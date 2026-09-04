"""Loader for the anonymized COLMAP-like RGB-D staging contract."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from vggt_omega.training.geometry import GeometryContractError, normalize_supervision

_ENTRY_PATTERN = re.compile(r"(scene_\d{6})/sequence_(\d{6})$")


class DataContractError(ValueError):
    """Raised when an exported scene or sample violates the loader contract."""


class ColmapRgbdDataset(Dataset[dict[str, Any]]):
    """Load fixed staging sequences and randomize only length/window/reference.

    Every returned item is already expressed relative to its randomized first
    camera and normalized by one common metric scale.  A regular PyTorch
    ``DataLoader`` therefore adds only the leading batch dimension.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        min_frames: int = 2,
        max_frames: int = 4,
        seed: int = 0,
        min_valid_depth_pixels: int = 1,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.min_frames = int(min_frames)
        self.max_frames = int(max_frames)
        self.seed = int(seed)
        self.min_valid_depth_pixels = int(min_valid_depth_pixels)
        self.epoch = 0
        if split not in {"train", "val", "smoke"}:
            raise DataContractError("split must be one of train, val, or smoke")
        if not 2 <= self.min_frames <= self.max_frames:
            raise DataContractError("frame counts must satisfy 2 <= min_frames <= max_frames")
        if self.min_valid_depth_pixels < 1:
            raise DataContractError("min_valid_depth_pixels must be positive")
        try:
            metadata = json.loads((self.root / "dataset.json").read_text())
        except (OSError, ValueError) as error:
            raise DataContractError("dataset.json is missing or invalid") from error
        if metadata.get("format") != "colmap_rgbd_v1":
            raise DataContractError("dataset format is not colmap_rgbd_v1")
        self.height = int(metadata.get("image", {}).get("height", 0))
        self.width = int(metadata.get("image", {}).get("width", 0))
        if (
            self.height <= 0
            or self.width <= 0
            or self.height % 16
            or self.width % 16
            or metadata.get("image", {}).get("channels") != 3
            or metadata.get("image", {}).get("dtype") != "uint8"
        ):
            raise DataContractError("dataset image dimensions are invalid")
        depth_metadata = metadata.get("depth", {})
        if depth_metadata.get("dtype") != "uint16" or depth_metadata.get("unit") != "millimeters":
            raise DataContractError("depth storage must be uint16 millimeters")
        if (
            depth_metadata.get("height") != self.height
            or depth_metadata.get("width") != self.width
            or depth_metadata.get("invalid_value") != 0
            or depth_metadata.get("mapped_depth_dilation_kernel") != 3
            or depth_metadata.get("mapped_depth_method") != "nearest_depth"
            or depth_metadata.get("additional_dilation") is not False
        ):
            raise DataContractError("depth metadata does not prove the mapped-depth contract")
        self.max_depth_mm = int(depth_metadata.get("max_depth_mm", 0))
        if self.max_depth_mm <= 0:
            raise DataContractError("max_depth_mm is invalid")
        camera_metadata = metadata.get("camera", {})
        if (
            camera_metadata.get("extrinsics") != "opencv_world_to_camera"
            or camera_metadata.get("intrinsics") != "pixel_units"
            or camera_metadata.get("source") != "aligned_trajectory_camera_to_world_inverted"
        ):
            raise DataContractError("camera metadata does not prove the required trajectory convention")
        sequence_metadata = metadata.get("sequences", {})
        self.stored_min_frames = int(sequence_metadata.get("min_length", 0))
        self.stored_max_frames = int(sequence_metadata.get("max_length", 0))
        if not 2 <= self.stored_min_frames <= self.stored_max_frames:
            raise DataContractError("dataset sequence length metadata is invalid")

        split_path = self.root / "splits" / f"{split}.txt"
        try:
            lines = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
        except OSError as error:
            raise DataContractError(f"split entry file is missing for {split}") from error
        if not lines:
            raise DataContractError(f"split entry file is empty for {split}")
        entries: list[tuple[str, int]] = []
        for line in lines:
            match = _ENTRY_PATTERN.fullmatch(line)
            if match is None or Path(line).is_absolute():
                raise DataContractError("split entry is not a generic relative sequence name")
            entries.append((match.group(1), int(match.group(2))))
        self.entries = entries
        self._scenes: dict[str, dict[str, np.ndarray]] = {}

    def set_epoch(self, epoch: int) -> None:
        """Change deterministic sampling without relying on process-global RNG state."""

        if epoch < 0:
            raise DataContractError("epoch must be non-negative")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.entries)

    def _load_scene(self, scene_name: str) -> dict[str, np.ndarray]:
        if scene_name in self._scenes:
            return self._scenes[scene_name]
        scene_root = self.root / "scenes" / scene_name
        camera_path = scene_root / "cameras.npz"
        sequence_path = scene_root / "sequences.npz"
        if camera_path.is_symlink() or sequence_path.is_symlink():
            raise DataContractError("scene array files must not be symlinks")
        try:
            with np.load(camera_path, allow_pickle=False) as cameras:
                if set(cameras.files) != {
                    "frame_ids",
                    "intrinsics",
                    "extrinsics_w2c",
                    "quality_flags",
                    "chunk_ids",
                }:
                    raise DataContractError(f"{scene_name} contains unexpected camera arrays")
                scene = {key: cameras[key].copy() for key in cameras.files}
            with np.load(sequence_path, allow_pickle=False) as sequences:
                if set(sequences.files) != {"sequences", "lengths", "split_ids", "chunk_ids"}:
                    raise DataContractError(f"{scene_name} contains unexpected sequence arrays")
                scene.update({f"sequence_{key}": sequences[key].copy() for key in sequences.files})
        except (OSError, ValueError, KeyError, DataContractError) as error:
            raise DataContractError(f"scene arrays are missing or invalid for {scene_name}") from error
        frame_count = len(scene.get("frame_ids", ()))
        required_camera_shapes = {
            "frame_ids": (frame_count,),
            "intrinsics": (frame_count, 3, 3),
            "extrinsics_w2c": (frame_count, 3, 4),
            "quality_flags": (frame_count,),
            "chunk_ids": (frame_count,),
        }
        for key, shape in required_camera_shapes.items():
            if key not in scene or scene[key].shape != shape:
                raise DataContractError(f"{scene_name} has invalid {key} shape")
        if (
            scene["frame_ids"].dtype != np.int64
            or scene["intrinsics"].dtype != np.float32
            or scene["extrinsics_w2c"].dtype != np.float32
            or scene["quality_flags"].dtype != np.bool_
            or scene["chunk_ids"].dtype != np.int64
        ):
            raise DataContractError(f"{scene_name} camera array dtypes violate the staging contract")
        if scene["frame_ids"].tolist() != list(range(frame_count)):
            raise DataContractError(f"{scene_name} frame_ids must be contiguous generic IDs")
        if (scene["chunk_ids"] < 0).any():
            raise DataContractError(f"{scene_name} chunk IDs must be non-negative")
        if not np.isfinite(scene["intrinsics"]).all() or not np.isfinite(scene["extrinsics_w2c"]).all():
            raise DataContractError(f"{scene_name} camera arrays contain NaN or Inf")
        if (
            (scene["intrinsics"][:, 0, 0] <= 0).any()
            or (scene["intrinsics"][:, 1, 1] <= 0).any()
            or not np.allclose(scene["intrinsics"][:, 2, :], (0.0, 0.0, 1.0), atol=1e-6)
            or not np.allclose(scene["intrinsics"][:, 0, 2], self.width / 2, atol=1e-4)
            or not np.allclose(scene["intrinsics"][:, 1, 2], self.height / 2, atol=1e-4)
        ):
            raise DataContractError(f"{scene_name} intrinsics violate the centered pinhole contract")
        sequence_count = len(scene.get("sequence_lengths", ()))
        if (
            scene.get("sequence_sequences", np.empty((0,))).shape != (sequence_count, self.stored_max_frames)
            or scene["sequence_sequences"].dtype != np.int64
        ):
            raise DataContractError(f"{scene_name} sequences must be a rank-two array")
        for key in ("sequence_lengths", "sequence_split_ids", "sequence_chunk_ids"):
            if key not in scene or scene[key].shape != (sequence_count,):
                raise DataContractError(f"{scene_name} has invalid {key} shape")
        if len(scene["sequence_sequences"]) != sequence_count:
            raise DataContractError(f"{scene_name} sequence arrays have inconsistent lengths")
        if (
            scene["sequence_lengths"].dtype != np.int64
            or scene["sequence_split_ids"].dtype != np.int8
            or scene["sequence_chunk_ids"].dtype != np.int64
            or (scene["sequence_lengths"] < self.stored_min_frames).any()
            or (scene["sequence_lengths"] > self.stored_max_frames).any()
            or not np.isin(scene["sequence_split_ids"], (0, 1, 2)).all()
            or (scene["sequence_chunk_ids"] < 0).any()
        ):
            raise DataContractError(f"{scene_name} sequence values violate the staging contract")
        for row, length, chunk_id in zip(
            scene["sequence_sequences"],
            scene["sequence_lengths"],
            scene["sequence_chunk_ids"],
            strict=True,
        ):
            count = int(length)
            active = row[:count]
            if (
                len(set(active.tolist())) != count
                or (active < 0).any()
                or (active >= frame_count).any()
                or np.any(row[count:] != -1)
                or (np.diff(active) != 1).any()
                or not np.all(scene["chunk_ids"][active] == int(chunk_id))
                or not scene["quality_flags"][active].all()
            ):
                raise DataContractError(f"{scene_name} contains an invalid stored sequence")
        self._scenes[scene_name] = scene
        return scene

    def _load_pixels(self, scene_name: str, frame_ids: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        rgb_frames: list[np.ndarray] = []
        depth_frames: list[np.ndarray] = []
        for frame_id in frame_ids:
            filename = f"frame_{int(frame_id):06d}.png"
            rgb_path = self.root / "scenes" / scene_name / "rgb" / filename
            depth_path = self.root / "scenes" / scene_name / "depth" / filename
            if rgb_path.is_symlink() or depth_path.is_symlink() or not rgb_path.is_file() or not depth_path.is_file():
                raise DataContractError("RGB and depth frames must be regular files, not symlinks")
            try:
                with Image.open(rgb_path) as image:
                    if image.mode != "RGB" or image.info:
                        raise DataContractError("RGB frame mode or metadata violates the staging contract")
                    rgb = np.array(image, dtype=np.uint8, copy=True)
                with Image.open(depth_path) as image:
                    if image.info:
                        raise DataContractError("depth frame metadata violates the staging contract")
                    depth = np.array(image, copy=True)
            except (OSError, ValueError, DataContractError) as error:
                raise DataContractError(f"generic frame {int(frame_id)} cannot be decoded") from error
            if rgb.shape != (self.height, self.width, 3) or rgb.dtype != np.uint8:
                raise DataContractError(f"generic RGB frame {int(frame_id)} has invalid shape or dtype")
            if depth.shape != (self.height, self.width) or depth.dtype != np.uint16:
                raise DataContractError(f"generic depth frame {int(frame_id)} has invalid shape or dtype")
            if int(depth.max(initial=0)) > self.max_depth_mm:
                raise DataContractError(f"generic depth frame {int(frame_id)} exceeds max_depth_mm")
            valid_count = int(np.count_nonzero(depth))
            if valid_count < self.min_valid_depth_pixels:
                raise DataContractError(
                    f"generic depth frame {int(frame_id)} has fewer than {self.min_valid_depth_pixels} valid pixels"
                )
            rgb_frames.append(rgb)
            depth_frames.append(depth)
        images = torch.from_numpy(np.stack(rgb_frames)).permute(0, 3, 1, 2).float().div_(255)
        depths = torch.from_numpy(np.stack(depth_frames).astype(np.float32)).div_(1000)
        return images, depths

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not 0 <= index < len(self.entries):
            raise IndexError(index)
        scene_name, sequence_id = self.entries[index]
        scene = self._load_scene(scene_name)
        if not 0 <= sequence_id < len(scene["sequence_lengths"]):
            raise DataContractError(f"sequence index {sequence_id} is out of range")
        expected_split_id = {"train": 0, "val": 1, "smoke": 2}[self.split]
        if int(scene["sequence_split_ids"][sequence_id]) != expected_split_id:
            raise DataContractError("split file points to a sequence from another split")
        stored_length = int(scene["sequence_lengths"][sequence_id])
        if stored_length < self.min_frames:
            raise DataContractError(
                f"stored sequence is shorter than requested min_frames ({stored_length} < {self.min_frames})"
            )
        upper_length = min(self.max_frames, stored_length)
        rng = np.random.default_rng(np.random.SeedSequence((self.seed, self.epoch, int(index))))
        sample_length = int(rng.integers(self.min_frames, upper_length + 1))
        start = int(rng.integers(0, stored_length - sample_length + 1))
        stored = scene["sequence_sequences"][sequence_id, :stored_length]
        selected = np.asarray(stored[start : start + sample_length], dtype=np.int64)
        if (selected < 0).any() or (selected >= len(scene["frame_ids"])).any():
            raise DataContractError("stored sequence contains an invalid frame ID")
        reference_offset = int(rng.integers(0, sample_length))
        selected = np.concatenate(
            (selected[reference_offset : reference_offset + 1], np.delete(selected, reference_offset))
        )
        if not scene["quality_flags"][selected].all():
            raise DataContractError("stored sequence contains a frame that failed pose quality checks")
        expected_chunk = int(scene["sequence_chunk_ids"][sequence_id])
        if not np.all(scene["chunk_ids"][selected] == expected_chunk):
            raise DataContractError("stored sequence crosses a trajectory chunk boundary")

        images, metric_depths = self._load_pixels(scene_name, selected)
        depth_masks = metric_depths > 0
        intrinsics = torch.from_numpy(scene["intrinsics"][selected].astype(np.float32, copy=True))
        extrinsics = torch.from_numpy(scene["extrinsics_w2c"][selected].astype(np.float32, copy=True))
        try:
            geometry = normalize_supervision(metric_depths, depth_masks, intrinsics, extrinsics)
        except GeometryContractError as error:
            raise DataContractError(f"dense geometry failed for sequence {sequence_id}: {error}") from error
        return {
            "images": images,
            "depths": geometry["depths"],
            "depth_masks": geometry["depth_masks"],
            "extrinsics": geometry["extrinsics"],
            "intrinsics": geometry["intrinsics"],
            "frame_ids": torch.from_numpy(selected.copy()),
            "cam_points": geometry["cam_points"],
            "world_points": geometry["world_points"],
            "normalization_scale_m": geometry["scale"],
            "scene_id": scene_name,
            "sequence_id": sequence_id,
        }
