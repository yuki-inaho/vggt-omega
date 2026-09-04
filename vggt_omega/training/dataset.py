"""Loader for the anonymized COLMAP-like RGB-D staging contract."""

from __future__ import annotations

import json
import math
import re
from itertools import combinations, pairwise
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


def overlap_curriculum_target(epoch: int, epochs: int, *, start_target: float, end_target: float) -> float:
    """Interpolate a high-to-medium overlap target at an epoch boundary."""

    if epoch < 0 or epochs < 1:
        raise DataContractError("overlap curriculum epoch values are invalid")
    if not 0 <= end_target <= start_target <= 1:
        raise DataContractError("overlap curriculum targets must satisfy 0 <= end <= start <= 1")
    progress = min(epoch, epochs - 1) / max(epochs - 1, 1)
    return float(start_target + (end_target - start_target) * progress)


def select_overlap_frame_offsets(
    pair_scores: np.ndarray,
    pair_valid: np.ndarray,
    *,
    sample_length: int,
    target: float,
    tolerance: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, bool]:
    """Select frame offsets around a target, with explicit nearest fallback."""

    scores = np.asarray(pair_scores)
    valid = np.asarray(pair_valid)
    if (
        scores.ndim != 2
        or scores.shape[0] != scores.shape[1]
        or valid.shape != scores.shape
        or valid.dtype != np.bool_
        or not np.isfinite(scores).all()
        or ((scores < 0) | (scores > 1)).any()
    ):
        raise DataContractError("overlap selection arrays violate the numeric contract")
    if not 2 <= sample_length <= scores.shape[0]:
        raise DataContractError("overlap sample_length is outside the profile shape")
    if not math.isfinite(target) or not 0 <= target <= 1 or not math.isfinite(tolerance) or tolerance < 0:
        raise DataContractError("overlap target and tolerance are invalid")
    candidates = np.argwhere(np.triu(valid, k=1))
    if not len(candidates):
        raise DataContractError("overlap profile contains no valid frame pair")

    def choose(values: np.ndarray) -> tuple[int, bool]:
        distance = np.abs(values - target)
        in_range = np.flatnonzero(distance <= tolerance)
        if len(in_range):
            return int(in_range[int(rng.integers(0, len(in_range)))]), False
        nearest = np.flatnonzero(np.isclose(distance, distance.min(), rtol=0, atol=1e-7))
        return int(nearest[int(rng.integers(0, len(nearest)))]), True

    candidate_scores = scores[candidates[:, 0], candidates[:, 1]]
    seed_index, used_fallback = choose(candidate_scores)
    selected = [int(value) for value in candidates[seed_index]]
    while len(selected) < sample_length:
        remaining = [index for index in range(scores.shape[0]) if index not in selected]
        aggregate_scores: list[float] = []
        eligible: list[int] = []
        for candidate in remaining:
            active_scores = [float(scores[candidate, current]) for current in selected if valid[candidate, current]]
            if active_scores:
                eligible.append(candidate)
                aggregate_scores.append(float(np.mean(active_scores)))
        if not eligible:
            raise DataContractError("overlap profile cannot extend the selected frame pair")
        selected_index, extension_fallback = choose(np.asarray(aggregate_scores, dtype=np.float32))
        used_fallback |= extension_fallback
        selected.append(eligible[selected_index])

    pair_values = [float(scores[first, second]) for first, second in combinations(selected, 2)]
    actual_score = float(np.mean(pair_values))
    reference_position = int(rng.integers(0, len(selected)))
    reference = selected.pop(reference_position)
    rng.shuffle(selected)
    return np.asarray([reference, *selected], dtype=np.int64), actual_score, used_fallback


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
        overlap_curriculum_enabled: bool = False,
        overlap_metric: str = "near_depth",
        overlap_start_target: float = 0.75,
        overlap_end_target: float = 0.5,
        overlap_target_tolerance: float = 0.05,
        overlap_curriculum_epochs: int = 1,
        filter_short_sequences: bool = False,
        flow_teacher_manifest: str | Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.min_frames = int(min_frames)
        self.max_frames = int(max_frames)
        self.seed = int(seed)
        self.min_valid_depth_pixels = int(min_valid_depth_pixels)
        self.overlap_curriculum_enabled = bool(overlap_curriculum_enabled)
        self.overlap_metric = str(overlap_metric)
        self.overlap_start_target = float(overlap_start_target)
        self.overlap_end_target = float(overlap_end_target)
        self.overlap_target_tolerance = float(overlap_target_tolerance)
        self.overlap_curriculum_epochs = int(overlap_curriculum_epochs)
        self.filter_short_sequences = bool(filter_short_sequences)
        self.flow_teacher_template: str | None = None
        self.epoch = 0
        if split not in {"train", "val", "smoke"}:
            raise DataContractError("split must be one of train, val, or smoke")
        if not 2 <= self.min_frames <= self.max_frames:
            raise DataContractError("frame counts must satisfy 2 <= min_frames <= max_frames")
        if self.min_valid_depth_pixels < 1:
            raise DataContractError("min_valid_depth_pixels must be positive")
        if self.overlap_metric not in {"all_depth", "near_depth"}:
            raise DataContractError("overlap_metric must be all_depth or near_depth")
        overlap_curriculum_target(
            0,
            self.overlap_curriculum_epochs,
            start_target=self.overlap_start_target,
            end_target=self.overlap_end_target,
        )
        if not math.isfinite(self.overlap_target_tolerance) or self.overlap_target_tolerance < 0:
            raise DataContractError("overlap_target_tolerance must be finite and non-negative")
        try:
            metadata = json.loads((self.root / "dataset.json").read_text())
        except (OSError, ValueError) as error:
            raise DataContractError("dataset.json is missing or invalid") from error
        self.dataset_format = str(metadata.get("format"))
        if self.dataset_format not in {"colmap_rgbd_v1", "colmap_rgbd_v2"}:
            raise DataContractError("dataset format is not a supported colmap_rgbd version")
        self.has_original_depth_observed = self.dataset_format == "colmap_rgbd_v2"
        if self.has_original_depth_observed:
            observed_metadata = metadata.get("original_depth_observed")
            if observed_metadata != {
                "directory": "original_depth_observed",
                "dtype": "uint8",
                "false_value": 0,
                "meaning": "pre_dilation_sensor_observation",
                "true_value": 255,
            }:
                raise DataContractError("v2 original-depth-observed metadata is missing or invalid")
        if flow_teacher_manifest is not None:
            manifest_path = Path(flow_teacher_manifest)
            if manifest_path.is_absolute() or ".." in manifest_path.parts:
                raise DataContractError("flow teacher manifest must be a safe relative path")
            try:
                flow_metadata = json.loads((self.root / manifest_path).read_text())
            except (OSError, ValueError) as error:
                raise DataContractError("flow teacher manifest is missing or invalid") from error
            expected_flow_metadata = {
                "coordinate_space": "pixel_displacement_xy",
                "file_template": "scenes/{scene_id}/flow_teacher/pair_{source_id:06d}_{target_id:06d}.npz",
                "flow_dtype": "float32",
                "format": "dynamic_flow_teacher_v1",
                "occlusion_dtype": "int8_tri_state",
                "schema_version": 1,
            }
            if flow_metadata != expected_flow_metadata:
                raise DataContractError("flow teacher manifest violates the explicit schema")
            self.flow_teacher_template = str(flow_metadata["file_template"])
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
        overlap_metadata = metadata.get("overlap")
        expected_overlap_keys = {
            "schema_version",
            "filename",
            "aggregation",
            "relative_depth_tolerance",
            "pixel_stride",
            "near_depth_m",
        }
        if (
            not isinstance(overlap_metadata, dict)
            or set(overlap_metadata) != expected_overlap_keys
            or overlap_metadata.get("schema_version") != 1
            or overlap_metadata.get("filename") != "overlap.npz"
            or overlap_metadata.get("aggregation") != "bidirectional_mean"
            or not isinstance(overlap_metadata.get("relative_depth_tolerance"), (int, float))
            or isinstance(overlap_metadata.get("relative_depth_tolerance"), bool)
            or not math.isfinite(float(overlap_metadata["relative_depth_tolerance"]))
            or float(overlap_metadata["relative_depth_tolerance"]) <= 0
            or not isinstance(overlap_metadata.get("pixel_stride"), int)
            or isinstance(overlap_metadata.get("pixel_stride"), bool)
            or int(overlap_metadata["pixel_stride"]) <= 0
            or not isinstance(overlap_metadata.get("near_depth_m"), (int, float))
            or isinstance(overlap_metadata.get("near_depth_m"), bool)
            or not math.isfinite(float(overlap_metadata["near_depth_m"]))
            or float(overlap_metadata["near_depth_m"]) <= 0
        ):
            raise DataContractError("overlap profile metadata is missing or invalid")
        self.overlap_metadata = dict(overlap_metadata)
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
        for scene_name in {scene_name for scene_name, _ in entries}:
            overlap_path = self.root / "scenes" / scene_name / str(self.overlap_metadata["filename"])
            if overlap_path.is_symlink() or not overlap_path.is_file():
                raise DataContractError("overlap profile file is missing or is not a regular file")
        if self.filter_short_sequences:
            eligible: list[tuple[str, int]] = []
            for scene_name, sequence_id in self.entries:
                scene = self._load_scene(scene_name)
                if not 0 <= sequence_id < len(scene["sequence_lengths"]):
                    raise DataContractError(f"sequence index {sequence_id} is out of range")
                if int(scene["sequence_lengths"][sequence_id]) >= self.min_frames:
                    eligible.append((scene_name, sequence_id))
            if not eligible:
                raise DataContractError("split contains no sequence long enough for the requested frame count")
            self.entries = eligible

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
            overlap_path = scene_root / str(self.overlap_metadata["filename"])
            if overlap_path.is_symlink():
                raise DataContractError("overlap profile must not be a symlink")
            with np.load(overlap_path, allow_pickle=False) as overlap:
                if set(overlap.files) != {"all_depth", "near_depth", "pair_valid"}:
                    raise DataContractError(f"{scene_name} contains unexpected overlap arrays")
                scene.update({f"overlap_{key}": overlap[key].copy() for key in overlap.files})
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
        overlap_shape = (sequence_count, self.stored_max_frames, self.stored_max_frames)
        overlap_all = scene.get("overlap_all_depth")
        overlap_near = scene.get("overlap_near_depth")
        overlap_valid = scene.get("overlap_pair_valid")
        if (
            overlap_all is None
            or overlap_near is None
            or overlap_valid is None
            or overlap_all.shape != overlap_shape
            or overlap_near.shape != overlap_shape
            or overlap_valid.shape != overlap_shape
            or overlap_all.dtype != np.float32
            or overlap_near.dtype != np.float32
            or overlap_valid.dtype != np.bool_
            or not np.isfinite(overlap_all).all()
            or not np.isfinite(overlap_near).all()
            or ((overlap_all < 0) | (overlap_all > 1)).any()
            or ((overlap_near < 0) | (overlap_near > 1)).any()
            or not np.allclose(overlap_all, overlap_all.transpose(0, 2, 1), atol=1e-6)
            or not np.allclose(overlap_near, overlap_near.transpose(0, 2, 1), atol=1e-6)
        ):
            raise DataContractError(f"{scene_name} overlap profile violates the numeric contract")
        expected_valid = np.zeros(overlap_shape, dtype=np.bool_)
        for sequence_id, length in enumerate(scene["sequence_lengths"]):
            count = int(length)
            expected_valid[sequence_id, :count, :count] = True
            np.fill_diagonal(expected_valid[sequence_id], False)
        if not np.array_equal(overlap_valid, expected_valid):
            raise DataContractError(f"{scene_name} overlap pair mask violates the sequence contract")
        self._scenes[scene_name] = scene
        return scene

    def _load_pixels(
        self, scene_name: str, frame_ids: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        rgb_frames: list[np.ndarray] = []
        depth_frames: list[np.ndarray] = []
        observed_frames: list[np.ndarray] = []
        for frame_id in frame_ids:
            filename = f"frame_{int(frame_id):06d}.png"
            rgb_path = self.root / "scenes" / scene_name / "rgb" / filename
            depth_path = self.root / "scenes" / scene_name / "depth" / filename
            observed_path = self.root / "scenes" / scene_name / "original_depth_observed" / filename
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
                if self.has_original_depth_observed:
                    if observed_path.is_symlink() or not observed_path.is_file():
                        raise DataContractError("v2 original-depth-observed mask must be a regular file")
                    with Image.open(observed_path) as image:
                        if image.info:
                            raise DataContractError("original-depth-observed metadata violates the staging contract")
                        observed = np.array(image, copy=True)
            except (OSError, ValueError, DataContractError) as error:
                raise DataContractError(f"generic frame {int(frame_id)} cannot be decoded") from error
            if rgb.shape != (self.height, self.width, 3) or rgb.dtype != np.uint8:
                raise DataContractError(f"generic RGB frame {int(frame_id)} has invalid shape or dtype")
            if depth.shape != (self.height, self.width) or depth.dtype != np.uint16:
                raise DataContractError(f"generic depth frame {int(frame_id)} has invalid shape or dtype")
            if self.has_original_depth_observed:
                if (
                    observed.shape != (self.height, self.width)
                    or observed.dtype != np.uint8
                    or not np.isin(observed, (0, 255)).all()
                    or np.any((observed == 255) & (depth == 0))
                ):
                    raise DataContractError("original-depth-observed mask violates the v2 pixel contract")
                observed_frames.append(observed == 255)
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
        observed_masks = torch.from_numpy(np.stack(observed_frames)) if self.has_original_depth_observed else None
        return images, depths, observed_masks

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
        stored = scene["sequence_sequences"][sequence_id, :stored_length]
        sampling_metrics: dict[str, torch.Tensor] = {}
        if self.overlap_curriculum_enabled:
            target = overlap_curriculum_target(
                self.epoch,
                self.overlap_curriculum_epochs,
                start_target=self.overlap_start_target,
                end_target=self.overlap_end_target,
            )
            offsets, actual_score, used_fallback = select_overlap_frame_offsets(
                scene[f"overlap_{self.overlap_metric}"][sequence_id, :stored_length, :stored_length],
                scene["overlap_pair_valid"][sequence_id, :stored_length, :stored_length],
                sample_length=sample_length,
                target=target,
                tolerance=self.overlap_target_tolerance,
                rng=rng,
            )
            selected = np.asarray(stored[offsets], dtype=np.int64)
            sampling_metrics = {
                "sampling_overlap_score": torch.tensor(actual_score, dtype=torch.float32),
                "sampling_overlap_target": torch.tensor(target, dtype=torch.float32),
                "sampling_overlap_fallback": torch.tensor(float(used_fallback), dtype=torch.float32),
            }
        else:
            start = int(rng.integers(0, stored_length - sample_length + 1))
            selected = np.asarray(stored[start : start + sample_length], dtype=np.int64)
            reference_offset = int(rng.integers(0, sample_length))
            selected = np.concatenate(
                (selected[reference_offset : reference_offset + 1], np.delete(selected, reference_offset))
            )
        if (selected < 0).any() or (selected >= len(scene["frame_ids"])).any():
            raise DataContractError("stored sequence contains an invalid frame ID")
        if not scene["quality_flags"][selected].all():
            raise DataContractError("stored sequence contains a frame that failed pose quality checks")
        expected_chunk = int(scene["sequence_chunk_ids"][sequence_id])
        if not np.all(scene["chunk_ids"][selected] == expected_chunk):
            raise DataContractError("stored sequence crosses a trajectory chunk boundary")

        images, metric_depths, original_observed = self._load_pixels(scene_name, selected)
        depth_masks = metric_depths > 0
        intrinsics = torch.from_numpy(scene["intrinsics"][selected].astype(np.float32, copy=True))
        extrinsics = torch.from_numpy(scene["extrinsics_w2c"][selected].astype(np.float32, copy=True))
        try:
            geometry = normalize_supervision(metric_depths, depth_masks, intrinsics, extrinsics)
        except GeometryContractError as error:
            raise DataContractError(f"dense geometry failed for sequence {sequence_id}: {error}") from error
        result = {
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
            **sampling_metrics,
        }
        if original_observed is not None:
            result["original_depth_observed_mask"] = original_observed
        if self.flow_teacher_template is not None:
            result.update(self._load_flow_teacher(scene_name, selected))
        return result

    def _load_flow_teacher(self, scene_name: str, selected: np.ndarray) -> dict[str, torch.Tensor]:
        ordered_positions = np.argsort(selected, kind="stable")
        pair_positions: list[tuple[int, int]] = []
        for left, right in pairwise(ordered_positions):
            pair_positions.extend(((int(left), int(right)), (int(right), int(left))))
        flows: list[np.ndarray] = []
        confidences: list[np.ndarray] = []
        occlusions: list[np.ndarray] = []
        assert self.flow_teacher_template is not None
        for source_position, target_position in pair_positions:
            source_id = int(selected[source_position])
            target_id = int(selected[target_position])
            relative = self.flow_teacher_template.format(
                scene_id=scene_name,
                source_id=source_id,
                target_id=target_id,
            )
            path = self.root / relative
            if path.is_symlink() or not path.is_file():
                raise DataContractError("directed flow teacher pair is missing or is not a regular file")
            try:
                with np.load(path, allow_pickle=False) as payload:
                    if set(payload.files) != {"confidence", "flow_xy", "occlusion_label"}:
                        raise ValueError("unexpected flow teacher arrays")
                    flow = payload["flow_xy"].copy()
                    confidence = payload["confidence"].copy()
                    occlusion = payload["occlusion_label"].copy()
            except (OSError, ValueError) as error:
                raise DataContractError("directed flow teacher pair is invalid") from error
            if (
                flow.shape != (self.height, self.width, 2)
                or flow.dtype != np.float32
                or confidence.shape != (self.height, self.width)
                or confidence.dtype != np.float32
                or occlusion.shape != (self.height, self.width)
                or occlusion.dtype != np.int8
                or not np.isfinite(flow).all()
                or not np.isfinite(confidence).all()
                or ((confidence < 0) | (confidence > 1)).any()
                or not np.isin(occlusion, (-1, 0, 1)).all()
            ):
                raise DataContractError("directed flow teacher arrays violate shape, dtype, or range")
            flows.append(flow)
            confidences.append(confidence)
            occlusions.append(occlusion)
        return {
            "motion_pixel_flow_xy": torch.from_numpy(np.stack(flows)),
            "motion_flow_confidence": torch.from_numpy(np.stack(confidences)),
            "motion_flow_occlusion_label": torch.from_numpy(np.stack(occlusions)),
        }
