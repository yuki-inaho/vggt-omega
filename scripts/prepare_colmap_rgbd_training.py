"""Export RGB-D and aligned poses to a privacy-safe VGGT-Omega staging set.

Only the already mapped, 3x3-dilated depth images are consumed.  This module
deliberately contains no depth dilation operation, and COLMAP poses or sparse
points are not accepted as inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from itertools import combinations, pairwise
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import torch
from PIL import Image

from vggt_omega.training.overlap import bidirectional_rgbd_overlap

DATASET_FORMAT = "colmap_rgbd_v1"
SCENE_NAME = "scene_000000"
SPLIT_NAMES = ("train", "val", "smoke")
_SCENE_FILE = re.compile(r"scenes/scene_\d{6}/(?:rgb|depth)/frame_\d{6}\.png$")
_SCENE_ARRAY = re.compile(r"scenes/scene_\d{6}/(?:cameras|sequences|overlap)\.npz$")
_SPLIT_FILE = re.compile(r"splits/(?:train|val|smoke)\.txt$")
_SEQUENCE_ENTRY = re.compile(r"scene_\d{6}/sequence_(\d{6})$")
_GENERIC_PRIVATE_PATTERNS = (
    rb"/home/",
    rb"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}",
    rb"\d{4}-\d{2}-\d{2}",
    rb"camera_[lr]",
)


class ExportContractError(ValueError):
    """Raised when private source data cannot satisfy the staging contract."""


def _require_finite_matrix(name: str, matrix: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != shape:
        raise ExportContractError(f"{name} must have shape {shape}, got {value.shape}")
    if not np.isfinite(value).all():
        raise ExportContractError(f"{name} contains NaN or Inf")
    return value


def compute_principal_crop(
    intrinsics: np.ndarray,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> tuple[tuple[float, float, float, float], np.ndarray]:
    """Compute a maximal aspect-preserving crop centered on the principal point."""

    matrix = _require_finite_matrix("intrinsics", intrinsics, (3, 3))
    source_width, source_height = map(int, source_size)
    target_width, target_height = map(int, target_size)
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ExportContractError("source and target dimensions must be positive")
    cx, cy = float(matrix[0, 2]), float(matrix[1, 2])
    if not (0 < cx < source_width and 0 < cy < source_height):
        raise ExportContractError("principal point must be strictly inside the source image")
    aspect = target_width / target_height
    half_width_limit = min(cx, source_width - cx)
    half_height_limit = min(cy, source_height - cy)
    half_height = min(half_height_limit, half_width_limit / aspect)
    half_width = half_height * aspect
    if half_width <= 0 or half_height <= 0:
        raise ExportContractError("principal-point crop is empty")
    crop = (cx - half_width, cy - half_height, cx + half_width, cy + half_height)
    scale_x = target_width / (2 * half_width)
    scale_y = target_height / (2 * half_height)
    updated = matrix.copy()
    updated[0, 0] *= scale_x
    updated[0, 1] *= scale_x
    updated[0, 2] = (cx - crop[0]) * scale_x
    updated[1, 0] *= scale_y
    updated[1, 1] *= scale_y
    updated[1, 2] = (cy - crop[1]) * scale_y
    return crop, updated


def _read_rgb_camera_yaml(path: Path) -> tuple[int, int, np.ndarray, str | None]:
    text = path.read_text()

    def integer_field(field: str) -> int:
        match = re.search(rf"(?m)^\s*{re.escape(field)}\s*:\s*(\d+)\s*$", text)
        if match is None:
            raise ExportContractError(f"RGB camera YAML is missing {field}")
        return int(match.group(1))

    matrix_match = re.search(r"(?ms)^\s*K\s*:\s*\[(.*?)\]", text)
    if matrix_match is None:
        raise ExportContractError("RGB camera YAML is missing K")
    try:
        values = [float(item.strip()) for item in matrix_match.group(1).split(",")]
    except ValueError as error:
        raise ExportContractError("RGB camera YAML K is not numeric") from error
    if len(values) != 9:
        raise ExportContractError("RGB camera YAML K must contain nine values")
    serial_match = re.search(r"(?m)^\s*serial_number\s*:\s*(\S+)\s*$", text)
    serial = serial_match.group(1) if serial_match else None
    intrinsics = _require_finite_matrix("RGB camera YAML K", np.asarray(values).reshape(3, 3), (3, 3))
    if intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
        raise ExportContractError("RGB camera YAML focal lengths must be positive")
    if not np.allclose(intrinsics[2], (0.0, 0.0, 1.0), atol=1e-8):
        raise ExportContractError("RGB camera YAML K has an invalid homogeneous row")
    return integer_field("width"), integer_field("height"), intrinsics, serial


def _pair_key(path: Path, suffix: str) -> str:
    stem = path.stem
    marker = f"_{suffix}"
    if not stem.endswith(marker):
        raise ExportContractError(f"expected a filename ending in {marker}.png")
    return stem[: -len(marker)]


def _load_trajectory(path: Path, expected_names: set[str]) -> tuple[list[dict[str, Any]], np.ndarray]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, Mapping):
        raise ExportContractError("trajectory JSON must contain an object")
    if not str(payload.get("pose_convention", "")).startswith("camera_to_world"):
        raise ExportContractError("trajectory pose_convention must be camera_to_world")
    raw_frames = payload.get("frames")
    if not isinstance(raw_frames, list) or not raw_frames:
        raise ExportContractError("trajectory contains no frames")
    frames: list[dict[str, Any]] = []
    for record_index, raw_frame in enumerate(raw_frames):
        if not isinstance(raw_frame, Mapping):
            raise ExportContractError(f"trajectory frame {record_index} is not an object")
        frame_record = cast(Mapping[str, Any], raw_frame)
        for field in ("frame_index", "image_name", "camera_to_world"):
            if field not in frame_record:
                raise ExportContractError(f"trajectory frame {record_index} is missing {field}")
        try:
            frame_index = int(frame_record["frame_index"])
        except (TypeError, ValueError) as error:
            raise ExportContractError(f"trajectory frame {record_index} has a non-integer frame_index") from error
        image_name = frame_record["image_name"]
        if not isinstance(image_name, str) or not image_name or Path(image_name).name != image_name:
            raise ExportContractError(f"trajectory frame {record_index} has an invalid image_name")
        frame = dict(frame_record)
        frame["frame_index"] = frame_index
        frames.append(frame)
    frames.sort(key=lambda frame: int(frame["frame_index"]))
    if int(payload.get("frame_count", -1)) != len(frames):
        raise ExportContractError("trajectory frame_count does not match its frame records")
    indices = [int(frame["frame_index"]) for frame in frames]
    if indices != list(range(len(frames))):
        raise ExportContractError("trajectory frame_index values must be contiguous from zero")
    names = [str(frame["image_name"]) for frame in frames]
    if len(set(names)) != len(names) or set(names) != expected_names:
        raise ExportContractError("trajectory and RGB filenames do not have a one-to-one match")

    raw_chunks = payload.get("chunk_scales")
    if not isinstance(raw_chunks, list) or not raw_chunks:
        raise ExportContractError("trajectory contains no chunk metadata")
    chunk_ids = np.full(len(frames), -1, dtype=np.int64)
    for fallback_index, chunk in enumerate(raw_chunks):
        if not isinstance(chunk, Mapping):
            raise ExportContractError(f"trajectory chunk {fallback_index} is not an object")
        chunk_record = cast(Mapping[str, Any], chunk)
        try:
            chunk_index = int(chunk_record.get("chunk_index", fallback_index))
        except (TypeError, ValueError) as error:
            raise ExportContractError(f"trajectory chunk {fallback_index} has an invalid chunk_index") from error
        global_indices = chunk_record.get("global_indices")
        if chunk_index < 0 or not isinstance(global_indices, list):
            raise ExportContractError(f"trajectory chunk {fallback_index} has invalid membership metadata")
        for frame_index in global_indices:
            try:
                index = int(frame_index)
            except (TypeError, ValueError) as error:
                raise ExportContractError(f"trajectory chunk {fallback_index} contains a non-integer frame") from error
            if not 0 <= index < len(frames):
                raise ExportContractError(f"trajectory chunk {fallback_index} references an out-of-range frame")
            if chunk_ids[index] < 0:
                chunk_ids[index] = chunk_index
    if (chunk_ids < 0).any():
        raise ExportContractError("trajectory chunk metadata does not cover every frame")
    return frames, chunk_ids


def _invert_camera_to_world(camera_to_world: Any, frame_index: int) -> np.ndarray:
    matrix = _require_finite_matrix(f"camera_to_world for frame {frame_index}", camera_to_world, (4, 4))
    if not np.allclose(matrix[3], (0, 0, 0, 1), atol=1e-7):
        raise ExportContractError(f"camera_to_world for frame {frame_index} has an invalid last row")
    rotation = matrix[:3, :3]
    determinant = float(np.linalg.det(rotation))
    if not math.isfinite(determinant) or abs(determinant - 1.0) > 1e-4:
        raise ExportContractError(f"camera_to_world for frame {frame_index} has an invalid rotation")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
        raise ExportContractError(f"camera_to_world for frame {frame_index} has a non-orthogonal rotation")
    try:
        return np.linalg.inv(matrix)[:3]
    except np.linalg.LinAlgError as error:
        raise ExportContractError(f"camera_to_world for frame {frame_index} is singular") from error


def _camera_centers(extrinsics_w2c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotations = extrinsics_w2c[:, :3, :3]
    translations = extrinsics_w2c[:, :3, 3]
    centers = -np.einsum("nij,nj->ni", rotations.transpose(0, 2, 1), translations)
    camera_to_world_rotations = rotations.transpose(0, 2, 1)
    return centers, camera_to_world_rotations


def _rotation_delta_degrees(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.T @ second
    cosine = float(np.clip((np.trace(relative) - 1) / 2, -1, 1))
    return math.degrees(math.acos(cosine))


def _split_labels(
    frame_count: int,
    guard_frames: int,
    split_fractions: tuple[float, float, float],
    min_sequence_length: int,
) -> np.ndarray:
    if guard_frames < 0:
        raise ExportContractError("guard_frames must be non-negative")
    fractions = np.asarray(split_fractions, dtype=np.float64)
    if fractions.shape != (3,) or (fractions <= 0).any() or not np.isclose(fractions.sum(), 1.0):
        raise ExportContractError("split_fractions must be three positive values summing to one")
    usable = frame_count - 2 * guard_frames
    if usable < 3 * min_sequence_length:
        raise ExportContractError("not enough frames for three disjoint splits and guard bands")
    train_count = int(math.floor(usable * fractions[0]))
    val_count = int(math.floor(usable * fractions[1]))
    smoke_count = usable - train_count - val_count
    counts = (train_count, val_count, smoke_count)
    if min(counts) < min_sequence_length:
        raise ExportContractError("a requested split is too short for one sequence")
    labels = np.full(frame_count, -1, dtype=np.int8)
    cursor = 0
    labels[cursor : cursor + train_count] = 0
    cursor += train_count + guard_frames
    labels[cursor : cursor + val_count] = 1
    cursor += val_count + guard_frames
    labels[cursor : cursor + smoke_count] = 2
    return labels


def _make_sequences(
    extrinsics_w2c: np.ndarray,
    chunk_ids: np.ndarray,
    split_labels: np.ndarray,
    min_sequence_length: int,
    max_sequence_length: int,
    max_translation_m: float,
    max_rotation_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 2 <= min_sequence_length <= max_sequence_length:
        raise ExportContractError("sequence lengths must satisfy 2 <= min <= max")
    centers, rotations = _camera_centers(extrinsics_w2c)
    rows: list[list[int]] = []
    lengths: list[int] = []
    sequence_splits: list[int] = []
    sequence_chunks: list[int] = []
    for split_id in range(3):
        split_indices = np.flatnonzero(split_labels == split_id).tolist()
        runs: list[list[int]] = []
        current: list[int] = []
        for frame_index in split_indices:
            if current:
                previous = current[-1]
                translation = float(np.linalg.norm(centers[frame_index] - centers[previous]))
                rotation = _rotation_delta_degrees(rotations[previous], rotations[frame_index])
                continues = (
                    frame_index == previous + 1
                    and chunk_ids[frame_index] == chunk_ids[previous]
                    and translation <= max_translation_m
                    and rotation <= max_rotation_deg
                )
                if not continues:
                    runs.append(current)
                    current = []
            current.append(frame_index)
        if current:
            runs.append(current)
        for run in runs:
            for start in range(max(0, len(run) - min_sequence_length + 1)):
                sequence = run[start : start + max_sequence_length]
                if len(sequence) < min_sequence_length:
                    continue
                rows.append(sequence + [-1] * (max_sequence_length - len(sequence)))
                lengths.append(len(sequence))
                sequence_splits.append(split_id)
                sequence_chunks.append(int(chunk_ids[sequence[0]]))
        if split_id not in sequence_splits:
            raise ExportContractError(f"{SPLIT_NAMES[split_id]} has no valid chunk-local sequence")
    return (
        np.asarray(rows, dtype=np.int64),
        np.asarray(lengths, dtype=np.int64),
        np.asarray(sequence_splits, dtype=np.int8),
        np.asarray(sequence_chunks, dtype=np.int64),
    )


def _warp_image(
    array: np.ndarray, crop: tuple[float, float, float, float], target_size: tuple[int, int], depth: bool
) -> np.ndarray:
    target_width, target_height = target_size
    x0, y0, x1, y1 = crop
    scale_x = target_width / (x1 - x0)
    scale_y = target_height / (y1 - y0)
    source_to_target = np.asarray([[scale_x, 0.0, -x0 * scale_x], [0.0, scale_y, -y0 * scale_y]], dtype=np.float64)
    interpolation = cv2.INTER_NEAREST if depth else cv2.INTER_CUBIC
    return cv2.warpAffine(
        array,
        source_to_target,
        (target_width, target_height),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0.0, 0.0, 0.0),
    )


def _write_split_files(root: Path, sequence_splits: np.ndarray) -> None:
    split_dir = root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split_id, split_name in enumerate(SPLIT_NAMES):
        sequence_ids = np.flatnonzero(sequence_splits == split_id)
        content = "".join(f"{SCENE_NAME}/sequence_{int(index):06d}\n" for index in sequence_ids)
        (split_dir / f"{split_name}.txt").write_text(content)


def _write_overlap_profile(
    root: Path,
    sequences: np.ndarray,
    lengths: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics_w2c: np.ndarray,
    *,
    relative_depth_tolerance: float,
    pixel_stride: int,
    near_depth_m: float,
) -> None:
    """Write numeric-only pair-overlap arrays for anonymous sequences."""

    if relative_depth_tolerance <= 0 or pixel_stride < 1 or near_depth_m <= 0:
        raise ExportContractError("overlap profile options must be positive")
    sequence_count, max_frames = sequences.shape
    all_depth = np.zeros((sequence_count, max_frames, max_frames), dtype=np.float32)
    near_depth = np.zeros_like(all_depth)
    pair_valid = np.zeros((sequence_count, max_frames, max_frames), dtype=np.bool_)
    scene_root = root / f"scenes/{SCENE_NAME}"

    @lru_cache(maxsize=16)
    def load_depth(frame_id: int) -> torch.Tensor:
        path = scene_root / "depth" / f"frame_{frame_id:06d}.png"
        try:
            with Image.open(path) as image:
                depth = np.array(image, dtype=np.uint16, copy=True)
        except (OSError, ValueError) as error:
            raise ExportContractError(f"generic depth frame {frame_id} cannot be read for overlap") from error
        return torch.from_numpy(depth.astype(np.float32)).div_(1000)

    cached_pairs: dict[tuple[int, int], tuple[float, float]] = {}
    camera_intrinsics = torch.from_numpy(np.asarray(intrinsics, dtype=np.float32))
    camera_extrinsics = torch.from_numpy(np.asarray(extrinsics_w2c, dtype=np.float32))
    for sequence_id, (row, length) in enumerate(zip(sequences, lengths, strict=True)):
        count = int(length)
        for first_offset in range(count):
            for second_offset in range(first_offset + 1, count):
                first_id = int(row[first_offset])
                second_id = int(row[second_offset])
                pair_key = (min(first_id, second_id), max(first_id, second_id))
                if pair_key not in cached_pairs:
                    first_depth = load_depth(pair_key[0])
                    second_depth = load_depth(pair_key[1])
                    arguments = (
                        first_depth,
                        second_depth,
                        camera_intrinsics[pair_key[0]],
                        camera_intrinsics[pair_key[1]],
                        camera_extrinsics[pair_key[0]],
                        camera_extrinsics[pair_key[1]],
                    )
                    score_all = bidirectional_rgbd_overlap(
                        *arguments,
                        relative_depth_tolerance=relative_depth_tolerance,
                        pixel_stride=pixel_stride,
                    )
                    score_near = bidirectional_rgbd_overlap(
                        *arguments,
                        relative_depth_tolerance=relative_depth_tolerance,
                        pixel_stride=pixel_stride,
                        max_depth_m=near_depth_m,
                    )
                    cached_pairs[pair_key] = (float(score_all), float(score_near))
                score_all, score_near = cached_pairs[pair_key]
                all_depth[sequence_id, first_offset, second_offset] = score_all
                all_depth[sequence_id, second_offset, first_offset] = score_all
                near_depth[sequence_id, first_offset, second_offset] = score_near
                near_depth[sequence_id, second_offset, first_offset] = score_near
                pair_valid[sequence_id, first_offset, second_offset] = True
                pair_valid[sequence_id, second_offset, first_offset] = True
    np.savez_compressed(
        scene_root / "overlap.npz",
        all_depth=all_depth,
        near_depth=near_depth,
        pair_valid=pair_valid,
    )


def _private_tokens(
    source_root: Path,
    trajectory_path: Path,
    camera_yaml: Path,
    rgb_paths: Sequence[Path],
    serial: str | None,
) -> tuple[str, ...]:
    candidates = {
        "/home/",
        source_root.name,
        str(source_root.absolute()),
        str(trajectory_path.absolute()),
        str(camera_yaml.absolute()),
    }
    for path in rgb_paths:
        candidates.update(re.findall(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", path.stem))
        candidates.update(re.findall(r"\d{4}-\d{2}-\d{2}", path.stem))
        candidates.update(re.findall(r"camera_[lr]", path.stem, flags=re.IGNORECASE))
    if serial:
        candidates.add(serial)
    return tuple(sorted(token for token in candidates if len(token) >= 6))


def _allowed_relative_path(relative: str, is_directory: bool) -> bool:
    if relative in {"dataset.json", "reports/export_validation.json"}:
        return not is_directory
    if _SCENE_FILE.fullmatch(relative) or _SCENE_ARRAY.fullmatch(relative) or _SPLIT_FILE.fullmatch(relative):
        return not is_directory
    if not is_directory:
        return False
    return bool(
        re.fullmatch(r"(?:scenes|splits|reports|scenes/scene_\d{6}|scenes/scene_\d{6}/(?:rgb|depth))", relative)
    )


def _compile_token_pattern(tokens: Sequence[str]) -> tuple[re.Pattern[bytes] | None, int]:
    encoded = {token.lower().encode() for token in tokens if token}
    ordered = sorted(encoded, key=lambda token: (-len(token), token))
    pattern_parts = [*_GENERIC_PRIVATE_PATTERNS, *(re.escape(token) for token in ordered)]
    longest = max(36, *(map(len, ordered))) if ordered else 36
    return re.compile(b"|".join(pattern_parts)), longest


def _contains_token(path: Path, pattern: re.Pattern[bytes] | None, longest: int) -> bool:
    if pattern is None or not path.is_file() or path.is_symlink():
        return False
    tail = b""
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                haystack = (tail + block).lower()
                if pattern.search(haystack) is not None:
                    return True
                tail = haystack[-max(longest - 1, 0) :]
    except OSError:
        return True
    return False


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from _walk_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_strings(child)


def _split_validation(root: Path, dataset: Mapping[str, Any] | None) -> dict[str, int]:
    report = {
        "frame_overlap_count": 0,
        "pair_overlap_count": 0,
        "chunk_boundary_violation_count": 0,
        "motion_threshold_violation_count": 0,
        "quality_flag_violation_count": 0,
        "entry_violation_count": 0,
        "structure_violation_count": 0,
    }
    sequence_path = root / f"scenes/{SCENE_NAME}/sequences.npz"
    camera_path = root / f"scenes/{SCENE_NAME}/cameras.npz"
    if not sequence_path.is_file() or not camera_path.is_file():
        report["structure_violation_count"] = 1
        return report
    try:
        with np.load(sequence_path, allow_pickle=False) as data:
            if set(data.files) != {"sequences", "lengths", "split_ids", "chunk_ids"}:
                raise ValueError("unexpected sequence arrays")
            sequences = data["sequences"]
            lengths = data["lengths"]
            sequence_splits = data["split_ids"]
            sequence_chunks = data["chunk_ids"]
        with np.load(camera_path, allow_pickle=False) as data:
            frame_ids = data["frame_ids"]
            frame_chunks = data["chunk_ids"]
            quality_flags = data["quality_flags"]
            extrinsics = data["extrinsics_w2c"]
        frame_count = len(frame_ids)
        if (
            frame_ids.shape != (frame_count,)
            or frame_chunks.shape != (frame_count,)
            or quality_flags.shape != (frame_count,)
            or extrinsics.shape != (frame_count, 3, 4)
            or not np.isfinite(extrinsics).all()
        ):
            raise ValueError("invalid camera arrays for sequence validation")
        sequence_count = len(lengths)
        if (
            sequences.ndim != 2
            or lengths.shape != (sequence_count,)
            or sequence_splits.shape != (sequence_count,)
            or sequence_chunks.shape != (sequence_count,)
            or sequences.dtype != np.int64
            or lengths.dtype != np.int64
            or sequence_splits.dtype != np.int8
            or sequence_chunks.dtype != np.int64
        ):
            raise ValueError("invalid sequence array contract")
        if dataset is None:
            raise ValueError("dataset metadata is unavailable")
        sequence_metadata = dataset.get("sequences", {})
        min_length = int(sequence_metadata["min_length"])
        max_length = int(sequence_metadata["max_length"])
        max_translation_m = float(sequence_metadata["max_translation_m"])
        max_rotation_deg = float(sequence_metadata["max_rotation_deg"])
        if not 2 <= min_length <= max_length or max_length != sequences.shape[1]:
            raise ValueError("invalid sequence length metadata")
        if (
            not math.isfinite(max_translation_m)
            or not math.isfinite(max_rotation_deg)
            or max_translation_m <= 0
            or max_rotation_deg <= 0
        ):
            raise ValueError("invalid sequence motion thresholds")
        if (frame_chunks < 0).any() or (sequence_chunks < 0).any():
            raise ValueError("chunk IDs must be non-negative")
        declared_splits = dataset["splits"]
        if any(
            int(declared_splits[name]) != int(np.sum(sequence_splits == split_id))
            for split_id, name in enumerate(SPLIT_NAMES)
        ):
            raise ValueError("declared split counts do not match sequence arrays")
        centers, rotations = _camera_centers(extrinsics)
    except Exception:
        report["structure_violation_count"] = 1
        return report

    frames_by_split: list[set[int]] = [set(), set(), set()]
    pairs_by_split: list[set[tuple[int, int]]] = [set(), set(), set()]
    expected_entries: list[set[str]] = [set(), set(), set()]
    for sequence_id, (row, length, split_id, chunk_id) in enumerate(
        zip(sequences, lengths, sequence_splits, sequence_chunks, strict=True)
    ):
        split = int(split_id)
        count = int(length)
        if split not in range(3) or not min_length <= count <= max_length:
            report["structure_violation_count"] += 1
            continue
        frames = [int(item) for item in row[:count]]
        if (
            len(set(frames)) != len(frames)
            or any(index < 0 or index >= len(frame_ids) for index in frames)
            or np.any(row[count:] != -1)
        ):
            report["structure_violation_count"] += 1
            continue
        split = int(split_id)
        frames_by_split[split].update(frames)
        pairs_by_split[split].update(
            (min(frames[first], frames[second]), max(frames[first], frames[second]))
            for first in range(len(frames))
            for second in range(first + 1, len(frames))
        )
        expected_entries[split].add(f"{SCENE_NAME}/sequence_{sequence_id:06d}")
        if any(int(frame_chunks[index]) != int(chunk_id) for index in frames):
            report["chunk_boundary_violation_count"] += 1
        if not np.asarray(quality_flags[frames], dtype=np.bool_).all():
            report["quality_flag_violation_count"] += 1
        for previous, current in pairwise(frames):
            translation = float(np.linalg.norm(centers[current] - centers[previous]))
            rotation = _rotation_delta_degrees(rotations[previous], rotations[current])
            if current != previous + 1 or translation > max_translation_m or rotation > max_rotation_deg:
                report["motion_threshold_violation_count"] += 1

    for split_id, split_name in enumerate(SPLIT_NAMES):
        path = root / "splits" / f"{split_name}.txt"
        try:
            raw_lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
            actual = set(raw_lines)
            entries_valid = (
                len(actual) == len(raw_lines)
                and all(
                    _SEQUENCE_ENTRY.fullmatch(line) is not None and not Path(line).is_absolute() for line in raw_lines
                )
                and actual == expected_entries[split_id]
            )
        except (OSError, UnicodeError):
            entries_valid = False
        if not entries_valid:
            report["entry_violation_count"] += 1

    frame_overlap = sum(len(frames_by_split[a] & frames_by_split[b]) for a, b in combinations(range(3), 2))
    pair_overlap = sum(len(pairs_by_split[a] & pairs_by_split[b]) for a, b in combinations(range(3), 2))
    report["frame_overlap_count"] = frame_overlap
    report["pair_overlap_count"] = pair_overlap
    return report


def _validate_png_contract(
    root: Path,
    frame_count: int,
    width: int,
    height: int,
    max_depth_mm: int,
) -> int:
    violations = 0
    expected_names = {f"frame_{index:06d}.png" for index in range(frame_count)}
    scene_root = root / "scenes" / SCENE_NAME
    rgb_paths = list((scene_root / "rgb").glob("frame_*.png"))
    depth_paths = list((scene_root / "depth").glob("frame_*.png"))
    if {path.name for path in rgb_paths} != expected_names or {path.name for path in depth_paths} != expected_names:
        violations += 1
    for path in rgb_paths:
        try:
            if path.is_symlink() or not path.is_file():
                raise ValueError("not a regular RGB file")
            with Image.open(path) as image:
                rgb = np.asarray(image)
                if image.mode != "RGB" or image.size != (width, height):
                    raise ValueError("invalid RGB image metadata")
            if rgb.shape != (height, width, 3) or rgb.dtype != np.uint8:
                raise ValueError("invalid RGB pixels")
        except Exception:
            violations += 1
    for path in depth_paths:
        try:
            if path.is_symlink() or not path.is_file():
                raise ValueError("not a regular depth file")
            with Image.open(path) as image:
                depth = np.asarray(image)
                if image.size != (width, height):
                    raise ValueError("invalid depth image metadata")
            if depth.shape != (height, width) or depth.dtype != np.uint16:
                raise ValueError("invalid depth pixels")
            if not np.any(depth) or int(depth.max(initial=0)) > max_depth_mm:
                raise ValueError("depth is empty or exceeds the configured limit")
        except Exception:
            violations += 1
    return violations


def validate_staging(
    root: Path,
    private_tokens: Sequence[str] = (),
    *,
    require_report: bool = True,
) -> dict[str, Any]:
    """Validate structure and privacy without returning any matched token text."""

    root = Path(root)
    paths = list(root.rglob("*")) if root.is_dir() else []
    symlink_count = int(root.is_symlink()) + sum(path.is_symlink() for path in paths)
    non_regular_file_count = sum(not path.is_dir() and not path.is_file() and not path.is_symlink() for path in paths)
    invalid_path_count = sum(
        not _allowed_relative_path(path.relative_to(root).as_posix(), path.is_dir()) for path in paths
    )
    token_pattern, longest_token = _compile_token_pattern(private_tokens)
    private_token_hit_count = sum(_contains_token(path, token_pattern, longest_token) for path in paths)
    png_metadata_count = 0
    unreadable_png_count = 0
    for path in root.rglob("*.png") if root.is_dir() else ():
        if path.is_symlink():
            continue
        try:
            with Image.open(path) as image:
                if image.info:
                    png_metadata_count += 1
        except (OSError, ValueError):
            unreadable_png_count += 1

    absolute_string_count = 0
    for path in root.rglob("*.json") if root.is_dir() else ():
        if path.is_symlink():
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError):
            absolute_string_count += 1
            continue
        for value in _walk_strings(payload):
            if Path(value).is_absolute():
                absolute_string_count += 1

    dataset_path = root / "dataset.json"
    dataset: Mapping[str, Any] | None = None
    dataset_ok = False
    frame_count = 0
    rgb_count = len(list(root.glob(f"scenes/{SCENE_NAME}/rgb/frame_*.png"))) if root.is_dir() else 0
    depth_count = len(list(root.glob(f"scenes/{SCENE_NAME}/depth/frame_*.png"))) if root.is_dir() else 0
    camera_count = 0
    frame_contract_violation_count = 0
    scene_contract_violation_count = 0
    try:
        dataset = json.loads(dataset_path.read_text())
        if not isinstance(dataset, Mapping):
            raise ValueError("dataset metadata must be an object")
        if set(dataset) != {
            "schema_version",
            "format",
            "scene_count",
            "frame_count",
            "image",
            "depth",
            "camera",
            "sequences",
            "overlap",
            "splits",
        }:
            raise ValueError("dataset metadata contains unexpected fields")
        image_metadata = dataset["image"]
        depth_metadata = dataset["depth"]
        camera_metadata = dataset["camera"]
        sequence_metadata = dataset["sequences"]
        overlap_metadata = dataset["overlap"]
        split_metadata = dataset["splits"]
        if (
            set(image_metadata) != {"height", "width", "channels", "dtype"}
            or set(depth_metadata)
            != {
                "height",
                "width",
                "dtype",
                "unit",
                "invalid_value",
                "mapped_depth_dilation_kernel",
                "mapped_depth_method",
                "additional_dilation",
                "max_depth_mm",
            }
            or set(camera_metadata) != {"extrinsics", "intrinsics", "source"}
            or set(sequence_metadata)
            != {
                "min_length",
                "max_length",
                "max_translation_m",
                "max_rotation_deg",
                "guard_frames",
            }
            or set(overlap_metadata)
            != {
                "schema_version",
                "filename",
                "aggregation",
                "relative_depth_tolerance",
                "pixel_stride",
                "near_depth_m",
            }
            or set(split_metadata) != set(SPLIT_NAMES)
        ):
            raise ValueError("dataset nested metadata contains unexpected fields")
        frame_count = int(dataset["frame_count"])
        width = int(image_metadata["width"])
        height = int(image_metadata["height"])
        max_depth_mm = int(depth_metadata["max_depth_mm"])
        camera_path = root / f"scenes/{SCENE_NAME}/cameras.npz"
        with np.load(camera_path, allow_pickle=False) as cameras:
            if set(cameras.files) != {"frame_ids", "intrinsics", "extrinsics_w2c", "quality_flags", "chunk_ids"}:
                raise ValueError("unexpected camera arrays")
            camera_count = len(cameras["frame_ids"])
            frame_ids = cameras["frame_ids"]
            intrinsics = cameras["intrinsics"]
            extrinsics = cameras["extrinsics_w2c"]
            quality_flags = cameras["quality_flags"]
            chunk_ids = cameras["chunk_ids"]
        rotations = extrinsics[:, :3, :3]
        sequence_path = root / f"scenes/{SCENE_NAME}/sequences.npz"
        with np.load(sequence_path, allow_pickle=False) as sequence_arrays:
            sequence_shape = sequence_arrays["sequences"].shape
            sequence_lengths = sequence_arrays["lengths"]
        overlap_path = root / f"scenes/{SCENE_NAME}/{overlap_metadata['filename']}"
        with np.load(overlap_path, allow_pickle=False) as profile:
            if set(profile.files) != {"all_depth", "near_depth", "pair_valid"}:
                raise ValueError("unexpected overlap arrays")
            overlap_all = profile["all_depth"]
            overlap_near = profile["near_depth"]
            overlap_valid = profile["pair_valid"]
        expected_overlap_shape = (sequence_shape[0], sequence_shape[1], sequence_shape[1])
        expected_pair_valid = np.zeros(expected_overlap_shape, dtype=np.bool_)
        for sequence_id, length in enumerate(sequence_lengths):
            count = int(length)
            expected_pair_valid[sequence_id, :count, :count] = True
            np.fill_diagonal(expected_pair_valid[sequence_id], False)
        overlap_arrays_ok = bool(
            overlap_all.shape == expected_overlap_shape
            and overlap_near.shape == expected_overlap_shape
            and overlap_valid.shape == expected_overlap_shape
            and overlap_all.dtype == np.float32
            and overlap_near.dtype == np.float32
            and overlap_valid.dtype == np.bool_
            and np.isfinite(overlap_all).all()
            and np.isfinite(overlap_near).all()
            and ((overlap_all >= 0) & (overlap_all <= 1)).all()
            and ((overlap_near >= 0) & (overlap_near <= 1)).all()
            and np.allclose(overlap_all, overlap_all.transpose(0, 2, 1), atol=1e-6)
            and np.allclose(overlap_near, overlap_near.transpose(0, 2, 1), atol=1e-6)
            and np.array_equal(overlap_valid, expected_pair_valid)
            and np.all(overlap_all[~overlap_valid] == 0)
            and np.all(overlap_near[~overlap_valid] == 0)
        )
        camera_arrays_ok = bool(
            frame_ids.shape == (camera_count,)
            and frame_ids.dtype == np.int64
            and frame_ids.tolist() == list(range(camera_count))
            and intrinsics.shape == (camera_count, 3, 3)
            and intrinsics.dtype == np.float32
            and extrinsics.shape == (camera_count, 3, 4)
            and extrinsics.dtype == np.float32
            and quality_flags.shape == (camera_count,)
            and quality_flags.dtype == np.bool_
            and chunk_ids.shape == (camera_count,)
            and chunk_ids.dtype == np.int64
            and np.isfinite(intrinsics).all()
            and np.isfinite(extrinsics).all()
            and (intrinsics[:, 0, 0] > 0).all()
            and (intrinsics[:, 1, 1] > 0).all()
            and np.allclose(intrinsics[:, 2, :], (0.0, 0.0, 1.0), atol=1e-6)
            and np.allclose(intrinsics[:, 0, 2], width / 2, atol=1e-4)
            and np.allclose(intrinsics[:, 1, 2], height / 2, atol=1e-4)
            and np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-4)
            and np.allclose(np.linalg.det(rotations), 1.0, atol=1e-4)
        )
        metadata_ok = bool(
            dataset.get("schema_version") == 1
            and dataset.get("format") == DATASET_FORMAT
            and dataset.get("scene_count") == 1
            and frame_count > 0
            and width > 0
            and height > 0
            and width % 16 == 0
            and height % 16 == 0
            and image_metadata.get("channels") == 3
            and image_metadata.get("dtype") == "uint8"
            and depth_metadata.get("height") == height
            and depth_metadata.get("width") == width
            and depth_metadata.get("dtype") == "uint16"
            and depth_metadata.get("unit") == "millimeters"
            and depth_metadata.get("invalid_value") == 0
            and depth_metadata.get("mapped_depth_dilation_kernel") == 3
            and depth_metadata.get("mapped_depth_method") == "nearest_depth"
            and depth_metadata.get("additional_dilation") is False
            and max_depth_mm > 0
            and camera_metadata.get("extrinsics") == "opencv_world_to_camera"
            and camera_metadata.get("intrinsics") == "pixel_units"
            and camera_metadata.get("source") == "aligned_trajectory_camera_to_world_inverted"
            and overlap_metadata.get("schema_version") == 1
            and overlap_metadata.get("filename") == "overlap.npz"
            and overlap_metadata.get("aggregation") == "bidirectional_mean"
            and math.isfinite(float(overlap_metadata.get("relative_depth_tolerance", 0)))
            and float(overlap_metadata.get("relative_depth_tolerance", 0)) > 0
            and isinstance(overlap_metadata.get("pixel_stride"), int)
            and not isinstance(overlap_metadata.get("pixel_stride"), bool)
            and int(overlap_metadata.get("pixel_stride", 0)) > 0
            and math.isfinite(float(overlap_metadata.get("near_depth_m", 0)))
            and float(overlap_metadata.get("near_depth_m", 0)) > 0
            and overlap_arrays_ok
        )
        frame_contract_violation_count = _validate_png_contract(
            root,
            frame_count=frame_count,
            width=width,
            height=height,
            max_depth_mm=max_depth_mm,
        )
        scene_dir = root / "scenes"
        scene_names = {path.name for path in scene_dir.iterdir() if path.is_dir() and not path.is_symlink()}
        scene_contract_violation_count = int(scene_names != {SCENE_NAME})
        report_path = root / "reports" / "export_validation.json"
        if require_report and (report_path.is_symlink() or not report_path.is_file()):
            scene_contract_violation_count += 1
        dataset_ok = bool(
            metadata_ok
            and camera_arrays_ok
            and frame_count == rgb_count == depth_count == camera_count
            and frame_contract_violation_count == 0
            and scene_contract_violation_count == 0
        )
    except Exception:
        dataset_ok = False
        frame_contract_violation_count = max(frame_contract_violation_count, 1)
    split_report = _split_validation(root, dataset)
    privacy = {
        "symlink_count": symlink_count,
        "non_regular_file_count": non_regular_file_count,
        "invalid_path_count": invalid_path_count,
        "private_token_hit_count": private_token_hit_count,
        "png_metadata_count": png_metadata_count,
        "unreadable_png_count": unreadable_png_count,
        "absolute_string_count": absolute_string_count,
    }
    valid = dataset_ok and not any(privacy.values()) and not any(split_report.values())
    return {
        "schema_version": 1,
        "valid": bool(valid),
        "privacy": privacy,
        "dataset": {
            "contract_valid": bool(dataset_ok),
            "frame_count": frame_count,
            "rgb_count": rgb_count,
            "depth_count": depth_count,
            "camera_count": camera_count,
            "frame_contract_violation_count": frame_contract_violation_count,
            "scene_contract_violation_count": scene_contract_violation_count,
        },
        "splits": split_report,
    }


def export_colmap_rgbd_training(
    source_rgbd_root: Path,
    trajectory_path: Path,
    rgb_camera_yaml: Path,
    output_root: Path,
    *,
    height: int = 384,
    width: int = 512,
    max_depth_mm: int = 1300,
    expected_dilation_kernel: int = 3,
    expected_frame_count: int | None = None,
    min_sequence_length: int = 2,
    max_sequence_length: int = 4,
    max_translation_m: float = 0.30,
    max_rotation_deg: float = 3.0,
    guard_frames: int = 12,
    split_fractions: tuple[float, float, float] = (0.8, 0.15, 0.05),
) -> dict[str, Any]:
    """Create one anonymous scene from a private RGB-D sequence and aligned poses."""

    source_rgbd_root = Path(source_rgbd_root)
    trajectory_path = Path(trajectory_path)
    rgb_camera_yaml = Path(rgb_camera_yaml)
    output_root = Path(output_root)
    if output_root.is_symlink():
        raise ExportContractError("output_root must not be a symlink")
    if output_root.exists() and any(output_root.iterdir()):
        raise ExportContractError("output_root already exists and is not empty")
    if expected_frame_count is not None and expected_frame_count <= 0:
        raise ExportContractError("expected_frame_count must be positive when provided")
    if expected_dilation_kernel != 3:
        raise ExportContractError("only the verified 3x3 mapped-depth provenance is accepted")
    readme = source_rgbd_root / "README.md"
    provenance = readme.read_text().lower() if readme.is_file() else ""
    if "3x3" not in provenance or "nearest-depth dilation" not in provenance:
        raise ExportContractError("mapped depth provenance does not confirm 3x3 nearest-depth dilation")
    if width % 16 or height % 16:
        raise ExportContractError("target width and height must be multiples of patch size 16")
    if max_depth_mm <= 0:
        raise ExportContractError("max_depth_mm must be positive")

    rgb_paths = sorted((source_rgbd_root / "rgb").glob("*.png"))
    depth_paths = sorted((source_rgbd_root / "mapped_depth").glob("*.png"))
    if not rgb_paths or len(rgb_paths) != len(depth_paths):
        raise ExportContractError("RGB and mapped-depth counts must be equal and non-zero")
    rgb_by_key = {_pair_key(path, "rgb"): path for path in rgb_paths}
    depth_by_key = {_pair_key(path, "depth"): path for path in depth_paths}
    if (
        len(rgb_by_key) != len(rgb_paths)
        or len(depth_by_key) != len(depth_paths)
        or set(rgb_by_key) != set(depth_by_key)
    ):
        raise ExportContractError("RGB and mapped-depth stems do not have a one-to-one match")
    frames, chunk_ids = _load_trajectory(trajectory_path, {path.name for path in rgb_paths})
    if expected_frame_count is not None and len(frames) != expected_frame_count:
        raise ExportContractError(
            f"source does not match the expected frame count ({len(frames)} != {expected_frame_count})"
        )
    source_width, source_height, intrinsics, serial = _read_rgb_camera_yaml(rgb_camera_yaml)
    crop, updated_intrinsics = compute_principal_crop(intrinsics, (source_width, source_height), (width, height))

    frame_count = len(frames)
    extrinsics = np.stack(
        [_invert_camera_to_world(frame["camera_to_world"], index) for index, frame in enumerate(frames)]
    ).astype(np.float32)
    split_labels = _split_labels(frame_count, guard_frames, split_fractions, min_sequence_length)
    sequences, sequence_lengths, sequence_splits, sequence_chunks = _make_sequences(
        extrinsics,
        chunk_ids,
        split_labels,
        min_sequence_length,
        max_sequence_length,
        max_translation_m,
        max_rotation_deg,
    )

    scene_root = output_root / f"scenes/{SCENE_NAME}"
    rgb_output = scene_root / "rgb"
    depth_output = scene_root / "depth"
    rgb_output.mkdir(parents=True, exist_ok=True)
    depth_output.mkdir(parents=True, exist_ok=True)
    for generic_index, frame in enumerate(frames):
        rgb_path = source_rgbd_root / "rgb" / str(frame["image_name"])
        if not rgb_path.is_file():
            raise ExportContractError(f"trajectory frame {generic_index} has no RGB file")
        key = _pair_key(rgb_path, "rgb")
        depth_path = depth_by_key.get(key)
        if depth_path is None:
            raise ExportContractError(f"trajectory frame {generic_index} has no mapped depth")
        with Image.open(rgb_path) as image:
            if image.size != (source_width, source_height):
                raise ExportContractError(f"RGB frame {generic_index} has an unexpected shape")
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        with Image.open(depth_path) as image:
            depth = np.asarray(image)
        if depth.shape != (source_height, source_width) or depth.dtype != np.uint16:
            raise ExportContractError(f"mapped depth frame {generic_index} must be uint16 on the RGB pixel grid")
        if int(depth.max(initial=0)) > max_depth_mm:
            raise ExportContractError(f"mapped depth frame {generic_index} exceeds max_depth_mm")
        generic_name = f"frame_{generic_index:06d}.png"
        exported_rgb = _warp_image(rgb, crop, (width, height), depth=False).astype(np.uint8)
        exported_depth = _warp_image(depth, crop, (width, height), depth=True).astype(np.uint16)
        Image.fromarray(exported_rgb).save(rgb_output / generic_name, optimize=False)
        Image.fromarray(exported_depth).save(depth_output / generic_name, optimize=False)

    repeated_intrinsics = np.repeat(updated_intrinsics[None], frame_count, axis=0).astype(np.float32)
    np.savez_compressed(
        scene_root / "cameras.npz",
        frame_ids=np.arange(frame_count, dtype=np.int64),
        intrinsics=repeated_intrinsics,
        extrinsics_w2c=extrinsics,
        quality_flags=np.ones(frame_count, dtype=np.bool_),
        chunk_ids=chunk_ids,
    )
    np.savez_compressed(
        scene_root / "sequences.npz",
        sequences=sequences,
        lengths=sequence_lengths,
        split_ids=sequence_splits,
        chunk_ids=sequence_chunks,
    )
    overlap_relative_depth_tolerance = 0.03
    overlap_pixel_stride = 8
    overlap_near_depth_m = 1.2
    _write_overlap_profile(
        output_root,
        sequences,
        sequence_lengths,
        repeated_intrinsics,
        extrinsics,
        relative_depth_tolerance=overlap_relative_depth_tolerance,
        pixel_stride=overlap_pixel_stride,
        near_depth_m=overlap_near_depth_m,
    )
    _write_split_files(output_root, sequence_splits)
    dataset = {
        "schema_version": 1,
        "format": DATASET_FORMAT,
        "scene_count": 1,
        "frame_count": frame_count,
        "image": {"height": height, "width": width, "channels": 3, "dtype": "uint8"},
        "depth": {
            "height": height,
            "width": width,
            "dtype": "uint16",
            "unit": "millimeters",
            "invalid_value": 0,
            "mapped_depth_dilation_kernel": expected_dilation_kernel,
            "mapped_depth_method": "nearest_depth",
            "additional_dilation": False,
            "max_depth_mm": max_depth_mm,
        },
        "camera": {
            "extrinsics": "opencv_world_to_camera",
            "intrinsics": "pixel_units",
            "source": "aligned_trajectory_camera_to_world_inverted",
        },
        "sequences": {
            "min_length": min_sequence_length,
            "max_length": max_sequence_length,
            "max_translation_m": max_translation_m,
            "max_rotation_deg": max_rotation_deg,
            "guard_frames": guard_frames,
        },
        "overlap": {
            "schema_version": 1,
            "filename": "overlap.npz",
            "aggregation": "bidirectional_mean",
            "relative_depth_tolerance": overlap_relative_depth_tolerance,
            "pixel_stride": overlap_pixel_stride,
            "near_depth_m": overlap_near_depth_m,
        },
        "splits": {name: int(np.sum(sequence_splits == index)) for index, name in enumerate(SPLIT_NAMES)},
    }
    (output_root / "dataset.json").write_text(json.dumps(dataset, indent=2, sort_keys=True) + "\n")
    (output_root / "reports").mkdir(parents=True, exist_ok=True)
    tokens = _private_tokens(source_rgbd_root, trajectory_path, rgb_camera_yaml, rgb_paths, serial)
    report = validate_staging(output_root, private_tokens=tokens, require_report=False)
    (output_root / "reports/export_validation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not report["valid"]:
        failed = sorted(key for key, value in report["privacy"].items() if value)
        failed.extend(key for key, value in report["splits"].items() if value)
        if not report["dataset"]["contract_valid"]:
            failed.append("dataset_contract")
        raise ExportContractError(f"staging validation failed: {', '.join(failed)}")
    return report


def prepare_overlap_profile(
    output_root: Path,
    *,
    relative_depth_tolerance: float = 0.03,
    pixel_stride: int = 8,
    near_depth_m: float = 1.2,
) -> dict[str, Any]:
    """Upgrade an existing anonymous staging set with a numeric overlap profile."""

    output_root = Path(output_root)
    dataset_path = output_root / "dataset.json"
    scene_root = output_root / f"scenes/{SCENE_NAME}"
    overlap_path = scene_root / "overlap.npz"
    try:
        dataset = json.loads(dataset_path.read_text())
    except (OSError, ValueError) as error:
        raise ExportContractError("dataset.json is missing or invalid") from error
    if not isinstance(dataset, dict) or dataset.get("format") != DATASET_FORMAT:
        raise ExportContractError("dataset format is not the supported anonymous staging contract")
    if "overlap" in dataset or overlap_path.exists():
        raise ExportContractError("overlap profile already exists; refusing to overwrite it")
    try:
        with np.load(scene_root / "cameras.npz", allow_pickle=False) as cameras:
            if set(cameras.files) != {"frame_ids", "intrinsics", "extrinsics_w2c", "quality_flags", "chunk_ids"}:
                raise ValueError("unexpected camera arrays")
            intrinsics = cameras["intrinsics"].copy()
            extrinsics = cameras["extrinsics_w2c"].copy()
        with np.load(scene_root / "sequences.npz", allow_pickle=False) as stored_sequences:
            if set(stored_sequences.files) != {"sequences", "lengths", "split_ids", "chunk_ids"}:
                raise ValueError("unexpected sequence arrays")
            sequences = stored_sequences["sequences"].copy()
            lengths = stored_sequences["lengths"].copy()
    except (OSError, ValueError, KeyError) as error:
        raise ExportContractError("anonymous camera or sequence arrays are invalid") from error
    _write_overlap_profile(
        output_root,
        sequences,
        lengths,
        intrinsics,
        extrinsics,
        relative_depth_tolerance=relative_depth_tolerance,
        pixel_stride=pixel_stride,
        near_depth_m=near_depth_m,
    )
    dataset["overlap"] = {
        "schema_version": 1,
        "filename": "overlap.npz",
        "aggregation": "bidirectional_mean",
        "relative_depth_tolerance": relative_depth_tolerance,
        "pixel_stride": pixel_stride,
        "near_depth_m": near_depth_m,
    }
    dataset_path.write_text(json.dumps(dataset, indent=2, sort_keys=True) + "\n")
    report = validate_staging(output_root)
    (output_root / "reports/export_validation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not report["valid"]:
        raise ExportContractError("overlap profile upgrade failed strict staging validation")
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-rgbd-root", type=Path)
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--rgb-camera-yaml", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--prepare-overlap-profile", action="store_true")
    parser.add_argument("--private-token", action="append", default=[])
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--max-depth-mm", type=int, default=1300)
    parser.add_argument("--expected-dilation-kernel", type=int, default=3)
    parser.add_argument("--expected-frame-count", type=int)
    parser.add_argument("--guard-frames", type=int, default=12)
    parser.add_argument("--max-translation-m", type=float, default=0.30)
    parser.add_argument("--max-rotation-deg", type=float, default=3.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.validate_only and args.prepare_overlap_profile:
        print("error: --validate-only and --prepare-overlap-profile are mutually exclusive")
        return 2
    if args.prepare_overlap_profile:
        try:
            report = prepare_overlap_profile(args.output_root)
        except (ExportContractError, OSError, ValueError) as error:
            print(f"error: {error}")
            return 2
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.validate_only:
        source_arguments = (args.source_rgbd_root, args.trajectory, args.rgb_camera_yaml)
        if any(argument is not None for argument in source_arguments) and not all(
            argument is not None for argument in source_arguments
        ):
            print("error: all three private source arguments are required when any is used for validation")
            return 2
        private_tokens = list(args.private_token)
        if all(argument is not None for argument in source_arguments):
            source_root = cast(Path, args.source_rgbd_root)
            trajectory_path = cast(Path, args.trajectory)
            camera_yaml = cast(Path, args.rgb_camera_yaml)
            try:
                rgb_paths = sorted((source_root / "rgb").glob("*.png"))
                _, _, _, serial = _read_rgb_camera_yaml(camera_yaml)
                private_tokens.extend(_private_tokens(source_root, trajectory_path, camera_yaml, rgb_paths, serial))
            except (ExportContractError, OSError, ValueError) as error:
                print(f"error: cannot derive privacy checks from private source inputs: {error}")
                return 2
        report = validate_staging(args.output_root, private_tokens=private_tokens)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["valid"] else 2
    try:
        if args.source_rgbd_root is None or args.trajectory is None or args.rgb_camera_yaml is None:
            raise ExportContractError(
                "--source-rgbd-root, --trajectory, and --rgb-camera-yaml are required unless --validate-only is used"
            )
        report = export_colmap_rgbd_training(
            source_rgbd_root=args.source_rgbd_root,
            trajectory_path=args.trajectory,
            rgb_camera_yaml=args.rgb_camera_yaml,
            output_root=args.output_root,
            height=args.height,
            width=args.width,
            max_depth_mm=args.max_depth_mm,
            expected_dilation_kernel=args.expected_dilation_kernel,
            expected_frame_count=args.expected_frame_count,
            guard_frames=args.guard_frames,
            max_translation_m=args.max_translation_m,
            max_rotation_deg=args.max_rotation_deg,
        )
    except (ExportContractError, OSError, ValueError) as error:
        print(f"error: {error}")
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
