"""Build and validate a small RGB-D/VGGT4D/COLMAP integration pilot.

The script keeps the source dataset immutable.  It creates a manifest-backed
view of selected RGB/depth frames, derives a metric rail trajectory from
VGGT4D outputs, and filters a copied COLMAP match database with measured depth,
dynamic masks, and the scaled camera poses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sqlite3
import struct
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

MAX_IMAGE_ID = 2_147_483_647

# Repository and cache locations are resolved from the environment so the
# script is not tied to one machine's layout.  The defaults assume the sibling
# repositories sit next to this checkout.
VGGT_OMEGA_REPO = Path(os.environ.get("VGGT_OMEGA_REPO", Path(__file__).resolve().parents[1]))
SIBLING_ROOT = Path(os.environ.get("RGBD_WORKFLOW_REPO_ROOT", VGGT_OMEGA_REPO.parent))
VGGT4D_REPO = Path(os.environ.get("VGGT4D_REPO", SIBLING_ROOT / "VGGT4D"))
OPEND4RT_REPO = Path(os.environ.get("OPEND4RT_REPO", SIBLING_ROOT / "Open-d4rt"))
MAMBAGLUE_REPO = Path(os.environ.get("MAMBAGLUE_REPO", SIBLING_ROOT / "MambaGlue"))
COLMAP_CACHE_DIR = Path(os.environ.get("COLMAP_CACHE_DIR", Path.home() / ".cache" / "colmap"))

DEFAULT_VGGT4D_CHECKPOINT = VGGT4D_REPO / "ckpts" / "model_tracker_fixed_e20.pt"
DEFAULT_VGGT_OMEGA_CHECKPOINT = VGGT_OMEGA_REPO / "checkpoints" / "vggt_omega_1b_512.pt"
DEFAULT_OPEND4RT_CHECKPOINT = (
    OPEND4RT_REPO / "checkpoints" / "OpenD4RT_32CLIP_9Dataset_NoAUG" / "opend4rt.ckpt"
)
DEFAULT_MAMBAGLUE_CHECKPOINT = (
    MAMBAGLUE_REPO / ".cache" / "torch" / "hub" / "checkpoints" / "superpoint_mambaglue_v0.1.tar"
)
DEFAULT_INTRINSICS = (554.6940307617188, 561.6201782226562, 395.7606150224128, 295.52493750108624)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _depth_name_for_rgb(image_name: str) -> str:
    suffix = "_rgb.png"
    if not image_name.endswith(suffix):
        raise ValueError(f"RGB filename does not end with {suffix!r}: {image_name}")
    return f"{image_name[: -len(suffix)]}_depth.png"


def _safe_symlink(source: Path, destination: Path) -> None:
    if destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        raise FileExistsError(f"symlink points to a different source: {destination}")
    if destination.exists():
        raise FileExistsError(f"pilot path already exists and is not the expected symlink: {destination}")
    destination.symlink_to(source.resolve())


def _build_manifest(source: Path, start: int, stride: int, count: int) -> dict[str, Any]:
    if start < 0:
        raise ValueError("start must be non-negative")
    if stride < 1:
        raise ValueError("stride must be at least 1")
    if count < 1:
        raise ValueError("count must be at least 1")
    rgb_dir = source / "rgb"
    depth_dir = source / "mapped_depth"
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"RGB directory does not exist: {rgb_dir}")
    if not depth_dir.is_dir():
        raise FileNotFoundError(f"mapped depth directory does not exist: {depth_dir}")

    rgb_paths = sorted(rgb_dir.glob("*.png"))
    selected = rgb_paths[start : start + stride * count : stride]
    if len(selected) != count:
        raise ValueError(f"requested {count} frames but selected {len(selected)} from {len(rgb_paths)} RGB files")

    first_image = cv2.imread(str(selected[0]), cv2.IMREAD_UNCHANGED)
    if first_image is None:
        raise ValueError(f"could not read RGB image: {selected[0]}")
    height, width = first_image.shape[:2]
    frames: list[dict[str, Any]] = []
    for frame_index, (source_index, rgb_path) in enumerate(
        zip(range(start, start + stride * count, stride), selected, strict=True)
    ):
        depth_path = depth_dir / _depth_name_for_rgb(rgb_path.name)
        if not depth_path.is_file():
            raise FileNotFoundError(f"mapped depth is missing for {rgb_path.name}: {depth_path}")
        image = cv2.imread(str(rgb_path), cv2.IMREAD_UNCHANGED)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if image is None or image.shape[:2] != (height, width):
            raise ValueError(f"RGB shape differs from {(height, width)}: {rgb_path}")
        if depth is None or depth.shape[:2] != (height, width) or depth.dtype != np.uint16:
            actual = None if depth is None else (depth.shape, str(depth.dtype))
            raise ValueError(f"mapped depth must be {width}x{height} uint16: {depth_path} ({actual})")
        frames.append(
            {
                "frame_index": frame_index,
                "source_index": source_index,
                "image_name": rgb_path.name,
                "depth_name": depth_path.name,
                "rgb_source": str(rgb_path.resolve()),
                "depth_source": str(depth_path.resolve()),
                "image_sha256": sha256_file(rgb_path),
                "depth_sha256": sha256_file(depth_path),
            }
        )
    return {
        "schema_version": 1,
        "source_dataset": str(source.resolve()),
        "selection": {"start": start, "stride": stride, "count": count},
        "image_width": width,
        "image_height": height,
        "depth_unit": "millimeters_uint16_zero_invalid",
        "frames": frames,
    }


def prepare_pilot(source: Path, output: Path, start: int, stride: int, count: int) -> dict[str, Any]:
    """Create an idempotent manifest and symlink view without modifying source."""
    source = source.resolve()
    manifest = _build_manifest(source, start=start, stride=stride, count=count)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing != manifest:
            raise FileExistsError(f"pilot has a different manifest and will not be overwritten: {manifest_path}")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f"non-empty pilot output has no manifest and will not be modified: {output}")

    image_dir = output / "images"
    depth_dir = output / "mapped_depth"
    log_dir = output / "logs"
    colmap_dir = output / "colmap"
    for directory in (image_dir, depth_dir, log_dir, colmap_dir):
        directory.mkdir(parents=True, exist_ok=True)
    for frame in manifest["frames"]:
        _safe_symlink(Path(frame["rgb_source"]), image_dir / frame["image_name"])
        _safe_symlink(Path(frame["depth_source"]), depth_dir / frame["depth_name"])
    _write_json(manifest_path, manifest)
    return manifest


def robust_metric_scale(
    measured_depths_m: Sequence[np.ndarray],
    predicted_depths: Sequence[np.ndarray],
    min_depth_m: float = 0.1,
    max_depth_m: float = 1.3,
) -> dict[str, Any]:
    if len(measured_depths_m) != len(predicted_depths):
        raise ValueError("measured and predicted depth sequences must have equal length")
    ratios: list[np.ndarray] = []
    per_frame_counts: list[int] = []
    for measured, predicted in zip(measured_depths_m, predicted_depths, strict=True):
        if measured.shape != predicted.shape:
            raise ValueError(f"depth shapes differ: {measured.shape} != {predicted.shape}")
        valid = (
            np.isfinite(measured)
            & np.isfinite(predicted)
            & (measured >= min_depth_m)
            & (measured <= max_depth_m)
            & (predicted > 0.0)
        )
        frame_ratios = measured[valid].astype(np.float64) / predicted[valid].astype(np.float64)
        per_frame_counts.append(int(frame_ratios.size))
        if frame_ratios.size:
            ratios.append(frame_ratios)
    if not ratios:
        raise ValueError("no valid depth ratios are available for metric scale")
    all_ratios = np.concatenate(ratios)
    low, high = np.percentile(all_ratios, [5.0, 95.0])
    trimmed = all_ratios[(all_ratios >= low) & (all_ratios <= high)]
    if not trimmed.size:
        raise ValueError("no valid depth ratios remain after percentile trimming")
    scale = float(np.median(trimmed))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"metric scale is not positive and finite: {scale}")
    return {
        "scale": scale,
        "raw_sample_count": int(all_ratios.size),
        "trimmed_sample_count": int(trimmed.size),
        "ratio_percentile_05": float(low),
        "ratio_percentile_95": float(high),
        "per_frame_valid_counts": per_frame_counts,
        "min_depth_m": min_depth_m,
        "max_depth_m": max_depth_m,
    }


def quaternion_wxyz_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    qw, qx, qy, qz = np.asarray(quaternion, dtype=np.float64)
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("quaternion must have a positive finite norm")
    qw, qx, qy, qz = (qw / norm, qx / norm, qy / norm, qz / norm)
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def _rotation_difference_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.T @ second
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def analyze_rail_keyframes(
    positions: np.ndarray,
    rotations: np.ndarray,
    min_translation_m: float = 0.08,
    min_rotation_deg: float = 3.0,
    max_gap: int = 3,
) -> dict[str, Any]:
    positions = np.asarray(positions, dtype=np.float64)
    rotations = np.asarray(rotations, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 2:
        raise ValueError("positions must have shape (N, 3) with N >= 2")
    if rotations.shape != (len(positions), 3, 3):
        raise ValueError("rotations must have shape (N, 3, 3)")
    if max_gap < 1:
        raise ValueError("max_gap must be at least 1")
    if not np.isfinite(positions).all() or not np.isfinite(rotations).all():
        raise ValueError("positions and rotations must be finite")

    centroid = positions.mean(axis=0)
    centered = positions - centroid
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    rail_axis = vh[0]
    if np.dot(rail_axis, positions[-1] - positions[0]) < 0.0:
        rail_axis = -rail_axis
    centered_along = centered @ rail_axis
    orthogonal = centered - np.outer(centered_along, rail_axis)
    residuals = np.linalg.norm(orthogonal, axis=1)
    coordinates = (positions - positions[0]) @ rail_axis

    keyframes = [0]
    reasons: dict[int, list[str]] = {0: ["first"]}
    last = 0
    for index in range(1, len(positions) - 1):
        translation = abs(float(coordinates[index] - coordinates[last]))
        rotation = _rotation_difference_deg(rotations[last], rotations[index])
        selected_for: list[str] = []
        if translation >= min_translation_m:
            selected_for.append("rail_translation")
        if rotation >= min_rotation_deg:
            selected_for.append("rotation")
        if index - last >= max_gap:
            selected_for.append("max_gap")
        if selected_for:
            keyframes.append(index)
            reasons[index] = selected_for
            last = index
    final_index = len(positions) - 1
    if final_index not in keyframes:
        keyframes.append(final_index)
        reasons[final_index] = ["last"]
    elif "last" not in reasons[final_index]:
        reasons[final_index].append("last")

    explained = 0.0
    total = float(np.sum(singular_values**2))
    if total > 0.0:
        explained = float(singular_values[0] ** 2 / total)
    return {
        "rail_axis": rail_axis.tolist(),
        "rail_centroid_m": centroid.tolist(),
        "rail_coordinates_m": coordinates.tolist(),
        "orthogonal_residuals_m": residuals.tolist(),
        "orthogonal_rms_m": float(np.sqrt(np.mean(residuals**2))),
        "orthogonal_max_m": float(np.max(residuals)),
        "rail_explained_variance_ratio": explained,
        "keyframe_indices": keyframes,
        "keyframe_reasons": {str(index): reasons[index] for index in keyframes},
        "thresholds": {
            "min_translation_m": min_translation_m,
            "min_rotation_deg": min_rotation_deg,
            "max_gap": max_gap,
        },
    }


def _load_tum_poses(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.atleast_2d(np.loadtxt(path, dtype=np.float64))
    if rows.shape[1] != 8:
        raise ValueError(f"TUM trajectory must have 8 columns: {path}")
    timestamps = rows[:, 0]
    positions = rows[:, 1:4]
    rotations = np.stack([quaternion_wxyz_to_matrix(row[4:8]) for row in rows])
    return timestamps, positions, rotations


def analyze_trajectory(
    pilot_root: Path,
    vggt_dir: Path,
    min_depth_m: float = 0.1,
    max_depth_m: float = 1.3,
    min_translation_m: float = 0.08,
    min_rotation_deg: float = 3.0,
    max_gap: int = 3,
) -> dict[str, Any]:
    manifest = json.loads((pilot_root / "manifest.json").read_text())
    frames = manifest["frames"]
    timestamps, positions, rotations = _load_tum_poses(vggt_dir / "pred_traj.txt")
    if len(frames) != len(positions):
        raise ValueError(f"manifest/VGGT4D frame count differs: {len(frames)} != {len(positions)}")

    predicted_depths: list[np.ndarray] = []
    measured_depths: list[np.ndarray] = []
    output_shape: tuple[int, int] | None = None
    for index, frame in enumerate(frames):
        predicted = np.load(vggt_dir / f"frame_{index:04d}.npy").astype(np.float32)
        if predicted.ndim != 2:
            raise ValueError(f"predicted depth must be 2-D: frame {index} has {predicted.shape}")
        if output_shape is None:
            output_shape = predicted.shape
        elif predicted.shape != output_shape:
            raise ValueError(f"VGGT4D depth shapes differ: {predicted.shape} != {output_shape}")
        measured_mm = cv2.imread(str(pilot_root / "mapped_depth" / frame["depth_name"]), cv2.IMREAD_UNCHANGED)
        if measured_mm is None or measured_mm.dtype != np.uint16:
            raise ValueError(f"could not read uint16 mapped depth for frame {index}")
        measured = cv2.resize(
            measured_mm.astype(np.float32) / 1000.0,
            (predicted.shape[1], predicted.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        predicted_depths.append(predicted)
        measured_depths.append(measured)

    expected_shape = (392, 518) if (manifest["image_width"], manifest["image_height"]) == (800, 600) else None
    if expected_shape is not None and output_shape != expected_shape:
        raise ValueError(f"VGGT4D preprocessing contract differs: expected {expected_shape}, got {output_shape}")
    scale = robust_metric_scale(measured_depths, predicted_depths, min_depth_m=min_depth_m, max_depth_m=max_depth_m)
    scaled_positions = positions * scale["scale"]
    rail = analyze_rail_keyframes(
        scaled_positions,
        rotations,
        min_translation_m=min_translation_m,
        min_rotation_deg=min_rotation_deg,
        max_gap=max_gap,
    )

    output_dir = pilot_root / "trajectory"
    output_dir.mkdir(parents=True, exist_ok=True)
    scaled_tum_path = output_dir / "scaled_pred_traj.txt"
    source_rows = np.atleast_2d(np.loadtxt(vggt_dir / "pred_traj.txt", dtype=np.float64))
    source_rows[:, 1:4] = scaled_positions
    np.savetxt(scaled_tum_path, source_rows, fmt="%.10g")

    output_frames = []
    for index, frame in enumerate(frames):
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = rotations[index]
        c2w[:3, 3] = scaled_positions[index]
        output_frames.append(
            {
                "frame_index": index,
                "image_name": frame["image_name"],
                "timestamp": float(timestamps[index]),
                "camera_to_world": c2w.tolist(),
                "rail_coordinate_m": rail["rail_coordinates_m"][index],
                "orthogonal_residual_m": rail["orthogonal_residuals_m"][index],
                "is_keyframe": index in rail["keyframe_indices"],
            }
        )
    result = {
        "schema_version": 1,
        "pose_convention": "camera_to_world_tum_xyz_qw_qx_qy_qz",
        "depth_preprocess": {
            "source_shape_hw": [manifest["image_height"], manifest["image_width"]],
            "model_shape_hw": list(output_shape or ()),
            "interpolation": "nearest",
            "crop": None,
        },
        "metric_scale": scale,
        "rail": rail,
        "keyframe_indices": rail["keyframe_indices"],
        "keyframe_image_names": [frames[index]["image_name"] for index in rail["keyframe_indices"]],
        "scaled_tum_path": str(scaled_tum_path),
        "frames": output_frames,
    }
    _write_json(output_dir / "rail_keyframes.json", result)
    (output_dir / "keyframes.txt").write_text("\n".join(result["keyframe_image_names"]) + "\n")
    return result


def image_ids_to_pair_id(image_id1: int, image_id2: int) -> int:
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return MAX_IMAGE_ID * image_id1 + image_id2


def pair_id_to_image_ids(pair_id: int) -> tuple[int, int]:
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) // MAX_IMAGE_ID
    return image_id1, image_id2


def _read_keypoints(connection: sqlite3.Connection) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    for image_id, rows, cols, blob in connection.execute("SELECT image_id, rows, cols, data FROM keypoints"):
        data = (
            np.empty((0, cols), dtype=np.float32) if not rows else np.frombuffer(blob, np.float32).reshape(rows, cols)
        )
        result[image_id] = data[:, :2].copy()
    return result


def _sample_nearest(array: np.ndarray, x: float, y: float, source_width: int, source_height: int) -> Any:
    target_x = min(array.shape[1] - 1, max(0, int(math.floor(x * array.shape[1] / source_width))))
    target_y = min(array.shape[0] - 1, max(0, int(math.floor(y * array.shape[0] / source_height))))
    return array[target_y, target_x]


def _backproject_world(
    x: float,
    y: float,
    depth_m: float,
    intrinsic: tuple[float, float, float, float],
    camera_to_world: np.ndarray,
) -> np.ndarray:
    fx, fy, cx, cy = intrinsic
    camera_point = np.array([(x - cx) * depth_m / fx, (y - cy) * depth_m / fy, depth_m])
    return camera_to_world[:3, :3] @ camera_point + camera_to_world[:3, 3]


def filter_colmap_matches(
    raw_db: Path,
    filtered_db: Path,
    manifest_path: Path,
    vggt_dir: Path,
    trajectory_path: Path,
    fx: float = DEFAULT_INTRINSICS[0],
    fy: float = DEFAULT_INTRINSICS[1],
    cx: float = DEFAULT_INTRINSICS[2],
    cy: float = DEFAULT_INTRINSICS[3],
    min_depth_m: float = 0.1,
    max_depth_m: float = 1.3,
    absolute_tolerance_m: float = 0.05,
    relative_tolerance: float = 0.05,
    missing_depth_policy: str = "keep",
) -> dict[str, Any]:
    if missing_depth_policy not in {"keep", "reject"}:
        raise ValueError("missing_depth_policy must be 'keep' or 'reject'")
    if filtered_db.exists():
        raise FileExistsError(f"filtered database already exists and will not be replaced: {filtered_db}")
    if not raw_db.is_file():
        raise FileNotFoundError(f"raw database does not exist: {raw_db}")
    manifest = json.loads(manifest_path.read_text())
    trajectory = json.loads(trajectory_path.read_text())
    width = int(manifest["image_width"])
    height = int(manifest["image_height"])
    frames_by_name = {frame["image_name"]: frame for frame in manifest["frames"]}
    trajectory_frames = {
        int(frame.get("frame_index", frame.get("global_index"))): frame for frame in trajectory["frames"]
    }
    poses_by_index = {
        index: np.asarray(frame["camera_to_world"], dtype=np.float64) for index, frame in trajectory_frames.items()
    }
    if set(poses_by_index) != {int(frame["frame_index"]) for frame in manifest["frames"]}:
        raise ValueError("trajectory frame indices do not match manifest")

    raw_hash_before = sha256_file(raw_db)
    filtered_db.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw_db, filtered_db)
    totals = {
        "before": 0,
        "kept": 0,
        "kept_depth_consistent": 0,
        "kept_depth_unavailable": 0,
        "invalid_coordinate": 0,
        "dynamic": 0,
        "invalid_depth": 0,
        "inconsistent_3d": 0,
    }
    pair_summaries: list[dict[str, Any]] = []
    connection = sqlite3.connect(filtered_db)
    try:
        images = {int(image_id): name for image_id, name in connection.execute("SELECT image_id, name FROM images")}
        keypoints = _read_keypoints(connection)
        cached: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for image_id, name in images.items():
            if name not in frames_by_name:
                raise ValueError(f"COLMAP image is not present in pilot manifest: {name}")
            frame = frames_by_name[name]
            index = int(frame["frame_index"])
            depth_path = Path(
                frame.get(
                    "depth_source",
                    manifest_path.parent / "mapped_depth" / frame["depth_name"],
                )
            )
            depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            mask_path = Path(
                trajectory_frames[index].get("dynamic_mask_path", vggt_dir / f"dynamic_mask_{index:04d}.png")
            )
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if depth_mm is None or depth_mm.dtype != np.uint16 or depth_mm.shape != (height, width):
                raise ValueError(f"invalid mapped depth for frame {index}: {frame['depth_name']}")
            if mask is None:
                raise FileNotFoundError(f"dynamic mask is missing for frame {index}")
            cached[image_id] = (depth_mm.astype(np.float32) / 1000.0, mask, poses_by_index[index])

        rows = list(connection.execute("SELECT pair_id, rows, cols, data FROM matches ORDER BY pair_id"))
        for pair_id, row_count, cols, blob in rows:
            if cols != 2:
                raise ValueError(f"match pair {pair_id} has {cols} columns instead of 2")
            image_id1, image_id2 = pair_id_to_image_ids(int(pair_id))
            if image_id1 not in images or image_id2 not in images:
                raise ValueError(f"match pair references unknown image ids: {(image_id1, image_id2)}")
            matches = (
                np.empty((0, 2), dtype=np.uint32)
                if not row_count
                else np.frombuffer(blob, np.uint32).reshape(row_count, 2).copy()
            )
            pair_counts = {key: 0 for key in totals if key != "before"}
            pair_counts["before"] = int(len(matches))
            kept: list[np.ndarray] = []
            depth1, mask1, pose1 = cached[image_id1]
            depth2, mask2, pose2 = cached[image_id2]
            points1 = keypoints[image_id1]
            points2 = keypoints[image_id2]
            for match in matches:
                index1, index2 = (int(match[0]), int(match[1]))
                if index1 >= len(points1) or index2 >= len(points2):
                    pair_counts["invalid_coordinate"] += 1
                    continue
                x1, y1 = points1[index1]
                x2, y2 = points2[index2]
                if not (0 <= x1 < width and 0 <= y1 < height and 0 <= x2 < width and 0 <= y2 < height):
                    pair_counts["invalid_coordinate"] += 1
                    continue
                if (
                    _sample_nearest(mask1, x1, y1, width, height) > 0
                    or _sample_nearest(mask2, x2, y2, width, height) > 0
                ):
                    pair_counts["dynamic"] += 1
                    continue
                z1 = float(depth1[int(math.floor(y1)), int(math.floor(x1))])
                z2 = float(depth2[int(math.floor(y2)), int(math.floor(x2))])
                if not (
                    math.isfinite(z1)
                    and math.isfinite(z2)
                    and min_depth_m <= z1 <= max_depth_m
                    and min_depth_m <= z2 <= max_depth_m
                ):
                    if missing_depth_policy == "reject":
                        pair_counts["invalid_depth"] += 1
                        continue
                    pair_counts["kept"] += 1
                    pair_counts["kept_depth_unavailable"] += 1
                    kept.append(match)
                    continue
                world1 = _backproject_world(x1, y1, z1, (fx, fy, cx, cy), pose1)
                world2 = _backproject_world(x2, y2, z2, (fx, fy, cx, cy), pose2)
                tolerance = max(absolute_tolerance_m, relative_tolerance * (z1 + z2) / 2.0)
                if float(np.linalg.norm(world1 - world2)) > tolerance:
                    pair_counts["inconsistent_3d"] += 1
                    continue
                pair_counts["kept"] += 1
                pair_counts["kept_depth_consistent"] += 1
                kept.append(match)
            kept_array = np.asarray(kept, dtype=np.uint32).reshape(-1, 2)
            connection.execute(
                "UPDATE matches SET rows=?, cols=2, data=? WHERE pair_id=?",
                (len(kept_array), kept_array.tobytes(), pair_id),
            )
            for key in totals:
                totals[key] += pair_counts[key]
            pair_summaries.append(
                {
                    "pair_id": int(pair_id),
                    "image_id1": image_id1,
                    "image_id2": image_id2,
                    "image_name1": images[image_id1],
                    "image_name2": images[image_id2],
                    **pair_counts,
                }
            )
        connection.execute("DELETE FROM two_view_geometries")
        connection.commit()
    except Exception:
        connection.close()
        filtered_db.unlink(missing_ok=True)
        raise
    finally:
        if connection:
            connection.close()

    raw_hash_after = sha256_file(raw_db)
    if raw_hash_after != raw_hash_before:
        raise RuntimeError("raw COLMAP database changed during filtering")
    terminal_total = sum(
        totals[key] for key in ("kept", "invalid_coordinate", "dynamic", "invalid_depth", "inconsistent_3d")
    )
    if totals["before"] != terminal_total:
        raise RuntimeError("match filtering counts do not add up")
    summary = {
        "schema_version": 1,
        "raw_database": str(raw_db.resolve()),
        "raw_database_sha256_before": raw_hash_before,
        "raw_database_sha256_after": raw_hash_after,
        "filtered_database": str(filtered_db.resolve()),
        "thresholds": {
            "min_depth_m": min_depth_m,
            "max_depth_m": max_depth_m,
            "absolute_tolerance_m": absolute_tolerance_m,
            "relative_tolerance": relative_tolerance,
            "missing_depth_policy": missing_depth_policy,
        },
        "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        "totals": totals,
        "pairs": pair_summaries,
    }
    _write_json(filtered_db.parent / "match_filter_summary.json", summary)
    return summary


def _write_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    if points.shape != colors.shape or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points and colors must both have shape (N, 3)")
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode()
    vertex = struct.Struct("<fffBBB")
    with path.open("wb") as stream:
        stream.write(header)
        for point, color in zip(points, colors, strict=True):
            stream.write(vertex.pack(float(point[0]), float(point[1]), float(point[2]), *map(int, color)))


def export_dynamic_rgbd_map(
    pilot_root: Path,
    vggt_dir: Path,
    trajectory_path: Path,
    output_path: Path,
    fx: float = DEFAULT_INTRINSICS[0],
    fy: float = DEFAULT_INTRINSICS[1],
    cx: float = DEFAULT_INTRINSICS[2],
    cy: float = DEFAULT_INTRINSICS[3],
    min_depth_m: float = 0.1,
    max_depth_m: float = 1.3,
    pixel_stride: int = 2,
    voxel_size_m: float = 0.005,
) -> dict[str, Any]:
    """Fuse measured RGB-D pixels selected by VGGT4D dynamic masks into a PLY."""
    if pixel_stride < 1:
        raise ValueError("pixel_stride must be at least 1")
    if voxel_size_m <= 0.0:
        raise ValueError("voxel_size_m must be positive")
    manifest = json.loads((pilot_root / "manifest.json").read_text())
    trajectory = json.loads(trajectory_path.read_text())
    trajectory_frames = {
        int(frame.get("frame_index", frame.get("global_index"))): frame for frame in trajectory["frames"]
    }
    poses = {
        index: np.asarray(frame["camera_to_world"], dtype=np.float64) for index, frame in trajectory_frames.items()
    }
    all_points: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    per_frame_counts: list[int] = []
    for frame in manifest["frames"]:
        index = int(frame["frame_index"])
        image = cv2.imread(str(pilot_root / "images" / frame["image_name"]), cv2.IMREAD_COLOR)
        depth_mm = cv2.imread(str(pilot_root / "mapped_depth" / frame["depth_name"]), cv2.IMREAD_UNCHANGED)
        dynamic_path = Path(
            trajectory_frames[index].get("dynamic_mask_path", vggt_dir / f"dynamic_mask_{index:04d}.png")
        )
        dynamic = cv2.imread(str(dynamic_path), cv2.IMREAD_GRAYSCALE)
        if image is None or depth_mm is None or dynamic is None:
            raise FileNotFoundError(f"RGB, depth, or dynamic mask is missing for frame {index}")
        if depth_mm.dtype != np.uint16 or image.shape[:2] != depth_mm.shape:
            raise ValueError(f"invalid RGB-D shapes for frame {index}")
        height, width = depth_mm.shape
        dynamic_full = cv2.resize(dynamic, (width, height), interpolation=cv2.INTER_NEAREST) > 0
        depth_m = depth_mm.astype(np.float32) / 1000.0
        rows, cols = np.indices((height, width))
        selected = (
            dynamic_full
            & np.isfinite(depth_m)
            & (depth_m >= min_depth_m)
            & (depth_m <= max_depth_m)
            & (rows % pixel_stride == 0)
            & (cols % pixel_stride == 0)
        )
        y = rows[selected].astype(np.float64)
        x = cols[selected].astype(np.float64)
        z = depth_m[selected].astype(np.float64)
        camera_points = np.stack(((x - cx) * z / fx, (y - cy) * z / fy, z), axis=1)
        pose = poses[index]
        world_points = camera_points @ pose[:3, :3].T + pose[:3, 3]
        colors = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)[selected]
        all_points.append(world_points)
        all_colors.append(colors)
        per_frame_counts.append(int(len(world_points)))
    if not any(per_frame_counts):
        raise ValueError("no valid dynamic RGB-D points were selected")
    points = np.concatenate(all_points).astype(np.float32)
    colors = np.concatenate(all_colors).astype(np.uint8)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    colors = colors[finite]
    voxel_keys = np.floor(points / voxel_size_m).astype(np.int64)
    _, first_indices = np.unique(voxel_keys, axis=0, return_index=True)
    first_indices.sort()
    points = points[first_indices]
    colors = colors[first_indices]
    _write_binary_ply(output_path, points, colors)
    summary = {
        "schema_version": 1,
        "description": "initial metric dynamic evidence map; moving objects are not motion-compensated",
        "output_path": str(output_path.resolve()),
        "frame_count": len(manifest["frames"]),
        "per_frame_points_before_voxel": per_frame_counts,
        "point_count_before_voxel": int(sum(per_frame_counts)),
        "point_count_after_voxel": int(len(points)),
        "pixel_stride": pixel_stride,
        "voxel_size_m": voxel_size_m,
        "depth_range_m": [min_depth_m, max_depth_m],
        "pose_source": str(trajectory_path.resolve()),
    }
    _write_json(output_path.with_suffix(".json"), summary)
    return summary


def _database_metrics(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    connection = sqlite3.connect(path)
    try:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        result: dict[str, Any] = {}
        for table in ("cameras", "images", "keypoints", "descriptors", "matches", "two_view_geometries"):
            if table not in tables:
                result[table] = None
                continue
            result[table] = int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            if table in {"keypoints", "descriptors", "matches", "two_view_geometries"}:
                result[f"{table}_rows"] = int(
                    connection.execute(f"SELECT COALESCE(SUM(rows), 0) FROM {table}").fetchone()[0]
                )
        return result
    finally:
        connection.close()


def _parse_model_analyzer(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    fields: dict[str, Any] = {}
    integer_labels = {
        "Registered frames": "registered_frames",
        "Registered images": "registered_images",
        "Points": "points3d",
        "Observations": "observations",
    }
    float_labels = {"Mean reprojection error": "mean_reprojection_error_px"}
    for line in path.read_text().splitlines():
        message = line.split("] ", 1)[-1]
        for label, key in integer_labels.items():
            if message.startswith(f"{label}:"):
                fields[key] = int(message.split(":", 1)[1].strip())
        for label, key in float_labels.items():
            if message.startswith(f"{label}:"):
                fields[key] = float(message.split(":", 1)[1].strip().removesuffix("px"))
    return fields


def _component_status(path: Path, missing_reason: str) -> dict[str, Any]:
    if path.is_file():
        return {"status": "ready", "path": str(path.resolve()), "size_bytes": path.stat().st_size}
    return {"status": "blocked", "expected_path": str(path), "reason": missing_reason}


def _opend4rt_metrics(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with np.load(path) as data:
        xyz = np.asarray(data["point_xyz_ref0"])
        uv = np.asarray(data["point_uv_px"])
        visibility = np.asarray(data["point_visibility"], dtype=bool)
        dynamic = np.asarray(data["point_is_dynamic"], dtype=bool)
        return {
            "path": str(path.resolve()),
            "file_size_bytes": path.stat().st_size,
            "frame_count": int(xyz.shape[0]),
            "point_count": int(xyz.shape[1]),
            "point_xyz_shape": list(xyz.shape),
            "point_uv_shape": list(uv.shape),
            "finite_xyz": bool(np.isfinite(xyz).all()),
            "finite_uv": bool(np.isfinite(uv).all()),
            "dynamic_point_count": int(dynamic.sum()),
            "visible_observation_count": int(visibility.sum()),
        }


def summarize_workflow(
    pilot_root: Path,
    vggt4d_checkpoint: Path = DEFAULT_VGGT4D_CHECKPOINT,
    vggt_omega_checkpoint: Path = DEFAULT_VGGT_OMEGA_CHECKPOINT,
    opend4rt_checkpoint: Path = DEFAULT_OPEND4RT_CHECKPOINT,
) -> dict[str, Any]:
    manifest_path = pilot_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {"frames": []}
    frame_count = len(manifest["frames"])
    vggt_dir = pilot_root / "vggt4d"
    trajectory_path = pilot_root / "trajectory" / "rail_keyframes.json"
    trajectory = json.loads(trajectory_path.read_text()) if trajectory_path.is_file() else None
    filter_path = pilot_root / "colmap" / "match_filter_summary.json"
    filter_summary = json.loads(filter_path.read_text()) if filter_path.is_file() else None
    raw_metrics = _database_metrics(pilot_root / "colmap" / "raw.db")
    filtered_metrics = _database_metrics(pilot_root / "colmap" / "filtered.db")
    before_ba = _parse_model_analyzer(pilot_root / "logs" / "model_analyzer_before_ba.log")
    after_ba = _parse_model_analyzer(pilot_root / "logs" / "model_analyzer_after_ba.log")
    dynamic_summary_path = pilot_root / "dynamic_map" / "dynamic_rgbd_map.json"
    dynamic_summary = json.loads(dynamic_summary_path.read_text()) if dynamic_summary_path.is_file() else None
    opend4rt_path = pilot_root / "opend4rt" / "dense_scene_8f_32x24.npz"
    opend4rt_metrics = _opend4rt_metrics(opend4rt_path)

    manifest_ok = frame_count == 8 and all(
        (pilot_root / "images" / frame["image_name"]).is_file()
        and (pilot_root / "mapped_depth" / frame["depth_name"]).is_file()
        for frame in manifest["frames"]
    )
    vggt_ok = (
        frame_count == 8
        and all(
            len(list(vggt_dir.glob(pattern))) == 8 for pattern in ("frame_*.npy", "conf_*.npy", "dynamic_mask_*.png")
        )
        and (vggt_dir / "pred_traj.txt").is_file()
    )
    trajectory_ok = bool(
        trajectory
        and math.isfinite(trajectory["metric_scale"]["scale"])
        and math.isfinite(trajectory["rail"]["orthogonal_rms_m"])
        and len(trajectory["keyframe_indices"]) >= 2
    )
    raw_ok = bool(
        raw_metrics
        and raw_metrics.get("images") == 8
        and raw_metrics.get("keypoints") == 8
        and raw_metrics.get("descriptors") == 8
        and raw_metrics.get("matches_rows", 0) > 0
    )
    filter_ok = bool(
        filter_summary
        and filter_summary["raw_database_sha256_before"] == filter_summary["raw_database_sha256_after"]
        and filter_summary["totals"]["kept"] > 0
        and filtered_metrics
        and filtered_metrics.get("matches_rows") == filter_summary["totals"]["kept"]
    )
    reconstruction_ok = bool(
        after_ba
        and after_ba.get("registered_images", 0) >= 4
        and after_ba.get("points3d", 0) > 0
        and math.isfinite(after_ba.get("mean_reprojection_error_px", math.nan))
    )
    quality_logs = [pilot_root / "logs" / name for name in ("pytest.log", "quality.log", "regression.log")]
    quality_ok = all(path.is_file() for path in quality_logs)
    components = {
        "vggt4d": _component_status(vggt4d_checkpoint, "official VGGT4D checkpoint is missing"),
        "vggt_omega": _component_status(
            vggt_omega_checkpoint,
            "facebook/VGGT-Omega is gated; obtain access and place the official checkpoint here",
        ),
        "opend4rt": _component_status(
            opend4rt_checkpoint,
            "OpenD4RT model.yaml exists but the official opend4rt.ckpt has not been downloaded",
        ),
        "mambaglue": {
            **_component_status(
                DEFAULT_MAMBAGLUE_CHECKPOINT,
                "MambaGlue SuperPoint checkpoint is missing",
            ),
            "selected_for_pilot": False,
            "selection_reason": "the accepted ALIKED+LightGlue alternative writes COLMAP-native descriptors and matches",
        },
        "colmap_aliked": _component_status(
            COLMAP_CACHE_DIR
            / "39c423d0a6f03d39ec89d3d1d61853765c2fb6a8b8381376c703e5758778a547-aliked-n16rot.onnx",
            "COLMAP ALIKED ONNX checkpoint has not been downloaded",
        ),
        "colmap_lightglue": _component_status(
            COLMAP_CACHE_DIR
            / "b9a5de7204648b18a8cf5dcac819f9d30de1a5961ef03756803c8b86c2dceb8d-aliked-lightglue.onnx",
            "COLMAP ALIKED-LightGlue ONNX checkpoint has not been downloaded",
        ),
    }
    components["opend4rt"]["inference_status"] = "complete" if opend4rt_metrics else "not_run"
    if opend4rt_metrics:
        components["opend4rt"]["inference_artifact"] = str(opend4rt_path.resolve())
    inventory_ok = all(
        item["status"] in {"ready", "blocked"} and (item.get("path") or item.get("reason"))
        for item in components.values()
    )
    dod = {
        "DoD-1_manifest": manifest_ok,
        "DoD-2_vggt4d": vggt_ok and components["vggt4d"]["status"] == "ready",
        "DoD-3_metric_rail_keyframes": trajectory_ok,
        "DoD-4_raw_aliked_lightglue": raw_ok,
        "DoD-5_rgbd_filtered_db": filter_ok,
        "DoD-6_glomap_bundle_adjustment": reconstruction_ok,
        "DoD-7_quality_logs": quality_ok,
        "DoD-8_component_inventory": bool(inventory_ok),
    }
    result = {
        "schema_version": 1,
        "pilot_root": str(pilot_root.resolve()),
        "core_workflow_success": all(dod[key] for key in list(dod)[:6]),
        "overall_documented_success": all(dod.values()),
        "optional_model_components_complete": all(
            components[name]["status"] == "ready" for name in ("vggt_omega", "opend4rt")
        ),
        "definition_of_done": dod,
        "components": components,
        "selected_pipeline": {
            "pose_and_dynamic_backend": "VGGT4D official model_tracker_fixed_e20.pt",
            "feature_backend": "COLMAP ALIKED_N16ROT ONNX",
            "matching_backend": "COLMAP ALIKED_LIGHTGLUE ONNX sequential overlap=3",
            "global_sfm_backend": "COLMAP integrated GLOMAP global_mapper",
            "global_positioning_and_ba_device": "CPU explicitly selected because this Ceres build lacks cuDSS",
        },
        "metrics": {
            "frame_count": frame_count,
            "metric_scale": trajectory["metric_scale"] if trajectory else None,
            "rail": trajectory["rail"] if trajectory else None,
            "raw_database": raw_metrics,
            "filtered_database": filtered_metrics,
            "match_filter": filter_summary["totals"] if filter_summary else None,
            "glomap_before_explicit_ba": before_ba,
            "after_explicit_ba": after_ba,
            "dynamic_map": dynamic_summary,
            "opend4rt_dense_scene": opend4rt_metrics,
        },
        "artifacts": {
            "manifest": str(manifest_path),
            "trajectory": str(trajectory_path),
            "match_filter": str(filter_path),
            "sparse_model": str(pilot_root / "colmap" / "sparse" / "0"),
            "bundle_adjusted_model": str(pilot_root / "colmap" / "ba"),
            "dynamic_map": str(pilot_root / "dynamic_map" / "dynamic_rgbd_map.ply"),
            "opend4rt_dense_scene": str(opend4rt_path),
            "opend4rt_rerun": str(pilot_root / "opend4rt" / "dense_scene_8f_32x24.rrd"),
        },
    }
    _write_json(pilot_root / "workflow_summary.json", result)
    return result


def _add_depth_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--min-depth-m", type=float, default=0.1)
    parser.add_argument("--max-depth-m", type=float, default=1.3)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Create a manifest-backed pilot view")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--start", type=int, default=0)
    prepare.add_argument("--stride", type=int, default=3)
    prepare.add_argument("--count", type=int, default=8)

    analyze = subparsers.add_parser("analyze-trajectory", help="Metric-scale poses and select rail keyframes")
    analyze.add_argument("--pilot-root", type=Path, required=True)
    analyze.add_argument("--vggt-dir", type=Path, required=True)
    _add_depth_options(analyze)
    analyze.add_argument("--min-translation-m", type=float, default=0.08)
    analyze.add_argument("--min-rotation-deg", type=float, default=3.0)
    analyze.add_argument("--max-gap", type=int, default=3)

    filter_parser = subparsers.add_parser("filter-matches", help="Filter a copied COLMAP match database")
    filter_parser.add_argument("--raw-db", type=Path, required=True)
    filter_parser.add_argument("--filtered-db", type=Path, required=True)
    filter_parser.add_argument("--manifest", type=Path, required=True)
    filter_parser.add_argument("--vggt-dir", type=Path, required=True)
    filter_parser.add_argument("--trajectory", type=Path, required=True)
    filter_parser.add_argument("--fx", type=float, default=DEFAULT_INTRINSICS[0])
    filter_parser.add_argument("--fy", type=float, default=DEFAULT_INTRINSICS[1])
    filter_parser.add_argument("--cx", type=float, default=DEFAULT_INTRINSICS[2])
    filter_parser.add_argument("--cy", type=float, default=DEFAULT_INTRINSICS[3])
    _add_depth_options(filter_parser)
    filter_parser.add_argument("--absolute-tolerance-m", type=float, default=0.05)
    filter_parser.add_argument("--relative-tolerance", type=float, default=0.05)
    filter_parser.add_argument("--missing-depth-policy", choices=("keep", "reject"), default="keep")

    dynamic = subparsers.add_parser("export-dynamic-map", help="Fuse masked measured RGB-D into a PLY")
    dynamic.add_argument("--pilot-root", type=Path, required=True)
    dynamic.add_argument("--vggt-dir", type=Path, required=True)
    dynamic.add_argument("--trajectory", type=Path, required=True)
    dynamic.add_argument("--output", type=Path, required=True)
    dynamic.add_argument("--fx", type=float, default=DEFAULT_INTRINSICS[0])
    dynamic.add_argument("--fy", type=float, default=DEFAULT_INTRINSICS[1])
    dynamic.add_argument("--cx", type=float, default=DEFAULT_INTRINSICS[2])
    dynamic.add_argument("--cy", type=float, default=DEFAULT_INTRINSICS[3])
    _add_depth_options(dynamic)
    dynamic.add_argument("--pixel-stride", type=int, default=2)
    dynamic.add_argument("--voxel-size-m", type=float, default=0.005)

    summarize = subparsers.add_parser("summarize", help="Write the auditable workflow summary")
    summarize.add_argument("--pilot-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare_pilot(args.source, args.output, args.start, args.stride, args.count)
    elif args.command == "analyze-trajectory":
        result = analyze_trajectory(
            args.pilot_root,
            args.vggt_dir,
            min_depth_m=args.min_depth_m,
            max_depth_m=args.max_depth_m,
            min_translation_m=args.min_translation_m,
            min_rotation_deg=args.min_rotation_deg,
            max_gap=args.max_gap,
        )
    elif args.command == "filter-matches":
        result = filter_colmap_matches(
            raw_db=args.raw_db,
            filtered_db=args.filtered_db,
            manifest_path=args.manifest,
            vggt_dir=args.vggt_dir,
            trajectory_path=args.trajectory,
            fx=args.fx,
            fy=args.fy,
            cx=args.cx,
            cy=args.cy,
            min_depth_m=args.min_depth_m,
            max_depth_m=args.max_depth_m,
            absolute_tolerance_m=args.absolute_tolerance_m,
            relative_tolerance=args.relative_tolerance,
            missing_depth_policy=args.missing_depth_policy,
        )
    elif args.command == "export-dynamic-map":
        result = export_dynamic_rgbd_map(
            pilot_root=args.pilot_root,
            vggt_dir=args.vggt_dir,
            trajectory_path=args.trajectory,
            output_path=args.output,
            fx=args.fx,
            fy=args.fy,
            cx=args.cx,
            cy=args.cy,
            min_depth_m=args.min_depth_m,
            max_depth_m=args.max_depth_m,
            pixel_stride=args.pixel_stride,
            voxel_size_m=args.voxel_size_m,
        )
    elif args.command == "summarize":
        result = summarize_workflow(args.pilot_root)
    else:  # pragma: no cover - argparse enforces the choices
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
