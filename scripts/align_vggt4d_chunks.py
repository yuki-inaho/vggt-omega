"""Metric-scale and align resumable VGGT4D chunks through shared frames."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rgbd_sfm_pilot import (  # noqa: E402
    analyze_rail_keyframes,
    quaternion_wxyz_to_matrix,
    robust_metric_scale,
)


def matrix_to_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Convert a proper 3x3 rotation matrix to a canonical wxyz quaternion."""
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("rotation must be a finite 3x3 matrix")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * math.sqrt(trace + 1.0)
        qw = 0.25 * scale
        qx = (matrix[2, 1] - matrix[1, 2]) / scale
        qy = (matrix[0, 2] - matrix[2, 0]) / scale
        qz = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = 2.0 * math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
        qw = (matrix[2, 1] - matrix[1, 2]) / scale
        qx = 0.25 * scale
        qy = (matrix[0, 1] + matrix[1, 0]) / scale
        qz = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = 2.0 * math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
        qw = (matrix[0, 2] - matrix[2, 0]) / scale
        qx = (matrix[0, 1] + matrix[1, 0]) / scale
        qy = 0.25 * scale
        qz = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = 2.0 * math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
        qw = (matrix[1, 0] - matrix[0, 1]) / scale
        qx = (matrix[0, 2] + matrix[2, 0]) / scale
        qy = (matrix[1, 2] + matrix[2, 1]) / scale
        qz = 0.25 * scale
    quaternion = np.array([qw, qx, qy, qz], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0.0:
        quaternion = -quaternion
    return quaternion


def average_rigid_transforms(transforms: Sequence[np.ndarray]) -> np.ndarray:
    if not transforms:
        raise ValueError("at least one rigid transform is required")
    matrices = np.asarray(transforms, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1:] != (4, 4):
        raise ValueError("transforms must have shape (N, 4, 4)")
    u, _, vh = np.linalg.svd(matrices[:, :3, :3].sum(axis=0))
    rotation = u @ vh
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vh
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = np.median(matrices[:, :3, 3], axis=0)
    return result


def _rotation_difference_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first[:3, :3].T @ second[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def align_local_chunk(
    local_poses: Mapping[int, np.ndarray], global_poses: Mapping[int, np.ndarray]
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    shared = sorted(set(local_poses) & set(global_poses))
    if not shared:
        raise ValueError("chunk has no shared frame with the aligned trajectory")
    candidates = [global_poses[index] @ np.linalg.inv(local_poses[index]) for index in shared]
    global_from_local = average_rigid_transforms(candidates)
    aligned = {index: global_from_local @ pose for index, pose in local_poses.items()}
    translation_residuals = [
        float(np.linalg.norm(aligned[index][:3, 3] - global_poses[index][:3, 3])) for index in shared
    ]
    rotation_residuals = [_rotation_difference_deg(aligned[index], global_poses[index]) for index in shared]
    edge = {
        "shared_global_indices": shared,
        "global_from_local": global_from_local.tolist(),
        "translation_residuals_m": translation_residuals,
        "translation_rms_m": float(np.sqrt(np.mean(np.square(translation_residuals)))),
        "translation_max_m": float(np.max(translation_residuals)),
        "rotation_residuals_deg": rotation_residuals,
        "rotation_rms_deg": float(np.sqrt(np.mean(np.square(rotation_residuals)))),
        "rotation_max_deg": float(np.max(rotation_residuals)),
    }
    return aligned, edge


def _load_tum(path: Path) -> list[np.ndarray]:
    rows = np.atleast_2d(np.loadtxt(path, dtype=np.float64))
    if rows.shape[1] != 8:
        raise ValueError(f"expected 8 TUM columns in {path}, got {rows.shape}")
    poses: list[np.ndarray] = []
    for row in rows:
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = quaternion_wxyz_to_matrix(row[4:8])
        pose[:3, 3] = row[1:4]
        poses.append(pose)
    return poses


def _depth_name(image_name: str) -> str:
    if not image_name.endswith("_rgb.png"):
        raise ValueError(f"cannot derive mapped depth name from {image_name}")
    return image_name.removesuffix("_rgb.png") + "_depth.png"


def _metric_chunk(
    chunk_dir: Path,
    global_indices: Sequence[int],
    frame_names: Sequence[str],
    mapped_depth_dir: Path,
    min_depth_m: float,
    max_depth_m: float,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    local_poses = _load_tum(chunk_dir / "pred_traj.txt")
    if len(local_poses) != len(global_indices) or len(frame_names) != len(global_indices):
        raise ValueError(f"chunk frame/pose count differs in {chunk_dir}")
    measured_depths: list[np.ndarray] = []
    predicted_depths: list[np.ndarray] = []
    for local_index, name in enumerate(frame_names):
        predicted = np.load(chunk_dir / f"frame_{local_index:04d}.npy").astype(np.float32)
        measured_mm = cv2.imread(str(mapped_depth_dir / _depth_name(name)), cv2.IMREAD_UNCHANGED)
        if measured_mm is None or measured_mm.dtype != np.uint16:
            raise ValueError(f"missing uint16 mapped depth for {name}")
        measured = cv2.resize(
            measured_mm.astype(np.float32) / 1000.0,
            (predicted.shape[1], predicted.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        measured_depths.append(measured)
        predicted_depths.append(predicted)
    scale = robust_metric_scale(
        measured_depths,
        predicted_depths,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
    )
    metric_poses: dict[int, np.ndarray] = {}
    for global_index, pose in zip(global_indices, local_poses, strict=True):
        metric_pose = pose.copy()
        metric_pose[:3, 3] *= scale["scale"]
        metric_poses[int(global_index)] = metric_pose
    return metric_poses, scale


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def project_positions_to_rail(positions: np.ndarray, rail: Mapping[str, Any]) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float64)
    axis = np.asarray(rail["rail_axis"], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    centroid = np.asarray(rail["rail_centroid_m"], dtype=np.float64)
    coordinates = (positions - centroid) @ axis
    return centroid + coordinates[:, None] * axis


def constrain_positions_near_rail(
    positions: np.ndarray,
    rail: Mapping[str, Any],
    target_rms_m: float = 0.005,
    max_retained_fraction: float = 0.10,
) -> tuple[np.ndarray, float]:
    if target_rms_m <= 0 or not 0 < max_retained_fraction < 1:
        raise ValueError("rail constraint parameters must be positive and retain fraction in (0, 1)")
    positions = np.asarray(positions, dtype=np.float64)
    projected = project_positions_to_rail(positions, rail)
    rms = float(rail["orthogonal_rms_m"])
    retained_fraction = min(max_retained_fraction, target_rms_m / rms) if rms > 0 else 0.0
    return projected + retained_fraction * (positions - projected), retained_fraction


def align_chunk_run(
    chunk_run_dir: Path,
    mapped_depth_dir: Path,
    output_dir: Path,
    min_depth_m: float = 0.1,
    max_depth_m: float = 1.3,
    min_translation_m: float = 0.08,
    min_rotation_deg: float = 3.0,
    max_gap: int = 12,
) -> dict[str, Any]:
    manifest_path = chunk_run_dir / "chunks_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    frames_by_index = {int(frame["global_index"]): frame for frame in manifest["frames"]}
    global_poses: dict[int, np.ndarray] = {}
    dynamic_masks: dict[int, str] = {}
    chunk_scales: list[dict[str, Any]] = []
    alignment_edges: list[dict[str, Any]] = []
    for chunk in manifest["chunks"]:
        chunk_index = int(chunk["chunk_index"])
        chunk_dir = chunk_run_dir / f"chunk_{chunk_index:06d}"
        global_indices = [int(index) for index in chunk["global_indices"]]
        frame_names = chunk["frame_names"]
        metric_poses, scale = _metric_chunk(
            chunk_dir,
            global_indices,
            frame_names,
            mapped_depth_dir,
            min_depth_m,
            max_depth_m,
        )
        if not global_poses:
            aligned = metric_poses
            transform = np.eye(4, dtype=np.float64)
            edge = None
        else:
            aligned, edge = align_local_chunk(metric_poses, global_poses)
            edge = {"chunk_index": chunk_index, **edge}
            transform = np.asarray(edge["global_from_local"])
            alignment_edges.append(edge)
        chunk_scales.append(
            {
                "chunk_index": chunk_index,
                "global_indices": global_indices,
                "metric_scale": scale,
                "global_from_local": transform.tolist(),
            }
        )
        for local_index, global_index in enumerate(global_indices):
            if global_index not in global_poses:
                global_poses[global_index] = aligned[global_index]
            dynamic_masks.setdefault(
                global_index,
                str((chunk_dir / f"dynamic_mask_{local_index:04d}.png").resolve()),
            )
    expected_indices = set(frames_by_index)
    if set(global_poses) != expected_indices:
        missing = sorted(expected_indices - set(global_poses))
        raise ValueError(f"aligned trajectory does not cover all selected frames: {missing}")

    ordered_indices = sorted(global_poses)
    positions = np.stack([global_poses[index][:3, 3] for index in ordered_indices])
    rotations = np.stack([global_poses[index][:3, :3] for index in ordered_indices])
    rail = analyze_rail_keyframes(
        positions,
        rotations,
        min_translation_m=min_translation_m,
        min_rotation_deg=min_rotation_deg,
        max_gap=max_gap,
    )
    constrained_positions, retained_fraction = constrain_positions_near_rail(positions, rail)
    constrained_residuals = np.linalg.norm(
        np.cross(
            constrained_positions - np.asarray(rail["rail_centroid_m"]),
            np.asarray(rail["rail_axis"]),
        ),
        axis=1,
    )
    rail["downstream_pose_policy"] = (
        "camera centers constrained near PCA rail with adaptive non-collinear residual; rotations unchanged"
    )
    rail["downstream_retained_orthogonal_fraction"] = retained_fraction
    rail["constrained_orthogonal_rms_m"] = float(np.sqrt(np.mean(constrained_residuals**2)))
    output_frames: list[dict[str, Any]] = []
    tum_rows: list[list[float]] = []
    for order_index, global_index in enumerate(ordered_indices):
        unconstrained_pose = global_poses[global_index]
        pose = unconstrained_pose.copy()
        pose[:3, 3] = constrained_positions[order_index]
        quaternion = matrix_to_quaternion_wxyz(pose[:3, :3])
        frame = frames_by_index[global_index]
        output_frames.append(
            {
                **frame,
                "frame_index": global_index,
                "image_name": frame["name"],
                "camera_to_world": pose.tolist(),
                "camera_to_world_unconstrained": unconstrained_pose.tolist(),
                "dynamic_mask_path": dynamic_masks[global_index],
                "rail_coordinate_m": rail["rail_coordinates_m"][order_index],
                "orthogonal_residual_m": rail["orthogonal_residuals_m"][order_index],
                "is_keyframe": order_index in rail["keyframe_indices"],
            }
        )
        tum_rows.append([float(global_index), *pose[:3, 3].tolist(), *quaternion.tolist()])
    result = {
        "schema_version": 1,
        "source_chunk_manifest": str(manifest_path.resolve()),
        "mapped_depth_dir": str(mapped_depth_dir.resolve()),
        "pose_convention": "camera_to_world_tum_xyz_qw_qx_qy_qz",
        "frame_count": len(output_frames),
        "chunk_count": len(chunk_scales),
        "chunk_scales": chunk_scales,
        "alignment_edges": alignment_edges,
        "rail": rail,
        "keyframe_indices": rail["keyframe_indices"],
        "keyframe_image_names": [output_frames[index]["name"] for index in rail["keyframe_indices"]],
        "frames": output_frames,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "aligned_trajectory.json", result)
    np.savetxt(output_dir / "aligned_trajectory.tum", np.asarray(tum_rows), fmt="%.10g")
    (output_dir / "keyframes.txt").write_text("\n".join(result["keyframe_image_names"]) + "\n")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-run-dir", type=Path, required=True)
    parser.add_argument("--mapped-depth-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-depth-m", type=float, default=0.1)
    parser.add_argument("--max-depth-m", type=float, default=1.3)
    parser.add_argument("--min-translation-m", type=float, default=0.08)
    parser.add_argument("--min-rotation-deg", type=float, default=3.0)
    parser.add_argument("--max-gap", type=int, default=12)
    args = parser.parse_args(argv)
    result = align_chunk_run(
        chunk_run_dir=args.chunk_run_dir,
        mapped_depth_dir=args.mapped_depth_dir,
        output_dir=args.output_dir,
        min_depth_m=args.min_depth_m,
        max_depth_m=args.max_depth_m,
        min_translation_m=args.min_translation_m,
        min_rotation_deg=args.min_rotation_deg,
        max_gap=args.max_gap,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
