#!/usr/bin/env python3
"""Align masked RGB-D point clouds from overlapping VGGT pose chunks.

This is the first pose-graph stage: infer overlapping chunks, derive each
adjacent chunk transform from their shared-frame VGGT camera poses, and use the
resulting chained global poses to transform camera-frame masked PLY files.
Point-cloud registration and loop closure can refine these initial poses later.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import torch

from run_vggt_rgbd_pose_workflow import (
    DEPTH_SUFFIX,
    RGB_SUFFIX,
    estimate_scale,
    load_inputs,
)
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_checkpoint_state_dict
from vggt_omega.utils.pose_enc import encoding_to_camera

MASKED_CLOUD_DIR: Final = "point_clouds_left_third_or_stem_foreground_voxel_0025m"
MASKED_CLOUD_SUFFIX: Final = "_masked_voxel_0025m.ply"


@dataclass(frozen=True)
class AlignmentConfig:
    session_dir: Path
    checkpoint: Path
    output_dir: Path
    chunk_size: int
    stride: int
    max_chunks: int | None
    width: int
    height: int
    min_depth_m: float
    max_depth_m: float
    fusion_voxel_size_m: float


@dataclass(frozen=True)
class ChunkResult:
    start_index: int
    frame_stems: tuple[str, ...]
    scale: float
    scale_valid_pixels: int
    camera_to_local_world: np.ndarray


def parse_args() -> AlignmentConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=6)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--min-depth-m", type=float, default=0.10)
    parser.add_argument("--max-depth-m", type=float, default=5.00)
    parser.add_argument("--fusion-voxel-size-m", type=float, default=0.005)
    args = parser.parse_args()
    if args.chunk_size < 2:
        raise ValueError("--chunk-size must be >= 2")
    if not 0 < args.stride < args.chunk_size:
        raise ValueError("--stride must satisfy 0 < stride < chunk-size")
    if args.max_chunks is not None and args.max_chunks < 2:
        raise ValueError("--max-chunks must be >= 2 when specified")
    if args.width <= 0 or args.height <= 0 or args.width % 16 or args.height % 16:
        raise ValueError("--width and --height must be positive multiples of 16")
    if not 0 < args.min_depth_m < args.max_depth_m:
        raise ValueError("Depth limits must satisfy 0 < min < max")
    if args.fusion_voxel_size_m <= 0:
        raise ValueError("--fusion-voxel-size-m must be positive")
    return AlignmentConfig(**vars(args))


def collect_all_pairs(session_dir: Path) -> list[tuple[Path, Path]]:
    rgb_dir = session_dir / "rgb"
    depth_dir = session_dir / "mapped_depth_dense"
    if not rgb_dir.is_dir() or not depth_dir.is_dir():
        raise FileNotFoundError("Session must contain rgb/ and mapped_depth_dense/")
    pairs: list[tuple[Path, Path]] = []
    for rgb_path in sorted(rgb_dir.glob(f"*{RGB_SUFFIX}")):
        stem = rgb_path.name.removesuffix(RGB_SUFFIX)
        depth_path = depth_dir / f"{stem}{DEPTH_SUFFIX}"
        if depth_path.is_file():
            pairs.append((rgb_path, depth_path))
    return pairs


def chunk_start_indices(frame_count: int, chunk_size: int, stride: int) -> list[int]:
    if frame_count < chunk_size:
        return []
    starts = list(range(0, frame_count - chunk_size + 1, stride))
    final_start = frame_count - chunk_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def make_homogeneous(extrinsics: np.ndarray) -> np.ndarray:
    if extrinsics.ndim != 3 or extrinsics.shape[1:] != (3, 4):
        raise ValueError(f"Expected [N, 3, 4] extrinsics, got {extrinsics.shape}")
    result = np.repeat(np.eye(4, dtype=np.float64)[None], len(extrinsics), axis=0)
    result[:, :3, :] = extrinsics
    return result


def average_se3(transforms: list[np.ndarray]) -> np.ndarray:
    """Return a robust small-sample SE(3) average using median t and SVD R."""
    if not transforms:
        raise ValueError("At least one transform is required")
    stack = np.asarray(transforms, dtype=np.float64)
    if stack.shape[1:] != (4, 4):
        raise ValueError(f"Expected [N, 4, 4] transforms, got {stack.shape}")
    u, _, vt = np.linalg.svd(stack[:, :3, :3].sum(axis=0))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = np.median(stack[:, :3, 3], axis=0)
    return result


def rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def estimate_adjacent_chunk_transform(
    previous: ChunkResult, current: ChunkResult
) -> tuple[np.ndarray, dict[str, object]]:
    """Estimate T_current_local_from_previous_local from shared camera poses."""
    previous_indices = {stem: index for index, stem in enumerate(previous.frame_stems)}
    current_indices = {stem: index for index, stem in enumerate(current.frame_stems)}
    shared = sorted(previous_indices.keys() & current_indices.keys())
    if not shared:
        raise RuntimeError("Adjacent chunks have no shared frames")

    candidates: list[np.ndarray] = []
    for stem in shared:
        previous_pose = previous.camera_to_local_world[previous_indices[stem]]
        current_pose = current.camera_to_local_world[current_indices[stem]]
        candidates.append(current_pose @ np.linalg.inv(previous_pose))
    transform = average_se3(candidates)
    translation_errors = [float(np.linalg.norm(candidate[:3, 3] - transform[:3, 3])) for candidate in candidates]
    rotation_errors = [rotation_angle_deg(transform[:3, :3].T @ candidate[:3, :3]) for candidate in candidates]
    metrics: dict[str, object] = {
        "shared_frames": shared,
        "translation_residual_m_median": float(np.median(translation_errors)),
        "translation_residual_m_max": float(np.max(translation_errors)),
        "rotation_residual_deg_median": float(np.median(rotation_errors)),
        "rotation_residual_deg_max": float(np.max(rotation_errors)),
    }
    return transform, metrics


def infer_chunk(
    model: VGGTOmega,
    pairs: list[tuple[Path, Path]],
    start_index: int,
    config: AlignmentConfig,
) -> ChunkResult:
    images_cpu, metric_depth_m = load_inputs(pairs, config.width, config.height)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        predictions = model(images_cpu.to("cuda"))
        pose_encoding = predictions["pose_enc"].float().cpu()
        predicted_depth = predictions["depth"].float().cpu().numpy()[0, ..., 0]
        extrinsics, _ = encoding_to_camera(pose_encoding, (config.height, config.width))
    scale, valid_pixels = estimate_scale(
        predicted_depth,
        metric_depth_m,
        config.min_depth_m,
        config.max_depth_m,
    )
    scaled_extrinsics = extrinsics.numpy()[0]
    scaled_extrinsics[:, :3, 3] *= scale
    camera_to_local = np.linalg.inv(make_homogeneous(scaled_extrinsics))
    frame_stems = tuple(rgb.name.removesuffix(RGB_SUFFIX) for rgb, _ in pairs)
    return ChunkResult(start_index, frame_stems, scale, valid_pixels, camera_to_local)


def read_open3d_binary_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the fixed Open3D binary PLY layout generated by the RGB-D exporter."""
    expected_properties = [
        "property double x",
        "property double y",
        "property double z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
    ]
    with path.open("rb") as handle:
        header_lines: list[str] = []
        while True:
            line = handle.readline()
            if not line:
                raise RuntimeError(f"Missing PLY end_header: {path}")
            decoded = line.decode("ascii").rstrip("\n")
            header_lines.append(decoded)
            if decoded == "end_header":
                break
        if "format binary_little_endian 1.0" not in header_lines:
            raise RuntimeError(f"Unsupported PLY format: {path}")
        properties = [line for line in header_lines if line.startswith("property ")]
        if properties != expected_properties:
            raise RuntimeError(f"Unsupported PLY properties in {path}: {properties}")
        vertex_line = next((line for line in header_lines if line.startswith("element vertex ")), None)
        if vertex_line is None:
            raise RuntimeError(f"Missing PLY vertex count: {path}")
        vertex_count = int(vertex_line.rsplit(" ", 1)[1])
        dtype = np.dtype(
            [
                ("x", "<f8"),
                ("y", "<f8"),
                ("z", "<f8"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ]
        )
        vertices = np.fromfile(handle, dtype=dtype, count=vertex_count)
    if len(vertices) != vertex_count:
        raise RuntimeError(f"Truncated PLY payload: {path}")
    points = np.column_stack((vertices["x"], vertices["y"], vertices["z"]))
    colors = np.column_stack((vertices["red"], vertices["green"], vertices["blue"]))
    return points, colors


def write_open3d_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    if points.shape != colors.shape or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points and colors must both have shape [N, 3]")
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment Created by VGGT RGB-D chunk alignment\n"
        f"element vertex {len(points)}\n"
        "property double x\n"
        "property double y\n"
        "property double z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    dtype = np.dtype(
        [
            ("x", "<f8"),
            ("y", "<f8"),
            ("z", "<f8"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    vertices = np.empty(len(points), dtype=dtype)
    for coordinate, name in enumerate(("x", "y", "z")):
        vertices[name] = points[:, coordinate]
    for channel, name in enumerate(("red", "green", "blue")):
        vertices[name] = np.clip(np.rint(colors[:, channel]), 0, 255).astype(np.uint8)
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        vertices.tofile(handle)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def voxel_downsample(points: np.ndarray, colors: np.ndarray, voxel_size_m: float) -> tuple[np.ndarray, np.ndarray]:
    keys = np.floor(points / voxel_size_m).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    count = np.bincount(inverse).astype(np.float64)
    sampled_points = np.column_stack([np.bincount(inverse, weights=points[:, axis]) / count for axis in range(3)])
    sampled_colors = np.column_stack([np.bincount(inverse, weights=colors[:, axis]) / count for axis in range(3)])
    return sampled_points, sampled_colors


def global_frame_poses(
    chunks: list[ChunkResult], chunk_to_global: list[np.ndarray]
) -> tuple[list[str], np.ndarray, dict[str, int]]:
    candidates: dict[str, list[np.ndarray]] = {}
    for chunk, global_from_local in zip(chunks, chunk_to_global):
        for stem, camera_to_local in zip(chunk.frame_stems, chunk.camera_to_local_world):
            candidates.setdefault(stem, []).append(global_from_local @ camera_to_local)
    stems = sorted(candidates)
    poses = np.stack([average_se3(candidates[stem]) for stem in stems])
    observation_counts = {stem: len(candidates[stem]) for stem in stems}
    return stems, poses, observation_counts


def summarize_edge_residuals(edges: list[dict[str, object]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key in (
        "translation_residual_m_median",
        "translation_residual_m_max",
        "rotation_residual_deg_median",
        "rotation_residual_deg_max",
    ):
        values = np.asarray([edge[key] for edge in edges], dtype=np.float64)
        result[f"{key}_p50"] = float(np.percentile(values, 50))
        result[f"{key}_p95"] = float(np.percentile(values, 95))
        result[f"{key}_worst"] = float(np.max(values))
    return result


def main() -> int:
    config = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    all_pairs = collect_all_pairs(config.session_dir)
    starts = chunk_start_indices(len(all_pairs), config.chunk_size, config.stride)
    if config.max_chunks is not None:
        starts = starts[: config.max_chunks]
    if len(starts) < 2:
        raise RuntimeError("At least two overlapping chunks are required")

    model = VGGTOmega().eval().to("cuda")
    model.load_state_dict(load_checkpoint_state_dict(config.checkpoint))
    chunks: list[ChunkResult] = []
    for chunk_index, start in enumerate(starts):
        pairs = all_pairs[start : start + config.chunk_size]
        chunk = infer_chunk(model, pairs, start, config)
        chunks.append(chunk)
        print(f"chunk {chunk_index}: start={start} scale={chunk.scale:.6f}")

    chunk_to_global = [np.eye(4, dtype=np.float64)]
    edges: list[dict[str, object]] = []
    for index in range(1, len(chunks)):
        previous_to_current, metrics = estimate_adjacent_chunk_transform(chunks[index - 1], chunks[index])
        chunk_to_global.append(chunk_to_global[-1] @ np.linalg.inv(previous_to_current))
        edges.append(
            {
                "source_chunk": index - 1,
                "target_chunk": index,
                **metrics,
                "transform_target_from_source": previous_to_current.tolist(),
            }
        )

    frame_stems, camera_to_global, observation_counts = global_frame_poses(chunks, chunk_to_global)
    point_cloud_dir = config.session_dir / MASKED_CLOUD_DIR
    aligned_dir = config.output_dir / "aligned_masked_clouds"
    all_points: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    input_point_count = 0
    for stem, global_pose in zip(frame_stems, camera_to_global):
        source = point_cloud_dir / f"{stem}{MASKED_CLOUD_SUFFIX}"
        points, colors = read_open3d_binary_ply(source)
        aligned = transform_points(points, global_pose)
        write_open3d_binary_ply(aligned_dir / f"{stem}_aligned.ply", aligned, colors)
        all_points.append(aligned)
        all_colors.append(colors)
        input_point_count += len(aligned)

    fused_points, fused_colors = voxel_downsample(
        np.concatenate(all_points),
        np.concatenate(all_colors),
        config.fusion_voxel_size_m,
    )
    fused_path = config.output_dir / "fused_masked_vggt_initial_pose.ply"
    write_open3d_binary_ply(fused_path, fused_points, fused_colors)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        config.output_dir / "vggt_initial_pose_alignment.npz",
        frame_stems=np.asarray(frame_stems),
        camera_to_global=camera_to_global,
        chunk_to_global=np.stack(chunk_to_global),
    )
    summary = {
        "mode": "vggt_initial_pose_shared_frame_chain",
        "session": config.session_dir.name,
        "chunk_size": config.chunk_size,
        "stride": config.stride,
        "chunk_count": len(chunks),
        "unique_frame_count": len(frame_stems),
        "chunk_scales": [chunk.scale for chunk in chunks],
        "frame_observation_counts": observation_counts,
        "edges": edges,
        "edge_residual_summary": summarize_edge_residuals(edges),
        "input_point_count": input_point_count,
        "fused_point_count": len(fused_points),
        "fusion_voxel_size_m": config.fusion_voxel_size_m,
        "outputs": {
            "aligned_cloud_dir": str(aligned_dir),
            "fused_cloud": str(fused_path),
            "poses": str(config.output_dir / "vggt_initial_pose_alignment.npz"),
        },
    }
    summary_path = config.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "chunk_count": len(chunks),
                "unique_frame_count": len(frame_stems),
                "input_point_count": input_point_count,
                "fused_point_count": len(fused_points),
                "edge_residual_summary": summary["edge_residual_summary"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
