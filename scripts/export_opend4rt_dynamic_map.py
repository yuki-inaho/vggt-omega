"""Fuse OpenD4RT dynamic tracks with measured RGB-D and aligned metric poses."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rgbd_sfm_pilot import DEFAULT_INTRINSICS, _write_binary_ply  # noqa: E402


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def export_opend4rt_dynamic_map(
    workflow_root: Path,
    chunks_dir: Path,
    trajectory_path: Path,
    output_path: Path,
    fx: float = DEFAULT_INTRINSICS[0],
    fy: float = DEFAULT_INTRINSICS[1],
    cx: float = DEFAULT_INTRINSICS[2],
    cy: float = DEFAULT_INTRINSICS[3],
    min_depth_m: float = 0.1,
    max_depth_m: float = 1.3,
    voxel_size_m: float = 0.005,
) -> dict[str, Any]:
    manifest = json.loads((workflow_root / "manifest.json").read_text())
    chunk_manifest = json.loads((chunks_dir / "chunks_manifest.json").read_text())
    trajectory = json.loads(trajectory_path.read_text())
    frames = {int(frame["frame_index"]): frame for frame in manifest["frames"]}
    poses = {
        int(frame.get("frame_index", frame.get("global_index"))): np.asarray(frame["camera_to_world"], dtype=np.float64)
        for frame in trajectory["frames"]
    }
    seen: set[int] = set()
    per_frame_counts = [0] * len(frames)
    all_points: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    dynamic_tracks = 0
    visible_dynamic_observations = 0

    for chunk in chunk_manifest["chunks"]:
        scene_path = chunks_dir / f"chunk_{int(chunk['chunk_index']):06d}" / "dense_scene.npz"
        with np.load(scene_path, allow_pickle=False) as scene:
            uv = np.asarray(scene["point_uv_px"], dtype=np.float64)
            visibility = np.asarray(scene["point_visibility"], dtype=bool)
            dynamic = np.asarray(scene["point_is_dynamic"], dtype=bool)
            indices = np.asarray(scene["global_indices"], dtype=np.int64)
        dynamic_tracks += int(dynamic.sum())
        for local_index, global_index_raw in enumerate(indices):
            global_index = int(global_index_raw)
            if global_index in seen:
                continue
            seen.add(global_index)
            frame = frames[global_index]
            image = cv2.imread(str(workflow_root / "images" / frame["image_name"]), cv2.IMREAD_COLOR)
            depth_mm = cv2.imread(str(workflow_root / "mapped_depth" / frame["depth_name"]), cv2.IMREAD_UNCHANGED)
            if image is None or depth_mm is None:
                raise FileNotFoundError(f"missing RGB-D for frame {global_index}")
            height, width = depth_mm.shape
            selected = dynamic & visibility[local_index] & np.isfinite(uv[local_index]).all(axis=1)
            selected_uv = uv[local_index][selected]
            x = np.rint(selected_uv[:, 0]).astype(np.int64)
            y = np.rint(selected_uv[:, 1]).astype(np.int64)
            in_bounds = (x >= 0) & (x < width) & (y >= 0) & (y < height)
            x, y = x[in_bounds], y[in_bounds]
            depth_m = depth_mm[y, x].astype(np.float64) / 1000.0
            valid_depth = np.isfinite(depth_m) & (depth_m >= min_depth_m) & (depth_m <= max_depth_m)
            x, y, depth_m = x[valid_depth], y[valid_depth], depth_m[valid_depth]
            visible_dynamic_observations += int(len(depth_m))
            if not len(depth_m):
                continue
            camera_points = np.stack(((x - cx) * depth_m / fx, (y - cy) * depth_m / fy, depth_m), axis=1)
            pose = poses[global_index]
            world_points = camera_points @ pose[:3, :3].T + pose[:3, 3]
            colors = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)[y, x]
            all_points.append(world_points)
            all_colors.append(colors)
            per_frame_counts[global_index] = int(len(world_points))

    if seen != set(frames):
        missing = sorted(set(frames) - seen)
        raise ValueError(f"OpenD4RT chunks do not cover all workflow frames: {missing[:10]}")
    if not all_points:
        raise ValueError("no valid OpenD4RT dynamic RGB-D observations")
    points = np.concatenate(all_points).astype(np.float32)
    colors = np.concatenate(all_colors).astype(np.uint8)
    voxel_keys = np.floor(points / voxel_size_m).astype(np.int64)
    _, selected_indices = np.unique(voxel_keys, axis=0, return_index=True)
    selected_indices.sort()
    points = points[selected_indices]
    colors = colors[selected_indices]
    _write_binary_ply(output_path, points, colors)
    summary = {
        "schema_version": 1,
        "description": "metric dynamic evidence map from OpenD4RT labels/tracks and measured RGB-D",
        "output_path": str(output_path.resolve()),
        "frame_count": len(frames),
        "covered_frame_count": len(seen),
        "chunk_count": int(chunk_manifest["chunk_count"]),
        "dynamic_tracks_summed_per_chunk": dynamic_tracks,
        "visible_dynamic_observations_with_depth": visible_dynamic_observations,
        "per_frame_points_before_voxel": per_frame_counts,
        "point_count_before_voxel": int(sum(per_frame_counts)),
        "point_count_after_voxel": int(len(points)),
        "voxel_size_m": voxel_size_m,
        "depth_range_m": [min_depth_m, max_depth_m],
        "pose_source": str(trajectory_path.resolve()),
        "opend4rt_chunks": str(chunks_dir.resolve()),
    }
    _write_json(output_path.with_suffix(".json"), summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow-root", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = export_opend4rt_dynamic_map(args.workflow_root, args.chunks, args.trajectory, args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
