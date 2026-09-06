#!/usr/bin/env python3
"""Reconstruct a staged RGB-D sequence with VGGT-Omega and write RGB-D videos.

The released VGGT-Omega model predicts relative depth and camera poses from
RGB.  This script uses the measured RGB-D depth only to recover metric scale,
anchors each short inference window to the staged camera trajectory, and then
exports a fused point cloud plus two compact videos:

* ``rgbd_prediction.mp4``: input RGB, VGGT metric-depth prediction, and the
  measured RGB-D reference depth;
* ``rgbd_reconstruction_render.mp4``: input RGB, fused-cloud reprojection, and
  the reprojection depth at the same staged camera poses.

The staged dataset is deliberately read through its public on-disk contract;
no private source paths or frame names are written to the summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from vggt_omega.pipeline import SceneResult, VGGTOmegaPipeline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--head-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--point-stride", type=int, default=8)
    parser.add_argument("--max-points", type=int, default=1_500_000)
    parser.add_argument("--render-max-points", type=int, default=250_000)
    parser.add_argument("--render-splat-radius", type=int, default=1)
    parser.add_argument("--video-width", type=int, default=320)
    parser.add_argument("--video-height", type=int, default=240)
    parser.add_argument("--video-fps", type=float, default=8.0)
    parser.add_argument("--max-windows", type=int, default=0, help="0 means all trajectory chunks")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_dataset_arrays(data_root: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metadata = json.loads((data_root / "dataset.json").read_text(encoding="utf-8"))
    scene_root = data_root / "scenes" / "scene_000000"
    with np.load(scene_root / "cameras.npz", allow_pickle=False) as cameras:
        camera_arrays = {key: cameras[key].copy() for key in cameras.files}
    required = {"frame_ids", "intrinsics", "extrinsics_w2c", "chunk_ids"}
    if not required.issubset(camera_arrays):
        raise ValueError(f"camera archive is missing required arrays: {sorted(required - set(camera_arrays))}")
    if metadata.get("format") not in {"colmap_rgbd_v1", "colmap_rgbd_v2"}:
        raise ValueError("unsupported staged RGB-D dataset format")
    height = int(metadata["image"]["height"])
    width = int(metadata["image"]["width"])
    if (height, width) != (480, 640):
        raise ValueError(f"this export expects the 640x480 RGB-D contract, got {(width, height)}")
    return metadata, {"scene_root": scene_root, **camera_arrays}


def _chunk_windows(camera_arrays: dict[str, np.ndarray], window_size: int) -> list[np.ndarray]:
    frame_ids = np.asarray(camera_arrays["frame_ids"], dtype=np.int64)
    chunk_ids = np.asarray(camera_arrays["chunk_ids"], dtype=np.int64)
    windows: list[np.ndarray] = []
    for chunk_id in np.unique(chunk_ids):
        ids = frame_ids[chunk_ids == chunk_id]
        if len(ids) < 2:
            continue
        count = min(int(window_size), len(ids))
        windows.append(ids[:count].copy())
    if not windows:
        raise ValueError("staged trajectory has no chunk with at least two frames")
    return windows


def _read_rgb_depth(scene_root: Path, frame_id: int, target_width: int, target_height: int) -> tuple[np.ndarray, np.ndarray]:
    filename = f"frame_{int(frame_id):06d}.png"
    rgb_path = scene_root / "rgb" / filename
    depth_path = scene_root / "depth" / filename
    with Image.open(rgb_path) as image:
        rgb_image = image.convert("RGB")
        rgb_resized = np.array(
            rgb_image.resize((target_width, target_height), Image.Resampling.BICUBIC), dtype=np.uint8, copy=True
        )
    with Image.open(depth_path) as image:
        depth = np.array(image, dtype=np.uint16, copy=True)
    depth_resized = np.asarray(
        Image.fromarray(depth).resize((target_width, target_height), Image.Resampling.NEAREST), dtype=np.uint16
    )
    return rgb_resized, depth_resized.astype(np.float32, copy=False) * 0.001


def _load_model_inputs(
    scene_root: Path, frame_ids: np.ndarray, target_width: int, target_height: int
) -> tuple[torch.Tensor, np.ndarray, list[np.ndarray]]:
    images: list[torch.Tensor] = []
    metric_depth: list[np.ndarray] = []
    original_rgb: list[np.ndarray] = []
    for frame_id in frame_ids:
        rgb, depth = _read_rgb_depth(scene_root, int(frame_id), target_width, target_height)
        original_rgb.append(rgb)
        metric_depth.append(depth)
        images.append(torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255))
    return torch.stack(images), np.stack(metric_depth), original_rgb


def _metric_scale(predicted_depth: np.ndarray, measured_depth: np.ndarray) -> tuple[float, int]:
    valid = (
        np.isfinite(predicted_depth)
        & np.isfinite(measured_depth)
        & (predicted_depth > 1e-5)
        & (measured_depth >= 0.10)
        & (measured_depth <= 1.30)
    )
    ratios = (measured_depth[valid] / predicted_depth[valid]).astype(np.float64)
    if ratios.size < 1000:
        raise RuntimeError(f"only {ratios.size} valid RGB-D pixels are available for metric scale")
    low, high = np.quantile(ratios, (0.05, 0.95))
    trimmed = ratios[(ratios >= low) & (ratios <= high)]
    scale = float(np.median(trimmed))
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("metric scale is not finite and positive")
    return scale, int(trimmed.size)


def _invert_w2c(extrinsic_w2c: np.ndarray) -> np.ndarray:
    homogeneous = np.eye(4, dtype=np.float64)
    homogeneous[:3, :4] = np.asarray(extrinsic_w2c, dtype=np.float64)
    inverse = np.linalg.inv(homogeneous)
    if not np.isfinite(inverse).all():
        raise ValueError("camera-to-world inversion produced non-finite values")
    return inverse


def _transform_points(points: np.ndarray, camera_to_world: np.ndarray) -> np.ndarray:
    return points @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]


def _select_finite_points(
    scene: SceneResult,
    metric_depth: np.ndarray,
    camera_to_world_anchor: np.ndarray,
    scale: float,
    point_stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    result = scene.with_world_points()
    assert result.world_points is not None
    world_points: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    image_array = result.images.detach().cpu().numpy().transpose(0, 2, 3, 1)
    depth_array = result.depth[..., 0].astype(np.float32)
    confidence = result.depth_conf.astype(np.float32)
    for index in range(len(depth_array)):
        finite_conf = confidence[index][np.isfinite(confidence[index])]
        threshold = float(np.quantile(finite_conf, 0.20)) if finite_conf.size else -np.inf
        valid = (
            np.isfinite(result.world_points[index]).all(axis=-1)
            & np.isfinite(depth_array[index])
            & (depth_array[index] > 1e-5)
            & np.isfinite(metric_depth[index])
            & (metric_depth[index] > 0)
            & (confidence[index] >= threshold)
        )
        sampled = np.zeros_like(valid)
        sampled[::point_stride, ::point_stride] = True
        valid &= sampled
        local = result.world_points[index][valid].astype(np.float32) * np.float32(scale)
        if not len(local):
            continue
        world_points.append(_transform_points(local, camera_to_world_anchor).astype(np.float32))
        colors.append(np.rint(image_array[index][valid] * 255).clip(0, 255).astype(np.uint8))
    if not world_points:
        raise RuntimeError("no finite RGB-D points survived the reconstruction mask")
    return np.concatenate(world_points, axis=0), np.concatenate(colors, axis=0)


def _write_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape:
        raise ValueError("PLY points and colors must have matching Nx3 shapes")
    if not np.isfinite(points).all() or colors.dtype != np.uint8:
        raise ValueError("PLY arrays must be finite and colors must be uint8")
    vertices = np.empty(
        len(points),
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    vertices["x"], vertices["y"], vertices["z"] = points.T
    vertices["red"], vertices["green"], vertices["blue"] = colors.T
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    ).encode("ascii")
    with path.open("wb") as stream:
        stream.write(header)
        vertices.tofile(stream)


def _depth_color(depth_m: np.ndarray, *, max_depth_m: float = 1.30) -> np.ndarray:
    finite = np.isfinite(depth_m) & (depth_m > 0)
    normalized = np.zeros(depth_m.shape, dtype=np.float32)
    normalized[finite] = np.clip(depth_m[finite] / max_depth_m, 0, 1)
    color = cv2.applyColorMap(np.rint(normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    color[~finite] = 0
    return color


def _label(image: np.ndarray, text: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 24), (0, 0, 0), thickness=-1)
    cv2.putText(result, text, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return result


def _render_point_cloud(
    points: np.ndarray,
    colors: np.ndarray,
    extrinsic_w2c: np.ndarray,
    intrinsic: np.ndarray,
    width: int,
    height: int,
    splat_radius: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    rotation = np.asarray(extrinsic_w2c[:3, :3], dtype=np.float32)
    translation = np.asarray(extrinsic_w2c[:3, 3], dtype=np.float32)
    camera_points = points @ rotation.T + translation
    z = camera_points[:, 2]
    valid = np.isfinite(camera_points).all(axis=1) & (z > 1e-4) & (z < 5.0)
    if not valid.any():
        return np.zeros((height, width, 3), dtype=np.uint8), np.zeros((height, width), dtype=np.float32)
    camera_points = camera_points[valid]
    z = z[valid]
    colors = colors[valid]
    scale_x = width / 640.0
    scale_y = height / 480.0
    fx = float(intrinsic[0, 0]) * scale_x
    fy = float(intrinsic[1, 1]) * scale_y
    cx = float(intrinsic[0, 2]) * scale_x
    cy = float(intrinsic[1, 2]) * scale_y
    x = np.rint(fx * camera_points[:, 0] / z + cx).astype(np.int64)
    y = np.rint(fy * camera_points[:, 1] / z + cy).astype(np.int64)
    in_bounds = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    if not in_bounds.any():
        return np.zeros((height, width, 3), dtype=np.uint8), np.zeros((height, width), dtype=np.float32)
    x = x[in_bounds]
    y = y[in_bounds]
    z = z[in_bounds]
    colors = colors[in_bounds]
    # A sparse RGB-D point cloud is easier to inspect when each point covers a
    # small pixel footprint.  Keep the nearest point per destination pixel so
    # this remains a conventional z-buffer rather than an alpha blend.
    radius = max(int(splat_radius), 0)
    if radius:
        offsets = np.asarray(
            [(dx, dy) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)], dtype=np.int64
        )
        x = (x[:, None] + offsets[None, :, 0]).reshape(-1)
        y = (y[:, None] + offsets[None, :, 1]).reshape(-1)
        z = np.repeat(z, len(offsets))
        colors = np.repeat(colors, len(offsets), axis=0)
        footprint_valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        x = x[footprint_valid]
        y = y[footprint_valid]
        z = z[footprint_valid]
        colors = colors[footprint_valid]
    pixel = y * width + x
    order = np.lexsort((z, pixel))
    sorted_pixels = pixel[order]
    _, first = np.unique(sorted_pixels, return_index=True)
    chosen = order[first]
    rendered_rgb = np.zeros((height * width, 3), dtype=np.uint8)
    rendered_depth = np.zeros(height * width, dtype=np.float32)
    rendered_rgb[pixel[chosen]] = colors[chosen]
    rendered_depth[pixel[chosen]] = z[chosen]
    return rendered_rgb.reshape(height, width, 3), rendered_depth.reshape(height, width)


def _open_writer(path: Path, width: int, height: int, fps: float) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not open MP4 writer: {path}")
    return writer


def _subsample(points: np.ndarray, colors: np.ndarray, maximum: int) -> tuple[np.ndarray, np.ndarray]:
    if len(points) <= maximum:
        return points, colors
    indices = np.linspace(0, len(points) - 1, maximum, dtype=np.int64)
    return points[indices], colors[indices]


def main() -> int:
    args = _parser().parse_args()
    if (
        args.window_size < 2
        or args.point_stride < 1
        or args.max_points < 1
        or args.render_max_points < 1
        or args.render_splat_radius < 0
    ):
        raise ValueError("window-size, point-stride, and point limits must be positive")
    if args.video_width < 16 or args.video_height < 16 or args.video_fps <= 0:
        raise ValueError("video dimensions and FPS are invalid")
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA is required for the requested device")

    data_root = args.data_root.expanduser().resolve()
    base_checkpoint = args.base_checkpoint.expanduser().resolve()
    head_checkpoint = args.head_checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata, arrays = _load_dataset_arrays(data_root)
    scene_root = Path(arrays["scene_root"])
    windows = _chunk_windows(arrays, args.window_size)
    if args.max_windows > 0:
        windows = windows[: args.max_windows]

    # The accepted VGGT-Omega head was trained at 384x512 (the 640x480 source
    # has the same 4:3 aspect ratio), so preserve that model grid exactly.
    model_height, model_width = 384, 512
    pipeline = VGGTOmegaPipeline(
        base_checkpoint,
        head_checkpoint_path=head_checkpoint,
        device=args.device,
    )
    if pipeline.recommended_input_shape not in {None, (model_height, model_width)}:
        raise ValueError(f"head checkpoint recommends {pipeline.recommended_input_shape}, not {(model_height, model_width)}")

    fused_points: list[np.ndarray] = []
    fused_colors: list[np.ndarray] = []
    video_records: list[dict[str, Any]] = []
    predicted_depths: list[np.ndarray] = []
    measured_depths: list[np.ndarray] = []
    global_camera_to_world: list[np.ndarray] = []
    target_extrinsics: list[np.ndarray] = []
    target_intrinsics: list[np.ndarray] = []
    scales: list[float] = []
    valid_scale_pixels: list[int] = []
    original_rgbs: list[np.ndarray] = []
    start_time = time.perf_counter()

    for window_index, frame_ids in enumerate(windows):
        images, measured_depth, _ = _load_model_inputs(scene_root, frame_ids, model_width, model_height)
        scene = pipeline.run(images)
        predicted_depth = scene.depth[..., 0].astype(np.float32)
        scale, scale_pixels = _metric_scale(predicted_depth, measured_depth)
        anchor_id = int(frame_ids[0])
        anchor_index = int(np.flatnonzero(arrays["frame_ids"] == anchor_id)[0])
        anchor_c2w = _invert_w2c(arrays["extrinsics_w2c"][anchor_index])
        points, colors = _select_finite_points(
            scene,
            measured_depth,
            anchor_c2w,
            scale,
            args.point_stride,
        )
        fused_points.append(points)
        fused_colors.append(colors)

        representative_id = int(frame_ids[0])
        representative_index = int(np.flatnonzero(frame_ids == representative_id)[0])
        representative_global_c2w = anchor_c2w.copy()
        predicted_depths.append(predicted_depth[representative_index] * np.float32(scale))
        measured_depths.append(measured_depth[representative_index])
        source_rgb, _ = _read_rgb_depth(scene_root, representative_id, model_width, model_height)
        original_rgbs.append(source_rgb)
        global_camera_to_world.append(representative_global_c2w)
        target_extrinsics.append(arrays["extrinsics_w2c"][anchor_index].copy())
        target_intrinsics.append(arrays["intrinsics"][anchor_index].copy())
        scales.append(scale)
        valid_scale_pixels.append(scale_pixels)
        video_records.append(
            {
                "chunk_index": int(window_index),
                "frame_id": representative_id,
                "window_frame_ids": [int(value) for value in frame_ids],
                "metric_scale_m": scale,
                "scale_valid_pixels": scale_pixels,
                "predicted_depth_finite_fraction": float(np.isfinite(predicted_depth).mean()),
            }
        )
        if (window_index + 1) % 10 == 0 or window_index + 1 == len(windows):
            elapsed = time.perf_counter() - start_time
            print(f"processed {window_index + 1}/{len(windows)} windows ({elapsed:.1f}s)", flush=True)

    points = np.concatenate(fused_points, axis=0)
    colors = np.concatenate(fused_colors, axis=0)
    points, colors = _subsample(points, colors, args.max_points)
    _write_binary_ply(output_dir / "fused_reconstruction.ply", points, colors)

    predicted_depths_array = np.stack(predicted_depths).astype(np.float32)
    measured_depths_array = np.stack(measured_depths).astype(np.float32)
    original_rgbs_array = np.stack(original_rgbs).astype(np.uint8)
    np.savez_compressed(
        output_dir / "reconstruction_predictions.npz",
        frame_ids=np.asarray([record["frame_id"] for record in video_records], dtype=np.int64),
        metric_scale_m=np.asarray(scales, dtype=np.float32),
        predicted_depth_m=predicted_depths_array,
        measured_depth_m=measured_depths_array,
        camera_to_world=np.stack(global_camera_to_world).astype(np.float32),
        target_extrinsics_w2c=np.stack(target_extrinsics).astype(np.float32),
        target_intrinsics=np.stack(target_intrinsics).astype(np.float32),
    )

    render_points, render_colors = _subsample(points, colors, args.render_max_points)
    prediction_writer = _open_writer(
        output_dir / "rgbd_prediction.mp4", args.video_width * 3, args.video_height, args.video_fps
    )
    render_writer = _open_writer(
        output_dir / "rgbd_reconstruction_render.mp4", args.video_width * 3, args.video_height, args.video_fps
    )
    try:
        for index in range(len(video_records)):
            rgb = cv2.resize(original_rgbs_array[index], (args.video_width, args.video_height), interpolation=cv2.INTER_AREA)
            predicted_depth = cv2.resize(
                _depth_color(predicted_depths_array[index]), (args.video_width, args.video_height), interpolation=cv2.INTER_LINEAR
            )
            measured_depth = cv2.resize(
                _depth_color(measured_depths_array[index]), (args.video_width, args.video_height), interpolation=cv2.INTER_NEAREST
            )
            prediction_panel = np.concatenate(
                (
                    _label(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), "input RGB"),
                    _label(predicted_depth, "VGGT metric depth"),
                    _label(measured_depth, "RGB-D reference depth"),
                ),
                axis=1,
            )
            prediction_writer.write(prediction_panel)

            rendered_rgb, rendered_depth = _render_point_cloud(
                render_points,
                render_colors,
                np.asarray(target_extrinsics[index]),
                np.asarray(target_intrinsics[index]),
                args.video_width,
                args.video_height,
                args.render_splat_radius,
            )
            rendered_depth_color = _depth_color(rendered_depth)
            render_panel = np.concatenate(
                (
                    _label(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), "input RGB"),
                    _label(cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR), "fused-cloud RGB"),
                    _label(rendered_depth_color, "fused-cloud depth"),
                ),
                axis=1,
            )
            render_writer.write(render_panel)
    finally:
        prediction_writer.release()
        render_writer.release()

    summary = {
        "format_version": 1,
        "status": "passed",
        "dataset_format": metadata["format"],
        "source_frame_count": int(metadata["frame_count"]),
        "processed_chunk_count": len(windows),
        "processed_representative_frame_count": len(video_records),
        "model_grid_hw": [model_height, model_width],
        "video_size_wh": [args.video_width * 3, args.video_height],
        "video_fps": float(args.video_fps),
        "point_count": int(len(points)),
        "render_point_count": int(len(render_points)),
        "finite_point_fraction": float(np.isfinite(points).all(axis=1).mean()),
        "metric_scale_m": {
            "min": float(np.min(scales)),
            "median": float(np.median(scales)),
            "max": float(np.max(scales)),
            "valid_pixels_min": int(np.min(valid_scale_pixels)),
        },
        "checkpoints": {
            "base_sha256": _sha256(base_checkpoint),
            "head_sha256": _sha256(head_checkpoint),
        },
        "outputs": {
            "fused_point_cloud": "fused_reconstruction.ply",
            "prediction_npz": "reconstruction_predictions.npz",
            "rgbd_prediction_video": "rgbd_prediction.mp4",
            "rgbd_reconstruction_render_video": "rgbd_reconstruction_render.mp4",
        },
        "video_records": video_records,
        "settings": {
            "window_size": int(args.window_size),
            "point_stride": int(args.point_stride),
            "max_points": int(args.max_points),
            "render_max_points": int(args.render_max_points),
            "render_splat_radius": int(args.render_splat_radius),
            "alignment": "each VGGT window anchored by the staged camera-to-world pose of its first frame",
            "scale_source": "trimmed median of measured RGB-D depth / VGGT predicted depth",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("status", "processed_chunk_count", "point_count", "outputs")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
