"""Export privacy-minimal baseline/finetuned reconstructions for one anonymous sequence."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from vggt_omega.pipeline import SceneResult, VGGTOmegaPipeline
from vggt_omega.training.dataset import ColmapRgbdDataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--after-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--near-depth-m", type=float, default=1.2)
    parser.add_argument("--max-points", type=int, default=250_000)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape:
        raise ValueError("PLY points/colors must have matching Nx3 shapes")
    if not np.isfinite(points).all() or colors.dtype != np.uint8:
        raise ValueError("PLY points must be finite and colors must be uint8")
    vertices = np.empty(
        len(points),
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    vertices["x"], vertices["y"], vertices["z"] = points.T.astype(np.float32)
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


def _select_points(
    scene: SceneResult,
    *,
    scale_m: float,
    near_depth_m: float | None,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    result = scene.with_world_points()
    assert result.world_points is not None
    points = result.world_points.astype(np.float32) * scale_m
    depth = result.depth[..., 0].astype(np.float32) * scale_m
    colors = np.rint(result.images.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255).clip(0, 255).astype(np.uint8)
    valid = np.isfinite(points).all(axis=-1) & np.isfinite(depth) & (depth > 0)
    if near_depth_m is not None:
        valid &= depth < near_depth_m
    selected_points = points[valid]
    selected_colors = colors[valid]
    if len(selected_points) > max_points:
        indices = np.linspace(0, len(selected_points) - 1, max_points, dtype=np.int64)
        selected_points = selected_points[indices]
        selected_colors = selected_colors[indices]
    return selected_points, selected_colors


def _depth_preview(depth_m: np.ndarray, near_depth_m: float) -> np.ndarray:
    first = depth_m[0].astype(np.float32)
    finite = np.isfinite(first) & (first > 0)
    normalized = np.zeros_like(first, dtype=np.float32)
    normalized[finite] = np.clip(first[finite] / near_depth_m, 0, 1)
    red = np.rint(normalized * 255).astype(np.uint8)
    return np.stack((red, 255 - red, np.zeros_like(red)), axis=-1)


def _mean_valid_absolute_difference(first: np.ndarray, second: np.ndarray) -> float:
    valid = np.isfinite(first) & np.isfinite(second) & (first > 0) & (second > 0)
    if not valid.any():
        return 0.0
    return float(np.abs(first[valid] - second[valid]).mean(dtype=np.float64))


def _depth_diagnostics(depth_m: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(depth_m) & (depth_m > 0)
    horizontal = _mean_valid_absolute_difference(depth_m[..., 1:], depth_m[..., :-1])
    vertical = _mean_valid_absolute_difference(depth_m[..., 1:, :], depth_m[..., :-1, :])
    temporal = _mean_valid_absolute_difference(depth_m[1:], depth_m[:-1]) if depth_m.shape[0] > 1 else 0.0
    return {
        "finite_depth_fraction": float(finite.mean(dtype=np.float64)),
        "spatial_edge_l1_m": float((horizontal + vertical) / 2.0),
        "temporal_depth_change_proxy_m": temporal,
    }


def _first_camera_identity_error(extrinsic: np.ndarray) -> float:
    expected = np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)
    return float(np.abs(extrinsic[0].astype(np.float64) - expected).max())


def _validate_reconstruction_contract(
    baseline: SceneResult,
    after: SceneResult,
    *,
    scale_m: float,
    first_camera_tolerance: float = 1e-2,
) -> None:
    if not np.isfinite(scale_m) or scale_m <= 0:
        raise ValueError("scene normalization scale must be finite and positive")
    if baseline.depth.shape != after.depth.shape or baseline.images.shape != after.images.shape:
        raise ValueError("baseline and after reconstructions must describe the same frames and image grid")
    for label, scene in (("baseline", baseline), ("after", after)):
        result = scene.with_world_points()
        assert result.world_points is not None
        arrays = (result.depth, result.extrinsic, result.intrinsic, result.world_points)
        if any(not np.isfinite(array).all() for array in arrays):
            raise ValueError(f"{label} reconstruction contains non-finite geometry")
        if not (result.depth > 0).all():
            raise ValueError(f"{label} reconstruction contains non-positive depth")
        if _first_camera_identity_error(result.extrinsic) > first_camera_tolerance:
            raise ValueError(f"{label} first camera is not normalized to the reference frame")


def _scene_artifacts(
    scene: SceneResult,
    *,
    prefix: str,
    output_dir: Path,
    scale_m: float,
    near_depth_m: float,
    max_points: int,
) -> dict[str, Any]:
    all_points, all_colors = _select_points(
        scene,
        scale_m=scale_m,
        near_depth_m=None,
        max_points=max_points,
    )
    near_points, near_colors = _select_points(
        scene,
        scale_m=scale_m,
        near_depth_m=near_depth_m,
        max_points=max_points,
    )
    _write_binary_ply(output_dir / f"{prefix}_all.ply", all_points, all_colors)
    _write_binary_ply(output_dir / f"{prefix}_near.ply", near_points, near_colors)
    depth_m = scene.depth[..., 0].astype(np.float32) * scale_m
    Image.fromarray(_depth_preview(depth_m, near_depth_m)).save(output_dir / f"{prefix}_depth.png")
    return {
        "all_point_count": len(all_points),
        "finite_camera_count": int(np.isfinite(scene.extrinsic).all(axis=(1, 2)).sum()),
        "first_camera_identity_max_abs": _first_camera_identity_error(scene.extrinsic),
        "finite_intrinsic_count": int(np.isfinite(scene.intrinsic).all(axis=(1, 2)).sum()),
        "frame_count": int(scene.depth.shape[0]),
        "near_point_count": len(near_points),
        **_depth_diagnostics(depth_m),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.sample_index < 0 or args.near_depth_m <= 0 or args.max_points < 1:
        raise ValueError("sample-index, near-depth-m, and max-points are invalid")
    data_root = Path(args.data_root).expanduser().resolve()
    base_checkpoint = Path(args.base_checkpoint).expanduser().resolve()
    after_checkpoint = Path(args.after_checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    dataset = ColmapRgbdDataset(
        data_root,
        split="smoke",
        min_frames=4,
        max_frames=4,
        filter_short_sequences=True,
        seed=2026,
        min_valid_depth_pixels=1024,
    )
    dataset.set_epoch(0)
    if args.sample_index >= len(dataset):
        raise IndexError("sample-index is outside the anonymous smoke split")
    sample = dataset[args.sample_index]
    images = sample["images"]
    scale_m = float(sample["normalization_scale_m"])
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_pipeline = VGGTOmegaPipeline(base_checkpoint, device=args.device)
    baseline = baseline_pipeline.run(images).with_world_points()
    del baseline_pipeline
    after_pipeline = VGGTOmegaPipeline(
        base_checkpoint,
        head_checkpoint_path=after_checkpoint,
        device=args.device,
    )
    after = after_pipeline.run(images).with_world_points()
    del after_pipeline
    _validate_reconstruction_contract(baseline, after, scale_m=scale_m)
    baseline_summary = _scene_artifacts(
        baseline,
        prefix="baseline",
        output_dir=output_dir,
        scale_m=scale_m,
        near_depth_m=args.near_depth_m,
        max_points=args.max_points,
    )
    after_summary = _scene_artifacts(
        after,
        prefix="after",
        output_dir=output_dir,
        scale_m=scale_m,
        near_depth_m=args.near_depth_m,
        max_points=args.max_points,
    )
    panel = np.concatenate(
        (
            _depth_preview(baseline.depth[..., 0] * scale_m, args.near_depth_m),
            _depth_preview(after.depth[..., 0] * scale_m, args.near_depth_m),
        ),
        axis=1,
    )
    Image.fromarray(panel).save(output_dir / "baseline_after_depth.png")
    summary = {
        "after": after_summary,
        "after_checkpoint_sha256": _sha256(after_checkpoint),
        "base_checkpoint_sha256": _sha256(base_checkpoint),
        "baseline": baseline_summary,
        "correspondence": {
            "frame_count_equal": baseline_summary["frame_count"] == after_summary["frame_count"],
            "image_shape_equal": list(baseline.images.shape) == list(after.images.shape),
        },
        "format_version": 1,
        "near_depth_m": float(args.near_depth_m),
        "sample_index": int(args.sample_index),
    }
    temporary = output_dir / ".summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
