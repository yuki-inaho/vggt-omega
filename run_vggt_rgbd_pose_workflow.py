#!/usr/bin/env python3
"""Run fast VGGT BF16 inference, RGB-D scale correction, and pose plotting.

The workflow consumes one standardized robot-data session containing ``rgb/``
and RGB-FoV aligned ``mapped_depth_dense/`` images.  VGGT predicts relative
depth and camera poses from RGB only; valid RGB-D pixels recover one robust
metric scale which is applied to predicted depth and pose translations.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_checkpoint_state_dict
from vggt_omega.utils.pose_enc import encoding_to_camera

DEPTH_SUFFIX: Final = "_depth.png"
RGB_SUFFIX: Final = "_rgb.png"


@dataclass(frozen=True)
class WorkflowConfig:
    session_dir: Path
    checkpoint: Path
    output_dir: Path
    num_frames: int
    width: int
    height: int
    min_depth_m: float
    max_depth_m: float


def parse_args() -> WorkflowConfig:
    parser = argparse.ArgumentParser(description="VGGT BF16 RGB-D scale-corrected pose workflow.")
    parser.add_argument("--session-dir", type=Path, required=True, help="Standardized session directory.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="VGGT-Omega checkpoint.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for NPZ, JSON, and plot outputs.")
    parser.add_argument("--num-frames", type=int, default=6, help="Contiguous RGB frames to infer (default: 6).")
    parser.add_argument("--width", type=int, default=640, help="Inference width; must be divisible by 16.")
    parser.add_argument("--height", type=int, default=480, help="Inference height; must be divisible by 16.")
    parser.add_argument("--min-depth-m", type=float, default=0.10, help="Minimum valid metric depth in metres.")
    parser.add_argument("--max-depth-m", type=float, default=5.00, help="Maximum valid metric depth in metres.")
    args = parser.parse_args()
    if args.num_frames < 2:
        raise ValueError("--num-frames must be >= 2 for a pose trajectory.")
    if args.width <= 0 or args.height <= 0 or args.width % 16 or args.height % 16:
        raise ValueError("--width and --height must be positive multiples of 16.")
    if not 0 < args.min_depth_m < args.max_depth_m:
        raise ValueError("Depth limits must satisfy 0 < min < max.")
    return WorkflowConfig(**vars(args))


def collect_pairs(session_dir: Path, num_frames: int) -> list[tuple[Path, Path]]:
    rgb_dir = session_dir / "rgb"
    depth_dir = session_dir / "mapped_depth_dense"
    if not rgb_dir.is_dir() or not depth_dir.is_dir():
        raise FileNotFoundError("Session must contain rgb/ and mapped_depth_dense/ directories.")
    pairs: list[tuple[Path, Path]] = []
    for rgb_path in sorted(rgb_dir.glob(f"*{RGB_SUFFIX}")):
        stem = rgb_path.name.removesuffix(RGB_SUFFIX)
        depth_path = depth_dir / f"{stem}{DEPTH_SUFFIX}"
        if depth_path.is_file():
            pairs.append((rgb_path, depth_path))
        if len(pairs) == num_frames:
            break
    if len(pairs) != num_frames:
        raise RuntimeError(f"Requested {num_frames} matched RGB-D pairs, found {len(pairs)}.")
    return pairs


def load_inputs(pairs: list[tuple[Path, Path]], width: int, height: int) -> tuple[torch.Tensor, np.ndarray]:
    rgb_frames: list[np.ndarray] = []
    depth_frames: list[np.ndarray] = []
    for rgb_path, depth_path in pairs:
        with Image.open(rgb_path) as image:
            rgb = image.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
        with Image.open(depth_path) as image:
            depth = image.resize((width, height), Image.Resampling.NEAREST)
            depth_array = np.asarray(depth, dtype=np.uint16)
        rgb_frames.append(np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        depth_frames.append(depth_array.astype(np.float32) * 0.001)
    return torch.from_numpy(np.stack(rgb_frames)), np.stack(depth_frames)


def estimate_scale(
    predicted_depth: np.ndarray, metric_depth_m: np.ndarray, min_depth_m: float, max_depth_m: float
) -> tuple[float, int]:
    valid = (
        np.isfinite(predicted_depth)
        & np.isfinite(metric_depth_m)
        & (predicted_depth > 1e-6)
        & (metric_depth_m >= min_depth_m)
        & (metric_depth_m <= max_depth_m)
    )
    ratios = metric_depth_m[valid] / predicted_depth[valid]
    if ratios.size < 1_000:
        raise RuntimeError(f"Only {ratios.size} valid RGB-D pixels remain for scale estimation.")
    low, high = np.quantile(ratios, (0.05, 0.95))
    trimmed = ratios[(ratios >= low) & (ratios <= high)]
    return float(np.median(trimmed)), int(trimmed.size)


def camera_centres(extrinsics: np.ndarray) -> np.ndarray:
    rotations = extrinsics[:, :3, :3]
    translations = extrinsics[:, :3, 3]
    return -np.einsum("nij,nj->ni", np.transpose(rotations, (0, 2, 1)), translations)


def extrinsics_to_camera_to_global(extrinsics: np.ndarray) -> np.ndarray:
    """Convert VGGT camera-from-world 3x4 poses to homogeneous camera-to-global poses."""
    if extrinsics.ndim != 3 or extrinsics.shape[1:] != (3, 4):
        raise ValueError(f"Expected (N, 3, 4) camera-from-world poses, got {extrinsics.shape}.")
    homogeneous = np.broadcast_to(np.eye(4, dtype=np.float64), (len(extrinsics), 4, 4)).copy()
    homogeneous[:, :3, :4] = extrinsics
    camera_to_global = np.linalg.inv(homogeneous)
    if not np.isfinite(camera_to_global).all():
        raise ValueError("VGGT camera-to-global conversion produced non-finite values.")
    return camera_to_global


def save_pose_plot(centres_m: np.ndarray, output_path: Path) -> None:
    figure = plt.figure(figsize=(8, 7))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot(centres_m[:, 0], centres_m[:, 1], centres_m[:, 2], "o-", label="VGGT RGB-D scaled pose")
    markers = axis.scatter(
        centres_m[:, 0], centres_m[:, 1], centres_m[:, 2], c=np.arange(len(centres_m)), cmap="viridis", s=42
    )
    axis.set_xlabel("X [m]")
    axis.set_ylabel("Y [m]")
    axis.set_zlabel("Z [m]")
    axis.set_title("VGGT camera centres (RGB-D metric scale corrected)")
    axis.legend()
    figure.colorbar(markers, ax=axis, pad=0.12, label="Frame index")
    axis.set_box_aspect(np.ptp(centres_m, axis=0) + 1e-6)
    axis.view_init(elev=25, azim=-60)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_translation_plot(centres_m: np.ndarray, output_path: Path) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    frame_indices = np.arange(len(centres_m))
    for axis, coordinate, label in zip(axes, range(3), ("X", "Y", "Z")):
        axis.plot(frame_indices, centres_m[:, coordinate], "o-", color="tab:blue")
        axis.set_ylabel(f"{label} [m]")
        axis.grid(alpha=0.3)
    axes[-1].set_xlabel("Frame index")
    figure.suptitle("VGGT RGB-D scaled camera-centre translation")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    config = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the BF16 fast path.")
    pairs = collect_pairs(config.session_dir, config.num_frames)
    images_cpu, metric_depth_m = load_inputs(pairs, config.width, config.height)

    model = VGGTOmega().eval().to("cuda")
    model.load_state_dict(load_checkpoint_state_dict(config.checkpoint))
    # Match the chunk-alignment path: on CUDA, native BF16 autocast is the
    # measured fast path for the 1B checkpoint while pose/depth outputs are
    # converted to float32 below before metric-scale estimation.
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        predictions = model(images_cpu.to("cuda"))
        pose_encoding = predictions["pose_enc"].float().cpu()
        raw_depth = predictions["depth"].float().cpu().numpy()[0, ..., 0]
        extrinsics, intrinsics = encoding_to_camera(pose_encoding, (config.height, config.width))

    scale, valid_pixels = estimate_scale(raw_depth, metric_depth_m, config.min_depth_m, config.max_depth_m)
    scaled_extrinsics = extrinsics.cpu().numpy()[0]
    scaled_extrinsics[:, :3, 3] *= scale
    centres_m = camera_centres(scaled_extrinsics)
    frame_stems = np.asarray([rgb.name.removesuffix(RGB_SUFFIX) for rgb, _ in pairs])
    camera_to_global = extrinsics_to_camera_to_global(scaled_extrinsics)
    scaled_depth_m = raw_depth * scale

    config.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        config.output_dir / "vggt_rgbd_pose_results.npz",
        raw_depth=raw_depth,
        scaled_depth_m=scaled_depth_m,
        metric_depth_m=metric_depth_m,
        extrinsics_camera_from_world=scaled_extrinsics,
        intrinsics=intrinsics.cpu().numpy()[0],
        camera_centres_m=centres_m,
        frame_stems=frame_stems,
        camera_to_global=camera_to_global,
    )
    save_pose_plot(centres_m, config.output_dir / "camera_trajectory_rgbd_scaled.png")
    save_translation_plot(centres_m, config.output_dir / "camera_translation_by_frame.png")
    summary = {
        "backend": "pytorch_native_bf16_autocast",
        "num_frames": config.num_frames,
        "input_size_wh": [config.width, config.height],
        "rgb_frames": [rgb.name for rgb, _ in pairs],
        "vggt_pose_convention": "camera_to_global",
        "rgbd_scale": scale,
        "scale_valid_pixels_after_trim": valid_pixels,
        "outputs": [
            "vggt_rgbd_pose_results.npz",
            "camera_trajectory_rgbd_scaled.png",
            "camera_translation_by_frame.png",
        ],
    }
    (config.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
