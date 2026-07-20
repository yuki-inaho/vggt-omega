#!/usr/bin/env python3
"""Export RGB-D scale-corrected VGGT poses for every boundary-window dataset.

Each output NPZ contains ``frame_stems`` and homogeneous ``camera_to_global``
poses, the convention consumed by ViGG's RGB-D registration script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from run_vggt_rgbd_pose_workflow import (
    DEPTH_SUFFIX,
    RGB_SUFFIX,
    camera_centres,
    estimate_scale,
    extrinsics_to_camera_to_global,
    load_inputs,
)
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_checkpoint_state_dict
from vggt_omega.utils.pose_enc import encoding_to_camera


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--min-depth-m", type=float, default=0.10)
    parser.add_argument("--max-depth-m", type=float, default=5.00)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def pairs_for_window(window: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for rgb in sorted((window / "rgb").glob(f"*{RGB_SUFFIX}")):
        depth = window / "mapped_depth_dense" / f"{rgb.name.removesuffix(RGB_SUFFIX)}{DEPTH_SUFFIX}"
        if not depth.is_file():
            raise FileNotFoundError(f"Missing dense aligned depth: {depth}")
        pairs.append((rgb, depth))
    if len(pairs) < 2:
        raise RuntimeError(f"Need at least two RGB-D frames in {window}")
    return pairs


def append_status(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for VGGT pose export")
    if args.width % 16 or args.height % 16:
        raise ValueError("--width and --height must be divisible by 16")
    windows = sorted(path for path in args.windows_root.glob("boundary_*") if path.is_dir())
    if args.max_windows is not None:
        windows = windows[: args.max_windows]
    if not windows:
        raise FileNotFoundError(f"No boundary_* directories in {args.windows_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    status_path = args.output_root / "vggt_export_status.jsonl"
    model = VGGTOmega().eval().to("cuda")
    model.load_state_dict(load_checkpoint_state_dict(args.checkpoint))
    completed = 0
    failed = 0
    for index, window in enumerate(windows, 1):
        output_dir = args.output_root / window.name
        output_npz = output_dir / "vggt_pose_for_vigg.npz"
        if args.resume and output_npz.is_file():
            print(f"[{index}/{len(windows)}] skip {window.name}", flush=True)
            completed += 1
            continue
        try:
            pairs = pairs_for_window(window)
            images_cpu, metric_depth_m = load_inputs(pairs, args.width, args.height)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                predictions = model(images_cpu.to("cuda"))
                pose_encoding = predictions["pose_enc"].float().cpu()
                raw_depth = predictions["depth"].float().cpu().numpy()[0, ..., 0]
                extrinsics, intrinsics = encoding_to_camera(pose_encoding, (args.height, args.width))
            scale, valid_pixels = estimate_scale(raw_depth, metric_depth_m, args.min_depth_m, args.max_depth_m)
            scaled_extrinsics = extrinsics.cpu().numpy()[0]
            scaled_extrinsics[:, :3, 3] *= scale
            centres = camera_centres(scaled_extrinsics)
            frame_stems = np.asarray([rgb.name.removesuffix(RGB_SUFFIX) for rgb, _ in pairs])
            output_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output_npz,
                frame_stems=frame_stems,
                camera_to_global=extrinsics_to_camera_to_global(scaled_extrinsics),
                extrinsics_camera_from_world=scaled_extrinsics,
                intrinsics=intrinsics.cpu().numpy()[0],
                camera_centres_m=centres,
                raw_depth=raw_depth,
                scaled_depth_m=raw_depth * scale,
                metric_depth_m=metric_depth_m,
            )
            summary = {
                "window": window.name,
                "num_frames": len(pairs),
                "frame_stems": frame_stems.tolist(),
                "vigg_pose_convention": "camera_to_global",
                "rgbd_scale": scale,
                "scale_valid_pixels_after_trim": valid_pixels,
                "output": str(output_npz),
            }
            (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            append_status(status_path, {"window": window.name, "status": "ok", **summary})
            completed += 1
            print(f"[{index}/{len(windows)}] ok {window.name} frames={len(pairs)} scale={scale:.5f}", flush=True)
        except Exception as error:
            failed += 1
            append_status(status_path, {"window": window.name, "status": "error", "error": repr(error)})
            print(f"[{index}/{len(windows)}] ERROR {window.name}: {error!r}", flush=True)
        finally:
            torch.cuda.empty_cache()
    completion = {"windows_requested": len(windows), "completed": completed, "failed": failed}
    (args.output_root / "vggt_export_complete.json").write_text(json.dumps(completion, indent=2), encoding="utf-8")
    print(json.dumps(completion), flush=True)


if __name__ == "__main__":
    main()
