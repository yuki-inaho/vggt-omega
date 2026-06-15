"""Command-line utilities for VGGT-Omega development smoke checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from .colmap_export import (
    camera_from_intrinsics,
    collect_image_paths,
    export_scene_outputs,
    load_camera_from_dataset_info,
)
from .pipeline import VGGTOmegaPipeline
from .preprocess import load_images_from_paths, preprocess_images, read_images_from_video


def _collect_image_paths(args: argparse.Namespace) -> list[Path]:
    max_images = args.num_frames if args.num_frames > 0 else None
    return collect_image_paths(args.images, max_images=max_images)


def _collect_input_tensor(args: argparse.Namespace) -> torch.Tensor:
    if args.images:
        image_paths = _collect_image_paths(args)
        return load_images_from_paths(image_paths, image_resolution=args.image_resolution)

    if args.video:
        frames = read_images_from_video(args.video, sample_fps=args.sample_fps, max_frames=args.num_frames)
        return preprocess_images(frames, image_resolution=args.image_resolution)

    raise SystemExit("Provide either --images <dir> or --video <path>")


def _cmd_smoke(args: argparse.Namespace) -> int:
    pipeline = VGGTOmegaPipeline(
        checkpoint_path=args.checkpoint,
        device=args.device,
        enable_alignment=args.enable_alignment,
    )
    images = _collect_input_tensor(args)
    scene = pipeline.run(images).with_world_points()
    print(
        "smoke ok: "
        f"images={tuple(images.shape)} "
        f"depth={scene.depth.shape} "
        f"world_points={scene.world_points.shape if scene.world_points is not None else None}"
    )
    return 0


def _cmd_export_colmap(args: argparse.Namespace) -> int:
    image_paths = _collect_image_paths(args)
    pipeline = VGGTOmegaPipeline(
        checkpoint_path=args.checkpoint,
        device=args.device,
        enable_alignment=args.enable_alignment,
    )
    images = load_images_from_paths(image_paths, image_resolution=args.image_resolution)
    scene = pipeline.run(images).with_world_points()

    if args.dataset_info:
        camera = load_camera_from_dataset_info(args.dataset_info)
    else:
        first_intrinsic = scene.intrinsic[0]
        camera = camera_from_intrinsics(first_intrinsic, width=images.shape[-1], height=images.shape[-2])

    summary = export_scene_outputs(
        scene,
        image_paths,
        args.output,
        camera,
        copy_input_images=args.copy_images,
        run_settings={
            "checkpoint": str(args.checkpoint),
            "image_resolution": int(args.image_resolution),
            "dataset_info": str(args.dataset_info) if args.dataset_info else None,
            "device": str(args.device) if args.device else None,
            "enable_alignment": bool(args.enable_alignment),
            "copy_images": bool(args.copy_images),
        },
    )
    pose_summary = summary["pose_summary"]
    print(
        "export-colmap ok: "
        f"images={summary['num_images']} "
        f"sparse={summary['paths']['sparse_dir']} "
        f"npz={summary['paths']['predictions_npz']} "
        f"all_identity_pose={pose_summary['all_identity_pose']}"
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VGGT-Omega CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="Run a small inference smoke test")
    smoke.add_argument("--checkpoint", required=True, help="Path to a VGGT-Omega .pt checkpoint")
    src = smoke.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", help="Directory of input images")
    src.add_argument("--video", help="Path to an input video")
    smoke.add_argument("--num-frames", type=int, default=4, help="Limit number of frames")
    smoke.add_argument("--sample-fps", type=float, default=1.0, help="Sampling FPS for --video")
    smoke.add_argument("--image-resolution", type=int, default=512)
    smoke.add_argument("--device", help="Torch device, e.g. cuda, cuda:0, or cpu")
    smoke.add_argument("--enable-alignment", action="store_true", help="Enable the text-alignment head")
    smoke.set_defaults(func=_cmd_smoke)

    export = subparsers.add_parser("export-colmap", help="Run inference and export COLMAP text files")
    export.add_argument("--checkpoint", required=True, help="Path to a VGGT-Omega .pt checkpoint")
    export.add_argument("--images", required=True, help="Directory of input images")
    export.add_argument("--output", required=True, help="Output root for predictions.npz and sparse/0/*.txt")
    export.add_argument("--dataset-info", help="Optional dataset_info.json with original RGB camera intrinsics")
    export.add_argument("--num-frames", type=int, default=0, help="Limit number of frames; 0 means all")
    export.add_argument("--image-resolution", type=int, default=512)
    export.add_argument("--device", help="Torch device, e.g. cuda, cuda:0, or cpu")
    export.add_argument("--enable-alignment", action="store_true", help="Enable the text-alignment head")
    export.add_argument("--copy-images", action="store_true", help="Copy input images under <output>/images")
    export.set_defaults(func=_cmd_export_colmap)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
