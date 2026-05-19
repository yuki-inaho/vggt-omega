"""Command-line utilities for VGGT-Omega development smoke checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from .pipeline import VGGTOmegaPipeline
from .preprocess import load_images_from_paths, preprocess_images, read_images_from_video


def _collect_input_tensor(args: argparse.Namespace) -> torch.Tensor:
    if args.images:
        image_paths = sorted(Path(args.images).glob("*"))
        image_paths = [p for p in image_paths if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        if args.num_frames > 0:
            image_paths = image_paths[: args.num_frames]
        if not image_paths:
            raise SystemExit(f"No images found under {args.images}")
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
