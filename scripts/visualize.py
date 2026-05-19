"""CLI for visualizing VGGT-Omega inference output with Rerun.

Three modes are supported:

* ``viewer``     — spawn the local Rerun viewer (requires a display).
* ``rrd``        — write the recording to a ``.rrd`` file for later inspection.
* ``screenshot`` — start ``rerun --serve-web`` in a subprocess and use Playwright
  (Chromium) to grab a PNG of the running viewer. Useful from headless boxes.

Example::

    uv run python scripts/visualize.py \\
        --checkpoint checkpoints/vggt_omega_1b_512.pt \\
        --video examples/forest_road.mp4 --num-frames 6 \\
        --mode screenshot --output outputs/forest.png
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal

import torch

from vggt_omega.pipeline import VGGTOmegaPipeline
from vggt_omega.preprocess import load_images_from_paths, read_images_from_video
from vggt_omega.visualize import save_results_to_rrd, visualize_results

Mode = Literal["viewer", "rrd", "screenshot"]


# ---------------------------------------------------------------------------
# Inference helpers.
# ---------------------------------------------------------------------------


def _collect_input_tensor(args: argparse.Namespace) -> torch.Tensor:
    if args.images and args.video:
        raise SystemExit("--images and --video are mutually exclusive")
    if args.images:
        image_paths = sorted(Path(args.images).glob("*"))
        image_paths = [p for p in image_paths if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        if args.num_frames > 0:
            image_paths = image_paths[: args.num_frames]
        if not image_paths:
            raise SystemExit(f"No images found under {args.images}")
        return load_images_from_paths(image_paths, image_resolution=args.image_resolution)
    if args.video:
        frames = read_images_from_video(args.video, sample_fps=args.sample_fps)
        if args.num_frames > 0:
            frames = frames[: args.num_frames]
        # Reuse the path-based loader via the in-memory pipeline.
        from vggt_omega.preprocess import preprocess_images

        return preprocess_images(frames, image_resolution=args.image_resolution)
    raise SystemExit("Provide either --images <dir> or --video <path>")


def _run_inference(args: argparse.Namespace):
    pipeline = VGGTOmegaPipeline(checkpoint_path=args.checkpoint)
    images = _collect_input_tensor(args)
    return pipeline.run(images).with_world_points()


# ---------------------------------------------------------------------------
# Modes.
# ---------------------------------------------------------------------------


def _mode_viewer(args: argparse.Namespace) -> None:
    scene = _run_inference(args)
    visualize_results(
        scene,
        conf_percent=args.conf_percent,
        mask_black_bg=args.mask_black_bg,
        mask_white_bg=args.mask_white_bg,
        max_points=args.max_points,
        spawn=True,
    )


def _mode_rrd(args: argparse.Namespace) -> Path:
    scene = _run_inference(args)
    out = Path(args.output) if args.output else Path("outputs/scene.rrd")
    save_results_to_rrd(
        scene,
        out,
        conf_percent=args.conf_percent,
        mask_black_bg=args.mask_black_bg,
        mask_white_bg=args.mask_white_bg,
        max_points=args.max_points,
    )
    print(f"Wrote rerun recording to {out}")
    return out


def _mode_screenshot(args: argparse.Namespace) -> Path:
    rrd_path = _mode_rrd(args) if not args.rrd_input else Path(args.rrd_input)
    out_png = Path(args.output) if args.output else Path("outputs/scene.png")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    _capture_with_playwright(rrd_path, out_png, args)
    print(f"Wrote screenshot to {out_png}")
    return out_png


def _capture_with_playwright(rrd_path: Path, out_png: Path, args: argparse.Namespace) -> None:
    if shutil.which("rerun") is None:
        raise SystemExit("`rerun` CLI not on PATH; install rerun-sdk and ensure entrypoint is exposed")

    serve_cmd = [
        "rerun",
        "--serve-web",
        "--web-viewer-port",
        str(args.web_port),
        "--port",
        str(args.grpc_port),
        str(rrd_path),
    ]
    proc = subprocess.Popen(serve_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        time.sleep(args.serve_wait)
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch()
            try:
                context = browser.new_context(viewport={"width": args.width, "height": args.height})
                page = context.new_page()
                page.goto(f"http://localhost:{args.web_port}/?url=http://localhost:{args.grpc_port}", wait_until="load")
                page.wait_for_timeout(int(args.render_wait * 1000))
                page.screenshot(path=str(out_png), full_page=False)
                browser.close()
            finally:
                browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


# ---------------------------------------------------------------------------
# argparse plumbing.
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Rerun visualization for VGGT-Omega")
    p.add_argument("--checkpoint", required=False, help="Path to a VGGT-Omega .pt checkpoint")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--images", help="Directory of input images")
    src.add_argument("--video", help="Path to an input video")
    p.add_argument("--num-frames", type=int, default=0, help="Limit number of frames (0 = all)")
    p.add_argument("--sample-fps", type=float, default=1.0, help="Sampling FPS for --video")
    p.add_argument("--image-resolution", type=int, default=512)

    p.add_argument("--mode", choices=("viewer", "rrd", "screenshot"), default="rrd")
    p.add_argument("--output", help="Output path (.rrd for rrd, .png for screenshot)")
    p.add_argument("--rrd-input", help="Reuse an existing .rrd file (skip inference)")

    # Visualization filters.
    p.add_argument("--conf-percent", type=float, default=50.0)
    p.add_argument("--mask-black-bg", action="store_true")
    p.add_argument("--mask-white-bg", action="store_true")
    p.add_argument("--max-points", type=int, default=1_000_000)

    # Screenshot-only settings.
    p.add_argument("--width", type=int, default=1600)
    p.add_argument("--height", type=int, default=900)
    p.add_argument("--web-port", type=int, default=9090)
    p.add_argument("--grpc-port", type=int, default=9876)
    p.add_argument("--serve-wait", type=float, default=2.0, help="Seconds to wait for `rerun --serve-web` to come up")
    p.add_argument("--render-wait", type=float, default=4.0, help="Seconds to wait for the viewer to render")

    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.rrd_input and args.checkpoint is None:
        raise SystemExit("--checkpoint is required unless --rrd-input is provided")

    if args.mode == "viewer":
        _mode_viewer(args)
    elif args.mode == "rrd":
        _mode_rrd(args)
    elif args.mode == "screenshot":
        _mode_screenshot(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
