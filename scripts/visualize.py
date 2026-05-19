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
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal
from urllib import request
from urllib.parse import quote

import torch

from vggt_omega.pipeline import VGGTOmegaPipeline
from vggt_omega.preprocess import load_images_from_paths, preprocess_images, read_images_from_video
from vggt_omega.visualize import save_results_to_rrd, visualize_results

Mode = Literal["viewer", "rrd", "screenshot"]
CHROMIUM_GL_ARGS = [
    "--use-gl=angle",
    "--use-angle=gl",
    "--ignore-gpu-blocklist",
    "--enable-gpu-rasterization",
]


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
        frames = read_images_from_video(args.video, sample_fps=args.sample_fps, max_frames=args.num_frames)
        return preprocess_images(frames, image_resolution=args.image_resolution)
    raise SystemExit("Provide either --images <dir> or --video <path>")


def _run_inference(args: argparse.Namespace):
    pipeline = VGGTOmegaPipeline(
        checkpoint_path=args.checkpoint,
        device=args.device,
        enable_alignment=args.enable_alignment,
    )
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
        filter_depth_edges=args.filter_depth_edges,
        depth_edge_rtol=args.depth_edge_rtol,
        max_points=args.max_points,
        spawn=True,
    )


def _mode_rrd(args: argparse.Namespace, output: Path | None = None) -> Path:
    scene = _run_inference(args)
    out = output or (Path(args.output) if args.output else Path("outputs/scene.rrd"))
    save_results_to_rrd(
        scene,
        out,
        conf_percent=args.conf_percent,
        mask_black_bg=args.mask_black_bg,
        mask_white_bg=args.mask_white_bg,
        filter_depth_edges=args.filter_depth_edges,
        depth_edge_rtol=args.depth_edge_rtol,
        max_points=args.max_points,
    )
    print(f"Wrote rerun recording to {out}")
    return out


def _mode_screenshot(args: argparse.Namespace) -> Path:
    out_png = Path(args.output) if args.output else Path("outputs/scene.png")
    rrd_path = (
        Path(args.rrd_input)
        if args.rrd_input
        else _mode_rrd(args, output=Path(args.rrd_output) if args.rrd_output else out_png.with_suffix(".rrd"))
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    _capture_with_playwright(rrd_path, out_png, args)
    print(f"Wrote screenshot to {out_png}")
    return out_png


def _capture_with_playwright(rrd_path: Path, out_png: Path, args: argparse.Namespace) -> None:
    if not rrd_path.is_file():
        raise SystemExit(f"RRD input not found: {rrd_path}")
    if shutil.which("rerun") is None:
        raise SystemExit("`rerun` CLI not on PATH; install rerun-sdk and ensure entrypoint is exposed")

    web_port, grpc_port = _resolve_ports(args.web_port, args.grpc_port, strict=args.strict_ports)
    serve_cmd = [
        "rerun",
        "--serve-web",
        "--web-viewer-port",
        str(web_port),
        "--port",
        str(grpc_port),
        str(rrd_path),
    ]
    proc = subprocess.Popen(serve_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        _wait_for_http(f"http://127.0.0.1:{web_port}", timeout_s=args.serve_wait)
        from playwright.sync_api import sync_playwright

        recording_url = quote(f"rerun+http://127.0.0.1:{grpc_port}/proxy", safe="")
        renderer_query = f"&renderer={args.renderer}" if args.renderer else ""
        with sync_playwright() as p:
            browser = p.chromium.launch(args=CHROMIUM_GL_ARGS)
            context = browser.new_context(viewport={"width": args.width, "height": args.height})
            try:
                page = context.new_page()
                page.goto(f"http://127.0.0.1:{web_port}/?url={recording_url}{renderer_query}", wait_until="load")
                page.wait_for_timeout(int(args.render_wait * 1000))
                _raise_if_viewer_error(page)
                page.screenshot(path=str(out_png), full_page=False)
            finally:
                context.close()
                browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def _raise_if_viewer_error(page) -> None:
    body_text = page.locator("body").inner_text(timeout=1000)
    if "An error occurred during loading" in body_text:
        detail = " ".join(line.strip() for line in body_text.splitlines() if line.strip())
        raise SystemExit(f"Rerun web viewer failed to load the recording: {detail[:240]}")


def _resolve_ports(web_port: int, grpc_port: int, *, strict: bool) -> tuple[int, int]:
    if strict:
        _assert_port_available(web_port)
        _assert_port_available(grpc_port)
        return web_port, grpc_port

    resolved_web = _find_available_port(web_port)
    resolved_grpc = _find_available_port(grpc_port if grpc_port != resolved_web else resolved_web + 1)
    if (resolved_web, resolved_grpc) != (web_port, grpc_port):
        print(
            f"Requested Rerun ports {web_port}/{grpc_port} are busy; "
            f"using {resolved_web}/{resolved_grpc} for this screenshot."
        )
    return resolved_web, resolved_grpc


def _find_available_port(start_port: int) -> int:
    port = start_port
    while port < 65535:
        if _port_is_available(port):
            return port
        port += 1
    raise SystemExit(f"No available TCP port found at or above {start_port}")


def _assert_port_available(port: int) -> None:
    if not _port_is_available(port):
        raise SystemExit(f"TCP port {port} is already in use")


def _port_is_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _wait_for_http(url: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with request.urlopen(url, timeout=0.5) as response:
                if response.status < 500:
                    return
        except Exception as exc:
            last_error = exc
        time.sleep(0.1)
    raise SystemExit(f"Timed out waiting for Rerun web viewer at {url}: {last_error}")


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
    p.add_argument("--device", help="Torch device, e.g. cuda, cuda:0, or cpu")
    p.add_argument("--enable-alignment", action="store_true", help="Enable the text-alignment head")

    p.add_argument("--mode", choices=("viewer", "rrd", "screenshot"), default="rrd")
    p.add_argument("--output", help="Output path (.rrd for rrd, .png for screenshot)")
    p.add_argument("--rrd-input", help="Screenshot mode only: reuse an existing .rrd file and skip inference")
    p.add_argument("--rrd-output", help="Screenshot mode only: where to save the intermediate .rrd")

    # Visualization filters.
    p.add_argument("--conf-percent", type=float, default=50.0)
    p.add_argument("--mask-black-bg", action="store_true")
    p.add_argument("--mask-white-bg", action="store_true")
    p.add_argument("--filter-depth-edges", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--depth-edge-rtol", type=float, default=0.03)
    p.add_argument("--max-points", type=int, default=1_000_000)

    # Screenshot-only settings.
    p.add_argument("--width", type=int, default=1600)
    p.add_argument("--height", type=int, default=900)
    p.add_argument("--renderer", choices=("webgl", "webgpu"), default="webgl")
    p.add_argument("--web-port", type=int, default=9090)
    p.add_argument("--grpc-port", type=int, default=9876)
    p.add_argument("--strict-ports", action="store_true", help="Fail instead of choosing nearby free ports")
    p.add_argument("--serve-wait", type=float, default=2.0, help="Seconds to wait for `rerun --serve-web` to come up")
    p.add_argument("--render-wait", type=float, default=4.0, help="Seconds to wait for the viewer to render")

    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.rrd_input and args.mode != "screenshot":
        raise SystemExit("--rrd-input is only valid with --mode screenshot")
    if args.rrd_output and args.mode != "screenshot":
        raise SystemExit("--rrd-output is only valid with --mode screenshot")
    if not args.rrd_input:
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required unless screenshot mode uses --rrd-input")
        if not args.images and not args.video:
            raise SystemExit("Provide either --images <dir> or --video <path>")

    if args.mode == "viewer":
        _mode_viewer(args)
    elif args.mode == "rrd":
        _mode_rrd(args)
    elif args.mode == "screenshot":
        _mode_screenshot(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
