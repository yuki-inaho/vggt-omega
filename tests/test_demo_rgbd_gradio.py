from __future__ import annotations

from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

import demo_rgbd_gradio


def _dataset(root: Path) -> None:
    scene = root / "scenes" / "scene_000000"
    for index in range(2):
        name = f"frame_{index:06d}.png"
        rgb = np.full((8, 12, 3), 40 + index * 20, dtype=np.uint8)
        depth = np.full((8, 12), 300 + index * 100, dtype=np.uint16)
        mask = np.full((8, 12), 255, dtype=np.uint8)
        for directory, array in (("rgb", rgb), ("depth", depth), ("valid_mask", mask)):
            path = scene / directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(array).save(path)


def test_callbacks_index_and_render_selected_rgb_frames(tmp_path: Path) -> None:
    _dataset(tmp_path)
    config = demo_rgbd_gradio.loader_config("rgb", "depth", "valid_mask", "", "", "", 0.001, False, True)

    choices, status = demo_rgbd_gradio.list_frame_choices(tmp_path, config)
    gallery, rows, render_status = demo_rgbd_gradio.visualize_selection(
        tmp_path,
        choices,
        None,
        config,
        depth_ceiling_m=1.0,
        auto_depth_ceiling=False,
        overlay_alpha=0.4,
    )

    assert len(choices) == 2
    assert "2 RGB-D pairs" in status
    assert len(gallery) == 8
    assert len(rows) == 2
    assert "2 frame(s)" in render_status
    assert "1.000 m" in render_status


def test_uploaded_rgb_is_resolved_without_ui_specific_loader_logic(tmp_path: Path) -> None:
    _dataset(tmp_path)
    upload = tmp_path / "uploads" / "frame_000001.png"
    upload.parent.mkdir()
    upload.write_bytes((tmp_path / "scenes/scene_000000/rgb/frame_000001.png").read_bytes())
    config = demo_rgbd_gradio.loader_config("rgb", "depth", "valid_mask", "", "", "", 0.001, False, True)

    gallery, rows, _ = demo_rgbd_gradio.visualize_selection(
        tmp_path,
        [],
        [upload],
        config,
        depth_ceiling_m=1.0,
        auto_depth_ceiling=False,
        overlay_alpha=0.4,
    )

    assert len(gallery) == 4
    assert isinstance(rows[0][0], str)
    assert rows[0][0].endswith("frame_000001.png")


def test_uploaded_rgb_takes_precedence_over_stale_dropdown_selection(tmp_path: Path) -> None:
    _dataset(tmp_path)
    upload = tmp_path / "uploads" / "frame_000001.png"
    upload.parent.mkdir()
    upload.write_bytes((tmp_path / "scenes/scene_000000/rgb/frame_000001.png").read_bytes())
    config = demo_rgbd_gradio.loader_config("rgb", "depth", "valid_mask", "", "", "", 0.001, False, True)

    gallery, rows, status = demo_rgbd_gradio.visualize_selection(
        tmp_path,
        ["scenes/scene_000000/rgb/frame_000000.png"],
        [upload],
        config,
        depth_ceiling_m=1.0,
        auto_depth_ceiling=False,
        overlay_alpha=0.4,
    )

    assert len(gallery) == 4
    assert len(rows) == 1
    assert rows[0][0] == "scenes/scene_000000/rgb/frame_000001.png"
    assert "uploaded RGB" in status


def test_cli_and_ui_build_without_a_model_or_cuda(tmp_path: Path) -> None:
    _dataset(tmp_path)
    args = demo_rgbd_gradio.parse_args(["--dataset-root", str(tmp_path), "--server-port", "9876"])

    demo = demo_rgbd_gradio.build_ui(args.dataset_root, demo_rgbd_gradio.config_from_args(args))

    assert args.server_port == 9876
    assert isinstance(demo, gr.Blocks)
    config = demo.get_config_file()
    labels = {component["props"].get("label") for component in config["components"]}
    assert {"RGB frames", "RGB uploads", "RGB-D views", "Frame statistics"} <= labels
    api_names = {dependency.get("api_name") for dependency in config["dependencies"]}
    assert "visualize_uploaded_rgb" in api_names
