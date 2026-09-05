from __future__ import annotations

from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

import demo_rgbd_gradio
from vggt_omega.omnivggt_inference import OmniVggtInferenceResult


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
    assert {
        "RGB frames",
        "RGB uploads",
        "RGB-D views",
        "Frame statistics",
        "OmniVGGT predictions",
        "OmniVGGT frame comparison",
        "Predicted cameras",
        "OmniVGGT 3D reconstruction",
        "Download OmniVGGT GLB",
    } <= labels
    api_names = {dependency.get("api_name") for dependency in config["dependencies"]}
    assert "visualize_uploaded_rgb" in api_names
    assert "run_omnivggt" in api_names


def test_omnivggt_callback_resolves_rgbd_and_returns_model_outputs(tmp_path: Path, monkeypatch) -> None:
    _dataset(tmp_path)
    output_glb = tmp_path / "result.glb"
    output_glb.write_bytes(b"glTF")
    captured = {}

    def fake_runtime(repository: str, checkpoint: str, device: str):
        captured["runtime"] = (repository, checkpoint, device)
        return object(), object()

    def fake_infer(model, pose_decoder, prepared, **kwargs):
        captured["prepared"] = prepared
        captured["kwargs"] = kwargs
        return OmniVggtInferenceResult(
            gallery=((Image.new("RGB", (4, 4)), "predicted depth"),),
            frame_statistics=((prepared.frame_ids[0], 100.0, 0.3, 0.4, 0.75, 0.01, 2.0),),
            camera_statistics=((prepared.frame_ids[0], 0.0, 0.0, 0.0, 100.0, 101.0, 6.0, 4.0),),
            glb_path=output_glb,
            exported_points=96,
            inference_seconds=1.25,
        )

    monkeypatch.setattr(demo_rgbd_gradio, "_get_omnivggt_runtime", fake_runtime)
    monkeypatch.setattr(demo_rgbd_gradio, "infer_and_render", fake_infer)
    config = demo_rgbd_gradio.loader_config("rgb", "depth", "valid_mask", "", "", "", 0.001, False, True)

    gallery, frame_rows, camera_rows, model_path, download_path, status = demo_rgbd_gradio.run_omnivggt_selection(
        tmp_path,
        ["scenes/scene_000000/rgb/frame_000000.png"],
        None,
        config,
        official_repository="/workspace/external/OmniVGGT-official",
        checkpoint="/workspace/models/OmniVGGT/OmniVGGT.safetensors",
        device="cpu",
        target_size=28,
        confidence_percentile=25,
        max_points=1000,
    )

    assert len(gallery) == 1
    assert str(frame_rows[0][0]).endswith("frame_000000.png")
    assert camera_rows[0][4:6] == [100.0, 101.0]
    assert model_path == download_path == str(output_glb)
    assert "1.250 s" in status
    assert "96" in status
    assert captured["runtime"][2] == "cpu"
    assert captured["prepared"].images.shape == (1, 3, 14, 28)
