from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from vggt_omega.pipeline import SceneResult, VGGTOmegaPipeline, _predictions_to_scene_result, autodetect_device


def _fake_predictions(num_frames: int = 2, h: int = 16, w: int = 24) -> dict:
    g = torch.Generator().manual_seed(0)
    images = torch.rand(1, num_frames, 3, h, w, generator=g)
    raw_rot = torch.randn(1, num_frames, 3, 3, generator=g, dtype=torch.float64)
    u, _, vh = torch.linalg.svd(raw_rot)
    det = torch.det(u @ vh)
    u[..., :, -1] *= det.unsqueeze(-1)
    R = (u @ vh).float()
    quat = _mat_to_quat(R)
    fov = torch.full((1, num_frames, 2), 1.0)
    pose_enc = torch.cat([torch.zeros(1, num_frames, 3), quat, fov], dim=-1)
    depth = torch.rand(1, num_frames, h, w, 1, generator=g) + 0.5
    depth_conf = torch.rand(1, num_frames, h, w, generator=g) + 1.0
    cam_reg = torch.rand(1, num_frames, 17, 32, generator=g)
    return {
        "images": images,
        "pose_enc": pose_enc,
        "depth": depth,
        "depth_conf": depth_conf,
        "camera_and_register_tokens": cam_reg,
    }


def _mat_to_quat(R: torch.Tensor) -> torch.Tensor:
    from vggt_omega.utils.rotation import mat_to_quat

    return mat_to_quat(R)


def test_predictions_to_scene_result_shapes() -> None:
    preds = _fake_predictions(num_frames=3, h=16, w=24)
    scene = _predictions_to_scene_result(preds)
    assert isinstance(scene, SceneResult)
    assert scene.pose_enc.shape == (3, 9)
    assert scene.extrinsic.shape == (3, 3, 4)
    assert scene.intrinsic.shape == (3, 3, 3)
    assert scene.depth.shape == (3, 16, 24, 1)
    assert scene.depth_conf.shape == (3, 16, 24)
    assert scene.camera_tokens is not None
    assert scene.register_tokens is not None
    assert scene.camera_tokens.shape == (3, 1, 32)
    assert scene.register_tokens.shape == (3, 16, 32)


def test_with_world_points_lazy_fills() -> None:
    preds = _fake_predictions()
    scene = _predictions_to_scene_result(preds)
    assert scene.world_points is None
    scene.with_world_points()
    assert scene.world_points is not None
    assert scene.world_points.shape[-1] == 3


def test_as_npz_dict_contains_world_points() -> None:
    preds = _fake_predictions()
    scene = _predictions_to_scene_result(preds)
    out = scene.as_npz_dict()
    assert "world_points_from_depth" in out
    assert out["depth"].shape == scene.depth.shape
    assert isinstance(out["images"], np.ndarray)


def test_autodetect_device_returns_torch_device() -> None:
    dev = autodetect_device()
    assert isinstance(dev, torch.device)
    assert dev.type in {"cuda", "cpu"}


class _TinyDenseHead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trunk = torch.nn.Linear(2, 2)
        self.proj = torch.nn.Linear(2, 1)
        self.proj_conf = torch.nn.Linear(2, 1)


class _TinyOmega(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.aggregator = torch.nn.Linear(2, 2)
        self.camera_head = torch.nn.Linear(2, 2)
        self.dense_head = _TinyDenseHead()


def _head_payload(model: torch.nn.Module, base_path: Path) -> dict:
    names = {
        name
        for name, _ in model.named_parameters()
        if (name.startswith("camera_head.") or name.startswith("dense_head."))
        and not name.startswith("dense_head.proj_conf.")
    }
    state = {name: torch.full_like(model.state_dict()[name], 7) for name in names}
    return {
        "format_version": 1,
        "kind": "best",
        "parameter_state": "x",
        "model_state": state,
        "metadata": {
            "base_checkpoint": {
                "size_bytes": base_path.stat().st_size,
                "sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
            }
        },
        "config": {"model": {"image_height": 384, "image_width": 512}},
    }


def test_pipeline_strictly_overlays_head_and_reads_training_shape(tmp_path: Path) -> None:
    model = _TinyOmega()
    original_aggregator = model.aggregator.weight.detach().clone()
    original_confidence = model.dense_head.proj_conf.weight.detach().clone()
    base_path = tmp_path / "base.pt"
    base_path.write_bytes(b"released base")
    head_path = tmp_path / "best.pt"
    torch.save(_head_payload(model, base_path), head_path)

    shape = VGGTOmegaPipeline._apply_head_checkpoint(model, head_path, base_path)

    assert shape == (384, 512)
    assert torch.equal(model.aggregator.weight, original_aggregator)
    assert torch.equal(model.dense_head.proj_conf.weight, original_confidence)
    assert torch.equal(model.camera_head.weight, torch.full_like(model.camera_head.weight, 7))
    assert torch.equal(model.dense_head.proj.weight, torch.full_like(model.dense_head.proj.weight, 7))


def test_pipeline_rejects_incomplete_or_wrong_base_head(tmp_path: Path) -> None:
    model = _TinyOmega()
    base_path = tmp_path / "base.pt"
    base_path.write_bytes(b"released base")
    payload = _head_payload(model, base_path)
    payload["model_state"].pop("camera_head.weight")
    head_path = tmp_path / "best.pt"
    torch.save(payload, head_path)

    with pytest.raises(ValueError, match="exactly match"):
        VGGTOmegaPipeline._apply_head_checkpoint(model, head_path, base_path)

    payload = _head_payload(model, base_path)
    payload["metadata"]["base_checkpoint"]["sha256"] = "0" * 64
    torch.save(payload, head_path)
    with pytest.raises(ValueError, match="different base"):
        VGGTOmegaPipeline._apply_head_checkpoint(model, head_path, base_path)


@pytest.mark.gpu
def test_pipeline_run_smoke() -> None:
    from vggt_omega.pipeline import VGGTOmegaPipeline
    from vggt_omega.preprocess import preprocess_images, read_images_from_video

    ckpt = Path("checkpoints/vggt_omega_1b_512.pt")
    video = Path("examples/forest_road.mp4")
    if not ckpt.is_file() or not video.is_file() or not torch.cuda.is_available():
        pytest.skip("smoke test requires GPU, 512 checkpoint, and example video")

    pipe = VGGTOmegaPipeline(ckpt)
    frames = read_images_from_video(video, sample_fps=1.0, max_frames=2)
    images = preprocess_images(frames, image_resolution=512)
    scene = pipe.run(images).with_world_points()
    assert scene.depth.shape[0] == 2
    assert scene.world_points is not None
    assert scene.world_points.shape[-1] == 3
