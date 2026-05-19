from __future__ import annotations

import numpy as np
import pytest
import torch

from vggt_omega.pipeline import SceneResult, _predictions_to_scene_result, autodetect_device


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


@pytest.mark.gpu
def test_pipeline_run_smoke() -> None:
    from pathlib import Path

    from vggt_omega.pipeline import VGGTOmegaPipeline
    from vggt_omega.utils.load_fn import load_and_preprocess_images

    ckpt = Path("checkpoints/vggt_omega_1b_512.pt")
    if not ckpt.is_file() or not torch.cuda.is_available():
        pytest.skip("smoke test requires GPU and 512 checkpoint")

    pipe = VGGTOmegaPipeline(ckpt)
    images = load_and_preprocess_images(
        sorted(str(p) for p in Path("/tmp/vggt_cli_test/images").glob("*.png"))[:2],
        image_resolution=512,
    )
    scene = pipe.run(images).with_world_points()
    assert scene.depth.shape[0] == 2
    assert scene.world_points.shape[-1] == 3
