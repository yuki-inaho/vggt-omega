from __future__ import annotations

import numpy as np
import torch
from torch import nn

import vggt_omega.inference as inference_mod
from vggt_omega.pipeline import SceneResult


def _fake_scene(images: torch.Tensor) -> SceneResult:
    n, _, h, w = images.shape
    return SceneResult(
        images=images,
        pose_enc=np.zeros((n, 9), dtype=np.float32),
        extrinsic=np.tile(np.eye(3, 4, dtype=np.float32)[None], (n, 1, 1)),
        intrinsic=np.tile(
            np.array([[w * 1.0, 0, w / 2], [0, h * 1.0, h / 2], [0, 0, 1]], dtype=np.float32)[None],
            (n, 1, 1),
        ),
        depth=np.ones((n, h, w, 1), dtype=np.float32),
        depth_conf=np.ones((n, h, w), dtype=np.float32),
        world_points=np.zeros((n, h, w, 3), dtype=np.float32),
    )


def test_inference_wrapper_registers_underlying_model(monkeypatch) -> None:
    class FakePipeline:
        def __init__(self, checkpoint_path, device, enable_alignment=False) -> None:
            self.model = nn.Linear(1, 1)
            self.device = torch.device(device)

        def run(self, images: torch.Tensor) -> SceneResult:
            return _fake_scene(images)

    monkeypatch.setattr(inference_mod, "VGGTOmegaPipeline", FakePipeline)
    monkeypatch.setattr(inference_mod, "preprocess_images", lambda *args, **kwargs: torch.zeros(2, 3, 8, 8))

    model = inference_mod.VGGTOmegaInference(checkpoint_path="dummy.pt", device="cpu")
    assert sum(param.numel() for param in model.parameters()) > 0
    assert next(iter(model.state_dict())).startswith("model.")

    results = model([np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((8, 8, 3), dtype=np.uint8)])
    assert len(results) == 2
    assert results[0].point_map_by_unprojection.shape == (8, 8, 3)

    model.to("cpu")
    assert model.pipeline.device.type == "cpu"
