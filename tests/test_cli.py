from __future__ import annotations

import numpy as np
import torch

import vggt_omega.cli as cli
from vggt_omega.pipeline import SceneResult


class _FakePipeline:
    def __init__(self, checkpoint_path, device=None, enable_alignment=False) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.enable_alignment = enable_alignment

    def run(self, images: torch.Tensor) -> SceneResult:
        n, _, h, w = images.shape
        return SceneResult(
            images=images,
            pose_enc=np.zeros((n, 9), dtype=np.float32),
            extrinsic=np.tile(np.eye(3, 4, dtype=np.float32)[None], (n, 1, 1)),
            intrinsic=np.tile(np.eye(3, dtype=np.float32)[None], (n, 1, 1)),
            depth=np.ones((n, h, w, 1), dtype=np.float32),
            depth_conf=np.ones((n, h, w), dtype=np.float32),
            world_points=np.zeros((n, h, w, 3), dtype=np.float32),
        )


def test_smoke_cli_runs_video_path_with_mocked_pipeline(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "VGGTOmegaPipeline", _FakePipeline)
    monkeypatch.setattr(cli, "read_images_from_video", lambda *args, **kwargs: [np.zeros((8, 8, 3), dtype=np.uint8)])
    monkeypatch.setattr(cli, "preprocess_images", lambda *args, **kwargs: torch.zeros(1, 3, 8, 8))

    code = cli.main(
        [
            "smoke",
            "--checkpoint",
            "dummy.pt",
            "--video",
            "input.mp4",
            "--device",
            "cpu",
            "--num-frames",
            "1",
        ]
    )

    assert code == 0
    assert "smoke ok" in capsys.readouterr().out
