from __future__ import annotations

import torch

from vggt_omega.utils.pose_enc import encoding_to_camera, extri_intri_to_pose_encoding


def _random_extrinsics(batch: int = 1, num_frames: int = 3) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    raw = torch.randn(batch, num_frames, 3, 3, generator=g, dtype=torch.float64)
    u, _, vh = torch.linalg.svd(raw)
    det = torch.det(u @ vh)
    u[..., :, -1] *= det.unsqueeze(-1)
    R = (u @ vh).float()
    T = torch.randn(batch, num_frames, 3, generator=g).float()
    return torch.cat([R, T.unsqueeze(-1)], dim=-1)


def _intrinsics(batch: int, num_frames: int, fx: float, fy: float, cx: float, cy: float) -> torch.Tensor:
    intr = torch.zeros(batch, num_frames, 3, 3)
    intr[..., 0, 0] = fx
    intr[..., 1, 1] = fy
    intr[..., 0, 2] = cx
    intr[..., 1, 2] = cy
    intr[..., 2, 2] = 1.0
    return intr


def test_extri_intri_pose_encoding_roundtrip() -> None:
    extr = _random_extrinsics(batch=1, num_frames=4)
    H, W = 384, 512
    intr = _intrinsics(1, 4, fx=480.0, fy=420.0, cx=W / 2, cy=H / 2)

    pose_enc = extri_intri_to_pose_encoding(extr, intr, (H, W))
    assert pose_enc.shape == (1, 4, 9)

    extr_rec, intr_rec = encoding_to_camera(pose_enc, (H, W))
    assert extr_rec.shape == extr.shape
    assert intr_rec.shape == intr.shape
    assert torch.allclose(extr_rec, extr, atol=1e-4)
    assert torch.allclose(intr_rec, intr, atol=1e-3)


def test_encoding_skip_intrinsics() -> None:
    extr = _random_extrinsics()
    intr = _intrinsics(1, 3, fx=500.0, fy=500.0, cx=256.0, cy=192.0)
    enc = extri_intri_to_pose_encoding(extr, intr, (384, 512))
    extr_rec, intr_rec = encoding_to_camera(enc, (384, 512), build_intrinsics=False)
    assert intr_rec is None
    assert torch.allclose(extr_rec, extr, atol=1e-4)
