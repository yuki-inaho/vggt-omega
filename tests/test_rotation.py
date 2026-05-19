from __future__ import annotations

import torch

from vggt_omega.utils.rotation import mat_to_quat, quat_to_mat, standardize_quaternion


def _random_rotations(num: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    raw = torch.randn(num, 3, 3, generator=g, dtype=torch.float64)
    u, _, vh = torch.linalg.svd(raw)
    det = torch.det(u @ vh)
    u[..., :, -1] *= det.unsqueeze(-1)
    return (u @ vh).float()


def test_quat_to_mat_roundtrip_preserves_rotation() -> None:
    rotations = _random_rotations(16)
    q = mat_to_quat(rotations)
    rec = quat_to_mat(q)
    assert torch.allclose(rec, rotations, atol=1e-5)


def test_quaternion_is_standardized() -> None:
    q = mat_to_quat(_random_rotations(8))
    assert torch.all(q[..., 3] >= 0)


def test_standardize_flips_negative_real_part() -> None:
    q = torch.tensor([[0.1, 0.2, 0.3, -0.4]])
    out = standardize_quaternion(q)
    assert out[0, 3] > 0
    assert torch.allclose(out, -q)


def test_identity_rotation_roundtrip() -> None:
    eye = torch.eye(3).unsqueeze(0)
    rec = quat_to_mat(mat_to_quat(eye))
    assert torch.allclose(rec, eye, atol=1e-6)
