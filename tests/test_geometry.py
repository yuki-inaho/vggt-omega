from __future__ import annotations

import numpy as np
import torch

from vggt_omega.utils.geometry import closed_form_inverse_se3


def _random_se3(batch: int = 4, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    raw = torch.randn(batch, 3, 3, generator=g, dtype=torch.float64)
    u, _, vh = torch.linalg.svd(raw)
    det = torch.det(u @ vh)
    u[..., :, -1] *= det.unsqueeze(-1)
    R = (u @ vh).float()
    T = torch.randn(batch, 3, generator=g).float()
    se3 = torch.eye(4).unsqueeze(0).repeat(batch, 1, 1)
    se3[:, :3, :3] = R
    se3[:, :3, 3] = T
    return se3


def test_inverse_se3_torch_is_actual_inverse() -> None:
    se3 = _random_se3()
    inv = closed_form_inverse_se3(se3)
    identity = torch.eye(4).unsqueeze(0).expand_as(se3)
    assert torch.allclose(se3 @ inv, identity, atol=1e-5)
    assert torch.allclose(inv @ se3, identity, atol=1e-5)


def test_inverse_se3_numpy_matches_torch() -> None:
    se3 = _random_se3(batch=2).numpy()
    inv = closed_form_inverse_se3(se3)
    eye = np.tile(np.eye(4), (len(se3), 1, 1))
    np.testing.assert_allclose(se3 @ inv, eye, atol=1e-4)


def test_invalid_shape_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        closed_form_inverse_se3(torch.zeros(2, 3, 3))
