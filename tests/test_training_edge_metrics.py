from __future__ import annotations

import pytest
import torch

from vggt_omega.training.edge_metrics import edge_3d_error_proxy


def _intrinsics(height: int, width: int, focal: float = 8.0) -> torch.Tensor:
    return torch.tensor(
        [[[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]]],
        dtype=torch.float32,
    )


def test_edge_proxy_perfect_plane_has_no_edges_and_zero_error() -> None:
    target = torch.ones(1, 5, 7)
    metrics = edge_3d_error_proxy(
        target,
        target,
        _intrinsics(5, 7),
        torch.ones_like(target, dtype=torch.bool),
        max_near_depth_m=1.2,
    )

    assert metrics["all_3d_error"] == pytest.approx(0)
    assert metrics["all_edge_pixels"] == 0
    assert metrics["all_edge_3d_error_proxy"] == pytest.approx(0)


def test_edge_proxy_perfect_step_is_zero_but_blurred_step_is_positive() -> None:
    target = torch.ones(1, 6, 8)
    target[..., 4:] = 2
    blurred = target.clone()
    blurred[..., 3] = 1.4
    blurred[..., 4] = 1.6
    mask = torch.ones_like(target, dtype=torch.bool)

    perfect = edge_3d_error_proxy(target, target, _intrinsics(6, 8), mask, max_near_depth_m=2.5)
    imperfect = edge_3d_error_proxy(blurred, target, _intrinsics(6, 8), mask, max_near_depth_m=2.5)

    assert perfect["all_edge_pixels"] > 0
    assert perfect["all_edge_3d_error_proxy"] == pytest.approx(0)
    assert imperfect["all_edge_3d_error_proxy"] > 0


def test_edge_proxy_invalid_depth_boundary_is_not_an_edge() -> None:
    target = torch.ones(1, 5, 7)
    target[..., :, 3:] = 2
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[..., :, 3:] = False

    metrics = edge_3d_error_proxy(target, target, _intrinsics(5, 7), mask, max_near_depth_m=2.5)

    assert metrics["all_edge_pixels"] == 0
    assert metrics["all_edge_coverage"] == pytest.approx(0)


def test_edge_proxy_separates_near_and_all_and_intrinsics_scale() -> None:
    target = torch.ones(1, 6, 8)
    target[..., 4:] = 2
    prediction = target + 0.1
    mask = torch.ones_like(target, dtype=torch.bool)

    wide = edge_3d_error_proxy(prediction, target, _intrinsics(6, 8, focal=4), mask, max_near_depth_m=1.2)
    narrow = edge_3d_error_proxy(prediction, target, _intrinsics(6, 8, focal=16), mask, max_near_depth_m=1.2)

    assert wide["all_valid_pixels"] == 48
    assert wide["near_valid_pixels"] == 24
    assert wide["all_edge_pixels"] > wide["near_edge_pixels"] > 0
    assert wide["all_3d_error"] > narrow["all_3d_error"]
    assert wide["all_non_edge_3d_error"] > 0
