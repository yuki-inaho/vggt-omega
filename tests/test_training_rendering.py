from __future__ import annotations

import pytest
import torch

from vggt_omega.training.losses import build_camera_pose_target
from vggt_omega.training.rendering import (
    compute_sequence_photometric_loss,
    masked_photometric_l1,
    soft_zbuffer_reproject,
)


def _camera(height: int, width: int, *, tx: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    intrinsics = torch.tensor([[[4.0, 0.0, width / 2], [0.0, 4.0, height / 2], [0.0, 0.0, 1.0]]], dtype=torch.float32)
    extrinsics = torch.eye(4, dtype=torch.float32)[None, :3].clone()
    extrinsics[:, 0, 3] = tx
    return intrinsics, extrinsics


def test_soft_reprojection_identity_reconstructs_plane() -> None:
    height, width = 4, 5
    rgb = torch.linspace(0, 1, height * width).reshape(1, 1, height, width).repeat(1, 3, 1, 1)
    depth = torch.ones((1, height, width), dtype=torch.float32)
    intrinsics, extrinsics = _camera(height, width)

    result = soft_zbuffer_reproject(rgb, depth, intrinsics, intrinsics, extrinsics, extrinsics)

    assert result["visibility"].all()
    assert torch.allclose(result["rgb"], rgb, atol=1e-6)
    assert masked_photometric_l1(result["rgb"], rgb, result["visibility"]).item() == pytest.approx(0.0)


def test_soft_reprojection_known_translation_moves_source_pixel() -> None:
    height, width = 5, 7
    rgb = torch.zeros((1, 3, height, width), dtype=torch.float32)
    rgb[:, :, 2, 3] = 1.0
    depth = torch.ones((1, height, width), dtype=torch.float32)
    intrinsics, source_pose = _camera(height, width)
    _, target_pose = _camera(height, width, tx=0.25)

    result = soft_zbuffer_reproject(rgb, depth, intrinsics, intrinsics, source_pose, target_pose)

    assert torch.allclose(result["rgb"][0, :, 2, 4], torch.ones(3), atol=1e-5)


def test_soft_reprojection_target_depth_rejects_occluded_points() -> None:
    height, width = 4, 5
    rgb = torch.ones((1, 3, height, width), dtype=torch.float32)
    depth = torch.ones((1, height, width), dtype=torch.float32)
    target_depth = torch.full_like(depth, 0.5)
    intrinsics, extrinsics = _camera(height, width)

    result = soft_zbuffer_reproject(
        rgb,
        depth,
        intrinsics,
        intrinsics,
        extrinsics,
        extrinsics,
        target_depth=target_depth,
        relative_depth_tolerance=0.01,
    )

    assert not result["visibility"].any()
    assert torch.count_nonzero(result["rgb"]) == 0


def test_soft_reprojection_near_and_static_masks_limit_visibility() -> None:
    rgb = torch.ones((1, 3, 2, 2), dtype=torch.float32)
    depth = torch.tensor([[[1.0, 1.3], [1.1, 1.0]]], dtype=torch.float32)
    static_mask = torch.tensor([[[True, True], [False, True]]])
    intrinsics, extrinsics = _camera(2, 2)

    result = soft_zbuffer_reproject(
        rgb,
        depth,
        intrinsics,
        intrinsics,
        extrinsics,
        extrinsics,
        source_mask=static_mask,
        max_depth_m=1.2,
    )

    assert result["visibility"].sum().item() == 2


def test_soft_reprojection_depth_gradient_is_finite_and_nonzero() -> None:
    height, width = 5, 7
    rgb = torch.linspace(0, 1, width).reshape(1, 1, 1, width).repeat(1, 3, height, 1)
    depth = torch.ones((1, height, width), dtype=torch.float32, requires_grad=True)
    intrinsics, source_pose = _camera(height, width)
    _, target_pose = _camera(height, width, tx=0.1)

    result = soft_zbuffer_reproject(rgb, depth, intrinsics, intrinsics, source_pose, target_pose)
    loss = result["rgb"].square().mean()
    loss.backward()

    assert depth.grad is not None
    assert torch.isfinite(depth.grad).all()
    assert torch.count_nonzero(depth.grad).item() > 0


def test_empty_visibility_returns_graph_connected_zero_loss() -> None:
    rgb = torch.ones((1, 3, 2, 2), dtype=torch.float32, requires_grad=True)
    visibility = torch.zeros((1, 2, 2), dtype=torch.bool)

    loss = masked_photometric_l1(rgb, torch.zeros_like(rgb), visibility)

    assert loss.item() == pytest.approx(0.0)
    loss.backward()
    assert rgb.grad is not None and torch.count_nonzero(rgb.grad) == 0


def test_sequence_photometric_identity_is_zero_and_graph_connected() -> None:
    height, width = 4, 5
    images = torch.rand((1, 1, 3, height, width), dtype=torch.float32).repeat(1, 2, 1, 1, 1)
    target_depth = torch.ones((1, 2, height, width), dtype=torch.float32)
    predicted_depth = target_depth[..., None].clone().requires_grad_()
    intrinsics, extrinsics = _camera(height, width)
    intrinsics = intrinsics[:, None].repeat(1, 2, 1, 1)
    extrinsics = extrinsics[:, None].repeat(1, 2, 1, 1)
    predicted_pose = build_camera_pose_target(extrinsics, intrinsics, (height, width)).detach().requires_grad_()
    batch = {
        "images": images,
        "depths": target_depth,
        "depth_masks": torch.ones_like(target_depth, dtype=torch.bool),
        "normalization_scale_m": torch.ones(1),
    }

    result = compute_sequence_photometric_loss(
        {"pose_enc": predicted_pose, "depth": predicted_depth},
        batch,
        backend="soft",
        max_depth_m=1.2,
    )

    assert result["photometric"].item() == pytest.approx(0.0, abs=1e-6)
    assert result["photometric_visibility"].item() == pytest.approx(1.0)
    result["photometric"].backward()
    assert predicted_depth.grad is not None and torch.isfinite(predicted_depth.grad).all()
    assert predicted_pose.grad is not None and torch.isfinite(predicted_pose.grad).all()
