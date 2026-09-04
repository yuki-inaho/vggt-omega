from __future__ import annotations

import math

import pytest
import torch

from vggt_omega.training.losses import (
    build_camera_pose_target,
    compute_camera_depth_loss,
    compute_camera_loss,
    compute_depth_loss,
)


def _known_camera() -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    angle = math.pi / 2
    rotation = torch.tensor(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    extrinsics = torch.zeros(1, 1, 3, 4)
    extrinsics[0, 0, :3, :3] = rotation
    extrinsics[0, 0, :, 3] = torch.tensor([1.0, 2.0, 3.0])

    intrinsics = torch.zeros(1, 1, 3, 3)
    intrinsics[..., 0, 0] = 4.0
    intrinsics[..., 1, 1] = 2.0
    intrinsics[..., 0, 2] = 4.0
    intrinsics[..., 1, 2] = 2.0
    intrinsics[..., 2, 2] = 1.0
    return extrinsics, intrinsics, (4, 8)


def test_build_camera_pose_target_uses_xyzw_and_vertical_horizontal_fov_order() -> None:
    extrinsics, intrinsics, image_size = _known_camera()

    target = build_camera_pose_target(extrinsics, intrinsics, image_size)

    expected = torch.tensor(
        [[[[1.0, 2.0, 3.0, 0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5), math.pi / 2, math.pi / 2]]]]
    ).reshape(1, 1, 9)
    assert target.shape == (1, 1, 9)
    assert torch.allclose(target, expected, atol=1e-6)


def test_compute_camera_loss_is_zero_for_exact_match_and_matches_known_components() -> None:
    extrinsics, intrinsics, image_size = _known_camera()
    target = build_camera_pose_target(extrinsics, intrinsics, image_size)

    exact = compute_camera_loss(target, extrinsics, intrinsics, image_size)
    assert exact["camera"].item() == pytest.approx(0.0)

    prediction = target.clone()
    prediction[..., :3] += 1.0
    prediction[..., 3:7] += 2.0
    prediction[..., 7:] += 4.0
    losses = compute_camera_loss(prediction, extrinsics, intrinsics, image_size)

    assert losses["camera_translation"].item() == pytest.approx(1.0)
    assert losses["camera_rotation"].item() == pytest.approx(2.0)
    assert losses["camera_fov"].item() == pytest.approx(4.0)
    assert losses["camera"].item() == pytest.approx(5.0)


def test_compute_camera_loss_supports_independent_motion_weights() -> None:
    extrinsics, intrinsics, image_size = _known_camera()
    target = build_camera_pose_target(extrinsics, intrinsics, image_size)
    prediction = target.clone()
    prediction[..., :3] += 1.0
    prediction[..., 3:7] += 2.0
    prediction[..., 7:] += 4.0

    losses = compute_camera_loss(
        prediction,
        extrinsics,
        intrinsics,
        image_size,
        translation_weight=4.0,
        rotation_weight=2.0,
        fov_weight=0.1,
    )

    assert losses["camera_translation"].item() == pytest.approx(1.0)
    assert losses["camera_rotation"].item() == pytest.approx(2.0)
    assert losses["camera_fov"].item() == pytest.approx(4.0)
    assert losses["camera"].item() == pytest.approx(8.4)


def test_compute_depth_loss_uses_only_true_mask_values() -> None:
    prediction = torch.tensor([[[[[1.0], [2.0]], [[3.0], [4.0]]]]], requires_grad=True)
    target = torch.tensor([[[[1.0, 4.0], [1.0, 8.0]]]])
    mask = torch.tensor([[[[True, False], [True, False]]]])

    loss = compute_depth_loss(prediction, target, mask)

    assert loss.item() == pytest.approx(1.0)
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_compute_depth_loss_empty_mask_returns_finite_graph_connected_zero() -> None:
    prediction = torch.ones(1, 1, 2, 2, 1, requires_grad=True)
    target = torch.zeros(1, 1, 2, 2)
    mask = torch.zeros(1, 1, 2, 2, dtype=torch.bool)

    loss = compute_depth_loss(prediction, target, mask)

    assert loss.item() == pytest.approx(0.0)
    assert loss.requires_grad
    assert torch.isfinite(loss)
    loss.backward()
    assert prediction.grad is not None
    assert torch.count_nonzero(prediction.grad).item() == 0


def test_compute_depth_loss_empty_mask_zero_does_not_overflow_reduction() -> None:
    prediction = torch.full(
        (1, 1, 2, 2, 1),
        torch.finfo(torch.float32).max,
        requires_grad=True,
    )
    target = torch.zeros(1, 1, 2, 2)
    mask = torch.zeros(1, 1, 2, 2, dtype=torch.bool)

    loss = compute_depth_loss(prediction, target, mask)

    assert loss.item() == pytest.approx(0.0)
    assert torch.isfinite(loss)
    loss.backward()
    assert prediction.grad is not None
    assert torch.count_nonzero(prediction.grad).item() == 0


def test_compute_depth_loss_honors_minimum_valid_pixel_threshold() -> None:
    prediction = torch.ones(1, 1, 2, 2, 1, requires_grad=True)
    target = torch.zeros(1, 1, 2, 2)
    mask = torch.tensor([[[[True, False], [True, False]]]])

    loss = compute_depth_loss(prediction, target, mask, min_valid_pixels=3)

    assert loss.item() == pytest.approx(0.0)
    loss.backward()
    assert prediction.grad is not None
    assert torch.count_nonzero(prediction.grad).item() == 0


def test_compute_camera_depth_loss_applies_fixed_objective_weights() -> None:
    extrinsics, intrinsics, image_size = _known_camera()
    pose_target = build_camera_pose_target(extrinsics, intrinsics, image_size)
    pose_prediction = pose_target.clone()
    pose_prediction[..., :3] += 1.0
    pose_prediction[..., 3:7] += 2.0
    pose_prediction[..., 7:] += 4.0
    depth_prediction = torch.tensor([[[[[2.0]]]]], requires_grad=True)
    batch = {
        "images": torch.zeros(1, 1, 3, *image_size),
        "depths": torch.tensor([[[[1.0]]]]),
        "depth_masks": torch.tensor([[[[True]]]]),
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
    }

    losses = compute_camera_depth_loss(
        {"pose_enc": pose_prediction, "depth": depth_prediction},
        batch,
    )

    assert losses["camera"].item() == pytest.approx(5.0)
    assert losses["depth"].item() == pytest.approx(1.0)
    assert losses["objective"].item() == pytest.approx(26.0)


@pytest.mark.parametrize("which", ["pose", "depth", "target_depth"])
def test_losses_reject_nonfinite_values(which: str) -> None:
    extrinsics, intrinsics, image_size = _known_camera()
    pose = build_camera_pose_target(extrinsics, intrinsics, image_size)
    depth = torch.ones(1, 1, 1, 1, 1)
    target_depth = torch.ones(1, 1, 1, 1)
    mask = torch.ones(1, 1, 1, 1, dtype=torch.bool)
    if which == "pose":
        pose[..., 0] = torch.nan
        with pytest.raises(ValueError, match="predicted_pose"):
            compute_camera_loss(pose, extrinsics, intrinsics, image_size)
    else:
        if which == "depth":
            depth[..., 0] = torch.inf
        else:
            target_depth[..., 0] = torch.nan
        with pytest.raises(ValueError, match="depth"):
            compute_depth_loss(depth, target_depth, mask)


def test_losses_reject_shapes_that_would_otherwise_broadcast() -> None:
    extrinsics, intrinsics, image_size = _known_camera()
    bad_pose = torch.zeros(1, 9)
    with pytest.raises(ValueError, match="predicted_pose"):
        compute_camera_loss(bad_pose, extrinsics, intrinsics, image_size)

    prediction = torch.zeros(1, 1, 2, 2, 1)
    target = torch.zeros(1, 2, 2)
    mask = torch.ones(1, 1, 2, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match="target_depth"):
        compute_depth_loss(prediction, target, mask)
