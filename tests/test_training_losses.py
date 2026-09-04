from __future__ import annotations

import math

import pytest
import torch

from vggt_omega.training.losses import (
    build_camera_pose_target,
    compute_camera_depth_loss,
    compute_camera_loss,
    compute_depth_loss,
    compute_pairwise_pose_loss,
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


def test_compute_camera_depth_loss_masks_depth_in_metric_space() -> None:
    extrinsics, intrinsics, image_size = _known_camera()
    pose_target = build_camera_pose_target(extrinsics, intrinsics, image_size)
    batch = {
        "images": torch.zeros(1, 1, 3, *image_size),
        "depths": torch.tensor([[[[1.0, 3.0]]]]),
        "depth_masks": torch.tensor([[[[True, True]]]]),
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
        "normalization_scale_m": torch.tensor(0.5),
    }
    depth_prediction = torch.tensor([[[[[2.0], [0.0]]]]], requires_grad=True)

    losses = compute_camera_depth_loss(
        {"pose_enc": pose_target, "depth": depth_prediction},
        batch,
        max_metric_depth_m=1.2,
    )

    assert losses["depth"].item() == pytest.approx(1.0)
    losses["objective"].backward()
    assert depth_prediction.grad is not None
    assert depth_prediction.grad[0, 0, 0, 0, 0].item() != 0.0
    assert depth_prediction.grad[0, 0, 0, 1, 0].item() == 0.0


def test_metric_depth_mask_requires_normalization_scale() -> None:
    extrinsics, intrinsics, image_size = _known_camera()
    pose_target = build_camera_pose_target(extrinsics, intrinsics, image_size)
    batch = {
        "images": torch.zeros(1, 1, 3, *image_size),
        "depths": torch.ones(1, 1, 1, 1),
        "depth_masks": torch.ones(1, 1, 1, 1, dtype=torch.bool),
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
    }

    with pytest.raises(KeyError, match="normalization_scale_m"):
        compute_camera_depth_loss(
            {"pose_enc": pose_target, "depth": torch.ones(1, 1, 1, 1, 1)},
            batch,
            max_metric_depth_m=1.2,
        )


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


def _two_camera_setup() -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    extrinsics = torch.eye(4, dtype=torch.float32)[None, None, :3].repeat(1, 2, 1, 1)
    extrinsics[0, 1, 0, 3] = 1.0
    intrinsics = torch.eye(3, dtype=torch.float32)[None, None].repeat(1, 2, 1, 1)
    intrinsics[..., 0, 0] = 4.0
    intrinsics[..., 1, 1] = 4.0
    intrinsics[..., 0, 2] = 4.0
    intrinsics[..., 1, 2] = 2.0
    return extrinsics, intrinsics, (4, 8)


def test_pairwise_pose_loss_is_zero_for_exact_pose_and_quaternion_sign() -> None:
    extrinsics, intrinsics, image_size = _two_camera_setup()
    pose = build_camera_pose_target(extrinsics, intrinsics, image_size)

    exact = compute_pairwise_pose_loss(pose, extrinsics, image_size)
    sign_flipped = pose.clone()
    sign_flipped[..., 3:7] *= -1
    flipped = compute_pairwise_pose_loss(sign_flipped, extrinsics, image_size)

    assert exact["pairwise_pose"].item() == pytest.approx(0.0, abs=1e-6)
    assert flipped["pairwise_rotation"].item() == pytest.approx(0.0, abs=1e-6)
    assert exact["rpa_5"].item() == pytest.approx(1.0)
    assert exact["rpa_15"].item() == pytest.approx(1.0)
    assert exact["rpa_30"].item() == pytest.approx(1.0)


def test_pairwise_pose_loss_matches_known_rotation_and_translation_direction() -> None:
    extrinsics, intrinsics, image_size = _two_camera_setup()
    predicted_extrinsics = extrinsics.clone()
    predicted_extrinsics[0, 1, :3, :3] = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
    )
    predicted_extrinsics[0, 1, :3, 3] = torch.tensor([0.0, 1.0, 0.0])
    predicted_pose = build_camera_pose_target(predicted_extrinsics, intrinsics, image_size)

    losses = compute_pairwise_pose_loss(predicted_pose, extrinsics, image_size)

    assert losses["pairwise_rotation"].item() == pytest.approx(math.pi / 2, rel=1e-5)
    assert losses["pairwise_translation_direction"].item() == pytest.approx(math.pi / 2, rel=1e-5)
    assert losses["pairwise_translation_magnitude"].item() == pytest.approx(0.0, abs=1e-6)
    assert losses["pairwise_valid_direction_fraction"].item() == pytest.approx(1.0)
    assert losses["pairwise_rotation_degrees"].item() == pytest.approx(90.0, rel=1e-5)
    assert losses["pairwise_translation_direction_degrees"].item() == pytest.approx(90.0, rel=1e-5)
    assert losses["rpa_5"].item() == pytest.approx(0.0)
    assert losses["rpa_15"].item() == pytest.approx(0.0)
    assert losses["rpa_30"].item() == pytest.approx(0.0)


def test_pairwise_zero_baseline_masks_direction_but_keeps_magnitude() -> None:
    extrinsics, intrinsics, image_size = _two_camera_setup()
    extrinsics[0, 1, :3, 3] = 0.0
    predicted_extrinsics = extrinsics.clone()
    predicted_extrinsics[0, 1, 0, 3] = 1.0
    predicted_pose = build_camera_pose_target(predicted_extrinsics, intrinsics, image_size)

    losses = compute_pairwise_pose_loss(predicted_pose, extrinsics, image_size)

    assert losses["pairwise_translation_direction"].item() == pytest.approx(0.0)
    assert losses["pairwise_valid_direction_fraction"].item() == pytest.approx(0.0)
    assert losses["pairwise_translation_magnitude"].item() == pytest.approx(1.0)
    assert losses["rpa_5"].item() == pytest.approx(0.0)


def test_pairwise_pose_loss_rejects_nonfinite_prediction() -> None:
    extrinsics, intrinsics, image_size = _two_camera_setup()
    pose = build_camera_pose_target(extrinsics, intrinsics, image_size)
    pose[0, 1, 0] = float("nan")

    with pytest.raises(ValueError, match="predicted_pose"):
        compute_pairwise_pose_loss(pose, extrinsics, image_size)


def test_camera_depth_loss_adds_explicit_photometric_objective() -> None:
    height, width = 4, 5
    images = torch.rand((1, 1, 3, height, width), dtype=torch.float32).repeat(1, 2, 1, 1, 1)
    depths = torch.ones((1, 2, height, width), dtype=torch.float32)
    extrinsics = torch.eye(4, dtype=torch.float32)[None, None, :3].repeat(1, 2, 1, 1)
    intrinsics = torch.tensor(
        [[[[4.0, 0.0, width / 2], [0.0, 4.0, height / 2], [0.0, 0.0, 1.0]]]],
        dtype=torch.float32,
    ).repeat(1, 2, 1, 1)
    pose = build_camera_pose_target(extrinsics, intrinsics, (height, width))
    predictions = {"pose_enc": pose, "depth": depths[..., None]}
    batch = {
        "images": images,
        "depths": depths,
        "depth_masks": torch.ones_like(depths, dtype=torch.bool),
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
        "normalization_scale_m": torch.ones(1),
    }

    losses = compute_camera_depth_loss(
        predictions,
        batch,
        photometric_weight=0.25,
        renderer_options={"backend": "soft", "max_depth_m": 1.2},
    )

    assert losses["photometric"].item() == pytest.approx(0.0, abs=1e-6)
    assert losses["photometric_visibility"].item() == pytest.approx(1.0)
    assert losses["objective"].item() == pytest.approx(0.0, abs=1e-6)
