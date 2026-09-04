from __future__ import annotations

import pytest
import torch

from vggt_omega.training.gpa_loss import (
    gpa_edge_aware_smoothness,
    gpa_sequence_loss,
    inverse_warp_source_to_target,
    sample_gpa_anchor_indices,
    sample_temporal_window_indices,
    transform_intrinsics_for_image_affine,
)


def _cameras(
    batch: int,
    frames: int,
    height: int,
    width: int,
    *,
    focal: float = 4.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    intrinsics = (
        torch.tensor(
            [[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        .reshape(1, 1, 3, 3)
        .repeat(batch, frames, 1, 1)
    )
    extrinsics = torch.eye(4, dtype=torch.float32)[:3].reshape(1, 1, 3, 4).repeat(batch, frames, 1, 1)
    return intrinsics, extrinsics


def test_gpa_inverse_warp_identity_has_zero_photo_and_geometry_cost() -> None:
    height, width = 5, 7
    target = torch.rand(1, 3, height, width)
    depth = torch.ones(1, height, width)
    intrinsics, extrinsics = _cameras(1, 2, height, width)

    result = inverse_warp_source_to_target(
        target,
        target,
        depth,
        depth,
        intrinsics[:, 0],
        intrinsics[:, 1],
        extrinsics[:, 0],
        extrinsics[:, 1],
        mu=0.85,
        geometry_epsilon=1e-6,
    )

    assert result["valid_mask"].all()
    torch.testing.assert_close(result["warped_source"], target, atol=1e-6, rtol=1e-6)
    assert result["photometric_cost"].abs().max().item() < 1e-6
    assert result["structural_cost"].abs().max().item() < 1e-6


def test_gpa_inverse_warp_known_translation_samples_expected_source_pixel() -> None:
    height, width = 5, 7
    target = torch.zeros(1, 3, height, width)
    source = torch.zeros_like(target)
    source[:, :, 2, 4] = 1.0
    depth = torch.ones(1, height, width)
    intrinsics, extrinsics = _cameras(1, 2, height, width)
    extrinsics[:, 1, 0, 3] = 0.25

    result = inverse_warp_source_to_target(
        target,
        source,
        depth,
        depth,
        intrinsics[:, 0],
        intrinsics[:, 1],
        extrinsics[:, 0],
        extrinsics[:, 1],
        mu=0.0,
        geometry_epsilon=1e-6,
    )

    torch.testing.assert_close(result["warped_source"][0, :, 2, 3], torch.ones(3), atol=1e-6, rtol=0)
    assert result["valid_mask"][0, 2, 3]


def test_gpa_structural_cost_detects_source_depth_mismatch() -> None:
    height, width = 4, 6
    image = torch.zeros(1, 3, height, width)
    target_depth = torch.ones(1, height, width)
    source_depth = torch.full_like(target_depth, 0.5)
    intrinsics, extrinsics = _cameras(1, 2, height, width)

    result = inverse_warp_source_to_target(
        image,
        image,
        target_depth,
        source_depth,
        intrinsics[:, 0],
        intrinsics[:, 1],
        extrinsics[:, 0],
        extrinsics[:, 1],
        mu=0.0,
        geometry_epsilon=1e-6,
    )

    assert result["structural_cost"][result["valid_mask"]].mean().item() == pytest.approx(1 / 3, rel=1e-5)


def test_gpa_hard_view_selects_lowest_combined_source_cost() -> None:
    height, width = 5, 7
    target = torch.rand(1, 3, height, width)
    images = torch.stack((target[0], target[0], torch.ones_like(target[0])), dim=0).unsqueeze(0)
    depths = torch.ones(1, 3, height, width)
    intrinsics, extrinsics = _cameras(1, 3, height, width)

    result = gpa_sequence_loss(
        images,
        depths,
        intrinsics,
        extrinsics,
        valid_mask=torch.ones_like(depths, dtype=torch.bool),
        anchor_indices=torch.tensor([[0]]),
        mu=0.85,
        lambda_geo=0.1,
        lambda_smooth=0.001,
        auto_mask_delta=0.0,
        auto_mask_enabled=False,
    )

    assert result["valid_fraction"] > 0
    assert torch.all(result["selected_source_indices"][result["selected_valid_mask"]] == 1)
    assert result["physical"] < 1e-6


def test_gpa_hard_view_rejects_structurally_occluded_source() -> None:
    height, width = 5, 7
    target = torch.zeros(1, 3, height, width)
    images = target[:, None].repeat(1, 3, 1, 1, 1)
    depths = torch.ones(1, 3, height, width)
    depths[:, 1] = 0.25
    intrinsics, extrinsics = _cameras(1, 3, height, width)

    result = gpa_sequence_loss(
        images,
        depths,
        intrinsics,
        extrinsics,
        valid_mask=torch.ones_like(depths, dtype=torch.bool),
        anchor_indices=torch.tensor([[0]]),
        mu=0.0,
        lambda_geo=1.0,
        lambda_smooth=0.0,
        auto_mask_delta=0.0,
        auto_mask_enabled=False,
    )

    assert torch.all(result["selected_source_indices"][result["selected_valid_mask"]] == 2)
    assert result["structural"].item() == pytest.approx(0.0)


def test_gpa_auto_mask_and_dynamic_mask_can_produce_graph_connected_zero() -> None:
    height, width = 4, 6
    image = torch.rand(1, 1, 3, height, width).repeat(1, 2, 1, 1, 1)
    depths = torch.ones(1, 2, height, width, requires_grad=True)
    intrinsics, extrinsics = _cameras(1, 2, height, width)

    auto_masked = gpa_sequence_loss(
        image,
        depths,
        intrinsics,
        extrinsics,
        valid_mask=torch.ones_like(depths, dtype=torch.bool),
        anchor_indices=torch.tensor([[0]]),
        mu=0.85,
        lambda_geo=0.1,
        lambda_smooth=0.001,
        auto_mask_delta=0.0,
        auto_mask_enabled=True,
    )
    assert auto_masked["valid_fraction"].item() == 0.0
    auto_masked["objective"].backward()
    assert depths.grad is not None and torch.count_nonzero(depths.grad) == 0

    dynamic = torch.ones_like(depths, dtype=torch.bool)
    empty = gpa_sequence_loss(
        image,
        depths.detach().requires_grad_(),
        intrinsics,
        extrinsics,
        valid_mask=torch.ones_like(depths, dtype=torch.bool),
        dynamic_mask=dynamic,
        anchor_indices=torch.tensor([[0]]),
        mu=0.85,
        lambda_geo=0.1,
        lambda_smooth=0.001,
        auto_mask_delta=0.0,
        auto_mask_enabled=False,
    )
    assert empty["valid_fraction"].item() == 0.0
    assert empty["objective"].item() == 0.0


def test_gpa_rectangular_intrinsics_smoothness_and_backward_are_finite() -> None:
    height, width = 6, 10
    images = torch.rand(1, 3, 3, height, width)
    depths = (torch.rand(1, 3, height, width) + 0.8).requires_grad_()
    intrinsics, extrinsics = _cameras(1, 3, height, width, focal=7.0)
    extrinsics[:, 1, 0, 3] = 0.05
    extrinsics[:, 2, 0, 3] = -0.05

    smoothness = gpa_edge_aware_smoothness(depths[:, 0], images[:, 0])
    result = gpa_sequence_loss(
        images,
        depths,
        intrinsics,
        extrinsics,
        valid_mask=torch.ones_like(depths, dtype=torch.bool),
        anchor_indices=torch.tensor([[0, 1, 2]]),
        mu=0.85,
        lambda_geo=0.1,
        lambda_smooth=0.001,
        auto_mask_delta=0.0,
        auto_mask_enabled=False,
        max_depth=2.0,
    )

    assert torch.isfinite(smoothness)
    for key in ("objective", "physical", "photometric", "structural", "smoothness", "valid_fraction"):
        assert result[key].ndim == 0 and torch.isfinite(result[key])
    result["objective"].backward()
    assert depths.grad is not None and torch.isfinite(depths.grad).all()
    assert torch.count_nonzero(depths.grad) > 0


def test_gpa_samples_three_unique_valid_anchors_and_variable_stride_window() -> None:
    frame_mask = torch.tensor([[True, True, True, True, False], [True, True, True, False, False]])
    anchors = sample_gpa_anchor_indices(
        frame_mask,
        anchor_count=3,
        generator=torch.Generator().manual_seed(7),
    )

    assert anchors.shape == (2, 3)
    for batch_index in range(2):
        assert len(anchors[batch_index].unique()) == 3
        assert frame_mask[batch_index, anchors[batch_index]].all()

    window = sample_temporal_window_indices(
        frame_count=10,
        window_size=4,
        stride_options=(1, 2),
        generator=torch.Generator().manual_seed(2),
    )
    assert window.shape == (4,)
    assert torch.all(window[1:] - window[:-1] == window[1] - window[0])
    assert int(window[1] - window[0]) in {1, 2}
    assert int(window[0]) >= 0 and int(window[-1]) < 10


def test_gpa_image_affine_updates_intrinsics_by_left_multiplication() -> None:
    intrinsics, _ = _cameras(1, 2, 6, 10, focal=7.0)
    horizontal_flip = torch.tensor(
        [[-1.0, 0.0, 9.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )

    transformed = transform_intrinsics_for_image_affine(intrinsics, horizontal_flip)

    torch.testing.assert_close(transformed, horizontal_flip @ intrinsics)
    assert transformed[0, 0, 0, 2].item() == pytest.approx(4.0)


def test_gpa_rejects_nonfinite_geometry_instead_of_masking_it() -> None:
    images = torch.zeros(1, 2, 3, 4, 6)
    depths = torch.ones(1, 2, 4, 6)
    depths[0, 1, 0, 0] = torch.nan
    intrinsics, extrinsics = _cameras(1, 2, 4, 6)

    with pytest.raises(ValueError, match="finite"):
        gpa_sequence_loss(
            images,
            depths,
            intrinsics,
            extrinsics,
            valid_mask=torch.ones_like(depths, dtype=torch.bool),
            anchor_indices=torch.tensor([[0]]),
            mu=0.85,
            lambda_geo=0.1,
            lambda_smooth=0.001,
            auto_mask_delta=0.0,
        )


def test_gpa_union_mask_is_explicit_compatibility_mode() -> None:
    images = torch.rand(1, 1, 3, 4, 6).repeat(1, 2, 1, 1, 1)
    depths = torch.ones(1, 2, 4, 6)
    intrinsics, extrinsics = _cameras(1, 2, 4, 6)
    options = {
        "valid_mask": torch.ones_like(depths, dtype=torch.bool),
        "anchor_indices": torch.tensor([[0]]),
        "mu": 0.85,
        "lambda_geo": 0.1,
        "lambda_smooth": 0.001,
        "auto_mask_delta": 0.0,
    }

    intersection = gpa_sequence_loss(images, depths, intrinsics, extrinsics, **options)
    union = gpa_sequence_loss(images, depths, intrinsics, extrinsics, mask_mode="union", **options)

    assert intersection["valid_fraction"].item() == 0.0
    assert union["valid_fraction"].item() == 1.0
    with pytest.raises(ValueError, match="mask_mode"):
        gpa_sequence_loss(images, depths, intrinsics, extrinsics, mask_mode="implicit", **options)
