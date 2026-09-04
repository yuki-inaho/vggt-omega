from __future__ import annotations

import pytest
import torch

from vggt_omega.training.correspondence import (
    CORRESPONDENCE_COORDINATE_SPACE,
    FactoredCorrespondenceHead,
    build_rgbd_correspondence_targets,
    masked_generalized_charbonnier,
    validate_external_teacher_targets,
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


def test_rgbd_correspondence_identity_and_direction_reversal() -> None:
    height, width = 5, 7
    depths = torch.ones(1, 2, height, width)
    intrinsics, extrinsics = _cameras(1, 2, height, width)
    valid = torch.ones_like(depths, dtype=torch.bool)

    identity = build_rgbd_correspondence_targets(
        depths,
        intrinsics,
        extrinsics,
        torch.tensor([[[0, 1]]]),
        valid_mask=valid,
        relative_depth_tolerance=0.01,
    )
    assert identity["covisibility_mask"].all()
    torch.testing.assert_close(identity["flow_pixels"], torch.zeros_like(identity["flow_pixels"]), atol=1e-6, rtol=0)

    extrinsics[:, 1, 0, 3] = 0.25
    directed = build_rgbd_correspondence_targets(
        depths,
        intrinsics,
        extrinsics,
        torch.tensor([[[0, 1], [1, 0]]]),
        valid_mask=valid,
        relative_depth_tolerance=0.01,
    )
    assert directed["covisibility_mask"][0, 0, 2, 3]
    assert directed["covisibility_mask"][0, 1, 2, 3]
    assert directed["flow_pixels"][0, 0, 2, 3, 0].item() == pytest.approx(1.0)
    assert directed["flow_pixels"][0, 1, 2, 3, 0].item() == pytest.approx(-1.0)


def test_rgbd_correspondence_masks_nonoverlap_dynamic_and_far_pixels() -> None:
    height, width = 4, 6
    depths = torch.ones(1, 2, height, width)
    depths[..., 0, 0] = 1.5
    intrinsics, extrinsics = _cameras(1, 2, height, width)
    extrinsics[:, 1, 0, 3] = 100.0
    dynamic = torch.zeros_like(depths, dtype=torch.bool)
    dynamic[..., 1, 1] = True

    result = build_rgbd_correspondence_targets(
        depths,
        intrinsics,
        extrinsics,
        torch.tensor([[[0, 1]]]),
        valid_mask=torch.ones_like(depths, dtype=torch.bool),
        dynamic_mask=dynamic,
        max_depth=1.2,
        relative_depth_tolerance=0.01,
    )

    assert not result["covisibility_mask"].any()
    assert torch.count_nonzero(result["flow_pixels"]) == 0


def test_rgbd_correspondence_applies_dynamic_near_and_padding_contracts() -> None:
    height, width = 4, 6
    depths = torch.ones(1, 3, height, width)
    depths[:, 0, 0, 0] = 1.5
    intrinsics, extrinsics = _cameras(1, 3, height, width)
    valid = torch.ones_like(depths, dtype=torch.bool)
    dynamic = torch.zeros_like(depths, dtype=torch.bool)
    dynamic[:, 0, 1, 1] = True

    result = build_rgbd_correspondence_targets(
        depths,
        intrinsics,
        extrinsics,
        torch.tensor([[[0, 1]]]),
        valid_mask=valid,
        dynamic_mask=dynamic,
        frame_mask=torch.tensor([[True, True, False]]),
        max_depth=1.2,
        relative_depth_tolerance=0.01,
    )

    assert CORRESPONDENCE_COORDINATE_SPACE == "pixel_displacement_xy"
    assert not result["covisibility_mask"][0, 0, 0, 0]
    assert not result["covisibility_mask"][0, 0, 1, 1]
    assert result["covisibility_mask"][0, 0, 2, 2]
    with pytest.raises(ValueError, match="padding"):
        build_rgbd_correspondence_targets(
            depths,
            intrinsics,
            extrinsics,
            torch.tensor([[[0, 2]]]),
            valid_mask=valid,
            frame_mask=torch.tensor([[True, True, False]]),
            relative_depth_tolerance=0.01,
        )


def test_factored_head_rectangular_shape_chunk_parity_and_asymmetry() -> None:
    torch.manual_seed(3)
    head = FactoredCorrespondenceHead(geometry_dim=12, camera_dim=9, hidden_dim=16)
    geometry = torch.rand(2, 3, 6, 12)
    cameras = torch.rand(2, 3, 9)
    pairs = torch.tensor([[[0, 1], [0, 2], [1, 2]], [[0, 1], [0, 2], [1, 2]]])

    full = head(
        geometry,
        cameras,
        pairs,
        source_grid_hw=(2, 3),
        output_hw=(8, 12),
        pair_chunk_size=3,
    )
    chunked = head(
        geometry,
        cameras,
        pairs,
        source_grid_hw=(2, 3),
        output_hw=(8, 12),
        pair_chunk_size=1,
    )

    assert full.shape == (2, 3, 8, 12, 2)
    torch.testing.assert_close(chunked, full, atol=0, rtol=0)
    with torch.no_grad():
        head.output_projection.weight.fill_(0.1)
    changed_target = cameras.clone()
    changed_target[:, 1] += 1
    changed = head(
        geometry,
        changed_target,
        pairs,
        source_grid_hw=(2, 3),
        output_hw=(8, 12),
        pair_chunk_size=2,
    )
    assert not torch.equal(changed[:, 0], full[:, 0])


def test_correspondence_charbonnier_mask_and_backward() -> None:
    prediction = torch.tensor([[[[[0.0, 0.0], [2.0, 0.0]]]]], requires_grad=True)
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[[[True, False]]]])

    perfect = masked_generalized_charbonnier(prediction, target, mask, alpha=0.5, epsilon=0.01)
    assert perfect.item() == pytest.approx(0.0)
    perfect.backward()
    assert prediction.grad is not None and torch.count_nonzero(prediction.grad) == 0

    prediction.grad = None
    mask[..., 1] = True
    nonzero = masked_generalized_charbonnier(prediction, target, mask, alpha=0.5, epsilon=0.01)
    nonzero.backward()
    assert nonzero.item() > 0
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()
    assert prediction.grad[..., 1, 0].abs().item() > 0


def test_empty_correspondence_mask_returns_graph_connected_zero() -> None:
    prediction = torch.rand(1, 1, 2, 3, 2, requires_grad=True)
    loss = masked_generalized_charbonnier(
        prediction,
        torch.zeros_like(prediction),
        torch.zeros(1, 1, 2, 3, dtype=torch.bool),
        alpha=0.5,
        epsilon=0.01,
    )

    assert loss.item() == 0.0
    loss.backward()
    assert prediction.grad is not None and torch.count_nonzero(prediction.grad) == 0


def test_factored_head_has_finite_parameter_gradient() -> None:
    head = FactoredCorrespondenceHead(geometry_dim=12, camera_dim=9, hidden_dim=16)
    geometry = torch.rand(1, 2, 6, 12, requires_grad=True)
    cameras = torch.rand(1, 2, 9, requires_grad=True)
    prediction = head(
        geometry,
        cameras,
        torch.tensor([[[0, 1]]]),
        source_grid_hw=(2, 3),
        output_hw=(8, 12),
        pair_chunk_size=1,
    )
    loss = masked_generalized_charbonnier(
        prediction,
        torch.ones_like(prediction),
        torch.ones(1, 1, 8, 12, dtype=torch.bool),
        alpha=0.5,
        epsilon=0.01,
    )

    loss.backward()

    gradient = head.output_projection.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_external_teacher_extension_point_requires_explicit_generic_schema() -> None:
    pairs = torch.tensor([[[0, 1]]])
    flow = torch.zeros(1, 1, 4, 6, 2)
    mask = torch.ones(1, 1, 4, 6, dtype=torch.bool)
    payload = {
        "schema_version": 1,
        "coordinate_space": "pixel_displacement_xy",
        "pair_indices": pairs.clone(),
        "flow_pixels": flow,
        "covisibility_mask": mask,
    }

    validated = validate_external_teacher_targets(payload, expected_pair_indices=pairs, output_hw=(4, 6))

    assert validated["flow_pixels"] is flow
    assert validated["covisibility_mask"] is mask
    with pytest.raises(ValueError, match="schema"):
        validate_external_teacher_targets(
            {**payload, "schema_version": 2},
            expected_pair_indices=pairs,
            output_hw=(4, 6),
        )
    with pytest.raises(ValueError, match="fields"):
        validate_external_teacher_targets(
            {**payload, "teacher_confidence": torch.ones(1)},
            expected_pair_indices=pairs,
            output_hw=(4, 6),
        )
