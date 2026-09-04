from __future__ import annotations

import pytest
import torch

from vggt_omega.training.dynamic_losses import (
    bounded_self_supervised_area_prior,
    confidence_weighted_scene_flow_regression,
    dynamic_temporal_consistency_loss,
    edge_aware_dynamic_spatial_consistency_loss,
    multi_view_reprojection_loss,
    source_grid_forward_backward_3d_cycle_loss,
    temporal_target_depth_consistency_loss,
    tri_state_binary_cross_entropy,
)


def _pairs() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.tensor([[[0, 1], [1, 0]]]), torch.ones((1, 2), dtype=torch.bool)


def _pixel_grid(height: int, width: int) -> torch.Tensor:
    rows, columns = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    return torch.stack((columns, rows), dim=-1).to(torch.float32)[None, None]


def test_confidence_weighted_scene_flow_is_exact_zero_and_positive() -> None:
    target = torch.zeros((1, 2, 2, 3, 3))
    prediction = target.clone().requires_grad_()
    confidence = torch.ones((1, 2, 2, 3))
    valid = torch.ones_like(confidence, dtype=torch.bool)

    zero = confidence_weighted_scene_flow_regression(
        prediction,
        target,
        confidence,
        valid,
        alpha=0.5,
        epsilon=1e-3,
    )
    assert zero.dtype is torch.float32
    assert zero.item() == 0.0

    changed = prediction.detach().clone()
    changed[..., 0, 0, 0] = 1.0
    positive = confidence_weighted_scene_flow_regression(
        changed,
        target,
        confidence,
        valid,
        alpha=0.5,
        epsilon=1e-3,
    )
    assert positive.item() > 0.0


def test_losses_mask_before_arithmetic_and_empty_mask_has_finite_backward() -> None:
    prediction = torch.full((1, 1, 2, 3, 3), torch.nan, requires_grad=True)
    target = torch.full_like(prediction, torch.nan)
    confidence = torch.full((1, 1, 2, 3), torch.nan)
    valid = torch.zeros_like(confidence, dtype=torch.bool)

    loss = confidence_weighted_scene_flow_regression(
        prediction,
        target,
        confidence,
        valid,
        alpha=0.5,
        epsilon=1e-3,
    )
    loss.backward()
    assert loss.item() == 0.0
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_tri_state_bce_handles_visible_occluded_static_dynamic_and_unknown() -> None:
    labels = torch.tensor([[[[1, 0, -1]]]], dtype=torch.int8)
    known = labels >= 0
    domain = torch.ones_like(known)
    confidence = torch.tensor([[[[1.0, 0.5, float("nan")]]]])
    good_logits = torch.tensor([[[[12.0, -12.0, float("nan")]]]], requires_grad=True)
    bad_logits = torch.tensor([[[[-12.0, 12.0, float("nan")]]]])

    good = tri_state_binary_cross_entropy(
        good_logits,
        labels,
        known_mask=known,
        domain_mask=domain,
        confidence=confidence,
    )
    bad = tri_state_binary_cross_entropy(
        bad_logits,
        labels,
        known_mask=known,
        domain_mask=domain,
        confidence=confidence,
    )
    assert good.item() < 1e-4
    assert bad.item() > 10.0
    good.backward()
    assert torch.isfinite(good_logits.grad).all()


def test_tri_state_bce_balances_rare_positive_against_many_negatives() -> None:
    labels = torch.zeros((1, 1, 1, 101), dtype=torch.int8)
    labels[..., 0] = 1
    known = torch.ones_like(labels, dtype=torch.bool)
    logits = torch.full(labels.shape, -4.0, requires_grad=True)

    loss = tri_state_binary_cross_entropy(
        logits,
        labels,
        known_mask=known,
        domain_mask=known,
        confidence=torch.ones_like(logits),
    )

    assert loss.item() > 1.0
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_source_grid_3d_cycle_uses_reverse_bilinear_sample_on_rectangular_grid() -> None:
    height, width = 2, 4
    pairs, pair_valid = _pairs()
    forward = torch.zeros((1, 2, height, width, 3))
    forward[:, 0, ..., 0] = 1.0
    forward[:, 1, ..., 0] = -1.0
    pixel_flow = torch.zeros((1, 2, height, width, 2))
    domain = torch.ones((1, 2, height, width), dtype=torch.bool)
    confidence = torch.ones_like(domain, dtype=torch.float32)

    zero = source_grid_forward_backward_3d_cycle_loss(
        forward,
        pixel_flow,
        pairs,
        pair_valid,
        domain_mask=domain,
        confidence=confidence,
        missing_reverse_policy="reject",
    )
    assert zero.item() == 0.0

    pixel_flow[:, 0, ..., 0] = 0.5
    reverse_ramp = torch.arange(width, dtype=torch.float32).view(1, 1, 1, width)
    forward[:, 1, ..., 0] = -(1.0 + reverse_ramp)
    positive = source_grid_forward_backward_3d_cycle_loss(
        forward,
        pixel_flow,
        pairs,
        pair_valid,
        domain_mask=domain,
        confidence=confidence,
        missing_reverse_policy="reject",
    )
    assert positive.item() > 0.0


def test_reverse_pair_missing_is_rejected_or_treated_as_unknown() -> None:
    flow = torch.zeros((1, 1, 2, 3, 3), requires_grad=True)
    pixel_flow = torch.zeros((1, 1, 2, 3, 2))
    pairs = torch.tensor([[[0, 1]]])
    pair_valid = torch.ones((1, 1), dtype=torch.bool)
    domain = torch.ones((1, 1, 2, 3), dtype=torch.bool)
    confidence = torch.ones_like(domain, dtype=torch.float32)

    with pytest.raises(ValueError, match="reverse pair"):
        source_grid_forward_backward_3d_cycle_loss(
            flow,
            pixel_flow,
            pairs,
            pair_valid,
            domain_mask=domain,
            confidence=confidence,
            missing_reverse_policy="reject",
        )
    unknown = source_grid_forward_backward_3d_cycle_loss(
        flow,
        pixel_flow,
        pairs,
        pair_valid,
        domain_mask=domain,
        confidence=confidence,
        missing_reverse_policy="unknown",
    )
    unknown.backward()
    assert unknown.item() == 0.0
    assert flow.grad is not None and torch.isfinite(flow.grad).all()


def test_multi_view_reprojection_identity_and_offset() -> None:
    height, width = 2, 4
    pixels = _pixel_grid(height, width)
    points = torch.cat((pixels.expand(1, 1, -1, -1, -1), torch.ones((1, 1, height, width, 1))), dim=-1)
    w2c = torch.eye(4)[:3].reshape(1, 1, 3, 4)
    intrinsics = torch.eye(3).reshape(1, 1, 3, 3)
    valid = torch.ones((1, 1, height, width), dtype=torch.bool)
    confidence = torch.ones_like(valid, dtype=torch.float32)

    zero = multi_view_reprojection_loss(
        points,
        w2c,
        intrinsics,
        pixels.expand(1, 1, -1, -1, -1),
        confidence,
        valid,
    )
    positive = multi_view_reprojection_loss(
        points,
        w2c,
        intrinsics,
        pixels.expand(1, 1, -1, -1, -1) + 1,
        confidence,
        valid,
    )
    assert zero.item() == 0.0
    assert positive.item() > 0.0


def test_temporal_target_depth_consistency_samples_target_grid() -> None:
    height, width = 2, 4
    target_depth = torch.arange(1, width + 1, dtype=torch.float32).view(1, 1, 1, width).expand(1, 1, height, width)
    pixel_flow = torch.zeros((1, 1, height, width, 2))
    pixel_flow[..., 0] = 1.0
    predicted = torch.zeros_like(target_depth)
    predicted[..., :-1] = target_depth[..., 1:]
    valid = torch.ones_like(target_depth, dtype=torch.bool)
    confidence = torch.ones_like(target_depth)

    zero = temporal_target_depth_consistency_loss(
        predicted,
        target_depth,
        pixel_flow,
        confidence,
        source_valid_mask=valid,
        target_valid_mask=valid,
        epsilon=1e-6,
    )
    assert zero.item() == 0.0
    assert zero.dtype is torch.float32


def test_edge_aware_spatial_dynamic_loss_respects_image_edges() -> None:
    probability = torch.tensor([[[[0.0, 0.0, 1.0, 1.0]]]])
    flat_image = torch.zeros((1, 1, 3, 1, 4))
    edge_image = flat_image.clone()
    edge_image[..., 2:] = 1.0
    valid = torch.ones_like(probability, dtype=torch.bool)

    flat = edge_aware_dynamic_spatial_consistency_loss(probability, flat_image, valid, edge_scale=10.0)
    edged = edge_aware_dynamic_spatial_consistency_loss(probability, edge_image, valid, edge_scale=10.0)
    assert flat.item() > edged.item() >= 0.0


def test_dynamic_temporal_consistency_and_missing_reverse_policy() -> None:
    height, width = 2, 4
    pairs, pair_valid = _pairs()
    probability = torch.full((1, 2, height, width), 0.25)
    pixel_flow = torch.zeros((1, 2, height, width, 2))
    domain = torch.ones_like(probability, dtype=torch.bool)
    confidence = torch.ones_like(probability)
    zero = dynamic_temporal_consistency_loss(
        probability,
        pixel_flow,
        pairs,
        pair_valid,
        domain_mask=domain,
        confidence=confidence,
        missing_reverse_policy="reject",
    )
    assert zero.item() == 0.0

    with pytest.raises(ValueError, match="reverse pair"):
        dynamic_temporal_consistency_loss(
            probability[:, :1],
            pixel_flow[:, :1],
            pairs[:, :1],
            pair_valid[:, :1],
            domain_mask=domain[:, :1],
            confidence=confidence[:, :1],
            missing_reverse_policy="reject",
        )


def test_area_prior_penalizes_all_static_and_all_dynamic_per_pair() -> None:
    valid = torch.ones((1, 2, 2, 4), dtype=torch.bool)
    all_static = torch.zeros((1, 2, 2, 4), requires_grad=True)
    all_dynamic = torch.ones((1, 2, 2, 4), requires_grad=True)
    in_bounds = torch.full((1, 2, 2, 4), 0.2)

    low = bounded_self_supervised_area_prior(all_static, valid, minimum_fraction=0.1, maximum_fraction=0.4)
    high = bounded_self_supervised_area_prior(all_dynamic, valid, minimum_fraction=0.1, maximum_fraction=0.4)
    zero = bounded_self_supervised_area_prior(in_bounds, valid, minimum_fraction=0.1, maximum_fraction=0.4)
    assert low.item() > 0.0
    assert high.item() > 0.0
    assert zero.item() == 0.0
    (low + high).backward()
    assert torch.isfinite(all_static.grad).all()
    assert torch.isfinite(all_dynamic.grad).all()


def test_area_prior_empty_mask_is_graph_connected_and_validation_is_strict() -> None:
    probabilities = torch.full((1, 1, 2, 3), torch.nan, requires_grad=True)
    valid = torch.zeros_like(probabilities, dtype=torch.bool)
    loss = bounded_self_supervised_area_prior(
        probabilities,
        valid,
        minimum_fraction=0.1,
        maximum_fraction=0.4,
    )
    loss.backward()
    assert loss.item() == 0.0
    assert torch.isfinite(probabilities.grad).all()

    with pytest.raises(ValueError, match="minimum_fraction"):
        bounded_self_supervised_area_prior(
            torch.zeros((1, 1, 2, 3)),
            valid,
            minimum_fraction=0.5,
            maximum_fraction=0.4,
        )


def test_valid_nonfinite_values_are_rejected_instead_of_nan_times_zero() -> None:
    prediction = torch.zeros((1, 1, 1, 1, 3))
    prediction[..., 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        confidence_weighted_scene_flow_regression(
            prediction,
            torch.zeros_like(prediction),
            torch.ones((1, 1, 1, 1)),
            torch.ones((1, 1, 1, 1), dtype=torch.bool),
            alpha=0.5,
            epsilon=1e-3,
        )


def test_every_loss_returns_graph_connected_zero_when_its_domain_is_empty() -> None:
    height, width = 2, 3
    pairs, pair_valid = _pairs()
    empty = torch.zeros((1, 2, height, width), dtype=torch.bool)
    pair_field = torch.full((1, 2, height, width), torch.nan)
    pair_vector2 = torch.full((1, 2, height, width, 2), torch.nan)
    confidence = pair_field.clone()
    leaves = [
        torch.full((1, 2, height, width, 3), torch.nan, requires_grad=True),
        pair_field.clone().requires_grad_(),
        torch.full((1, 2, height, width, 3), torch.nan, requires_grad=True),
        pair_field.clone().requires_grad_(),
        pair_field.clone().requires_grad_(),
        pair_field.clone().requires_grad_(),
        pair_field.clone().requires_grad_(),
        pair_field.clone().requires_grad_(),
    ]
    losses = [
        confidence_weighted_scene_flow_regression(
            leaves[0],
            torch.full_like(leaves[0], torch.nan),
            confidence,
            empty,
            alpha=0.5,
            epsilon=1e-3,
        ),
        tri_state_binary_cross_entropy(
            leaves[1],
            torch.full_like(leaves[1], -1, dtype=torch.int8),
            known_mask=empty,
            domain_mask=empty,
            confidence=confidence,
        ),
        source_grid_forward_backward_3d_cycle_loss(
            leaves[2],
            pair_vector2,
            pairs,
            pair_valid,
            domain_mask=empty,
            confidence=confidence,
            missing_reverse_policy="reject",
        ),
        multi_view_reprojection_loss(
            leaves[3][..., None].expand(-1, -1, -1, -1, 3),
            torch.full((1, 2, 3, 4), torch.nan),
            torch.full((1, 2, 3, 3), torch.nan),
            pair_vector2,
            confidence,
            empty,
        ),
        temporal_target_depth_consistency_loss(
            leaves[4],
            pair_field,
            pair_vector2,
            confidence,
            source_valid_mask=empty,
            target_valid_mask=empty,
            epsilon=1e-6,
        ),
        edge_aware_dynamic_spatial_consistency_loss(
            leaves[5],
            torch.full((1, 2, 3, height, width), torch.nan),
            empty,
            edge_scale=1.0,
        ),
        dynamic_temporal_consistency_loss(
            leaves[6],
            pair_vector2,
            pairs,
            pair_valid,
            domain_mask=empty,
            confidence=confidence,
            missing_reverse_policy="reject",
        ),
        bounded_self_supervised_area_prior(
            leaves[7],
            empty,
            minimum_fraction=0.1,
            maximum_fraction=0.4,
        ),
    ]
    total = torch.stack(losses).sum()
    total.backward()
    assert total.item() == 0.0
    assert all(leaf.grad is not None and torch.isfinite(leaf.grad).all() for leaf in leaves)


def test_scene_flow_reduction_is_fp32_for_half_precision_inputs() -> None:
    prediction = torch.zeros((1, 1, 2, 3, 3), dtype=torch.float16, requires_grad=True)
    target = torch.zeros_like(prediction)
    confidence = torch.ones((1, 1, 2, 3), dtype=torch.float16)
    valid = torch.ones_like(confidence, dtype=torch.bool)
    loss = confidence_weighted_scene_flow_regression(
        prediction,
        target,
        confidence,
        valid,
        alpha=0.5,
        epsilon=1e-3,
    )
    loss.backward()
    assert loss.dtype is torch.float32
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()
