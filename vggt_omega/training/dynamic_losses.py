"""Pure, mask-safe losses for optional dynamic-geometry training.

The functions in this module operate on the public source-pixel-aligned 4D
contract.  They deliberately do not own curriculum state or loss weights.
Callers must provide every mask, confidence, threshold, and coefficient from
resolved configuration.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F

MissingReversePolicy = Literal["reject", "unknown"]


def _require_scalar(
    name: str,
    value: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_open: bool = False,
    maximum_open: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite scalar")
    result = float(value)
    below = minimum is not None and (result <= minimum if minimum_open else result < minimum)
    above = maximum is not None and (result >= maximum if maximum_open else result > maximum)
    if below or above:
        raise ValueError(f"{name} is outside its permitted interval")
    return result


def _require_float_tensor(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating-point tensor")


def _require_mask(name: str, value: torch.Tensor, shape: torch.Size, device: torch.device) -> None:
    if value.shape != shape or value.dtype is not torch.bool or value.device != device:
        raise ValueError(f"{name} must be bool with shape {tuple(shape)} on the shared device")


def _expanded_mask(mask: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    result = mask
    while result.ndim < value.ndim:
        result = result.unsqueeze(-1)
    return result.expand_as(value)


def _safe_on_mask(name: str, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = _expanded_mask(mask, value)
    if bool((expanded & ~torch.isfinite(value)).any()):
        raise ValueError(f"{name} must be finite on every active element")
    return torch.where(expanded, value.float(), torch.zeros_like(value, dtype=torch.float32))


def _graph_connected_zero(*values: torch.Tensor) -> torch.Tensor:
    if not values:
        raise ValueError("at least one tensor is required for graph-connected zero")
    zero = torch.zeros((), dtype=torch.float32, device=values[0].device)
    for value in values:
        safe = torch.where(torch.isfinite(value), value.float(), torch.zeros_like(value, dtype=torch.float32))
        zero = zero + safe.sum() * 0.0
    return zero


def _safe_confidence(
    confidence: torch.Tensor,
    active_domain: torch.Tensor,
    *,
    reference: torch.Tensor,
) -> torch.Tensor:
    _require_float_tensor("confidence", confidence)
    if confidence.shape != active_domain.shape or confidence.device != reference.device:
        raise ValueError("confidence must match the active domain on the shared device")
    safe = _safe_on_mask("confidence", confidence, active_domain)
    if bool((active_domain & ((safe < 0) | (safe > 1))).any()):
        raise ValueError("confidence must be within [0,1] on the active domain")
    return safe


def _weighted_mean(
    loss_map: torch.Tensor,
    confidence: torch.Tensor,
    active: torch.Tensor,
    *,
    graph_values: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    safe_loss = _safe_on_mask("loss", loss_map, active)
    weights = torch.where(active, confidence.float(), torch.zeros_like(confidence, dtype=torch.float32))
    denominator = weights.sum()
    if not bool(denominator > 0):
        return _graph_connected_zero(*graph_values)
    return (safe_loss * weights).sum() / denominator


def confidence_weighted_scene_flow_regression(
    prediction: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    alpha: float,
    epsilon: float,
) -> torch.Tensor:
    """Return robust confidence-weighted 3D displacement regression.

    The generalized Charbonnier value is shifted by its zero-residual value,
    so identical finite fields produce an exact scalar zero.
    """

    if prediction.ndim != 5 or prediction.shape[-1] != 3 or target.shape != prediction.shape:
        raise ValueError("prediction and target must match [B,Q,H,W,3]")
    _require_float_tensor("prediction", prediction)
    _require_float_tensor("target", target)
    if target.device != prediction.device:
        raise ValueError("prediction and target must share a device")
    _require_mask("valid_mask", valid_mask, prediction.shape[:-1], prediction.device)
    alpha = _require_scalar("alpha", alpha, minimum=0.0, maximum=1.0, minimum_open=True)
    epsilon = _require_scalar("epsilon", epsilon, minimum=0.0, minimum_open=True)
    safe_confidence = _safe_confidence(confidence, valid_mask, reference=prediction)
    active = valid_mask & (safe_confidence > 0)
    safe_prediction = _safe_on_mask("prediction", prediction, active)
    safe_target = _safe_on_mask("target", target, active)
    squared_error = (safe_prediction - safe_target).square().sum(dim=-1)
    robust = (squared_error + epsilon**2).pow(alpha) - epsilon ** (2 * alpha)
    return _weighted_mean(
        robust,
        safe_confidence,
        active,
        graph_values=(prediction, target),
    )


def tri_state_binary_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    known_mask: torch.Tensor,
    domain_mask: torch.Tensor,
    confidence: torch.Tensor,
) -> torch.Tensor:
    """Class-balanced BCE for ``-1=unknown, 0=false, 1=true`` labels."""

    if logits.ndim != 4:
        raise ValueError("logits must have shape [B,Q,H,W]")
    _require_float_tensor("logits", logits)
    if labels.shape != logits.shape or labels.dtype is not torch.int8 or labels.device != logits.device:
        raise ValueError("labels must be int8 and match logits on the shared device")
    _require_mask("known_mask", known_mask, logits.shape, logits.device)
    _require_mask("domain_mask", domain_mask, logits.shape, logits.device)
    if bool(((labels < -1) | (labels > 1)).any()):
        raise ValueError("labels must contain only -1, 0, or 1")
    if bool((known_mask & (~domain_mask | (labels < 0))).any()):
        raise ValueError("known_mask may select only known labels inside the domain")
    base_active = known_mask & domain_mask
    safe_confidence = _safe_confidence(confidence, base_active, reference=logits)
    active = base_active & (safe_confidence > 0)
    safe_logits = _safe_on_mask("logits", logits, active)
    targets = torch.where(active, labels.float(), torch.zeros_like(logits, dtype=torch.float32))
    loss_map = F.binary_cross_entropy_with_logits(safe_logits, targets, reduction="none")
    class_losses = [
        _weighted_mean(
            loss_map,
            safe_confidence,
            active & (labels == class_label),
            graph_values=(logits,),
        )
        for class_label in (0, 1)
        if bool((active & (labels == class_label)).any())
    ]
    if not class_losses:
        return _graph_connected_zero(logits)
    return torch.stack(class_losses).mean()


def _validate_directed_fields(
    field: torch.Tensor,
    pixel_flow: torch.Tensor,
    pair_indices: torch.Tensor,
    pair_valid_mask: torch.Tensor,
    domain_mask: torch.Tensor,
    confidence: torch.Tensor,
    *,
    channels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if field.ndim != 5 or field.shape[-1] != channels:
        raise ValueError(f"field must have shape [B,Q,H,W,{channels}]")
    _require_float_tensor("field", field)
    expected_flow = (*field.shape[:-1], 2)
    if pixel_flow.shape != expected_flow:
        raise ValueError("pixel_flow must match [B,Q,H,W,2]")
    _require_float_tensor("pixel_flow", pixel_flow)
    batch_size, pair_count, height, width, _ = field.shape
    if pair_indices.shape != (batch_size, pair_count, 2) or pair_indices.dtype is not torch.long:
        raise ValueError("pair_indices must be int64 with shape [B,Q,2]")
    _require_mask(
        "pair_valid_mask",
        pair_valid_mask,
        torch.Size((batch_size, pair_count)),
        field.device,
    )
    _require_mask(
        "domain_mask",
        domain_mask,
        torch.Size((batch_size, pair_count, height, width)),
        field.device,
    )
    values = (pixel_flow, pair_indices, pair_valid_mask, domain_mask, confidence)
    if any(value.device != field.device for value in values):
        raise ValueError("directed fields must share a device")
    valid_pairs = pair_indices[pair_valid_mask]
    invalid_pairs = pair_indices[~pair_valid_mask]
    if valid_pairs.numel() and bool((valid_pairs < 0).any() | (valid_pairs[:, 0] == valid_pairs[:, 1]).any()):
        raise ValueError("valid directed pairs require distinct non-negative frame indices")
    if invalid_pairs.numel() and not bool((invalid_pairs == -1).all()):
        raise ValueError("invalid directed pairs must use the (-1,-1) sentinel")
    for batch_index in range(batch_size):
        rows = pair_indices[batch_index, pair_valid_mask[batch_index]]
        if rows.numel() and len({tuple(row.tolist()) for row in rows}) != rows.shape[0]:
            raise ValueError("valid directed pairs must be unique within each sample")
    pair_domain = pair_valid_mask[:, :, None, None] & domain_mask
    safe_confidence = _safe_confidence(confidence, pair_domain, reference=field)
    active = pair_domain & (safe_confidence > 0)
    _safe_on_mask("field", field, active)
    _safe_on_mask("pixel_flow", pixel_flow, active)
    return safe_confidence, active


def _reverse_pair_offsets(
    pair_indices: torch.Tensor,
    pair_valid_mask: torch.Tensor,
    *,
    policy: MissingReversePolicy,
) -> torch.Tensor:
    if policy not in {"reject", "unknown"}:
        raise ValueError("missing_reverse_policy must be 'reject' or 'unknown'")
    offsets = torch.full_like(pair_valid_mask, -1, dtype=torch.long)
    for batch_index in range(pair_indices.shape[0]):
        lookup = {
            tuple(pair_indices[batch_index, offset].tolist()): offset
            for offset in range(pair_indices.shape[1])
            if bool(pair_valid_mask[batch_index, offset])
        }
        for pair_offset, pair in enumerate(pair_indices[batch_index]):
            if not bool(pair_valid_mask[batch_index, pair_offset]):
                continue
            reverse = lookup.get((int(pair[1]), int(pair[0])))
            if reverse is None:
                if policy == "reject":
                    raise ValueError("every valid directed pair requires a reverse pair")
                continue
            offsets[batch_index, pair_offset] = reverse
    return offsets


def _sampling_grid(pixel_flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = pixel_flow.shape[-3:-1]
    rows = torch.arange(height, device=pixel_flow.device, dtype=torch.float32)
    columns = torch.arange(width, device=pixel_flow.device, dtype=torch.float32)
    vertical, horizontal = torch.meshgrid(rows, columns, indexing="ij")
    target_x = horizontal + pixel_flow[..., 0]
    target_y = vertical + pixel_flow[..., 1]
    grid = torch.stack(
        (
            2 * (target_x + 0.5) / width - 1,
            2 * (target_y + 0.5) / height - 1,
        ),
        dim=-1,
    )
    in_bounds = (target_x >= 0) & (target_x <= width - 1) & (target_y >= 0) & (target_y <= height - 1)
    return grid, in_bounds


def _sample_reverse_on_source_grid(
    field: torch.Tensor,
    pixel_flow: torch.Tensor,
    confidence: torch.Tensor,
    active: torch.Tensor,
    reverse_offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sampled = torch.zeros_like(field, dtype=torch.float32)
    sampled_confidence = torch.zeros_like(confidence, dtype=torch.float32)
    sampled_valid = torch.zeros_like(active)
    safe_field = _safe_on_mask("field", field, active)
    safe_pixel_flow = _safe_on_mask("pixel_flow", pixel_flow, active)
    for batch_index in range(field.shape[0]):
        for pair_offset in range(field.shape[1]):
            reverse_offset = int(reverse_offsets[batch_index, pair_offset])
            if reverse_offset < 0:
                continue
            grid, in_bounds = _sampling_grid(safe_pixel_flow[batch_index, pair_offset])
            reverse_field = safe_field[batch_index, reverse_offset].permute(2, 0, 1)[None]
            reverse_confidence = confidence[batch_index, reverse_offset][None, None]
            reverse_active = active[batch_index, reverse_offset][None, None]
            sampled[batch_index, pair_offset] = F.grid_sample(
                reverse_field,
                grid[None],
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )[0].permute(1, 2, 0)
            sampled_confidence[batch_index, pair_offset] = F.grid_sample(
                reverse_confidence,
                grid[None],
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )[0, 0]
            mask_coverage = F.grid_sample(
                reverse_active.float(),
                grid[None],
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )[0, 0]
            sampled_valid[batch_index, pair_offset] = in_bounds & (mask_coverage >= 1 - 1e-6)
    return sampled, sampled_confidence, sampled_valid


def source_grid_forward_backward_3d_cycle_loss(
    scene_flow: torch.Tensor,
    source_to_target_pixel_flow: torch.Tensor,
    pair_indices: torch.Tensor,
    pair_valid_mask: torch.Tensor,
    *,
    domain_mask: torch.Tensor,
    confidence: torch.Tensor,
    missing_reverse_policy: MissingReversePolicy,
) -> torch.Tensor:
    """Cycle source-aligned 3D flow through a reverse bilinear lookup."""

    safe_confidence, active = _validate_directed_fields(
        scene_flow,
        source_to_target_pixel_flow,
        pair_indices,
        pair_valid_mask,
        domain_mask,
        confidence,
        channels=3,
    )
    reverse_offsets = _reverse_pair_offsets(
        pair_indices,
        pair_valid_mask,
        policy=missing_reverse_policy,
    )
    safe_scene_flow = _safe_on_mask("scene_flow", scene_flow, active)
    sampled_reverse, reverse_confidence, sampled_valid = _sample_reverse_on_source_grid(
        scene_flow,
        source_to_target_pixel_flow,
        safe_confidence,
        active,
        reverse_offsets,
    )
    final_active = active & sampled_valid & (reverse_confidence > 0)
    weights = torch.minimum(safe_confidence, reverse_confidence)
    cycle = (safe_scene_flow + sampled_reverse).abs().mean(dim=-1)
    return _weighted_mean(cycle, weights, final_active, graph_values=(scene_flow,))


def multi_view_reprojection_loss(
    canonical_points_at_target_time: torch.Tensor,
    target_extrinsics_w2c: torch.Tensor,
    target_intrinsics: torch.Tensor,
    target_pixel_xy: torch.Tensor,
    confidence: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Project target-time canonical points and compare target pixel coordinates."""

    points = canonical_points_at_target_time
    if points.ndim != 5 or points.shape[-1] != 3:
        raise ValueError("canonical_points_at_target_time must have shape [B,Q,H,W,3]")
    _require_float_tensor("canonical_points_at_target_time", points)
    batch_size, pair_count, height, width, _ = points.shape
    if target_pixel_xy.shape != (batch_size, pair_count, height, width, 2):
        raise ValueError("target_pixel_xy must have shape [B,Q,H,W,2]")
    if target_extrinsics_w2c.shape != (batch_size, pair_count, 3, 4):
        raise ValueError("target_extrinsics_w2c must have shape [B,Q,3,4]")
    if target_intrinsics.shape != (batch_size, pair_count, 3, 3):
        raise ValueError("target_intrinsics must have shape [B,Q,3,3]")
    for name, value in (
        ("target_pixel_xy", target_pixel_xy),
        ("target_extrinsics_w2c", target_extrinsics_w2c),
        ("target_intrinsics", target_intrinsics),
    ):
        _require_float_tensor(name, value)
        if value.device != points.device:
            raise ValueError("reprojection inputs must share a device")
    _require_mask("valid_mask", valid_mask, points.shape[:-1], points.device)
    safe_confidence = _safe_confidence(confidence, valid_mask, reference=points)
    active = valid_mask & (safe_confidence > 0)
    pair_active = active.any(dim=(-2, -1))
    safe_points = _safe_on_mask("canonical_points_at_target_time", points, active)
    safe_pixels = _safe_on_mask("target_pixel_xy", target_pixel_xy, active)
    safe_extrinsics = _safe_on_mask("target_extrinsics_w2c", target_extrinsics_w2c, pair_active)
    safe_intrinsics = _safe_on_mask("target_intrinsics", target_intrinsics, pair_active)
    rotation = safe_extrinsics[..., :3]
    translation = safe_extrinsics[..., 3]
    camera_points = torch.einsum("bqhwj,bqij->bqhwi", safe_points, rotation) + translation[:, :, None, None]
    homogeneous = torch.einsum("bqhwj,bqij->bqhwi", camera_points, safe_intrinsics)
    z = camera_points[..., 2]
    positive_z = z > torch.finfo(torch.float32).eps
    safe_z = torch.where(active & positive_z, z, torch.ones_like(z))
    projected = homogeneous[..., :2] / safe_z[..., None]
    final_active = active & positive_z
    pixel_error = (projected - safe_pixels).square().sum(dim=-1).sqrt()
    return _weighted_mean(pixel_error, safe_confidence, final_active, graph_values=(points,))


def temporal_target_depth_consistency_loss(
    predicted_target_depth: torch.Tensor,
    target_depth_grid: torch.Tensor,
    source_to_target_pixel_flow: torch.Tensor,
    confidence: torch.Tensor,
    *,
    source_valid_mask: torch.Tensor,
    target_valid_mask: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Compare source-aligned predicted target depth to sampled target depth."""

    if predicted_target_depth.ndim != 4 or target_depth_grid.shape != predicted_target_depth.shape:
        raise ValueError("predicted_target_depth and target_depth_grid must match [B,Q,H,W]")
    _require_float_tensor("predicted_target_depth", predicted_target_depth)
    _require_float_tensor("target_depth_grid", target_depth_grid)
    expected_flow = (*predicted_target_depth.shape, 2)
    if source_to_target_pixel_flow.shape != expected_flow:
        raise ValueError("source_to_target_pixel_flow must have shape [B,Q,H,W,2]")
    _require_float_tensor("source_to_target_pixel_flow", source_to_target_pixel_flow)
    if any(
        value.device != predicted_target_depth.device
        for value in (target_depth_grid, source_to_target_pixel_flow, confidence, source_valid_mask, target_valid_mask)
    ):
        raise ValueError("temporal depth inputs must share a device")
    _require_mask(
        "source_valid_mask",
        source_valid_mask,
        predicted_target_depth.shape,
        predicted_target_depth.device,
    )
    _require_mask(
        "target_valid_mask",
        target_valid_mask,
        predicted_target_depth.shape,
        predicted_target_depth.device,
    )
    epsilon = _require_scalar("epsilon", epsilon, minimum=0.0, minimum_open=True)
    safe_confidence = _safe_confidence(confidence, source_valid_mask, reference=predicted_target_depth)
    active = source_valid_mask & (safe_confidence > 0)
    safe_flow = _safe_on_mask("source_to_target_pixel_flow", source_to_target_pixel_flow, active)
    safe_target = _safe_on_mask("target_depth_grid", target_depth_grid, target_valid_mask)
    if bool((target_valid_mask & (safe_target <= 0)).any()):
        raise ValueError("target_depth_grid must be positive on its valid mask")

    batch_size, pair_count, height, width = predicted_target_depth.shape
    flat_flow = safe_flow.reshape(batch_size * pair_count, height, width, 2)
    grid, in_bounds = _sampling_grid(flat_flow)
    sampled_target = F.grid_sample(
        safe_target.reshape(batch_size * pair_count, 1, height, width),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[:, 0].reshape_as(predicted_target_depth)
    mask_coverage = F.grid_sample(
        target_valid_mask.reshape(batch_size * pair_count, 1, height, width).float(),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[:, 0].reshape_as(predicted_target_depth)
    sampled_valid = in_bounds.reshape_as(predicted_target_depth) & (mask_coverage >= 1 - 1e-6)
    final_active = active & sampled_valid & (sampled_target > 0)
    safe_prediction = _safe_on_mask("predicted_target_depth", predicted_target_depth, final_active)
    if bool((final_active & (safe_prediction <= 0)).any()):
        raise ValueError("predicted_target_depth must be positive on active sampled pixels")
    error = (safe_prediction - sampled_target).abs() / (safe_prediction + sampled_target + epsilon)
    return _weighted_mean(error, safe_confidence, final_active, graph_values=(predicted_target_depth,))


def edge_aware_dynamic_spatial_consistency_loss(
    dynamic_probability: torch.Tensor,
    source_image: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    edge_scale: float,
) -> torch.Tensor:
    """Penalize probability variation except across strong source-image edges."""

    if dynamic_probability.ndim != 4:
        raise ValueError("dynamic_probability must have shape [B,Q,H,W]")
    _require_float_tensor("dynamic_probability", dynamic_probability)
    batch_size, pair_count, height, width = dynamic_probability.shape
    if source_image.shape != (batch_size, pair_count, 3, height, width):
        raise ValueError("source_image must have shape [B,Q,3,H,W]")
    _require_float_tensor("source_image", source_image)
    if source_image.device != dynamic_probability.device:
        raise ValueError("dynamic probability and source image must share a device")
    _require_mask("valid_mask", valid_mask, dynamic_probability.shape, dynamic_probability.device)
    edge_scale = _require_scalar("edge_scale", edge_scale, minimum=0.0)
    safe_probability = _safe_on_mask("dynamic_probability", dynamic_probability, valid_mask)
    if bool((valid_mask & ((safe_probability < 0) | (safe_probability > 1))).any()):
        raise ValueError("dynamic_probability must be within [0,1] on valid pixels")
    safe_image = _safe_on_mask("source_image", source_image, valid_mask[:, :, None])

    horizontal_valid = valid_mask[..., :, :-1] & valid_mask[..., :, 1:]
    vertical_valid = valid_mask[..., :-1, :] & valid_mask[..., 1:, :]
    horizontal_probability = (safe_probability[..., :, 1:] - safe_probability[..., :, :-1]).abs()
    vertical_probability = (safe_probability[..., 1:, :] - safe_probability[..., :-1, :]).abs()
    horizontal_edge = (safe_image[..., :, 1:] - safe_image[..., :, :-1]).abs().mean(dim=2)
    vertical_edge = (safe_image[..., 1:, :] - safe_image[..., :-1, :]).abs().mean(dim=2)
    horizontal_loss = horizontal_probability * torch.exp(-edge_scale * horizontal_edge)
    vertical_loss = vertical_probability * torch.exp(-edge_scale * vertical_edge)
    count = horizontal_valid.sum() + vertical_valid.sum()
    if not bool(count > 0):
        return _graph_connected_zero(dynamic_probability)
    total = torch.where(horizontal_valid, horizontal_loss, torch.zeros_like(horizontal_loss)).sum()
    total = total + torch.where(vertical_valid, vertical_loss, torch.zeros_like(vertical_loss)).sum()
    return total / count.float()


def dynamic_temporal_consistency_loss(
    dynamic_probability: torch.Tensor,
    source_to_target_pixel_flow: torch.Tensor,
    pair_indices: torch.Tensor,
    pair_valid_mask: torch.Tensor,
    *,
    domain_mask: torch.Tensor,
    confidence: torch.Tensor,
    missing_reverse_policy: MissingReversePolicy,
) -> torch.Tensor:
    """Warp reverse-pair dynamic probability onto each source grid."""

    if dynamic_probability.ndim != 4:
        raise ValueError("dynamic_probability must have shape [B,Q,H,W]")
    field = dynamic_probability[..., None]
    safe_confidence, active = _validate_directed_fields(
        field,
        source_to_target_pixel_flow,
        pair_indices,
        pair_valid_mask,
        domain_mask,
        confidence,
        channels=1,
    )
    safe_probability = _safe_on_mask("dynamic_probability", dynamic_probability, active)
    if bool((active & ((safe_probability < 0) | (safe_probability > 1))).any()):
        raise ValueError("dynamic_probability must be within [0,1] on active pixels")
    reverse_offsets = _reverse_pair_offsets(
        pair_indices,
        pair_valid_mask,
        policy=missing_reverse_policy,
    )
    sampled_reverse, reverse_confidence, sampled_valid = _sample_reverse_on_source_grid(
        field,
        source_to_target_pixel_flow,
        safe_confidence,
        active,
        reverse_offsets,
    )
    final_active = active & sampled_valid & (reverse_confidence > 0)
    weights = torch.minimum(safe_confidence, reverse_confidence)
    temporal_error = (safe_probability - sampled_reverse[..., 0]).abs()
    return _weighted_mean(
        temporal_error,
        weights,
        final_active,
        graph_values=(dynamic_probability,),
    )


def bounded_self_supervised_area_prior(
    dynamic_probability: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    minimum_fraction: float,
    maximum_fraction: float,
) -> torch.Tensor:
    """Apply a squared hinge to each pair's predicted dynamic area fraction."""

    if dynamic_probability.ndim != 4:
        raise ValueError("dynamic_probability must have shape [B,Q,H,W]")
    _require_float_tensor("dynamic_probability", dynamic_probability)
    _require_mask("valid_mask", valid_mask, dynamic_probability.shape, dynamic_probability.device)
    minimum_fraction = _require_scalar(
        "minimum_fraction",
        minimum_fraction,
        minimum=0.0,
        maximum=1.0,
        minimum_open=True,
        maximum_open=True,
    )
    maximum_fraction = _require_scalar(
        "maximum_fraction",
        maximum_fraction,
        minimum=0.0,
        maximum=1.0,
        minimum_open=True,
        maximum_open=True,
    )
    if minimum_fraction >= maximum_fraction:
        raise ValueError("minimum_fraction must be smaller than maximum_fraction")
    safe_probability = _safe_on_mask("dynamic_probability", dynamic_probability, valid_mask)
    if bool((valid_mask & ((safe_probability < 0) | (safe_probability > 1))).any()):
        raise ValueError("dynamic_probability must be within [0,1] on valid pixels")
    count = valid_mask.sum(dim=(-2, -1))
    pair_valid = count > 0
    if not bool(pair_valid.any()):
        return _graph_connected_zero(dynamic_probability)
    area = (safe_probability * valid_mask.float()).sum(dim=(-2, -1)) / count.clamp_min(1).float()
    lower = torch.relu(minimum_fraction - area).square()
    upper = torch.relu(area - maximum_fraction).square()
    return (lower + upper)[pair_valid].mean()
