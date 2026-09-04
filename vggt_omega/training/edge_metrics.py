"""Edge-aware camera-space depth metrics.

This is deliberately named a proxy: the PPD paper describes a Canny/Chamfer
evaluation but its exact evaluation implementation and thresholds are not
published in the inspected repository.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def edge_3d_error_proxy(
    prediction: torch.Tensor,
    target: torch.Tensor,
    intrinsics: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    max_near_depth_m: float,
    relative_edge_threshold: float = 0.05,
    edge_radius: int = 1,
) -> dict[str, float]:
    """Measure per-pixel 3D error around valid GT depth discontinuities."""

    if prediction.ndim != 3 or target.shape != prediction.shape:
        raise ValueError("prediction and target must have matching [B,H,W] shapes")
    batch_size, height, width = prediction.shape
    if intrinsics.shape != (batch_size, 3, 3):
        raise ValueError("intrinsics must have shape [B,3,3]")
    if valid_mask.shape != prediction.shape or valid_mask.dtype is not torch.bool:
        raise ValueError("valid_mask must be bool and match depth shape")
    for name, value in (("prediction", prediction), ("target", target), ("intrinsics", intrinsics)):
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite and floating point")
    if (
        valid_mask.device != prediction.device
        or target.device != prediction.device
        or intrinsics.device != prediction.device
    ):
        raise ValueError("all inputs must share a device")
    if torch.any(prediction[valid_mask] <= 0) or torch.any(target[valid_mask] <= 0):
        raise ValueError("depth must be positive on valid pixels")
    if not math.isfinite(max_near_depth_m) or max_near_depth_m <= 0:
        raise ValueError("max_near_depth_m must be finite and positive")
    if not math.isfinite(relative_edge_threshold) or relative_edge_threshold <= 0:
        raise ValueError("relative_edge_threshold must be finite and positive")
    if isinstance(edge_radius, bool) or not isinstance(edge_radius, int) or edge_radius < 0:
        raise ValueError("edge_radius must be a non-negative integer")

    usable = valid_mask & (target > 0) & (prediction > 0)
    edge_mask = torch.zeros_like(usable)
    horizontal_pair = usable[..., :, 1:] & usable[..., :, :-1]
    horizontal_jump = (target[..., :, 1:] - target[..., :, :-1]).abs() / torch.minimum(
        target[..., :, 1:], target[..., :, :-1]
    ).clamp_min(torch.finfo(target.dtype).eps)
    horizontal_edge = horizontal_pair & (horizontal_jump > relative_edge_threshold)
    edge_mask[..., :, 1:] |= horizontal_edge
    edge_mask[..., :, :-1] |= horizontal_edge
    vertical_pair = usable[..., 1:, :] & usable[..., :-1, :]
    vertical_jump = (target[..., 1:, :] - target[..., :-1, :]).abs() / torch.minimum(
        target[..., 1:, :], target[..., :-1, :]
    ).clamp_min(torch.finfo(target.dtype).eps)
    vertical_edge = vertical_pair & (vertical_jump > relative_edge_threshold)
    edge_mask[..., 1:, :] |= vertical_edge
    edge_mask[..., :-1, :] |= vertical_edge
    if edge_radius:
        kernel_size = 2 * edge_radius + 1
        edge_mask = F.max_pool2d(
            edge_mask[:, None].float(),
            kernel_size=kernel_size,
            stride=1,
            padding=edge_radius,
        )[:, 0].bool()
    edge_mask &= usable

    rows = torch.arange(height, dtype=target.dtype, device=target.device)
    columns = torch.arange(width, dtype=target.dtype, device=target.device)
    vertical, horizontal = torch.meshgrid(rows, columns, indexing="ij")
    pixels = torch.stack((horizontal, vertical, torch.ones_like(horizontal)), dim=0)
    pixels = pixels.reshape(1, 3, height * width).expand(batch_size, -1, -1)
    rays = torch.linalg.solve(intrinsics, pixels).transpose(1, 2).reshape(batch_size, height, width, 3)
    point_error = (rays * (prediction - target).unsqueeze(-1)).norm(dim=-1)

    metrics: dict[str, float] = {}
    near = usable & (target < max_near_depth_m)
    for prefix, scope in (("all", usable), ("near", near)):
        scope_count = int(scope.sum())
        scope_edge = scope & edge_mask
        scope_non_edge = scope & ~edge_mask
        edge_count = int(scope_edge.sum())
        metrics[f"{prefix}_valid_pixels"] = float(scope_count)
        metrics[f"{prefix}_edge_pixels"] = float(edge_count)
        metrics[f"{prefix}_edge_coverage"] = edge_count / scope_count if scope_count else 0.0
        metrics[f"{prefix}_3d_error"] = _masked_mean(point_error, scope)
        metrics[f"{prefix}_edge_3d_error_proxy"] = _masked_mean(point_error, scope_edge)
        metrics[f"{prefix}_non_edge_3d_error"] = _masked_mean(point_error, scope_non_edge)
    return metrics


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any()):
        return 0.0
    return float(values[mask].double().mean())
