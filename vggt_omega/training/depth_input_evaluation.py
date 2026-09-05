# SPDX-License-Identifier: Apache-2.0
"""Pure masks and sufficient statistics for paired RGB-D evaluation."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations

import torch


@dataclass(frozen=True)
class AvailabilityCase:
    """One deterministic subset of frames receiving mapped-depth input."""

    case_id: str
    provided_frames: int
    mask: tuple[bool, ...]


@dataclass(frozen=True)
class DepthSufficientStatistics:
    """Additive depth-error totals with explicit units and denominators."""

    normalized_absolute_error_sum: float = 0.0
    metric_absolute_error_sum_m: float = 0.0
    near_absolute_error_sum_m: float = 0.0
    valid_pixel_count: int = 0
    near_valid_pixel_count: int = 0

    @property
    def normalized_mae(self) -> float | None:
        return _optional_mean(self.normalized_absolute_error_sum, self.valid_pixel_count)

    @property
    def metric_mae_m(self) -> float | None:
        return _optional_mean(self.metric_absolute_error_sum_m, self.valid_pixel_count)

    @property
    def near_mae_m(self) -> float | None:
        return _optional_mean(self.near_absolute_error_sum_m, self.near_valid_pixel_count)


@dataclass(frozen=True)
class PairedDepthStatistics:
    """Baseline and candidate totals evaluated on one shared mask."""

    baseline: DepthSufficientStatistics
    candidate: DepthSufficientStatistics


@dataclass(frozen=True)
class InputDepthHoldout:
    """Conditioning tensors with selected known pixels hidden from the input."""

    depth: torch.Tensor
    visible_mask: torch.Tensor
    holdout_mask: torch.Tensor


def all_depth_availability_cases(num_frames: int) -> tuple[AvailabilityCase, ...]:
    """Enumerate all frame subsets, grouped by their provided-frame count."""

    if isinstance(num_frames, bool) or not isinstance(num_frames, int) or num_frames < 1:
        raise ValueError("num_frames must be a positive integer")
    cases: list[AvailabilityCase] = []
    for provided_frames in range(num_frames + 1):
        for selected in combinations(range(num_frames), provided_frames):
            mask = tuple(index in selected for index in range(num_frames))
            bits = "".join("1" if value else "0" for value in mask)
            cases.append(AvailabilityCase(f"k{provided_frames}-{bits}", provided_frames, mask))
    return tuple(cases)


def depth_sufficient_statistics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    normalization_scale_m: torch.Tensor,
    *,
    max_near_depth_m: float = 1.2,
) -> DepthSufficientStatistics:
    """Return pixel-additive normalized, metric, and near-depth L1 totals."""

    _validate_depth_inputs(prediction, target, valid_mask, normalization_scale_m)
    if not math.isfinite(max_near_depth_m) or max_near_depth_m <= 0:
        raise ValueError("max_near_depth_m must be finite and positive")
    scalar_prediction = prediction[..., 0].float()
    float_target = target.float()
    scale = normalization_scale_m.float().reshape(target.shape[0], 1, 1, 1)
    metric_target = float_target * scale
    normalized_error = (scalar_prediction - float_target).abs()
    metric_error = normalized_error * scale
    near_mask = valid_mask & (metric_target > 0) & (metric_target < max_near_depth_m)
    return DepthSufficientStatistics(
        normalized_absolute_error_sum=_masked_sum(normalized_error, valid_mask),
        metric_absolute_error_sum_m=_masked_sum(metric_error, valid_mask),
        near_absolute_error_sum_m=_masked_sum(metric_error, near_mask),
        valid_pixel_count=int(valid_mask.sum().item()),
        near_valid_pixel_count=int(near_mask.sum().item()),
    )


def paired_depth_statistics(
    baseline_prediction: torch.Tensor,
    candidate_prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    normalization_scale_m: torch.Tensor,
    *,
    max_near_depth_m: float = 1.2,
) -> PairedDepthStatistics:
    """Score two predictions against exactly the same target pixels."""

    if baseline_prediction.shape != candidate_prediction.shape:
        raise ValueError("baseline and candidate predictions must have identical shapes")
    return PairedDepthStatistics(
        baseline=depth_sufficient_statistics(
            baseline_prediction,
            target,
            valid_mask,
            normalization_scale_m,
            max_near_depth_m=max_near_depth_m,
        ),
        candidate=depth_sufficient_statistics(
            candidate_prediction,
            target,
            valid_mask,
            normalization_scale_m,
            max_near_depth_m=max_near_depth_m,
        ),
    )


def merge_depth_statistics(items: Iterable[DepthSufficientStatistics]) -> DepthSufficientStatistics:
    """Merge arbitrary batch partitions without averaging batch means."""

    normalized_sum = metric_sum = near_sum = 0.0
    valid_count = near_count = 0
    for item in items:
        if not isinstance(item, DepthSufficientStatistics):
            raise TypeError("items must contain DepthSufficientStatistics")
        normalized_sum += item.normalized_absolute_error_sum
        metric_sum += item.metric_absolute_error_sum_m
        near_sum += item.near_absolute_error_sum_m
        valid_count += item.valid_pixel_count
        near_count += item.near_valid_pixel_count
    return DepthSufficientStatistics(normalized_sum, metric_sum, near_sum, valid_count, near_count)


def metric_result(error_sum: float, count: int) -> dict[str, float | int | str | None]:
    """Render an additive metric without representing empty subsets as zero error."""

    if not math.isfinite(error_sum) or error_sum < 0:
        raise ValueError("error_sum must be finite and non-negative")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("count must be a non-negative integer")
    if count == 0:
        if error_sum != 0:
            raise ValueError("empty metrics cannot have a non-zero error sum")
        return {"value": None, "count": 0, "reason": "not_applicable"}
    return {"value": error_sum / count, "count": count}


def build_input_depth_holdout(
    depth: torch.Tensor,
    valid_mask: torch.Tensor,
    frame_ids: torch.Tensor,
    *,
    patch_size: int = 16,
    divisor: int = 5,
) -> InputDepthHoldout:
    """Hide deterministic patch tiles in the input while preserving target tensors."""

    if not isinstance(depth, torch.Tensor) or depth.ndim != 5 or depth.shape[2] != 1:
        raise ValueError("depth must have shape [B,S,1,H,W]")
    if not depth.is_floating_point():
        raise TypeError("depth must use a floating-point dtype")
    if not isinstance(valid_mask, torch.Tensor) or valid_mask.shape != depth.shape or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be bool and exactly match depth")
    if not isinstance(frame_ids, torch.Tensor) or frame_ids.shape != depth.shape[:2]:
        raise ValueError("frame_ids must have shape [B,S]")
    if frame_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TypeError("frame_ids must use an integer dtype")
    if valid_mask.device != depth.device or frame_ids.device != depth.device:
        raise ValueError("depth, valid_mask, and frame_ids must use the same device")
    if isinstance(patch_size, bool) or not isinstance(patch_size, int) or patch_size < 1:
        raise ValueError("patch_size must be a positive integer")
    if isinstance(divisor, bool) or not isinstance(divisor, int) or divisor < 2:
        raise ValueError("divisor must be an integer of at least two")
    height, width = depth.shape[-2:]
    if height % patch_size or width % patch_size:
        raise ValueError("depth dimensions must be divisible by patch_size")
    observed_depth = depth[valid_mask]
    if not torch.isfinite(observed_depth).all() or torch.any(observed_depth <= 0):
        raise ValueError("valid depth values must be finite and positive")

    patch_y = torch.arange(height, device=depth.device).div(patch_size, rounding_mode="floor")
    patch_x = torch.arange(width, device=depth.device).div(patch_size, rounding_mode="floor")
    tile_phase = patch_y[:, None] + 2 * patch_x[None, :]
    selected_tiles = (tile_phase[None, None] + frame_ids[:, :, None, None]) % divisor == 0
    holdout_mask = valid_mask & selected_tiles[:, :, None]
    visible_mask = valid_mask & ~holdout_mask
    frame_axes = (2, 3, 4)
    if torch.any(visible_mask.sum(dim=frame_axes) == 0):
        raise ValueError("every frame must contain visible valid depth")
    if torch.any(holdout_mask.sum(dim=frame_axes) == 0):
        raise ValueError("every frame must contain holdout valid depth")
    input_depth = torch.where(visible_mask, depth, torch.zeros_like(depth))
    return InputDepthHoldout(input_depth, visible_mask, holdout_mask)


def _validate_depth_inputs(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    normalization_scale_m: torch.Tensor,
) -> None:
    if not isinstance(target, torch.Tensor) or target.ndim != 4:
        raise ValueError("target must have shape [B,S,H,W]")
    if not isinstance(prediction, torch.Tensor) or prediction.shape != (*target.shape, 1):
        raise ValueError("prediction must have target shape plus a final singleton channel")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("prediction and target must use floating-point dtypes")
    if not isinstance(valid_mask, torch.Tensor) or valid_mask.shape != target.shape or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be bool and exactly match target")
    if not isinstance(normalization_scale_m, torch.Tensor) or normalization_scale_m.numel() != target.shape[0]:
        raise ValueError("normalization_scale_m must contain one value per sample")
    if any(value.device != target.device for value in (prediction, valid_mask, normalization_scale_m)):
        raise ValueError("all depth metric tensors must use the same device")
    scale = normalization_scale_m.float()
    if not torch.isfinite(scale).all() or torch.any(scale <= 0):
        raise ValueError("normalization_scale_m must be finite and positive")
    if not torch.isfinite(prediction[..., 0][valid_mask]).all() or not torch.isfinite(target[valid_mask]).all():
        raise ValueError("evaluated prediction and target values must be finite")


def _masked_sum(values: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any()):
        return 0.0
    return float(values[mask].double().sum().item())


def _optional_mean(total: float, count: int) -> float | None:
    result = metric_result(total, count)
    value = result["value"]
    return None if value is None else float(value)
