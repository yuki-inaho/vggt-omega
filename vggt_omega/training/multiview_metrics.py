"""Visibility-aware multi-view depth consistency metrics."""

from __future__ import annotations

import math

import torch


def directional_depth_consistency(
    source_depth: torch.Tensor,
    target_depth: torch.Tensor,
    source_intrinsics: torch.Tensor,
    target_intrinsics: torch.Tensor,
    source_extrinsics_w2c: torch.Tensor,
    target_extrinsics_w2c: torch.Tensor,
    *,
    source_mask: torch.Tensor | None = None,
    target_mask: torch.Tensor | None = None,
    max_depth_m: float | None = None,
    occlusion_tolerance: float = 0.03,
    pixel_stride: int = 1,
) -> dict[str, float]:
    """Project source depth into one target and report error plus coverage."""

    if source_depth.ndim != 2 or target_depth.ndim != 2:
        raise ValueError("source_depth and target_depth must be rank two")
    if not source_depth.is_floating_point() or not target_depth.is_floating_point():
        raise ValueError("depths must be floating point")
    if source_depth.device != target_depth.device:
        raise ValueError("depths must share a device")
    for name, value, shape in (
        ("source_intrinsics", source_intrinsics, (3, 3)),
        ("target_intrinsics", target_intrinsics, (3, 3)),
        ("source_extrinsics_w2c", source_extrinsics_w2c, (3, 4)),
        ("target_extrinsics_w2c", target_extrinsics_w2c, (3, 4)),
    ):
        if value.shape != shape or not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite floating-point with shape {shape}")
        if value.device != source_depth.device:
            raise ValueError("depths and camera tensors must share a device")
    if not math.isfinite(occlusion_tolerance) or occlusion_tolerance < 0:
        raise ValueError("occlusion_tolerance must be finite and non-negative")
    if isinstance(pixel_stride, bool) or not isinstance(pixel_stride, int) or pixel_stride < 1:
        raise ValueError("pixel_stride must be a positive integer")
    if max_depth_m is not None and (not math.isfinite(max_depth_m) or max_depth_m <= 0):
        raise ValueError("max_depth_m must be finite and positive when provided")
    source_mask = _depth_mask("source_mask", source_mask, source_depth)
    target_mask = _depth_mask("target_mask", target_mask, target_depth)

    source_height, source_width = source_depth.shape
    target_height, target_width = target_depth.shape
    rows = torch.arange(0, source_height, pixel_stride, device=source_depth.device)
    columns = torch.arange(0, source_width, pixel_stride, device=source_depth.device)
    vertical, horizontal = torch.meshgrid(rows, columns, indexing="ij")
    sampled_depth = source_depth[vertical, horizontal]
    sampled_mask = source_mask[vertical, horizontal]
    source_valid = sampled_mask & torch.isfinite(sampled_depth) & (sampled_depth > 0)
    if max_depth_m is not None:
        source_valid &= sampled_depth < max_depth_m
    source_count = int(source_valid.sum())
    if source_count == 0:
        return _empty_direction(source_count)

    dtype = source_depth.dtype
    pixels = torch.stack(
        (horizontal.to(dtype), vertical.to(dtype), torch.ones_like(horizontal, dtype=dtype)),
        dim=-1,
    )
    rays = torch.linalg.solve(source_intrinsics, pixels.reshape(-1, 3).T).T.reshape_as(pixels)
    source_points = rays * torch.nan_to_num(sampled_depth).unsqueeze(-1)
    source_rotation = source_extrinsics_w2c[:, :3]
    source_translation = source_extrinsics_w2c[:, 3]
    world_points = (source_points - source_translation) @ source_rotation
    target_rotation = target_extrinsics_w2c[:, :3]
    target_translation = target_extrinsics_w2c[:, 3]
    target_points = world_points @ target_rotation.T + target_translation
    projected_z = target_points[..., 2]
    homogeneous = target_points @ target_intrinsics.T
    safe_z = torch.where(projected_z.abs() > torch.finfo(dtype).eps, projected_z, torch.ones_like(projected_z))
    projected_x = (homogeneous[..., 0] / safe_z).round().to(torch.int64)
    projected_y = (homogeneous[..., 1] / safe_z).round().to(torch.int64)
    in_bounds = (
        (projected_z > 0)
        & (projected_x >= 0)
        & (projected_x < target_width)
        & (projected_y >= 0)
        & (projected_y < target_height)
    )
    safe_x = projected_x.clamp(0, target_width - 1)
    safe_y = projected_y.clamp(0, target_height - 1)
    observed_depth = target_depth[safe_y, safe_x]
    observed_valid = target_mask[safe_y, safe_x] & torch.isfinite(observed_depth) & (observed_depth > 0)
    if max_depth_m is not None:
        observed_valid &= observed_depth < max_depth_m
    not_occluded = projected_z <= observed_depth * (1 + occlusion_tolerance)
    visible = source_valid & in_bounds & observed_valid & not_occluded
    visible_count = int(visible.sum())
    if visible_count == 0:
        return _empty_direction(source_count)
    absolute_error = (projected_z - observed_depth).abs()
    relative_error = absolute_error / observed_depth.clamp_min(torch.finfo(dtype).eps)
    return {
        "depth_error": float(absolute_error[visible].double().mean()),
        "relative_error": float(relative_error[visible].double().mean()),
        "coverage": visible_count / source_count,
        "visible_points": float(visible_count),
        "source_points": float(source_count),
    }


def sequence_multiview_consistency(
    depths: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics_w2c: torch.Tensor,
    *,
    valid_mask: torch.Tensor,
    dynamic_mask: torch.Tensor | None = None,
    frame_mask: torch.Tensor | None = None,
    max_depth_m: float | None = None,
    occlusion_tolerance: float = 0.03,
    pixel_stride: int = 1,
) -> dict[str, float]:
    """Aggregate both directions of every valid unordered frame pair."""

    if depths.ndim != 4:
        raise ValueError("depths must have shape [B,S,H,W]")
    batch_size, frame_count, _, _ = depths.shape
    if intrinsics.shape != (batch_size, frame_count, 3, 3):
        raise ValueError("intrinsics must have shape [B,S,3,3]")
    if extrinsics_w2c.shape != (batch_size, frame_count, 3, 4):
        raise ValueError("extrinsics_w2c must have shape [B,S,3,4]")
    if valid_mask.shape != depths.shape or valid_mask.dtype is not torch.bool:
        raise ValueError("valid_mask must be bool and match depths")
    if dynamic_mask is None:
        dynamic_mask = torch.zeros_like(valid_mask)
    if dynamic_mask.shape != depths.shape or dynamic_mask.dtype is not torch.bool:
        raise ValueError("dynamic_mask must be bool and match depths")
    if frame_mask is None:
        frame_mask = torch.ones((batch_size, frame_count), dtype=torch.bool, device=depths.device)
    if frame_mask.shape != (batch_size, frame_count) or frame_mask.dtype is not torch.bool:
        raise ValueError("frame_mask must be bool with shape [B,S]")
    if any(
        value.device != depths.device for value in (intrinsics, extrinsics_w2c, valid_mask, dynamic_mask, frame_mask)
    ):
        raise ValueError("all sequence tensors must share a device")

    static_valid = valid_mask & ~dynamic_mask
    directions: list[dict[str, float]] = []
    pair_count = 0
    for batch_index in range(batch_size):
        valid_frames = torch.nonzero(frame_mask[batch_index], as_tuple=False).flatten().tolist()
        for offset, first in enumerate(valid_frames):
            for second in valid_frames[offset + 1 :]:
                pair_count += 1
                for source, target in ((first, second), (second, first)):
                    directions.append(
                        directional_depth_consistency(
                            depths[batch_index, source],
                            depths[batch_index, target],
                            intrinsics[batch_index, source],
                            intrinsics[batch_index, target],
                            extrinsics_w2c[batch_index, source],
                            extrinsics_w2c[batch_index, target],
                            source_mask=static_valid[batch_index, source],
                            target_mask=static_valid[batch_index, target],
                            max_depth_m=max_depth_m,
                            occlusion_tolerance=occlusion_tolerance,
                            pixel_stride=pixel_stride,
                        )
                    )
    direction_count = len(directions)
    visible_directions = [value for value in directions if value["visible_points"] > 0]
    total_visible = sum(value["visible_points"] for value in visible_directions)
    weighted_error = sum(value["depth_error"] * value["visible_points"] for value in visible_directions)
    weighted_relative = sum(value["relative_error"] * value["visible_points"] for value in visible_directions)
    return {
        "directional_depth_error": sum(value["depth_error"] for value in directions) / direction_count
        if direction_count
        else 0.0,
        "symmetric_depth_error": weighted_error / total_visible if total_visible else 0.0,
        "symmetric_relative_error": weighted_relative / total_visible if total_visible else 0.0,
        "symmetric_coverage": sum(value["coverage"] for value in directions) / direction_count
        if direction_count
        else 0.0,
        "pair_count": float(pair_count),
        "direction_count": float(direction_count),
        "visible_direction_count": float(len(visible_directions)),
    }


def _depth_mask(name: str, value: torch.Tensor | None, depth: torch.Tensor) -> torch.Tensor:
    if value is None:
        return torch.ones_like(depth, dtype=torch.bool)
    if value.shape != depth.shape or value.dtype is not torch.bool or value.device != depth.device:
        raise ValueError(f"{name} must be bool and match its depth tensor")
    return value


def _empty_direction(source_count: int) -> dict[str, float]:
    return {
        "depth_error": 0.0,
        "relative_error": 0.0,
        "coverage": 0.0,
        "visible_points": 0.0,
        "source_points": float(source_count),
    }
