"""Pure RGB-D view-overlap geometry for curriculum sampling."""

from __future__ import annotations

import torch


def _validate_inputs(
    source_depth: torch.Tensor,
    target_depth: torch.Tensor,
    source_intrinsics: torch.Tensor,
    target_intrinsics: torch.Tensor,
    source_extrinsics_w2c: torch.Tensor,
    target_extrinsics_w2c: torch.Tensor,
    *,
    relative_depth_tolerance: float,
    pixel_stride: int,
    max_depth_m: float | None,
) -> None:
    if source_depth.ndim != 2 or target_depth.ndim != 2:
        raise ValueError("source_depth and target_depth must be rank-two tensors")
    if source_depth.device != target_depth.device:
        raise ValueError("source_depth and target_depth must be on the same device")
    if not source_depth.is_floating_point() or not target_depth.is_floating_point():
        raise ValueError("source_depth and target_depth must be floating-point tensors")
    for name, value in (
        ("source_intrinsics", source_intrinsics),
        ("target_intrinsics", target_intrinsics),
    ):
        if value.shape != (3, 3) or not torch.isfinite(value).all():
            raise ValueError(f"{name} must be a finite 3x3 tensor")
    for name, value in (
        ("source_extrinsics_w2c", source_extrinsics_w2c),
        ("target_extrinsics_w2c", target_extrinsics_w2c),
    ):
        if value.shape != (3, 4) or not torch.isfinite(value).all():
            raise ValueError(f"{name} must be a finite 3x4 tensor")
    if relative_depth_tolerance <= 0:
        raise ValueError("relative_depth_tolerance must be positive")
    if not isinstance(pixel_stride, int) or isinstance(pixel_stride, bool) or pixel_stride < 1:
        raise ValueError("pixel_stride must be a positive integer")
    if max_depth_m is not None and max_depth_m <= 0:
        raise ValueError("max_depth_m must be positive when provided")


def directional_rgbd_overlap(
    source_depth: torch.Tensor,
    target_depth: torch.Tensor,
    source_intrinsics: torch.Tensor,
    target_intrinsics: torch.Tensor,
    source_extrinsics_w2c: torch.Tensor,
    target_extrinsics_w2c: torch.Tensor,
    *,
    relative_depth_tolerance: float = 0.03,
    pixel_stride: int = 1,
    max_depth_m: float | None = None,
) -> torch.Tensor:
    """Return the visible fraction of valid source depth in a target view.

    Extrinsics follow OpenCV world-to-camera convention. A source point is a
    match only when it projects inside the target and agrees with target Z
    depth within ``relative_depth_tolerance``. Empty source support is defined
    as a finite zero.
    """

    _validate_inputs(
        source_depth,
        target_depth,
        source_intrinsics,
        target_intrinsics,
        source_extrinsics_w2c,
        target_extrinsics_w2c,
        relative_depth_tolerance=relative_depth_tolerance,
        pixel_stride=pixel_stride,
        max_depth_m=max_depth_m,
    )
    device = source_depth.device
    dtype = source_depth.dtype
    source_intrinsics = source_intrinsics.to(device=device, dtype=dtype)
    target_intrinsics = target_intrinsics.to(device=device, dtype=dtype)
    source_extrinsics_w2c = source_extrinsics_w2c.to(device=device, dtype=dtype)
    target_extrinsics_w2c = target_extrinsics_w2c.to(device=device, dtype=dtype)

    source_height, source_width = source_depth.shape
    target_height, target_width = target_depth.shape
    rows = torch.arange(0, source_height, pixel_stride, device=device)
    columns = torch.arange(0, source_width, pixel_stride, device=device)
    vertical, horizontal = torch.meshgrid(rows, columns, indexing="ij")
    sampled_depth = source_depth[vertical, horizontal]
    source_valid = torch.isfinite(sampled_depth) & (sampled_depth > 0)
    if max_depth_m is not None:
        source_valid &= sampled_depth < max_depth_m
    source_count = source_valid.count_nonzero()
    if not bool(source_count):
        return torch.nan_to_num(source_depth).sum() * 0.0

    pixels = torch.stack(
        (
            horizontal.to(dtype=dtype),
            vertical.to(dtype=dtype),
            torch.ones_like(horizontal, dtype=dtype),
        ),
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
    target_z = target_points[..., 2]
    projected = target_points @ target_intrinsics.T
    safe_z = torch.where(target_z.abs() > torch.finfo(dtype).eps, target_z, torch.ones_like(target_z))
    target_u = projected[..., 0] / safe_z
    target_v = projected[..., 1] / safe_z
    target_x = target_u.round().to(torch.int64)
    target_y = target_v.round().to(torch.int64)
    in_bounds = (
        torch.isfinite(target_u)
        & torch.isfinite(target_v)
        & (target_z > 0)
        & (target_x >= 0)
        & (target_x < target_width)
        & (target_y >= 0)
        & (target_y < target_height)
    )
    safe_x = target_x.clamp(0, target_width - 1)
    safe_y = target_y.clamp(0, target_height - 1)
    observed_target_depth = target_depth[safe_y, safe_x]
    target_valid = torch.isfinite(observed_target_depth) & (observed_target_depth > 0)
    if max_depth_m is not None:
        target_valid &= observed_target_depth < max_depth_m
    relative_error = (observed_target_depth - target_z).abs() / observed_target_depth.clamp_min(torch.finfo(dtype).eps)
    matches = source_valid & in_bounds & target_valid & (relative_error <= relative_depth_tolerance)
    return matches.count_nonzero().to(dtype=dtype) / source_count.to(dtype=dtype)


def bidirectional_rgbd_overlap(
    first_depth: torch.Tensor,
    second_depth: torch.Tensor,
    first_intrinsics: torch.Tensor,
    second_intrinsics: torch.Tensor,
    first_extrinsics_w2c: torch.Tensor,
    second_extrinsics_w2c: torch.Tensor,
    *,
    relative_depth_tolerance: float = 0.03,
    pixel_stride: int = 1,
    max_depth_m: float | None = None,
) -> torch.Tensor:
    """Return the arithmetic mean of both directional RGB-D overlaps."""

    options = {
        "relative_depth_tolerance": relative_depth_tolerance,
        "pixel_stride": pixel_stride,
        "max_depth_m": max_depth_m,
    }
    forward = directional_rgbd_overlap(
        first_depth,
        second_depth,
        first_intrinsics,
        second_intrinsics,
        first_extrinsics_w2c,
        second_extrinsics_w2c,
        **options,
    )
    backward = directional_rgbd_overlap(
        second_depth,
        first_depth,
        second_intrinsics,
        first_intrinsics,
        second_extrinsics_w2c,
        first_extrinsics_w2c,
        **options,
    )
    return (forward + backward) / 2
