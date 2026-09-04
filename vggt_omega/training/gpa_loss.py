"""Clean-room geometry/physics-aware losses inspired by GPA-VGGT.

The paper specifies the loss equations but does not publish its training code
or coefficient values.  Callers therefore provide every paper coefficient via
configuration; defaults here are deliberately absent from the public API.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def sample_gpa_anchor_indices(
    frame_mask: torch.Tensor,
    *,
    anchor_count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample unique anchor frames independently for every sequence."""

    if frame_mask.ndim != 2 or frame_mask.dtype is not torch.bool:
        raise ValueError("frame_mask must be bool with shape [B,S]")
    if isinstance(anchor_count, bool) or not isinstance(anchor_count, int) or anchor_count < 1:
        raise ValueError("anchor_count must be a positive integer")
    selected: list[torch.Tensor] = []
    for sample_mask in frame_mask:
        candidates = torch.nonzero(sample_mask, as_tuple=False).flatten()
        if len(candidates) < anchor_count:
            raise ValueError("each sequence must contain at least anchor_count valid frames")
        order = torch.randperm(len(candidates), generator=generator, device=frame_mask.device)
        selected.append(candidates[order[:anchor_count]])
    return torch.stack(selected)


def sample_temporal_window_indices(
    *,
    frame_count: int,
    window_size: int,
    stride_options: tuple[int, ...],
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample one fixed-stride window without silently truncating it."""

    for name, value in (("frame_count", frame_count), ("window_size", window_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not stride_options or any(
        isinstance(stride, bool) or not isinstance(stride, int) or stride < 1 for stride in stride_options
    ):
        raise ValueError("stride_options must contain positive integers")
    valid_strides = tuple(stride for stride in stride_options if (window_size - 1) * stride < frame_count)
    if not valid_strides:
        raise ValueError("no temporal stride can fit the requested window")
    generator_device = generator.device
    stride_offset = int(torch.randint(len(valid_strides), (), generator=generator, device=generator_device))
    stride = valid_strides[stride_offset]
    maximum_start = frame_count - 1 - (window_size - 1) * stride
    start = int(torch.randint(maximum_start + 1, (), generator=generator, device=generator_device))
    return start + torch.arange(window_size, device=generator_device) * stride


def transform_intrinsics_for_image_affine(
    intrinsics: torch.Tensor,
    image_affine: torch.Tensor,
) -> torch.Tensor:
    """Update pixel-space intrinsics after applying ``image_affine`` to images."""

    if intrinsics.shape[-2:] != (3, 3) or image_affine.shape[-2:] != (3, 3):
        raise ValueError("intrinsics and image_affine must end in [3,3]")
    if not intrinsics.is_floating_point() or not image_affine.is_floating_point():
        raise ValueError("intrinsics and image_affine must be floating point")
    if not torch.isfinite(intrinsics).all() or not torch.isfinite(image_affine).all():
        raise ValueError("intrinsics and image_affine must be finite")
    if intrinsics.device != image_affine.device:
        raise ValueError("intrinsics and image_affine must share a device")
    try:
        return torch.matmul(image_affine, intrinsics)
    except RuntimeError as error:
        raise ValueError("image_affine leading dimensions are not broadcastable to intrinsics") from error


def _require_scalar(name: str, value: float, *, minimum: float, maximum: float | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite scalar")
    if value < minimum or (maximum is not None and value > maximum):
        interval = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{name} must be {interval}")


def _photometric_cost(target: torch.Tensor, reconstructed: torch.Tensor, *, mu: float) -> torch.Tensor:
    l1 = (target - reconstructed).abs().mean(dim=1)
    c1 = 0.01**2
    c2 = 0.03**2
    mean_target = F.avg_pool2d(target, 3, stride=1, padding=1)
    mean_reconstructed = F.avg_pool2d(reconstructed, 3, stride=1, padding=1)
    variance_target = F.avg_pool2d(target.square(), 3, stride=1, padding=1) - mean_target.square()
    variance_reconstructed = F.avg_pool2d(reconstructed.square(), 3, stride=1, padding=1) - mean_reconstructed.square()
    covariance = F.avg_pool2d(target * reconstructed, 3, stride=1, padding=1) - (mean_target * mean_reconstructed)
    numerator = (2 * mean_target * mean_reconstructed + c1) * (2 * covariance + c2)
    denominator = (mean_target.square() + mean_reconstructed.square() + c1) * (
        variance_target + variance_reconstructed + c2
    )
    ssim = numerator / denominator.clamp_min(torch.finfo(target.dtype).eps)
    dissimilarity = ((1 - ssim) / 2).clamp(0, 1).mean(dim=1)
    return mu * dissimilarity + (1 - mu) * l1


def _validate_pair(
    target_image: torch.Tensor,
    source_image: torch.Tensor,
    target_depth: torch.Tensor,
    source_depth: torch.Tensor,
    target_intrinsics: torch.Tensor,
    source_intrinsics: torch.Tensor,
    target_extrinsics_w2c: torch.Tensor,
    source_extrinsics_w2c: torch.Tensor,
) -> tuple[int, int, int]:
    if target_image.ndim != 4 or target_image.shape[1] != 3 or source_image.shape != target_image.shape:
        raise ValueError("target_image and source_image must match [B,3,H,W]")
    batch_size, _, height, width = target_image.shape
    if target_depth.shape != (batch_size, height, width) or source_depth.shape != target_depth.shape:
        raise ValueError("target_depth and source_depth must match [B,H,W]")
    if target_intrinsics.shape != (batch_size, 3, 3) or source_intrinsics.shape != target_intrinsics.shape:
        raise ValueError("intrinsics must match [B,3,3]")
    expected_extrinsics = (batch_size, 3, 4)
    if target_extrinsics_w2c.shape != expected_extrinsics or source_extrinsics_w2c.shape != expected_extrinsics:
        raise ValueError("extrinsics must match [B,3,4]")
    tensors = (
        target_image,
        source_image,
        target_depth,
        source_depth,
        target_intrinsics,
        source_intrinsics,
        target_extrinsics_w2c,
        source_extrinsics_w2c,
    )
    if any(not value.is_floating_point() or not torch.isfinite(value).all() for value in tensors):
        raise ValueError("GPA pair tensors must be finite floating point")
    if any(value.device != target_image.device for value in tensors):
        raise ValueError("GPA pair tensors must share a device")
    return batch_size, height, width


def _pair_mask(name: str, value: torch.Tensor | None, depth: torch.Tensor) -> torch.Tensor:
    if value is None:
        return torch.ones_like(depth, dtype=torch.bool)
    if value.shape != depth.shape or value.dtype is not torch.bool or value.device != depth.device:
        raise ValueError(f"{name} must be bool and match depth")
    return value


def inverse_warp_source_to_target(
    target_image: torch.Tensor,
    source_image: torch.Tensor,
    target_depth: torch.Tensor,
    source_depth: torch.Tensor,
    target_intrinsics: torch.Tensor,
    source_intrinsics: torch.Tensor,
    target_extrinsics_w2c: torch.Tensor,
    source_extrinsics_w2c: torch.Tensor,
    *,
    mu: float,
    geometry_epsilon: float,
    target_mask: torch.Tensor | None = None,
    source_mask: torch.Tensor | None = None,
    max_depth: float | None = None,
) -> dict[str, torch.Tensor]:
    """Inverse-warp one source into a target using predicted depth and W2C."""

    batch_size, height, width = _validate_pair(
        target_image,
        source_image,
        target_depth,
        source_depth,
        target_intrinsics,
        source_intrinsics,
        target_extrinsics_w2c,
        source_extrinsics_w2c,
    )
    _require_scalar("mu", mu, minimum=0.0, maximum=1.0)
    _require_scalar("geometry_epsilon", geometry_epsilon, minimum=0.0)
    if geometry_epsilon == 0:
        raise ValueError("geometry_epsilon must be positive")
    if max_depth is not None:
        _require_scalar("max_depth", max_depth, minimum=0.0)
        if max_depth == 0:
            raise ValueError("max_depth must be positive")
    target_mask = _pair_mask("target_mask", target_mask, target_depth)
    source_mask = _pair_mask("source_mask", source_mask, source_depth)

    dtype = target_depth.dtype
    rows = torch.arange(height, dtype=dtype, device=target_depth.device)
    columns = torch.arange(width, dtype=dtype, device=target_depth.device)
    vertical, horizontal = torch.meshgrid(rows, columns, indexing="ij")
    pixels = torch.stack((horizontal, vertical, torch.ones_like(horizontal)), dim=-1)
    pixels = pixels.reshape(1, height * width, 3).expand(batch_size, -1, -1)
    rays = torch.linalg.solve(target_intrinsics, pixels.transpose(1, 2)).transpose(1, 2)
    target_points = rays * target_depth.reshape(batch_size, -1, 1)

    target_rotation = target_extrinsics_w2c[:, :3, :3]
    target_translation = target_extrinsics_w2c[:, :3, 3]
    world_points = (target_points - target_translation[:, None]) @ target_rotation
    source_rotation = source_extrinsics_w2c[:, :3, :3]
    source_translation = source_extrinsics_w2c[:, :3, 3]
    source_points = world_points @ source_rotation.transpose(-1, -2) + source_translation[:, None]
    computed_source_depth = source_points[..., 2].reshape(batch_size, height, width)
    homogeneous = source_points @ source_intrinsics.transpose(-1, -2)
    safe_z = torch.where(
        source_points[..., 2].abs() > torch.finfo(dtype).eps,
        source_points[..., 2],
        torch.ones_like(source_points[..., 2]),
    )
    source_x = (homogeneous[..., 0] / safe_z).reshape(batch_size, height, width)
    source_y = (homogeneous[..., 1] / safe_z).reshape(batch_size, height, width)
    normalized_x = 2 * (source_x + 0.5) / width - 1
    normalized_y = 2 * (source_y + 0.5) / height - 1
    sampling_grid = torch.stack((normalized_x, normalized_y), dim=-1)

    warped_source = F.grid_sample(
        source_image,
        sampling_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    projected_source_depth = F.grid_sample(
        source_depth[:, None],
        sampling_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[:, 0]
    sampled_source_mask = (
        F.grid_sample(
            source_mask[:, None].to(dtype),
            sampling_grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        )[:, 0]
        > 0.5
    )
    in_bounds = (source_x >= 0) & (source_x <= width - 1) & (source_y >= 0) & (source_y <= height - 1)
    valid = (
        target_mask
        & sampled_source_mask
        & (target_depth > 0)
        & (computed_source_depth > 0)
        & (projected_source_depth > 0)
        & in_bounds
    )
    if max_depth is not None:
        valid &= (target_depth < max_depth) & (projected_source_depth < max_depth)

    photometric = _photometric_cost(target_image, warped_source, mu=mu)
    identity = _photometric_cost(target_image, source_image, mu=mu)
    structural = (computed_source_depth - projected_source_depth).abs() / (
        computed_source_depth + projected_source_depth + geometry_epsilon
    ).clamp_min(geometry_epsilon)
    return {
        "warped_source": warped_source,
        "projected_source_depth": projected_source_depth,
        "computed_source_depth": computed_source_depth,
        "photometric_cost": photometric,
        "identity_photometric_cost": identity,
        "structural_cost": structural,
        "sampling_grid": sampling_grid,
        "valid_mask": valid,
    }


def _smoothness_map(depth: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    disparity = depth.clamp_min(torch.finfo(depth.dtype).eps).reciprocal()
    mean_disparity = disparity.mean(dim=(-2, -1), keepdim=True).clamp_min(torch.finfo(depth.dtype).eps)
    disparity = disparity / mean_disparity
    disparity_dx = (disparity[..., :, 1:] - disparity[..., :, :-1]).abs()
    disparity_dy = (disparity[..., 1:, :] - disparity[..., :-1, :]).abs()
    image_dx = (image[..., :, 1:] - image[..., :, :-1]).abs().mean(dim=1)
    image_dy = (image[..., 1:, :] - image[..., :-1, :]).abs().mean(dim=1)
    result = torch.zeros_like(depth)
    result[..., :, :-1] += disparity_dx * torch.exp(-image_dx)
    result[..., :-1, :] += disparity_dy * torch.exp(-image_dy)
    return result


def gpa_edge_aware_smoothness(
    depth: torch.Tensor,
    image: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return first-order edge-aware normalized-disparity smoothness."""

    if depth.ndim != 3 or image.shape != (depth.shape[0], 3, depth.shape[1], depth.shape[2]):
        raise ValueError("depth/image must have shapes [B,H,W] and [B,3,H,W]")
    if not depth.is_floating_point() or not image.is_floating_point():
        raise ValueError("depth/image must be floating point")
    if not torch.isfinite(depth).all() or not torch.isfinite(image).all():
        raise ValueError("depth/image must be finite")
    mask = _pair_mask("mask", mask, depth)
    if not bool(mask.any()):
        return depth.reshape(-1)[0] * 0
    return _smoothness_map(depth, image)[mask].mean()


def _validate_sequence(
    images: torch.Tensor,
    depths: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics_w2c: torch.Tensor,
    valid_mask: torch.Tensor,
    dynamic_mask: torch.Tensor | None,
    frame_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if images.ndim != 5 or images.shape[2] != 3:
        raise ValueError("images must have shape [B,S,3,H,W]")
    batch_size, frame_count, _, height, width = images.shape
    if frame_count < 2 or depths.shape != (batch_size, frame_count, height, width):
        raise ValueError("depths must match [B,S,H,W] with S >= 2")
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
        frame_mask = torch.ones((batch_size, frame_count), dtype=torch.bool, device=images.device)
    if frame_mask.shape != (batch_size, frame_count) or frame_mask.dtype is not torch.bool:
        raise ValueError("frame_mask must be bool with shape [B,S]")
    tensors = (images, depths, intrinsics, extrinsics_w2c)
    if any(not value.is_floating_point() or not torch.isfinite(value).all() for value in tensors):
        raise ValueError("GPA sequence tensors must be finite floating point")
    if any(value.device != images.device for value in (*tensors, valid_mask, dynamic_mask, frame_mask)):
        raise ValueError("GPA sequence tensors must share a device")
    return dynamic_mask, frame_mask


def gpa_sequence_loss(
    images: torch.Tensor,
    depths: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics_w2c: torch.Tensor,
    *,
    valid_mask: torch.Tensor,
    anchor_indices: torch.Tensor,
    mu: float,
    lambda_geo: float,
    lambda_smooth: float,
    auto_mask_delta: float,
    geometry_epsilon: float = 1e-6,
    dynamic_mask: torch.Tensor | None = None,
    frame_mask: torch.Tensor | None = None,
    auto_mask_enabled: bool = True,
    mask_mode: str = "intersection",
    max_depth: float | None = None,
) -> dict[str, torch.Tensor]:
    """Apply GPA-style hard source selection to multiple explicit anchors."""

    dynamic_mask, frame_mask = _validate_sequence(
        images,
        depths,
        intrinsics,
        extrinsics_w2c,
        valid_mask,
        dynamic_mask,
        frame_mask,
    )
    _require_scalar("mu", mu, minimum=0.0, maximum=1.0)
    _require_scalar("lambda_geo", lambda_geo, minimum=0.0)
    _require_scalar("lambda_smooth", lambda_smooth, minimum=0.0)
    _require_scalar("auto_mask_delta", auto_mask_delta, minimum=0.0)
    if not isinstance(auto_mask_enabled, bool):
        raise ValueError("auto_mask_enabled must be boolean")
    if mask_mode not in {"intersection", "union"}:
        raise ValueError("mask_mode must be intersection or union")
    batch_size, frame_count, _, height, width = images.shape
    if anchor_indices.ndim != 2 or anchor_indices.shape[0] != batch_size or anchor_indices.dtype is not torch.long:
        raise ValueError("anchor_indices must be int64 with shape [B,A]")
    if anchor_indices.device != images.device:
        raise ValueError("anchor_indices must share the image device")
    if torch.any((anchor_indices < 0) | (anchor_indices >= frame_count)):
        raise ValueError("anchor_indices are outside the frame range")
    if not frame_mask.gather(1, anchor_indices).all():
        raise ValueError("anchor_indices must select valid frames")

    anchor_count = anchor_indices.shape[1]
    selected_sources = torch.full(
        (batch_size, anchor_count, height, width),
        -1,
        dtype=torch.long,
        device=images.device,
    )
    selected_valid = torch.zeros_like(selected_sources, dtype=torch.bool)
    physical_sum = depths.reshape(-1)[0] * 0
    photo_sum = physical_sum.clone()
    structural_sum = physical_sum.clone()
    smooth_sum = physical_sum.clone()
    total_valid = 0
    static_valid = valid_mask & ~dynamic_mask & frame_mask[:, :, None, None]

    for batch_index in range(batch_size):
        for anchor_offset, anchor_tensor in enumerate(anchor_indices[batch_index]):
            anchor = int(anchor_tensor)
            sources = [
                index for index in range(frame_count) if index != anchor and bool(frame_mask[batch_index, index])
            ]
            if not sources:
                continue
            pair_results = [
                inverse_warp_source_to_target(
                    images[batch_index : batch_index + 1, anchor],
                    images[batch_index : batch_index + 1, source],
                    depths[batch_index : batch_index + 1, anchor],
                    depths[batch_index : batch_index + 1, source],
                    intrinsics[batch_index : batch_index + 1, anchor],
                    intrinsics[batch_index : batch_index + 1, source],
                    extrinsics_w2c[batch_index : batch_index + 1, anchor],
                    extrinsics_w2c[batch_index : batch_index + 1, source],
                    mu=mu,
                    geometry_epsilon=geometry_epsilon,
                    target_mask=static_valid[batch_index : batch_index + 1, anchor],
                    source_mask=static_valid[batch_index : batch_index + 1, source],
                    max_depth=max_depth,
                )
                for source in sources
            ]
            costs = torch.stack(
                [result["photometric_cost"] + lambda_geo * result["structural_cost"] for result in pair_results]
            )[:, 0]
            pair_valid = torch.stack([result["valid_mask"] for result in pair_results])[:, 0]
            masked_costs = torch.where(pair_valid, costs, torch.full_like(costs, torch.inf))
            minimum_cost, selected_offset = masked_costs.min(dim=0)
            projection_valid = pair_valid.any(dim=0)
            identity_costs = torch.stack([result["identity_photometric_cost"] for result in pair_results])[:, 0]
            identity_valid = torch.stack(
                [static_valid[batch_index, anchor] & static_valid[batch_index, source] for source in sources]
            )
            minimum_identity = (
                torch.where(
                    identity_valid,
                    identity_costs,
                    torch.full_like(identity_costs, torch.inf),
                )
                .min(dim=0)
                .values
            )
            selected_photo = torch.stack([result["photometric_cost"] for result in pair_results])[:, 0].gather(
                0, selected_offset.unsqueeze(0)
            )[0]
            selected_structural = torch.stack([result["structural_cost"] for result in pair_results])[:, 0].gather(
                0, selected_offset.unsqueeze(0)
            )[0]
            auto_mask = selected_photo < (1 + auto_mask_delta) * minimum_identity
            if not auto_mask_enabled:
                auto_mask = torch.ones_like(auto_mask)
            valid = projection_valid & auto_mask if mask_mode == "intersection" else projection_valid | auto_mask
            valid &= torch.isfinite(minimum_cost)
            source_lookup = torch.tensor(sources, dtype=torch.long, device=images.device)
            selected_sources[batch_index, anchor_offset] = torch.where(
                valid,
                source_lookup[selected_offset],
                torch.full_like(selected_offset, -1),
            )
            selected_valid[batch_index, anchor_offset] = valid
            count = int(valid.sum())
            if count == 0:
                continue
            physical_sum = physical_sum + minimum_cost[valid].sum()
            photo_sum = photo_sum + selected_photo[valid].sum()
            structural_sum = structural_sum + selected_structural[valid].sum()
            smooth_sum = (
                smooth_sum
                + _smoothness_map(
                    depths[batch_index : batch_index + 1, anchor],
                    images[batch_index : batch_index + 1, anchor],
                )[0][valid].sum()
            )
            total_valid += count

    if total_valid == 0:
        zero = depths.reshape(-1)[0] * 0 + images.reshape(-1)[0] * 0
        physical = photometric = structural = smoothness = objective = zero
    else:
        denominator = depths.new_tensor(float(total_valid))
        physical = physical_sum / denominator
        photometric = photo_sum / denominator
        structural = structural_sum / denominator
        smoothness = smooth_sum / denominator
        objective = physical + lambda_smooth * smoothness
    return {
        "objective": objective,
        "physical": physical,
        "photometric": photometric,
        "structural": structural,
        "smoothness": smoothness,
        "valid_fraction": depths.new_tensor(total_valid / (batch_size * anchor_count * height * width)),
        "selected_source_indices": selected_sources,
        "selected_valid_mask": selected_valid,
        "anchor_count": depths.new_tensor(float(anchor_count)),
    }
