"""Differentiable training-only reprojection backends and photometric loss."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch


def _require_finite(name: str, value: torch.Tensor) -> None:
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be a finite floating-point tensor")


def soft_zbuffer_reproject(
    source_rgb: torch.Tensor,
    source_depth: torch.Tensor,
    source_intrinsics: torch.Tensor,
    target_intrinsics: torch.Tensor,
    source_extrinsics_w2c: torch.Tensor,
    target_extrinsics_w2c: torch.Tensor,
    *,
    source_mask: torch.Tensor | None = None,
    target_depth: torch.Tensor | None = None,
    target_mask: torch.Tensor | None = None,
    max_depth_m: float | None = None,
    relative_depth_tolerance: float = 0.03,
    z_temperature: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Forward-splat one RGB-D view with bilinear and soft Z weights."""

    if source_rgb.ndim != 4 or source_rgb.shape[1] != 3:
        raise ValueError("source_rgb must have shape [B,3,H,W]")
    batch_size, _, height, width = source_rgb.shape
    if source_depth.shape != (batch_size, height, width):
        raise ValueError("source_depth must have shape [B,H,W]")
    if source_intrinsics.shape != (batch_size, 3, 3) or target_intrinsics.shape != (batch_size, 3, 3):
        raise ValueError("intrinsics must have shape [B,3,3]")
    if source_extrinsics_w2c.shape != (batch_size, 3, 4) or target_extrinsics_w2c.shape != (
        batch_size,
        3,
        4,
    ):
        raise ValueError("extrinsics must have shape [B,3,4]")
    for name, value in (
        ("source_rgb", source_rgb),
        ("source_depth", source_depth),
        ("source_intrinsics", source_intrinsics),
        ("target_intrinsics", target_intrinsics),
        ("source_extrinsics_w2c", source_extrinsics_w2c),
        ("target_extrinsics_w2c", target_extrinsics_w2c),
    ):
        _require_finite(name, value)
    if not math.isfinite(relative_depth_tolerance) or relative_depth_tolerance <= 0:
        raise ValueError("relative_depth_tolerance must be finite and positive")
    if not math.isfinite(z_temperature) or z_temperature <= 0:
        raise ValueError("z_temperature must be finite and positive")
    if max_depth_m is not None and (not math.isfinite(max_depth_m) or max_depth_m <= 0):
        raise ValueError("max_depth_m must be finite and positive when provided")
    if source_mask is None:
        source_mask = torch.ones_like(source_depth, dtype=torch.bool)
    if source_mask.shape != source_depth.shape or source_mask.dtype is not torch.bool:
        raise ValueError("source_mask must be bool with shape [B,H,W]")
    if target_depth is not None:
        if target_depth.shape != source_depth.shape:
            raise ValueError("target_depth must have shape [B,H,W]")
        _require_finite("target_depth", target_depth)
    if target_mask is not None and (target_mask.shape != source_depth.shape or target_mask.dtype is not torch.bool):
        raise ValueError("target_mask must be bool with shape [B,H,W]")

    dtype = source_depth.dtype
    device = source_depth.device
    rows = torch.arange(height, device=device, dtype=dtype)
    columns = torch.arange(width, device=device, dtype=dtype)
    vertical, horizontal = torch.meshgrid(rows, columns, indexing="ij")
    pixels = torch.stack((horizontal, vertical, torch.ones_like(horizontal)), dim=-1)
    pixels = pixels.reshape(1, height * width, 3).expand(batch_size, -1, -1)
    rays = torch.linalg.solve(source_intrinsics, pixels.transpose(1, 2)).transpose(1, 2)
    flat_depth = source_depth.reshape(batch_size, -1)
    source_points = rays * flat_depth.unsqueeze(-1)

    source_rotation = source_extrinsics_w2c[:, :3, :3]
    source_translation = source_extrinsics_w2c[:, :3, 3]
    world_points = (source_points - source_translation[:, None]) @ source_rotation
    target_rotation = target_extrinsics_w2c[:, :3, :3]
    target_translation = target_extrinsics_w2c[:, :3, 3]
    target_points = world_points @ target_rotation.transpose(-1, -2) + target_translation[:, None]
    target_z = target_points[..., 2]
    homogeneous = target_points @ target_intrinsics.transpose(-1, -2)
    safe_z = torch.where(target_z.abs() > torch.finfo(dtype).eps, target_z, torch.ones_like(target_z))
    projected_x = homogeneous[..., 0] / safe_z
    projected_y = homogeneous[..., 1] / safe_z

    source_valid = source_mask.reshape(batch_size, -1) & (flat_depth > 0) & (target_z > 0)
    if max_depth_m is not None:
        source_valid &= flat_depth < max_depth_m
    minimum_z = torch.where(source_valid, target_z, torch.full_like(target_z, torch.inf)).amin(dim=1, keepdim=True)
    minimum_z = torch.where(torch.isfinite(minimum_z), minimum_z, torch.zeros_like(minimum_z))
    soft_z_weight = torch.exp(-(target_z - minimum_z) / z_temperature)

    floor_x = torch.floor(projected_x)
    floor_y = torch.floor(projected_y)
    fraction_x = projected_x - floor_x
    fraction_y = projected_y - floor_y
    neighbor_x = torch.stack((floor_x, floor_x + 1, floor_x, floor_x + 1), dim=-1)
    neighbor_y = torch.stack((floor_y, floor_y, floor_y + 1, floor_y + 1), dim=-1)
    bilinear_weight = torch.stack(
        (
            (1 - fraction_x) * (1 - fraction_y),
            fraction_x * (1 - fraction_y),
            (1 - fraction_x) * fraction_y,
            fraction_x * fraction_y,
        ),
        dim=-1,
    )
    neighbor_x_int = neighbor_x.to(torch.int64)
    neighbor_y_int = neighbor_y.to(torch.int64)
    in_bounds = (neighbor_x_int >= 0) & (neighbor_x_int < width) & (neighbor_y_int >= 0) & (neighbor_y_int < height)
    safe_x = neighbor_x_int.clamp(0, width - 1)
    safe_y = neighbor_y_int.clamp(0, height - 1)
    neighbor_index = safe_y * width + safe_x
    contribution_valid = source_valid.unsqueeze(-1) & in_bounds
    if target_depth is not None:
        observed_depth = target_depth.reshape(batch_size, -1).gather(1, neighbor_index.reshape(batch_size, -1))
        observed_depth = observed_depth.reshape_as(neighbor_index)
        depth_valid = observed_depth > 0
        if target_mask is not None:
            gathered_mask = target_mask.reshape(batch_size, -1).gather(1, neighbor_index.reshape(batch_size, -1))
            depth_valid &= gathered_mask.reshape_as(neighbor_index)
        relative_error = (observed_depth - target_z.unsqueeze(-1)).abs() / observed_depth.clamp_min(
            torch.finfo(dtype).eps
        )
        contribution_valid &= depth_valid & (relative_error <= relative_depth_tolerance)
    elif target_mask is not None:
        gathered_mask = target_mask.reshape(batch_size, -1).gather(1, neighbor_index.reshape(batch_size, -1))
        contribution_valid &= gathered_mask.reshape_as(neighbor_index)

    weights = bilinear_weight * soft_z_weight.unsqueeze(-1) * contribution_valid.to(dtype=dtype)
    flat_indices = neighbor_index.reshape(batch_size, -1)
    flat_weights = weights.reshape(batch_size, -1)
    denominator = torch.zeros((batch_size, height * width), dtype=dtype, device=device)
    denominator.scatter_add_(1, flat_indices, flat_weights)
    colors = source_rgb.permute(0, 2, 3, 1).reshape(batch_size, -1, 3)
    colors = colors.unsqueeze(2).expand(-1, -1, 4, -1).reshape(batch_size, -1, 3)
    numerator = torch.zeros((batch_size, height * width, 3), dtype=dtype, device=device)
    numerator.scatter_add_(1, flat_indices.unsqueeze(-1).expand(-1, -1, 3), colors * flat_weights.unsqueeze(-1))
    visibility = denominator > torch.finfo(dtype).eps
    rendered = numerator / denominator.clamp_min(torch.finfo(dtype).eps).unsqueeze(-1)
    return {
        "rgb": rendered.reshape(batch_size, height, width, 3).permute(0, 3, 1, 2),
        "visibility": visibility.reshape(batch_size, height, width),
        "weight": denominator.reshape(batch_size, height, width),
    }


def masked_photometric_l1(
    rendered_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    visibility: torch.Tensor,
) -> torch.Tensor:
    """Return visible RGB L1 or a graph-connected zero for empty visibility."""

    if rendered_rgb.ndim != 4 or rendered_rgb.shape[1] != 3 or target_rgb.shape != rendered_rgb.shape:
        raise ValueError("rendered_rgb and target_rgb must have matching [B,3,H,W] shapes")
    if visibility.shape != (rendered_rgb.shape[0], rendered_rgb.shape[2], rendered_rgb.shape[3]):
        raise ValueError("visibility must have shape [B,H,W]")
    if visibility.dtype is not torch.bool:
        raise ValueError("visibility must have bool dtype")
    _require_finite("rendered_rgb", rendered_rgb)
    _require_finite("target_rgb", target_rgb)
    if not bool(visibility.any()):
        return rendered_rgb.reshape(-1)[0] * 0.0
    visible_channels = visibility[:, None].expand_as(rendered_rgb)
    return (rendered_rgb[visible_channels] - target_rgb[visible_channels]).abs().mean()


def gsplat_reproject(
    source_rgb: torch.Tensor,
    source_depth: torch.Tensor,
    source_intrinsics: torch.Tensor,
    target_intrinsics: torch.Tensor,
    source_extrinsics_w2c: torch.Tensor,
    target_extrinsics_w2c: torch.Tensor,
    *,
    source_mask: torch.Tensor | None = None,
    target_depth: torch.Tensor | None = None,
    target_mask: torch.Tensor | None = None,
    max_depth_m: float | None = None,
    relative_depth_tolerance: float = 0.03,
    gaussian_radius_pixels: float = 0.75,
    opacity: float = 0.95,
) -> dict[str, torch.Tensor]:
    """Render RGB-D pixels as explicit world-space Gaussians via gsplat."""

    try:
        from gsplat import rasterization
    except (ImportError, OSError) as error:
        raise RuntimeError("gsplat backend was selected but its CUDA extension is not importable") from error
    if source_depth.device.type != "cuda":
        raise RuntimeError("gsplat backend requires CUDA tensors")
    if source_rgb.ndim != 4 or source_rgb.shape[1] != 3:
        raise ValueError("source_rgb must have shape [B,3,H,W]")
    batch_size, _, height, width = source_rgb.shape
    if source_depth.shape != (batch_size, height, width):
        raise ValueError("source_depth must have shape [B,H,W]")
    if source_intrinsics.shape != (batch_size, 3, 3) or target_intrinsics.shape != (batch_size, 3, 3):
        raise ValueError("intrinsics must have shape [B,3,3]")
    if source_extrinsics_w2c.shape != (batch_size, 3, 4) or target_extrinsics_w2c.shape != (
        batch_size,
        3,
        4,
    ):
        raise ValueError("extrinsics must have shape [B,3,4]")
    for name, value in (
        ("source_rgb", source_rgb),
        ("source_depth", source_depth),
        ("source_intrinsics", source_intrinsics),
        ("target_intrinsics", target_intrinsics),
        ("source_extrinsics_w2c", source_extrinsics_w2c),
        ("target_extrinsics_w2c", target_extrinsics_w2c),
    ):
        _require_finite(name, value)
    if not math.isfinite(relative_depth_tolerance) or relative_depth_tolerance <= 0:
        raise ValueError("relative_depth_tolerance must be finite and positive")
    if not math.isfinite(gaussian_radius_pixels) or gaussian_radius_pixels <= 0:
        raise ValueError("gaussian_radius_pixels must be finite and positive")
    if not math.isfinite(opacity) or not 0 < opacity <= 1:
        raise ValueError("opacity must be finite and within (0, 1]")
    if max_depth_m is not None and (not math.isfinite(max_depth_m) or max_depth_m <= 0):
        raise ValueError("max_depth_m must be finite and positive when provided")
    if source_mask is None:
        source_mask = torch.ones_like(source_depth, dtype=torch.bool)
    if source_mask.shape != source_depth.shape or source_mask.dtype is not torch.bool:
        raise ValueError("source_mask must be bool with shape [B,H,W]")
    if target_depth is not None:
        if target_depth.shape != source_depth.shape:
            raise ValueError("target_depth must have shape [B,H,W]")
        _require_finite("target_depth", target_depth)
    if target_mask is not None and (target_mask.shape != source_depth.shape or target_mask.dtype is not torch.bool):
        raise ValueError("target_mask must be bool with shape [B,H,W]")

    device = source_depth.device
    rows = torch.arange(height, device=device, dtype=torch.float32)
    columns = torch.arange(width, device=device, dtype=torch.float32)
    vertical, horizontal = torch.meshgrid(rows, columns, indexing="ij")
    pixels = torch.stack((horizontal, vertical, torch.ones_like(horizontal)), dim=-1).reshape(-1, 3)
    rendered_batches: list[torch.Tensor] = []
    visibility_batches: list[torch.Tensor] = []
    alpha_batches: list[torch.Tensor] = []
    for batch_index in range(batch_size):
        depth = source_depth[batch_index].float().reshape(-1)
        valid = source_mask[batch_index].reshape(-1) & (depth > 0)
        if max_depth_m is not None:
            valid &= depth < max_depth_m
        if not bool(valid.any()):
            zero_rgb = source_rgb[batch_index].float() * 0.0
            rendered_batches.append(zero_rgb)
            visibility_batches.append(torch.zeros((height, width), dtype=torch.bool, device=device))
            alpha_batches.append(torch.zeros((height, width), dtype=torch.float32, device=device))
            continue
        source_k = source_intrinsics[batch_index].float()
        rays = torch.linalg.solve(source_k, pixels.T).T[valid]
        camera_points = rays * depth[valid, None]
        source_rotation = source_extrinsics_w2c[batch_index, :3, :3].float()
        source_translation = source_extrinsics_w2c[batch_index, :3, 3].float()
        means = (camera_points - source_translation) @ source_rotation
        focal = (source_k[0, 0] + source_k[1, 1]) / 2
        isotropic_scale = (depth[valid] / focal * gaussian_radius_pixels).clamp_min(1e-5)
        scales = isotropic_scale[:, None].expand(-1, 3)
        quaternions = torch.zeros((len(means), 4), dtype=torch.float32, device=device)
        quaternions[:, 0] = 1.0
        opacities = torch.full((len(means),), opacity, dtype=torch.float32, device=device)
        colors = source_rgb[batch_index].permute(1, 2, 0).reshape(-1, 3)[valid].float()
        view = torch.eye(4, dtype=torch.float32, device=device)[None]
        view[:, :3, :] = target_extrinsics_w2c[batch_index].float()
        rendered, alpha, _ = rasterization(
            means=means,
            quats=quaternions,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=view,
            Ks=target_intrinsics[batch_index : batch_index + 1].float(),
            width=width,
            height=height,
            packed=False,
            render_mode="RGB+ED",
        )
        rgb = rendered[0, ..., :3].permute(2, 0, 1)
        rendered_depth = rendered[0, ..., 3]
        alpha_image = alpha[0, ..., 0]
        visibility = alpha_image > 1e-4
        if target_depth is not None:
            observed = target_depth[batch_index].float()
            visibility &= observed > 0
            visibility &= (rendered_depth - observed).abs() / observed.clamp_min(1e-7) <= relative_depth_tolerance
        if target_mask is not None:
            visibility &= target_mask[batch_index]
        rendered_batches.append(rgb)
        visibility_batches.append(visibility)
        alpha_batches.append(alpha_image)
    return {
        "rgb": torch.stack(rendered_batches),
        "visibility": torch.stack(visibility_batches),
        "weight": torch.stack(alpha_batches),
    }


def reproject_rgbd(backend: str, *args: torch.Tensor, **kwargs: object) -> dict[str, torch.Tensor]:
    """Dispatch an explicitly selected renderer without implicit fallback."""

    if backend == "soft":
        return soft_zbuffer_reproject(*args, **kwargs)
    if backend == "gsplat":
        return gsplat_reproject(*args, **kwargs)
    raise ValueError("renderer backend must be explicitly set to soft or gsplat")


def compute_sequence_photometric_loss(
    predictions: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    *,
    backend: str,
    max_depth_m: float,
    relative_depth_tolerance: float = 0.03,
    pose_source: str = "predicted",
    use_target_depth: bool = True,
    z_temperature: float = 0.1,
    gaussian_radius_pixels: float = 0.75,
    opacity: float = 0.95,
) -> dict[str, torch.Tensor]:
    """Render the reference RGB into later frames and return mean L1/coverage."""

    from vggt_omega.utils.pose_enc import encoding_to_camera

    images = batch.get("images")
    target_depths = batch.get("depths")
    depth_masks = batch.get("depth_masks")
    normalization_scale = batch.get("normalization_scale_m")
    predicted_pose = predictions.get("pose_enc")
    predicted_depth = predictions.get("depth")
    values = (images, target_depths, depth_masks, normalization_scale, predicted_pose, predicted_depth)
    if not all(isinstance(value, torch.Tensor) for value in values):
        raise KeyError("photometric loss requires tensor images/depths/masks/scale/pose/depth")
    assert isinstance(images, torch.Tensor)
    assert isinstance(target_depths, torch.Tensor)
    assert isinstance(depth_masks, torch.Tensor)
    assert isinstance(normalization_scale, torch.Tensor)
    assert isinstance(predicted_pose, torch.Tensor)
    assert isinstance(predicted_depth, torch.Tensor)
    if images.ndim != 5 or images.shape[2] != 3:
        raise ValueError("images must have shape [B,S,3,H,W]")
    batch_size, frame_count, _, height, width = images.shape
    if target_depths.shape != (batch_size, frame_count, height, width):
        raise ValueError("target depths do not match image shape")
    if depth_masks.shape != target_depths.shape or depth_masks.dtype is not torch.bool:
        raise ValueError("depth masks must be bool and match target depths")
    if predicted_depth.shape != (*target_depths.shape, 1) or predicted_pose.shape != (batch_size, frame_count, 9):
        raise ValueError("predicted depth or pose shape is invalid")
    if normalization_scale.numel() != batch_size:
        raise ValueError("normalization_scale_m must contain one value per sample")
    if pose_source not in {"predicted", "ground_truth"}:
        raise ValueError("pose_source must be predicted or ground_truth")
    if not isinstance(use_target_depth, bool):
        raise ValueError("use_target_depth must be boolean")
    if not math.isfinite(max_depth_m) or max_depth_m <= 0:
        raise ValueError("max_depth_m must be finite and positive")
    for name, value in (
        ("images", images),
        ("target_depths", target_depths),
        ("normalization_scale_m", normalization_scale),
        ("predicted_pose", predicted_pose),
        ("predicted_depth", predicted_depth),
    ):
        _require_finite(name, value)

    if pose_source == "predicted":
        pose = predicted_pose.float()
        quaternion = pose[..., 3:7]
        quaternion_norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
        identity = torch.zeros_like(quaternion)
        identity[..., 3] = 1.0
        pose = torch.cat(
            (pose[..., :3], torch.where(quaternion_norm > 1e-8, quaternion, identity), pose[..., 7:]), dim=-1
        )
        extrinsics, intrinsics = encoding_to_camera(pose, (height, width), build_intrinsics=True)
        assert intrinsics is not None
    else:
        extrinsics = batch.get("extrinsics")
        intrinsics = batch.get("intrinsics")
        if not isinstance(extrinsics, torch.Tensor) or not isinstance(intrinsics, torch.Tensor):
            raise KeyError("ground_truth pose source requires batch extrinsics and intrinsics")
        extrinsics = extrinsics.float()
        intrinsics = intrinsics.float()

    metric_depths = target_depths.float() * normalization_scale.float().reshape(batch_size, 1, 1, 1)
    static_mask = ~batch["dynamic_masks"] if "dynamic_masks" in batch else torch.ones_like(depth_masks)
    if static_mask.shape != depth_masks.shape or static_mask.dtype is not torch.bool:
        raise ValueError("dynamic_masks must be bool and match depth masks")
    near_masks = depth_masks & static_mask & (metric_depths > 0) & (metric_depths < max_depth_m)
    if frame_count < 2:
        zero = predicted_depth.reshape(-1)[0] * 0.0
        return {"photometric": zero, "photometric_visibility": zero}

    source_rgb = images[:, 0].float()
    source_depth = predicted_depth[:, 0, ..., 0].float()
    losses: list[torch.Tensor] = []
    coverages: list[torch.Tensor] = []
    for target_index in range(1, frame_count):
        common_options = {
            "source_mask": near_masks[:, 0],
            "target_depth": target_depths[:, target_index].float() if use_target_depth else None,
            "target_mask": near_masks[:, target_index],
            "relative_depth_tolerance": relative_depth_tolerance,
        }
        if backend == "soft":
            backend_options = {"z_temperature": z_temperature}
        elif backend == "gsplat":
            backend_options = {"gaussian_radius_pixels": gaussian_radius_pixels, "opacity": opacity}
        else:
            raise ValueError("renderer backend must be explicitly set to soft or gsplat")
        rendered = reproject_rgbd(
            backend,
            source_rgb,
            source_depth,
            intrinsics[:, 0],
            intrinsics[:, target_index],
            extrinsics[:, 0],
            extrinsics[:, target_index],
            **common_options,
            **backend_options,
        )
        losses.append(masked_photometric_l1(rendered["rgb"], images[:, target_index].float(), rendered["visibility"]))
        coverages.append(rendered["visibility"].float().mean())
    return {
        "photometric": torch.stack(losses).mean(),
        "photometric_visibility": torch.stack(coverages).mean(),
    }
