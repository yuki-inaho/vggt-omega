"""Dense RGB-D geometry used by supervised VGGT-Omega fine-tuning.

The public training boundary uses OpenCV camera coordinates and world-to-camera
extrinsics.  All helpers fail on malformed calibration instead of silently
clamping it into a plausible-looking result.
"""

from __future__ import annotations

from typing import TypedDict

import torch
from torch import Tensor


class GeometryContractError(ValueError):
    """Raised when camera or dense-geometry input violates the data contract."""


class NormalizedSupervision(TypedDict):
    depths: Tensor
    depth_masks: Tensor
    intrinsics: Tensor
    extrinsics: Tensor
    cam_points: Tensor
    world_points: Tensor
    scale: Tensor


def _require_floating(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise GeometryContractError(f"{name} must be a floating-point torch.Tensor")
    if not torch.isfinite(value).all():
        raise GeometryContractError(f"{name} contains NaN or Inf")


def _homogeneous(transform: Tensor, name: str) -> Tensor:
    _require_floating(name, transform)
    if transform.shape[-2:] == (3, 4):
        result = torch.zeros((*transform.shape[:-2], 4, 4), dtype=transform.dtype, device=transform.device)
        result[..., :3, :] = transform
        result[..., 3, 3] = 1
    elif transform.shape[-2:] == (4, 4):
        result = transform.clone()
        expected = torch.zeros_like(result[..., 3, :])
        expected[..., 3] = 1
        if not torch.allclose(result[..., 3, :], expected, atol=1e-6, rtol=1e-6):
            raise GeometryContractError(f"{name} has an invalid homogeneous last row")
    else:
        raise GeometryContractError(f"{name} must end in shape [3,4] or [4,4]")
    rotation = result[..., :3, :3]
    determinant = torch.linalg.det(rotation)
    if (determinant.abs() <= torch.finfo(result.dtype).eps).any():
        raise GeometryContractError(f"{name} contains a singular rotation")
    identity = torch.eye(3, dtype=result.dtype, device=result.device).expand_as(rotation)
    if not torch.allclose(rotation.transpose(-1, -2) @ rotation, identity, atol=1e-4, rtol=1e-4):
        raise GeometryContractError(f"{name} contains a non-orthogonal rotation")
    if not torch.allclose(determinant, torch.ones_like(determinant), atol=1e-4, rtol=1e-4):
        raise GeometryContractError(f"{name} rotation determinant must be one")
    return result


def camera_to_world_to_world_to_camera(camera_to_world: Tensor) -> Tensor:
    """Invert camera-to-world matrices and return OpenCV ``[..., 3, 4]`` poses."""

    homogeneous = _homogeneous(camera_to_world, "camera_to_world")
    try:
        inverse = torch.linalg.inv(homogeneous)
    except RuntimeError as error:
        raise GeometryContractError("camera_to_world is singular") from error
    if not torch.isfinite(inverse).all():
        raise GeometryContractError("camera_to_world inverse contains NaN or Inf")
    return inverse[..., :3, :]


def _expanded_matrices(matrices: Tensor, point_prefix: tuple[int, ...], name: str) -> Tensor:
    matrix_prefix = matrices.shape[:-2]
    if len(matrix_prefix) > len(point_prefix):
        raise GeometryContractError(f"{name} has more leading dimensions than points")
    for matrix_size, point_size in zip(matrix_prefix, point_prefix):
        if matrix_size not in (1, point_size):
            raise GeometryContractError(f"{name} leading dimensions do not match points")
    singleton_count = len(point_prefix) - len(matrix_prefix)
    return matrices.reshape(*matrix_prefix, *([1] * singleton_count), *matrices.shape[-2:])


def unproject_depth(depths: Tensor, intrinsics: Tensor) -> Tensor:
    """Unproject camera-Z depth to ``[..., H, W, 3]`` camera-space points."""

    _require_floating("depths", depths)
    _require_floating("intrinsics", intrinsics)
    if depths.ndim < 2:
        raise GeometryContractError("depths must end in shape [H,W]")
    if intrinsics.shape[-2:] != (3, 3):
        raise GeometryContractError("intrinsics must end in shape [3,3]")
    depth_prefix = depths.shape[:-2]
    intrinsics_prefix = intrinsics.shape[:-2]
    try:
        common_prefix = torch.broadcast_shapes(depth_prefix, intrinsics_prefix)
    except RuntimeError as error:
        raise GeometryContractError("depth and intrinsics leading dimensions are not broadcastable") from error
    depths = torch.broadcast_to(depths, (*common_prefix, *depths.shape[-2:]))
    intrinsics = torch.broadcast_to(intrinsics, (*common_prefix, 3, 3))
    determinant = torch.linalg.det(intrinsics)
    if (determinant.abs() <= torch.finfo(intrinsics.dtype).eps).any():
        raise GeometryContractError("intrinsics contains a singular matrix")
    try:
        inverse_intrinsics = torch.linalg.inv(intrinsics)
    except RuntimeError as error:
        raise GeometryContractError("intrinsics contains a singular matrix") from error
    height, width = depths.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=depths.dtype, device=depths.device),
        torch.arange(width, dtype=depths.dtype, device=depths.device),
        indexing="ij",
    )
    pixels = torch.stack((xx, yy, torch.ones_like(xx)), dim=-1)
    rays = torch.einsum("...ij,hwj->...hwi", inverse_intrinsics, pixels)
    return rays * depths.unsqueeze(-1)


def project_points(points: Tensor, intrinsics: Tensor) -> tuple[Tensor, Tensor]:
    """Project camera-space points, returning pixel coordinates and camera Z."""

    _require_floating("points", points)
    _require_floating("intrinsics", intrinsics)
    if points.ndim < 2 or points.shape[-1] != 3:
        raise GeometryContractError("points must end in shape [...,3]")
    if intrinsics.shape[-2:] != (3, 3):
        raise GeometryContractError("intrinsics must end in shape [3,3]")
    expanded = _expanded_matrices(intrinsics, points.shape[:-1], "intrinsics")
    homogeneous_pixels = torch.matmul(expanded, points.unsqueeze(-1)).squeeze(-1)
    depth = points[..., 2]
    if (depth.abs() <= torch.finfo(depth.dtype).eps).any():
        raise GeometryContractError("cannot project a point with zero camera depth")
    return homogeneous_pixels[..., :2] / depth.unsqueeze(-1), depth


def transform_points(points: Tensor, transforms: Tensor) -> Tensor:
    """Apply rigid ``[...,3,4]`` or ``[...,4,4]`` transforms to 3D points."""

    _require_floating("points", points)
    if points.ndim < 2 or points.shape[-1] != 3:
        raise GeometryContractError("points must end in shape [...,3]")
    homogeneous = _homogeneous(transforms, "transforms")
    expanded = _expanded_matrices(homogeneous, points.shape[:-1], "transforms")
    ones = torch.ones_like(points[..., :1])
    homogeneous_points = torch.cat((points, ones), dim=-1)
    return torch.matmul(expanded, homogeneous_points.unsqueeze(-1)).squeeze(-1)[..., :3]


def reference_extrinsics(extrinsics_w2c: Tensor) -> Tensor:
    """Express a sequence of W2C poses in its first camera coordinate system."""

    if extrinsics_w2c.ndim < 3 or extrinsics_w2c.shape[-2:] not in {(3, 4), (4, 4)}:
        raise GeometryContractError("extrinsics_w2c must have shape [...,S,3,4] or [...,S,4,4]")
    if extrinsics_w2c.shape[-3] < 1:
        raise GeometryContractError("extrinsics_w2c sequence is empty")
    homogeneous = _homogeneous(extrinsics_w2c, "extrinsics_w2c")
    try:
        inverse_first = torch.linalg.inv(homogeneous[..., 0, :, :])
    except RuntimeError as error:
        raise GeometryContractError("first extrinsic is singular") from error
    referenced = homogeneous @ inverse_first.unsqueeze(-3)
    return referenced[..., :3, :]


def normalize_supervision(
    depths: Tensor,
    depth_masks: Tensor,
    intrinsics: Tensor,
    extrinsics_w2c: Tensor,
) -> NormalizedSupervision:
    """Create dense points, first-camera reference poses, and one scene scale.

    This function intentionally accepts one scene sample (``S`` frames), not a
    collated batch.  Dataset samples are normalized independently before the
    data loader adds a batch dimension.
    """

    _require_floating("depths", depths)
    _require_floating("intrinsics", intrinsics)
    _require_floating("extrinsics_w2c", extrinsics_w2c)
    if depths.ndim != 3:
        raise GeometryContractError("depths must have shape [S,H,W]")
    if not isinstance(depth_masks, Tensor) or depth_masks.dtype is not torch.bool:
        raise GeometryContractError("depth_masks must be a bool torch.Tensor")
    if depth_masks.shape != depths.shape:
        raise GeometryContractError("depth_masks shape must equal depths shape")
    sequence_length = depths.shape[0]
    if intrinsics.shape != (sequence_length, 3, 3):
        raise GeometryContractError("intrinsics must have shape [S,3,3]")
    if extrinsics_w2c.shape != (sequence_length, 3, 4):
        raise GeometryContractError("extrinsics_w2c must have shape [S,3,4]")
    if not depth_masks.any():
        raise GeometryContractError("depth mask contains no valid points")
    if (depths[depth_masks] <= 0).any():
        raise GeometryContractError("valid depth values must be positive")

    referenced_extrinsics = reference_extrinsics(extrinsics_w2c)
    camera_points = unproject_depth(depths, intrinsics)
    reference_c2w = camera_to_world_to_world_to_camera(referenced_extrinsics)
    world_points = transform_points(camera_points, reference_c2w)
    point_norms = torch.linalg.vector_norm(world_points[depth_masks], dim=-1)
    scale = point_norms.mean()
    if not torch.isfinite(scale) or scale <= torch.finfo(scale.dtype).eps:
        raise GeometryContractError("mean valid point norm is zero or non-finite")

    normalized_extrinsics = referenced_extrinsics.clone()
    normalized_extrinsics[..., :3, 3] /= scale
    normalized_camera_points = camera_points / scale
    normalized_world_points = world_points / scale
    normalized_camera_points = torch.where(depth_masks.unsqueeze(-1), normalized_camera_points, 0)
    normalized_world_points = torch.where(depth_masks.unsqueeze(-1), normalized_world_points, 0)
    return {
        "depths": depths / scale,
        "depth_masks": depth_masks,
        "intrinsics": intrinsics,
        "extrinsics": normalized_extrinsics,
        "cam_points": normalized_camera_points,
        "world_points": normalized_world_points,
        "scale": scale,
    }
