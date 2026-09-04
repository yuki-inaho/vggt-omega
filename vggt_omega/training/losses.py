"""Supervised camera and mapped-depth losses for VGGT-Omega fine-tuning."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from vggt_omega.utils.pose_enc import extri_intri_to_pose_encoding


def build_camera_pose_target(
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    image_size_hw: tuple[int, int],
) -> torch.Tensor:
    """Build the official ``[t, q_xyzw, FoV_vertical, FoV_horizontal]`` target."""

    _validate_camera_inputs(extrinsics, intrinsics, image_size_hw)
    target = extri_intri_to_pose_encoding(extrinsics, intrinsics, image_size_hw)
    _require_finite("camera_pose_target", target)
    return target


def compute_camera_loss(
    predicted_pose: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    image_size_hw: tuple[int, int],
    *,
    translation_weight: float = 1.0,
    rotation_weight: float = 1.0,
    fov_weight: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Return camera L1 loss and its translation/rotation/FoV components.

    ``mat_to_quat`` inside the official target utility standardizes the XYZW
    target to a non-negative scalar component. The fixed training objective
    therefore compares the predicted and standardized target quaternions
    directly instead of introducing a second hand-written quaternion path.
    """

    target_pose = build_camera_pose_target(extrinsics, intrinsics, image_size_hw)
    if predicted_pose.shape != target_pose.shape:
        raise ValueError(
            f"predicted_pose must have shape {tuple(target_pose.shape)}, got {tuple(predicted_pose.shape)}"
        )
    if not predicted_pose.is_floating_point():
        raise TypeError(f"predicted_pose must be floating point, got {predicted_pose.dtype}")
    _require_finite("predicted_pose", predicted_pose)
    _validate_nonnegative_finite_weight("translation_weight", translation_weight)
    _validate_nonnegative_finite_weight("rotation_weight", rotation_weight)
    _validate_nonnegative_finite_weight("fov_weight", fov_weight)

    target_pose = target_pose.to(device=predicted_pose.device, dtype=predicted_pose.dtype)
    translation = (predicted_pose[..., :3] - target_pose[..., :3]).abs().mean()
    rotation = (predicted_pose[..., 3:7] - target_pose[..., 3:7]).abs().mean()
    fov = (predicted_pose[..., 7:] - target_pose[..., 7:]).abs().mean()
    camera = float(translation_weight) * translation + float(rotation_weight) * rotation + float(fov_weight) * fov
    return {
        "camera": camera,
        "camera_translation": translation,
        "camera_rotation": rotation,
        "camera_fov": fov,
    }


def compute_depth_loss(
    predicted_depth: torch.Tensor,
    target_depth: torch.Tensor,
    depth_mask: torch.Tensor,
    *,
    min_valid_pixels: int = 1,
) -> torch.Tensor:
    """Compute masked depth L1, or a graph-connected zero for sparse/empty masks."""

    if target_depth.ndim != 4:
        raise ValueError(f"target_depth must have shape [B,S,H,W], got {tuple(target_depth.shape)}")
    if target_depth.numel() == 0:
        raise ValueError("target_depth must not be empty")
    expected_prediction_shape = (*target_depth.shape, 1)
    if predicted_depth.shape != expected_prediction_shape:
        raise ValueError(
            "predicted_depth must have target_depth.shape + (1,), "
            f"expected {expected_prediction_shape}, got {tuple(predicted_depth.shape)}"
        )
    if depth_mask.shape != target_depth.shape:
        raise ValueError(f"depth_mask must have shape {tuple(target_depth.shape)}, got {tuple(depth_mask.shape)}")
    if depth_mask.dtype is not torch.bool:
        raise TypeError(f"depth_mask must have dtype bool, got {depth_mask.dtype}")
    if not predicted_depth.is_floating_point() or not target_depth.is_floating_point():
        raise TypeError("predicted_depth and target_depth must be floating point")
    if isinstance(min_valid_pixels, bool) or not isinstance(min_valid_pixels, int) or min_valid_pixels < 1:
        raise ValueError(f"min_valid_pixels must be a positive integer, got {min_valid_pixels!r}")
    _require_finite("predicted_depth", predicted_depth)
    _require_finite("target_depth", target_depth)

    scalar_prediction = predicted_depth[..., 0]
    if int(depth_mask.sum().item()) < min_valid_pixels:
        # Use one finite element rather than a full reduction: summing many
        # individually finite large values can overflow before multiplication
        # by zero, producing NaN instead of the required graph-connected zero.
        return scalar_prediction.reshape(-1)[0] * 0.0
    return (scalar_prediction[depth_mask] - target_depth[depth_mask]).abs().mean()


def compute_camera_depth_loss(
    predictions: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    *,
    camera_weight: float = 5.0,
    depth_weight: float = 1.0,
    translation_weight: float = 1.0,
    rotation_weight: float = 1.0,
    fov_weight: float = 0.5,
    min_valid_depth_pixels: int = 1,
) -> dict[str, torch.Tensor]:
    """Compute the fixed weighted camera/depth objective for one normalized batch."""

    predicted_pose = _require_mapping_tensor(predictions, "pose_enc", "predictions")
    predicted_depth = _require_mapping_tensor(predictions, "depth", "predictions")
    images = _require_mapping_tensor(batch, "images", "batch")
    target_depth = _require_mapping_tensor(batch, "depths", "batch")
    depth_mask = _require_mapping_tensor(batch, "depth_masks", "batch")
    extrinsics = _require_mapping_tensor(batch, "extrinsics", "batch")
    intrinsics = _require_mapping_tensor(batch, "intrinsics", "batch")

    if images.ndim != 5 or images.shape[2] != 3:
        raise ValueError(f"batch images must have shape [B,S,3,H,W], got {tuple(images.shape)}")
    _validate_nonnegative_finite_weight("camera_weight", camera_weight)
    _validate_nonnegative_finite_weight("depth_weight", depth_weight)

    camera_losses = compute_camera_loss(
        predicted_pose,
        extrinsics,
        intrinsics,
        (int(images.shape[-2]), int(images.shape[-1])),
        translation_weight=translation_weight,
        rotation_weight=rotation_weight,
        fov_weight=fov_weight,
    )
    depth = compute_depth_loss(
        predicted_depth,
        target_depth,
        depth_mask,
        min_valid_pixels=min_valid_depth_pixels,
    )
    objective = float(camera_weight) * camera_losses["camera"] + float(depth_weight) * depth
    return {**camera_losses, "depth": depth, "objective": objective}


def _validate_camera_inputs(
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    image_size_hw: tuple[int, int],
) -> None:
    if extrinsics.ndim != 4 or extrinsics.shape[-2:] != (3, 4):
        raise ValueError(f"extrinsics must have shape [B,S,3,4], got {tuple(extrinsics.shape)}")
    if intrinsics.ndim != 4 or intrinsics.shape[-2:] != (3, 3):
        raise ValueError(f"intrinsics must have shape [B,S,3,3], got {tuple(intrinsics.shape)}")
    if extrinsics.shape[:2] != intrinsics.shape[:2]:
        raise ValueError(
            f"extrinsics/intrinsics batch-frame shapes differ: {extrinsics.shape[:2]} != {intrinsics.shape[:2]}"
        )
    if not extrinsics.is_floating_point() or not intrinsics.is_floating_point():
        raise TypeError("extrinsics and intrinsics must be floating point")
    if len(image_size_hw) != 2 or any(isinstance(value, bool) or int(value) <= 0 for value in image_size_hw):
        raise ValueError(f"image_size_hw must contain positive height and width, got {image_size_hw!r}")
    _require_finite("extrinsics", extrinsics)
    _require_finite("intrinsics", intrinsics)
    if torch.any(intrinsics[..., 0, 0] <= 0) or torch.any(intrinsics[..., 1, 1] <= 0):
        raise ValueError("intrinsics focal lengths must be positive")


def _require_mapping_tensor(mapping: Mapping[str, torch.Tensor], key: str, owner: str) -> torch.Tensor:
    value = mapping.get(key)
    if not isinstance(value, torch.Tensor):
        raise KeyError(f"{owner} must contain tensor key {key!r}")
    return value


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN or Inf values")


def _validate_nonnegative_finite_weight(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
