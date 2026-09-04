"""Supervised camera and mapped-depth losses for VGGT-Omega fine-tuning."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from vggt_omega.utils.pose_enc import encoding_to_camera, extri_intri_to_pose_encoding


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


def _relative_pair_transforms(extrinsics: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    frame_count = int(extrinsics.shape[1])
    pair_indices = torch.triu_indices(frame_count, frame_count, offset=1, device=extrinsics.device)
    first_rotation = extrinsics[:, pair_indices[0], :3, :3]
    second_rotation = extrinsics[:, pair_indices[1], :3, :3]
    first_translation = extrinsics[:, pair_indices[0], :3, 3]
    second_translation = extrinsics[:, pair_indices[1], :3, 3]
    relative_rotation = second_rotation @ first_rotation.transpose(-1, -2)
    relative_translation = second_translation - torch.einsum("bpij,bpj->bpi", relative_rotation, first_translation)
    return relative_rotation, relative_translation


def _rotation_geodesic_radians(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    error = predicted @ target.transpose(-1, -2)
    skew = torch.stack(
        (
            error[..., 2, 1] - error[..., 1, 2],
            error[..., 0, 2] - error[..., 2, 0],
            error[..., 1, 0] - error[..., 0, 1],
        ),
        dim=-1,
    )
    sine = torch.linalg.vector_norm(skew, dim=-1) / 2
    cosine = ((torch.diagonal(error, dim1=-2, dim2=-1).sum(dim=-1) - 1) / 2).clamp(-1, 1)
    return torch.atan2(sine, cosine)


def compute_pairwise_pose_loss(
    predicted_pose: torch.Tensor,
    target_extrinsics: torch.Tensor,
    image_size_hw: tuple[int, int],
    *,
    rotation_weight: float = 1.0,
    translation_direction_weight: float = 1.0,
    translation_magnitude_weight: float = 1.0,
    baseline_epsilon: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Compare all relative camera transforms in a sequence."""

    if predicted_pose.ndim != 3 or predicted_pose.shape[-1] != 9:
        raise ValueError("predicted_pose must have shape [B,S,9]")
    if target_extrinsics.shape != (*predicted_pose.shape[:2], 3, 4):
        raise ValueError("target_extrinsics must have shape [B,S,3,4]")
    if not predicted_pose.is_floating_point() or not target_extrinsics.is_floating_point():
        raise TypeError("pairwise pose inputs must be floating point")
    _require_finite("predicted_pose", predicted_pose)
    _require_finite("target_extrinsics", target_extrinsics)
    for name, value in (
        ("relative_rotation_weight", rotation_weight),
        ("relative_translation_direction_weight", translation_direction_weight),
        ("relative_translation_magnitude_weight", translation_magnitude_weight),
    ):
        _validate_nonnegative_finite_weight(name, value)
    if not math.isfinite(baseline_epsilon) or baseline_epsilon <= 0:
        raise ValueError("baseline_epsilon must be finite and positive")
    if predicted_pose.shape[1] < 2:
        zero = predicted_pose.reshape(-1)[0] * 0.0
        return {
            "pairwise_pose": zero,
            "pairwise_rotation": zero,
            "pairwise_translation_direction": zero,
            "pairwise_translation_magnitude": zero,
            "pairwise_valid_direction_fraction": zero,
            "pairwise_rotation_degrees": zero,
            "pairwise_translation_direction_degrees": zero,
            "rpa_5": zero,
            "rpa_15": zero,
            "rpa_30": zero,
        }

    predicted_quaternion = predicted_pose[..., 3:7]
    quaternion_norm = torch.linalg.vector_norm(predicted_quaternion, dim=-1, keepdim=True)
    identity_quaternion = torch.zeros_like(predicted_quaternion)
    identity_quaternion[..., 3] = 1.0
    safe_quaternion = torch.where(
        quaternion_norm > baseline_epsilon,
        predicted_quaternion,
        identity_quaternion,
    )
    decoded_pose = torch.cat((predicted_pose[..., :3], safe_quaternion, predicted_pose[..., 7:]), dim=-1)
    predicted_extrinsics, _ = encoding_to_camera(decoded_pose, image_size_hw, build_intrinsics=False)
    target_extrinsics = target_extrinsics.to(device=predicted_pose.device, dtype=predicted_pose.dtype)
    predicted_rotation, predicted_translation = _relative_pair_transforms(predicted_extrinsics)
    target_rotation, target_translation = _relative_pair_transforms(target_extrinsics)

    rotation_angles = _rotation_geodesic_radians(predicted_rotation, target_rotation)
    rotation = rotation_angles.mean()
    rotation_degrees = rotation_angles * (180 / math.pi)
    predicted_magnitude = torch.linalg.vector_norm(predicted_translation, dim=-1)
    target_magnitude = torch.linalg.vector_norm(target_translation, dim=-1)
    magnitude = (predicted_magnitude - target_magnitude).abs().mean()
    valid_direction = target_magnitude > baseline_epsilon
    if bool(valid_direction.any()):
        predicted_unit = predicted_translation / predicted_magnitude.clamp_min(baseline_epsilon).unsqueeze(-1)
        target_unit = target_translation / target_magnitude.clamp_min(baseline_epsilon).unsqueeze(-1)
        cross = torch.linalg.cross(predicted_unit, target_unit, dim=-1)
        sine = torch.linalg.vector_norm(cross, dim=-1)
        cosine = (predicted_unit * target_unit).sum(dim=-1).clamp(-1, 1)
        direction_angles = torch.atan2(sine, cosine)
        zero_prediction = predicted_magnitude <= baseline_epsilon
        direction_angles = torch.where(zero_prediction & valid_direction, math.pi / 2, direction_angles)
        direction = direction_angles[valid_direction].mean()
        direction_degrees_values = direction_angles * (180 / math.pi)
        direction_degrees = direction_degrees_values[valid_direction].mean()
        rpa = {
            threshold: (
                (rotation_degrees[valid_direction] <= threshold)
                & (direction_degrees_values[valid_direction] <= threshold)
            )
            .to(dtype=predicted_pose.dtype)
            .mean()
            for threshold in (5, 15, 30)
        }
    else:
        direction = predicted_pose.reshape(-1)[0] * 0.0
        direction_degrees = direction
        rpa = dict.fromkeys((5, 15, 30), direction)
    valid_fraction = valid_direction.to(dtype=predicted_pose.dtype).mean()
    pairwise = (
        float(rotation_weight) * rotation
        + float(translation_direction_weight) * direction
        + float(translation_magnitude_weight) * magnitude
    )
    return {
        "pairwise_pose": pairwise,
        "pairwise_rotation": rotation,
        "pairwise_translation_direction": direction,
        "pairwise_translation_magnitude": magnitude,
        "pairwise_valid_direction_fraction": valid_fraction,
        "pairwise_rotation_degrees": rotation_degrees.mean(),
        "pairwise_translation_direction_degrees": direction_degrees,
        "rpa_5": rpa[5],
        "rpa_15": rpa[15],
        "rpa_30": rpa[30],
    }


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
    max_metric_depth_m: float | None = None,
    relative_pose_weight: float = 0.0,
    relative_rotation_weight: float = 1.0,
    relative_translation_direction_weight: float = 1.0,
    relative_translation_magnitude_weight: float = 1.0,
    photometric_weight: float = 0.0,
    renderer_options: Mapping[str, object] | None = None,
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
    _validate_nonnegative_finite_weight("relative_pose_weight", relative_pose_weight)
    _validate_nonnegative_finite_weight("photometric_weight", photometric_weight)

    effective_depth_mask = depth_mask
    if max_metric_depth_m is not None:
        if (
            isinstance(max_metric_depth_m, bool)
            or not isinstance(max_metric_depth_m, (int, float))
            or not math.isfinite(float(max_metric_depth_m))
            or max_metric_depth_m <= 0
        ):
            raise ValueError("max_metric_depth_m must be a finite positive number or None")
        normalization_scale = _require_mapping_tensor(batch, "normalization_scale_m", "batch")
        batch_size = int(target_depth.shape[0])
        if normalization_scale.numel() != batch_size:
            raise ValueError("normalization_scale_m must contain exactly one scale per sample")
        normalization_scale = normalization_scale.to(device=target_depth.device, dtype=target_depth.dtype).reshape(
            batch_size, 1, 1, 1
        )
        _require_finite("normalization_scale_m", normalization_scale)
        if torch.any(normalization_scale <= 0):
            raise ValueError("normalization_scale_m must contain positive values")
        effective_depth_mask = depth_mask & (target_depth * normalization_scale < float(max_metric_depth_m))

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
        effective_depth_mask,
        min_valid_pixels=min_valid_depth_pixels,
    )
    pairwise_losses = compute_pairwise_pose_loss(
        predicted_pose,
        extrinsics,
        (int(images.shape[-2]), int(images.shape[-1])),
        rotation_weight=relative_rotation_weight,
        translation_direction_weight=relative_translation_direction_weight,
        translation_magnitude_weight=relative_translation_magnitude_weight,
    )
    objective = (
        float(camera_weight) * camera_losses["camera"]
        + float(depth_weight) * depth
        + float(relative_pose_weight) * pairwise_losses["pairwise_pose"]
    )
    photometric_losses: dict[str, torch.Tensor] = {}
    if photometric_weight > 0:
        if renderer_options is None:
            raise ValueError("renderer_options are required when photometric_weight is positive")
        from vggt_omega.training.rendering import compute_sequence_photometric_loss

        photometric_losses = compute_sequence_photometric_loss(predictions, batch, **dict(renderer_options))
        objective = objective + float(photometric_weight) * photometric_losses["photometric"]
    return {
        **camera_losses,
        "depth": depth,
        **pairwise_losses,
        **photometric_losses,
        "objective": objective,
    }


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
