"""Optional dynamic-geometry primitives for VGGT-Omega training.

The public tensors in this module use a first-camera canonical frame.  Motion
is a source-pixel-aligned 3D displacement, not optical flow or camera motion.
Unknown labels are represented explicitly and never inferred as static.
"""

from __future__ import annotations

import math
from itertools import pairwise

import torch
import torch.nn.functional as F
from torch import nn


def build_temporal_pairs(
    frame_ids: torch.Tensor,
    frame_mask: torch.Tensor,
    *,
    motion_pair_indices: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Build time-sorted adjacent pairs in both directions and pad the batch."""

    if frame_ids.ndim != 2 or frame_ids.dtype != torch.long:
        raise ValueError("frame_ids must be int64 with shape [B,S]")
    if frame_mask.shape != frame_ids.shape or frame_mask.dtype is not torch.bool:
        raise ValueError("frame_mask must be bool with shape [B,S]")
    if frame_ids.device != frame_mask.device:
        raise ValueError("frame_ids and frame_mask must share a device")
    batch_size, frame_count = frame_ids.shape
    if batch_size == 0 or frame_count == 0:
        raise ValueError("frame_ids must contain a non-empty batch and frame dimension")
    if not frame_mask[:, 0].all():
        raise ValueError("position zero must be a valid canonical reference")

    valid_counts = frame_mask.sum(dim=1)
    if torch.any(valid_counts < 2):
        raise ValueError("each sample must contain at least two valid frames")
    pair_count = 2 * (int(valid_counts.max().item()) - 1)
    generated = torch.full(
        (batch_size, pair_count, 2),
        -1,
        dtype=torch.long,
        device=frame_ids.device,
    )
    valid_pairs = torch.zeros((batch_size, pair_count), dtype=torch.bool, device=frame_ids.device)
    deltas = torch.zeros((batch_size, pair_count), dtype=torch.long, device=frame_ids.device)

    for batch_index in range(batch_size):
        positions = torch.nonzero(frame_mask[batch_index], as_tuple=False).flatten()
        active_ids = frame_ids[batch_index, positions]
        if torch.any(active_ids < 0):
            raise ValueError("active frame IDs must be non-negative")
        if torch.unique(active_ids).numel() != active_ids.numel():
            raise ValueError("duplicate active frame IDs are not allowed")
        order = torch.argsort(active_ids, stable=True)
        positions = positions[order]
        cursor = 0
        for left, right in pairwise(positions):
            generated[batch_index, cursor] = torch.stack((left, right))
            generated[batch_index, cursor + 1] = torch.stack((right, left))
            cursor += 2
        valid_pairs[batch_index, :cursor] = True
        source = generated[batch_index, :cursor, 0]
        target = generated[batch_index, :cursor, 1]
        deltas[batch_index, :cursor] = frame_ids[batch_index, target] - frame_ids[batch_index, source]
        if torch.any(deltas[batch_index, :cursor] == 0):
            raise ValueError("temporal pairs must have a non-zero frame delta")

    if motion_pair_indices is not None and (
        motion_pair_indices.dtype != torch.long
        or motion_pair_indices.device != frame_ids.device
        or motion_pair_indices.shape != generated.shape
        or not torch.equal(motion_pair_indices, generated)
    ):
        raise ValueError("motion_pair_indices must equal the adjacent_bidirectional graph")

    return {
        "motion_pair_indices": generated,
        "motion_pair_valid_mask": valid_pairs,
        "motion_time_delta_frames": deltas,
    }


def canonical_points_from_depth(
    depths: torch.Tensor,
    depth_masks: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics_w2c: torch.Tensor,
    frame_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Unproject depth into the position-zero camera coordinate system."""

    batch_size, _, height, width = _validate_geometry_inputs(
        depths,
        depth_masks,
        intrinsics,
        extrinsics_w2c,
        frame_mask,
    )
    homogeneous = _homogeneous_extrinsics(extrinsics_w2c)
    reference_inverse = torch.linalg.inv(homogeneous[:, :1])
    rebased_homogeneous = homogeneous @ reference_inverse
    identity = torch.eye(4, dtype=rebased_homogeneous.dtype, device=rebased_homogeneous.device)
    if not torch.allclose(rebased_homogeneous[:, 0], identity.expand(batch_size, 4, 4), atol=1e-5, rtol=0):
        raise ValueError("failed to rebase the position-zero camera to identity")

    inverse_intrinsics = torch.linalg.inv(intrinsics)
    pixel_grid = _pixel_grid(height, width, dtype=depths.dtype, device=depths.device)
    rays = torch.einsum("bsij,hwj->bshwi", inverse_intrinsics, pixel_grid)
    depth_valid = depth_masks & frame_mask[:, :, None, None] & torch.isfinite(depths) & (depths > 0)
    camera_points = rays * torch.where(depth_valid, depths, torch.zeros_like(depths)).unsqueeze(-1)
    rotation = rebased_homogeneous[..., :3, :3]
    translation = rebased_homogeneous[..., :3, 3]
    canonical_points = torch.einsum(
        "bsji,bshwj->bshwi",
        rotation,
        camera_points - translation[:, :, None, None, :],
    )
    finite_points = torch.isfinite(canonical_points).all(dim=-1)
    valid = depth_valid & finite_points
    canonical_points = torch.where(valid.unsqueeze(-1), canonical_points, torch.zeros_like(canonical_points))
    return {
        "canonical_points_current": canonical_points,
        "canonical_points_valid_mask": valid,
        "rebased_extrinsics_w2c": rebased_homogeneous[..., :3, :],
    }


def partition_dynamic_probability(
    dynamic_probability: torch.Tensor,
    motion_visibility_probability: torch.Tensor,
    motion_domain_mask: torch.Tensor,
    *,
    ready: bool,
    visibility_threshold: float,
    static_probability_max: float,
    dynamic_probability_min: float,
) -> dict[str, torch.Tensor]:
    """Partition valid motion pixels into dynamic, static, and unknown sets."""

    _validate_probability_tensor("dynamic_probability", dynamic_probability)
    _validate_probability_tensor("motion_visibility_probability", motion_visibility_probability)
    if dynamic_probability.shape != motion_visibility_probability.shape:
        raise ValueError("probability tensors must have matching shapes")
    if motion_domain_mask.shape != dynamic_probability.shape or motion_domain_mask.dtype is not torch.bool:
        raise ValueError("motion_domain_mask must be bool and match the probability shape")
    if motion_domain_mask.device != dynamic_probability.device:
        raise ValueError("probability tensors and motion_domain_mask must share a device")
    if not isinstance(ready, bool):
        raise ValueError("ready must be boolean")
    visibility_threshold = _unit_interval("visibility_threshold", visibility_threshold)
    static_probability_max = _unit_interval("static_probability_max", static_probability_max)
    dynamic_probability_min = _unit_interval("dynamic_probability_min", dynamic_probability_min)
    if static_probability_max >= dynamic_probability_min:
        raise ValueError("static_probability_max must be less than dynamic_probability_min")

    if not ready:
        visible = torch.zeros_like(motion_domain_mask)
        dynamic = torch.zeros_like(motion_domain_mask)
        static = torch.zeros_like(motion_domain_mask)
        unknown = motion_domain_mask.clone()
    else:
        visible = motion_domain_mask & (motion_visibility_probability >= visibility_threshold)
        dynamic = visible & (dynamic_probability >= dynamic_probability_min)
        static = visible & (dynamic_probability <= static_probability_max)
        unknown = motion_domain_mask & ~(dynamic | static)
    return {
        "motion_visibility_mask": visible,
        "dynamic_mask": dynamic,
        "static_mask": static,
        "dynamic_unknown_mask": unknown,
    }


class CanonicalMotionHead(nn.Module):
    """Small pair-factorized decoder for source-aligned canonical motion."""

    def __init__(self, *, feature_dim: int, hidden_dim: int, relative_camera_dim: int) -> None:
        super().__init__()
        for name, value in (
            ("feature_dim", feature_dim),
            ("hidden_dim", hidden_dim),
            ("relative_camera_dim", relative_camera_dim),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.feature_dim = feature_dim
        self.relative_camera_dim = relative_camera_dim
        pair_dim = 2 * feature_dim + relative_camera_dim + 1
        self.pair_encoder = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.flow_decoder = nn.Linear(hidden_dim, 3)
        self.visibility_decoder = nn.Linear(hidden_dim, 1)
        self.dynamic_decoder = nn.Linear(hidden_dim, 1)
        self._reset_output_layers()

    def _reset_output_layers(self) -> None:
        nn.init.zeros_(self.flow_decoder.weight)
        nn.init.zeros_(self.flow_decoder.bias)
        initial_logit = math.log(0.01 / 0.99)
        for decoder in (self.visibility_decoder, self.dynamic_decoder):
            nn.init.zeros_(decoder.weight)
            nn.init.constant_(decoder.bias, initial_logit)

    def forward(
        self,
        patch_features: torch.Tensor,
        relative_camera: torch.Tensor,
        motion_time_delta_frames: torch.Tensor,
        motion_pair_indices: torch.Tensor,
        motion_pair_valid_mask: torch.Tensor,
        *,
        patch_grid_hw: tuple[int, int],
        output_hw: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        batch_size, _, patch_height, patch_width = self._validate_forward_inputs(
            patch_features,
            relative_camera,
            motion_time_delta_frames,
            motion_pair_indices,
            motion_pair_valid_mask,
            patch_grid_hw,
            output_hw,
        )
        safe_pairs = motion_pair_indices.clamp_min(0)
        batch_indices = torch.arange(batch_size, device=patch_features.device)[:, None]
        source = patch_features[batch_indices, safe_pairs[..., 0]]
        target = patch_features[batch_indices, safe_pairs[..., 1]]
        camera = relative_camera[:, :, None, :].expand(-1, -1, source.shape[2], -1)
        delta = motion_time_delta_frames.to(patch_features.dtype)[:, :, None, None]
        delta = delta.expand(-1, -1, source.shape[2], -1)
        encoded = self.pair_encoder(torch.cat((source, target - source, camera, delta), dim=-1))

        flow = self._decode_dense(self.flow_decoder(encoded), patch_height, patch_width, output_hw)
        visibility_logits = self._decode_dense(
            self.visibility_decoder(encoded), patch_height, patch_width, output_hw
        ).squeeze(-1)
        dynamic_logits = self._decode_dense(
            self.dynamic_decoder(encoded), patch_height, patch_width, output_hw
        ).squeeze(-1)
        visibility = torch.sigmoid(visibility_logits)
        dynamic = torch.sigmoid(dynamic_logits)
        dense_valid = motion_pair_valid_mask[:, :, None, None]
        return {
            "canonical_scene_flow": torch.where(dense_valid.unsqueeze(-1), flow, torch.zeros_like(flow)),
            "motion_visibility_logits": torch.where(
                dense_valid, visibility_logits, torch.zeros_like(visibility_logits)
            ),
            "motion_visibility_probability": torch.where(dense_valid, visibility, torch.zeros_like(visibility)),
            "dynamic_logits": torch.where(dense_valid, dynamic_logits, torch.zeros_like(dynamic_logits)),
            "dynamic_probability": torch.where(dense_valid, dynamic, torch.zeros_like(dynamic)),
        }

    @staticmethod
    def _decode_dense(
        patch_values: torch.Tensor,
        patch_height: int,
        patch_width: int,
        output_hw: tuple[int, int],
    ) -> torch.Tensor:
        batch_size, pair_count, _, channels = patch_values.shape
        values = patch_values.reshape(batch_size * pair_count, patch_height, patch_width, channels).permute(0, 3, 1, 2)
        values = F.interpolate(values, size=output_hw, mode="bilinear", align_corners=False)
        return values.permute(0, 2, 3, 1).reshape(batch_size, pair_count, *output_hw, channels)

    def _validate_forward_inputs(
        self,
        patch_features: torch.Tensor,
        relative_camera: torch.Tensor,
        motion_time_delta_frames: torch.Tensor,
        motion_pair_indices: torch.Tensor,
        motion_pair_valid_mask: torch.Tensor,
        patch_grid_hw: tuple[int, int],
        output_hw: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        if patch_features.ndim != 4 or patch_features.shape[-1] != self.feature_dim:
            raise ValueError(f"patch_features must have shape [B,S,P,{self.feature_dim}]")
        if not patch_features.is_floating_point() or not torch.isfinite(patch_features).all():
            raise ValueError("patch_features must be finite and floating point")
        batch_size, frame_count, patch_count, _ = patch_features.shape
        if (
            motion_pair_indices.ndim != 3
            or motion_pair_indices.shape[0] != batch_size
            or motion_pair_indices.shape[2] != 2
        ):
            raise ValueError("motion_pair_indices must have shape [B,Q,2]")
        pair_count = motion_pair_indices.shape[1]
        if motion_pair_indices.dtype != torch.long:
            raise ValueError("motion_pair_indices must be int64")
        if motion_pair_valid_mask.shape != (batch_size, pair_count) or motion_pair_valid_mask.dtype is not torch.bool:
            raise ValueError("motion_pair_valid_mask must be bool with shape [B,Q]")
        if relative_camera.shape != (batch_size, pair_count, self.relative_camera_dim):
            raise ValueError(f"relative_camera must have shape [B,Q,{self.relative_camera_dim}]")
        if not relative_camera.is_floating_point() or not torch.isfinite(relative_camera).all():
            raise ValueError("relative_camera must be finite and floating point")
        if motion_time_delta_frames.shape != (batch_size, pair_count) or motion_time_delta_frames.dtype != torch.long:
            raise ValueError("motion_time_delta_frames must be int64 with shape [B,Q]")
        tensors = (
            relative_camera,
            motion_time_delta_frames,
            motion_pair_indices,
            motion_pair_valid_mask,
        )
        if any(tensor.device != patch_features.device for tensor in tensors):
            raise ValueError("all motion-head inputs must share a device")
        _validate_hw("patch_grid_hw", patch_grid_hw)
        _validate_hw("output_hw", output_hw)
        patch_height, patch_width = patch_grid_hw
        if patch_height * patch_width != patch_count:
            raise ValueError("patch_grid_hw does not match the patch token count")
        valid_indices = motion_pair_indices[motion_pair_valid_mask]
        if valid_indices.numel() and (
            torch.any(valid_indices < 0)
            or torch.any(valid_indices >= frame_count)
            or torch.any(valid_indices[:, 0] == valid_indices[:, 1])
        ):
            raise ValueError("valid temporal pairs contain invalid or identical frame indices")
        if torch.any(motion_pair_indices[~motion_pair_valid_mask] != -1):
            raise ValueError("padded pair indices must be -1")
        if torch.any(motion_time_delta_frames[motion_pair_valid_mask] == 0):
            raise ValueError("valid temporal pairs must have non-zero frame deltas")
        return batch_size, pair_count, patch_height, patch_width


def build_rgbd_motion_targets(
    depths: torch.Tensor,
    depth_masks: torch.Tensor,
    original_depth_observed_mask: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics_w2c: torch.Tensor,
    normalization_scale_m: torch.Tensor,
    frame_ids: torch.Tensor,
    frame_mask: torch.Tensor,
    motion_pair_indices: torch.Tensor,
    pixel_flow_xy: torch.Tensor | None,
    flow_confidence: torch.Tensor | None,
    *,
    flow_occlusion_label: torch.Tensor | None = None,
    depth_unit: str = "scene_normalized",
    static_off_m: float = 0.01,
    dynamic_on_m: float = 0.03,
    flow_confidence_min: float = 0.8,
    forward_backward_cycle_px: float = 1.0,
    depth_discontinuity_relative: float = 0.03,
) -> dict[str, torch.Tensor]:
    """Create conservative camera-compensated motion labels from RGB-D and flow."""

    if depth_unit != "scene_normalized":
        raise ValueError("depth_unit must be 'scene_normalized'")
    batch_size, _, height, width = _validate_geometry_inputs(
        depths,
        depth_masks,
        intrinsics,
        extrinsics_w2c,
        frame_mask,
    )
    if original_depth_observed_mask.shape != depths.shape or original_depth_observed_mask.dtype is not torch.bool:
        raise ValueError("original_depth_observed_mask must be bool and match depths")
    if original_depth_observed_mask.device != depths.device:
        raise ValueError("original_depth_observed_mask must share the geometry device")
    if torch.any(original_depth_observed_mask & ~depth_masks):
        raise ValueError("original_depth_observed_mask must be a subset of depth_masks")
    if normalization_scale_m.shape != (batch_size,) or not normalization_scale_m.is_floating_point():
        raise ValueError("normalization_scale_m must be floating point with shape [B]")
    if (
        normalization_scale_m.device != depths.device
        or not torch.isfinite(normalization_scale_m).all()
        or torch.any(normalization_scale_m <= 0)
    ):
        raise ValueError("normalization_scale_m must be finite, positive, and share the geometry device")
    pair_metadata = build_temporal_pairs(frame_ids, frame_mask, motion_pair_indices=motion_pair_indices)
    pair_valid = pair_metadata["motion_pair_valid_mask"]
    pair_count = motion_pair_indices.shape[1]

    static_off_m = _nonnegative_finite_float("static_off_m", static_off_m)
    dynamic_on_m = _nonnegative_finite_float("dynamic_on_m", dynamic_on_m)
    if static_off_m >= dynamic_on_m:
        raise ValueError("static_off_m must be less than dynamic_on_m")
    flow_confidence_min = _unit_interval("flow_confidence_min", flow_confidence_min)
    forward_backward_cycle_px = _nonnegative_finite_float("forward_backward_cycle_px", forward_backward_cycle_px)
    depth_discontinuity_relative = _nonnegative_finite_float(
        "depth_discontinuity_relative", depth_discontinuity_relative
    )
    depths_fp32 = depths.float()
    intrinsics_fp32 = intrinsics.float()
    extrinsics_fp32 = extrinsics_w2c.float()
    normalization_scale_fp32 = normalization_scale_m.float()
    pixel_flow_fp32 = None if pixel_flow_xy is None else pixel_flow_xy.float()
    flow_confidence_fp32 = None if flow_confidence is None else flow_confidence.float()
    with torch.autocast(device_type=depths.device.type, enabled=False):
        geometry = canonical_points_from_depth(
            depths_fp32,
            depth_masks,
            intrinsics_fp32,
            extrinsics_fp32,
            frame_mask,
        )
    current_points = geometry["canonical_points_current"]
    current_valid = geometry["canonical_points_valid_mask"]

    flow_target = torch.zeros(
        (batch_size, pair_count, height, width, 3),
        dtype=torch.float32,
        device=depths.device,
    )
    metric_flow_target = torch.zeros_like(flow_target)
    visibility_label = torch.full(
        (batch_size, pair_count, height, width),
        -1,
        dtype=torch.int8,
        device=depths.device,
    )
    visibility_known = torch.zeros_like(visibility_label, dtype=torch.bool)
    visibility_confidence = torch.zeros(visibility_label.shape, dtype=torch.float32, device=depths.device)
    dynamic_label = torch.full_like(visibility_label, -1)
    target_confidence = torch.zeros(visibility_label.shape, dtype=torch.float32, device=depths.device)

    teacher_available = pixel_flow_xy is not None and flow_confidence is not None
    if teacher_available:
        assert pixel_flow_fp32 is not None
        assert flow_confidence_fp32 is not None
        _validate_flow_teacher(
            pixel_flow_fp32,
            flow_confidence_fp32,
            batch_size,
            pair_count,
            height,
            width,
            depths_fp32,
        )
        if flow_occlusion_label is not None:
            if flow_occlusion_label.shape != visibility_label.shape or flow_occlusion_label.dtype != torch.int8:
                raise ValueError("flow_occlusion_label must be int8 with shape [B,Q,H,W]")
            if flow_occlusion_label.device != depths.device:
                raise ValueError("flow_occlusion_label must share the geometry device")
            if torch.any((flow_occlusion_label < -1) | (flow_occlusion_label > 1)):
                raise ValueError("flow_occlusion_label values must be -1, 0, or 1")
        with torch.autocast(device_type=depths.device.type, enabled=False):
            _populate_motion_targets(
                current_points=current_points,
                current_valid=current_valid,
                depths=depths_fp32,
                depth_masks=depth_masks,
                original_depth_observed_mask=original_depth_observed_mask,
                normalization_scale_m=normalization_scale_fp32,
                pair_indices=motion_pair_indices,
                pair_valid=pair_valid,
                pixel_flow_xy=pixel_flow_fp32,
                flow_confidence=flow_confidence_fp32,
                flow_occlusion_label=flow_occlusion_label,
                flow_target=flow_target,
                metric_flow_target=metric_flow_target,
                visibility_label=visibility_label,
                visibility_known=visibility_known,
                visibility_confidence=visibility_confidence,
                dynamic_label=dynamic_label,
                target_confidence=target_confidence,
                static_off_m=static_off_m,
                dynamic_on_m=dynamic_on_m,
                flow_confidence_min=flow_confidence_min,
                forward_backward_cycle_px=forward_backward_cycle_px,
                depth_discontinuity_relative=depth_discontinuity_relative,
            )

    return {
        "target_canonical_scene_flow": flow_target,
        "target_scene_flow_m": metric_flow_target,
        "target_visibility_label": visibility_label,
        "target_visibility_known_mask": visibility_known,
        "target_visibility_confidence": visibility_confidence,
        "target_dynamic_label": dynamic_label,
        "target_confidence": target_confidence,
        "motion_frame_ids": frame_ids,
        "motion_frame_mask": frame_mask,
        "motion_pair_indices": motion_pair_indices,
        **pair_metadata,
    }


def _populate_motion_targets(
    *,
    current_points: torch.Tensor,
    current_valid: torch.Tensor,
    depths: torch.Tensor,
    depth_masks: torch.Tensor,
    original_depth_observed_mask: torch.Tensor,
    normalization_scale_m: torch.Tensor,
    pair_indices: torch.Tensor,
    pair_valid: torch.Tensor,
    pixel_flow_xy: torch.Tensor,
    flow_confidence: torch.Tensor,
    flow_occlusion_label: torch.Tensor | None,
    flow_target: torch.Tensor,
    metric_flow_target: torch.Tensor,
    visibility_label: torch.Tensor,
    visibility_known: torch.Tensor,
    visibility_confidence: torch.Tensor,
    dynamic_label: torch.Tensor,
    target_confidence: torch.Tensor,
    static_off_m: float,
    dynamic_on_m: float,
    flow_confidence_min: float,
    forward_backward_cycle_px: float,
    depth_discontinuity_relative: float,
) -> None:
    batch_size, pair_count, height, width, _ = pixel_flow_xy.shape
    pixel_grid = _pixel_grid(height, width, dtype=depths.dtype, device=depths.device)[..., :2]
    reverse_lookup = _reverse_pair_lookup(pair_indices, pair_valid)
    for batch_index in range(batch_size):
        for pair_index in range(pair_count):
            if not bool(pair_valid[batch_index, pair_index]):
                continue
            source_index = int(pair_indices[batch_index, pair_index, 0].item())
            target_index = int(pair_indices[batch_index, pair_index, 1].item())
            reverse_index = reverse_lookup[batch_index][pair_index]
            if reverse_index < 0:
                continue
            _populate_one_pair(
                source_points=current_points[batch_index, source_index],
                source_valid=current_valid[batch_index, source_index]
                & original_depth_observed_mask[batch_index, source_index],
                target_points=current_points[batch_index, target_index],
                target_depth=depths[batch_index, target_index],
                target_depth_mask=depth_masks[batch_index, target_index],
                target_observed=original_depth_observed_mask[batch_index, target_index],
                forward_flow=pixel_flow_xy[batch_index, pair_index],
                reverse_flow=pixel_flow_xy[batch_index, reverse_index],
                forward_confidence=flow_confidence[batch_index, pair_index],
                reverse_confidence=flow_confidence[batch_index, reverse_index],
                explicit_occlusion=None
                if flow_occlusion_label is None
                else flow_occlusion_label[batch_index, pair_index],
                pixel_grid=pixel_grid,
                normalization_scale_m=normalization_scale_m[batch_index],
                output_flow=flow_target[batch_index, pair_index],
                output_metric_flow=metric_flow_target[batch_index, pair_index],
                output_visibility_label=visibility_label[batch_index, pair_index],
                output_visibility_known=visibility_known[batch_index, pair_index],
                output_visibility_confidence=visibility_confidence[batch_index, pair_index],
                output_dynamic_label=dynamic_label[batch_index, pair_index],
                output_confidence=target_confidence[batch_index, pair_index],
                static_off_m=static_off_m,
                dynamic_on_m=dynamic_on_m,
                flow_confidence_min=flow_confidence_min,
                forward_backward_cycle_px=forward_backward_cycle_px,
                depth_discontinuity_relative=depth_discontinuity_relative,
            )


def _populate_one_pair(
    *,
    source_points: torch.Tensor,
    source_valid: torch.Tensor,
    target_points: torch.Tensor,
    target_depth: torch.Tensor,
    target_depth_mask: torch.Tensor,
    target_observed: torch.Tensor,
    forward_flow: torch.Tensor,
    reverse_flow: torch.Tensor,
    forward_confidence: torch.Tensor,
    reverse_confidence: torch.Tensor,
    explicit_occlusion: torch.Tensor | None,
    pixel_grid: torch.Tensor,
    normalization_scale_m: torch.Tensor,
    output_flow: torch.Tensor,
    output_metric_flow: torch.Tensor,
    output_visibility_label: torch.Tensor,
    output_visibility_known: torch.Tensor,
    output_visibility_confidence: torch.Tensor,
    output_dynamic_label: torch.Tensor,
    output_confidence: torch.Tensor,
    static_off_m: float,
    dynamic_on_m: float,
    flow_confidence_min: float,
    forward_backward_cycle_px: float,
    depth_discontinuity_relative: float,
) -> None:
    height, width = source_valid.shape
    flow_finite = torch.isfinite(forward_flow).all(dim=-1)
    confidence_finite = torch.isfinite(forward_confidence)
    reliable_source = source_valid & flow_finite & confidence_finite & (forward_confidence >= flow_confidence_min)
    target_xy = pixel_grid + torch.where(flow_finite.unsqueeze(-1), forward_flow, torch.zeros_like(forward_flow))
    in_bounds = (
        (target_xy[..., 0] >= 0)
        & (target_xy[..., 0] <= width - 1)
        & (target_xy[..., 1] >= 0)
        & (target_xy[..., 1] <= height - 1)
    )

    known_oob = reliable_source & ~in_bounds
    if explicit_occlusion is not None:
        known_occluded = reliable_source & (explicit_occlusion == 0)
    else:
        known_occluded = torch.zeros_like(source_valid)
    known_occluded |= known_oob
    output_visibility_label[known_occluded] = 0
    output_visibility_known[known_occluded] = True
    output_visibility_confidence[known_occluded] = forward_confidence[known_occluded].clamp(0, 1)

    reverse_sample, reverse_confidence_sample, stencil_valid = _sample_reverse_teacher(
        reverse_flow,
        reverse_confidence,
        target_xy,
    )
    reverse_finite = torch.isfinite(reverse_sample).all(dim=-1) & torch.isfinite(reverse_confidence_sample)
    cycle_error = torch.linalg.vector_norm(forward_flow + reverse_sample, dim=-1)
    cycle_pass = (
        stencil_valid
        & reverse_finite
        & (reverse_confidence_sample >= flow_confidence_min)
        & (cycle_error <= forward_backward_cycle_px)
    )
    selected = _select_target_neighbors(
        target_xy,
        target_depth,
        target_depth_mask,
        target_observed,
        depth_discontinuity_relative=depth_discontinuity_relative,
    )
    candidate = reliable_source & in_bounds & cycle_pass & selected["known"] & ~known_occluded
    if explicit_occlusion is not None:
        candidate &= explicit_occlusion != 0
    selected_points = target_points.reshape(height * width, 3)[selected["flat_index"]]
    scene_flow = selected_points - source_points
    scene_flow_m = scene_flow * normalization_scale_m
    finite_motion = torch.isfinite(scene_flow_m).all(dim=-1)
    candidate &= finite_motion
    confidence = forward_confidence.clamp(0, 1) * reverse_confidence_sample.clamp(0, 1) * selected["offset_confidence"]
    candidate &= confidence > 0

    output_flow[candidate] = scene_flow[candidate]
    output_metric_flow[candidate] = scene_flow_m[candidate]
    output_visibility_label[candidate] = 1
    output_visibility_known[candidate] = True
    output_visibility_confidence[candidate] = confidence[candidate]
    output_confidence[candidate] = confidence[candidate]
    motion_norm = torch.linalg.vector_norm(scene_flow_m, dim=-1)
    static = candidate & (motion_norm <= static_off_m)
    dynamic = candidate & (motion_norm >= dynamic_on_m)
    output_dynamic_label[static] = 0
    output_dynamic_label[dynamic] = 1


def _sample_reverse_teacher(
    reverse_flow: torch.Tensor,
    reverse_confidence: torch.Tensor,
    target_xy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    height, width = reverse_confidence.shape
    safe_xy = torch.where(torch.isfinite(target_xy), target_xy, torch.zeros_like(target_xy))
    normalized_x = 2 * (safe_xy[..., 0] + 0.5) / width - 1
    normalized_y = 2 * (safe_xy[..., 1] + 0.5) / height - 1
    sample_grid = torch.stack((normalized_x, normalized_y), dim=-1)[None]
    sampled_flow = F.grid_sample(
        reverse_flow.permute(2, 0, 1)[None],
        sample_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[0].permute(1, 2, 0)
    sampled_confidence = F.grid_sample(
        reverse_confidence[None, None],
        sample_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[0, 0]
    floor_xy = torch.floor(safe_xy)
    stencil_valid = (
        (floor_xy[..., 0] >= 0)
        & (floor_xy[..., 0] + 1 < width)
        & (floor_xy[..., 1] >= 0)
        & (floor_xy[..., 1] + 1 < height)
        & torch.isfinite(target_xy).all(dim=-1)
    )
    return sampled_flow, sampled_confidence, stencil_valid


def _select_target_neighbors(
    target_xy: torch.Tensor,
    target_depth: torch.Tensor,
    target_depth_mask: torch.Tensor,
    target_observed: torch.Tensor,
    *,
    depth_discontinuity_relative: float,
) -> dict[str, torch.Tensor]:
    height, width = target_depth.shape
    base = torch.floor(torch.where(torch.isfinite(target_xy), target_xy, torch.zeros_like(target_xy))).long()
    best_distance = torch.full((height, width), torch.inf, dtype=target_depth.dtype, device=target_depth.device)
    best_flat = torch.zeros((height, width), dtype=torch.long, device=target_depth.device)
    candidate_depths: list[torch.Tensor] = []
    candidate_observed: list[torch.Tensor] = []
    for offset_y in (-1, 0, 1):
        for offset_x in (-1, 0, 1):
            candidate_x = base[..., 0] + offset_x
            candidate_y = base[..., 1] + offset_y
            in_bounds = (candidate_x >= 0) & (candidate_x < width) & (candidate_y >= 0) & (candidate_y < height)
            safe_x = candidate_x.clamp(0, width - 1)
            safe_y = candidate_y.clamp(0, height - 1)
            flat = safe_y * width + safe_x
            depth = target_depth.reshape(-1)[flat]
            valid = in_bounds & target_depth_mask.reshape(-1)[flat] & torch.isfinite(depth) & (depth > 0)
            observed = valid & target_observed.reshape(-1)[flat]
            distance = (candidate_x.to(target_xy.dtype) - target_xy[..., 0]).square() + (
                candidate_y.to(target_xy.dtype) - target_xy[..., 1]
            ).square()
            better = valid & ((distance < best_distance) | ((distance == best_distance) & (flat < best_flat)))
            best_distance = torch.where(better, distance, best_distance)
            best_flat = torch.where(better, flat, best_flat)
            candidate_depths.append(torch.where(observed, depth, torch.full_like(depth, torch.nan)))
            candidate_observed.append(observed)
    local_depths = torch.stack(candidate_depths, dim=-1)
    local_count = torch.stack(candidate_observed, dim=-1).sum(dim=-1)
    local_median = torch.nanmedian(local_depths, dim=-1).values
    selected_depth = target_depth.reshape(-1)[best_flat]
    selected_observed = target_observed.reshape(-1)[best_flat]
    discontinuity = (selected_depth - local_median).abs() / local_median.abs().clamp_min(1e-6)
    known = (
        torch.isfinite(best_distance)
        & selected_observed
        & (local_count >= 3)
        & torch.isfinite(local_median)
        & (discontinuity <= depth_discontinuity_relative)
    )
    offset_confidence = torch.exp(-best_distance / 2)
    offset_confidence = torch.where(known, offset_confidence, torch.zeros_like(offset_confidence))
    return {"flat_index": best_flat, "known": known, "offset_confidence": offset_confidence}


def _reverse_pair_lookup(pair_indices: torch.Tensor, pair_valid: torch.Tensor) -> list[list[int]]:
    lookup: list[list[int]] = []
    for batch_index in range(pair_indices.shape[0]):
        reverse_by_pair = [-1] * pair_indices.shape[1]
        mapping: dict[tuple[int, int], int] = {}
        for pair_index in range(pair_indices.shape[1]):
            if bool(pair_valid[batch_index, pair_index]):
                source, target = pair_indices[batch_index, pair_index].tolist()
                mapping[(source, target)] = pair_index
        for pair_index in range(pair_indices.shape[1]):
            if bool(pair_valid[batch_index, pair_index]):
                source, target = pair_indices[batch_index, pair_index].tolist()
                reverse_by_pair[pair_index] = mapping.get((target, source), -1)
        lookup.append(reverse_by_pair)
    return lookup


def _validate_geometry_inputs(
    depths: torch.Tensor,
    depth_masks: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics_w2c: torch.Tensor,
    frame_mask: torch.Tensor,
) -> tuple[int, int, int, int]:
    if depths.ndim != 4 or not depths.is_floating_point():
        raise ValueError("depths must be floating point with shape [B,S,H,W]")
    batch_size, frame_count, height, width = depths.shape
    if min(batch_size, frame_count, height, width) <= 0:
        raise ValueError("depths dimensions must be non-empty")
    if depth_masks.shape != depths.shape or depth_masks.dtype is not torch.bool:
        raise ValueError("depth_masks must be bool and match depths")
    if intrinsics.shape != (batch_size, frame_count, 3, 3):
        raise ValueError("intrinsics must have shape [B,S,3,3]")
    if extrinsics_w2c.shape != (batch_size, frame_count, 3, 4):
        raise ValueError("extrinsics_w2c must have shape [B,S,3,4]")
    if frame_mask.shape != (batch_size, frame_count) or frame_mask.dtype is not torch.bool:
        raise ValueError("frame_mask must be bool with shape [B,S]")
    tensors = (depth_masks, intrinsics, extrinsics_w2c, frame_mask)
    if any(tensor.device != depths.device for tensor in tensors):
        raise ValueError("geometry tensors must share a device")
    if not intrinsics.is_floating_point() or not extrinsics_w2c.is_floating_point():
        raise ValueError("intrinsics and extrinsics_w2c must be floating point")
    if not torch.isfinite(intrinsics).all() or not torch.isfinite(extrinsics_w2c).all():
        raise ValueError("camera parameters must be finite")
    valid_depth = depth_masks & frame_mask[:, :, None, None]
    if torch.any(valid_depth & (~torch.isfinite(depths) | (depths <= 0))):
        raise ValueError("valid depths must be finite and positive")
    if not frame_mask[:, 0].all():
        raise ValueError("position zero must be a valid canonical reference")
    return batch_size, frame_count, height, width


def _validate_flow_teacher(
    pixel_flow_xy: torch.Tensor,
    flow_confidence: torch.Tensor,
    batch_size: int,
    pair_count: int,
    height: int,
    width: int,
    reference: torch.Tensor,
) -> None:
    if pixel_flow_xy.shape != (batch_size, pair_count, height, width, 2):
        raise ValueError("pixel_flow_xy must have shape [B,Q,H,W,2]")
    if flow_confidence.shape != (batch_size, pair_count, height, width):
        raise ValueError("flow_confidence must have shape [B,Q,H,W]")
    if not pixel_flow_xy.is_floating_point() or not flow_confidence.is_floating_point():
        raise ValueError("flow teacher tensors must be floating point")
    if pixel_flow_xy.device != reference.device or flow_confidence.device != reference.device:
        raise ValueError("flow teacher tensors must share the geometry device")


def _homogeneous_extrinsics(extrinsics_w2c: torch.Tensor) -> torch.Tensor:
    batch_size, frame_count = extrinsics_w2c.shape[:2]
    result = (
        torch.eye(4, dtype=extrinsics_w2c.dtype, device=extrinsics_w2c.device)
        .expand(batch_size, frame_count, 4, 4)
        .clone()
    )
    result[..., :3, :] = extrinsics_w2c
    return result


def _pixel_grid(height: int, width: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    vertical, horizontal = torch.meshgrid(
        torch.arange(height, dtype=dtype, device=device),
        torch.arange(width, dtype=dtype, device=device),
        indexing="ij",
    )
    return torch.stack((horizontal, vertical, torch.ones_like(horizontal)), dim=-1)


def _validate_probability_tensor(name: str, value: torch.Tensor) -> None:
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite and floating point")
    if torch.any((value < 0) | (value > 1)):
        raise ValueError(f"{name} must contain values in [0,1]")


def _unit_interval(name: str, value: float) -> float:
    numeric = _nonnegative_finite_float(name, value)
    if numeric > 1:
        raise ValueError(f"{name} must be in [0,1]")
    return numeric


def _nonnegative_finite_float(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return float(value)


def _validate_hw(name: str, value: tuple[int, int]) -> None:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value)
    ):
        raise ValueError(f"{name} must be a pair of positive integers")
