"""Explicit temporal and geometry conditions for multi-frame depth refinement.

Pixel-Perfect Video Depth publishes an inference-only temporal attention path.
This module reuses that public architectural idea without claiming an
unpublished PPVD training recipe, and combines it with VGGT camera geometry.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from vggt_omega.training.rendering import soft_zbuffer_reproject


class TemporalSemanticMixer(nn.Module):
    """Mix same-location semantic tokens across frames with a named reference."""

    def __init__(self, *, hidden_dim: int, num_heads: int, depth: int) -> None:
        super().__init__()
        for name, value in (("hidden_dim", hidden_dim), ("num_heads", num_heads), ("depth", depth)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.reference_projection = nn.Linear(hidden_dim, hidden_dim)
        self.reference_gate = nn.Parameter(torch.zeros(()))
        self.blocks = nn.ModuleList([_TemporalBlock(hidden_dim, num_heads) for _ in range(depth)])

    def forward(
        self,
        features: torch.Tensor,
        token_mask: torch.Tensor,
        *,
        frame_mask: torch.Tensor | None = None,
        reference_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if features.ndim != 4 or features.shape[-1] != self.hidden_dim:
            raise ValueError(f"features must have shape [B,S,T,{self.hidden_dim}]")
        if not features.is_floating_point() or not torch.isfinite(features).all():
            raise ValueError("features must be finite and floating point")
        batch_size, frame_count, token_count, hidden_dim = features.shape
        if token_mask.shape != (batch_size, frame_count, token_count) or token_mask.dtype is not torch.bool:
            raise ValueError("token_mask must be bool with shape [B,S,T]")
        if frame_mask is None:
            frame_mask = torch.ones((batch_size, frame_count), dtype=torch.bool, device=features.device)
        if frame_mask.shape != (batch_size, frame_count) or frame_mask.dtype is not torch.bool:
            raise ValueError("frame_mask must be bool with shape [B,S]")
        if token_mask.device != features.device or frame_mask.device != features.device:
            raise ValueError("features and masks must share a device")
        valid = token_mask & frame_mask.unsqueeze(-1)
        masked_features = torch.where(valid.unsqueeze(-1), features, torch.zeros_like(features))
        if frame_count == 1:
            return masked_features

        if reference_indices is None:
            reference_indices = torch.zeros(batch_size, dtype=torch.long, device=features.device)
        if reference_indices.shape != (batch_size,) or reference_indices.dtype != torch.long:
            raise ValueError("reference_indices must be int64 with shape [B]")
        if reference_indices.device != features.device:
            raise ValueError("reference_indices must share the feature device")
        if torch.any((reference_indices < 0) | (reference_indices >= frame_count)):
            raise ValueError("reference_indices are outside the frame range")
        if not frame_mask.gather(1, reference_indices[:, None]).all():
            raise ValueError("reference_indices must select valid, non-padding frames")

        batch_indices = torch.arange(batch_size, device=features.device)
        reference = masked_features[batch_indices, reference_indices]
        reference_valid = valid[batch_indices, reference_indices]
        reference_condition = self.reference_projection(reference) * self.reference_gate
        reference_condition = torch.where(
            reference_valid.unsqueeze(-1), reference_condition, torch.zeros_like(reference_condition)
        )
        mixed = masked_features + reference_condition[:, None]
        temporal = mixed.permute(0, 2, 1, 3).reshape(batch_size * token_count, frame_count, hidden_dim)
        temporal_valid = valid.permute(0, 2, 1).reshape(batch_size * token_count, frame_count)
        for block in self.blocks:
            temporal = block(temporal, temporal_valid)
        mixed = temporal.reshape(batch_size, token_count, frame_count, hidden_dim).permute(0, 2, 1, 3)
        return torch.where(valid.unsqueeze(-1), mixed, torch.zeros_like(mixed))


class _TemporalBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def forward(self, features: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        safe_features = features
        key_padding_mask = ~valid
        all_invalid = ~valid.any(dim=1)
        if all_invalid.any():
            safe_features = features.clone()
            key_padding_mask = key_padding_mask.clone()
            safe_features[all_invalid, 0] = 0
            key_padding_mask[all_invalid, 0] = False
        normalized = self.attention_norm(safe_features)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        features = safe_features + attended
        features = features + self.mlp(self.mlp_norm(features))
        return torch.where(valid.unsqueeze(-1), features, torch.zeros_like(features))


def build_warped_neighbor_condition(
    images: torch.Tensor,
    depths: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics_w2c: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    dynamic_mask: torch.Tensor | None = None,
    frame_mask: torch.Tensor | None = None,
    max_depth_m: float,
    relative_depth_tolerance: float = 0.03,
    z_temperature: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Average visible neighboring RGB warped into each target camera."""

    if images.ndim != 5 or images.shape[2] != 3:
        raise ValueError("images must have shape [B,S,3,H,W]")
    batch_size, frame_count, _, height, width = images.shape
    expected_depth_shape = (batch_size, frame_count, height, width)
    if depths.shape != expected_depth_shape:
        raise ValueError("depths must have shape [B,S,H,W]")
    if intrinsics.shape != (batch_size, frame_count, 3, 3):
        raise ValueError("intrinsics must have shape [B,S,3,3]")
    if extrinsics_w2c.shape != (batch_size, frame_count, 3, 4):
        raise ValueError("extrinsics_w2c must have shape [B,S,3,4]")
    for name, value in (
        ("images", images),
        ("depths", depths),
        ("intrinsics", intrinsics),
        ("extrinsics_w2c", extrinsics_w2c),
    ):
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite and floating point")
    if valid_mask.shape != expected_depth_shape or valid_mask.dtype is not torch.bool:
        raise ValueError("valid_mask must be bool with shape [B,S,H,W]")
    if dynamic_mask is None:
        dynamic_mask = torch.zeros_like(valid_mask)
    if dynamic_mask.shape != expected_depth_shape or dynamic_mask.dtype is not torch.bool:
        raise ValueError("dynamic_mask must be bool with shape [B,S,H,W]")
    if frame_mask is None:
        frame_mask = torch.ones((batch_size, frame_count), dtype=torch.bool, device=images.device)
    if frame_mask.shape != (batch_size, frame_count) or frame_mask.dtype is not torch.bool:
        raise ValueError("frame_mask must be bool with shape [B,S]")
    if any(mask.device != images.device for mask in (valid_mask, dynamic_mask, frame_mask)):
        raise ValueError("images and masks must share a device")
    if not math.isfinite(max_depth_m) or max_depth_m <= 0:
        raise ValueError("max_depth_m must be finite and positive")

    static_near_mask = valid_mask & ~dynamic_mask & frame_mask[:, :, None, None] & (depths > 0) & (depths < max_depth_m)
    warped_rgb = torch.zeros_like(images)
    contributor_count = torch.zeros(expected_depth_shape, dtype=images.dtype, device=images.device)
    for target_index in range(frame_count):
        for source_index in range(frame_count):
            if source_index == target_index:
                continue
            rendered = soft_zbuffer_reproject(
                images[:, source_index],
                depths[:, source_index],
                intrinsics[:, source_index],
                intrinsics[:, target_index],
                extrinsics_w2c[:, source_index],
                extrinsics_w2c[:, target_index],
                source_mask=static_near_mask[:, source_index],
                target_depth=depths[:, target_index],
                target_mask=static_near_mask[:, target_index],
                max_depth_m=max_depth_m,
                relative_depth_tolerance=relative_depth_tolerance,
                z_temperature=z_temperature,
            )
            visibility = rendered["visibility"]
            warped_rgb[:, target_index] += rendered["rgb"] * visibility[:, None]
            contributor_count[:, target_index] += visibility.to(images.dtype)
    visibility = contributor_count > 0
    warped_rgb = warped_rgb / contributor_count.clamp_min(1).unsqueeze(2)
    warped_rgb = torch.where(visibility.unsqueeze(2), warped_rgb, torch.zeros_like(warped_rgb))
    condition = torch.cat((warped_rgb, visibility.unsqueeze(2).to(images.dtype)), dim=2)
    return {
        "warped_rgb": warped_rgb,
        "visibility": visibility,
        "contributor_count": contributor_count,
        "condition": condition,
    }


def multiframe_tracking_scalars(
    *,
    frame_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    dynamic_mask: torch.Tensor,
    warped_visibility: torch.Tensor,
    reference_indices: torch.Tensor,
    preserve_frame_order: bool,
) -> dict[str, float]:
    """Summarize only anonymous mask/reference statistics for scalar logging."""

    if frame_mask.ndim != 2 or frame_mask.dtype is not torch.bool:
        raise ValueError("frame_mask must be bool with shape [B,S]")
    batch_size, frame_count = frame_mask.shape
    if valid_mask.ndim != 4 or valid_mask.shape[:2] != (batch_size, frame_count):
        raise ValueError("valid_mask must have shape [B,S,H,W]")
    if valid_mask.dtype is not torch.bool:
        raise ValueError("valid_mask must be boolean")
    if dynamic_mask.shape != valid_mask.shape or dynamic_mask.dtype is not torch.bool:
        raise ValueError("dynamic_mask must be bool and match valid_mask")
    if warped_visibility.shape != valid_mask.shape or warped_visibility.dtype is not torch.bool:
        raise ValueError("warped_visibility must be bool and match valid_mask")
    if reference_indices.shape != (batch_size,) or reference_indices.dtype != torch.long:
        raise ValueError("reference_indices must be int64 with shape [B]")
    if not isinstance(preserve_frame_order, bool):
        raise ValueError("preserve_frame_order must be boolean")
    devices = {value.device for value in (frame_mask, valid_mask, dynamic_mask, warped_visibility, reference_indices)}
    if len(devices) != 1:
        raise ValueError("tracking tensors must share a device")
    if torch.any((reference_indices < 0) | (reference_indices >= frame_count)):
        raise ValueError("reference_indices are outside the frame range")
    if not frame_mask.gather(1, reference_indices[:, None]).all():
        raise ValueError("reference_indices must select valid frames")

    spatial_frame_mask = frame_mask[:, :, None, None].expand_as(valid_mask)
    available_pixels = spatial_frame_mask.sum().clamp_min(1)
    return {
        "multiframe_frame_count": float(frame_mask.sum(dim=1).float().mean()),
        "multiframe_reference_index": float(reference_indices.float().mean()),
        "multiframe_padding_fraction": float((~frame_mask).float().mean()),
        "multiframe_valid_fraction": float((valid_mask & spatial_frame_mask).sum() / available_pixels),
        "multiframe_dynamic_fraction": float((dynamic_mask & spatial_frame_mask).sum() / available_pixels),
        "multiframe_warped_visibility": float((warped_visibility & spatial_frame_mask).sum() / available_pixels),
        "multiframe_preserve_frame_order": float(preserve_frame_order),
    }
