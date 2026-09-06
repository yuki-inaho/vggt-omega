"""Factored directed-correspondence supervision for VGGT-Omega.

This is a clean-room implementation of the Flow3R factorization idea.  The
public Flow3R repository does not include its VGGT training head or complete
training pipeline, so no external source code or unpublished schema is copied.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import nn

CORRESPONDENCE_COORDINATE_SPACE = "pixel_displacement_xy"
_EXTERNAL_TEACHER_FIELDS = {
    "schema_version",
    "coordinate_space",
    "pair_indices",
    "flow_pixels",
    "covisibility_mask",
}


def validate_external_teacher_targets(
    payload: Mapping[str, object],
    *,
    expected_pair_indices: torch.Tensor,
    output_hw: tuple[int, int],
) -> dict[str, torch.Tensor]:
    """Validate a generic teacher payload without assuming a UFM format."""

    if set(payload) != _EXTERNAL_TEACHER_FIELDS:
        raise ValueError(
            "external teacher fields must match the explicit schema: "
            f"missing={sorted(_EXTERNAL_TEACHER_FIELDS - set(payload))}, "
            f"unexpected={sorted(set(payload) - _EXTERNAL_TEACHER_FIELDS)}"
        )
    if payload["schema_version"] != 1:
        raise ValueError("unsupported external teacher schema version")
    if payload["coordinate_space"] != CORRESPONDENCE_COORDINATE_SPACE:
        raise ValueError("external teacher coordinate_space must be pixel_displacement_xy")
    pair_indices = payload["pair_indices"]
    flow = payload["flow_pixels"]
    mask = payload["covisibility_mask"]
    if not isinstance(pair_indices, torch.Tensor) or not torch.equal(pair_indices, expected_pair_indices):
        raise ValueError("external teacher pair_indices must exactly match the requested directed pairs")
    if not isinstance(flow, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise ValueError("external teacher flow_pixels and covisibility_mask must be tensors")
    height, width = _positive_hw("output_hw", output_hw)
    expected_flow_shape = (*expected_pair_indices.shape[:2], height, width, 2)
    if flow.shape != expected_flow_shape or mask.shape != expected_flow_shape[:-1]:
        raise ValueError("external teacher flow or mask shape does not match pairs/output_hw")
    if not flow.is_floating_point() or not torch.isfinite(flow).all():
        raise ValueError("external teacher flow must be finite floating point")
    if mask.dtype is not torch.bool:
        raise ValueError("external teacher covisibility_mask must be boolean")
    if any(value.device != expected_pair_indices.device for value in (pair_indices, flow, mask)):
        raise ValueError("external teacher tensors must share the requested pair device")
    return {"flow_pixels": flow, "covisibility_mask": mask}


class FactoredCorrespondenceHead(nn.Module):
    """Predict a source-to-target residual over camera/depth reprojection flow."""

    def __init__(self, *, geometry_dim: int, camera_dim: int, hidden_dim: int) -> None:
        super().__init__()
        for name, value in (
            ("geometry_dim", geometry_dim),
            ("camera_dim", camera_dim),
            ("hidden_dim", hidden_dim),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.geometry_dim = geometry_dim
        self.camera_dim = camera_dim
        self.hidden_dim = hidden_dim
        self.source_projection = nn.Linear(geometry_dim, hidden_dim)
        self.target_camera_projection = nn.Linear(camera_dim, hidden_dim)
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.output_projection = nn.Linear(hidden_dim, 2)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        geometry_features: torch.Tensor,
        camera_features: torch.Tensor,
        pair_indices: torch.Tensor,
        *,
        source_grid_hw: tuple[int, int],
        output_hw: tuple[int, int],
        pair_chunk_size: int,
    ) -> torch.Tensor:
        if geometry_features.ndim != 4 or geometry_features.shape[-1] != self.geometry_dim:
            raise ValueError(f"geometry_features must have shape [B,S,P,{self.geometry_dim}]")
        batch_size, frame_count, patch_count, _ = geometry_features.shape
        if camera_features.shape != (batch_size, frame_count, self.camera_dim):
            raise ValueError(f"camera_features must have shape [B,S,{self.camera_dim}]")
        if pair_indices.ndim != 3 or pair_indices.shape[0] != batch_size or pair_indices.shape[2] != 2:
            raise ValueError("pair_indices must have shape [B,Q,2]")
        if pair_indices.dtype is not torch.long:
            raise ValueError("pair_indices must use int64 dtype")
        values = (geometry_features, camera_features)
        if any(not value.is_floating_point() or not torch.isfinite(value).all() for value in values):
            raise ValueError("geometry and camera features must be finite floating point")
        if any(value.device != geometry_features.device for value in (*values, pair_indices)):
            raise ValueError("correspondence head inputs must share a device")
        if torch.any((pair_indices < 0) | (pair_indices >= frame_count)):
            raise ValueError("pair_indices are outside the frame range")
        if torch.any(pair_indices[..., 0] == pair_indices[..., 1]):
            raise ValueError("directed correspondence pairs must use distinct frames")
        grid_height, grid_width = _positive_hw("source_grid_hw", source_grid_hw)
        output_height, output_width = _positive_hw("output_hw", output_hw)
        if patch_count != grid_height * grid_width:
            raise ValueError("source_grid_hw does not match patch count")
        if isinstance(pair_chunk_size, bool) or not isinstance(pair_chunk_size, int) or pair_chunk_size < 1:
            raise ValueError("pair_chunk_size must be a positive integer")

        batch_indices = torch.arange(batch_size, device=geometry_features.device)[:, None]
        predictions: list[torch.Tensor] = []
        for start in range(0, pair_indices.shape[1], pair_chunk_size):
            pairs = pair_indices[:, start : start + pair_chunk_size]
            source_features = geometry_features[batch_indices, pairs[..., 0]]
            target_camera = camera_features[batch_indices, pairs[..., 1]]
            hidden = self.source_projection(source_features) + self.target_camera_projection(target_camera)[:, :, None]
            hidden = hidden + self.fusion(hidden)
            patch_flow = self.output_projection(hidden)
            chunk_size = patch_flow.shape[1]
            patch_flow = patch_flow.reshape(batch_size * chunk_size, grid_height, grid_width, 2).permute(0, 3, 1, 2)
            dense_flow = F.interpolate(
                patch_flow,
                size=(output_height, output_width),
                mode="bilinear",
                align_corners=False,
            )
            predictions.append(
                dense_flow.permute(0, 2, 3, 1).reshape(batch_size, chunk_size, output_height, output_width, 2)
            )
        if not predictions:
            raise ValueError("pair_indices must contain at least one directed pair")
        return torch.cat(predictions, dim=1)


def project_depth_correspondence_flow(
    depths: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics_w2c: torch.Tensor,
    pair_indices: torch.Tensor,
) -> torch.Tensor:
    """Project predicted source depth into each target camera as dense pixel flow."""

    if depths.ndim != 4:
        raise ValueError("depths must have shape [B,S,H,W]")
    batch_size, frame_count, height, width = depths.shape
    if intrinsics.shape != (batch_size, frame_count, 3, 3):
        raise ValueError("intrinsics must have shape [B,S,3,3]")
    if extrinsics_w2c.shape != (batch_size, frame_count, 3, 4):
        raise ValueError("extrinsics_w2c must have shape [B,S,3,4]")
    if pair_indices.ndim != 3 or pair_indices.shape[0] != batch_size or pair_indices.shape[2] != 2:
        raise ValueError("pair_indices must have shape [B,Q,2]")
    if pair_indices.dtype is not torch.long:
        raise ValueError("pair_indices must use int64 dtype")
    values = (depths, intrinsics, extrinsics_w2c)
    if any(not value.is_floating_point() or not torch.isfinite(value).all() for value in values):
        raise ValueError("predicted correspondence geometry must be finite floating point")
    if any(value.device != depths.device for value in (*values, pair_indices)):
        raise ValueError("predicted correspondence inputs must share a device")
    if torch.any((pair_indices < 0) | (pair_indices >= frame_count)):
        raise ValueError("pair_indices are outside the frame range")
    if torch.any(pair_indices[..., 0] == pair_indices[..., 1]):
        raise ValueError("directed correspondence pairs must use distinct frames")

    rows = torch.arange(height, dtype=depths.dtype, device=depths.device)
    columns = torch.arange(width, dtype=depths.dtype, device=depths.device)
    vertical, horizontal = torch.meshgrid(rows, columns, indexing="ij")
    pixels = torch.stack((horizontal, vertical, torch.ones_like(horizontal)), dim=-1).reshape(-1, 3)
    pixel_xy = pixels[:, :2].reshape(height, width, 2)
    batches: list[torch.Tensor] = []
    for batch_index in range(batch_size):
        flows: list[torch.Tensor] = []
        for source_value, target_value in pair_indices[batch_index]:
            source = int(source_value)
            target = int(target_value)
            source_points = torch.linalg.solve(intrinsics[batch_index, source], pixels.T).T
            source_points = source_points * depths[batch_index, source].reshape(-1, 1)
            source_rotation = extrinsics_w2c[batch_index, source, :3, :3]
            source_translation = extrinsics_w2c[batch_index, source, :3, 3]
            world_points = (source_points - source_translation) @ source_rotation
            target_rotation = extrinsics_w2c[batch_index, target, :3, :3]
            target_translation = extrinsics_w2c[batch_index, target, :3, 3]
            target_points = world_points @ target_rotation.T + target_translation
            target_z = target_points[:, 2]
            safe_z = torch.where(
                target_z.abs() > torch.finfo(depths.dtype).eps,
                target_z,
                torch.ones_like(target_z),
            )
            projected = target_points @ intrinsics[batch_index, target].T
            target_xy = torch.stack((projected[:, 0] / safe_z, projected[:, 1] / safe_z), dim=-1)
            flows.append(target_xy.reshape(height, width, 2) - pixel_xy)
        batches.append(torch.stack(flows))
    return torch.stack(batches)


def _positive_hw(name: str, value: tuple[int, int]) -> tuple[int, int]:
    if len(value) != 2 or any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in value):
        raise ValueError(f"{name} must contain two positive integers")
    return value


def _sequence_mask(name: str, value: torch.Tensor | None, depth: torch.Tensor, *, default: bool) -> torch.Tensor:
    if value is None:
        return torch.full_like(depth, default, dtype=torch.bool)
    if value.shape != depth.shape or value.dtype is not torch.bool or value.device != depth.device:
        raise ValueError(f"{name} must be bool and match depths")
    return value


def build_rgbd_correspondence_targets(
    depths: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics_w2c: torch.Tensor,
    pair_indices: torch.Tensor,
    *,
    valid_mask: torch.Tensor,
    relative_depth_tolerance: float,
    dynamic_mask: torch.Tensor | None = None,
    frame_mask: torch.Tensor | None = None,
    max_depth: float | None = None,
) -> dict[str, torch.Tensor]:
    """Build source-to-target ``(du,dv)`` pixel flow and covisibility."""

    if depths.ndim != 4:
        raise ValueError("depths must have shape [B,S,H,W]")
    batch_size, frame_count, height, width = depths.shape
    if intrinsics.shape != (batch_size, frame_count, 3, 3):
        raise ValueError("intrinsics must have shape [B,S,3,3]")
    if extrinsics_w2c.shape != (batch_size, frame_count, 3, 4):
        raise ValueError("extrinsics_w2c must have shape [B,S,3,4]")
    if pair_indices.ndim != 3 or pair_indices.shape[0] != batch_size or pair_indices.shape[2] != 2:
        raise ValueError("pair_indices must have shape [B,Q,2]")
    if pair_indices.dtype is not torch.long:
        raise ValueError("pair_indices must use int64 dtype")
    values = (depths, intrinsics, extrinsics_w2c)
    if any(not value.is_floating_point() or not torch.isfinite(value).all() for value in values):
        raise ValueError("RGB-D correspondence geometry must be finite floating point")
    if any(value.device != depths.device for value in (*values, pair_indices, valid_mask)):
        raise ValueError("RGB-D correspondence inputs must share a device")
    valid_mask = _sequence_mask("valid_mask", valid_mask, depths, default=True)
    dynamic_mask = _sequence_mask("dynamic_mask", dynamic_mask, depths, default=False)
    if frame_mask is None:
        frame_mask = torch.ones((batch_size, frame_count), dtype=torch.bool, device=depths.device)
    if frame_mask.shape != (batch_size, frame_count) or frame_mask.dtype is not torch.bool:
        raise ValueError("frame_mask must be bool with shape [B,S]")
    if frame_mask.device != depths.device:
        raise ValueError("frame_mask must share the depth device")
    if torch.any((pair_indices < 0) | (pair_indices >= frame_count)):
        raise ValueError("pair_indices are outside the frame range")
    if torch.any(pair_indices[..., 0] == pair_indices[..., 1]):
        raise ValueError("directed correspondence pairs must use distinct frames")
    if not frame_mask.gather(1, pair_indices.reshape(batch_size, -1)).all():
        raise ValueError("pair_indices must select valid, non-padding frames")
    if (
        isinstance(relative_depth_tolerance, bool)
        or not isinstance(relative_depth_tolerance, (int, float))
        or not math.isfinite(relative_depth_tolerance)
        or relative_depth_tolerance < 0
    ):
        raise ValueError("relative_depth_tolerance must be finite and non-negative")
    if max_depth is not None and (
        isinstance(max_depth, bool)
        or not isinstance(max_depth, (int, float))
        or not math.isfinite(max_depth)
        or max_depth <= 0
    ):
        raise ValueError("max_depth must be finite and positive")

    pair_count = pair_indices.shape[1]
    flow = torch.zeros((batch_size, pair_count, height, width, 2), dtype=depths.dtype, device=depths.device)
    covisibility = torch.zeros((batch_size, pair_count, height, width), dtype=torch.bool, device=depths.device)
    rows = torch.arange(height, dtype=depths.dtype, device=depths.device)
    columns = torch.arange(width, dtype=depths.dtype, device=depths.device)
    vertical, horizontal = torch.meshgrid(rows, columns, indexing="ij")
    pixels = torch.stack((horizontal, vertical, torch.ones_like(horizontal)), dim=-1).reshape(-1, 3)
    pixel_xy = pixels[:, :2].reshape(height, width, 2)
    static_valid = valid_mask & ~dynamic_mask

    for batch_index in range(batch_size):
        for pair_offset, pair in enumerate(pair_indices[batch_index]):
            source = int(pair[0])
            target = int(pair[1])
            source_depth = depths[batch_index, source]
            source_points = torch.linalg.solve(intrinsics[batch_index, source], pixels.T).T * source_depth.reshape(
                -1, 1
            )
            source_rotation = extrinsics_w2c[batch_index, source, :3, :3]
            source_translation = extrinsics_w2c[batch_index, source, :3, 3]
            world_points = (source_points - source_translation) @ source_rotation
            target_rotation = extrinsics_w2c[batch_index, target, :3, :3]
            target_translation = extrinsics_w2c[batch_index, target, :3, 3]
            target_points = world_points @ target_rotation.T + target_translation
            target_z = target_points[:, 2]
            projected = target_points @ intrinsics[batch_index, target].T
            safe_z = torch.where(
                target_z.abs() > torch.finfo(depths.dtype).eps,
                target_z,
                torch.ones_like(target_z),
            )
            target_x = (projected[:, 0] / safe_z).reshape(height, width)
            target_y = (projected[:, 1] / safe_z).reshape(height, width)
            grid = torch.stack((2 * (target_x + 0.5) / width - 1, 2 * (target_y + 0.5) / height - 1), dim=-1)
            target_depth = F.grid_sample(
                depths[batch_index : batch_index + 1, target, None],
                grid[None],
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )[0, 0]
            target_static = (
                F.grid_sample(
                    static_valid[batch_index : batch_index + 1, target, None].to(depths.dtype),
                    grid[None],
                    mode="nearest",
                    padding_mode="zeros",
                    align_corners=False,
                )[0, 0]
                > 0.5
            )
            in_bounds = (target_x >= 0) & (target_x <= width - 1) & (target_y >= 0) & (target_y <= height - 1)
            target_z = target_z.reshape(height, width)
            visible = (
                static_valid[batch_index, source]
                & (source_depth > 0)
                & target_static
                & (target_z > 0)
                & (target_depth > 0)
                & in_bounds
            )
            relative_error = (target_z - target_depth).abs() / target_depth.clamp_min(torch.finfo(depths.dtype).eps)
            visible &= relative_error <= relative_depth_tolerance
            if max_depth is not None:
                visible &= (source_depth < max_depth) & (target_depth < max_depth)
            pair_flow = torch.stack((target_x, target_y), dim=-1) - pixel_xy
            flow[batch_index, pair_offset] = torch.where(visible[..., None], pair_flow, torch.zeros_like(pair_flow))
            covisibility[batch_index, pair_offset] = visible
    return {"flow_pixels": flow, "covisibility_mask": covisibility}


def masked_generalized_charbonnier(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    alpha: float,
    epsilon: float,
) -> torch.Tensor:
    """Robust directed-flow loss with exact zero at zero residual."""

    if prediction.ndim != 5 or prediction.shape[-1] != 2 or target.shape != prediction.shape:
        raise ValueError("prediction and target must match [B,Q,H,W,2]")
    if mask.shape != prediction.shape[:-1] or mask.dtype is not torch.bool:
        raise ValueError("mask must be bool with shape [B,Q,H,W]")
    if any(value.device != prediction.device for value in (target, mask)):
        raise ValueError("prediction, target, and mask must share a device")
    if any(not value.is_floating_point() or not torch.isfinite(value).all() for value in (prediction, target)):
        raise ValueError("prediction and target must be finite floating point")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("alpha must be finite and within (0, 1]")
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if not bool(mask.any()):
        return prediction.reshape(-1)[0] * 0
    squared_error = (prediction - target).square().sum(dim=-1)
    robust = (squared_error + epsilon**2).pow(alpha) - epsilon ** (2 * alpha)
    return robust[mask].mean()


def masked_endpoint_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Return mean flow endpoint error in pixels over covisible targets."""

    if prediction.ndim != 5 or prediction.shape[-1] != 2 or target.shape != prediction.shape:
        raise ValueError("prediction and target must match [B,Q,H,W,2]")
    if mask.shape != prediction.shape[:-1] or mask.dtype is not torch.bool:
        raise ValueError("mask must be bool with shape [B,Q,H,W]")
    if any(value.device != prediction.device for value in (target, mask)):
        raise ValueError("prediction, target, and mask must share a device")
    if any(not value.is_floating_point() or not torch.isfinite(value).all() for value in (prediction, target)):
        raise ValueError("prediction and target must be finite floating point")
    if not bool(mask.any()):
        return prediction.reshape(-1)[0] * 0
    return torch.linalg.vector_norm(prediction - target, dim=-1)[mask].mean()
