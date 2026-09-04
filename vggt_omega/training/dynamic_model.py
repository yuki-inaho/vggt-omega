"""Opt-in dynamic-geometry wrapper with an unchanged legacy forward path."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from vggt_omega.training.dynamic_geometry import (
    CanonicalMotionHead,
    build_temporal_pairs,
    canonical_points_from_depth,
    partition_dynamic_probability,
)
from vggt_omega.utils.pose_enc import encoding_to_camera

_FROZEN_CONFIDENCE_MARKER = ".dense_head.proj_conf."


class DynamicGeometryTrainingModel(nn.Module):
    """Add explicit 4D outputs while preserving the wrapped model's forward."""

    def __init__(
        self,
        wrapped_model: nn.Module,
        *,
        feature_dim: int,
        hidden_dim: int,
        relative_camera_dim: int,
        pair_chunk_size: int,
        refinement_seed: int,
        visibility_threshold: float,
        static_probability_max: float,
        dynamic_probability_min: float,
        joint_base_parameter_prefixes: Sequence[str],
    ) -> None:
        super().__init__()
        if not callable(getattr(wrapped_model, "prepare_dynamic_context", None)):
            raise TypeError("dynamic geometry requires a model with prepare_dynamic_context")
        if relative_camera_dim != 14:
            raise ValueError("relative_camera_dim must be 14 for relative SE(3) plus focal ratios")
        if isinstance(pair_chunk_size, bool) or not isinstance(pair_chunk_size, int) or pair_chunk_size < 1:
            raise ValueError("pair_chunk_size must be a positive integer")
        if isinstance(refinement_seed, bool) or not isinstance(refinement_seed, int) or refinement_seed < 0:
            raise ValueError("refinement_seed must be a non-negative integer")
        prefixes = tuple(joint_base_parameter_prefixes)
        if any(not isinstance(prefix, str) or not prefix for prefix in prefixes):
            raise ValueError("joint_base_parameter_prefixes must contain non-empty strings")
        self.wrapped_model = wrapped_model
        self.dynamic_geometry_head = CanonicalMotionHead(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            relative_camera_dim=relative_camera_dim,
        )
        self.pair_chunk_size = pair_chunk_size
        self.refinement_seed = refinement_seed
        self.visibility_threshold = float(visibility_threshold)
        self.static_probability_max = float(static_probability_max)
        self.dynamic_probability_min = float(dynamic_probability_min)
        self.joint_base_parameter_prefixes = prefixes
        self.register_buffer("dynamic_geometry_ready", torch.tensor(False, dtype=torch.bool), persistent=True)
        self._configure_optimizer_candidates()

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Delegate exactly to the wrapped legacy path."""

        return self.wrapped_model(images)

    @torch.no_grad()
    def forward_refine(self, images: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Delegate legacy refined-depth validation through the opt-in wrapper."""

        refine = getattr(self.wrapped_model, "forward_refine", None)
        if not callable(refine):
            raise TypeError("wrapped model does not provide forward_refine")
        return refine(images, **kwargs)

    def forward_dynamic(
        self,
        images: torch.Tensor,
        *,
        frame_ids: torch.Tensor,
        frame_mask: torch.Tensor,
        motion_pair_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict source-aligned canonical motion for explicit temporal pairs."""

        self._validate_sequence_inputs(images, frame_ids, frame_mask)
        pair_metadata = build_temporal_pairs(
            frame_ids,
            frame_mask,
            motion_pair_indices=motion_pair_indices,
        )
        contexts = self._compact_contexts(images, frame_ids, frame_mask)
        depth = contexts["depth"]
        pose = contexts["pose_enc"]
        patch_features = contexts["patch_features"]
        patch_grid_value = contexts["patch_grid_hw"]
        patch_grid_hw = (int(patch_grid_value[0]), int(patch_grid_value[1]))
        height, width = images.shape[-2:]
        extrinsics, intrinsics = encoding_to_camera(pose.float(), (height, width), build_intrinsics=True)
        assert intrinsics is not None
        extrinsics, intrinsics = _finite_padding_cameras(
            extrinsics.float(),
            intrinsics.float(),
            frame_mask,
            image_hw=(height, width),
        )
        point_geometry = canonical_points_from_depth(
            depth[..., 0].float(),
            contexts["depth_valid_mask"],
            intrinsics,
            extrinsics,
            frame_mask,
        )
        relative_camera = _relative_camera_features(
            extrinsics,
            intrinsics,
            pair_metadata["motion_pair_indices"],
            pair_metadata["motion_pair_valid_mask"],
        ).to(patch_features)
        head = self._head_outputs(
            patch_features,
            relative_camera,
            pair_metadata,
            patch_grid_hw=patch_grid_hw,
            output_hw=(height, width),
        )
        pairs = pair_metadata["motion_pair_indices"]
        pair_valid = pair_metadata["motion_pair_valid_mask"]
        safe_pairs = pairs.clamp_min(0)
        batch_indices = torch.arange(images.shape[0], device=images.device)[:, None]
        source_points = point_geometry["canonical_points_current"][batch_indices, safe_pairs[..., 0]]
        source_valid = point_geometry["canonical_points_valid_mask"][batch_indices, safe_pairs[..., 0]]
        motion_domain = source_valid & pair_valid[:, :, None, None]
        scene_flow = torch.where(
            motion_domain.unsqueeze(-1),
            head["canonical_scene_flow"].float(),
            torch.zeros_like(head["canonical_scene_flow"], dtype=torch.float32),
        )
        target_points = torch.where(
            motion_domain.unsqueeze(-1),
            source_points + scene_flow,
            torch.zeros_like(source_points),
        )
        target_extrinsics = point_geometry["rebased_extrinsics_w2c"][batch_indices, safe_pairs[..., 1]]
        target_camera_points = (
            torch.einsum(
                "bqij,bqhwj->bqhwi",
                target_extrinsics[..., :3],
                target_points,
            )
            + target_extrinsics[..., 3][:, :, None, None, :]
        )
        target_depth = torch.where(
            motion_domain,
            target_camera_points[..., 2],
            torch.zeros_like(target_camera_points[..., 2]),
        )
        masks = partition_dynamic_probability(
            head["dynamic_probability"],
            head["motion_visibility_probability"],
            motion_domain,
            ready=bool(self.dynamic_geometry_ready.item()),
            visibility_threshold=self.visibility_threshold,
            static_probability_max=self.static_probability_max,
            dynamic_probability_min=self.dynamic_probability_min,
        )
        public_context = {
            key: value
            for key, value in contexts.items()
            if key not in {"patch_features", "patch_grid_hw", "patch_valid_mask", "depth_valid_mask"}
        }
        public_context.update(
            {
                "canonical_reference_indices": torch.zeros(images.shape[0], dtype=torch.long, device=images.device),
                "motion_frame_ids": frame_ids,
                "motion_frame_mask": frame_mask,
                "canonical_points_current": point_geometry["canonical_points_current"],
                "canonical_points_valid_mask": point_geometry["canonical_points_valid_mask"],
                "rebased_extrinsics_w2c": point_geometry["rebased_extrinsics_w2c"],
                "predicted_intrinsics": intrinsics,
                **pair_metadata,
                "canonical_scene_flow": scene_flow,
                "canonical_points_at_target_time": target_points,
                "depth_at_target_time_in_target_camera": target_depth.unsqueeze(-1),
                "motion_domain_mask": motion_domain,
                "motion_visibility_logits": head["motion_visibility_logits"],
                "motion_visibility_probability": head["motion_visibility_probability"],
                "dynamic_logits": head["dynamic_logits"],
                "dynamic_probability": head["dynamic_probability"],
                **masks,
                "dynamic_geometry_ready": self.dynamic_geometry_ready.clone(),
            }
        )
        return public_context

    def set_dynamic_stage(self, stage: str) -> None:
        """Apply the exact trainable allowlist for one curriculum stage."""

        if stage not in {"disabled", "motion_only", "visibility_dynamic", "joint"}:
            raise ValueError(f"unsupported dynamic stage: {stage}")
        self.requires_grad_(False)
        if stage == "disabled":
            return
        if stage == "motion_only":
            self.dynamic_geometry_head.pair_encoder.requires_grad_(True)
            self.dynamic_geometry_head.flow_decoder.requires_grad_(True)
            return
        if stage == "visibility_dynamic":
            self.dynamic_geometry_head.visibility_decoder.requires_grad_(True)
            self.dynamic_geometry_head.dynamic_decoder.requires_grad_(True)
            return
        if not self.joint_base_parameter_prefixes:
            raise ValueError("joint stage requires non-empty joint_base_parameter_prefixes")
        self.dynamic_geometry_head.requires_grad_(True)
        matched = set()
        for name, parameter in self.named_parameters():
            if self._is_joint_base_parameter(name):
                parameter.requires_grad_(True)
                matched.add(name)
        if not matched:
            raise ValueError("joint_base_parameter_prefixes did not match model parameters")

    def set_dynamic_geometry_ready(self, ready: bool) -> None:
        if not isinstance(ready, bool):
            raise TypeError("ready must be boolean")
        self.dynamic_geometry_ready.fill_(ready)

    def _configure_optimizer_candidates(self) -> None:
        self.wrapped_model.requires_grad_(False)
        self.dynamic_geometry_head.requires_grad_(True)
        for name, parameter in self.named_parameters():
            if self._is_joint_base_parameter(name):
                parameter.requires_grad_(True)

    def _is_joint_base_parameter(self, name: str) -> bool:
        return _FROZEN_CONFIDENCE_MARKER not in name and any(
            name.startswith(prefix) for prefix in self.joint_base_parameter_prefixes
        )

    def _compact_contexts(
        self,
        images: torch.Tensor,
        frame_ids: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, frame_count, _, height, width = images.shape
        grouped: dict[int, list[int]] = defaultdict(list)
        positions: dict[int, torch.Tensor] = {}
        for sample in range(batch_size):
            valid_positions = torch.nonzero(frame_mask[sample], as_tuple=False).flatten()
            positions[sample] = valid_positions
            grouped[len(valid_positions)].append(sample)

        collected: dict[int, dict[str, torch.Tensor]] = {}
        for count, samples in sorted(grouped.items()):
            compact_images = torch.stack([images[sample, positions[sample]] for sample in samples])
            noise = torch.cat(
                [
                    _stable_noise(
                        compact_images[index : index + 1],
                        frame_ids[sample, positions[sample]],
                        base_seed=self.refinement_seed,
                    )
                    for index, sample in enumerate(samples)
                ],
                dim=0,
            )
            context = self.wrapped_model.prepare_dynamic_context(compact_images, initial_noise=noise)
            if context["depth"].shape[:2] != (len(samples), count):
                raise ValueError("dynamic context returned an invalid depth shape")
            for index, sample in enumerate(samples):
                collected[sample] = {
                    key: value[index]
                    for key, value in context.items()
                    if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == len(samples)
                }
                collected[sample]["patch_grid_hw"] = context["patch_grid_hw"]

        first = collected[0]
        patch_count, feature_dim = first["patch_features"].shape[-2:]
        depth = images.new_zeros((batch_size, frame_count, height, width, 1))
        pose = images.new_zeros((batch_size, frame_count, 9))
        patch = images.new_zeros((batch_size, frame_count, patch_count, feature_dim))
        patch_mask = torch.zeros((batch_size, frame_count, patch_count), dtype=torch.bool, device=images.device)
        for sample in range(batch_size):
            valid_positions = positions[sample]
            item = collected[sample]
            if item["patch_features"].shape[-2:] != (patch_count, feature_dim):
                raise ValueError("grouped dynamic contexts returned inconsistent patch shapes")
            depth[sample, valid_positions] = item["depth"]
            pose[sample, valid_positions] = item["pose_enc"]
            patch[sample, valid_positions] = item["patch_features"]
            patch_mask[sample, valid_positions] = item["patch_valid_mask"]
        depth_valid = frame_mask[:, :, None, None] & torch.isfinite(depth[..., 0]) & (depth[..., 0] > 0)
        return {
            "depth": depth,
            "pose_enc": pose,
            "patch_features": patch,
            "patch_grid_hw": first["patch_grid_hw"],
            "patch_valid_mask": patch_mask,
            "depth_valid_mask": depth_valid,
        }

    def _head_outputs(
        self,
        patch_features: torch.Tensor,
        relative_camera: torch.Tensor,
        pair_metadata: Mapping[str, torch.Tensor],
        *,
        patch_grid_hw: tuple[int, int],
        output_hw: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        outputs: dict[str, list[torch.Tensor]] = defaultdict(list)
        pair_count = pair_metadata["motion_pair_indices"].shape[1]
        for start in range(0, pair_count, self.pair_chunk_size):
            stop = min(start + self.pair_chunk_size, pair_count)
            chunk = self.dynamic_geometry_head(
                patch_features,
                relative_camera[:, start:stop],
                pair_metadata["motion_time_delta_frames"][:, start:stop],
                pair_metadata["motion_pair_indices"][:, start:stop],
                pair_metadata["motion_pair_valid_mask"][:, start:stop],
                patch_grid_hw=patch_grid_hw,
                output_hw=output_hw,
            )
            for key, value in chunk.items():
                outputs[key].append(value)
        return {key: torch.cat(values, dim=1) for key, values in outputs.items()}

    @staticmethod
    def _validate_sequence_inputs(images: torch.Tensor, frame_ids: torch.Tensor, frame_mask: torch.Tensor) -> None:
        if images.ndim != 5 or not images.is_floating_point():
            raise ValueError("images must be floating point with shape [B,S,3,H,W]")
        if images.shape[2] != 3 or min(images.shape) <= 0 or not torch.isfinite(images).all():
            raise ValueError("images must be finite non-empty RGB sequences")
        if frame_ids.shape != images.shape[:2] or frame_ids.dtype != torch.long:
            raise ValueError("frame_ids must be int64 with shape [B,S]")
        if frame_mask.shape != images.shape[:2] or frame_mask.dtype is not torch.bool:
            raise ValueError("frame_mask must be bool with shape [B,S]")
        if frame_ids.device != images.device or frame_mask.device != images.device:
            raise ValueError("images, frame_ids, and frame_mask must share a device")


def _stable_noise(images: torch.Tensor, frame_ids: torch.Tensor, *, base_seed: int) -> torch.Tensor:
    payload = frame_ids.detach().cpu().contiguous().numpy().tobytes()
    digest = hashlib.sha256(base_seed.to_bytes(8, "little", signed=False) + payload).digest()
    seed = int.from_bytes(digest[:8], "little") % (2**63 - 1)
    generator = torch.Generator(device=images.device).manual_seed(seed)
    _, frame_count, _, height, width = images.shape
    return torch.randn(
        (frame_count, 1, height, width),
        dtype=images.dtype,
        device=images.device,
        generator=generator,
    )


def _finite_padding_cameras(
    extrinsics_w2c: torch.Tensor,
    intrinsics: torch.Tensor,
    frame_mask: torch.Tensor,
    *,
    image_hw: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace out-of-domain padding cameras without changing valid frames."""

    batch_size, frame_count = frame_mask.shape
    height, width = image_hw
    padding_extrinsics = torch.zeros(
        (batch_size, frame_count, 3, 4),
        dtype=extrinsics_w2c.dtype,
        device=extrinsics_w2c.device,
    )
    padding_extrinsics[..., :3] = torch.eye(
        3,
        dtype=extrinsics_w2c.dtype,
        device=extrinsics_w2c.device,
    )
    padding_intrinsics = torch.zeros_like(intrinsics)
    padding_intrinsics[..., 0, 0] = 1.0
    padding_intrinsics[..., 1, 1] = 1.0
    padding_intrinsics[..., 0, 2] = width / 2.0
    padding_intrinsics[..., 1, 2] = height / 2.0
    padding_intrinsics[..., 2, 2] = 1.0
    camera_mask = frame_mask[..., None, None]
    return (
        torch.where(camera_mask, extrinsics_w2c, padding_extrinsics),
        torch.where(camera_mask, intrinsics, padding_intrinsics),
    )


def _relative_camera_features(
    extrinsics_w2c: torch.Tensor,
    intrinsics: torch.Tensor,
    pair_indices: torch.Tensor,
    pair_valid: torch.Tensor,
) -> torch.Tensor:
    batch_size, frame_count = extrinsics_w2c.shape[:2]
    homogeneous = (
        torch.eye(4, dtype=extrinsics_w2c.dtype, device=extrinsics_w2c.device)
        .expand(batch_size, frame_count, 4, 4)
        .clone()
    )
    homogeneous[..., :3, :] = extrinsics_w2c
    safe = pair_indices.clamp_min(0)
    batch_indices = torch.arange(batch_size, device=extrinsics_w2c.device)[:, None]
    source = homogeneous[batch_indices, safe[..., 0]]
    target = homogeneous[batch_indices, safe[..., 1]]
    relative = target @ torch.linalg.inv(source)
    source_k = intrinsics[batch_indices, safe[..., 0]]
    target_k = intrinsics[batch_indices, safe[..., 1]]
    focal_ratio = torch.stack(
        (
            torch.log(target_k[..., 0, 0] / source_k[..., 0, 0]),
            torch.log(target_k[..., 1, 1] / source_k[..., 1, 1]),
        ),
        dim=-1,
    )
    features = torch.cat((relative[..., :3, :3].flatten(-2), relative[..., :3, 3], focal_ratio), dim=-1)
    return torch.where(pair_valid.unsqueeze(-1), features, torch.zeros_like(features))
