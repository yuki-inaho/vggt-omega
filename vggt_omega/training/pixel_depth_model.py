"""Optional Pixel-Perfect-style multi-frame depth training wrapper."""

from __future__ import annotations

import math

import torch
from torch import nn

from vggt_omega.training.correspondence import FactoredCorrespondenceHead
from vggt_omega.training.multiframe import TemporalSemanticMixer, build_warped_neighbor_condition
from vggt_omega.training.pixel_depth import (
    PixelDepthFlowRefiner,
    SemanticPromptAdapter,
    decode_log_depth_residual,
    depth_gradient_matching_loss,
    encode_log_depth_residual,
    euler_flow_sample,
    flow_interpolate,
    masked_velocity_mse,
    sample_flow_noise,
)
from vggt_omega.utils.pose_enc import encoding_to_camera


class BoundedResidualGate(nn.Module):
    """A checkpointed scalar gate whose forward value stays within [0, 1]."""

    def __init__(self, initial_value: float) -> None:
        super().__init__()
        if not math.isfinite(initial_value) or not 0.0 <= initial_value <= 1.0:
            raise ValueError("residual gate initial value must be finite and within [0, 1]")
        self.value = nn.Parameter(torch.tensor(float(initial_value)))

    def forward(self) -> torch.Tensor:
        bounded = self.value.clamp(0.0, 1.0)
        # Preserve bounded inference while keeping a raw value outside the
        # interval recoverable by gradient descent.
        return self.value + (bounded - self.value).detach()


class PixelPerfectDepthTrainingModel(nn.Module):
    """Keep VGGT as a pretrained anchor and refine its depth in pixel space."""

    def __init__(
        self,
        base_model: nn.Module,
        *,
        semantic_input_dim: int,
        hidden_dim: int,
        refiner_depth: int,
        num_heads: int,
        coarse_patch_size: int,
        fine_patch_size: int,
        temporal_depth: int,
        log_residual_scale: float,
        gradient_loss_weight: float,
        ode_steps: int,
        max_depth_m: float,
        geometry_enabled: bool,
        reference_mode: str = "random_valid",
        residual_gate_initial: float = 1.0,
        correspondence_enabled: bool = False,
        correspondence_hidden_dim: int = 256,
        correspondence_pair_chunk_size: int = 2,
    ) -> None:
        super().__init__()
        if not isinstance(base_model, nn.Module):
            raise TypeError("base_model must be a torch module")
        if reference_mode not in {"first", "random_valid"}:
            raise ValueError("reference_mode must be first or random_valid")
        for name, value in (
            ("log_residual_scale", log_residual_scale),
            ("max_depth_m", max_depth_m),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(gradient_loss_weight) or gradient_loss_weight < 0:
            raise ValueError("gradient_loss_weight must be finite and non-negative")
        if isinstance(ode_steps, bool) or not isinstance(ode_steps, int) or ode_steps <= 0:
            raise ValueError("ode_steps must be a positive integer")
        if not isinstance(geometry_enabled, bool):
            raise ValueError("geometry_enabled must be boolean")

        self.base_model = base_model
        self.patch_feature_dim = semantic_input_dim
        self.semantic_adapter = SemanticPromptAdapter(
            input_dim=semantic_input_dim,
            prompt_dim=hidden_dim,
            hidden_dim=hidden_dim,
        )
        self.temporal_mixer = TemporalSemanticMixer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            depth=temporal_depth,
        )
        self.refiner = PixelDepthFlowRefiner(
            hidden_dim=hidden_dim,
            depth=refiner_depth,
            num_heads=num_heads,
            coarse_patch_size=coarse_patch_size,
            fine_patch_size=fine_patch_size,
            in_channels=8 if geometry_enabled else 4,
        )
        self.residual_gate = BoundedResidualGate(residual_gate_initial)
        self.correspondence_head = (
            FactoredCorrespondenceHead(
                geometry_dim=semantic_input_dim,
                camera_dim=9,
                hidden_dim=correspondence_hidden_dim,
            )
            if correspondence_enabled
            else None
        )
        self.correspondence_active = correspondence_enabled
        if (
            isinstance(correspondence_pair_chunk_size, bool)
            or not isinstance(correspondence_pair_chunk_size, int)
            or correspondence_pair_chunk_size < 1
        ):
            raise ValueError("correspondence_pair_chunk_size must be a positive integer")
        self.correspondence_pair_chunk_size = correspondence_pair_chunk_size
        self.log_residual_scale = float(log_residual_scale)
        self.gradient_loss_weight = float(gradient_loss_weight)
        self.ode_steps = ode_steps
        self.max_depth_m = float(max_depth_m)
        self.geometry_enabled = geometry_enabled
        self.reference_mode = reference_mode

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Preserve the legacy base-model call unless refinement is explicit."""

        return self.base_model(images)

    def train(self, mode: bool = True) -> PixelPerfectDepthTrainingModel:
        super().train(mode)
        aggregator = getattr(self.base_model, "aggregator", None)
        if isinstance(aggregator, nn.Module):
            aggregator.eval()
        return self

    def forward_train(
        self,
        images: torch.Tensor,
        target_depth: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        generator: torch.Generator,
        dynamic_mask: torch.Tensor | None = None,
        frame_mask: torch.Tensor | None = None,
        normalization_scale_m: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        base, prompt, prompt_mask, geometry = self._prepare_context(
            images,
            valid_mask=valid_mask,
            dynamic_mask=dynamic_mask,
            frame_mask=frame_mask,
            normalization_scale_m=normalization_scale_m,
            generator=generator,
            training=True,
        )
        base_depth = _base_depth(base)
        if target_depth.shape != base_depth.shape or valid_mask.shape != target_depth.shape:
            raise ValueError("target_depth and valid_mask must match the base [B,S,H,W] depth")
        clean = encode_log_depth_residual(
            target_depth,
            base_depth,
            valid_mask,
            log_residual_scale=self.log_residual_scale,
        )
        flat_clean = clean.flatten(0, 1).unsqueeze(1)
        noise = sample_flow_noise(flat_clean, generator=generator)
        timestep = torch.sigmoid(
            torch.randn(
                (flat_clean.shape[0],),
                device=flat_clean.device,
                dtype=flat_clean.dtype,
                generator=generator,
            )
        )
        state, target_velocity = flow_interpolate(flat_clean, noise, timestep)
        predicted_velocity = self._predict_velocity(images, state, prompt, prompt_mask, geometry, timestep)
        flat_mask = valid_mask.flatten(0, 1).unsqueeze(1)
        flow_loss = masked_velocity_mse(predicted_velocity, target_velocity, flat_mask)
        clean_prediction = state - timestep[:, None, None, None] * predicted_velocity
        gradient_loss = depth_gradient_matching_loss(clean_prediction, flat_clean, flat_mask)
        residual_gate = self.residual_gate()
        refined_depth = decode_log_depth_residual(
            base_depth.flatten(0, 1),
            clean_prediction[:, 0] * residual_gate,
            log_residual_scale=self.log_residual_scale,
        ).reshape_as(base_depth)
        result = _without_patch_outputs(base)
        result["base_depth"] = base["depth"]
        result["depth"] = refined_depth[..., None]
        result["flow"] = flow_loss
        result["flow_gradient"] = gradient_loss
        result["flow_objective"] = flow_loss + self.gradient_loss_weight * gradient_loss
        result["residual_gate"] = residual_gate
        result["ode_steps"] = flow_loss.new_tensor(float(self.ode_steps))
        result["temporal_enabled"] = flow_loss.new_tensor(1.0)
        self._add_correspondence_outputs(result, base, images, frame_mask)
        return result

    @torch.no_grad()
    def forward_refine(
        self,
        images: torch.Tensor,
        *,
        generator: torch.Generator,
        valid_mask: torch.Tensor | None = None,
        dynamic_mask: torch.Tensor | None = None,
        frame_mask: torch.Tensor | None = None,
        normalization_scale_m: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        base, prompt, prompt_mask, geometry = self._prepare_context(
            images,
            valid_mask=valid_mask,
            dynamic_mask=dynamic_mask,
            frame_mask=frame_mask,
            normalization_scale_m=normalization_scale_m,
            generator=generator,
            training=False,
        )
        base_depth = _base_depth(base)
        initial_noise = sample_flow_noise(base_depth.flatten(0, 1).unsqueeze(1), generator=generator)

        def velocity_fn(state: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            return self._predict_velocity(images, state, prompt, prompt_mask, geometry, timestep)

        residual = euler_flow_sample(velocity_fn, initial_noise, steps=self.ode_steps)[:, 0]
        residual_gate = self.residual_gate()
        refined_depth = decode_log_depth_residual(
            base_depth.flatten(0, 1),
            residual * residual_gate,
            log_residual_scale=self.log_residual_scale,
        ).reshape_as(base_depth)
        result = _without_patch_outputs(base)
        result["base_depth"] = base["depth"]
        result["depth"] = refined_depth[..., None]
        result["residual_gate"] = residual_gate
        self._add_correspondence_outputs(result, base, images, frame_mask)
        return result

    def prepare_dynamic_context(
        self,
        images: torch.Tensor,
        *,
        initial_noise: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        dynamic_mask: torch.Tensor | None = None,
        frame_mask: torch.Tensor | None = None,
        normalization_scale_m: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return refined depth and patch features from one deterministic base pass.

        The caller owns ``initial_noise`` so grouping padded sequences cannot
        change the sample's refinement noise or consume the training RNG.
        Legacy ``forward`` and ``forward_refine`` retain their existing APIs.
        """

        generator = torch.Generator(device=images.device).manual_seed(0)
        base, prompt, prompt_mask, geometry = self._prepare_context(
            images,
            valid_mask=valid_mask,
            dynamic_mask=dynamic_mask,
            frame_mask=frame_mask,
            normalization_scale_m=normalization_scale_m,
            generator=generator,
            training=False,
        )
        base_depth = _base_depth(base)
        expected_noise = base_depth.flatten(0, 1).unsqueeze(1)
        if initial_noise.shape != expected_noise.shape:
            raise ValueError(
                "initial_noise must match flattened base depth shape "
                f"{tuple(expected_noise.shape)}, got {tuple(initial_noise.shape)}"
            )
        if (
            not initial_noise.is_floating_point()
            or initial_noise.dtype != expected_noise.dtype
            or initial_noise.device != expected_noise.device
        ):
            raise TypeError("initial_noise must match the base depth dtype and device")
        if not torch.isfinite(initial_noise).all():
            raise ValueError("initial_noise must contain only finite values")

        def velocity_fn(state: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            return self._predict_velocity(images, state, prompt, prompt_mask, geometry, timestep)

        residual = euler_flow_sample(velocity_fn, initial_noise, steps=self.ode_steps)[:, 0]
        residual_gate = self.residual_gate()
        refined_depth = decode_log_depth_residual(
            base_depth.flatten(0, 1),
            residual * residual_gate,
            log_residual_scale=self.log_residual_scale,
        ).reshape_as(base_depth)
        result = dict(base)
        result["base_depth"] = base["depth"]
        result["depth"] = refined_depth[..., None]
        result["residual_gate"] = residual_gate
        return result

    def set_curriculum_trainable(
        self,
        *,
        train_refiner: bool,
        train_correspondence: bool,
        train_base_heads: bool = False,
    ) -> None:
        """Toggle stage groups without changing the optimizer parameter set."""

        if not all(isinstance(flag, bool) for flag in (train_refiner, train_correspondence, train_base_heads)):
            raise ValueError("curriculum trainable flags must be boolean")
        self.base_model.requires_grad_(False)
        if train_base_heads:
            for name in ("camera_head", "dense_head"):
                module = getattr(self.base_model, name, None)
                if isinstance(module, nn.Module):
                    module.requires_grad_(True)
            dense_head = getattr(self.base_model, "dense_head", None)
            confidence = getattr(dense_head, "proj_conf", None)
            if isinstance(confidence, nn.Module):
                confidence.requires_grad_(False)
        for module in (self.semantic_adapter, self.temporal_mixer, self.refiner, self.residual_gate):
            module.requires_grad_(train_refiner)
        if self.correspondence_head is None:
            if train_correspondence:
                raise ValueError("correspondence stage requested without a correspondence head")
        else:
            self.correspondence_head.requires_grad_(train_correspondence)
        self.correspondence_active = train_correspondence

    def _add_correspondence_outputs(
        self,
        result: dict[str, torch.Tensor],
        base: dict[str, torch.Tensor],
        images: torch.Tensor,
        frame_mask: torch.Tensor | None,
    ) -> None:
        if self.correspondence_head is None or not self.correspondence_active:
            return
        batch_size, frame_count, _, height, width = images.shape
        if frame_mask is not None and not bool(frame_mask.all()):
            raise ValueError("correspondence head currently requires batches without padded frames")
        pairs = torch.tensor(
            [(source, target) for source in range(frame_count) for target in range(frame_count) if source != target],
            dtype=torch.long,
            device=images.device,
        )
        if pairs.numel() == 0:
            raise ValueError("correspondence head requires at least two frames")
        pair_indices = pairs[None].expand(batch_size, -1, -1)
        source_grid_value = base["patch_grid_hw"]
        source_grid = (int(source_grid_value[0]), int(source_grid_value[1]))
        result["correspondence_flow_pixels"] = self.correspondence_head(
            base["patch_features"],
            base["pose_enc"],
            pair_indices,
            source_grid_hw=source_grid,
            output_hw=(height, width),
            pair_chunk_size=self.correspondence_pair_chunk_size,
        )
        result["correspondence_pair_indices"] = pair_indices

    def _prepare_context(
        self,
        images: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None,
        dynamic_mask: torch.Tensor | None,
        frame_mask: torch.Tensor | None,
        normalization_scale_m: torch.Tensor | None,
        generator: torch.Generator,
        training: bool,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor | None]:
        base = self.base_model(images, return_patch_features=True)
        features = base["patch_features"]
        feature_mask = base["patch_valid_mask"]
        source_grid_value = base["patch_grid_hw"]
        source_grid = (int(source_grid_value[0]), int(source_grid_value[1]))
        batch_size, frame_count, _, height, width = images.shape
        target_grid = (
            math.ceil(height / self.refiner.coarse_patch_size),
            math.ceil(width / self.refiner.coarse_patch_size),
        )
        prompt, prompt_mask = self.semantic_adapter(
            features,
            feature_mask,
            source_grid_hw=source_grid,
            target_grid_hw=target_grid,
        )
        prompt = prompt.reshape(batch_size, frame_count, -1, self.refiner.hidden_dim)
        prompt_mask = prompt_mask.reshape(batch_size, frame_count, -1)
        if frame_mask is None:
            frame_mask = torch.ones((batch_size, frame_count), dtype=torch.bool, device=images.device)
        reference_indices = self._reference_indices(frame_mask, generator, training=training)
        prompt = self.temporal_mixer(
            prompt,
            prompt_mask,
            frame_mask=frame_mask,
            reference_indices=reference_indices,
        )
        geometry = self._geometry_condition(
            images,
            base,
            valid_mask=valid_mask,
            dynamic_mask=dynamic_mask,
            frame_mask=frame_mask,
            normalization_scale_m=normalization_scale_m,
        )
        return base, prompt.flatten(0, 1), prompt_mask.flatten(0, 1), geometry

    def _reference_indices(
        self,
        frame_mask: torch.Tensor,
        generator: torch.Generator,
        *,
        training: bool,
    ) -> torch.Tensor:
        if not frame_mask.any(dim=1).all():
            raise ValueError("every sample must contain at least one valid frame")
        if self.reference_mode == "first" or not training:
            return frame_mask.to(torch.int64).argmax(dim=1)
        indices = []
        for sample_mask in frame_mask:
            valid_indices = torch.nonzero(sample_mask, as_tuple=False).flatten()
            selected = torch.randint(len(valid_indices), (), generator=generator, device=frame_mask.device)
            indices.append(valid_indices[selected])
        return torch.stack(indices)

    def _geometry_condition(
        self,
        images: torch.Tensor,
        base: dict[str, torch.Tensor],
        *,
        valid_mask: torch.Tensor | None,
        dynamic_mask: torch.Tensor | None,
        frame_mask: torch.Tensor,
        normalization_scale_m: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if not self.geometry_enabled:
            return None
        base_depth = _base_depth(base).detach()
        batch_size = base_depth.shape[0]
        usable = torch.isfinite(base_depth) & (base_depth > 0)
        if valid_mask is not None:
            usable &= valid_mask
        if normalization_scale_m is not None:
            scale = normalization_scale_m.reshape(batch_size, 1, 1, 1).to(base_depth)
            usable &= base_depth * scale < self.max_depth_m
        pose = base["pose_enc"].detach().float()
        extrinsics, intrinsics = encoding_to_camera(pose, images.shape[-2:], build_intrinsics=True)
        assert intrinsics is not None
        condition = build_warped_neighbor_condition(
            images.float(),
            base_depth.float(),
            intrinsics.float(),
            extrinsics.float(),
            usable,
            dynamic_mask=dynamic_mask,
            frame_mask=frame_mask,
            max_depth_m=1e30,
        )["condition"]
        return condition.flatten(0, 1).to(images)

    def _predict_velocity(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        prompt: torch.Tensor,
        prompt_mask: torch.Tensor,
        geometry: torch.Tensor | None,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        flat_images = images.flatten(0, 1)
        components = [flat_images, state]
        if geometry is not None:
            components.append(geometry)
        return self.refiner(torch.cat(components, dim=1), prompt, prompt_mask, timestep)


def _base_depth(base: dict[str, torch.Tensor]) -> torch.Tensor:
    depth = base.get("depth")
    if not isinstance(depth, torch.Tensor) or depth.ndim != 5 or depth.shape[-1] != 1:
        raise ValueError("base model depth must have shape [B,S,H,W,1]")
    return depth[..., 0]


def _without_patch_outputs(base: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: value for key, value in base.items() if key not in {"patch_features", "patch_grid_hw", "patch_valid_mask"}
    }
