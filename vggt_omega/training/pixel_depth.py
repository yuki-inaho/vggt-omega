"""Pixel-space depth refinement primitives inspired by Pixel-Perfect Depth.

The released PPD model predicts per-image relative log depth.  VGGT-Omega also
needs a sequence-consistent scale for camera/depth reconstruction, so this
module instead represents a bounded log-depth residual around VGGT's positive
base prediction.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn


def flow_interpolate(
    clean: torch.Tensor,
    noise: torch.Tensor,
    timestep: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interpolate PPD's straight probability path from clean (t=0) to noise (t=1)."""

    _validate_matching_float_tensors("clean", clean, "noise", noise)
    if clean.ndim < 1:
        raise ValueError("clean and noise must include a batch dimension")
    batch_size = clean.shape[0]
    if timestep.shape != (batch_size,) or not timestep.is_floating_point():
        raise ValueError(f"timestep must be floating point with shape [{batch_size}]")
    if timestep.device != clean.device:
        raise ValueError("timestep must share the clean/noise device")
    if not torch.isfinite(timestep).all() or torch.any((timestep < 0) | (timestep > 1)):
        raise ValueError("timestep must contain finite values in [0,1]")
    interpolation_weight = timestep.reshape(batch_size, *((1,) * (clean.ndim - 1)))
    velocity = noise - clean
    return clean + interpolation_weight * velocity, velocity


def sample_flow_noise(
    reference: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample Gaussian flow noise using an explicit caller-owned RNG."""

    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator")
    if not reference.is_floating_point():
        raise TypeError("reference must be floating point")
    return torch.randn(
        reference.shape,
        generator=generator,
        device=reference.device,
        dtype=reference.dtype,
    )


def masked_velocity_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean squared velocity error over valid pixels, with graph-connected empty zero."""

    _validate_matching_float_tensors("prediction", prediction, "target", target)
    _validate_exact_mask(valid_mask, prediction)
    squared_error = (prediction - target).square()
    if not valid_mask.any():
        return prediction.sum() * 0
    return squared_error[valid_mask].mean()


def depth_gradient_matching_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """L1 match horizontal/vertical depth gradients where both endpoints are valid."""

    _validate_matching_float_tensors("prediction", prediction, "target", target)
    _validate_exact_mask(valid_mask, prediction)
    if prediction.ndim != 4:
        raise ValueError("prediction and target must have shape [N,C,H,W]")

    horizontal_valid = valid_mask[..., :, 1:] & valid_mask[..., :, :-1]
    vertical_valid = valid_mask[..., 1:, :] & valid_mask[..., :-1, :]
    horizontal_error = (
        (prediction[..., :, 1:] - prediction[..., :, :-1]) - (target[..., :, 1:] - target[..., :, :-1])
    ).abs()
    vertical_error = (
        (prediction[..., 1:, :] - prediction[..., :-1, :]) - (target[..., 1:, :] - target[..., :-1, :])
    ).abs()
    valid_error = torch.cat(
        [horizontal_error[horizontal_valid], vertical_error[vertical_valid]],
        dim=0,
    )
    if valid_error.numel() == 0:
        return prediction.sum() * 0
    return valid_error.mean()


def euler_flow_sample(
    velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    initial_noise: torch.Tensor,
    *,
    steps: int,
) -> torch.Tensor:
    """Integrate the learned velocity field backward from t=1 to t=0."""

    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if initial_noise.ndim < 1 or not initial_noise.is_floating_point():
        raise ValueError("initial_noise must be floating point with a batch dimension")
    if not torch.isfinite(initial_noise).all():
        raise ValueError("initial_noise must contain only finite values")

    state = initial_noise
    delta_t = -1.0 / steps
    for index in range(steps):
        timestep = torch.full(
            (state.shape[0],),
            1.0 - index / steps,
            device=state.device,
            dtype=state.dtype,
        )
        velocity = velocity_fn(state, timestep)
        if velocity.shape != state.shape or not velocity.is_floating_point():
            raise ValueError("velocity_fn must return a floating tensor matching the state shape")
        if velocity.device != state.device or not torch.isfinite(velocity).all():
            raise ValueError("velocity_fn output must be finite and share the state device")
        state = state + delta_t * velocity
    return state


def encode_log_depth_residual(
    target_depth: torch.Tensor,
    base_depth: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    log_residual_scale: float,
) -> torch.Tensor:
    """Encode a bounded, scale-equivariant clean residual target."""

    _validate_depth_inputs(target_depth, base_depth, valid_mask)
    scale = _positive_finite_float("log_residual_scale", log_residual_scale)
    if torch.any(~torch.isfinite(target_depth[valid_mask])):
        raise ValueError("target_depth contains non-finite valid values")
    if torch.any(target_depth[valid_mask] <= 0):
        raise ValueError("target_depth must be positive on valid pixels")

    safe_target = torch.where(valid_mask, target_depth, torch.ones_like(target_depth))
    residual = (torch.log(safe_target) - torch.log(base_depth.detach())) / scale
    residual = residual.clamp(-1.0, 1.0)
    return torch.where(valid_mask, residual, torch.zeros_like(residual))


def decode_log_depth_residual(
    base_depth: torch.Tensor,
    residual: torch.Tensor,
    *,
    log_residual_scale: float,
) -> torch.Tensor:
    """Decode a residual without requiring ground-truth depth percentiles."""

    if base_depth.shape != residual.shape:
        raise ValueError(f"base_depth and residual shapes differ: {base_depth.shape} != {residual.shape}")
    if not base_depth.is_floating_point() or not residual.is_floating_point():
        raise TypeError("base_depth and residual must be floating point")
    if not torch.isfinite(base_depth).all() or torch.any(base_depth <= 0):
        raise ValueError("base_depth must contain only finite positive values")
    if not torch.isfinite(residual).all():
        raise ValueError("residual must contain only finite values")
    scale = _positive_finite_float("log_residual_scale", log_residual_scale)
    return base_depth * torch.exp(scale * residual.clamp(-1.0, 1.0))


def l2_normalize_patch_features(
    features: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """L2-normalize semantic tokens and zero invalid or zero-norm tokens."""

    if features.ndim != 4:
        raise ValueError(f"features must have shape [B,S,T,D], got {tuple(features.shape)}")
    if not features.is_floating_point():
        raise TypeError("features must be floating point")
    if valid_mask.shape != features.shape[:3] or valid_mask.dtype is not torch.bool:
        raise ValueError("valid_mask must be bool with shape [B,S,T]")
    if not torch.isfinite(features).all():
        raise ValueError("features contains non-finite values")
    epsilon = _positive_finite_float("eps", eps)
    norms = torch.linalg.vector_norm(features, dim=-1, keepdim=True)
    normalized = features / norms.clamp_min(epsilon)
    usable = valid_mask.unsqueeze(-1) & (norms > epsilon)
    return torch.where(usable, normalized, torch.zeros_like(normalized))


class SemanticPromptAdapter(nn.Module):
    """Resize normalized VGGT patch tokens and project them to a prompt width."""

    def __init__(self, *, input_dim: int, prompt_dim: int, hidden_dim: int) -> None:
        super().__init__()
        for name, value in (("input_dim", input_dim), ("prompt_dim", prompt_dim), ("hidden_dim", hidden_dim)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.input_dim = input_dim
        self.prompt_dim = prompt_dim
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, prompt_dim),
        )

    def forward(
        self,
        features: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        source_grid_hw: tuple[int, int],
        target_grid_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_height, source_width = _grid_hw("source_grid_hw", source_grid_hw)
        target_height, target_width = _grid_hw("target_grid_hw", target_grid_hw)
        if features.ndim != 4 or features.shape[-1] != self.input_dim:
            raise ValueError(f"features must have shape [B,S,T,{self.input_dim}], got {tuple(features.shape)}")
        if features.shape[2] != source_height * source_width:
            raise ValueError("feature token count does not match the source grid")

        normalized = l2_normalize_patch_features(features, valid_mask)
        batch_size, frame_count, _, feature_dim = normalized.shape
        flat_batch = batch_size * frame_count
        feature_grid = normalized.reshape(flat_batch, source_height, source_width, feature_dim).permute(0, 3, 1, 2)
        mask_grid = valid_mask.reshape(flat_batch, 1, source_height, source_width).float()
        if (source_height, source_width) != (target_height, target_width):
            feature_grid = F.interpolate(
                feature_grid,
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            )
            mask_grid = F.interpolate(mask_grid, size=(target_height, target_width), mode="nearest")

        resized = feature_grid.permute(0, 2, 3, 1).reshape(flat_batch, target_height * target_width, feature_dim)
        resized_mask = mask_grid.reshape(flat_batch, target_height * target_width).bool()
        prompt = self.projection(resized)
        prompt = torch.where(resized_mask.unsqueeze(-1), prompt, torch.zeros_like(prompt))
        return prompt, resized_mask


class PixelDepthFlowRefiner(nn.Module):
    """A compact coarse-to-fine pixel-space velocity predictor.

    The diffusion state remains a full-resolution depth residual.  Transformer
    computation starts on a coarse patch grid and expands once to a fine grid,
    following the useful architectural principle of Cascade DiT without
    copying the 500M-parameter reference implementation.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        coarse_patch_size: int = 16,
        fine_patch_size: int = 8,
        in_channels: int = 4,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        for name, value in (
            ("hidden_dim", hidden_dim),
            ("depth", depth),
            ("num_heads", num_heads),
            ("coarse_patch_size", coarse_patch_size),
            ("fine_patch_size", fine_patch_size),
            ("in_channels", in_channels),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if depth % 2:
            raise ValueError("depth must be even so coarse and fine stages have equal length")
        if hidden_dim % num_heads or hidden_dim % 4:
            raise ValueError("hidden_dim must be divisible by num_heads and four")
        if coarse_patch_size != 2 * fine_patch_size:
            raise ValueError("coarse_patch_size must equal two times fine_patch_size")
        ratio = _positive_finite_float("mlp_ratio", mlp_ratio)

        self.hidden_dim = hidden_dim
        self.depth = depth
        self.coarse_patch_size = coarse_patch_size
        self.fine_patch_size = fine_patch_size
        self.in_channels = in_channels
        self.input_projection = nn.Conv2d(
            in_channels,
            hidden_dim,
            kernel_size=coarse_patch_size,
            stride=coarse_patch_size,
        )
        self.time_embedding = _TimestepEmbedding(hidden_dim)
        self.coarse_blocks = nn.ModuleList(
            [_AdaptiveTransformerBlock(hidden_dim, num_heads, ratio) for _ in range(depth // 2)]
        )
        self.semantic_fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, 4 * hidden_dim),
            nn.SiLU(),
            nn.Linear(4 * hidden_dim, 4 * hidden_dim),
        )
        self.fine_blocks = nn.ModuleList(
            [_AdaptiveTransformerBlock(hidden_dim, num_heads, ratio) for _ in range(depth // 2)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.output_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim))
        self.output_projection = nn.Linear(hidden_dim, fine_patch_size * fine_patch_size)
        self._initialize_conditioning()

    def _initialize_conditioning(self) -> None:
        for block in (*self.coarse_blocks, *self.fine_blocks):
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        inputs: torch.Tensor,
        semantic_prompt: torch.Tensor,
        semantic_mask: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != self.in_channels:
            raise ValueError(f"inputs must have shape [N,{self.in_channels},H,W], got {tuple(inputs.shape)}")
        if not inputs.is_floating_point() or not torch.isfinite(inputs).all():
            raise ValueError("inputs must be floating point and finite")
        batch_size, _, original_height, original_width = inputs.shape
        if timestep.shape != (batch_size,) or not timestep.is_floating_point():
            raise ValueError(f"timestep must be floating point with shape [{batch_size}]")
        if not torch.isfinite(timestep).all() or torch.any((timestep < 0) | (timestep > 1)):
            raise ValueError("timestep must contain finite values in [0,1]")
        if semantic_prompt.shape[0] != batch_size or semantic_prompt.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"semantic_prompt must have shape [N,T,{self.hidden_dim}], got {tuple(semantic_prompt.shape)}"
            )
        if not semantic_prompt.is_floating_point() or not torch.isfinite(semantic_prompt).all():
            raise ValueError("semantic_prompt must be floating point and finite")
        if semantic_mask.shape != semantic_prompt.shape[:2] or semantic_mask.dtype is not torch.bool:
            raise ValueError("semantic_mask must be bool with shape [N,T]")

        pad_height = (-original_height) % self.coarse_patch_size
        pad_width = (-original_width) % self.coarse_patch_size
        padded = F.pad(inputs, (0, pad_width, 0, pad_height))
        coarse_grid = self.input_projection(padded)
        coarse_height, coarse_width = coarse_grid.shape[-2:]
        coarse_tokens = coarse_grid.flatten(2).transpose(1, 2)
        if semantic_prompt.shape[1] != coarse_height * coarse_width:
            raise ValueError("semantic prompt token count does not match the padded coarse grid")

        time_embedding = self.time_embedding(timestep)
        coarse_tokens = coarse_tokens + _position_embedding_2d(
            coarse_height,
            coarse_width,
            self.hidden_dim,
            device=coarse_tokens.device,
            dtype=coarse_tokens.dtype,
        )
        for block in self.coarse_blocks:
            coarse_tokens = block(coarse_tokens, time_embedding)

        masked_prompt = torch.where(semantic_mask.unsqueeze(-1), semantic_prompt, torch.zeros_like(semantic_prompt))
        expanded = self.semantic_fusion(torch.cat([coarse_tokens, masked_prompt], dim=-1))
        fine_grid = expanded.reshape(
            batch_size,
            coarse_height,
            coarse_width,
            2,
            2,
            self.hidden_dim,
        )
        fine_grid = fine_grid.permute(0, 5, 1, 3, 2, 4).reshape(
            batch_size,
            self.hidden_dim,
            2 * coarse_height,
            2 * coarse_width,
        )
        fine_height, fine_width = fine_grid.shape[-2:]
        fine_tokens = fine_grid.flatten(2).transpose(1, 2)
        fine_tokens = fine_tokens + _position_embedding_2d(
            fine_height,
            fine_width,
            self.hidden_dim,
            device=fine_tokens.device,
            dtype=fine_tokens.dtype,
        )
        for block in self.fine_blocks:
            fine_tokens = block(fine_tokens, time_embedding)

        shift, scale = self.output_modulation(time_embedding).chunk(2, dim=-1)
        output_tokens = self.output_norm(fine_tokens) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        output_tokens = self.output_projection(output_tokens)
        velocity = output_tokens.reshape(
            batch_size,
            fine_height,
            fine_width,
            self.fine_patch_size,
            self.fine_patch_size,
            1,
        )
        velocity = velocity.permute(0, 5, 1, 3, 2, 4).reshape(
            batch_size,
            1,
            fine_height * self.fine_patch_size,
            fine_width * self.fine_patch_size,
        )
        return velocity[..., :original_height, :original_width]


class _TimestepEmbedding(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        return self.mlp(_sinusoidal_embedding(timestep, self.hidden_dim))


class _AdaptiveTransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm_mlp = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim))

    def forward(self, tokens: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        shift_attention, scale_attention, gate_attention, shift_mlp, scale_mlp, gate_mlp = self.modulation(
            time_embedding
        ).chunk(6, dim=-1)
        normalized = self.norm_attention(tokens) * (1 + scale_attention.unsqueeze(1))
        normalized = normalized + shift_attention.unsqueeze(1)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        tokens = tokens + gate_attention.unsqueeze(1) * attended
        normalized = self.norm_mlp(tokens) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        return tokens + gate_mlp.unsqueeze(1) * self.mlp(normalized)


def _sinusoidal_embedding(values: torch.Tensor, dimension: int) -> torch.Tensor:
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(10_000) * torch.arange(half, device=values.device, dtype=torch.float32) / max(half - 1, 1)
    )
    angles = values.float().unsqueeze(-1) * 1_000 * frequencies
    embedding = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
    if embedding.shape[-1] < dimension:
        embedding = F.pad(embedding, (0, dimension - embedding.shape[-1]))
    return embedding.to(dtype=values.dtype)


def _position_embedding_2d(
    height: int,
    width: int,
    dimension: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    quarter = dimension // 4
    frequencies = torch.exp(
        -math.log(10_000) * torch.arange(quarter, device=device, dtype=torch.float32) / max(quarter - 1, 1)
    )
    y_coordinates = torch.arange(height, device=device, dtype=torch.float32)
    x_coordinates = torch.arange(width, device=device, dtype=torch.float32)
    y_angles = y_coordinates[:, None] * frequencies[None]
    x_angles = x_coordinates[:, None] * frequencies[None]
    y_embedding = torch.cat([torch.sin(y_angles), torch.cos(y_angles)], dim=-1)
    x_embedding = torch.cat([torch.sin(x_angles), torch.cos(x_angles)], dim=-1)
    grid = torch.cat(
        [
            y_embedding[:, None, :].expand(height, width, -1),
            x_embedding[None, :, :].expand(height, width, -1),
        ],
        dim=-1,
    )
    return grid.reshape(1, height * width, dimension).to(dtype=dtype)


def _validate_depth_inputs(
    target_depth: torch.Tensor,
    base_depth: torch.Tensor,
    valid_mask: torch.Tensor,
) -> None:
    if target_depth.shape != base_depth.shape:
        raise ValueError(f"target_depth and base_depth shapes differ: {target_depth.shape} != {base_depth.shape}")
    if target_depth.ndim != 4:
        raise ValueError(f"depth tensors must have shape [B,S,H,W], got {tuple(target_depth.shape)}")
    if valid_mask.shape != target_depth.shape or valid_mask.dtype is not torch.bool:
        raise ValueError("valid_mask must be bool and match the depth shape")
    if not target_depth.is_floating_point() or not base_depth.is_floating_point():
        raise TypeError("target_depth and base_depth must be floating point")
    if target_depth.device != base_depth.device or valid_mask.device != target_depth.device:
        raise ValueError("target_depth, base_depth, and valid_mask must share a device")
    if not torch.isfinite(base_depth).all() or torch.any(base_depth <= 0):
        raise ValueError("base_depth must contain only finite positive values")


def _validate_matching_float_tensors(
    first_name: str,
    first: torch.Tensor,
    second_name: str,
    second: torch.Tensor,
) -> None:
    if first.shape != second.shape:
        raise ValueError(f"{first_name} and {second_name} shapes differ: {first.shape} != {second.shape}")
    if not first.is_floating_point() or not second.is_floating_point():
        raise TypeError(f"{first_name} and {second_name} must be floating point")
    if first.device != second.device:
        raise ValueError(f"{first_name} and {second_name} must share a device")
    if not torch.isfinite(first).all() or not torch.isfinite(second).all():
        raise ValueError(f"{first_name} and {second_name} must contain only finite values")


def _validate_exact_mask(mask: torch.Tensor, reference: torch.Tensor) -> None:
    if mask.shape != reference.shape or mask.dtype is not torch.bool:
        raise ValueError("valid_mask must be bool and exactly match the tensor shape")
    if mask.device != reference.device:
        raise ValueError("valid_mask must share the tensor device")


def _positive_finite_float(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def _grid_hw(name: str, value: tuple[int, int]) -> tuple[int, int]:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value)
    ):
        raise ValueError(f"{name} must contain two positive integers")
    return value
