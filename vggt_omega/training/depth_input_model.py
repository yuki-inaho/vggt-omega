# SPDX-License-Identifier: Apache-2.0
"""Mapped-depth token adapter for opt-in RGB-D training."""

from __future__ import annotations

import torch
from torch import nn


class MappedDepthTokenAdapter(nn.Module):
    """Encode mapped depth and validity into residual RGB patch tokens."""

    def __init__(self, *, patch_size: int = 16, embed_dim: int = 1024) -> None:
        super().__init__()
        if not isinstance(patch_size, int) or isinstance(patch_size, bool) or patch_size <= 0:
            raise ValueError("patch_size must be a positive integer")
        if not isinstance(embed_dim, int) or isinstance(embed_dim, bool) or embed_dim <= 0:
            raise ValueError("embed_dim must be a positive integer")
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth_patch_embed = nn.Conv2d(
            2,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.depth_placeholder = nn.Parameter(torch.zeros(embed_dim))

    def forward(
        self,
        depth: torch.Tensor,
        valid_mask: torch.Tensor,
        availability: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``[B*S, patches, embed_dim]`` residual tokens."""

        if not isinstance(depth, torch.Tensor) or depth.ndim != 5 or depth.shape[2] != 1:
            raise ValueError("depth must have shape [B,S,1,H,W]")
        if not depth.is_floating_point():
            raise ValueError("depth must use a floating-point dtype")
        if not isinstance(valid_mask, torch.Tensor) or valid_mask.shape != depth.shape:
            raise ValueError("valid_mask must exactly match depth shape [B,S,1,H,W]")
        if valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must use bool dtype")
        batch_size, num_frames, _, height, width = depth.shape
        if not isinstance(availability, torch.Tensor) or availability.shape != (batch_size, num_frames):
            raise ValueError("availability must have shape [B,S]")
        if availability.dtype != torch.bool:
            raise ValueError("availability must use bool dtype")
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError("depth height and width must be divisible by patch_size")
        if valid_mask.device != depth.device or availability.device != depth.device:
            raise ValueError("depth, valid_mask, and availability must use the same device")
        if self.depth_patch_embed.weight.device != depth.device or self.depth_placeholder.device != depth.device:
            raise ValueError("adapter parameters and depth must use the same device")
        if self.depth_patch_embed.weight.dtype != depth.dtype or self.depth_placeholder.dtype != depth.dtype:
            raise ValueError("adapter parameters and depth must use the same dtype")

        available_pixels = valid_mask & availability[:, :, None, None, None]
        frame_has_depth = valid_mask.flatten(2).any(dim=2)
        if torch.any(availability & ~frame_has_depth):
            raise ValueError("every available frame must contain at least one valid depth pixel")
        available_depth = depth[available_pixels]
        if not torch.isfinite(available_depth).all():
            raise ValueError("available valid depth values must be finite")
        if torch.any(available_depth <= 0):
            raise ValueError("available valid depth values must be positive")

        clean_depth = torch.where(available_pixels, depth, torch.zeros_like(depth))
        valid_count = available_pixels.sum(dim=(1, 2, 3, 4), keepdim=True)
        sequence_mean = clean_depth.sum(dim=(1, 2, 3, 4), keepdim=True) / valid_count.clamp_min(1)
        normalized_depth = clean_depth / sequence_mean.clamp_min(torch.finfo(depth.dtype).tiny)
        adapter_input = torch.cat((normalized_depth, available_pixels.to(dtype=depth.dtype)), dim=2)
        adapter_input = adapter_input.reshape(batch_size * num_frames, 2, height, width)
        encoded = self.depth_patch_embed(adapter_input).flatten(2).transpose(1, 2)

        patch_count = (height // self.patch_size) * (width // self.patch_size)
        placeholder = self.depth_placeholder.expand(batch_size * num_frames, patch_count, self.embed_dim)
        return torch.where(availability.reshape(-1, 1, 1), encoded, placeholder)


class DepthInputTrainingModel(nn.Module):
    """Keep mapped-depth conditioning outside the released base model."""

    def __init__(self, base_model: nn.Module, adapter: MappedDepthTokenAdapter) -> None:
        super().__init__()
        self.base_model = base_model
        self.adapter = adapter

    @property
    def aggregator(self) -> nn.Module:
        aggregator = getattr(self.base_model, "aggregator", None)
        if not isinstance(aggregator, nn.Module):
            raise ValueError("base_model must expose an aggregator module")
        return aggregator

    def forward(
        self,
        images: torch.Tensor,
        *,
        mapped_depth: torch.Tensor,
        valid_mask: torch.Tensor,
        availability: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if images.ndim != 5 or images.shape[:2] != mapped_depth.shape[:2]:
            raise ValueError("images and mapped_depth must share [B,S]")
        residual = self.adapter(mapped_depth, valid_mask, availability)
        return self.base_model(images, spatial_token_residual=residual)


def sample_depth_availability(
    batch_size: int,
    num_frames: int,
    *,
    seed: int,
    epoch: int,
    optimizer_step: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Select a uniform k in ``[0,S]`` and k unique frames per sample."""

    for value, name, allow_zero in (
        (batch_size, "batch_size", False),
        (num_frames, "num_frames", False),
        (seed, "seed", True),
        (epoch, "epoch", True),
        (optimizer_step, "optimizer_step", True),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < int(not allow_zero):
            raise ValueError(f"{name} is outside its supported range")
    mixed_seed = seed + epoch * 1_000_003 + optimizer_step * 10_007
    generator = torch.Generator(device="cpu").manual_seed(mixed_seed)
    counts = torch.randint(0, num_frames + 1, (batch_size, 1), generator=generator)
    scores = torch.rand(batch_size, num_frames, generator=generator)
    ranks = scores.argsort(dim=1).argsort(dim=1)
    return (ranks < counts).to(device=device)


def fixed_depth_availability(
    batch_size: int,
    num_frames: int,
    provided_frames: int,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    """Return a deterministic first-k mask for validation comparisons."""

    if (
        isinstance(provided_frames, bool)
        or not isinstance(provided_frames, int)
        or provided_frames < 0
        or provided_frames > num_frames
    ):
        raise ValueError("provided_frames must be within [0,num_frames]")
    indices = torch.arange(num_frames, device=device)
    return (indices < provided_frames).expand(batch_size, num_frames)
