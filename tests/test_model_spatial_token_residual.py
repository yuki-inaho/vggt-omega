from __future__ import annotations

import pytest
import torch
from torch import nn

from vggt_omega.models.aggregator import Aggregator
from vggt_omega.models.vggt_omega import VGGTOmega


class _TinyPatchEmbed(nn.Module):
    def __init__(self, patch_size: int, embed_dim: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = images.shape
        patch_count = (height // self.patch_size) * (width // self.patch_size)
        return torch.zeros(batch, patch_count, self.embed_dim, dtype=images.dtype, device=images.device)


class _TinyRope(nn.Module):
    def forward(self, *, H: int, W: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(H * W, 1), torch.zeros(H * W, 1)


class _IdentityAttention(nn.Module):
    def forward(self, tokens: torch.Tensor, rope: object) -> torch.Tensor:
        return tokens


def _tiny_aggregator() -> Aggregator:
    aggregator = Aggregator.__new__(Aggregator)
    nn.Module.__init__(aggregator)
    aggregator.patch_embed = _TinyPatchEmbed(patch_size=2, embed_dim=3)
    aggregator.rope_embed = _TinyRope()
    aggregator.frame_blocks = nn.ModuleList([_IdentityAttention()])
    aggregator.inter_frame_blocks = nn.ModuleList([_IdentityAttention()])
    aggregator.depth = 1
    aggregator.patch_size = 2
    aggregator.cached_layer_indices = {0}
    aggregator.camera_token = nn.Parameter(torch.zeros(1, 2, 1, 3))
    aggregator.register_token = nn.Parameter(torch.zeros(1, 2, 1, 3))
    aggregator.patch_token_start = 2
    aggregator.inter_frame_attention_types = ["global"]
    aggregator.register_buffer("_resnet_mean", torch.zeros(1, 1, 3, 1, 1), persistent=False)
    aggregator.register_buffer("_resnet_std", torch.ones(1, 1, 3, 1, 1), persistent=False)
    return aggregator


def test_aggregator_zero_residual_is_exact_parity_and_state_keys_do_not_change() -> None:
    aggregator = _tiny_aggregator()
    images = torch.zeros(1, 2, 3, 4, 6)
    state_keys = tuple(aggregator.state_dict())

    baseline, baseline_start = aggregator(images)
    with_zero, zero_start = aggregator(images, spatial_token_residual=torch.zeros(2, 6, 3))

    assert baseline_start == zero_start == 2
    assert tuple(aggregator.state_dict()) == state_keys
    torch.testing.assert_close(with_zero[-1], baseline[-1], rtol=0, atol=0)


def test_aggregator_adds_residual_only_to_spatial_tokens_once() -> None:
    aggregator = _tiny_aggregator()
    images = torch.zeros(1, 2, 3, 4, 6)
    baseline, patch_start = aggregator(images)
    injected, _ = aggregator(images, spatial_token_residual=torch.ones(2, 6, 3))

    difference = injected[-1] - baseline[-1]
    torch.testing.assert_close(difference[:, :, :patch_start], torch.zeros_like(difference[:, :, :patch_start]))
    torch.testing.assert_close(difference[:, :, patch_start:], torch.ones_like(difference[:, :, patch_start:]))


@pytest.mark.parametrize(
    "residual",
    (
        torch.zeros(1, 6, 3),
        torch.zeros(2, 5, 3),
        torch.zeros(2, 6, 4),
        torch.zeros(2, 6, 3, dtype=torch.float64),
    ),
)
def test_aggregator_rejects_residual_shape_or_dtype_mismatch(residual: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="spatial_token_residual"):
        _tiny_aggregator()(torch.zeros(1, 2, 3, 4, 6), spatial_token_residual=residual)


def test_aggregator_rejects_residual_device_mismatch() -> None:
    with pytest.raises(ValueError, match="spatial_token_residual device"):
        _tiny_aggregator()(
            torch.zeros(1, 2, 3, 4, 6),
            spatial_token_residual=torch.zeros(2, 6, 3, device="meta"),
        )


class _ResidualSpyAggregator(nn.Module):
    patch_size = 2

    def __init__(self) -> None:
        super().__init__()
        self.observed: torch.Tensor | None = None

    def forward(
        self,
        images: torch.Tensor,
        *,
        spatial_token_residual: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], int]:
        self.observed = spatial_token_residual
        batch, frames = images.shape[:2]
        return [torch.zeros(batch, frames, 2, 4)], 1


def test_vggt_omega_forwards_spatial_residual_without_adding_parameters() -> None:
    model = VGGTOmega.__new__(VGGTOmega)
    nn.Module.__init__(model)
    model.aggregator = _ResidualSpyAggregator()
    model.camera_head = None
    model.dense_head = None
    model.text_alignment_head = None
    residual = torch.ones(2, 6, 4)
    state_keys = tuple(model.state_dict())

    model(torch.zeros(1, 2, 3, 4, 6), spatial_token_residual=residual)

    assert model.aggregator.observed is residual
    assert tuple(model.state_dict()) == state_keys
