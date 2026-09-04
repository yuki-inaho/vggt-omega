from __future__ import annotations

import torch
from torch import nn

from vggt_omega.models import VGGTOmega


class _TinyAggregator(nn.Module):
    def __init__(self, patch_size: int = 16, embed_dim: int = 4) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, images: torch.Tensor) -> tuple[list[torch.Tensor], int]:
        batch, frames, _, height, width = images.shape
        patch_count = (height // self.patch_size) * (width // self.patch_size)
        token_count = 2 + patch_count
        token_values = torch.arange(token_count, dtype=images.dtype).reshape(1, 1, token_count, 1)
        tokens = self.scale * token_values.expand(batch, frames, token_count, self.embed_dim)
        return [tokens], 2


class _TinyCameraHead(nn.Module):
    def forward(self, tokens: list[torch.Tensor], *, patch_token_start: int) -> torch.Tensor:
        batch, frames = tokens[-1].shape[:2]
        return tokens[-1][..., 0, :1].expand(batch, frames, 9)


class _TinyDenseHead(nn.Module):
    def forward(
        self,
        tokens: list[torch.Tensor],
        *,
        images: torch.Tensor,
        patch_token_start: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, frames, _, height, width = images.shape
        value = tokens[-1][..., 0, :1].reshape(batch, frames, 1, 1, 1)
        depth = value.expand(batch, frames, height, width, 1).float()
        confidence = value[..., 0].expand(batch, frames, height, width).float()
        return depth, confidence


def _tiny_model() -> VGGTOmega:
    model = VGGTOmega.__new__(VGGTOmega)
    nn.Module.__init__(model)
    model.aggregator = _TinyAggregator()
    model.camera_head = _TinyCameraHead()
    model.dense_head = _TinyDenseHead()
    model.text_alignment_head = None
    return model


def test_forward_patch_features_are_opt_in_and_legacy_outputs_match() -> None:
    model = _tiny_model().train()
    images = torch.zeros(1, 2, 3, 32, 48)

    legacy = model(images)
    featured = model(images, return_patch_features=True)

    assert "patch_features" not in legacy
    assert "patch_grid_hw" not in legacy
    assert "patch_valid_mask" not in legacy
    assert set(featured) == {*legacy, "patch_features", "patch_grid_hw", "patch_valid_mask"}
    for key, value in legacy.items():
        torch.testing.assert_close(featured[key], value)
    assert featured["patch_features"].shape == (1, 2, 6, 4)
    torch.testing.assert_close(featured["patch_features"][0, 0, :, 0], torch.arange(2, 8, dtype=images.dtype))
    torch.testing.assert_close(featured["patch_grid_hw"], torch.tensor([2, 3]))
    assert featured["patch_valid_mask"].shape == (1, 2, 6)
    assert featured["patch_valid_mask"].all()


def test_forward_patch_features_retain_the_feature_graph() -> None:
    model = _tiny_model().train()
    featured = model(torch.zeros(1, 3, 3, 32, 48), return_patch_features=True)

    featured["patch_features"].sum().backward()

    assert model.aggregator.scale.grad is not None
    assert torch.isfinite(model.aggregator.scale.grad)
