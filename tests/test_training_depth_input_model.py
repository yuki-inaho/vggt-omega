from __future__ import annotations

import pytest
import torch
from torch import nn

from vggt_omega.training.depth_input_model import (
    DepthInputTrainingModel,
    MappedDepthTokenAdapter,
    fixed_depth_availability,
    sample_depth_availability,
)


def test_adapter_maps_640x480_depth_and_mask_to_vggt_patch_tokens() -> None:
    adapter = MappedDepthTokenAdapter(patch_size=16, embed_dim=1024)

    residual = adapter(
        torch.ones(1, 1, 1, 480, 640),
        torch.ones(1, 1, 1, 480, 640, dtype=torch.bool),
        torch.ones(1, 1, dtype=torch.bool),
    )

    assert residual.shape == (1, 1200, 1024)


def test_adapter_normalizes_valid_depth_by_sequence_mean_and_uses_mask_channel() -> None:
    adapter = MappedDepthTokenAdapter(patch_size=2, embed_dim=2)
    assert adapter.depth_patch_embed.bias is not None
    with torch.no_grad():
        adapter.depth_patch_embed.weight.zero_()
        adapter.depth_patch_embed.bias.zero_()
        adapter.depth_patch_embed.weight[0, 0].fill_(1)
        adapter.depth_patch_embed.weight[1, 1].fill_(1)
    depth = torch.stack((torch.ones(2, 2), torch.full((2, 2), 3.0))).reshape(1, 2, 1, 2, 2)
    mask = torch.ones_like(depth, dtype=torch.bool)

    residual = adapter(depth, mask, torch.ones(1, 2, dtype=torch.bool))

    torch.testing.assert_close(residual[:, :, 0], torch.tensor([[2.0], [6.0]]))
    torch.testing.assert_close(residual[:, :, 1], torch.tensor([[4.0], [4.0]]))


def test_adapter_uses_learned_placeholder_without_reading_unavailable_depth() -> None:
    adapter = MappedDepthTokenAdapter(patch_size=2, embed_dim=3)
    with torch.no_grad():
        adapter.depth_placeholder.fill_(7)
    depth = torch.ones(1, 2, 1, 4, 4)
    depth[:, 1].fill_(torch.nan)
    mask = torch.ones_like(depth, dtype=torch.bool)
    availability = torch.tensor([[True, False]])

    residual = adapter(depth, mask, availability)

    assert residual.shape == (2, 4, 3)
    torch.testing.assert_close(residual[1], torch.full((4, 3), 7.0))


def test_adapter_rejects_available_frame_without_valid_depth() -> None:
    adapter = MappedDepthTokenAdapter(patch_size=2, embed_dim=3)

    with pytest.raises(ValueError, match="available frame"):
        adapter(
            torch.ones(1, 1, 1, 4, 4),
            torch.zeros(1, 1, 1, 4, 4, dtype=torch.bool),
            torch.ones(1, 1, dtype=torch.bool),
        )


@pytest.mark.parametrize(
    ("depth", "mask", "availability"),
    (
        (torch.ones(1, 1, 4, 4), torch.ones(1, 1, 1, 4, 4, dtype=torch.bool), torch.ones(1, 1, dtype=torch.bool)),
        (torch.ones(1, 1, 1, 4, 4), torch.ones(1, 1, 4, 4, dtype=torch.bool), torch.ones(1, 1, dtype=torch.bool)),
        (torch.ones(1, 1, 1, 5, 4), torch.ones(1, 1, 1, 5, 4, dtype=torch.bool), torch.ones(1, 1, dtype=torch.bool)),
        (torch.ones(1, 1, 1, 4, 4), torch.ones(1, 1, 1, 4, 4), torch.ones(1, 1, dtype=torch.bool)),
        (torch.ones(1, 1, 1, 4, 4), torch.ones(1, 1, 1, 4, 4, dtype=torch.bool), torch.ones(1, 1)),
    ),
)
def test_adapter_rejects_shape_or_dtype_mismatch(
    depth: torch.Tensor,
    mask: torch.Tensor,
    availability: torch.Tensor,
) -> None:
    with pytest.raises(ValueError):
        MappedDepthTokenAdapter(patch_size=2, embed_dim=3)(depth, mask, availability)


def test_adapter_rejects_parameter_dtype_mismatch_instead_of_casting() -> None:
    with pytest.raises(ValueError, match="same dtype"):
        MappedDepthTokenAdapter(patch_size=2, embed_dim=3)(
            torch.ones(1, 1, 1, 4, 4, dtype=torch.float64),
            torch.ones(1, 1, 1, 4, 4, dtype=torch.bool),
            torch.ones(1, 1, dtype=torch.bool),
        )


def test_adapter_rejects_device_mismatch_instead_of_moving_inputs() -> None:
    with pytest.raises(ValueError, match="same device"):
        MappedDepthTokenAdapter(patch_size=2, embed_dim=3)(
            torch.ones(1, 1, 1, 4, 4),
            torch.ones(1, 1, 1, 4, 4, dtype=torch.bool, device="meta"),
            torch.ones(1, 1, dtype=torch.bool),
        )


def test_adapter_gradients_reach_encoder_and_placeholder() -> None:
    adapter = MappedDepthTokenAdapter(patch_size=2, embed_dim=3)
    depth = torch.ones(1, 2, 1, 4, 4)
    mask = torch.ones_like(depth, dtype=torch.bool)

    adapter(depth, mask, torch.tensor([[True, False]])).sum().backward()

    gradients = [parameter.grad for parameter in adapter.parameters()]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)


class _ConditionedBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.aggregator = nn.Identity()

    def forward(
        self,
        images: torch.Tensor,
        *,
        spatial_token_residual: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {"residual": spatial_token_residual, "images": images}


def test_depth_input_wrapper_state_strictly_round_trips() -> None:
    model = DepthInputTrainingModel(_ConditionedBase(), MappedDepthTokenAdapter(patch_size=2, embed_dim=3))
    state = model.state_dict()
    restored = DepthInputTrainingModel(_ConditionedBase(), MappedDepthTokenAdapter(patch_size=2, embed_dim=3))

    incompatible = restored.load_state_dict(state, strict=True)
    output = restored(
        torch.zeros(1, 2, 3, 4, 4),
        mapped_depth=torch.ones(1, 2, 1, 4, 4),
        valid_mask=torch.ones(1, 2, 1, 4, 4, dtype=torch.bool),
        availability=torch.tensor([[True, False]]),
    )

    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys
    assert output["residual"].shape == (2, 4, 3)


def test_depth_availability_is_seeded_and_fixed_k_covers_zero_through_four() -> None:
    first = sample_depth_availability(8, 4, seed=42, epoch=3, optimizer_step=7, device="cpu")
    second = sample_depth_availability(8, 4, seed=42, epoch=3, optimizer_step=7, device="cpu")
    changed = sample_depth_availability(8, 4, seed=42, epoch=3, optimizer_step=8, device="cpu")

    assert torch.equal(first, second)
    assert not torch.equal(first, changed)
    assert ((first.sum(dim=1) >= 0) & (first.sum(dim=1) <= 4)).all()
    for provided_frames in range(5):
        fixed = fixed_depth_availability(3, 4, provided_frames, device="cpu")
        assert fixed.shape == (3, 4)
        assert torch.equal(fixed.sum(dim=1), torch.full((3,), provided_frames))
