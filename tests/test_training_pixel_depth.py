from __future__ import annotations

import pytest
import torch

from vggt_omega.training.pixel_depth import (
    PixelDepthFlowRefiner,
    SemanticPromptAdapter,
    decode_log_depth_residual,
    depth_gradient_matching_loss,
    encode_log_depth_residual,
    euler_flow_sample,
    flow_interpolate,
    l2_normalize_patch_features,
    masked_velocity_mse,
    sample_flow_noise,
)


def test_log_depth_residual_round_trip_and_invalid_mask() -> None:
    base = torch.tensor([[[[0.4, 0.8], [1.6, 3.2]]]], dtype=torch.float64)
    target = torch.tensor([[[[0.5, 1.0], [2.0, 0.0]]]], dtype=torch.float64)
    mask = target > 0

    residual = encode_log_depth_residual(target, base, mask, log_residual_scale=1.0)
    decoded = decode_log_depth_residual(base, residual, log_residual_scale=1.0)

    torch.testing.assert_close(decoded[mask], target[mask], atol=1e-12, rtol=1e-12)
    assert residual[~mask].eq(0).all()


def test_log_depth_residual_is_scale_equivariant() -> None:
    base = torch.tensor([[[[0.4, 0.8, 1.6]]]], dtype=torch.float64)
    target = torch.tensor([[[[0.5, 1.0, 2.0]]]], dtype=torch.float64)
    mask = torch.ones_like(base, dtype=torch.bool)

    residual = encode_log_depth_residual(target, base, mask, log_residual_scale=1.0)
    scaled_residual = encode_log_depth_residual(7 * target, 7 * base, mask, log_residual_scale=1.0)
    scaled_decoded = decode_log_depth_residual(7 * base, scaled_residual, log_residual_scale=1.0)

    torch.testing.assert_close(residual, scaled_residual, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(scaled_decoded, 7 * target, atol=1e-12, rtol=1e-12)


def test_log_depth_residual_constant_and_empty_masks_are_finite() -> None:
    base = torch.ones((1, 2, 2, 2), requires_grad=True)
    target = torch.ones_like(base)
    valid = torch.ones_like(base, dtype=torch.bool)
    empty = torch.zeros_like(base, dtype=torch.bool)

    zero_residual = encode_log_depth_residual(target, base, valid, log_residual_scale=1.0)
    empty_residual = encode_log_depth_residual(target, base, empty, log_residual_scale=1.0)

    assert zero_residual.eq(0).all()
    assert empty_residual.eq(0).all()
    assert torch.isfinite(zero_residual).all()
    assert torch.isfinite(empty_residual).all()


@pytest.mark.parametrize("field", ["target", "base"])
def test_log_depth_residual_rejects_nonfinite_inputs(field: str) -> None:
    base = torch.ones((1, 1, 2, 2))
    target = torch.ones_like(base)
    mask = torch.ones_like(base, dtype=torch.bool)
    if field == "target":
        target[..., 0, 0] = torch.nan
    else:
        base[..., 0, 0] = torch.inf

    with pytest.raises(ValueError, match=field):
        encode_log_depth_residual(target, base, mask, log_residual_scale=1.0)


def test_log_depth_residual_rejects_nonpositive_base() -> None:
    base = torch.ones((1, 1, 2, 2))
    target = torch.ones_like(base)
    mask = torch.ones_like(base, dtype=torch.bool)
    base[..., 0, 0] = 0

    with pytest.raises(ValueError, match="base"):
        encode_log_depth_residual(target, base, mask, log_residual_scale=1.0)


def test_l2_normalize_patch_features_normalizes_valid_and_zeros_invalid() -> None:
    features = torch.tensor([[[[3.0, 4.0], [0.0, 0.0], [1.0, 2.0]]]])
    valid = torch.tensor([[[True, True, False]]])

    normalized = l2_normalize_patch_features(features, valid)

    torch.testing.assert_close(normalized[..., 0, :].norm(dim=-1), torch.ones(1, 1))
    assert torch.isfinite(normalized).all()
    assert normalized[..., 1:, :].eq(0).all()


def test_semantic_prompt_adapter_resizes_rectangular_grid_and_backpropagates() -> None:
    torch.manual_seed(0)
    adapter = SemanticPromptAdapter(input_dim=4, prompt_dim=6, hidden_dim=8)
    features = torch.randn(2, 3, 6, 4, requires_grad=True)
    valid = torch.ones(2, 3, 6, dtype=torch.bool)
    valid[0, 0, 0] = False

    prompt, prompt_mask = adapter(
        features,
        valid,
        source_grid_hw=(2, 3),
        target_grid_hw=(4, 6),
    )

    assert prompt.shape == (6, 24, 6)
    assert prompt_mask.shape == (6, 24)
    assert prompt_mask.any()
    assert (~prompt_mask).any()
    assert prompt[~prompt_mask].eq(0).all()
    assert torch.isfinite(prompt).all()
    prompt.square().mean().backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert all(parameter.grad is not None for parameter in adapter.parameters())


def test_semantic_prompt_adapter_rejects_grid_token_mismatch() -> None:
    adapter = SemanticPromptAdapter(input_dim=4, prompt_dim=6, hidden_dim=8)
    features = torch.ones(1, 2, 5, 4)
    valid = torch.ones(1, 2, 5, dtype=torch.bool)

    with pytest.raises(ValueError, match="source grid"):
        adapter(features, valid, source_grid_hw=(2, 3), target_grid_hw=(2, 3))


def test_semantic_prompt_adapter_rejects_nonfinite_features() -> None:
    adapter = SemanticPromptAdapter(input_dim=4, prompt_dim=6, hidden_dim=8)
    features = torch.ones(1, 1, 6, 4)
    features[..., 0, 0] = torch.nan
    valid = torch.ones(1, 1, 6, dtype=torch.bool)

    with pytest.raises(ValueError, match="non-finite"):
        adapter(features, valid, source_grid_hw=(2, 3), target_grid_hw=(2, 3))


def _tiny_refiner() -> PixelDepthFlowRefiner:
    return PixelDepthFlowRefiner(
        hidden_dim=32,
        depth=4,
        num_heads=4,
        coarse_patch_size=8,
        fine_patch_size=4,
        in_channels=4,
    )


@pytest.mark.parametrize("shape", [(2, 4, 32, 32), (3, 4, 32, 48)])
def test_pixel_depth_flow_refiner_preserves_square_and_rectangular_shapes(shape: tuple[int, ...]) -> None:
    torch.manual_seed(0)
    refiner = _tiny_refiner()
    inputs = torch.randn(shape)
    coarse_tokens = (shape[-2] // 8) * (shape[-1] // 8)
    semantics = torch.randn(shape[0], coarse_tokens, 32)
    semantic_mask = torch.ones(shape[0], coarse_tokens, dtype=torch.bool)
    timestep = torch.linspace(0, 1, shape[0])

    velocity = refiner(inputs, semantics, semantic_mask, timestep)

    assert velocity.shape == (shape[0], 1, shape[-2], shape[-1])
    assert torch.isfinite(velocity).all()


def test_pixel_depth_flow_refiner_pads_and_crops_odd_input() -> None:
    refiner = _tiny_refiner()
    inputs = torch.randn(2, 4, 31, 45)
    semantics = torch.randn(2, 24, 32)
    semantic_mask = torch.ones(2, 24, dtype=torch.bool)

    velocity = refiner(inputs, semantics, semantic_mask, torch.tensor([0.25, 0.75]))

    assert velocity.shape == (2, 1, 31, 45)
    assert torch.isfinite(velocity).all()


def test_pixel_depth_flow_refiner_is_deterministic_and_backpropagates() -> None:
    torch.manual_seed(123)
    refiner = _tiny_refiner()
    inputs = torch.randn(2, 4, 16, 24, requires_grad=True)
    semantics = torch.randn(2, 6, 32, requires_grad=True)
    semantic_mask = torch.ones(2, 6, dtype=torch.bool)
    timestep = torch.tensor([0.2, 0.8])

    first = refiner(inputs, semantics, semantic_mask, timestep)
    second = refiner(inputs, semantics, semantic_mask, timestep)
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    first.square().mean().backward()

    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert semantics.grad is not None and torch.isfinite(semantics.grad).all()
    parameter_grads = [parameter.grad for parameter in refiner.parameters() if parameter.grad is not None]
    assert parameter_grads
    assert all(torch.isfinite(grad).all() for grad in parameter_grads)


def test_flow_interpolate_has_paper_endpoint_and_velocity_contract() -> None:
    clean = torch.tensor([[[[1.0, -0.5]]]])
    noise = torch.tensor([[[[-1.0, 0.5]]]])

    at_clean, velocity = flow_interpolate(clean, noise, torch.tensor([0.0]))
    at_noise, _ = flow_interpolate(clean, noise, torch.tensor([1.0]))
    halfway, _ = flow_interpolate(clean, noise, torch.tensor([0.5]))

    torch.testing.assert_close(at_clean, clean)
    torch.testing.assert_close(at_noise, noise)
    torch.testing.assert_close(halfway, (clean + noise) / 2)
    torch.testing.assert_close(velocity, noise - clean)


def test_sample_flow_noise_uses_explicit_reproducible_generator() -> None:
    reference = torch.zeros(2, 1, 3, 4)
    first_generator = torch.Generator().manual_seed(8675309)
    second_generator = torch.Generator().manual_seed(8675309)

    first = sample_flow_noise(reference, generator=first_generator)
    second = sample_flow_noise(reference, generator=second_generator)

    torch.testing.assert_close(first, second, atol=0, rtol=0)
    assert first.shape == reference.shape
    assert first.dtype == reference.dtype


@pytest.mark.parametrize("steps", [1, 2, 8])
def test_euler_flow_sample_moves_from_noise_to_clean_for_constant_velocity(steps: int) -> None:
    clean = torch.tensor([[[[0.25, -0.75]]]])
    noise = torch.tensor([[[[1.0, 0.5]]]])
    constant_velocity = noise - clean

    sampled = euler_flow_sample(
        lambda state, timestep: constant_velocity.expand_as(state),
        noise,
        steps=steps,
    )

    torch.testing.assert_close(sampled, clean, atol=1e-6, rtol=1e-6)


def test_masked_velocity_mse_reduces_only_valid_pixels_and_empty_is_graph_connected() -> None:
    prediction = torch.tensor([[[[1.0, 4.0], [2.0, 9.0]]]], requires_grad=True)
    target = torch.tensor([[[[0.0, 0.0], [4.0, 0.0]]]])
    mask = torch.tensor([[[[True, False], [True, False]]]])

    loss = masked_velocity_mse(prediction, target, mask)
    torch.testing.assert_close(loss, torch.tensor(2.5))
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([[[[1.0, 0.0], [-2.0, 0.0]]]]))

    empty_prediction = prediction.detach().clone().requires_grad_(True)
    empty = masked_velocity_mse(empty_prediction, target, torch.zeros_like(mask))
    assert empty.item() == 0
    empty.backward()
    assert empty_prediction.grad is not None
    assert empty_prediction.grad.eq(0).all()


def test_depth_gradient_matching_loss_detects_edge_error_and_respects_mask() -> None:
    target = torch.zeros(1, 1, 4, 5)
    target[..., :, 3:] = 1
    exact = target.clone().requires_grad_(True)
    shifted = torch.zeros_like(target)
    shifted[..., :, 2:] = 1
    mask = torch.ones_like(target, dtype=torch.bool)

    exact_loss = depth_gradient_matching_loss(exact, target, mask)
    shifted_loss = depth_gradient_matching_loss(shifted, target, mask)
    assert exact_loss.item() == 0
    assert shifted_loss.item() > 0
    exact_loss.backward()
    assert exact.grad is not None and torch.isfinite(exact.grad).all()

    isolated_mask = torch.zeros_like(mask)
    isolated_mask[..., 0, 0] = True
    assert depth_gradient_matching_loss(shifted, target, isolated_mask).item() == 0
