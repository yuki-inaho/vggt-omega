from __future__ import annotations

import math

import pytest
import torch

from vggt_omega.training.checkpointing import load_resume_checkpoint, save_resume_checkpoint
from vggt_omega.training.model_factory import PreparedTrainingModel, attach_pixel_depth_model
from vggt_omega.training.optimizer_factory import build_amuse_optimizer, classify_amuse_parameters
from vggt_omega.training.pixel_depth_model import BoundedResidualGate, PixelPerfectDepthTrainingModel
from vggt_omega.training.runner import (
    _load_trainable_state,
    _select_trainable_state,
    train_one_epoch,
    validate_one_epoch,
)


class _TinyBase(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, images: torch.Tensor, *, return_patch_features: bool = False) -> dict[str, torch.Tensor]:
        batch, frames, _, height, width = images.shape
        result = {
            "pose_enc": torch.tensor([0, 0, 0, 0, 0, 0, 1, 1.2, 1.2], dtype=images.dtype, device=images.device)
            .reshape(1, 1, 9)
            .repeat(batch, frames, 1),
            "depth": self.scale.exp().expand(batch, frames, height, width, 1),
            "depth_conf": torch.ones(batch, frames, height, width, device=images.device),
        }
        if return_patch_features:
            result.update(
                {
                    "patch_features": torch.ones(batch, frames, (height // 8) * (width // 8), 12, device=images.device),
                    "patch_grid_hw": torch.tensor([height // 8, width // 8], device=images.device),
                    "patch_valid_mask": torch.ones(
                        batch, frames, (height // 8) * (width // 8), dtype=torch.bool, device=images.device
                    ),
                }
            )
        return result


def _model(
    *,
    residual_gate_initial: float = 1.0,
    correspondence_enabled: bool = False,
) -> PixelPerfectDepthTrainingModel:
    return PixelPerfectDepthTrainingModel(
        _TinyBase(),
        semantic_input_dim=12,
        hidden_dim=16,
        refiner_depth=2,
        num_heads=4,
        coarse_patch_size=8,
        fine_patch_size=4,
        temporal_depth=1,
        log_residual_scale=1.0,
        gradient_loss_weight=0.2,
        ode_steps=4,
        max_depth_m=1.2,
        geometry_enabled=False,
        residual_gate_initial=residual_gate_initial,
        correspondence_enabled=correspondence_enabled,
        correspondence_hidden_dim=16,
        correspondence_pair_chunk_size=1,
    )


def test_pixel_depth_wrapper_base_forward_preserves_legacy_keys_and_values() -> None:
    model = _model()
    images = torch.rand(1, 2, 3, 16, 24)

    expected = model.base_model(images)
    actual = model(images)

    assert actual.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key])


def test_pixel_depth_wrapper_training_returns_finite_flow_losses_and_gradients() -> None:
    torch.manual_seed(1)
    model = _model()
    images = torch.rand(1, 2, 3, 16, 24)
    target = torch.full((1, 2, 16, 24), 0.8)
    mask = torch.ones_like(target, dtype=torch.bool)
    generator = torch.Generator().manual_seed(99)

    result = model.forward_train(images, target, mask, generator=generator)

    assert result["depth"].shape == (1, 2, 16, 24, 1)
    for key in ("flow", "flow_gradient", "flow_objective"):
        assert result[key].ndim == 0 and torch.isfinite(result[key])
    result["flow_objective"].backward()
    refiner_grads = [parameter.grad for name, parameter in model.named_parameters() if name.startswith("refiner.")]
    assert any(gradient is not None for gradient in refiner_grads)


def test_pixel_depth_wrapper_refine_is_seed_deterministic_and_state_is_explicit() -> None:
    model = _model()
    images = torch.rand(1, 2, 3, 16, 24)

    first = model.forward_refine(images, generator=torch.Generator().manual_seed(4))
    second = model.forward_refine(images, generator=torch.Generator().manual_seed(4))

    torch.testing.assert_close(first["depth"], second["depth"], atol=0, rtol=0)


def test_dynamic_context_reuses_one_base_pass_with_caller_owned_noise() -> None:
    model = _model()
    images = torch.rand(1, 2, 3, 16, 24)
    noise = torch.randn(2, 1, 16, 24)
    global_state = torch.random.get_rng_state().clone()

    first = model.prepare_dynamic_context(images, initial_noise=noise)
    second = model.prepare_dynamic_context(images, initial_noise=noise)

    assert {"patch_features", "patch_grid_hw", "patch_valid_mask"} <= first.keys()
    torch.testing.assert_close(first["depth"], second["depth"], atol=0, rtol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), global_state, atol=0, rtol=0)


def test_dynamic_context_rejects_noise_shape_mismatch() -> None:
    model = _model()
    images = torch.rand(1, 2, 3, 16, 24)

    with pytest.raises(ValueError, match="initial_noise"):
        model.prepare_dynamic_context(images, initial_noise=torch.randn(1, 1, 16, 24))
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert any(name.startswith("refiner.") for name in trainable)
    assert any(name.startswith("semantic_adapter.") for name in trainable)
    assert any(name.startswith("temporal_mixer.") for name in trainable)


def test_zero_residual_gate_preserves_base_depth_exactly_and_has_gradient() -> None:
    model = _model(residual_gate_initial=0.0)
    images = torch.rand(1, 2, 3, 16, 24)
    baseline = model.base_model(images)["depth"].detach().clone()

    refined = model.forward_refine(images, generator=torch.Generator().manual_seed(4))

    torch.testing.assert_close(refined["depth"], baseline, atol=0, rtol=0)
    assert refined["residual_gate"].item() == 0.0

    target = torch.full((1, 2, 16, 24), 0.8)
    mask = torch.ones_like(target, dtype=torch.bool)
    trained = model.forward_train(
        images,
        target,
        mask,
        generator=torch.Generator().manual_seed(99),
    )
    torch.nn.functional.l1_loss(trained["depth"][..., 0], target).backward()
    assert model.residual_gate.value.grad is not None
    assert torch.isfinite(model.residual_gate.value.grad)
    assert model.residual_gate.value.grad.abs() > 0


def test_residual_gate_recovers_from_negative_raw_value_without_leaving_forward_bounds() -> None:
    gate = BoundedResidualGate(0.0)
    optimizer = torch.optim.SGD(gate.parameters(), lr=1.0)
    with torch.no_grad():
        gate.value.fill_(-0.25)

    effective = gate()
    assert effective.item() == 0.0
    (-effective).backward()
    assert gate.value.grad is not None
    assert gate.value.grad.item() == pytest.approx(-1.0)
    optimizer.step()

    assert gate().item() == pytest.approx(0.75)


def test_guarded_refiner_freezes_base_and_exposes_exact_trainable_groups() -> None:
    base = _TinyBase()
    prepared = PreparedTrainingModel(model=base, trainable_parameter_names=("scale",))
    config = {
        "semantic_input_dim": 12,
        "refiner": {
            "hidden_dim": 16,
            "depth": 2,
            "num_heads": 4,
            "coarse_patch_size": 8,
            "fine_patch_size": 4,
        },
        "temporal": {"depth": 1, "reference_mode": "first"},
        "geometry": {"enabled": False, "max_depth_m": 1.2},
        "flow": {
            "log_residual_scale": 1.0,
            "gradient_loss_weight": 0.2,
            "ode_steps": 4,
            "residual_gate_initial": 0.0,
        },
        "optimization": {"train_base_heads": False},
    }

    wrapped = attach_pixel_depth_model(prepared, config)
    names = wrapped.trainable_parameter_names

    assert not wrapped.model.base_model.scale.requires_grad
    assert any(name.startswith("semantic_adapter.") for name in names)
    assert any(name.startswith("temporal_mixer.") for name in names)
    assert any(name.startswith("refiner.") for name in names)
    assert names.count("residual_gate.value") == 1


def test_optional_correspondence_head_outputs_directed_pairs_and_obeys_stage_freeze() -> None:
    model = _model(correspondence_enabled=True)
    images = torch.rand(1, 2, 3, 16, 24)
    target = torch.full((1, 2, 16, 24), 0.8)
    mask = torch.ones_like(target, dtype=torch.bool)

    result = model.forward_train(
        images,
        target,
        mask,
        generator=torch.Generator().manual_seed(99),
    )

    assert result["correspondence_flow_pixels"].shape == (1, 2, 16, 24, 2)
    assert result["correspondence_geometric_flow_pixels"].shape == (1, 2, 16, 24, 2)
    assert result["correspondence_residual_flow_pixels"].shape == (1, 2, 16, 24, 2)
    torch.testing.assert_close(
        result["correspondence_flow_pixels"], torch.zeros_like(result["correspondence_flow_pixels"])
    )
    torch.testing.assert_close(result["correspondence_pair_indices"], torch.tensor([[[0, 1], [1, 0]]]))

    result["correspondence_flow_pixels"].sum().backward()
    assert model.base_model.scale.grad is None
    model.zero_grad(set_to_none=True)

    model.set_curriculum_trainable(train_refiner=False, train_correspondence=True)
    assert not any(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name.startswith(("semantic_adapter.", "temporal_mixer.", "refiner.", "residual_gate."))
    )
    assert all(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name.startswith("correspondence_head.")
    )

    model.set_curriculum_trainable(train_refiner=True, train_correspondence=False)
    inactive = model.forward_train(
        images,
        target,
        mask,
        generator=torch.Generator().manual_seed(99),
    )
    assert "correspondence_flow_pixels" not in inactive
    assert "correspondence_pair_indices" not in inactive

    forced_validation = model.forward_refine(
        images,
        generator=torch.Generator().manual_seed(99),
        include_correspondence=True,
    )
    assert forced_validation["correspondence_flow_pixels"].shape == (1, 2, 16, 24, 2)
    assert forced_validation["correspondence_pair_indices"].shape == (1, 2, 2)


def test_zero_weight_self_supervision_skips_auxiliary_computation() -> None:
    model = _model(correspondence_enabled=True)
    model.set_curriculum_trainable(train_refiner=True, train_correspondence=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    images = torch.rand(1, 2, 3, 16, 24)
    depths = torch.ones(1, 2, 16, 24)
    batch = {
        "images": images,
        "depths": depths,
        "depth_masks": torch.ones_like(depths, dtype=torch.bool),
        "intrinsics": torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 2, 1, 1),
        "extrinsics": torch.eye(4)[:3].reshape(1, 1, 3, 4).repeat(1, 2, 1, 1),
        "normalization_scale_m": torch.ones(1),
    }
    options = {
        "objective_weight": 1.0,
        "max_depth_m": 1.2,
        "gpa": {"enabled": True, "objective_weight": 0.0},
        "correspondence": {"enabled": True, "objective_weight": 0.0},
    }

    result = train_one_epoch(
        model=model,
        batches=[batch],
        optimizer=optimizer,
        device=torch.device("cpu"),
        gradient_clip_norm=1.0,
        gradient_accumulation_steps=1,
        min_valid_depth_pixels=1,
        global_step=0,
        logger=None,
        log_every_steps=1,
        pixel_depth_options=options,
        flow_generator=torch.Generator().manual_seed(5),
    )

    assert "gpa_objective" not in result.metrics
    assert "correspondence_objective" not in result.metrics


def test_correspondence_head_is_in_exact_checkpoint_and_amuse_groups() -> None:
    base = _TinyBase()
    prepared = PreparedTrainingModel(model=base, trainable_parameter_names=("scale",))
    config = {
        "semantic_input_dim": 12,
        "refiner": {
            "hidden_dim": 16,
            "depth": 2,
            "num_heads": 4,
            "coarse_patch_size": 8,
            "fine_patch_size": 4,
        },
        "temporal": {"depth": 1, "reference_mode": "first"},
        "geometry": {"enabled": False, "max_depth_m": 1.2},
        "flow": {
            "log_residual_scale": 1.0,
            "gradient_loss_weight": 0.2,
            "ode_steps": 4,
            "residual_gate_initial": 0.0,
        },
        "optimization": {"train_base_heads": False},
        "self_supervised": {
            "correspondence": {
                "enabled": True,
                "hidden_dim": 16,
                "pair_chunk_size": 1,
            }
        },
    }

    wrapped = attach_pixel_depth_model(prepared, config)
    grouping = classify_amuse_parameters(wrapped.model)

    assert any(name.startswith("correspondence_head.") for name in wrapped.trainable_parameter_names)
    assert grouping.fallback_names.count("correspondence_head.output_projection.weight") == 1
    assert set(grouping.muon_names) | set(grouping.fallback_names) == set(wrapped.trainable_parameter_names)

    optimizer = build_amuse_optimizer(
        wrapped.model,
        total_optimizer_steps=20,
        correspondence_output_lr=5e-4,
    ).optimizer
    assert [group["group_name"] for group in optimizer.param_groups] == [
        "muon",
        "fallback",
        "correspondence_output",
    ]
    assert optimizer.param_groups[2]["lr"] == pytest.approx(5e-4)
    assert optimizer.param_groups[2]["param_names"] == [
        "correspondence_head.output_projection.bias",
        "correspondence_head.output_projection.weight",
    ]


def test_runner_updates_pixel_depth_wrapper_and_reports_flow_metrics() -> None:
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    images = torch.rand(1, 2, 3, 16, 24)
    depths = torch.full((1, 2, 16, 24), 0.8)
    intrinsics = (
        torch.tensor([[8.0, 0.0, 12.0], [0.0, 8.0, 8.0], [0.0, 0.0, 1.0]]).reshape(1, 1, 3, 3).repeat(1, 2, 1, 1)
    )
    extrinsics = torch.eye(4)[:3].reshape(1, 1, 3, 4).repeat(1, 2, 1, 1)
    batch = {
        "images": images,
        "depths": depths,
        "depth_masks": torch.ones_like(depths, dtype=torch.bool),
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "normalization_scale_m": torch.ones(1),
    }

    result = train_one_epoch(
        model=model,
        batches=[batch],
        optimizer=optimizer,
        device=torch.device("cpu"),
        gradient_clip_norm=1.0,
        gradient_accumulation_steps=1,
        min_valid_depth_pixels=1,
        global_step=0,
        logger=None,
        log_every_steps=1,
        pixel_depth_options={"objective_weight": 1.0},
        flow_generator=torch.Generator().manual_seed(5),
    )

    assert result.global_step == 1
    assert {"flow", "flow_gradient", "flow_objective"} <= result.metrics.keys()
    assert all(torch.isfinite(torch.tensor(value)) for value in result.metrics.values())


def test_runner_combines_gpa_and_correspondence_curriculum_losses() -> None:
    model = _model(correspondence_enabled=True)
    model.set_curriculum_trainable(train_refiner=True, train_correspondence=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    images = torch.rand(1, 2, 3, 16, 24)
    depths = torch.ones(1, 2, 16, 24)
    intrinsics = (
        torch.tensor([[8.0, 0.0, 12.0], [0.0, 8.0, 8.0], [0.0, 0.0, 1.0]]).reshape(1, 1, 3, 3).repeat(1, 2, 1, 1)
    )
    extrinsics = torch.eye(4)[:3].reshape(1, 1, 3, 4).repeat(1, 2, 1, 1)
    extrinsics[:, 1, 0, 3] = 0.1
    batch = {
        "images": images,
        "depths": depths,
        "depth_masks": torch.ones_like(depths, dtype=torch.bool),
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "normalization_scale_m": torch.ones(1),
    }
    options = {
        "objective_weight": 0.5,
        "max_depth_m": 1.2,
        "curriculum_stage_index": 4,
        "gpa": {
            "enabled": True,
            "objective_weight": 0.1,
            "mu": 0.85,
            "lambda_geo": 0.1,
            "lambda_smooth": 0.001,
            "auto_mask_delta": 0.0,
            "geometry_epsilon": 1e-6,
            "auto_mask_enabled": False,
            "mask_mode": "intersection",
            "anchor_count": 2,
        },
        "correspondence": {
            "enabled": True,
            "objective_weight": 0.1,
            "alpha": 0.5,
            "epsilon": 0.01,
            "relative_depth_tolerance": 0.03,
        },
    }

    result = train_one_epoch(
        model=model,
        batches=[batch],
        optimizer=optimizer,
        device=torch.device("cpu"),
        gradient_clip_norm=1.0,
        gradient_accumulation_steps=1,
        min_valid_depth_pixels=1,
        global_step=0,
        logger=None,
        log_every_steps=1,
        pixel_depth_options=options,
        flow_generator=torch.Generator().manual_seed(5),
    )

    required = {
        "gpa_objective",
        "gpa_physical",
        "gpa_valid_fraction",
        "correspondence_objective",
        "correspondence_covisibility",
        "correspondence_pair_count",
        "residual_gate",
        "curriculum_stage_index",
    }
    assert required <= result.metrics.keys()
    assert all(math.isfinite(result.metrics[key]) for key in required)


def test_validation_reports_self_supervised_metrics_without_replacing_objective() -> None:
    model = _model(correspondence_enabled=True)
    images = torch.rand(1, 2, 3, 16, 24)
    depths = torch.ones(1, 2, 16, 24)
    intrinsics = (
        torch.tensor([[8.0, 0.0, 12.0], [0.0, 8.0, 8.0], [0.0, 0.0, 1.0]]).reshape(1, 1, 3, 3).repeat(1, 2, 1, 1)
    )
    extrinsics = torch.eye(4)[:3].reshape(1, 1, 3, 4).repeat(1, 2, 1, 1)
    batch = {
        "images": images,
        "depths": depths,
        "depth_masks": torch.ones_like(depths, dtype=torch.bool),
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "normalization_scale_m": torch.ones(1),
    }
    options = {
        "objective_weight": 0.5,
        "edge_objective_weight": 0.5,
        "multiview_objective_weight": 0.5,
        "ode_steps": 4,
        "max_depth_m": 1.2,
        "curriculum_stage_index": 2,
        "gpa": {
            "enabled": True,
            "objective_weight": 999.0,
            "mu": 0.85,
            "lambda_geo": 0.1,
            "lambda_smooth": 0.001,
            "auto_mask_delta": 0.0,
            "geometry_epsilon": 1e-6,
            "auto_mask_enabled": False,
            "mask_mode": "intersection",
            "anchor_count": 2,
        },
        "correspondence": {
            "enabled": True,
            "objective_weight": 999.0,
            "alpha": 0.5,
            "epsilon": 0.01,
            "relative_depth_tolerance": 0.03,
        },
    }

    metrics = validate_one_epoch(
        model=model,
        batches=[batch],
        device=torch.device("cpu"),
        min_valid_depth_pixels=1,
        pixel_depth_options=options,
        flow_generator=torch.Generator().manual_seed(8),
    )

    assert metrics["gpa_valid_fraction"] > 0
    assert metrics["correspondence_covisibility"] > 0
    assert metrics["curriculum_stage_index"] == 2
    assert metrics["objective"] < 999.0


def test_pixel_depth_exact_state_loader_rejects_missing_refiner_and_rng_resumes() -> None:
    model = _model()
    names = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    state = _select_trainable_state(model, names)
    missing = dict(state)
    missing.pop(next(name for name in names if name.startswith("refiner.")))

    with pytest.raises(ValueError, match="missing"):
        _load_trainable_state(model, missing, names)
    _load_trainable_state(model, state, names)

    generator = torch.Generator().manual_seed(123)
    _ = torch.randn(4, generator=generator)
    saved_rng_state = generator.get_state()
    expected_next = torch.randn(4, generator=generator)
    restored = torch.Generator()
    restored.set_state(saved_rng_state)
    torch.testing.assert_close(torch.randn(4, generator=restored), expected_next, atol=0, rtol=0)


def test_pixel_depth_checkpoint_resumes_model_optimizer_rng_and_provenance(tmp_path) -> None:
    model = _model()
    names = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=1e-3)
    generator = torch.Generator().manual_seed(77)
    _ = torch.randn(3, generator=generator)
    saved_rng = generator.get_state()
    expected_next = torch.randn(3, generator=generator)
    fingerprint = "a" * 64
    config = {"pixel_depth": {"enabled": True, "flow": {"ode_steps": 4}}}
    metadata = {"base_checkpoint": {"filename": "base.pt", "size_bytes": 1, "sha256": "b" * 64}}
    destination = tmp_path / "resume.pt"

    save_resume_checkpoint(
        destination,
        epoch=2,
        global_step=9,
        model=model,
        optimizer=optimizer,
        state_selector=lambda module: _select_trainable_state(module, names),
        group_fingerprint=fingerprint,
        config=config,
        metadata=metadata,
        training_state={"flow_rng_state": saved_rng},
    )
    restored_model = _model()
    restored_optimizer = torch.optim.AdamW(
        (parameter for parameter in restored_model.parameters() if parameter.requires_grad), lr=1e-3
    )
    resume = load_resume_checkpoint(
        destination,
        model=restored_model,
        optimizer=restored_optimizer,
        expected_group_fingerprint=fingerprint,
        state_loader=lambda module, state: _load_trainable_state(module, state, names),
    )

    assert resume.config == config
    assert resume.epoch == 2 and resume.global_step == 9
    assert resume.training_state is not None
    restored_generator = torch.Generator()
    restored_generator.set_state(resume.training_state["flow_rng_state"])
    torch.testing.assert_close(torch.randn(3, generator=restored_generator), expected_next, atol=0, rtol=0)
    expected_state = _select_trainable_state(model, names)
    actual_state = _select_trainable_state(restored_model, names)
    for name, value in expected_state.items():
        torch.testing.assert_close(actual_state[name], value)


def test_pixel_depth_validation_reports_composable_near_edge_objective() -> None:
    model = _model()
    images = torch.rand(1, 2, 3, 16, 24)
    depths = torch.full((1, 2, 16, 24), 0.8)
    depths[..., 12:] = 1.1
    batch = {
        "images": images,
        "depths": depths,
        "depth_masks": torch.ones_like(depths, dtype=torch.bool),
        "intrinsics": torch.tensor([[8.0, 0.0, 12.0], [0.0, 8.0, 8.0], [0.0, 0.0, 1.0]])
        .reshape(1, 1, 3, 3)
        .repeat(1, 2, 1, 1),
        "extrinsics": torch.eye(4)[:3].reshape(1, 1, 3, 4).repeat(1, 2, 1, 1),
        "normalization_scale_m": torch.ones(1),
    }
    options = {
        "objective_weight": 1.0,
        "edge_objective_weight": 0.5,
        "multiview_objective_weight": 0.25,
        "max_depth_m": 1.2,
        "ode_steps": 4,
    }

    metrics = validate_one_epoch(
        model=model,
        batches=[batch],
        device=torch.device("cpu"),
        min_valid_depth_pixels=1,
        pixel_depth_options=options,
        flow_generator=torch.Generator().manual_seed(8),
    )

    required = {
        "near_depth_mae_m",
        "edge_3d_error_proxy",
        "near_edge_3d_error_proxy",
        "edge_coverage",
        "near_edge_coverage",
        "multiview_depth_error",
        "multiview_relative_error",
        "multiview_coverage",
        "multiview_pair_count",
        "multiview_visible_direction_count",
        "near_edge_objective",
        "ode_steps",
    }
    assert required <= metrics.keys()
    assert metrics["near_edge_objective"] == pytest.approx(
        metrics["near_depth_mae_m"]
        + 0.5 * metrics["near_edge_3d_error_proxy"]
        + 0.25 * metrics["multiview_depth_error"]
    )
