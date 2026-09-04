from __future__ import annotations

import pytest
import torch
from torch import nn

from vggt_omega.training.model_factory import PreparedTrainingModel, attach_dynamic_geometry_model
from vggt_omega.training.optimizer_factory import build_adamw_optimizer, classify_amuse_parameters
from vggt_omega.training.runner import (
    _apply_dynamic_curriculum_stage,
    _dynamic_geometry_losses,
    _dynamic_geometry_runtime_options,
    _dynamic_guardrail_options,
    _guardrail_violations,
    _load_trainable_state,
    _select_trainable_state,
    train_one_epoch,
)


class _TinyDynamicContext(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_feature_dim = 8
        self.scale = nn.Parameter(torch.tensor(0.0))
        self.call_shapes: list[tuple[int, ...]] = []

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, frames, _, height, width = images.shape
        return {
            "pose_enc": self._pose(batch, frames, images),
            "depth": self.scale.exp().expand(batch, frames, height, width, 1),
        }

    def prepare_dynamic_context(
        self,
        images: torch.Tensor,
        *,
        initial_noise: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        self.call_shapes.append(tuple(images.shape))
        batch, frames, _, height, width = images.shape
        assert initial_noise.shape == (batch * frames, 1, height, width)
        grid = (2, 3)
        image_bias = images.mean(dim=(2, 3, 4), keepdim=True)
        depth = self.scale.exp() + image_bias
        return {
            "pose_enc": self._pose(batch, frames, images),
            "depth": depth.expand(batch, frames, height, width, 1),
            "patch_features": torch.ones(batch, frames, grid[0] * grid[1], 8, device=images.device),
            "patch_grid_hw": torch.tensor(grid, device=images.device),
            "patch_valid_mask": torch.ones(batch, frames, grid[0] * grid[1], dtype=torch.bool),
        }

    @staticmethod
    def _pose(batch: int, frames: int, images: torch.Tensor) -> torch.Tensor:
        pose = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.2, 1.2],
            dtype=images.dtype,
            device=images.device,
        )
        return pose.reshape(1, 1, 9).expand(batch, frames, 9).clone()


class _TinyDynamicContextWithConfidence(_TinyDynamicContext):
    def __init__(self) -> None:
        super().__init__()
        self.base_model = nn.Module()
        self.base_model.camera_head = nn.Linear(2, 2)
        self.base_model.dense_head = nn.Module()
        self.base_model.dense_head.proj = nn.Linear(2, 2)
        self.base_model.dense_head.proj_conf = nn.Linear(2, 1)


def _config(*, enabled: bool) -> dict[str, object]:
    return {
        "enabled": enabled,
        "contract_version": 1,
        "hidden_dim": 16,
        "pair_chunk_size": 2,
        "relative_camera_dim": 14,
        "visibility_threshold": 0.5,
        "static_probability_max": 0.25,
        "dynamic_probability_min": 0.75,
        "depth_source": "pixel_refined_fixed_noise",
        "refinement_seed": 7301,
        "joint_base_parameter_prefixes": [],
    }


def _prepared() -> PreparedTrainingModel:
    model = _TinyDynamicContext()
    return PreparedTrainingModel(model=model, trainable_parameter_names=("scale",))


def test_disabled_dynamic_attach_is_exact_noop() -> None:
    prepared = _prepared()
    before_state = tuple(prepared.model.state_dict())
    images = torch.rand(1, 2, 3, 5, 7)
    expected = prepared.model(images)

    actual = attach_dynamic_geometry_model(prepared, _config(enabled=False))

    assert actual is prepared
    assert tuple(actual.model.state_dict()) == before_state
    assert actual.trainable_parameter_names == ("scale",)
    observed = actual.model(images)
    assert observed.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(observed[key], expected[key], atol=0, rtol=0)


def test_enabled_dynamic_wrapper_preserves_legacy_forward_and_is_fail_closed() -> None:
    prepared = _prepared()
    images = torch.rand(1, 3, 3, 5, 7)
    expected = prepared.model(images)
    wrapped = attach_dynamic_geometry_model(prepared, _config(enabled=True))

    legacy = wrapped.model(images)
    assert legacy.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(legacy[key], expected[key], atol=0, rtol=0)

    dynamic = wrapped.model.forward_dynamic(
        images,
        frame_ids=torch.tensor([[2, 0, 1]]),
        frame_mask=torch.ones(1, 3, dtype=torch.bool),
    )
    assert dynamic["canonical_scene_flow"].shape == (1, 4, 5, 7, 3)
    assert dynamic["dynamic_geometry_ready"].item() is False
    assert not dynamic["dynamic_mask"].any()
    torch.testing.assert_close(dynamic["dynamic_unknown_mask"], dynamic["motion_domain_mask"])
    assert any(name.startswith("dynamic_geometry_head.") for name in wrapped.trainable_parameter_names)


def test_dynamic_wrapper_compacts_padding_and_is_seed_deterministic() -> None:
    wrapped = attach_dynamic_geometry_model(_prepared(), _config(enabled=True)).model
    images = torch.rand(2, 3, 3, 5, 7)
    images[1, 2] = 1000
    frame_ids = torch.tensor([[0, 1, 2], [5, 6, -1]])
    frame_mask = torch.tensor([[True, True, True], [True, True, False]])

    first = wrapped.forward_dynamic(images, frame_ids=frame_ids, frame_mask=frame_mask)
    second = wrapped.forward_dynamic(images, frame_ids=frame_ids, frame_mask=frame_mask)
    standalone = wrapped.forward_dynamic(
        images[1:2, :2],
        frame_ids=frame_ids[1:2, :2],
        frame_mask=torch.ones(1, 2, dtype=torch.bool),
    )

    torch.testing.assert_close(first["depth"], second["depth"], atol=0, rtol=0)
    torch.testing.assert_close(first["depth"][1, :2], standalone["depth"][0], atol=0, rtol=0)
    assert not first["canonical_points_valid_mask"][1, 2].any()
    assert (1, 2, 3, 5, 7) in wrapped.wrapped_model.call_shapes
    assert (1, 3, 3, 5, 7) in wrapped.wrapped_model.call_shapes


def test_dynamic_stage_trainable_allowlists_are_exact() -> None:
    wrapped = attach_dynamic_geometry_model(_prepared(), _config(enabled=True)).model

    wrapped.set_dynamic_stage("motion_only")
    motion_names = {name for name, parameter in wrapped.named_parameters() if parameter.requires_grad}
    assert motion_names
    assert all(
        name.startswith(("dynamic_geometry_head.pair_encoder.", "dynamic_geometry_head.flow_decoder."))
        for name in motion_names
    )

    wrapped.set_dynamic_stage("visibility_dynamic")
    classification_names = {name for name, parameter in wrapped.named_parameters() if parameter.requires_grad}
    assert classification_names
    assert all(
        name.startswith(
            (
                "dynamic_geometry_head.visibility_decoder.",
                "dynamic_geometry_head.dynamic_decoder.",
            )
        )
        for name in classification_names
    )

    with pytest.raises(ValueError, match="joint_base_parameter_prefixes"):
        wrapped.set_dynamic_stage("joint")


def test_dynamic_joint_candidates_keep_confidence_projection_frozen() -> None:
    prepared = PreparedTrainingModel(
        model=_TinyDynamicContextWithConfidence(),
        trainable_parameter_names=("scale",),
    )
    config = _config(enabled=True)
    config["joint_base_parameter_prefixes"] = [
        "wrapped_model.base_model.camera_head.",
        "wrapped_model.base_model.dense_head.",
    ]

    wrapped = attach_dynamic_geometry_model(prepared, config).model
    confidence_names = {
        name
        for name, parameter in wrapped.named_parameters()
        if ".dense_head.proj_conf." in name and parameter.requires_grad
    }
    assert confidence_names == set()

    wrapped.set_dynamic_stage("joint")
    trainable = {name for name, parameter in wrapped.named_parameters() if parameter.requires_grad}
    assert "wrapped_model.base_model.dense_head.proj.weight" in trainable
    assert not any(".dense_head.proj_conf." in name for name in trainable)
    grouping = classify_amuse_parameters(wrapped)
    grouped = set(grouping.muon_names) | set(grouping.fallback_names)
    assert grouped == trainable
    assert all(".dense_head.proj_conf." not in name for name in grouped)


def test_dynamic_optimizer_and_checkpoint_include_every_candidate_and_readiness() -> None:
    prepared = attach_dynamic_geometry_model(_prepared(), _config(enabled=True))
    assert prepared.trainable_parameter_names[-1] == "dynamic_geometry_ready"
    optimizer = build_adamw_optimizer(prepared.model, lr=1e-4)
    grouped = {name for group in optimizer.optimizer.param_groups for name in group["param_names"]}
    trainable = {name for name, parameter in prepared.model.named_parameters() if parameter.requires_grad}
    assert grouped == trainable

    prepared.model.set_dynamic_geometry_ready(True)
    state = _select_trainable_state(prepared.model, prepared.trainable_parameter_names)
    prepared.model.set_dynamic_geometry_ready(False)
    _load_trainable_state(prepared.model, state, prepared.trainable_parameter_names)
    assert prepared.model.dynamic_geometry_ready.item() is True
    restored = attach_dynamic_geometry_model(_prepared(), _config(enabled=True))
    _load_trainable_state(restored.model, state, restored.trainable_parameter_names)
    images = torch.rand(1, 2, 3, 5, 7)
    frame_ids = torch.tensor([[0, 1]])
    frame_mask = torch.ones((1, 2), dtype=torch.bool)
    expected = prepared.model.forward_dynamic(images, frame_ids=frame_ids, frame_mask=frame_mask)
    actual = restored.model.forward_dynamic(images, frame_ids=frame_ids, frame_mask=frame_mask)
    assert expected.keys() == actual.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)


def test_dynamic_runtime_stage_and_static_rgbd_objective_have_finite_backward() -> None:
    prepared = attach_dynamic_geometry_model(_prepared(), _config(enabled=True))
    model = prepared.model
    optimizer = build_adamw_optimizer(model, lr=1e-4).optimizer
    config = {
        **_config(enabled=True),
        "geometry": {"max_depth_m": 1.2},
        "pseudo_labels": {
            "static_off_m": 0.01,
            "dynamic_on_m": 0.03,
            "flow_confidence_min": 0.8,
            "forward_backward_cycle_px": 1.0,
            "depth_discontinuity_relative": 0.03,
        },
        "loss": {
            "scene_flow_weight": 1.0,
            "cycle_weight": 0.1,
            "reprojection_weight": 0.1,
            "temporal_depth_weight": 0.1,
            "visibility_weight": 0.2,
            "dynamic_weight": 0.2,
            "spatial_weight": 0.01,
            "temporal_mask_weight": 0.05,
            "area_prior_weight": 0.001,
            "charbonnier_alpha": 0.5,
            "charbonnier_epsilon": 0.001,
            "area_lower": 0.01,
            "area_upper": 0.5,
        },
        "readiness": {
            "min_visibility_positive_count": 1,
            "min_visibility_negative_count": 1,
            "min_visibility_known_coverage": 0.01,
            "min_visibility_precision": 0.8,
            "min_dynamic_static_count": 1,
            "min_dynamic_positive_count": 1,
            "min_dynamic_known_coverage": 0.01,
            "min_dynamic_precision": 0.8,
            "min_dynamic_recall": 0.5,
        },
        "curriculum": [
            {
                "name": "baseline_parity",
                "start_epoch": 0,
                "train_enabled": False,
                "dynamic_stage": "disabled",
                "learning_rate_scale": 0.0,
            },
            {
                "name": "visibility_dynamic",
                "start_epoch": 1,
                "train_enabled": True,
                "dynamic_stage": "visibility_dynamic",
                "learning_rate_scale": 0.5,
            },
        ],
    }
    baseline = _dynamic_geometry_runtime_options(config, epoch=0)
    active = _dynamic_geometry_runtime_options(config, epoch=1)
    assert baseline is not None and active is not None
    base_lrs = tuple(float(group["lr"]) for group in optimizer.param_groups)
    assert not _apply_dynamic_curriculum_stage(model, optimizer, baseline, base_learning_rates=base_lrs)
    assert _apply_dynamic_curriculum_stage(model, optimizer, active, base_learning_rates=base_lrs)
    assert all(group["lr"] == pytest.approx(5e-5) for group in optimizer.param_groups)

    images = torch.zeros((1, 3, 3, 5, 7))
    frame_ids = torch.tensor([[0, 1, 2]])
    frame_mask = torch.ones((1, 3), dtype=torch.bool)
    predictions = model.forward_dynamic(images, frame_ids=frame_ids, frame_mask=frame_mask)
    pair_count = predictions["motion_pair_indices"].shape[1]
    batch = {
        "images": images,
        "depths": predictions["depth"][..., 0].detach(),
        "depth_masks": torch.ones((1, 3, 5, 7), dtype=torch.bool),
        "original_depth_observed_mask": torch.ones((1, 3, 5, 7), dtype=torch.bool),
        "intrinsics": predictions["predicted_intrinsics"].detach(),
        "extrinsics": predictions["rebased_extrinsics_w2c"].detach(),
        "normalization_scale_m": torch.ones(1),
        "frame_ids": frame_ids,
        "frame_mask": frame_mask,
        "motion_pixel_flow_xy": torch.zeros((1, pair_count, 5, 7, 2)),
        "motion_flow_confidence": torch.ones((1, pair_count, 5, 7)),
    }
    losses = _dynamic_geometry_losses(predictions, batch, active)
    assert torch.isfinite(losses["dynamic_objective"])
    assert losses["dynamic_teacher_coverage"].item() > 0
    assert losses["dynamic_scene_flow_epe"].item() >= 0
    for metric in (
        "dynamic_f1",
        "dynamic_iou",
        "dynamic_static_false_positive_rate",
        "dynamic_visibility_f1",
        "dynamic_visibility_iou",
        "dynamic_visibility_recall",
    ):
        assert 0 <= losses[metric].item() <= 1
    losses["dynamic_objective"].backward()
    gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if name.startswith(("dynamic_geometry_head.visibility_decoder.", "dynamic_geometry_head.dynamic_decoder."))
    ]
    assert gradients and all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    optimizer.zero_grad(set_to_none=True)
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
        dynamic_geometry_options=active,
    )
    assert result.optimizer_steps == 1
    assert result.global_step == 1
    assert result.metrics["dynamic_teacher_coverage"] > 0
    assert torch.isfinite(torch.tensor(result.metrics["dynamic_objective"]))
    assert model.dynamic_geometry_ready.item() is False


def test_dynamic_guardrail_maps_near_camera_and_objective_without_hiding_degradation() -> None:
    options = _dynamic_guardrail_options(
        {
            "guardrail": {
                "max_near_depth_mae_m_degradation": 0.01,
                "max_camera_translation_degradation": 0.02,
                "max_objective_degradation": 0.03,
            }
        }
    )
    assert options is not None
    baseline = {"depth_lt_1p2m_mae_m": 0.1, "camera_translation": 0.1, "objective": 1.0}
    current = {"depth_lt_1p2m_mae_m": 0.111, "camera_translation": 0.119, "objective": 1.029}

    violations = _guardrail_violations(baseline, current, options)

    assert set(violations) == {"depth_lt_1p2m_mae_m"}
    assert violations["depth_lt_1p2m_mae_m"]["allowed"] == pytest.approx(0.11)
