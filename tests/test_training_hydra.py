from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from vggt_omega.training.config import validate_training_config

CONFIG_DIR = Path(__file__).parents[1] / "configs" / "training"


def _compose(*overrides: str, return_hydra_config: bool = False) -> DictConfig:
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        return compose(
            config_name="config",
            overrides=list(overrides),
            return_hydra_config=return_hydra_config,
        )


def test_default_training_config_composes_and_validates() -> None:
    cfg = _compose()

    validate_training_config(cfg)

    assert cfg.optimizer.name == "amuse"
    assert cfg.loss.name == "standard"
    assert cfg.trainer.name == "smoke"
    assert cfg.checkpoint.k == 3
    assert cfg.checkpoint.monitor == "val/objective"
    assert cfg.logging.name == "tensorboard"
    assert cfg.performance.compile.enabled is False
    assert cfg.performance.data_loader.prefetch_factor == 2
    assert cfg.performance.runtime_contracts.first_batch_only is True
    assert cfg.depth_input.enabled is False


def test_profiled_compile_performance_config_composes_and_validates() -> None:
    cfg = _compose(
        "data=colmap_rgbd_fixed4_b14",
        "pixel_depth=pixel_perfect_gpa_correspondence",
        "performance=profiled_compile",
        "trainer=finetune",
        "trainer.epochs=6",
    )

    validate_training_config(cfg)

    assert cfg.performance.compile.enabled is True
    assert cfg.performance.compile.backend == "inductor"
    assert cfg.performance.compile.mode == "default"
    assert cfg.performance.profiling.enabled is True
    assert cfg.performance.data_loader.persistent_workers is True


def test_compile_profile_requires_matching_pixel_depth_modules() -> None:
    cfg = _compose("performance=profiled_compile")

    with pytest.raises(ValueError, match="pixel_depth"):
        validate_training_config(cfg)


def test_compile_correspondence_target_requires_enabled_head() -> None:
    cfg = _compose("pixel_depth=pixel_perfect_guarded", "performance=profiled_compile")

    with pytest.raises(ValueError, match="correspondence_head"):
        validate_training_config(cfg)


@pytest.mark.parametrize("value", [0, -1, True])
def test_data_loader_prefetch_factor_must_be_positive_integer(value: object) -> None:
    cfg = _compose()
    cfg.performance.data_loader.prefetch_factor = value

    with pytest.raises(ValueError, match="prefetch_factor"):
        validate_training_config(cfg)


def test_profiling_requires_single_optimizer_step_per_profiled_batch() -> None:
    cfg = _compose()
    cfg.performance.profiling.enabled = True
    cfg.trainer.gradient_accumulation_steps = 2

    with pytest.raises(ValueError, match="gradient_accumulation_steps=1"):
        validate_training_config(cfg)


def test_camera_motion_curriculum_composes_and_validates() -> None:
    cfg = _compose("loss=camera_motion_curriculum", "trainer=finetune", "trainer.epochs=9")

    validate_training_config(cfg)

    assert [stage.start_epoch for stage in cfg.loss.curriculum] == [0, 3, 6]
    assert cfg.loss.curriculum[0].translation_weight == pytest.approx(4.0)
    assert cfg.loss.curriculum[0].rotation_weight == pytest.approx(2.0)
    assert cfg.loss.validation.translation_weight == pytest.approx(1.0)


def test_translation_focus_early_stopping_config_validates() -> None:
    cfg = _compose(
        "loss=camera_translation_focus",
        "trainer=finetune",
        "trainer.epochs=5",
        "checkpoint.monitor=val/camera_translation",
        "trainer.early_stopping.enabled=true",
        "trainer.early_stopping.monitor=val/camera_translation",
    )

    validate_training_config(cfg)

    assert cfg.loss.training.translation_weight == pytest.approx(4.0)
    assert cfg.trainer.early_stopping.patience == 2


def test_near_depth_focus_config_validates() -> None:
    cfg = _compose("loss=near_depth_focus", "trainer=finetune", "trainer.epochs=5")

    validate_training_config(cfg)

    assert cfg.loss.training.max_metric_depth_m == pytest.approx(1.2)
    assert cfg.loss.validation.max_metric_depth_m == pytest.approx(1.2)
    assert cfg.loss.training.depth_weight == pytest.approx(4.0)


def test_fixed_four_frame_batch14_profile_composes_and_validates() -> None:
    cfg = _compose("data=colmap_rgbd_fixed4_b14", "trainer=finetune")

    validate_training_config(cfg)

    assert cfg.data.min_frames == 4
    assert cfg.data.max_frames == 4
    assert cfg.data.batch_size == 14


def test_fixed_four_frame_batch8_profile_composes_and_validates() -> None:
    cfg = _compose("data=colmap_rgbd_fixed4_b8", "trainer=finetune")

    validate_training_config(cfg)

    assert cfg.data.min_frames == 4
    assert cfg.data.max_frames == 4
    assert cfg.data.batch_size == 8


def test_640x480_base_only_config_disables_all_optional_wrappers() -> None:
    cfg = _compose(
        "data=colmap_rgbd_640x480_fixed4",
        "model=omega_1b_640x480_base",
        "pixel_depth=disabled",
        "dynamic_geometry=disabled",
        "loss=standard",
        "trainer.sequence_frames=4",
    )

    validate_training_config(cfg)

    assert cfg.model.name == "omega_1b_640x480_base"
    assert cfg.model.image_height == 480
    assert cfg.model.image_width == 640
    assert cfg.model.patch_size == 16
    assert cfg.model.precision == "bf16"
    assert cfg.model.initial_head_checkpoint is None
    assert cfg.model.freeze_aggregator is True
    assert cfg.model.freeze_confidence is True
    assert cfg.pixel_depth.enabled is False
    assert cfg.dynamic_geometry.enabled is False
    assert cfg.depth_input.enabled is False
    assert cfg.loss.name == "standard"


def test_640x480_mapped_depth_config_is_opt_in_and_exclusive() -> None:
    cfg = _compose(
        "data=colmap_rgbd_640x480_fixed4",
        "model=omega_1b_640x480_base",
        "depth_input=mapped_depth",
        "trainer.sequence_frames=4",
    )

    validate_training_config(cfg)

    assert cfg.depth_input.enabled is True
    assert cfg.depth_input.patch_size == 16
    assert cfg.depth_input.embed_dim == 1024
    assert cfg.depth_input.validation_provided_frames == 4

    cfg.pixel_depth.enabled = True
    with pytest.raises(ValueError, match="cannot be enabled together"):
        validate_training_config(cfg)


def test_guarded_pixel_depth_profile_starts_at_exact_baseline() -> None:
    cfg = _compose("pixel_depth=pixel_perfect_guarded")

    validate_training_config(cfg)

    assert cfg.pixel_depth.flow.residual_gate_initial == pytest.approx(0.0)
    assert cfg.pixel_depth.optimization.train_base_heads is False


def test_gpa_correspondence_curriculum_profile_has_six_guarded_stages() -> None:
    cfg = _compose(
        "data=colmap_rgbd_fixed4_b14",
        "pixel_depth=pixel_perfect_gpa_correspondence",
        "loss=near_depth_focus",
        "trainer=finetune",
        "trainer.epochs=6",
    )

    validate_training_config(cfg)

    stages = cfg.pixel_depth.self_supervised.curriculum
    assert [stage.name for stage in stages] == [
        "baseline_parity",
        "residual_gate",
        "gpa_warmup",
        "correspondence_head",
        "joint_low_lr",
        "near_depth_recovery",
    ]
    assert [stage.start_epoch for stage in stages] == list(range(6))
    assert stages[0].train_enabled is False
    assert stages[3].train_refiner is False and stages[3].train_correspondence is True
    assert cfg.pixel_depth.flow.residual_gate_initial == pytest.approx(0.0)
    assert cfg.pixel_depth.optimization.train_base_heads is True
    assert stages[1].train_base_heads is False
    assert stages[2].train_base_heads is True
    assert cfg.pixel_depth.self_supervised.guardrail.enabled is True
    assert set(cfg.pixel_depth.self_supervised.guardrail.metrics) == {
        "near_depth_mae_m",
        "camera_translation",
        "objective",
    }


def test_gpa_anchor_count_cannot_exceed_guaranteed_frame_count() -> None:
    cfg = _compose(
        "pixel_depth=pixel_perfect_gpa_correspondence",
        "trainer=finetune",
        "trainer.epochs=6",
    )

    with pytest.raises(ValueError, match="anchor_count"):
        validate_training_config(cfg)


@pytest.mark.parametrize("value", [0.0, float("nan"), float("inf")])
def test_metric_depth_limit_must_be_finite_and_positive(value: float) -> None:
    cfg = _compose("loss=near_depth_focus", "trainer=finetune", "trainer.epochs=5")
    cfg.loss.training.max_metric_depth_m = value

    with pytest.raises(ValueError, match="max_metric_depth_m"):
        validate_training_config(cfg)


def test_pixel_depth_residual_gate_rejects_values_above_one() -> None:
    cfg = _compose("pixel_depth=pixel_perfect_guarded")
    cfg.pixel_depth.flow.residual_gate_initial = 1.01

    with pytest.raises(ValueError, match="residual_gate_initial"):
        validate_training_config(cfg)


def test_erayzer_overlap_pairwise_profiles_compose_with_warmup() -> None:
    cfg = _compose(
        "data=colmap_rgbd_overlap",
        "model=omega_1b_512_near_head",
        "loss=erayzer_pairwise_near",
        "trainer=finetune",
        "trainer.epochs=6",
    )

    validate_training_config(cfg)

    assert cfg.data.overlap_curriculum.enabled is True
    assert cfg.data.overlap_curriculum.metric == "near_depth"
    assert cfg.data.overlap_curriculum.start_target == pytest.approx(0.24386708438396454)
    assert [stage.relative_pose_weight for stage in cfg.loss.curriculum] == pytest.approx([0.02, 0.05, 0.1])
    assert cfg.loss.validation.relative_pose_weight == pytest.approx(0.1)


def test_rpa_checkpoint_monitor_requires_max_mode_and_validates() -> None:
    cfg = _compose(
        "data=colmap_rgbd_overlap",
        "loss=erayzer_pairwise_near",
        "trainer=finetune",
        "trainer.epochs=4",
        "checkpoint.monitor=val/rpa_15",
        "checkpoint.mode=max",
    )

    validate_training_config(cfg)

    assert cfg.checkpoint.monitor == "val/rpa_15"
    assert cfg.checkpoint.mode == "max"


def test_explicit_gsplat_renderer_profile_composes() -> None:
    cfg = _compose("renderer=gsplat")

    validate_training_config(cfg)

    assert cfg.renderer.backend == "gsplat"
    assert cfg.renderer.gaussian_radius_pixels == pytest.approx(0.75)
    assert cfg.renderer.opacity == pytest.approx(0.95)


def test_photometric_profile_composes_with_explicit_soft_renderer() -> None:
    cfg = _compose(
        "data=colmap_rgbd_overlap",
        "model=omega_1b_512_near_head",
        "loss=erayzer_photometric_near",
        "renderer=soft",
        "trainer=finetune",
        "trainer.epochs=4",
    )

    validate_training_config(cfg)

    assert cfg.loss.training.photometric_weight == pytest.approx(0.01)
    assert cfg.loss.validation.photometric_weight == pytest.approx(0.01)
    assert cfg.renderer.backend == "soft"
    assert cfg.renderer.pose_source == "predicted"
    assert cfg.renderer.use_target_depth is True


def test_early_stopping_monitor_must_match_checkpoint_monitor() -> None:
    cfg = _compose("trainer=finetune", "trainer.early_stopping.enabled=true")
    cfg.trainer.early_stopping.monitor = "val/camera_translation"

    with pytest.raises(ValueError, match="same monitor"):
        validate_training_config(cfg)


def test_curriculum_resume_keeps_initial_head_provenance() -> None:
    cfg = _compose("loss=camera_motion_curriculum", "trainer=finetune", "trainer.epochs=9")
    cfg.model.initial_head_checkpoint = "previous/checkpoints/last.pt"
    cfg.trainer.resume_from = "current/checkpoints/last.pt"

    validate_training_config(cfg)


@pytest.mark.parametrize(
    "path",
    ["/private/last.pt", "../outside/last.pt", ""],
)
def test_initial_head_checkpoint_must_be_private_safe_relative_path(path: str) -> None:
    cfg = _compose()
    cfg.model.initial_head_checkpoint = path

    with pytest.raises(ValueError, match="initial_head_checkpoint"):
        validate_training_config(cfg)


def test_dynamic_geometry_disabled_is_the_default_noop_profile() -> None:
    cfg = _compose()

    validate_training_config(cfg)

    assert cfg.dynamic_geometry.enabled is False
    assert cfg.dynamic_geometry.contract_version == 1


def test_dynamic_geometry_v1_curriculum_composes_and_validates() -> None:
    cfg = _compose(
        "data=colmap_rgbd_fixed4_b14",
        "pixel_depth=pixel_perfect_gpa_correspondence",
        "dynamic_geometry=v1",
        "trainer=finetune",
        "trainer.epochs=6",
    )

    validate_training_config(cfg)

    assert cfg.dynamic_geometry.depth_source == "pixel_refined_fixed_noise"
    assert cfg.dynamic_geometry.geometry.max_depth_m == pytest.approx(1.2)
    assert [stage.name for stage in cfg.dynamic_geometry.curriculum] == [
        "baseline_parity",
        "motion_only",
        "visibility_dynamic",
        "joint_low_lr",
    ]


def test_dynamic_classification_extended_profile_uses_classification_topk() -> None:
    cfg = _compose(
        "data=colmap_rgbd_fixed4_b8",
        "pixel_depth=pixel_perfect_gpa_correspondence",
        "dynamic_geometry=classification_extended",
        "checkpoint=topk_dynamic_classification",
        "trainer=finetune",
        "trainer.epochs=13",
        "trainer.max_train_steps=48",
    )

    validate_training_config(cfg)

    assert [stage.start_epoch for stage in cfg.dynamic_geometry.curriculum] == [0, 1, 2, 12]
    assert cfg.checkpoint.monitor == "val/dynamic_classification"
    assert cfg.checkpoint.mode == "min"


def test_dynamic_classification_lr_100x_changes_only_classification_stage_scale() -> None:
    baseline = _compose(
        "data=colmap_rgbd_fixed4_b8",
        "pixel_depth=pixel_perfect_gpa_correspondence",
        "dynamic_geometry=classification_extended",
        "trainer=finetune",
        "trainer.epochs=13",
        "trainer.max_train_steps=48",
    )
    experiment = _compose(
        "data=colmap_rgbd_fixed4_b8",
        "pixel_depth=pixel_perfect_gpa_correspondence",
        "dynamic_geometry=classification_lr_100x",
        "trainer=finetune",
        "trainer.epochs=13",
        "trainer.max_train_steps=48",
    )

    validate_training_config(experiment)
    baseline_dynamic = OmegaConf.to_container(baseline.dynamic_geometry, resolve=True)
    experiment_dynamic = OmegaConf.to_container(experiment.dynamic_geometry, resolve=True)
    assert isinstance(baseline_dynamic, dict) and isinstance(experiment_dynamic, dict)
    baseline_stages = baseline_dynamic.pop("curriculum")
    experiment_stages = experiment_dynamic.pop("curriculum")
    assert experiment_dynamic == baseline_dynamic
    assert isinstance(baseline_stages, list) and isinstance(experiment_stages, list)
    for index, (baseline_stage, experiment_stage) in enumerate(zip(baseline_stages, experiment_stages, strict=True)):
        if index == 2:
            assert {**experiment_stage, "learning_rate_scale": 0.5} == baseline_stage
            assert experiment_stage["learning_rate_scale"] == 100.0
        else:
            assert experiment_stage == baseline_stage


def test_dynamic_geometry_requires_pixel_depth_wrapper() -> None:
    cfg = _compose("dynamic_geometry=v1", "trainer=finetune", "trainer.epochs=6")

    with pytest.raises(ValueError, match=r"pixel_depth\.enabled"):
        validate_training_config(cfg)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("visibility_threshold", -0.1),
        ("static_probability_max", 0.9),
        ("dynamic_probability_min", 0.2),
        ("pair_chunk_size", 0),
    ],
)
def test_dynamic_geometry_rejects_invalid_thresholds(key: str, value: object) -> None:
    cfg = _compose(
        "data=colmap_rgbd_fixed4_b14",
        "pixel_depth=pixel_perfect_gpa_correspondence",
        "dynamic_geometry=v1",
        "trainer=finetune",
        "trainer.epochs=6",
    )
    cfg.dynamic_geometry[key] = value

    with pytest.raises(ValueError, match="dynamic_geometry"):
        validate_training_config(cfg)


def test_adamw_finetune_override_composes() -> None:
    cfg = _compose("optimizer=adamw", "trainer=finetune", "checkpoint.k=2")

    validate_training_config(cfg)

    assert cfg.optimizer.name == "adamw"
    assert cfg.trainer.name == "finetune"
    assert cfg.checkpoint.k == 2


def test_tensorboard_can_only_be_disabled_with_an_explicit_profile() -> None:
    cfg = _compose("logging=tensorboard_disabled")

    validate_training_config(cfg)

    assert cfg.logging.name == "tensorboard_disabled"
    assert cfg.logging.enabled is False


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (("optimizer=amuse", "trainer.strategy=fsdp"), "FSDP"),
        (("optimizer=amuse", "optimizer.scheduler=cosine"), "scheduler"),
        (("checkpoint.k=0",), "checkpoint.k"),
        (("model.image_height=385",), "patch size"),
    ],
)
def test_invalid_training_config_is_rejected(overrides: tuple[str, ...], message: str) -> None:
    cfg = _compose(*overrides)

    with pytest.raises(ValueError, match=message):
        validate_training_config(cfg)


def test_hydra_keeps_the_repository_working_directory() -> None:
    cfg = _compose(return_hydra_config=True)

    assert cfg.hydra.job.chdir is False
    assert cfg.hydra.output_subdir is None
    assert "outputs/training/" in cfg.hydra.run.dir
