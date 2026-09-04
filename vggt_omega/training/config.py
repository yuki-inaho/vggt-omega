"""Validation helpers for the Hydra training configuration."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from omegaconf import DictConfig

_LOSS_WEIGHT_KEYS = (
    "camera_weight",
    "depth_weight",
    "translation_weight",
    "rotation_weight",
    "fov_weight",
)
_OPTIONAL_LOSS_DEFAULTS = {
    "relative_pose_weight": 0.0,
    "relative_rotation_weight": 1.0,
    "relative_translation_direction_weight": 1.0,
    "relative_translation_magnitude_weight": 1.0,
    "photometric_weight": 0.0,
}
_VALIDATION_MONITORS = {
    "val/objective",
    "val/camera",
    "val/camera_translation",
    "val/camera_rotation",
    "val/camera_fov",
    "val/depth",
    "val/pairwise_pose",
    "val/pairwise_rotation_degrees",
    "val/pairwise_translation_direction_degrees",
    "val/pairwise_translation_magnitude",
    "val/rpa_5",
    "val/rpa_15",
    "val/rpa_30",
    "val/near_edge_objective",
    "val/dynamic_classification",
}


def _validate_loss_weights(value: object, owner: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{owner} must be a mapping")
    weights = cast(Mapping[str, object], value)
    for key in _LOSS_WEIGHT_KEYS:
        raw_weight = weights.get(key)
        if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
            raise ValueError(f"{owner}.{key} must be a finite non-negative number")
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"{owner}.{key} must be a finite non-negative number")
    for key, default in _OPTIONAL_LOSS_DEFAULTS.items():
        raw_weight = weights.get(key, default)
        if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
            raise ValueError(f"{owner}.{key} must be a finite non-negative number")
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"{owner}.{key} must be a finite non-negative number")
    raw_max_depth = weights.get("max_metric_depth_m")
    if raw_max_depth is not None:
        if isinstance(raw_max_depth, bool) or not isinstance(raw_max_depth, (int, float)):
            raise ValueError(f"{owner}.max_metric_depth_m must be a finite positive number or null")
        max_depth = float(raw_max_depth)
        if not math.isfinite(max_depth) or max_depth <= 0:
            raise ValueError(f"{owner}.max_metric_depth_m must be a finite positive number or null")


def _validate_pixel_self_supervision(value: object, *, epochs: int, min_frames: int) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise ValueError("pixel_depth.self_supervised must be a mapping")
    gpa = value.get("gpa")
    correspondence = value.get("correspondence")
    guardrail = value.get("guardrail")
    curriculum = value.get("curriculum")
    if (
        not isinstance(gpa, Mapping)
        or not isinstance(correspondence, Mapping)
        or not isinstance(guardrail, Mapping)
        or not isinstance(curriculum, Sequence)
        or isinstance(curriculum, (str, bytes))
    ):
        raise ValueError("pixel_depth.self_supervised requires gpa, correspondence, guardrail, and curriculum")
    for owner, enabled in (("gpa", gpa.get("enabled")), ("correspondence", correspondence.get("enabled"))):
        if not isinstance(enabled, bool):
            raise ValueError(f"pixel_depth.self_supervised.{owner}.enabled must be boolean")
    for key, lower, upper, strictly_positive in (
        ("mu", 0.0, 1.0, False),
        ("lambda_geo", 0.0, None, False),
        ("lambda_smooth", 0.0, None, False),
        ("auto_mask_delta", 0.0, None, False),
        ("geometry_epsilon", 0.0, None, True),
    ):
        raw = gpa.get(key)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
            raise ValueError(f"pixel_depth.self_supervised.gpa.{key} must be finite")
        scalar = float(raw)
        if scalar < lower or (strictly_positive and scalar == lower) or (upper is not None and scalar > upper):
            raise ValueError(f"pixel_depth.self_supervised.gpa.{key} is outside its supported range")
    if not isinstance(gpa.get("auto_mask_enabled"), bool):
        raise ValueError("pixel_depth.self_supervised.gpa.auto_mask_enabled must be boolean")
    if gpa.get("mask_mode") not in {"intersection", "union"}:
        raise ValueError("pixel_depth.self_supervised.gpa.mask_mode must be intersection or union")
    for owner, raw in (
        ("gpa.anchor_count", gpa.get("anchor_count")),
        ("correspondence.hidden_dim", correspondence.get("hidden_dim")),
        ("correspondence.pair_chunk_size", correspondence.get("pair_chunk_size")),
    ):
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
            raise ValueError(f"pixel_depth.self_supervised.{owner} must be a positive integer")
    if bool(gpa["enabled"]) and int(gpa["anchor_count"]) > min_frames:
        raise ValueError("pixel_depth.self_supervised.gpa.anchor_count cannot exceed data.min_frames")
    if bool(correspondence["enabled"]) and min_frames < 2:
        raise ValueError("pixel-depth correspondence requires at least two frames")
    for key, lower, upper in (
        ("alpha", 0.0, 1.0),
        ("epsilon", 0.0, None),
        ("relative_depth_tolerance", 0.0, None),
    ):
        raw = correspondence.get(key)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
            raise ValueError(f"pixel_depth.self_supervised.correspondence.{key} must be finite")
        scalar = float(raw)
        if scalar <= lower or (upper is not None and scalar > upper):
            raise ValueError(f"pixel_depth.self_supervised.correspondence.{key} is outside its supported range")

    if not isinstance(guardrail.get("enabled"), bool):
        raise ValueError("pixel_depth.self_supervised.guardrail.enabled must be boolean")
    guardrail_metrics = guardrail.get("metrics")
    expected_guardrail_metrics = {"near_depth_mae_m", "camera_translation", "objective"}
    if not isinstance(guardrail_metrics, Mapping) or set(guardrail_metrics) != expected_guardrail_metrics:
        raise ValueError(f"pixel-depth guardrail metrics must be exactly {sorted(expected_guardrail_metrics)}")
    for metric_name, thresholds in guardrail_metrics.items():
        if not isinstance(thresholds, Mapping):
            raise ValueError(f"pixel-depth guardrail {metric_name} thresholds must be a mapping")
        if set(thresholds) != {"max_relative_degradation", "max_absolute_degradation"}:
            raise ValueError(f"pixel-depth guardrail {metric_name} threshold fields are invalid")
        for threshold_name, raw in thresholds.items():
            if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)) or raw < 0:
                raise ValueError(
                    f"pixel-depth guardrail {metric_name}.{threshold_name} must be finite and non-negative"
                )

    expected_names = (
        "baseline_parity",
        "residual_gate",
        "gpa_warmup",
        "correspondence_head",
        "joint_low_lr",
        "near_depth_recovery",
    )
    names = tuple(stage.get("name") if isinstance(stage, Mapping) else None for stage in curriculum)
    if names != expected_names:
        raise ValueError(f"pixel-depth self-supervised curriculum stages must be {expected_names}")
    previous_start = -1
    for stage in curriculum:
        assert isinstance(stage, Mapping)
        start_epoch = stage.get("start_epoch")
        if isinstance(start_epoch, bool) or not isinstance(start_epoch, int) or start_epoch <= previous_start:
            raise ValueError("pixel-depth self-supervised curriculum epochs must be strictly increasing integers")
        if start_epoch < 0 or start_epoch >= epochs:
            raise ValueError("pixel-depth self-supervised curriculum stage starts outside configured epochs")
        for key in ("train_enabled", "train_refiner", "train_correspondence", "train_base_heads"):
            if not isinstance(stage.get(key), bool):
                raise ValueError(f"pixel-depth self-supervised curriculum {key} must be boolean")
        for key in ("flow_weight", "gpa_weight", "correspondence_weight", "learning_rate_scale"):
            raw = stage.get(key)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)) or raw < 0:
                raise ValueError(f"pixel-depth self-supervised curriculum {key} must be finite and non-negative")
        previous_start = start_epoch
    baseline = curriculum[0]
    assert isinstance(baseline, Mapping)
    if (
        bool(baseline["train_enabled"])
        or bool(baseline["train_base_heads"])
        or any(
            float(baseline[key]) != 0
            for key in ("flow_weight", "gpa_weight", "correspondence_weight", "learning_rate_scale")
        )
    ):
        raise ValueError("baseline_parity stage must skip training and use zero weights")


def _validate_dynamic_geometry(value: object, *, pixel_depth_enabled: bool, epochs: int) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("dynamic_geometry must be a mapping")
    enabled = value.get("enabled")
    if not isinstance(enabled, bool):
        raise ValueError("dynamic_geometry.enabled must be boolean")
    if value.get("contract_version") != 1:
        raise ValueError("dynamic_geometry.contract_version must be 1")
    if not enabled:
        return
    if not pixel_depth_enabled:
        raise ValueError("dynamic_geometry requires pixel_depth.enabled=true")
    if value.get("pair_mode") != "adjacent_bidirectional":
        raise ValueError("dynamic_geometry.pair_mode must be adjacent_bidirectional")
    if value.get("padding_base_forward") != "group_by_valid_count":
        raise ValueError("dynamic_geometry.padding_base_forward must be group_by_valid_count")
    if value.get("depth_source") != "pixel_refined_fixed_noise":
        raise ValueError("dynamic_geometry.depth_source must be pixel_refined_fixed_noise")
    for key in ("hidden_dim", "relative_camera_dim", "pair_chunk_size"):
        raw = value.get(key)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
            raise ValueError(f"dynamic_geometry.{key} must be a positive integer")
    refinement_seed = value.get("refinement_seed")
    if isinstance(refinement_seed, bool) or not isinstance(refinement_seed, int) or refinement_seed < 0:
        raise ValueError("dynamic_geometry.refinement_seed must be a non-negative integer")
    for key in ("visibility_threshold", "static_probability_max", "dynamic_probability_min"):
        raw = value.get(key)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
            raise ValueError(f"dynamic_geometry.{key} must be finite")
        if not 0 <= float(raw) <= 1:
            raise ValueError(f"dynamic_geometry.{key} must be in [0,1]")
    if float(value["static_probability_max"]) >= float(value["dynamic_probability_min"]):
        raise ValueError("dynamic_geometry static probability must be below dynamic probability")
    prefixes = value.get("joint_base_parameter_prefixes")
    if (
        not isinstance(prefixes, Sequence)
        or isinstance(prefixes, (str, bytes))
        or not prefixes
        or any(not isinstance(prefix, str) or not prefix for prefix in prefixes)
    ):
        raise ValueError("dynamic_geometry.joint_base_parameter_prefixes must contain non-empty strings")
    geometry = value.get("geometry")
    pseudo = value.get("pseudo_labels")
    readiness = value.get("readiness")
    loss = value.get("loss")
    teacher_ema = value.get("teacher_ema")
    guardrail = value.get("guardrail")
    curriculum = value.get("curriculum")
    if not all(isinstance(item, Mapping) for item in (geometry, pseudo, readiness, loss, teacher_ema, guardrail)):
        raise ValueError("dynamic_geometry nested options must be mappings")
    if not isinstance(curriculum, Sequence) or isinstance(curriculum, (str, bytes)):
        raise ValueError("dynamic_geometry.curriculum must be a sequence")
    assert isinstance(geometry, Mapping)
    max_depth = geometry.get("max_depth_m")
    if isinstance(max_depth, bool) or not isinstance(max_depth, (int, float)) or not math.isfinite(float(max_depth)):
        raise ValueError("dynamic_geometry.geometry.max_depth_m must be finite")
    if float(max_depth) <= 0:
        raise ValueError("dynamic_geometry.geometry.max_depth_m must be positive")
    assert isinstance(pseudo, Mapping)
    manifest = pseudo.get("teacher_artifact_manifest")
    if not isinstance(manifest, str) or not manifest or Path(manifest).is_absolute() or ".." in Path(manifest).parts:
        raise ValueError("dynamic_geometry pseudo teacher manifest must be a private-safe relative path")
    for key in (
        "static_off_m",
        "dynamic_on_m",
        "flow_confidence_min",
        "forward_backward_cycle_px",
        "depth_discontinuity_relative",
        "photo_sigma",
        "texture_gradient_scale",
        "flow_coherence_sigma_px",
        "max_flow_fraction",
    ):
        raw = pseudo.get(key)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
            raise ValueError(f"dynamic_geometry.pseudo_labels.{key} must be finite")
        if float(raw) < 0:
            raise ValueError(f"dynamic_geometry.pseudo_labels.{key} must be non-negative")
    if float(pseudo["static_off_m"]) >= float(pseudo["dynamic_on_m"]):
        raise ValueError("dynamic_geometry pseudo static threshold must be below dynamic threshold")
    if not 0 < float(pseudo["flow_confidence_min"]) <= 1:
        raise ValueError("dynamic_geometry flow confidence threshold must be in (0,1]")
    if not 0 < float(pseudo["max_flow_fraction"]) <= 1:
        raise ValueError("dynamic_geometry max_flow_fraction must be in (0,1]")
    assert isinstance(teacher_ema, Mapping)
    if not isinstance(teacher_ema.get("enabled"), bool):
        raise ValueError("dynamic_geometry.teacher_ema.enabled must be boolean")
    decay = teacher_ema.get("decay")
    if isinstance(decay, bool) or not isinstance(decay, (int, float)) or not 0 < float(decay) < 1:
        raise ValueError("dynamic_geometry.teacher_ema.decay must be in (0,1)")
    expected_names = ("baseline_parity", "motion_only", "visibility_dynamic", "joint_low_lr")
    names = tuple(stage.get("name") if isinstance(stage, Mapping) else None for stage in curriculum)
    if names != expected_names:
        raise ValueError(f"dynamic_geometry curriculum stages must be {expected_names}")
    previous_start = -1
    for stage in curriculum:
        assert isinstance(stage, Mapping)
        start = stage.get("start_epoch")
        if isinstance(start, bool) or not isinstance(start, int) or start <= previous_start or start >= epochs:
            raise ValueError("dynamic_geometry curriculum epochs must be strictly increasing and in range")
        if not isinstance(stage.get("train_enabled"), bool):
            raise ValueError("dynamic_geometry curriculum train_enabled must be boolean")
        scale = stage.get("learning_rate_scale")
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or float(scale) < 0:
            raise ValueError("dynamic_geometry curriculum learning_rate_scale must be non-negative")
        previous_start = start


def validate_training_config(cfg: DictConfig) -> None:
    """Reject unsafe or unsupported training combinations before allocating a model."""

    patch_size = int(cfg.model.patch_size)
    image_height = int(cfg.model.image_height)
    image_width = int(cfg.model.image_width)
    if patch_size <= 0 or image_height % patch_size or image_width % patch_size:
        raise ValueError(
            "model image dimensions must both be divisible by the patch size; "
            f"got H={image_height}, W={image_width}, patch size={patch_size}"
        )

    min_frames = int(cfg.data.min_frames)
    max_frames = int(cfg.data.max_frames)
    if min_frames < 2 or max_frames < min_frames:
        raise ValueError(f"invalid data frame range: min_frames={min_frames}, max_frames={max_frames}")
    sequence_frames = cfg.trainer.sequence_frames
    if sequence_frames is not None and not min_frames <= int(sequence_frames) <= max_frames:
        raise ValueError("trainer.sequence_frames must be within the configured data frame range")
    if int(cfg.data.batch_size) > 1 and sequence_frames is None and min_frames != max_frames:
        raise ValueError("variable-length frame sampling currently requires data.batch_size=1")
    overlap = cfg.data.overlap_curriculum
    if not isinstance(overlap.enabled, bool):
        raise ValueError("data.overlap_curriculum.enabled must be boolean")
    if str(overlap.metric) not in {"all_depth", "near_depth"}:
        raise ValueError("data.overlap_curriculum.metric must be all_depth or near_depth")
    overlap_start = float(overlap.start_target)
    overlap_end = float(overlap.end_target)
    overlap_tolerance = float(overlap.target_tolerance)
    if not math.isfinite(overlap_start) or not math.isfinite(overlap_end) or not 0 <= overlap_end <= overlap_start <= 1:
        raise ValueError("overlap curriculum targets must satisfy 0 <= end <= start <= 1")
    if not math.isfinite(overlap_tolerance) or overlap_tolerance < 0:
        raise ValueError("overlap curriculum target_tolerance must be finite and non-negative")
    if isinstance(overlap.epochs, bool) or int(overlap.epochs) < 1:
        raise ValueError("overlap curriculum epochs must be at least 1")

    if int(cfg.checkpoint.k) < 1:
        raise ValueError(f"checkpoint.k must be at least 1, got {cfg.checkpoint.k}")
    if str(cfg.checkpoint.mode) not in {"min", "max"}:
        raise ValueError(f"checkpoint.mode must be 'min' or 'max', got {cfg.checkpoint.mode!r}")
    if str(cfg.checkpoint.monitor) not in _VALIDATION_MONITORS:
        raise ValueError("checkpoint.monitor must name a supported validation scalar")

    strategy = str(cfg.trainer.strategy).lower()
    optimizer_name = str(cfg.optimizer.name).lower()
    scheduler = str(cfg.optimizer.scheduler).lower()
    if optimizer_name == "amuse":
        if strategy == "fsdp":
            raise ValueError("AMUSE with FSDP is unsupported in the initial training implementation")
        if scheduler != "none":
            raise ValueError("AMUSE owns its warmup schedule; an external scheduler is unsupported")
        warmup_ratio = float(cfg.optimizer.warmup_ratio)
        if not 0.0 < warmup_ratio <= 1.0:
            raise ValueError(f"AMUSE warmup_ratio must be in (0, 1], got {warmup_ratio}")
    elif optimizer_name == "adamw":
        if scheduler not in {"none", "constant", "cosine"}:
            raise ValueError(f"unsupported AdamW scheduler: {scheduler!r}")
    else:
        raise ValueError(f"unsupported optimizer: {optimizer_name!r}")

    if strategy not in {"single", "ddp", "fsdp"}:
        raise ValueError(f"unsupported trainer.strategy: {strategy!r}")
    if str(cfg.trainer.device) not in {"cpu", "cuda"}:
        raise ValueError("trainer.device must be 'cpu' or 'cuda'")
    if str(cfg.model.precision) not in {"fp32", "bf16"}:
        raise ValueError("model.precision must be 'fp32' or 'bf16'")
    compile_config = cfg.performance.compile
    if not isinstance(compile_config.enabled, bool):
        raise ValueError("performance.compile.enabled must be boolean")
    if str(compile_config.backend) not in {"inductor", "eager", "aot_eager"}:
        raise ValueError("performance.compile.backend is unsupported")
    if str(compile_config.mode) not in {"default", "reduce-overhead", "max-autotune"}:
        raise ValueError("performance.compile.mode is unsupported")
    if not isinstance(compile_config.fullgraph, bool) or not isinstance(compile_config.dynamic, bool):
        raise ValueError("performance.compile fullgraph/dynamic must be boolean")
    compile_targets = tuple(compile_config.targets)
    supported_compile_targets = {"semantic_adapter", "temporal_mixer", "refiner", "correspondence_head"}
    if not compile_targets or len(set(compile_targets)) != len(compile_targets):
        raise ValueError("performance.compile.targets must be unique and non-empty")
    if not set(compile_targets) <= supported_compile_targets:
        raise ValueError("performance.compile.targets contains an unsupported module")
    if bool(compile_config.enabled) and not bool(cfg.pixel_depth.enabled):
        raise ValueError("performance.compile requires pixel_depth.enabled=true for the configured targets")
    if bool(compile_config.enabled) and "correspondence_head" in compile_targets:
        self_supervised = cfg.pixel_depth.get("self_supervised")
        correspondence = self_supervised.get("correspondence") if self_supervised is not None else None
        if correspondence is None or not bool(correspondence.get("enabled", False)):
            raise ValueError(
                "performance.compile target correspondence_head requires "
                "pixel_depth.self_supervised.correspondence.enabled=true"
            )
    loader_performance = cfg.performance.data_loader
    if isinstance(loader_performance.prefetch_factor, bool) or int(loader_performance.prefetch_factor) < 1:
        raise ValueError("performance.data_loader.prefetch_factor must be a positive integer")
    if not isinstance(loader_performance.persistent_workers, bool):
        raise ValueError("performance.data_loader.persistent_workers must be boolean")
    profiling = cfg.performance.profiling
    if not isinstance(profiling.enabled, bool):
        raise ValueError("performance.profiling.enabled must be boolean")
    for name in ("warmup_steps", "active_steps"):
        raw = profiling[name]
        if isinstance(raw, bool) or int(raw) < (0 if name == "warmup_steps" else 1):
            raise ValueError(f"performance.profiling.{name} is outside its supported range")
    if bool(profiling.enabled) and int(cfg.trainer.gradient_accumulation_steps) != 1:
        raise ValueError("performance.profiling requires trainer.gradient_accumulation_steps=1")
    runtime_contracts = cfg.performance.runtime_contracts
    if not isinstance(runtime_contracts.enabled, bool) or not isinstance(runtime_contracts.first_batch_only, bool):
        raise ValueError("performance.runtime_contracts options must be boolean")
    renderer_backend = str(cfg.renderer.backend)
    if renderer_backend not in {"soft", "gsplat"}:
        raise ValueError("renderer.backend must be soft or gsplat")
    renderer_tolerance = float(cfg.renderer.relative_depth_tolerance)
    renderer_max_depth = float(cfg.renderer.max_depth_m)
    if not math.isfinite(renderer_tolerance) or renderer_tolerance <= 0:
        raise ValueError("renderer.relative_depth_tolerance must be finite and positive")
    if not math.isfinite(renderer_max_depth) or renderer_max_depth <= 0:
        raise ValueError("renderer.max_depth_m must be finite and positive")
    if str(cfg.renderer.pose_source) not in {"predicted", "ground_truth"}:
        raise ValueError("renderer.pose_source must be predicted or ground_truth")
    if not isinstance(cfg.renderer.use_target_depth, bool):
        raise ValueError("renderer.use_target_depth must be boolean")
    if renderer_backend == "soft":
        temperature = float(cfg.renderer.z_temperature)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("renderer.z_temperature must be finite and positive")
    else:
        radius = float(cfg.renderer.gaussian_radius_pixels)
        opacity = float(cfg.renderer.opacity)
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("renderer.gaussian_radius_pixels must be finite and positive")
        if not math.isfinite(opacity) or not 0 < opacity <= 1:
            raise ValueError("renderer.opacity must be finite and within (0, 1]")
    if float(cfg.trainer.gradient_clip_norm) <= 0:
        raise ValueError("trainer.gradient_clip_norm must be positive")
    if int(cfg.trainer.epochs) < 1:
        raise ValueError("trainer.epochs must be at least 1")

    pixel_depth = cfg.pixel_depth
    if not isinstance(pixel_depth.enabled, bool):
        raise ValueError("pixel_depth.enabled must be boolean")
    for owner, value in (
        ("pixel_depth.semantic_input_dim", pixel_depth.semantic_input_dim),
        ("pixel_depth.refiner.hidden_dim", pixel_depth.refiner.hidden_dim),
        ("pixel_depth.refiner.depth", pixel_depth.refiner.depth),
        ("pixel_depth.refiner.num_heads", pixel_depth.refiner.num_heads),
        ("pixel_depth.refiner.coarse_patch_size", pixel_depth.refiner.coarse_patch_size),
        ("pixel_depth.refiner.fine_patch_size", pixel_depth.refiner.fine_patch_size),
        ("pixel_depth.temporal.depth", pixel_depth.temporal.depth),
        ("pixel_depth.flow.ode_steps", pixel_depth.flow.ode_steps),
    ):
        if isinstance(value, bool) or int(value) < 1:
            raise ValueError(f"{owner} must be a positive integer")
    if int(pixel_depth.refiner.hidden_dim) % int(pixel_depth.refiner.num_heads):
        raise ValueError("pixel_depth refiner hidden_dim must be divisible by num_heads")
    if int(pixel_depth.refiner.coarse_patch_size) != 2 * int(pixel_depth.refiner.fine_patch_size):
        raise ValueError("pixel_depth coarse_patch_size must be twice fine_patch_size")
    if str(pixel_depth.temporal.reference_mode) not in {"first", "random_valid"}:
        raise ValueError("pixel_depth temporal reference_mode must be first or random_valid")
    if not isinstance(pixel_depth.temporal.preserve_frame_order, bool):
        raise ValueError("pixel_depth temporal preserve_frame_order must be boolean")
    if str(pixel_depth.flow.time_sampling) != "logit_normal":
        raise ValueError("pixel_depth flow time_sampling must be logit_normal")
    for owner, value, allow_zero in (
        ("pixel_depth.flow.log_residual_scale", pixel_depth.flow.log_residual_scale, False),
        ("pixel_depth.flow.gradient_loss_weight", pixel_depth.flow.gradient_loss_weight, True),
        ("pixel_depth.flow.objective_weight", pixel_depth.flow.objective_weight, True),
        ("pixel_depth.flow.edge_objective_weight", pixel_depth.flow.edge_objective_weight, True),
        ("pixel_depth.flow.multiview_objective_weight", pixel_depth.flow.multiview_objective_weight, True),
        ("pixel_depth.flow.residual_gate_initial", pixel_depth.flow.residual_gate_initial, True),
        ("pixel_depth.geometry.max_depth_m", pixel_depth.geometry.max_depth_m, False),
        ("pixel_depth.geometry.relative_depth_tolerance", pixel_depth.geometry.relative_depth_tolerance, False),
    ):
        scalar = float(value)
        if not math.isfinite(scalar) or scalar < 0 or (not allow_zero and scalar == 0):
            raise ValueError(f"{owner} must be finite and {'non-negative' if allow_zero else 'positive'}")
    residual_gate_initial = float(pixel_depth.flow.residual_gate_initial)
    if residual_gate_initial > 1.0:
        raise ValueError("pixel_depth.flow.residual_gate_initial must be within [0, 1]")
    if not isinstance(pixel_depth.optimization.train_base_heads, bool):
        raise ValueError("pixel_depth optimization train_base_heads must be boolean")
    _validate_pixel_self_supervision(
        pixel_depth.get("self_supervised"),
        epochs=int(cfg.trainer.epochs),
        min_frames=min_frames,
    )
    _validate_dynamic_geometry(
        cfg.dynamic_geometry,
        pixel_depth_enabled=bool(pixel_depth.enabled),
        epochs=int(cfg.trainer.epochs),
    )
    self_supervised = pixel_depth.get("self_supervised")
    if isinstance(self_supervised, Mapping):
        guardrail = self_supervised.get("guardrail")
        if isinstance(guardrail, Mapping) and bool(guardrail.get("enabled")):
            if not bool(cfg.checkpoint.save_last):
                raise ValueError("pixel-depth curriculum guardrail requires checkpoint.save_last=true")
            if int(cfg.trainer.validate_every_epochs) != 1:
                raise ValueError("pixel-depth curriculum guardrail requires validation every epoch")

    initial_head = cfg.model.initial_head_checkpoint
    if initial_head is not None:
        if not isinstance(initial_head, str) or not initial_head:
            raise ValueError("model.initial_head_checkpoint must be a non-empty relative path")
        initial_path = Path(initial_head)
        if initial_path.is_absolute() or ".." in initial_path.parts:
            raise ValueError("model.initial_head_checkpoint must be a private-safe relative path")

    _validate_loss_weights(cfg.loss.training, "loss.training")
    _validate_loss_weights(cfg.loss.validation, "loss.validation")
    previous_start = -1
    for index, stage in enumerate(cfg.loss.curriculum):
        if not isinstance(stage, Mapping):
            raise ValueError(f"loss.curriculum[{index}] must be a mapping")
        start_epoch = stage.get("start_epoch")
        if isinstance(start_epoch, bool) or not isinstance(start_epoch, int) or start_epoch < 0:
            raise ValueError(f"loss.curriculum[{index}].start_epoch must be a non-negative integer")
        if start_epoch <= previous_start:
            raise ValueError("loss curriculum start epochs must be strictly increasing")
        if index == 0 and start_epoch != 0:
            raise ValueError("loss curriculum must start at epoch 0")
        if start_epoch >= int(cfg.trainer.epochs):
            raise ValueError("loss curriculum stage starts outside the configured epoch range")
        _validate_loss_weights(stage, f"loss.curriculum[{index}]")
        previous_start = start_epoch

    early = cfg.trainer.early_stopping
    if not isinstance(early.enabled, bool):
        raise ValueError("trainer.early_stopping.enabled must be boolean")
    if str(early.monitor) not in _VALIDATION_MONITORS:
        raise ValueError("trainer.early_stopping.monitor must name a supported validation scalar")
    if str(early.mode) not in {"min", "max"}:
        raise ValueError("trainer.early_stopping.mode must be min or max")
    if isinstance(early.patience, bool) or int(early.patience) < 1:
        raise ValueError("trainer.early_stopping.patience must be at least 1")
    min_delta = float(early.min_delta)
    if not math.isfinite(min_delta) or min_delta < 0:
        raise ValueError("trainer.early_stopping.min_delta must be finite and non-negative")
    if bool(early.enabled) and (
        str(early.monitor) != str(cfg.checkpoint.monitor) or str(early.mode) != str(cfg.checkpoint.mode)
    ):
        raise ValueError("early stopping and checkpoint selection must use the same monitor and mode")
