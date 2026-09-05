"""Epoch primitives and orchestration for supervised VGGT-Omega training."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from vggt_omega.training.checkpointing import (
    TopKCheckpointManager,
    load_resume_checkpoint,
    optimizer_evaluation_state,
)
from vggt_omega.training.config import validate_training_config
from vggt_omega.training.dataset import ColmapRgbdDataset
from vggt_omega.training.depth_input_model import fixed_depth_availability, sample_depth_availability
from vggt_omega.training.losses import compute_camera_depth_loss, compute_depth_loss
from vggt_omega.training.model_factory import (
    PreparedTrainingModel,
    attach_depth_input_model,
    attach_dynamic_geometry_model,
    attach_pixel_depth_model,
    build_training_model,
)
from vggt_omega.training.optimizer_factory import build_adamw_optimizer, build_amuse_optimizer
from vggt_omega.training.performance import StepProfiler, validate_training_batch_contract
from vggt_omega.training.tensorboard import TensorBoardScalarLogger


class ScalarLogger(Protocol):
    def log_scalars(self, scalars: Mapping[str, float], *, step: int) -> None: ...


@dataclass(frozen=True)
class TrainEpochResult:
    metrics: dict[str, float]
    global_step: int
    optimizer_steps: int
    batches: int


@dataclass(frozen=True)
class _RuntimeOptimizer:
    optimizer: torch.optim.Optimizer
    scheduler: Any
    group_fingerprint: str


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
_STANDARD_LOSS_CONFIG = {
    "name": "standard",
    "training": {
        "camera_weight": 5.0,
        "depth_weight": 1.0,
        "translation_weight": 1.0,
        "rotation_weight": 1.0,
        "fov_weight": 0.5,
    },
    "curriculum": [],
    "validation": {
        "camera_weight": 5.0,
        "depth_weight": 1.0,
        "translation_weight": 1.0,
        "rotation_weight": 1.0,
        "fov_weight": 0.5,
    },
}
_STANDARD_EARLY_STOPPING_CONFIG = {
    "enabled": False,
    "monitor": "val/objective",
    "mode": "min",
    "patience": 2,
    "min_delta": 0.0,
}
_DEFAULT_DEPTH_EVALUATION_THRESHOLDS_M = (0.4, 0.8, 1.2)


def _loss_options(value: Mapping[str, Any]) -> dict[str, float]:
    options = {key: float(value[key]) for key in _LOSS_WEIGHT_KEYS}
    options.update({key: float(value[key]) for key in _OPTIONAL_LOSS_DEFAULTS if key in value})
    max_metric_depth_m = value.get("max_metric_depth_m")
    if max_metric_depth_m is not None:
        options["max_metric_depth_m"] = float(max_metric_depth_m)
    return options


def _training_loss_options(cfg: DictConfig, epoch: int) -> dict[str, float]:
    selected = cfg.loss.training
    for stage in cfg.loss.curriculum:
        if int(stage.start_epoch) > epoch:
            break
        selected = stage
    return _loss_options(selected)


def _renderer_options(value: Mapping[str, Any]) -> dict[str, object]:
    options: dict[str, object] = {
        "backend": str(value["backend"]),
        "max_depth_m": float(value["max_depth_m"]),
        "relative_depth_tolerance": float(value["relative_depth_tolerance"]),
        "pose_source": str(value.get("pose_source", "predicted")),
        "use_target_depth": bool(value.get("use_target_depth", True)),
    }
    if options["backend"] == "soft":
        options["z_temperature"] = float(value["z_temperature"])
    elif options["backend"] == "gsplat":
        options["gaussian_radius_pixels"] = float(value["gaussian_radius_pixels"])
        options["opacity"] = float(value["opacity"])
    return options


def _metric_improved(value: float, best: float | None, *, mode: str, min_delta: float) -> bool:
    if best is None:
        return True
    return value < best - min_delta if mode == "min" else value > best + min_delta


def _restore_early_stopping_state(
    training_state: Mapping[str, Any] | None,
    *,
    enabled: bool,
    monitor: str,
    mode: str,
    patience: int,
    min_delta: float,
) -> tuple[float | None, int, bool]:
    """Validate and restore early-stopping state at an epoch boundary."""

    if not enabled:
        return None, 0, False
    if training_state is None:
        raise ValueError("resume checkpoint is missing early-stopping training_state")
    raw = training_state.get("early_stopping")
    if not isinstance(raw, Mapping):
        raise ValueError("resume checkpoint training_state.early_stopping must be a dictionary")
    expected = {
        "enabled": enabled,
        "monitor": monitor,
        "mode": mode,
        "patience": patience,
        "min_delta": min_delta,
    }
    for key, expected_value in expected.items():
        if raw.get(key) != expected_value:
            raise ValueError(
                "resume checkpoint early-stopping configuration mismatch: "
                f"{key}={raw.get(key)!r}, expected={expected_value!r}"
            )
    best = raw.get("best")
    if best is not None:
        if isinstance(best, bool) or not isinstance(best, (int, float)) or not math.isfinite(float(best)):
            raise ValueError("resume checkpoint early-stopping best must be finite or null")
        best = float(best)
    bad_epochs = raw.get("bad_epochs")
    if isinstance(bad_epochs, bool) or not isinstance(bad_epochs, int) or bad_epochs < 0:
        raise ValueError("resume checkpoint early-stopping bad_epochs must be a non-negative integer")
    stopped = raw.get("stopped")
    if not isinstance(stopped, bool):
        raise ValueError("resume checkpoint early-stopping stopped must be boolean")
    if stopped != (bad_epochs >= patience):
        raise ValueError("resume checkpoint early-stopping stopped flag is inconsistent with patience")
    return best, bad_epochs, stopped


def _validate_loop_options(
    *,
    gradient_clip_norm: float,
    gradient_accumulation_steps: int,
    log_every_steps: int,
    max_optimizer_steps: int | None,
) -> None:
    if not math.isfinite(gradient_clip_norm) or gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be finite and positive")
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")
    if log_every_steps < 1:
        raise ValueError("log_every_steps must be at least 1")
    if max_optimizer_steps is not None and max_optimizer_steps < 1:
        raise ValueError("max_optimizer_steps must be None or at least 1")


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device=device, non_blocking=device.type == "cuda") if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _autocast_context(device: torch.device, precision: str):
    normalized = precision.lower()
    if normalized == "fp32":
        return nullcontext()
    if normalized != "bf16":
        raise ValueError(f"unsupported precision: {precision!r}; expected 'bf16' or 'fp32'")
    if device.type == "cuda":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 training was requested but this CUDA device does not support bf16")
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if device.type == "cpu":
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    raise ValueError(f"unsupported training device type: {device.type!r}")


def _ensure_finite_loss(losses: Mapping[str, torch.Tensor]) -> None:
    for name, value in losses.items():
        if value.ndim != 0:
            raise ValueError(f"loss {name!r} must be scalar, got shape {tuple(value.shape)}")
        if not torch.isfinite(value):
            raise ValueError(f"non-finite loss {name!r}: {float(value.detach())}")


def _optimizer_beta1(optimizer: torch.optim.Optimizer) -> float | None:
    beta1_init = getattr(optimizer, "beta1_init", None)
    if beta1_init is not None:
        return float(beta1_init)
    if optimizer.param_groups:
        betas = optimizer.param_groups[0].get("betas")
        if betas is not None:
            return float(betas[0])
    return None


def _training_scalars(
    losses: Mapping[str, torch.Tensor],
    *,
    grad_norm: float,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    scalars = {
        "train/objective": float(losses["objective"].detach()),
        "train/camera": float(losses["camera"].detach()),
        "train/camera_translation": float(losses["camera_translation"].detach()),
        "train/camera_rotation": float(losses["camera_rotation"].detach()),
        "train/camera_fov": float(losses["camera_fov"].detach()),
        "train/depth": float(losses["depth"].detach()),
        "train/grad_norm": grad_norm,
    }
    optional_metrics = {
        "sample_overlap": "train/sample_overlap",
        "overlap_target": "train/overlap_target",
        "overlap_fallback": "train/overlap_fallback",
        "pairwise_pose": "train/pairwise_pose",
        "pairwise_rotation": "train/pairwise_rotation",
        "pairwise_translation_direction": "train/pairwise_translation_direction",
        "pairwise_translation_magnitude": "train/pairwise_translation_magnitude",
        "pairwise_valid_direction_fraction": "train/pairwise_valid_direction_fraction",
        "pairwise_rotation_degrees": "train/pairwise_rotation_degrees",
        "pairwise_translation_direction_degrees": "train/pairwise_translation_direction_degrees",
        "rpa_5": "train/rpa_5",
        "rpa_15": "train/rpa_15",
        "rpa_30": "train/rpa_30",
        "photometric": "train/photometric",
        "photometric_visibility": "train/photometric_visibility",
        "flow": "train/flow",
        "flow_gradient": "train/flow_gradient",
        "flow_objective": "train/flow_objective",
        "ode_steps": "train/ode_steps",
        "temporal_enabled": "train/temporal_enabled",
        "residual_gate": "train/residual_gate",
        "curriculum_stage_index": "train/curriculum_stage_index",
        "gpa_objective": "train/gpa_objective",
        "gpa_physical": "train/gpa_physical",
        "gpa_photometric": "train/gpa_photometric",
        "gpa_structural": "train/gpa_structural",
        "gpa_smoothness": "train/gpa_smoothness",
        "gpa_valid_fraction": "train/gpa_valid_fraction",
        "correspondence_objective": "train/correspondence_objective",
        "correspondence_covisibility": "train/correspondence_covisibility",
        "correspondence_pair_count": "train/correspondence_pair_count",
        "dynamic_objective": "train/dynamic_objective",
        "dynamic_scene_flow": "train/dynamic_scene_flow",
        "dynamic_scene_flow_epe": "train/dynamic_scene_flow_epe",
        "dynamic_cycle": "train/dynamic_cycle",
        "dynamic_reprojection": "train/dynamic_reprojection",
        "dynamic_temporal_depth": "train/dynamic_temporal_depth",
        "dynamic_visibility": "train/dynamic_visibility",
        "dynamic_classification": "train/dynamic_classification",
        "dynamic_spatial": "train/dynamic_spatial",
        "dynamic_temporal_mask": "train/dynamic_temporal_mask",
        "dynamic_area_prior": "train/dynamic_area_prior",
        "dynamic_near_coverage": "train/dynamic_near_coverage",
        "dynamic_teacher_coverage": "train/dynamic_teacher_coverage",
        "dynamic_visibility_precision": "train/dynamic_visibility_precision",
        "dynamic_visibility_recall": "train/dynamic_visibility_recall",
        "dynamic_visibility_f1": "train/dynamic_visibility_f1",
        "dynamic_visibility_iou": "train/dynamic_visibility_iou",
        "dynamic_visibility_positive_count": "train/dynamic_visibility_positive_count",
        "dynamic_visibility_negative_count": "train/dynamic_visibility_negative_count",
        "dynamic_visibility_known_coverage": "train/dynamic_visibility_known_coverage",
        "dynamic_precision": "train/dynamic_precision",
        "dynamic_recall": "train/dynamic_recall",
        "dynamic_f1": "train/dynamic_f1",
        "dynamic_iou": "train/dynamic_iou",
        "dynamic_static_false_positive_rate": "train/dynamic_static_false_positive_rate",
        "dynamic_static_count": "train/dynamic_static_count",
        "dynamic_positive_count": "train/dynamic_positive_count",
        "dynamic_known_coverage": "train/dynamic_known_coverage",
        "dynamic_curriculum_stage_index": "train/dynamic_curriculum_stage_index",
    }
    for metric_name, tag in optional_metrics.items():
        if metric_name in losses:
            scalars[tag] = float(losses[metric_name].detach())
    for index, group in enumerate(optimizer.param_groups[:2]):
        scalars[f"optimizer/group_{index}_lr"] = float(group["lr"])
    beta1 = _optimizer_beta1(optimizer)
    if beta1 is not None:
        scalars["optimizer/beta1"] = beta1
    if device.type == "cuda":
        scalars["system/max_cuda_memory_gib"] = torch.cuda.max_memory_allocated(device) / (1024**3)
    return scalars


def _accumulate_metrics(totals: dict[str, float], losses: Mapping[str, torch.Tensor]) -> None:
    for name, value in losses.items():
        totals[name] = totals.get(name, 0.0) + float(value.detach())


def _mean_metrics(totals: Mapping[str, float], batches: int) -> dict[str, float]:
    if batches < 1:
        raise ValueError("epoch received no batches")
    return {name: value / batches for name, value in totals.items()}


def _validated_depth_thresholds_m(values: Iterable[float]) -> tuple[float, ...]:
    thresholds: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("depth thresholds must be finite positive numbers in strictly increasing order")
        threshold = float(value)
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("depth thresholds must be finite positive numbers in strictly increasing order")
        thresholds.append(threshold)
    if not thresholds or any(left >= right for left, right in pairwise(thresholds)):
        raise ValueError("depth thresholds must be finite positive numbers in strictly increasing order")
    return tuple(thresholds)


def _depth_threshold_label(threshold_m: float) -> str:
    return format(threshold_m, ".9g").replace(".", "p")


def _empty_depth_metric_totals() -> dict[str, float]:
    return {
        "absolute_error_m": 0.0,
        "absolute_relative_error": 0.0,
        "normalized_absolute_error": 0.0,
        "squared_error_m2": 0.0,
        "valid_pixels": 0.0,
    }


def _accumulate_depth_scope(
    totals: dict[str, float],
    *,
    scope_mask: torch.Tensor,
    absolute_error_m: torch.Tensor,
    absolute_relative_error: torch.Tensor,
    normalized_absolute_error: torch.Tensor,
) -> None:
    count = int(scope_mask.sum().item())
    if count == 0:
        return
    totals["valid_pixels"] += count
    totals["absolute_error_m"] += float(absolute_error_m[scope_mask].double().sum())
    totals["squared_error_m2"] += float(absolute_error_m[scope_mask].double().square().sum())
    totals["absolute_relative_error"] += float(absolute_relative_error[scope_mask].double().sum())
    totals["normalized_absolute_error"] += float(normalized_absolute_error[scope_mask].double().sum())


def _accumulate_metric_depth_metrics(
    totals: dict[str, dict[str, float]],
    *,
    predictions: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    thresholds_m: tuple[float, ...],
) -> bool:
    scale = batch.get("normalization_scale_m")
    if scale is None:
        return False
    if not isinstance(scale, torch.Tensor):
        raise TypeError("normalization_scale_m must be a tensor")
    target_depth = cast(torch.Tensor, batch["depths"])
    depth_mask = cast(torch.Tensor, batch["depth_masks"])
    predicted_depth = predictions["depth"][..., 0]
    batch_size = int(target_depth.shape[0])
    if scale.numel() != batch_size:
        raise ValueError("normalization_scale_m must contain exactly one scale per sample")
    scale = scale.to(device=target_depth.device, dtype=torch.float64).reshape(batch_size, 1, 1, 1)
    if not torch.isfinite(scale).all() or torch.any(scale <= 0):
        raise ValueError("normalization_scale_m must contain finite positive values")

    target = target_depth.to(dtype=torch.float64)
    prediction = predicted_depth.to(dtype=torch.float64)
    target_metric_m = target * scale
    normalized_absolute_error = (prediction - target).abs()
    absolute_error_m = normalized_absolute_error * scale
    valid = depth_mask & (target_metric_m > 0)
    absolute_relative_error = absolute_error_m / target_metric_m.clamp_min(torch.finfo(torch.float64).eps)
    _accumulate_depth_scope(
        totals["all"],
        scope_mask=valid,
        absolute_error_m=absolute_error_m,
        absolute_relative_error=absolute_relative_error,
        normalized_absolute_error=normalized_absolute_error,
    )
    for threshold_m in thresholds_m:
        _accumulate_depth_scope(
            totals[_depth_threshold_label(threshold_m)],
            scope_mask=valid & (target_metric_m < threshold_m),
            absolute_error_m=absolute_error_m,
            absolute_relative_error=absolute_relative_error,
            normalized_absolute_error=normalized_absolute_error,
        )
    return True


def _finalize_metric_depth_metrics(
    totals: Mapping[str, Mapping[str, float]], thresholds_m: tuple[float, ...]
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    all_count = float(totals["all"]["valid_pixels"])
    if all_count <= 0:
        raise ValueError("metric depth evaluation received no valid pixels")
    scopes = (("all", "depth_all"),) + tuple(
        (_depth_threshold_label(threshold), f"depth_lt_{_depth_threshold_label(threshold)}m")
        for threshold in thresholds_m
    )
    for scope, prefix in scopes:
        scope_totals = totals[scope]
        count = float(scope_totals["valid_pixels"])
        metrics[f"{prefix}_valid_pixels"] = count
        metrics[f"{prefix}_coverage"] = count / all_count
        if count == 0:
            continue
        metrics[f"{prefix}_mae_m"] = scope_totals["absolute_error_m"] / count
        metrics[f"{prefix}_rmse_m"] = math.sqrt(scope_totals["squared_error_m2"] / count)
        metrics[f"{prefix}_abs_rel"] = scope_totals["absolute_relative_error"] / count
        metrics[f"{prefix}_normalized_l1"] = scope_totals["normalized_absolute_error"] / count
    return metrics


def _take_optimizer_step(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    gradient_clip_norm: float,
) -> float:
    grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        max_norm=gradient_clip_norm,
        error_if_nonfinite=True,
    )
    grad_norm = float(grad_norm_tensor)
    if not math.isfinite(grad_norm):
        raise ValueError(f"non-finite gradient norm: {grad_norm}")
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return grad_norm


def train_one_epoch(
    *,
    model: nn.Module,
    batches: Iterable[Mapping[str, Any]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip_norm: float,
    gradient_accumulation_steps: int,
    min_valid_depth_pixels: int,
    global_step: int,
    logger: ScalarLogger | None,
    log_every_steps: int,
    max_optimizer_steps: int | None = None,
    scheduler: Any = None,
    precision: str = "fp32",
    loss_options: Mapping[str, float] | None = None,
    renderer_options: Mapping[str, object] | None = None,
    pixel_depth_options: Mapping[str, object] | None = None,
    dynamic_geometry_options: Mapping[str, object] | None = None,
    depth_input_options: Mapping[str, object] | None = None,
    flow_generator: torch.Generator | None = None,
    performance_options: Mapping[str, object] | None = None,
) -> TrainEpochResult:
    """Train one epoch; ``global_step`` counts optimizer rather than micro steps."""

    _validate_loop_options(
        gradient_clip_norm=gradient_clip_norm,
        gradient_accumulation_steps=gradient_accumulation_steps,
        log_every_steps=log_every_steps,
        max_optimizer_steps=max_optimizer_steps,
    )
    model.train()
    aggregator = getattr(model, "aggregator", None)
    if isinstance(aggregator, nn.Module):
        aggregator.eval()
    optimizer.zero_grad(set_to_none=True)

    totals: dict[str, float] = {}
    batch_count = 0
    pending_micro_batches = 0
    optimizer_steps = 0
    last_losses: Mapping[str, torch.Tensor] | None = None
    performance_options = dict(performance_options or {})
    profiling_options = performance_options.get("profiling", {})
    contract_options = performance_options.get("runtime_contracts", {})
    if not isinstance(profiling_options, Mapping) or not isinstance(contract_options, Mapping):
        raise ValueError("performance profiling/runtime_contracts must be mappings")
    profiler = StepProfiler(
        enabled=bool(profiling_options.get("enabled", False)),
        warmup_steps=int(profiling_options.get("warmup_steps", 0)),
        active_steps=int(profiling_options.get("active_steps", 1)),
        synchronize=(torch.cuda.synchronize if device.type == "cuda" else lambda: None),
    )

    for raw_batch in batches:
        batch = _move_batch(raw_batch, device)
        images = batch.get("images")
        if not isinstance(images, torch.Tensor):
            raise TypeError("training batch images must be a tensor")
        profiler.batch_ready(sample_count=int(images.shape[0]))
        if bool(contract_options.get("enabled", False)) and (
            not bool(contract_options.get("first_batch_only", True)) or batch_count == 0
        ):
            validate_training_batch_contract(batch)
        with _autocast_context(device, precision):
            if depth_input_options is not None:
                mapped_depth, valid_mask = _depth_input_batch(batch)
                availability = sample_depth_availability(
                    int(images.shape[0]),
                    int(images.shape[1]),
                    seed=cast(int, depth_input_options["seed"]),
                    epoch=cast(int, depth_input_options["epoch"]),
                    optimizer_step=global_step,
                    device=device,
                )
                predictions = model(
                    images,
                    mapped_depth=mapped_depth,
                    valid_mask=valid_mask,
                    availability=availability,
                )
            elif dynamic_geometry_options is not None:
                dynamic_forward = getattr(model, "forward_dynamic", None)
                if not callable(dynamic_forward):
                    raise ValueError("dynamic geometry training requires its wrapper")
                frame_ids = batch.get("frame_ids")
                if not isinstance(frame_ids, torch.Tensor):
                    raise ValueError("dynamic geometry training requires frame_ids")
                frame_mask = batch.get("frame_mask")
                if frame_mask is None:
                    frame_mask = torch.ones_like(frame_ids, dtype=torch.bool)
                predictions = dynamic_forward(
                    batch["images"],
                    frame_ids=frame_ids,
                    frame_mask=frame_mask,
                )
            elif pixel_depth_options is None:
                predictions = model(batch["images"])
            else:
                if flow_generator is None or not callable(getattr(model, "forward_train", None)):
                    raise ValueError("pixel-depth training requires its wrapper and explicit flow generator")
                predictions = model.forward_train(
                    batch["images"],
                    batch["depths"],
                    batch["depth_masks"],
                    generator=flow_generator,
                    dynamic_mask=batch.get("dynamic_masks"),
                    frame_mask=batch.get("frame_mask"),
                    normalization_scale_m=batch.get("normalization_scale_m"),
                )
            losses = compute_camera_depth_loss(
                predictions,
                batch,
                min_valid_depth_pixels=min_valid_depth_pixels,
                renderer_options=renderer_options,
                **dict(loss_options or {}),
            )
            losses = dict(losses)
            if dynamic_geometry_options is not None:
                auxiliary = _dynamic_geometry_losses(predictions, batch, dynamic_geometry_options)
                losses.update(auxiliary)
                losses["objective"] = losses["objective"] + auxiliary["dynamic_objective"]
            if pixel_depth_options is not None and dynamic_geometry_options is None:
                flow_objective = predictions.get("flow_objective")
                if not isinstance(flow_objective, torch.Tensor):
                    raise KeyError("pixel-depth wrapper did not return flow_objective")
                losses["flow"] = predictions["flow"]
                losses["flow_gradient"] = predictions["flow_gradient"]
                losses["flow_objective"] = flow_objective
                losses["ode_steps"] = predictions["ode_steps"]
                losses["temporal_enabled"] = predictions["temporal_enabled"]
                losses["residual_gate"] = predictions["residual_gate"]
                losses["curriculum_stage_index"] = flow_objective.new_tensor(
                    float(pixel_depth_options.get("curriculum_stage_index", 0))
                )
                losses["objective"] = (
                    losses["objective"] + float(pixel_depth_options["objective_weight"]) * flow_objective
                )
                if isinstance(pixel_depth_options.get("gpa"), Mapping) or isinstance(
                    pixel_depth_options.get("correspondence"), Mapping
                ):
                    if flow_generator is None:
                        raise ValueError("self-supervised training requires an explicit generator")
                    auxiliary = _self_supervised_losses(
                        predictions,
                        batch,
                        pixel_depth_options,
                        generator=flow_generator,
                    )
                    losses.update(auxiliary)
                    gpa_options = pixel_depth_options.get("gpa")
                    correspondence_options = pixel_depth_options.get("correspondence")
                    if isinstance(gpa_options, Mapping) and "gpa_objective" in auxiliary:
                        losses["objective"] = (
                            losses["objective"] + float(gpa_options["objective_weight"]) * auxiliary["gpa_objective"]
                        )
                    if isinstance(correspondence_options, Mapping) and "correspondence_objective" in auxiliary:
                        losses["objective"] = (
                            losses["objective"]
                            + float(correspondence_options["objective_weight"]) * auxiliary["correspondence_objective"]
                        )
            sampling_metrics = {
                "sampling_overlap_score": "sample_overlap",
                "sampling_overlap_target": "overlap_target",
                "sampling_overlap_fallback": "overlap_fallback",
            }
            for batch_key, metric_name in sampling_metrics.items():
                if batch_key in batch:
                    value = batch[batch_key]
                    if not isinstance(value, torch.Tensor):
                        raise TypeError(f"{batch_key} must be a tensor")
                    losses[metric_name] = value.float().mean()
        _ensure_finite_loss(losses)
        (losses["objective"] / gradient_accumulation_steps).backward()
        _accumulate_metrics(totals, losses)
        batch_count += 1
        pending_micro_batches += 1
        last_losses = losses

        reached_step_limit = False
        if pending_micro_batches >= gradient_accumulation_steps:
            grad_norm = _take_optimizer_step(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                gradient_clip_norm=gradient_clip_norm,
            )
            pending_micro_batches = 0
            optimizer_steps += 1
            global_step += 1
            if logger is not None and global_step % log_every_steps == 0:
                logger.log_scalars(
                    _training_scalars(losses, grad_norm=grad_norm, optimizer=optimizer, device=device),
                    step=global_step,
                )
            reached_step_limit = max_optimizer_steps is not None and optimizer_steps >= max_optimizer_steps
        profiler.batch_complete()
        if reached_step_limit:
            break

    if pending_micro_batches and (max_optimizer_steps is None or optimizer_steps < max_optimizer_steps):
        # The objective was divided by the configured accumulation factor. Do
        # not rescale a short final group: keeping this smaller step is stable
        # and deterministic across resume-at-epoch-boundary runs.
        assert last_losses is not None
        grad_norm = _take_optimizer_step(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            gradient_clip_norm=gradient_clip_norm,
        )
        optimizer_steps += 1
        global_step += 1
        if logger is not None and global_step % log_every_steps == 0:
            logger.log_scalars(
                _training_scalars(last_losses, grad_norm=grad_norm, optimizer=optimizer, device=device),
                step=global_step,
            )

    metrics = _mean_metrics(totals, batch_count)
    metrics.update(profiler.metrics())
    return TrainEpochResult(
        metrics=metrics,
        global_step=global_step,
        optimizer_steps=optimizer_steps,
        batches=batch_count,
    )


@torch.no_grad()
def validate_one_epoch(
    *,
    model: nn.Module,
    batches: Iterable[Mapping[str, Any]],
    device: torch.device,
    min_valid_depth_pixels: int,
    max_batches: int | None = None,
    precision: str = "fp32",
    loss_options: Mapping[str, float] | None = None,
    renderer_options: Mapping[str, object] | None = None,
    depth_thresholds_m: Iterable[float] = _DEFAULT_DEPTH_EVALUATION_THRESHOLDS_M,
    pixel_depth_options: Mapping[str, object] | None = None,
    dynamic_geometry_options: Mapping[str, object] | None = None,
    depth_input_options: Mapping[str, object] | None = None,
    flow_generator: torch.Generator | None = None,
    performance_options: Mapping[str, object] | None = None,
) -> dict[str, float]:
    """Evaluate one epoch using the model weights currently exposed by the optimizer."""

    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be None or at least 1")
    thresholds_m = _validated_depth_thresholds_m(depth_thresholds_m)
    model.eval()
    totals: dict[str, float] = {}
    metric_depth_totals = {
        "all": _empty_depth_metric_totals(),
        **{_depth_threshold_label(threshold): _empty_depth_metric_totals() for threshold in thresholds_m},
    }
    metric_depth_available: bool | None = None
    batch_count = 0
    performance_options = dict(performance_options or {})
    contract_options = performance_options.get("runtime_contracts", {})
    if not isinstance(contract_options, Mapping):
        raise ValueError("performance.runtime_contracts must be a mapping")
    for raw_batch in batches:
        batch = _move_batch(raw_batch, device)
        if bool(contract_options.get("enabled", False)) and (
            not bool(contract_options.get("first_batch_only", True)) or batch_count == 0
        ):
            validate_training_batch_contract(batch)
        with _autocast_context(device, precision):
            if depth_input_options is not None:
                images = batch.get("images")
                if not isinstance(images, torch.Tensor):
                    raise ValueError("depth-input validation requires images")
                mapped_depth, valid_mask = _depth_input_batch(batch)
                availability = fixed_depth_availability(
                    int(images.shape[0]),
                    int(images.shape[1]),
                    cast(int, depth_input_options["validation_provided_frames"]),
                    device=device,
                )
                predictions = model(
                    images,
                    mapped_depth=mapped_depth,
                    valid_mask=valid_mask,
                    availability=availability,
                )
            elif dynamic_geometry_options is not None:
                dynamic_forward = getattr(model, "forward_dynamic", None)
                if not callable(dynamic_forward):
                    raise ValueError("dynamic geometry validation requires its wrapper")
                frame_ids = batch.get("frame_ids")
                if not isinstance(frame_ids, torch.Tensor):
                    raise ValueError("dynamic geometry validation requires frame_ids")
                frame_mask = batch.get("frame_mask")
                if frame_mask is None:
                    frame_mask = torch.ones_like(frame_ids, dtype=torch.bool)
                predictions = dynamic_forward(
                    batch["images"],
                    frame_ids=frame_ids,
                    frame_mask=frame_mask,
                )
            elif pixel_depth_options is None:
                predictions = model(batch["images"])
            else:
                if flow_generator is None or not callable(getattr(model, "forward_refine", None)):
                    raise ValueError("pixel-depth validation requires its wrapper and explicit flow generator")
                predictions = model.forward_refine(
                    batch["images"],
                    generator=flow_generator,
                    valid_mask=batch.get("depth_masks"),
                    dynamic_mask=batch.get("dynamic_masks"),
                    frame_mask=batch.get("frame_mask"),
                    normalization_scale_m=batch.get("normalization_scale_m"),
                )
            losses = compute_camera_depth_loss(
                predictions,
                batch,
                min_valid_depth_pixels=min_valid_depth_pixels,
                renderer_options=renderer_options,
                **dict(loss_options or {}),
            )
            if depth_input_options is not None:
                losses = {
                    **losses,
                    **_depth_input_validation_metrics(
                        predictions,
                        batch,
                        availability,
                        min_valid_depth_pixels=min_valid_depth_pixels,
                    ),
                }
            if pixel_depth_options is not None:
                pixel_metrics = {
                    **losses,
                    **_pixel_depth_validation_metrics(predictions, batch, pixel_depth_options),
                }
                pixel_metrics["residual_gate"] = predictions["residual_gate"]
                pixel_metrics["curriculum_stage_index"] = predictions["depth"].new_tensor(
                    float(pixel_depth_options.get("curriculum_stage_index", 0))
                )
                if dynamic_geometry_options is None and (
                    isinstance(pixel_depth_options.get("gpa"), Mapping)
                    or isinstance(pixel_depth_options.get("correspondence"), Mapping)
                ):
                    if flow_generator is None:
                        raise ValueError("self-supervised validation requires an explicit generator")
                    pixel_metrics.update(
                        _self_supervised_losses(
                            predictions,
                            batch,
                            pixel_depth_options,
                            generator=flow_generator,
                        )
                    )
                losses = pixel_metrics
            if dynamic_geometry_options is not None:
                dynamic_metrics = _dynamic_geometry_losses(predictions, batch, dynamic_geometry_options)
                losses = {**losses, **dynamic_metrics}
        _ensure_finite_loss(losses)
        _accumulate_metrics(totals, losses)
        has_metric_depth = _accumulate_metric_depth_metrics(
            metric_depth_totals,
            predictions=predictions,
            batch=batch,
            thresholds_m=thresholds_m,
        )
        if metric_depth_available is not None and metric_depth_available != has_metric_depth:
            raise ValueError("normalization_scale_m must be present consistently across validation batches")
        metric_depth_available = has_metric_depth
        batch_count += 1
        if max_batches is not None and batch_count >= max_batches:
            break
    metrics = _mean_metrics(totals, batch_count)
    if dynamic_geometry_options is not None:
        teacher_coverage = metrics.get("dynamic_teacher_coverage")
        if teacher_coverage is None or not math.isfinite(teacher_coverage) or teacher_coverage <= 0:
            raise ValueError("dynamic validation teacher coverage must be finite and positive")
    if metric_depth_available:
        metrics.update(_finalize_metric_depth_metrics(metric_depth_totals, thresholds_m))
    return metrics


def _pixel_depth_validation_metrics(
    predictions: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    options: Mapping[str, object],
) -> dict[str, torch.Tensor]:
    from vggt_omega.training.edge_metrics import edge_3d_error_proxy
    from vggt_omega.training.multiview_metrics import sequence_multiview_consistency
    from vggt_omega.utils.pose_enc import encoding_to_camera

    predicted_depth = predictions["depth"][..., 0].float()
    target_depth = batch["depths"].float()
    valid_mask = batch["depth_masks"].bool()
    normalization_scale = batch["normalization_scale_m"].float()
    intrinsics = batch["intrinsics"].float()
    if predicted_depth.shape != target_depth.shape or normalization_scale.numel() != target_depth.shape[0]:
        raise ValueError("pixel-depth validation tensors have incompatible shapes")
    dynamic_mask = batch.get("dynamic_masks")
    if dynamic_mask is None:
        dynamic_mask = torch.zeros_like(valid_mask)
    static_valid = valid_mask & ~dynamic_mask.bool()
    scale = normalization_scale.reshape(-1, 1, 1, 1)
    predicted_metric = predicted_depth * scale
    target_metric = target_depth * scale
    max_depth_m = float(options["max_depth_m"])
    near_mask = static_valid & (target_metric > 0) & (target_metric < max_depth_m)
    near_mae = (
        (predicted_metric - target_metric).abs()[near_mask].mean()
        if bool(near_mask.any())
        else predicted_metric.sum() * 0
    )
    edge = edge_3d_error_proxy(
        predicted_metric.flatten(0, 1),
        target_metric.flatten(0, 1),
        intrinsics.flatten(0, 1),
        static_valid.flatten(0, 1),
        max_near_depth_m=max_depth_m,
    )
    pose = predictions["pose_enc"].float()
    quaternion = pose[..., 3:7]
    quaternion_norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    identity = torch.zeros_like(quaternion)
    identity[..., 3] = 1
    pose = torch.cat(
        (pose[..., :3], torch.where(quaternion_norm > 1e-8, quaternion, identity), pose[..., 7:]),
        dim=-1,
    )
    extrinsics, predicted_intrinsics = encoding_to_camera(pose, target_depth.shape[-2:], build_intrinsics=True)
    assert predicted_intrinsics is not None
    metric_extrinsics = extrinsics.clone()
    metric_extrinsics[..., :3, 3] *= normalization_scale[:, None, None]
    multiview = sequence_multiview_consistency(
        predicted_metric,
        predicted_intrinsics.float(),
        metric_extrinsics.float(),
        valid_mask=valid_mask,
        dynamic_mask=dynamic_mask.bool(),
        frame_mask=batch.get("frame_mask"),
        max_depth_m=max_depth_m,
    )
    device = predicted_depth.device
    dtype = predicted_depth.dtype

    def scalar(value: float) -> torch.Tensor:
        return torch.tensor(value, device=device, dtype=dtype)

    edge_error = scalar(edge["near_edge_3d_error_proxy"])
    multiview_error = scalar(multiview["symmetric_depth_error"])
    near_edge_objective = (
        near_mae
        + float(options["edge_objective_weight"]) * edge_error
        + float(options["multiview_objective_weight"]) * multiview_error
    )
    return {
        "near_depth_mae_m": near_mae,
        "edge_3d_error_proxy": scalar(edge["all_edge_3d_error_proxy"]),
        "near_edge_3d_error_proxy": edge_error,
        "edge_coverage": scalar(edge["all_edge_coverage"]),
        "near_edge_coverage": scalar(edge["near_edge_coverage"]),
        "multiview_depth_error": multiview_error,
        "multiview_relative_error": scalar(multiview["symmetric_relative_error"]),
        "multiview_coverage": scalar(multiview["symmetric_coverage"]),
        "multiview_pair_count": scalar(multiview["pair_count"]),
        "multiview_visible_direction_count": scalar(multiview["visible_direction_count"]),
        "near_edge_objective": near_edge_objective,
        "ode_steps": scalar(float(options["ode_steps"])),
    }


def _resolve_path(value: str | os.PathLike[str], original_cwd: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (original_cwd / path).resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _base_checkpoint_metadata(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "base_checkpoint": {
            "filename": path.name,
            "size_bytes": stat.st_size,
            "sha256": _sha256_file(path),
        }
    }


def _initialize_head_from_checkpoint(
    checkpoint_path: Path,
    *,
    model: nn.Module,
    trainable_parameter_names: tuple[str, ...],
    expected_base_checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise ValueError("initial head checkpoint must be a regular file")
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError("initial head checkpoint cannot be loaded safely") from error
    if not isinstance(payload, Mapping):
        raise ValueError("initial head checkpoint payload must be a mapping")
    if payload.get("format_version") != 1 or payload.get("kind") not in {"best", "resume"}:
        raise ValueError("initial head checkpoint has an unsupported format or kind")
    if payload.get("parameter_state") != "x":
        raise ValueError("initial head checkpoint must contain AMUSE evaluation weights x")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping) or metadata.get("base_checkpoint") != dict(expected_base_checkpoint):
        raise ValueError("initial head checkpoint base metadata does not match the configured pretrained model")
    state = payload.get("model_state")
    if not isinstance(state, Mapping):
        raise ValueError("initial head checkpoint model_state is missing or invalid")
    _load_trainable_state(model, cast(Mapping[str, torch.Tensor], state), trainable_parameter_names)
    return {
        "filename": checkpoint_path.name,
        "sha256": _sha256_file(checkpoint_path),
        "kind": payload.get("kind"),
        "epoch": payload.get("epoch"),
        "global_step": payload.get("global_step"),
        "parameter_state": "x",
    }


def _resolved_config(cfg: DictConfig) -> dict[str, Any]:
    value = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(value, dict):
        raise TypeError("resolved Hydra training configuration must be a dictionary")
    resolved = cast(dict[str, Any], value)
    trainer = resolved.get("trainer")
    if isinstance(trainer, dict):
        # A resume location is invocation state rather than training identity.
        # Persist one canonical config so pre- and post-resume best checkpoints
        # remain mutually auditable and no local run path leaks into artifacts.
        trainer["resume_from"] = None
    return resolved


def _pixel_depth_config(cfg: DictConfig) -> dict[str, Any] | None:
    if not bool(cfg.pixel_depth.enabled):
        return None
    value = OmegaConf.to_container(cfg.pixel_depth, resolve=True)
    if not isinstance(value, dict):
        raise TypeError("resolved pixel_depth configuration must be a dictionary")
    return cast(dict[str, Any], value)


def _depth_input_config(cfg: DictConfig) -> dict[str, Any] | None:
    if not bool(cfg.depth_input.enabled):
        return None
    value = OmegaConf.to_container(cfg.depth_input, resolve=True)
    if not isinstance(value, dict):
        raise TypeError("resolved depth_input configuration must be a dictionary")
    return cast(dict[str, Any], value)


def _depth_input_batch(batch: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    depths = batch.get("depths")
    masks = batch.get("depth_masks")
    if not isinstance(depths, torch.Tensor) or depths.ndim != 4:
        raise ValueError("depth-input training requires depths with shape [B,S,H,W]")
    if not isinstance(masks, torch.Tensor) or masks.shape != depths.shape or masks.dtype != torch.bool:
        raise ValueError("depth-input training requires bool depth_masks matching depths")
    return depths.unsqueeze(2), masks.unsqueeze(2)


def _depth_input_validation_metrics(
    predictions: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    availability: torch.Tensor,
    *,
    min_valid_depth_pixels: int,
) -> dict[str, torch.Tensor]:
    predicted = predictions.get("depth")
    target = batch.get("depths")
    valid = batch.get("depth_masks")
    if not isinstance(predicted, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise ValueError("depth-input validation requires predicted and target depth")
    if not isinstance(valid, torch.Tensor) or valid.shape != target.shape:
        raise ValueError("depth-input validation requires matching depth_masks")
    if availability.shape != target.shape[:2]:
        raise ValueError("depth-input availability must match target [B,S]")
    provided = availability[:, :, None, None]
    dtype = predicted.dtype
    return {
        "depth_provided": compute_depth_loss(
            predicted, target, valid & provided, min_valid_pixels=min_valid_depth_pixels
        ),
        "depth_unprovided": compute_depth_loss(
            predicted, target, valid & ~provided, min_valid_pixels=min_valid_depth_pixels
        ),
        "provided_frame_count": availability.sum(dim=1).to(dtype=dtype).mean(),
        "unprovided_frame_count": (~availability).sum(dim=1).to(dtype=dtype).mean(),
    }


def _performance_config(cfg: DictConfig) -> dict[str, Any]:
    value = OmegaConf.to_container(cfg.performance, resolve=True)
    if not isinstance(value, dict):
        raise TypeError("resolved performance configuration must be a dictionary")
    return cast(dict[str, Any], value)


def _compile_training_modules(model: nn.Module, options: Mapping[str, object]) -> dict[str, object]:
    enabled = bool(options.get("enabled", False))
    metadata: dict[str, object] = {
        "enabled": enabled,
        "backend": str(options.get("backend", "inductor")),
        "mode": str(options.get("mode", "default")),
        "fullgraph": bool(options.get("fullgraph", False)),
        "dynamic": bool(options.get("dynamic", False)),
        "targets": list(options.get("targets", [])),
    }
    if not enabled:
        return metadata
    for target in metadata["targets"]:
        if not isinstance(target, str):
            raise ValueError("compile targets must be module attribute names")
        module = getattr(model, target, None)
        if not isinstance(module, nn.Module):
            raise ValueError(f"missing compile target module: {target}")
        module.compile(
            backend=metadata["backend"],
            mode=None if metadata["mode"] == "default" else metadata["mode"],
            fullgraph=metadata["fullgraph"],
            dynamic=metadata["dynamic"],
        )
    return metadata


def _pixel_depth_runtime_options(
    config: Mapping[str, Any] | None,
    *,
    epoch: int | None = None,
) -> dict[str, object] | None:
    if config is None:
        return None
    flow = config.get("flow")
    geometry = config.get("geometry")
    if not isinstance(flow, Mapping) or not isinstance(geometry, Mapping):
        raise ValueError("pixel-depth config requires flow and geometry mappings")
    result: dict[str, object] = {**dict(flow), "max_depth_m": float(geometry["max_depth_m"])}
    self_supervised = config.get("self_supervised")
    if self_supervised is None:
        return result
    if not isinstance(self_supervised, Mapping):
        raise ValueError("pixel-depth self_supervised config must be a mapping")
    gpa = self_supervised.get("gpa")
    correspondence = self_supervised.get("correspondence")
    guardrail = self_supervised.get("guardrail")
    curriculum = self_supervised.get("curriculum")
    if (
        not isinstance(gpa, Mapping)
        or not isinstance(correspondence, Mapping)
        or not isinstance(guardrail, Mapping)
        or not isinstance(curriculum, list)
    ):
        raise ValueError("pixel-depth self supervision requires gpa, correspondence, guardrail, and curriculum")
    if epoch is None:
        epoch = 0
    active = [stage for stage in curriculum if isinstance(stage, Mapping) and int(stage["start_epoch"]) <= epoch]
    if not active:
        raise ValueError("pixel-depth self-supervised curriculum has no active stage")
    stage = active[-1]
    stage_index = curriculum.index(stage)
    result["objective_weight"] = float(stage["flow_weight"])
    result["gpa"] = {**dict(gpa), "objective_weight": float(stage["gpa_weight"])}
    result["correspondence"] = {
        **dict(correspondence),
        "objective_weight": float(stage["correspondence_weight"]),
    }
    result["guardrail"] = dict(guardrail)
    result["curriculum_stage_name"] = str(stage["name"])
    result["curriculum_stage_index"] = stage_index
    result["curriculum"] = dict(stage)
    return result


def _guardrail_violations(
    baseline: Mapping[str, float],
    current: Mapping[str, float],
    guardrail: Mapping[str, object],
) -> dict[str, dict[str, float]]:
    """Return lower-is-better metric violations against the accepted baseline."""

    if not bool(guardrail.get("enabled", False)):
        return {}
    metrics = guardrail.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("enabled curriculum guardrail requires metric thresholds")
    violations: dict[str, dict[str, float]] = {}
    for name, raw_thresholds in metrics.items():
        if name not in baseline or name not in current:
            raise ValueError(f"guardrail metric {name!r} is missing")
        baseline_value = float(baseline[name])
        current_value = float(current[name])
        if not math.isfinite(baseline_value) or not math.isfinite(current_value):
            raise ValueError(f"guardrail metric {name!r} must be finite")
        if not isinstance(raw_thresholds, Mapping):
            raise ValueError(f"guardrail metric {name!r} thresholds must be a mapping")
        relative = float(raw_thresholds["max_relative_degradation"])
        absolute = float(raw_thresholds["max_absolute_degradation"])
        allowed = baseline_value + max(absolute, abs(baseline_value) * relative)
        if current_value > allowed:
            violations[str(name)] = {
                "baseline": baseline_value,
                "current": current_value,
                "allowed": allowed,
                "excess": current_value - allowed,
            }
    return violations


def _guardrail_ablation(stage_name: str) -> str:
    choices = {
        "residual_gate": "reduce_flow_weight",
        "gpa_warmup": "set_gpa_weight_to_zero",
        "correspondence_head": "set_correspondence_weight_to_zero",
        "joint_low_lr": "freeze_base_heads",
        "near_depth_recovery": "set_flow_weight_to_zero",
    }
    return choices.get(stage_name, "hold_previous_accepted_checkpoint")


def _self_supervised_losses(
    predictions: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    options: Mapping[str, object],
    *,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    from vggt_omega.training.correspondence import (
        build_rgbd_correspondence_targets,
        masked_generalized_charbonnier,
    )
    from vggt_omega.training.gpa_loss import gpa_sequence_loss, sample_gpa_anchor_indices
    from vggt_omega.utils.pose_enc import encoding_to_camera

    gpa_options = options.get("gpa")
    correspondence_options = options.get("correspondence")
    if not isinstance(gpa_options, Mapping) or not isinstance(correspondence_options, Mapping):
        return {}
    gpa_active = bool(gpa_options.get("enabled", False)) and float(gpa_options.get("objective_weight", 0.0)) > 0
    correspondence_active = (
        bool(correspondence_options.get("enabled", False))
        and float(correspondence_options.get("objective_weight", 0.0)) > 0
    )
    if not gpa_active and not correspondence_active:
        return {}
    images = cast(torch.Tensor, batch["images"]).float()
    target_depth = cast(torch.Tensor, batch["depths"]).float()
    depth_mask = cast(torch.Tensor, batch["depth_masks"]).bool()
    intrinsics = cast(torch.Tensor, batch["intrinsics"]).float()
    target_extrinsics = cast(torch.Tensor, batch["extrinsics"]).float()
    predicted_depth = predictions["depth"][..., 0].float()
    batch_size, frame_count = target_depth.shape[:2]
    dynamic_mask = batch.get("dynamic_masks")
    if dynamic_mask is None:
        dynamic_mask = torch.zeros_like(depth_mask)
    dynamic_mask = cast(torch.Tensor, dynamic_mask).bool()
    frame_mask = batch.get("frame_mask")
    if frame_mask is None:
        frame_mask = torch.ones((batch_size, frame_count), dtype=torch.bool, device=images.device)
    frame_mask = cast(torch.Tensor, frame_mask).bool()
    normalization_scale = batch.get("normalization_scale_m")
    if not isinstance(normalization_scale, torch.Tensor) or normalization_scale.numel() != batch_size:
        raise ValueError("self-supervised near masking requires one normalization_scale_m per sample")
    metric_target = target_depth * normalization_scale.float().reshape(batch_size, 1, 1, 1)
    near_valid = depth_mask & ~dynamic_mask & (metric_target > 0) & (metric_target < float(options["max_depth_m"]))
    losses: dict[str, torch.Tensor] = {}

    if gpa_active:
        anchors = sample_gpa_anchor_indices(
            frame_mask,
            anchor_count=int(gpa_options["anchor_count"]),
            generator=generator,
        )
        predicted_extrinsics, _ = encoding_to_camera(
            predictions["pose_enc"].float(),
            images.shape[-2:],
            build_intrinsics=False,
        )
        gpa = gpa_sequence_loss(
            images,
            predicted_depth,
            intrinsics,
            predicted_extrinsics.float(),
            valid_mask=near_valid,
            anchor_indices=anchors,
            mu=float(gpa_options["mu"]),
            lambda_geo=float(gpa_options["lambda_geo"]),
            lambda_smooth=float(gpa_options["lambda_smooth"]),
            auto_mask_delta=float(gpa_options["auto_mask_delta"]),
            geometry_epsilon=float(gpa_options["geometry_epsilon"]),
            dynamic_mask=dynamic_mask,
            frame_mask=frame_mask,
            auto_mask_enabled=bool(gpa_options["auto_mask_enabled"]),
            mask_mode=str(gpa_options["mask_mode"]),
        )
        losses.update(
            {
                "gpa_objective": gpa["objective"],
                "gpa_physical": gpa["physical"],
                "gpa_photometric": gpa["photometric"],
                "gpa_structural": gpa["structural"],
                "gpa_smoothness": gpa["smoothness"],
                "gpa_valid_fraction": gpa["valid_fraction"],
            }
        )

    if correspondence_active:
        predicted_flow = predictions.get("correspondence_flow_pixels")
        pair_indices = predictions.get("correspondence_pair_indices")
        if not isinstance(predicted_flow, torch.Tensor) or not isinstance(pair_indices, torch.Tensor):
            raise KeyError("enabled correspondence training requires wrapper flow and pair outputs")
        targets = build_rgbd_correspondence_targets(
            target_depth,
            intrinsics,
            target_extrinsics,
            pair_indices,
            valid_mask=near_valid,
            dynamic_mask=dynamic_mask,
            frame_mask=frame_mask,
            relative_depth_tolerance=float(correspondence_options["relative_depth_tolerance"]),
        )
        correspondence = masked_generalized_charbonnier(
            predicted_flow.float(),
            targets["flow_pixels"],
            targets["covisibility_mask"],
            alpha=float(correspondence_options["alpha"]),
            epsilon=float(correspondence_options["epsilon"]),
        )
        losses.update(
            {
                "correspondence_objective": correspondence,
                "correspondence_covisibility": targets["covisibility_mask"].float().mean(),
                "correspondence_pair_count": predicted_flow.new_tensor(float(pair_indices.shape[1])),
            }
        )
    return losses


def _dynamic_geometry_config(cfg: DictConfig) -> dict[str, Any] | None:
    if not bool(cfg.dynamic_geometry.enabled):
        return None
    value = OmegaConf.to_container(cfg.dynamic_geometry, resolve=True)
    if not isinstance(value, dict):
        raise TypeError("resolved dynamic_geometry configuration must be a dictionary")
    return cast(dict[str, Any], value)


def _dynamic_geometry_runtime_options(
    config: Mapping[str, Any] | None,
    *,
    epoch: int = 0,
) -> dict[str, object] | None:
    if config is None:
        return None
    curriculum = config.get("curriculum")
    if not isinstance(curriculum, list):
        raise ValueError("dynamic geometry config requires a curriculum list")
    active = [stage for stage in curriculum if isinstance(stage, Mapping) and int(stage["start_epoch"]) <= epoch]
    if not active:
        raise ValueError("dynamic geometry curriculum has no active stage")
    stage = active[-1]
    result = dict(config)
    result["curriculum"] = dict(stage)
    result["curriculum_stage_index"] = curriculum.index(stage)
    result["curriculum_stage_name"] = str(stage["name"])
    return result


def _dynamic_guardrail_options(options: Mapping[str, object] | None) -> dict[str, object] | None:
    if options is None:
        return None
    raw = options.get("guardrail")
    if not isinstance(raw, Mapping):
        raise ValueError("dynamic geometry requires guardrail thresholds")
    return {
        "enabled": True,
        "metrics": {
            "depth_lt_1p2m_mae_m": {
                "max_relative_degradation": 0.0,
                "max_absolute_degradation": float(raw["max_near_depth_mae_m_degradation"]),
            },
            "camera_translation": {
                "max_relative_degradation": 0.0,
                "max_absolute_degradation": float(raw["max_camera_translation_degradation"]),
            },
            "objective": {
                "max_relative_degradation": 0.0,
                "max_absolute_degradation": float(raw["max_objective_degradation"]),
            },
        },
    }


def _apply_dynamic_curriculum_stage(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    options: Mapping[str, object],
    *,
    base_learning_rates: tuple[float, ...],
) -> bool:
    stage = options.get("curriculum")
    if not isinstance(stage, Mapping):
        raise ValueError("dynamic geometry runtime options require a curriculum stage")
    setter = getattr(model, "set_dynamic_stage", None)
    if not callable(setter):
        raise ValueError("dynamic geometry curriculum requires a stage-aware wrapper")
    setter(str(stage["dynamic_stage"]))
    if len(optimizer.param_groups) != len(base_learning_rates):
        raise ValueError("optimizer parameter-group count changed after dynamic curriculum initialization")
    scale = float(stage["learning_rate_scale"])
    _set_optimizer_stage_learning_rate_scale(optimizer, base_learning_rates, scale)
    return bool(stage["train_enabled"])


def _set_optimizer_stage_learning_rate_scale(
    optimizer: torch.optim.Optimizer,
    base_learning_rates: tuple[float, ...],
    scale: float,
) -> None:
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("curriculum learning-rate scale must be finite and non-negative")
    if len(optimizer.param_groups) != len(base_learning_rates):
        raise ValueError("optimizer parameter-group count changed after curriculum initialization")
    for group, base_learning_rate in zip(optimizer.param_groups, base_learning_rates, strict=True):
        group["lr"] = base_learning_rate * scale
        if "base_lr" in group:
            group["base_lr"] = base_learning_rate * scale


def _dynamic_geometry_losses(
    predictions: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    options: Mapping[str, object],
) -> dict[str, torch.Tensor]:
    """Build conservative near-range 4D supervision and collapse guards."""

    from vggt_omega.training.dynamic_geometry import build_rgbd_motion_targets
    from vggt_omega.training.dynamic_losses import (
        bounded_self_supervised_area_prior,
        confidence_weighted_scene_flow_regression,
        dynamic_temporal_consistency_loss,
        edge_aware_dynamic_spatial_consistency_loss,
        multi_view_reprojection_loss,
        source_grid_forward_backward_3d_cycle_loss,
        temporal_target_depth_consistency_loss,
        tri_state_binary_cross_entropy,
    )

    geometry_options = options.get("geometry")
    pseudo_options = options.get("pseudo_labels")
    loss_options = options.get("loss")
    if not all(isinstance(item, Mapping) for item in (geometry_options, pseudo_options, loss_options)):
        raise ValueError("dynamic geometry requires geometry, pseudo_labels, and loss mappings")
    geometry_options = cast(Mapping[str, object], geometry_options)
    pseudo_options = cast(Mapping[str, object], pseudo_options)
    loss_options = cast(Mapping[str, object], loss_options)
    depths = cast(torch.Tensor, batch["depths"]).float()
    depth_masks = cast(torch.Tensor, batch["depth_masks"]).bool()
    intrinsics = cast(torch.Tensor, batch["intrinsics"]).float()
    extrinsics = cast(torch.Tensor, batch["extrinsics"]).float()
    frame_ids = cast(torch.Tensor, batch["frame_ids"]).long()
    batch_size, frame_count, height, width = depths.shape
    frame_mask_value = batch.get("frame_mask")
    frame_mask = (
        torch.ones((batch_size, frame_count), dtype=torch.bool, device=depths.device)
        if frame_mask_value is None
        else cast(torch.Tensor, frame_mask_value).bool()
    )
    normalization_scale = cast(torch.Tensor, batch["normalization_scale_m"]).float().reshape(batch_size)
    original_observed = batch.get("original_depth_observed_mask")
    if not isinstance(original_observed, torch.Tensor):
        raise ValueError("dynamic geometry requires original_depth_observed_mask from staging schema v2")
    original_observed = original_observed.bool()
    pair_indices = predictions["motion_pair_indices"]
    pixel_flow = batch.get("motion_pixel_flow_xy")
    flow_confidence = batch.get("motion_flow_confidence")
    if (pixel_flow is None) != (flow_confidence is None):
        raise ValueError("motion pixel flow and confidence must be provided together")
    if isinstance(pixel_flow, torch.Tensor):
        pixel_flow = pixel_flow.detach().float()
        flow_confidence = cast(torch.Tensor, flow_confidence).detach().float()
    targets = build_rgbd_motion_targets(
        depths,
        depth_masks,
        original_observed,
        intrinsics,
        extrinsics,
        normalization_scale,
        frame_ids,
        frame_mask,
        pair_indices,
        pixel_flow,
        flow_confidence,
        flow_occlusion_label=cast(torch.Tensor | None, batch.get("motion_flow_occlusion_label")),
        static_off_m=float(pseudo_options["static_off_m"]),
        dynamic_on_m=float(pseudo_options["dynamic_on_m"]),
        flow_confidence_min=float(pseudo_options["flow_confidence_min"]),
        forward_backward_cycle_px=float(pseudo_options["forward_backward_cycle_px"]),
        depth_discontinuity_relative=float(pseudo_options["depth_discontinuity_relative"]),
    )
    safe_flow = (
        torch.zeros((batch_size, pair_indices.shape[1], height, width, 2), device=depths.device)
        if pixel_flow is None
        else pixel_flow
    )
    pair_valid = predictions["motion_pair_valid_mask"]
    safe_pairs = pair_indices.clamp_min(0)
    batch_indices = torch.arange(batch_size, device=depths.device)[:, None]
    source_indices = safe_pairs[..., 0]
    target_indices = safe_pairs[..., 1]
    source_metric_depth = depths[batch_indices, source_indices] * normalization_scale[:, None, None, None]
    near_domain = (
        predictions["motion_domain_mask"]
        & (source_metric_depth > 0)
        & (source_metric_depth < float(geometry_options["max_depth_m"]))
    )
    confidence = targets["target_confidence"] * near_domain.float()
    visibility_known = targets["target_visibility_known_mask"] & near_domain
    dynamic_known = (targets["target_dynamic_label"] >= 0) & near_domain
    target_depth_grid = depths[batch_indices, target_indices]
    target_valid_grid = depth_masks[batch_indices, target_indices] & pair_valid[:, :, None, None]
    source_images = cast(torch.Tensor, batch["images"])[batch_indices, source_indices].float()
    target_extrinsics = predictions["rebased_extrinsics_w2c"][batch_indices, target_indices]
    target_intrinsics = predictions["predicted_intrinsics"][batch_indices, target_indices]
    rows, columns = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=depths.device),
        torch.arange(width, dtype=torch.float32, device=depths.device),
        indexing="ij",
    )
    target_pixels = torch.stack((columns, rows), dim=-1)[None, None] + safe_flow
    scene_flow = confidence_weighted_scene_flow_regression(
        predictions["canonical_scene_flow"],
        targets["target_canonical_scene_flow"],
        confidence,
        near_domain,
        alpha=float(loss_options["charbonnier_alpha"]),
        epsilon=float(loss_options["charbonnier_epsilon"]),
    )
    cycle = source_grid_forward_backward_3d_cycle_loss(
        predictions["canonical_scene_flow"],
        safe_flow,
        pair_indices,
        pair_valid,
        domain_mask=near_domain,
        confidence=confidence,
        missing_reverse_policy="unknown",
    )
    reprojection = multi_view_reprojection_loss(
        predictions["canonical_points_at_target_time"],
        target_extrinsics,
        target_intrinsics,
        target_pixels,
        confidence,
        near_domain & (targets["target_visibility_label"] == 1),
    )
    temporal_depth = temporal_target_depth_consistency_loss(
        predictions["depth_at_target_time_in_target_camera"][..., 0],
        target_depth_grid,
        safe_flow,
        confidence,
        source_valid_mask=near_domain,
        target_valid_mask=target_valid_grid,
        epsilon=float(loss_options["charbonnier_epsilon"]),
    )
    visibility = tri_state_binary_cross_entropy(
        predictions["motion_visibility_logits"],
        targets["target_visibility_label"],
        known_mask=visibility_known,
        domain_mask=near_domain,
        confidence=targets["target_visibility_confidence"] * near_domain.float(),
    )
    dynamic = tri_state_binary_cross_entropy(
        predictions["dynamic_logits"],
        targets["target_dynamic_label"],
        known_mask=dynamic_known,
        domain_mask=near_domain,
        confidence=confidence,
    )
    spatial = edge_aware_dynamic_spatial_consistency_loss(
        predictions["dynamic_probability"], source_images, near_domain, edge_scale=10.0
    )
    temporal_mask = dynamic_temporal_consistency_loss(
        predictions["dynamic_probability"],
        safe_flow,
        pair_indices,
        pair_valid,
        domain_mask=near_domain,
        confidence=confidence,
        missing_reverse_policy="unknown",
    )
    area_prior = bounded_self_supervised_area_prior(
        predictions["dynamic_probability"],
        near_domain & (confidence > 0),
        minimum_fraction=float(loss_options["area_lower"]),
        maximum_fraction=float(loss_options["area_upper"]),
    )
    components = {
        "dynamic_scene_flow": scene_flow,
        "dynamic_cycle": cycle,
        "dynamic_reprojection": reprojection,
        "dynamic_temporal_depth": temporal_depth,
        "dynamic_visibility": visibility,
        "dynamic_classification": dynamic,
        "dynamic_spatial": spatial,
        "dynamic_temporal_mask": temporal_mask,
        "dynamic_area_prior": area_prior,
    }
    weights = {
        "dynamic_scene_flow": "scene_flow_weight",
        "dynamic_cycle": "cycle_weight",
        "dynamic_reprojection": "reprojection_weight",
        "dynamic_temporal_depth": "temporal_depth_weight",
        "dynamic_visibility": "visibility_weight",
        "dynamic_classification": "dynamic_weight",
        "dynamic_spatial": "spatial_weight",
        "dynamic_temporal_mask": "temporal_mask_weight",
        "dynamic_area_prior": "area_prior_weight",
    }
    objective = sum(
        (float(loss_options[weight_key]) * components[name] for name, weight_key in weights.items()),
        scene_flow.new_zeros(()),
    )
    visibility_prediction = predictions["motion_visibility_probability"] >= float(options["visibility_threshold"])
    dynamic_prediction = predictions["dynamic_probability"] >= float(options["dynamic_probability_min"])
    visibility_positive = visibility_known & (targets["target_visibility_label"] == 1)
    visibility_negative = visibility_known & (targets["target_visibility_label"] == 0)
    dynamic_positive = dynamic_known & (targets["target_dynamic_label"] == 1)
    dynamic_static = dynamic_known & (targets["target_dynamic_label"] == 0)
    visibility_predicted_positive = visibility_known & visibility_prediction
    dynamic_predicted_positive = dynamic_known & dynamic_prediction

    def ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        return numerator.float() / denominator.float().clamp_min(1)

    def classification_metrics(
        predicted_positive: torch.Tensor,
        target_positive: torch.Tensor,
        target_negative: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        true_positive = (predicted_positive & target_positive).sum()
        false_positive = (predicted_positive & target_negative).sum()
        false_negative = (~predicted_positive & target_positive).sum()
        precision = ratio(true_positive, true_positive + false_positive)
        recall = ratio(true_positive, true_positive + false_negative)
        f1 = ratio(2 * true_positive, 2 * true_positive + false_positive + false_negative)
        iou = ratio(true_positive, true_positive + false_positive + false_negative)
        return precision, recall, f1, iou

    visibility_known_count = visibility_known.sum()
    dynamic_known_count = dynamic_known.sum()
    near_count = near_domain.sum()
    visibility_precision, visibility_recall, visibility_f1, visibility_iou = classification_metrics(
        visibility_predicted_positive,
        visibility_positive,
        visibility_negative,
    )
    dynamic_precision, dynamic_recall, dynamic_f1, dynamic_iou = classification_metrics(
        dynamic_predicted_positive,
        dynamic_positive,
        dynamic_static,
    )
    epe_active = near_domain & (confidence > 0)
    safe_flow_error = torch.where(
        epe_active.unsqueeze(-1),
        predictions["canonical_scene_flow"].float() - targets["target_canonical_scene_flow"],
        torch.zeros_like(targets["target_canonical_scene_flow"]),
    )
    epe_map = torch.linalg.vector_norm(safe_flow_error, dim=-1)
    scene_flow_epe = (epe_map * confidence).sum() / confidence.sum().clamp_min(1)
    return {
        **components,
        "dynamic_objective": objective,
        "dynamic_scene_flow_epe": scene_flow_epe,
        "dynamic_near_coverage": near_domain.float().mean(),
        "dynamic_teacher_coverage": (confidence > 0).float().mean(),
        "dynamic_visibility_positive_count": visibility_positive.sum().float(),
        "dynamic_visibility_negative_count": visibility_negative.sum().float(),
        "dynamic_visibility_known_coverage": ratio(visibility_known_count, near_count),
        "dynamic_visibility_precision": visibility_precision,
        "dynamic_visibility_recall": visibility_recall,
        "dynamic_visibility_f1": visibility_f1,
        "dynamic_visibility_iou": visibility_iou,
        "dynamic_static_count": dynamic_static.sum().float(),
        "dynamic_positive_count": dynamic_positive.sum().float(),
        "dynamic_known_coverage": ratio(dynamic_known_count, near_count),
        "dynamic_precision": dynamic_precision,
        "dynamic_recall": dynamic_recall,
        "dynamic_f1": dynamic_f1,
        "dynamic_iou": dynamic_iou,
        "dynamic_static_false_positive_rate": ratio(
            (dynamic_predicted_positive & dynamic_static).sum(),
            dynamic_static.sum(),
        ),
        "dynamic_curriculum_stage_index": objective.new_tensor(float(options.get("curriculum_stage_index", 0))),
    }


def _dynamic_readiness_passed(
    metrics: Mapping[str, torch.Tensor | float],
    options: Mapping[str, object],
) -> bool:
    readiness = options.get("readiness")
    if not isinstance(readiness, Mapping):
        raise ValueError("dynamic geometry requires readiness thresholds")
    requirements = {
        "dynamic_visibility_positive_count": "min_visibility_positive_count",
        "dynamic_visibility_negative_count": "min_visibility_negative_count",
        "dynamic_visibility_known_coverage": "min_visibility_known_coverage",
        "dynamic_visibility_precision": "min_visibility_precision",
        "dynamic_static_count": "min_dynamic_static_count",
        "dynamic_positive_count": "min_dynamic_positive_count",
        "dynamic_known_coverage": "min_dynamic_known_coverage",
        "dynamic_precision": "min_dynamic_precision",
        "dynamic_recall": "min_dynamic_recall",
    }

    def scalar(value: torch.Tensor | float) -> float:
        return float(value.detach()) if isinstance(value, torch.Tensor) else float(value)

    return all(scalar(metrics[metric]) >= float(readiness[threshold]) for metric, threshold in requirements.items())


def _flow_generator(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def _apply_pixel_curriculum_stage(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    options: Mapping[str, object],
    *,
    base_learning_rates: tuple[float, ...],
) -> bool:
    stage = options.get("curriculum")
    if not isinstance(stage, Mapping):
        return True
    setter = getattr(model, "set_curriculum_trainable", None)
    if not callable(setter):
        raise ValueError("pixel-depth curriculum requires a stage-aware wrapper")
    setter(
        train_refiner=bool(stage["train_refiner"]),
        train_correspondence=bool(stage["train_correspondence"]),
        train_base_heads=bool(stage["train_base_heads"]),
    )
    if len(optimizer.param_groups) != len(base_learning_rates):
        raise ValueError("optimizer parameter-group count changed after curriculum initialization")
    scale = float(stage["learning_rate_scale"])
    _set_optimizer_stage_learning_rate_scale(optimizer, base_learning_rates, scale)
    return bool(stage["train_enabled"])


def _select_trainable_state(
    model: nn.Module,
    trainable_parameter_names: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    state = model.state_dict()
    missing = set(trainable_parameter_names) - set(state)
    if missing:
        raise ValueError(f"trainable parameter state is missing keys: {sorted(missing)}")
    return {name: state[name].detach().clone() for name in trainable_parameter_names}


def _load_trainable_state(
    model: nn.Module,
    state: Mapping[str, torch.Tensor],
    trainable_parameter_names: tuple[str, ...],
) -> None:
    expected = set(trainable_parameter_names)
    actual = set(state)
    if actual != expected:
        raise ValueError(
            "resume head state does not exactly match trainable parameters: "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )
    destinations = {**dict(model.named_parameters()), **dict(model.named_buffers())}
    absent_state = expected - set(destinations)
    if absent_state:
        raise ValueError(f"resume state names are not model parameters or buffers: {sorted(absent_state)}")
    with torch.no_grad():
        for name in trainable_parameter_names:
            source = state[name]
            destination = destinations[name]
            if source.shape != destination.shape:
                raise ValueError(
                    f"resume parameter shape mismatch for {name}: {tuple(source.shape)} != {tuple(destination.shape)}"
                )
            if source.dtype != destination.dtype:
                raise ValueError(f"resume parameter dtype mismatch for {name}: {source.dtype} != {destination.dtype}")
            destination.copy_(source.to(device=destination.device))


def _adamw_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    scheduler_name: str,
    total_optimizer_steps: int,
    warmup_ratio: float,
) -> Any:
    if scheduler_name == "none":
        return None
    warmup_steps = max(1, math.ceil(total_optimizer_steps * warmup_ratio))

    def lr_multiplier(step_index: int) -> float:
        current = step_index + 1
        if current <= warmup_steps:
            return current / warmup_steps
        if scheduler_name == "constant":
            return 1.0
        progress = min(1.0, (current - warmup_steps) / max(1, total_optimizer_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_multiplier)


def _build_optimizer_from_config(
    cfg: DictConfig,
    model: nn.Module,
    *,
    total_optimizer_steps: int,
) -> _RuntimeOptimizer:
    name = str(cfg.optimizer.name)
    if name == "amuse":
        result = build_amuse_optimizer(
            model,
            total_optimizer_steps=total_optimizer_steps,
            muon_lr=float(cfg.optimizer.muon_lr),
            aux_lr=float(cfg.optimizer.aux_lr),
            aux_update_type=str(cfg.optimizer.aux_update_type),
            beta1=float(cfg.optimizer.beta1),
            beta2=float(cfg.optimizer.beta2),
            momentum=float(cfg.optimizer.momentum),
            rho=float(cfg.optimizer.rho),
            r=float(cfg.optimizer.r),
            weight_lr_power=float(cfg.optimizer.weight_lr_power),
            warmup_ratio=float(cfg.optimizer.warmup_ratio),
            weight_decay=float(cfg.optimizer.weight_decay),
            weight_decay_at_y=float(cfg.optimizer.weight_decay_at_y),
            external_scheduler=None,
            distributed_strategy=str(cfg.trainer.strategy),
        )
        return _RuntimeOptimizer(result.optimizer, result.scheduler, result.group_fingerprint)
    if name == "adamw":
        beta_values = [float(value) for value in cfg.optimizer.betas]
        if len(beta_values) != 2:
            raise ValueError("optimizer.betas must contain exactly two values")
        result = build_adamw_optimizer(
            model,
            lr=float(cfg.optimizer.lr),
            betas=(beta_values[0], beta_values[1]),
            eps=float(cfg.optimizer.eps),
            weight_decay=float(cfg.optimizer.weight_decay),
        )
        scheduler = _adamw_scheduler(
            result.optimizer,
            scheduler_name=str(cfg.optimizer.scheduler),
            total_optimizer_steps=total_optimizer_steps,
            warmup_ratio=float(cfg.optimizer.warmup_ratio),
        )
        return _RuntimeOptimizer(result.optimizer, scheduler, result.group_fingerprint)
    raise ValueError(f"unsupported optimizer: {name!r}")


def _make_loader(
    dataset: ColmapRgbdDataset,
    cfg: DictConfig,
    *,
    epoch: int,
    training: bool,
) -> DataLoader:
    dataset.set_epoch(epoch if training else 0)
    generator = torch.Generator().manual_seed(int(cfg.seed) + (epoch if training else 0))
    num_workers = int(cfg.data.num_workers)
    loader_options: dict[str, object] = {}
    if num_workers > 0:
        loader_options = {
            "prefetch_factor": int(cfg.performance.data_loader.prefetch_factor),
            "persistent_workers": bool(cfg.performance.data_loader.persistent_workers),
        }
    return DataLoader(
        dataset,
        batch_size=int(cfg.data.batch_size),
        shuffle=training,
        num_workers=num_workers,
        pin_memory=bool(cfg.data.pin_memory),
        drop_last=False,
        generator=generator,
        **loader_options,
    )


def _total_optimizer_steps(cfg: DictConfig, dataset_length: int) -> int:
    batches_per_epoch = math.ceil(dataset_length / int(cfg.data.batch_size))
    steps_per_epoch = math.ceil(batches_per_epoch / int(cfg.trainer.gradient_accumulation_steps))
    train_epochs = int(cfg.trainer.epochs)
    pixel_config = _pixel_depth_config(cfg)
    dynamic_config = _dynamic_geometry_config(cfg)
    if dynamic_config is not None:
        train_epochs = sum(
            bool(
                cast(
                    Mapping[str, object], _dynamic_geometry_runtime_options(dynamic_config, epoch=epoch)["curriculum"]
                )["train_enabled"]
            )
            for epoch in range(train_epochs)
        )
    elif pixel_config is not None:
        train_epochs = sum(
            bool(
                cast(Mapping[str, object], _pixel_depth_runtime_options(pixel_config, epoch=epoch)["curriculum"])[
                    "train_enabled"
                ]
            )
            for epoch in range(train_epochs)
        )
    planned = steps_per_epoch * train_epochs
    max_steps = cfg.trainer.max_train_steps
    return max(1, min(planned, int(max_steps)) if max_steps is not None else planned)


def _enter_optimizer_training_mode(optimizer: torch.optim.Optimizer) -> None:
    train = getattr(optimizer, "train", None)
    if callable(train):
        train()


def _atomic_json(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _existing_completed_metrics(
    summary_path: Path,
    *,
    epochs_completed: int,
    global_step: int,
    group_fingerprint: str,
    base_checkpoint: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, float]]:
    if not summary_path.is_file():
        return {}, {}
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}, {}
    if (
        not isinstance(payload, dict)
        or payload.get("status") != "complete"
        or payload.get("epochs_completed") != epochs_completed
        or payload.get("global_step") != global_step
        or payload.get("group_fingerprint") != group_fingerprint
        or payload.get("base_checkpoint") != dict(base_checkpoint)
    ):
        return {}, {}

    return _finite_metric_mapping(payload.get("train")), _finite_metric_mapping(payload.get("validation"))


def _finite_metric_mapping(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    metrics: dict[str, float] = {}
    for key, raw_value in value.items():
        if not isinstance(key, str) or isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            return {}
        scalar = float(raw_value)
        if not math.isfinite(scalar):
            return {}
        metrics[key] = scalar
    return metrics


def _metrics_from_training_state(
    training_state: Mapping[str, Any] | None,
) -> tuple[dict[str, float], dict[str, float]]:
    if training_state is None:
        return {}, {}
    return (
        _finite_metric_mapping(training_state.get("latest_train")),
        _finite_metric_mapping(training_state.get("latest_validation")),
    )


def _validate_resume_config(stored: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    stored_copy = json.loads(json.dumps(dict(stored)))
    current_copy = json.loads(json.dumps(dict(current)))
    for config in (stored_copy, current_copy):
        trainer = config.get("trainer")
        if isinstance(trainer, dict):
            trainer["resume_from"] = None
            trainer.setdefault("early_stopping", json.loads(json.dumps(_STANDARD_EARLY_STOPPING_CONFIG)))
        model = config.get("model")
        if isinstance(model, dict):
            model.setdefault("initial_head_checkpoint", None)
        config.setdefault("loss", json.loads(json.dumps(_STANDARD_LOSS_CONFIG)))
    if stored_copy != current_copy:
        raise ValueError("resume checkpoint configuration does not match the current resolved configuration")


def _dataset_frame_range(cfg: DictConfig) -> tuple[int, int]:
    forced = cfg.trainer.sequence_frames
    if forced is not None:
        return int(forced), int(forced)
    return int(cfg.data.min_frames), int(cfg.data.max_frames)


def _runtime_metadata(device: torch.device, started_at: float) -> dict[str, float]:
    metadata = {"elapsed_seconds": time.perf_counter() - started_at}
    if device.type == "cuda":
        metadata["max_cuda_memory_gib"] = torch.cuda.max_memory_allocated(device) / (1024**3)
        metadata["max_cuda_memory_reserved_gib"] = torch.cuda.max_memory_reserved(device) / (1024**3)
    return metadata


def run_training(
    cfg: DictConfig,
    *,
    output_dir: str | os.PathLike[str],
    original_cwd: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Build the configured pipeline and run all requested epochs."""

    validate_training_config(cfg)
    if str(cfg.trainer.strategy) != "single":
        raise ValueError("the initial executable runner currently supports trainer.strategy=single only")
    cwd = Path(original_cwd) if original_cwd is not None else Path.cwd()
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    data_root = _resolve_path(str(cfg.data.root), cwd)
    checkpoint_path = _resolve_path(str(cfg.model.pretrained_checkpoint), cwd)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"pretrained checkpoint not found: {checkpoint_path}")
    device = torch.device(str(cfg.trainer.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training was requested but CUDA is unavailable")

    seed = int(cfg.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    torch.use_deterministic_algorithms(bool(cfg.trainer.deterministic), warn_only=True)

    min_frames, max_frames = _dataset_frame_range(cfg)
    dynamic_config = _dynamic_geometry_config(cfg)
    depth_input_config = _depth_input_config(cfg)
    is_smoke = str(cfg.trainer.name) == "smoke"
    train_split = str(cfg.data.smoke_split if is_smoke else cfg.data.train_split)
    val_split = str(cfg.data.smoke_split if is_smoke else cfg.data.val_split)
    dataset_options = {
        "min_frames": min_frames,
        "max_frames": max_frames,
        "filter_short_sequences": min_frames == max_frames,
        "seed": seed,
        "min_valid_depth_pixels": int(cfg.data.min_valid_depth_pixels),
        "overlap_metric": str(cfg.data.overlap_curriculum.metric),
        "overlap_start_target": float(cfg.data.overlap_curriculum.start_target),
        "overlap_end_target": float(cfg.data.overlap_curriculum.end_target),
        "overlap_target_tolerance": float(cfg.data.overlap_curriculum.target_tolerance),
        "overlap_curriculum_epochs": int(cfg.data.overlap_curriculum.epochs),
    }
    if dynamic_config is not None:
        pseudo_labels = dynamic_config.get("pseudo_labels")
        if not isinstance(pseudo_labels, Mapping):
            raise ValueError("dynamic geometry config requires pseudo_labels")
        dataset_options["flow_teacher_manifest"] = str(pseudo_labels["teacher_artifact_manifest"])
    train_dataset = ColmapRgbdDataset(
        data_root,
        split=train_split,
        overlap_curriculum_enabled=bool(cfg.data.overlap_curriculum.enabled),
        **dataset_options,
    )
    val_dataset = ColmapRgbdDataset(
        data_root,
        split=val_split,
        overlap_curriculum_enabled=False,
        **dataset_options,
    )

    prepared: PreparedTrainingModel = build_training_model(checkpoint_path, device=device)
    model = prepared.model
    base_metadata = _base_checkpoint_metadata(checkpoint_path)
    initial_head = cfg.model.initial_head_checkpoint
    if initial_head is not None:
        base_metadata["initial_head_checkpoint"] = _initialize_head_from_checkpoint(
            _resolve_path(str(initial_head), cwd),
            model=model,
            trainable_parameter_names=prepared.trainable_parameter_names,
            expected_base_checkpoint=base_metadata["base_checkpoint"],
        )
    if depth_input_config is not None:
        prepared = attach_depth_input_model(prepared, depth_input_config, device=device)
        model = prepared.model
    pixel_config = _pixel_depth_config(cfg)
    pixel_runtime_options = _pixel_depth_runtime_options(pixel_config)
    if pixel_config is not None:
        prepared = attach_pixel_depth_model(prepared, pixel_config, device=device)
        model = prepared.model
    dynamic_runtime_options = _dynamic_geometry_runtime_options(dynamic_config)
    if dynamic_config is not None:
        prepared = attach_dynamic_geometry_model(prepared, dynamic_config, device=device)
        model = prepared.model
    performance_config = _performance_config(cfg)
    compile_options = performance_config.get("compile")
    if not isinstance(compile_options, Mapping):
        raise ValueError("performance.compile must be a mapping")
    compile_metadata = _compile_training_modules(model, cast(Mapping[str, object], compile_options))
    flow_generator = (
        _flow_generator(device, seed + int(cfg.pixel_depth.flow.seed_offset)) if pixel_config is not None else None
    )
    total_steps = _total_optimizer_steps(cfg, len(train_dataset))
    optimizer_bundle = _build_optimizer_from_config(cfg, model, total_optimizer_steps=total_steps)
    optimizer = optimizer_bundle.optimizer
    scheduler = optimizer_bundle.scheduler
    base_learning_rates = tuple(float(group["lr"]) for group in optimizer.param_groups)
    resolved_config = _resolved_config(cfg)

    def state_selector(module: nn.Module) -> dict[str, torch.Tensor]:
        return _select_trainable_state(module, prepared.trainable_parameter_names)

    def state_loader(module: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
        _load_trainable_state(module, state, prepared.trainable_parameter_names)

    checkpoint_manager = TopKCheckpointManager(
        output_root / str(cfg.checkpoint.directory),
        k=int(cfg.checkpoint.k),
        monitor=str(cfg.checkpoint.monitor),
        mode=str(cfg.checkpoint.mode),
        save_last=bool(cfg.checkpoint.save_last),
    )
    start_epoch = 0
    global_step = 0
    resume_training_state: dict[str, Any] | None = None
    resume_from = cfg.trainer.resume_from
    if resume_from:
        resume_state = load_resume_checkpoint(
            _resolve_path(str(resume_from), cwd),
            model=model,
            optimizer=optimizer,
            expected_group_fingerprint=optimizer_bundle.group_fingerprint,
            state_loader=state_loader,
            map_location=device,
        )
        _validate_resume_config(resume_state.config, resolved_config)
        if resume_state.metadata != base_metadata:
            raise ValueError("resume checkpoint base metadata does not match the configured pretrained checkpoint")
        start_epoch = resume_state.epoch + 1
        global_step = resume_state.global_step
        resume_training_state = resume_state.training_state
        if pixel_config is not None:
            if resume_training_state is None or not isinstance(
                resume_training_state.get("flow_rng_state"), torch.Tensor
            ):
                raise ValueError("pixel-depth resume checkpoint is missing flow_rng_state")
            assert flow_generator is not None
            flow_generator.set_state(resume_training_state["flow_rng_state"].cpu())
    else:
        _enter_optimizer_training_mode(optimizer)

    # Persist only after every resume contract has succeeded so a rejected
    # invocation cannot alter the configuration of an existing valid run.
    _atomic_json(resolved_config, output_root / "resolved_config.json")

    logger = TensorBoardScalarLogger(
        output_root / str(cfg.logging.directory),
        enabled=bool(cfg.logging.enabled),
        rank=0,
    )
    epochs_completed = start_epoch
    early_config = cfg.trainer.early_stopping
    early_enabled = bool(early_config.enabled)
    early_monitor = str(early_config.monitor)
    early_mode = str(early_config.mode)
    early_patience = int(early_config.patience)
    early_min_delta = float(early_config.min_delta)
    early_best, early_bad_epochs, stopped_early = (
        _restore_early_stopping_state(
            resume_training_state,
            enabled=early_enabled,
            monitor=early_monitor,
            mode=early_mode,
            patience=early_patience,
            min_delta=early_min_delta,
        )
        if resume_from
        else (None, 0, False)
    )
    latest_train, latest_val = _existing_completed_metrics(
        output_root / "run_summary.json",
        epochs_completed=start_epoch,
        global_step=global_step,
        group_fingerprint=optimizer_bundle.group_fingerprint,
        base_checkpoint=base_metadata["base_checkpoint"],
    )
    if resume_from and (not latest_train or not latest_val):
        restored_train, restored_val = _metrics_from_training_state(resume_training_state)
        latest_train = latest_train or restored_train
        latest_val = latest_val or restored_val
    guardrail_baseline = (
        _finite_metric_mapping(resume_training_state.get("guardrail_baseline"))
        if resume_training_state is not None
        else {}
    )
    guardrail_event: dict[str, Any] | None = None
    initial_guardrail = (
        _dynamic_guardrail_options(dynamic_runtime_options)
        if dynamic_runtime_options is not None
        else None
        if pixel_runtime_options is None
        else pixel_runtime_options.get("guardrail")
    )
    guardrail_enabled = isinstance(initial_guardrail, Mapping) and bool(initial_guardrail.get("enabled", False))
    if resume_from and start_epoch > 0 and guardrail_enabled and not guardrail_baseline:
        raise ValueError("guarded curriculum resume checkpoint is missing guardrail_baseline")
    started_at = time.perf_counter()

    def early_stopping_payload() -> dict[str, Any]:
        return {
            "enabled": early_enabled,
            "monitor": early_monitor,
            "mode": early_mode,
            "patience": early_patience,
            "min_delta": early_min_delta,
            "best": early_best,
            "bad_epochs": early_bad_epochs,
            "stopped": stopped_early,
        }

    def progress_payload(status: str, *, exception_type: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": status,
            "epochs_completed": epochs_completed,
            "global_step": global_step,
            "train": latest_train,
            "validation": latest_val,
            "best": list(checkpoint_manager.entries),
            "base_checkpoint": base_metadata["base_checkpoint"],
            "initial_head_checkpoint": base_metadata.get("initial_head_checkpoint"),
            "early_stopping": early_stopping_payload(),
            "guardrail": {
                "enabled": guardrail_enabled,
                "baseline": guardrail_baseline,
                "event": guardrail_event,
            },
            "performance": {"compile": compile_metadata},
            "group_fingerprint": optimizer_bundle.group_fingerprint,
            **_runtime_metadata(device, started_at),
        }
        if exception_type is not None:
            payload["exception_type"] = exception_type
        return payload

    _atomic_json(progress_payload("running"), output_root / "progress.json")
    try:
        for epoch in range(start_epoch, int(cfg.trainer.epochs)):
            if stopped_early:
                break
            remaining_steps = None
            if cfg.trainer.max_train_steps is not None:
                remaining_steps = int(cfg.trainer.max_train_steps) - global_step
                if remaining_steps <= 0:
                    break
            pixel_runtime_options = _pixel_depth_runtime_options(pixel_config, epoch=epoch)
            dynamic_runtime_options = _dynamic_geometry_runtime_options(dynamic_config, epoch=epoch)
            depth_input_runtime_options = (
                None
                if depth_input_config is None
                else {
                    **depth_input_config,
                    "seed": seed + int(depth_input_config["seed_offset"]),
                    "epoch": epoch,
                }
            )
            train_enabled = (
                True
                if pixel_runtime_options is None and dynamic_runtime_options is None
                else _apply_dynamic_curriculum_stage(
                    model,
                    optimizer,
                    cast(Mapping[str, object], dynamic_runtime_options),
                    base_learning_rates=base_learning_rates,
                )
                if dynamic_runtime_options is not None
                else _apply_pixel_curriculum_stage(
                    model,
                    optimizer,
                    pixel_runtime_options,
                    base_learning_rates=base_learning_rates,
                )
            )
            if train_enabled:
                train_loader = _make_loader(train_dataset, cfg, epoch=epoch, training=True)
                train_result = train_one_epoch(
                    model=model,
                    batches=train_loader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=device,
                    gradient_clip_norm=float(cfg.trainer.gradient_clip_norm),
                    gradient_accumulation_steps=int(cfg.trainer.gradient_accumulation_steps),
                    min_valid_depth_pixels=int(cfg.data.min_valid_depth_pixels),
                    global_step=global_step,
                    logger=logger,
                    log_every_steps=int(cfg.trainer.log_every_steps),
                    max_optimizer_steps=remaining_steps,
                    precision=str(cfg.model.precision),
                    loss_options=_training_loss_options(cfg, epoch),
                    renderer_options=_renderer_options(cfg.renderer),
                    pixel_depth_options=pixel_runtime_options,
                    dynamic_geometry_options=dynamic_runtime_options,
                    depth_input_options=depth_input_runtime_options,
                    flow_generator=flow_generator,
                    performance_options=performance_config,
                )
            else:
                assert pixel_runtime_options is not None
                gate = getattr(getattr(model, "residual_gate", None), "forward", None)
                gate_value = float(gate().detach()) if callable(gate) else 0.0
                train_result = TrainEpochResult(
                    metrics={
                        "curriculum_stage_index": float(pixel_runtime_options["curriculum_stage_index"]),
                        "residual_gate": gate_value,
                    },
                    global_step=global_step,
                    optimizer_steps=0,
                    batches=0,
                )
            global_step = train_result.global_step
            latest_train = train_result.metrics
            performance_tags = {
                "profile_warmup_seconds": "train/profile_warmup_seconds",
                "profile_step_time_seconds": "train/profile_step_time_seconds",
                "profile_samples_per_second": "train/profile_samples_per_second",
                "profile_data_wait_fraction": "train/profile_data_wait_fraction",
                "profile_active_steps": "train/profile_active_steps",
            }
            logger.log_scalars(
                {tag: latest_train[name] for name, tag in performance_tags.items() if name in latest_train},
                step=global_step,
            )

            if (epoch + 1) % int(cfg.trainer.validate_every_epochs) == 0:
                val_loader = _make_loader(val_dataset, cfg, epoch=0, training=False)
                with optimizer_evaluation_state(optimizer):
                    latest_val = validate_one_epoch(
                        model=model,
                        batches=val_loader,
                        device=device,
                        min_valid_depth_pixels=int(cfg.data.min_valid_depth_pixels),
                        max_batches=(None if cfg.trainer.max_val_batches is None else int(cfg.trainer.max_val_batches)),
                        precision=str(cfg.model.precision),
                        loss_options=_loss_options(cfg.loss.validation),
                        renderer_options=_renderer_options(cfg.renderer),
                        pixel_depth_options=pixel_runtime_options,
                        dynamic_geometry_options=dynamic_runtime_options,
                        depth_input_options=depth_input_runtime_options,
                        flow_generator=(
                            None
                            if pixel_config is None
                            else _flow_generator(device, seed + int(cfg.pixel_depth.flow.seed_offset) + 1)
                        ),
                        performance_options=performance_config,
                    )
                    if dynamic_runtime_options is not None and _dynamic_readiness_passed(
                        latest_val,
                        dynamic_runtime_options,
                    ):
                        readiness_setter = getattr(model, "set_dynamic_geometry_ready", None)
                        if not callable(readiness_setter):
                            raise ValueError("dynamic geometry wrapper is missing its readiness setter")
                        readiness_setter(True)
                    validation_scalars = {f"val/{name}": value for name, value in latest_val.items()}
                    logger.log_scalars(validation_scalars, step=global_step)
                    guardrail_options = (
                        _dynamic_guardrail_options(dynamic_runtime_options)
                        if dynamic_runtime_options is not None
                        else None
                        if pixel_runtime_options is None
                        else pixel_runtime_options.get("guardrail")
                    )
                    stage_index = (
                        0
                        if pixel_runtime_options is None and dynamic_runtime_options is None
                        else int(
                            cast(Mapping[str, object], dynamic_runtime_options or pixel_runtime_options).get(
                                "curriculum_stage_index", 0
                            )
                        )
                    )
                    if isinstance(guardrail_options, Mapping) and bool(guardrail_options.get("enabled", False)):
                        guardrail_metrics = guardrail_options.get("metrics")
                        assert isinstance(guardrail_metrics, Mapping)
                        if stage_index == 0:
                            guardrail_baseline = {str(name): float(latest_val[str(name)]) for name in guardrail_metrics}
                        else:
                            violations = _guardrail_violations(
                                guardrail_baseline,
                                latest_val,
                                guardrail_options,
                            )
                            if violations:
                                attempted_global_step = global_step
                                stage_name = str(
                                    cast(Mapping[str, object], dynamic_runtime_options or pixel_runtime_options)[
                                        "curriculum_stage_name"
                                    ]
                                )
                                guardrail_event = {
                                    "triggered": True,
                                    "rejected_epoch": epoch,
                                    "rejected_global_step": attempted_global_step,
                                    "stage_index": stage_index,
                                    "stage_name": stage_name,
                                    "violations": violations,
                                    "recommended_single_variable_ablation": _guardrail_ablation(stage_name),
                                }
                                logger.log_scalars(
                                    {
                                        "val/guardrail_triggered": 1.0,
                                        **{
                                            f"val/guardrail_{name}_excess": values["excess"]
                                            for name, values in violations.items()
                                        },
                                    },
                                    step=attempted_global_step,
                                )
                                rollback = load_resume_checkpoint(
                                    checkpoint_manager.last_path,
                                    model=model,
                                    optimizer=optimizer,
                                    expected_group_fingerprint=optimizer_bundle.group_fingerprint,
                                    state_loader=state_loader,
                                    map_location=device,
                                )
                                if rollback.config != resolved_config or rollback.metadata != base_metadata:
                                    raise ValueError(
                                        "guardrail rollback checkpoint provenance does not match current run"
                                    )
                                if rollback.training_state is None:
                                    raise ValueError("guardrail rollback checkpoint is missing training state")
                                latest_train, latest_val = _metrics_from_training_state(rollback.training_state)
                                global_step = rollback.global_step
                                epochs_completed = rollback.epoch + 1
                                if flow_generator is not None:
                                    rollback_rng = rollback.training_state.get("flow_rng_state")
                                    if not isinstance(rollback_rng, torch.Tensor):
                                        raise ValueError("guardrail rollback checkpoint is missing flow_rng_state")
                                    flow_generator.set_state(rollback_rng.cpu())
                    if guardrail_event is None and early_enabled:
                        monitored_value = float(validation_scalars[early_monitor])
                        if _metric_improved(
                            monitored_value,
                            early_best,
                            mode=early_mode,
                            min_delta=early_min_delta,
                        ):
                            early_best = monitored_value
                            early_bad_epochs = 0
                        else:
                            early_bad_epochs += 1
                            stopped_early = early_bad_epochs >= early_patience
                    if guardrail_event is None:
                        checkpoint_manager.update(
                            epoch=epoch,
                            global_step=global_step,
                            metrics=validation_scalars,
                            model=model,
                            optimizer=optimizer,
                            state_selector=state_selector,
                            group_fingerprint=optimizer_bundle.group_fingerprint,
                            config=resolved_config,
                            metadata=base_metadata,
                            training_state={
                                "early_stopping": early_stopping_payload(),
                                "latest_train": latest_train,
                                "latest_validation": latest_val,
                                "guardrail_baseline": guardrail_baseline,
                                **(
                                    {"flow_rng_state": flow_generator.get_state().cpu()}
                                    if flow_generator is not None
                                    else {}
                                ),
                            },
                        )
                logger.flush()
            if guardrail_event is not None:
                _atomic_json(progress_payload("guardrail_stopped"), output_root / "progress.json")
                break
            epochs_completed = epoch + 1
            _atomic_json(progress_payload("running"), output_root / "progress.json")
            if stopped_early:
                break
    except Exception as error:
        _atomic_json(
            progress_payload("failed", exception_type=type(error).__name__),
            output_root / "progress.json",
        )
        raise
    finally:
        logger.close()

    summary = {
        "status": "guardrail_stopped" if guardrail_event is not None else "complete",
        "epochs_completed": epochs_completed,
        "global_step": global_step,
        "train": latest_train,
        "validation": latest_val,
        "best": list(checkpoint_manager.entries),
        "base_checkpoint": base_metadata["base_checkpoint"],
        "initial_head_checkpoint": base_metadata.get("initial_head_checkpoint"),
        "early_stopping": early_stopping_payload(),
        "stopped_early": stopped_early,
        "guardrail": {
            "enabled": guardrail_enabled,
            "baseline": guardrail_baseline,
            "event": guardrail_event,
        },
        "performance": {"compile": compile_metadata},
        "group_fingerprint": optimizer_bundle.group_fingerprint,
        **_runtime_metadata(device, started_at),
    }
    _atomic_json(summary, output_root / "run_summary.json")
    _atomic_json(summary, output_root / "progress.json")
    return summary
