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
from vggt_omega.training.losses import compute_camera_depth_loss
from vggt_omega.training.model_factory import PreparedTrainingModel, build_training_model
from vggt_omega.training.optimizer_factory import build_adamw_optimizer, build_amuse_optimizer
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

    for raw_batch in batches:
        batch = _move_batch(raw_batch, device)
        with _autocast_context(device, precision):
            predictions = model(batch["images"])
            losses = compute_camera_depth_loss(
                predictions,
                batch,
                min_valid_depth_pixels=min_valid_depth_pixels,
                **dict(loss_options or {}),
            )
        _ensure_finite_loss(losses)
        (losses["objective"] / gradient_accumulation_steps).backward()
        _accumulate_metrics(totals, losses)
        batch_count += 1
        pending_micro_batches += 1
        last_losses = losses

        if pending_micro_batches < gradient_accumulation_steps:
            continue

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
        if max_optimizer_steps is not None and optimizer_steps >= max_optimizer_steps:
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

    return TrainEpochResult(
        metrics=_mean_metrics(totals, batch_count),
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
    depth_thresholds_m: Iterable[float] = _DEFAULT_DEPTH_EVALUATION_THRESHOLDS_M,
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
    for raw_batch in batches:
        batch = _move_batch(raw_batch, device)
        with _autocast_context(device, precision):
            predictions = model(batch["images"])
            losses = compute_camera_depth_loss(
                predictions,
                batch,
                min_valid_depth_pixels=min_valid_depth_pixels,
                **dict(loss_options or {}),
            )
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
    if metric_depth_available:
        metrics.update(_finalize_metric_depth_metrics(metric_depth_totals, thresholds_m))
    return metrics


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


def _select_trainable_state(
    model: nn.Module,
    trainable_parameter_names: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    state = model.state_dict()
    missing = set(trainable_parameter_names) - set(state)
    if missing:
        raise ValueError(f"trainable parameter state is missing keys: {sorted(missing)}")
    return {name: state[name] for name in trainable_parameter_names}


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
    incompatible = model.load_state_dict(dict(state), strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(f"resume head state has unexpected keys: {incompatible.unexpected_keys}")


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
    return DataLoader(
        dataset,
        batch_size=int(cfg.data.batch_size),
        shuffle=training,
        num_workers=int(cfg.data.num_workers),
        pin_memory=bool(cfg.data.pin_memory),
        drop_last=False,
        generator=generator,
    )


def _total_optimizer_steps(cfg: DictConfig, dataset_length: int) -> int:
    batches_per_epoch = math.ceil(dataset_length / int(cfg.data.batch_size))
    steps_per_epoch = math.ceil(batches_per_epoch / int(cfg.trainer.gradient_accumulation_steps))
    planned = steps_per_epoch * int(cfg.trainer.epochs)
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
    is_smoke = str(cfg.trainer.name) == "smoke"
    train_split = str(cfg.data.smoke_split if is_smoke else cfg.data.train_split)
    val_split = str(cfg.data.smoke_split if is_smoke else cfg.data.val_split)
    dataset_options = {
        "min_frames": min_frames,
        "max_frames": max_frames,
        "seed": seed,
        "min_valid_depth_pixels": int(cfg.data.min_valid_depth_pixels),
    }
    train_dataset = ColmapRgbdDataset(data_root, split=train_split, **dataset_options)
    val_dataset = ColmapRgbdDataset(data_root, split=val_split, **dataset_options)

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
    total_steps = _total_optimizer_steps(cfg, len(train_dataset))
    optimizer_bundle = _build_optimizer_from_config(cfg, model, total_optimizer_steps=total_steps)
    optimizer = optimizer_bundle.optimizer
    scheduler = optimizer_bundle.scheduler
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
            )
            global_step = train_result.global_step
            latest_train = train_result.metrics

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
                    )
                    validation_scalars = {f"val/{name}": value for name, value in latest_val.items()}
                    logger.log_scalars(validation_scalars, step=global_step)
                    if early_enabled:
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
                        },
                    )
                logger.flush()
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
        "status": "complete",
        "epochs_completed": epochs_completed,
        "global_step": global_step,
        "train": latest_train,
        "validation": latest_val,
        "best": list(checkpoint_manager.entries),
        "base_checkpoint": base_metadata["base_checkpoint"],
        "initial_head_checkpoint": base_metadata.get("initial_head_checkpoint"),
        "early_stopping": early_stopping_payload(),
        "stopped_early": stopped_early,
        "group_fingerprint": optimizer_bundle.group_fingerprint,
        **_runtime_metadata(device, started_at),
    }
    _atomic_json(summary, output_root / "run_summary.json")
    _atomic_json(summary, output_root / "progress.json")
    return summary
