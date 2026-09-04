"""Validation helpers for the Hydra training configuration."""

from __future__ import annotations

import math
from collections.abc import Mapping
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
_VALIDATION_MONITORS = {
    "val/objective",
    "val/camera",
    "val/camera_translation",
    "val/camera_rotation",
    "val/camera_fov",
    "val/depth",
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
    if float(cfg.trainer.gradient_clip_norm) <= 0:
        raise ValueError("trainer.gradient_clip_norm must be positive")
    if int(cfg.trainer.epochs) < 1:
        raise ValueError("trainer.epochs must be at least 1")

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
