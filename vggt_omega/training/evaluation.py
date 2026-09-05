"""Strict, deterministic re-evaluation of saved top-K training checkpoints."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Protocol, cast

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from vggt_omega.training.dataset import ColmapRgbdDataset
from vggt_omega.training.depth_input_evaluation import (
    DepthSufficientStatistics,
    all_depth_availability_cases,
    build_input_depth_holdout,
    depth_sufficient_statistics,
    merge_depth_statistics,
    metric_result,
)
from vggt_omega.training.losses import compute_camera_loss, compute_pairwise_pose_loss
from vggt_omega.training.model_factory import (
    PreparedTrainingModel,
    attach_depth_input_model,
    attach_dynamic_geometry_model,
    attach_pixel_depth_model,
    build_training_model,
)
from vggt_omega.training.runner import (
    _autocast_context,
    _dynamic_geometry_runtime_options,
    _move_batch,
    _renderer_options,
    validate_one_epoch,
)

ModelFactory = Callable[..., PreparedTrainingModel]
Validator = Callable[..., Mapping[str, float]]

_DIGEST_LENGTH = 64
_MONITOR_TO_METRIC = {
    "val/objective": "objective",
    "val/camera": "camera",
    "val/camera_translation": "camera_translation",
    "val/camera_rotation": "camera_rotation",
    "val/camera_fov": "camera_fov",
    "val/depth": "depth",
    "val/pairwise_pose": "pairwise_pose",
    "val/pairwise_rotation_degrees": "pairwise_rotation_degrees",
    "val/pairwise_translation_direction_degrees": "pairwise_translation_direction_degrees",
    "val/pairwise_translation_magnitude": "pairwise_translation_magnitude",
    "val/rpa_5": "rpa_5",
    "val/rpa_15": "rpa_15",
    "val/rpa_30": "rpa_30",
    "val/near_edge_objective": "near_edge_objective",
    "val/dynamic_classification": "dynamic_classification",
}
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
_STANDARD_EARLY_STOPPING_CONFIG = {
    "enabled": False,
    "monitor": "val/objective",
    "mode": "min",
    "patience": 2,
    "min_delta": 0.0,
}


class EvaluationError(ValueError):
    """Raised when a completed run cannot be reproduced safely and exactly."""


class EvaluationDataset(Protocol):
    def __getitem__(self, index: int) -> Any: ...

    def __len__(self) -> int: ...

    def set_epoch(self, epoch: int) -> None: ...


DatasetFactory = Callable[..., EvaluationDataset]


def _read_json_object(path: Path, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise EvaluationError(f"{name} must be a regular file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvaluationError(f"{name} is not readable JSON") from error
    if not isinstance(payload, dict):
        raise EvaluationError(f"{name} must contain a JSON object")
    return payload


def _mapping(owner: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = owner.get(key)
    if not isinstance(value, Mapping):
        raise EvaluationError(f"{key} configuration must be an object")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvaluationError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: object, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result < 1:
        raise EvaluationError(f"{name} must be positive")
    return result


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise EvaluationError(f"{name} must be a finite number")
    return result


def _digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _DIGEST_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise EvaluationError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_config_path(value: object, original_cwd: Path, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise EvaluationError(f"{name} must be a non-empty path string")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (original_cwd / path).resolve()


def _safe_run_child(run_dir: Path, value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise EvaluationError(f"{name} must be a non-empty relative path")
    relative = Path(value)
    candidate = (run_dir / relative).resolve(strict=False)
    if relative.is_absolute() or candidate == run_dir or not candidate.is_relative_to(run_dir):
        raise EvaluationError(f"{name} must remain inside the run directory")
    return candidate


def _actual_base_metadata(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise EvaluationError("configured base checkpoint must be a regular file")
    stat = path.stat()
    return {"filename": path.name, "sha256": _sha256_file(path), "size_bytes": stat.st_size}


def _validate_base_metadata(value: object, actual: Mapping[str, Any], owner: str) -> None:
    if not isinstance(value, Mapping) or dict(value) != dict(actual):
        raise EvaluationError(f"{owner} base checkpoint metadata does not match the configured base checkpoint")


def _validated_initial_head_metadata(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise EvaluationError("summary initial head metadata must be an object")
    metadata = dict(value)
    filename = metadata.get("filename")
    if not isinstance(filename, str) or not filename or Path(filename).name != filename:
        raise EvaluationError("summary initial head filename must be a basename")
    _digest(metadata.get("sha256"), "summary initial head sha256")
    if metadata.get("kind") not in {"best", "resume"}:
        raise EvaluationError("summary initial head kind is invalid")
    _nonnegative_int(metadata.get("epoch"), "summary initial head epoch")
    _nonnegative_int(metadata.get("global_step"), "summary initial head global_step")
    if metadata.get("parameter_state") != "x":
        raise EvaluationError("summary initial head parameter state must be x")
    return metadata


def _validate_completed_summary(
    summary: Mapping[str, Any],
    config: Mapping[str, Any],
    actual_base: Mapping[str, Any],
) -> tuple[int, int, str, dict[str, Any]]:
    if summary.get("status") != "complete":
        raise EvaluationError("run summary status must be complete")
    completed_epochs = _nonnegative_int(summary.get("epochs_completed"), "summary epochs_completed")
    global_step = _nonnegative_int(summary.get("global_step"), "summary global_step")
    for metric_group in ("train", "validation"):
        raw_metrics = summary.get(metric_group)
        if not isinstance(raw_metrics, Mapping) or not raw_metrics:
            raise EvaluationError(f"summary {metric_group} metrics must be a non-empty object")
        for key, value in raw_metrics.items():
            if not isinstance(key, str) or not key:
                raise EvaluationError(f"summary {metric_group} metric names must be non-empty")
            _finite_float(value, f"summary {metric_group} metric {key}")
    trainer = _mapping(config, "trainer")
    configured_epochs = _positive_int(trainer.get("epochs"), "configured epochs")
    stopped_early = summary.get("stopped_early", False)
    if not isinstance(stopped_early, bool):
        raise EvaluationError("summary stopped_early must be boolean")
    raw_max_train_steps = trainer.get("max_train_steps")
    max_train_steps = (
        None if raw_max_train_steps is None else _positive_int(raw_max_train_steps, "configured max_train_steps")
    )
    if max_train_steps is not None and global_step > max_train_steps:
        raise EvaluationError("summary global_step exceeds configured max_train_steps")
    completed_by_step_limit = (
        max_train_steps is not None and global_step == max_train_steps and 0 < completed_epochs <= configured_epochs
    )
    if (not stopped_early and completed_epochs != configured_epochs and not completed_by_step_limit) or (
        stopped_early and not 0 < completed_epochs <= configured_epochs
    ):
        raise EvaluationError("run summary does not cover the expected configured epochs")
    raw_early_config = trainer.get("early_stopping")
    if raw_early_config is None:
        early_config: Mapping[str, Any] = _STANDARD_EARLY_STOPPING_CONFIG
    elif isinstance(raw_early_config, Mapping):
        early_config = raw_early_config
    else:
        raise EvaluationError("early_stopping configuration must be an object")
    raw_early_summary = summary.get("early_stopping")
    if raw_early_summary is None and not bool(early_config.get("enabled")) and not stopped_early:
        raw_early_summary = {
            **early_config,
            "best": None,
            "bad_epochs": 0,
            "stopped": False,
        }
    if not isinstance(raw_early_summary, Mapping):
        raise EvaluationError("summary early_stopping must be an object")
    early_summary = dict(raw_early_summary)
    enabled = early_config.get("enabled")
    monitor = early_config.get("monitor")
    mode = early_config.get("mode")
    patience = _positive_int(early_config.get("patience"), "early-stopping patience")
    min_delta = _finite_float(early_config.get("min_delta"), "early-stopping min_delta")
    if not isinstance(enabled, bool) or mode not in {"min", "max"} or not isinstance(monitor, str) or not monitor:
        raise EvaluationError("early-stopping configuration is invalid")
    if min_delta < 0:
        raise EvaluationError("early-stopping min_delta must be non-negative")
    expected_early = {
        "enabled": enabled,
        "monitor": monitor,
        "mode": mode,
        "patience": patience,
        "min_delta": min_delta,
    }
    if any(early_summary.get(key) != value for key, value in expected_early.items()):
        raise EvaluationError("summary early-stopping configuration does not match resolved config")
    best = early_summary.get("best")
    if best is not None:
        _finite_float(best, "summary early-stopping best")
    bad_epochs = _nonnegative_int(early_summary.get("bad_epochs"), "summary early-stopping bad_epochs")
    stopped = early_summary.get("stopped")
    if not isinstance(stopped, bool) or stopped != stopped_early:
        raise EvaluationError("summary early-stopping stopped flags are inconsistent")
    if enabled:
        checkpoint_config = _mapping(config, "checkpoint")
        if monitor != checkpoint_config.get("monitor") or mode != checkpoint_config.get("mode"):
            raise EvaluationError("early-stopping monitor/mode does not match checkpoint ranking")
        if completed_epochs > 0 and best is None:
            raise EvaluationError("enabled early stopping requires a best metric after validation")
        if stopped != (bad_epochs >= patience):
            raise EvaluationError("summary early-stopping patience state is inconsistent")
    elif best is not None or bad_epochs != 0 or stopped:
        raise EvaluationError("disabled early stopping has unexpected runtime state")
    group_fingerprint = _digest(summary.get("group_fingerprint"), "summary group_fingerprint")
    _validate_base_metadata(summary.get("base_checkpoint"), actual_base, "summary")
    return completed_epochs, global_step, group_fingerprint, early_summary


def _validate_last_checkpoint(
    path: Path,
    *,
    completed_epochs: int,
    global_step: int,
    group_fingerprint: str,
    resolved_config: Mapping[str, Any],
    actual_base: Mapping[str, Any],
    expected_initial_head: Mapping[str, Any] | None,
    expected_early_stopping: Mapping[str, Any],
) -> None:
    if path.is_symlink() or not path.is_file():
        raise EvaluationError("configured last checkpoint must be a regular file")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise EvaluationError("last checkpoint is unreadable") from error
    if not isinstance(payload, Mapping):
        raise EvaluationError("last checkpoint payload must be an object")
    if (
        payload.get("format_version") != 1
        or payload.get("kind") != "resume"
        or payload.get("checkpoint_role") != "last"
        or payload.get("parameter_state") != "x"
    ):
        raise EvaluationError("last checkpoint header is invalid")
    if payload.get("epoch") != completed_epochs - 1 or payload.get("global_step") != global_step:
        raise EvaluationError("last checkpoint position does not match the completed run")
    if payload.get("group_fingerprint") != group_fingerprint:
        raise EvaluationError("last checkpoint group fingerprint does not match the run summary")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise EvaluationError("last checkpoint metadata must be an object")
    _validate_base_metadata(metadata.get("base_checkpoint"), actual_base, "last checkpoint")
    if metadata.get("initial_head_checkpoint") != expected_initial_head:
        raise EvaluationError("last checkpoint initial head metadata does not match the run summary")
    if not _configs_match_across_resume(payload.get("config"), resolved_config):
        raise EvaluationError("last checkpoint configuration does not match resolved_config.json")
    training_state = payload.get("training_state")
    early_enabled = bool(expected_early_stopping.get("enabled"))
    if training_state is None and not early_enabled:
        return
    if not isinstance(training_state, Mapping) or training_state.get("early_stopping") != expected_early_stopping:
        raise EvaluationError("last checkpoint early-stopping state does not match the run summary")


def _leaderboard_entry(value: object, *, completed_epochs: int, final_step: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvaluationError("leaderboard entry must be an object")
    raw_entry = cast(dict[str, Any], value)
    epoch = _nonnegative_int(raw_entry.get("epoch"), "leaderboard epoch")
    global_step = _nonnegative_int(raw_entry.get("global_step"), "leaderboard global_step")
    metric = _finite_float(raw_entry.get("metric"), "leaderboard metric")
    filename = raw_entry.get("filename")
    if not isinstance(filename, str) or not filename or Path(filename).name != filename:
        raise EvaluationError("leaderboard filename must be a basename")
    if not filename.startswith("best_epoch_") or not filename.endswith(".pt"):
        raise EvaluationError("leaderboard filename is not a generic best checkpoint name")
    if epoch >= completed_epochs:
        raise EvaluationError("leaderboard epoch is outside the completed run")
    if global_step > final_step:
        raise EvaluationError("leaderboard global_step is after the completed run")
    return {"epoch": epoch, "filename": filename, "global_step": global_step, "metric": metric}


def _validate_leaderboard(
    leaderboard: Mapping[str, Any],
    summary: Mapping[str, Any],
    config: Mapping[str, Any],
    checkpoint_dir: Path,
    *,
    completed_epochs: int,
    final_step: int,
) -> list[dict[str, Any]]:
    checkpoint_config = _mapping(config, "checkpoint")
    k = _positive_int(checkpoint_config.get("k"), "checkpoint k")
    mode = checkpoint_config.get("mode")
    monitor = checkpoint_config.get("monitor")
    if mode not in {"min", "max"}:
        raise EvaluationError("checkpoint mode must be min or max")
    if monitor not in _MONITOR_TO_METRIC:
        raise EvaluationError("checkpoint monitor is unsupported")
    expected_header = {"format_version": 1, "k": k, "mode": mode, "monitor": monitor}
    if any(leaderboard.get(key) != expected for key, expected in expected_header.items()):
        raise EvaluationError("leaderboard header does not match resolved checkpoint configuration")
    raw_entries = leaderboard.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise EvaluationError("leaderboard must contain at least one best checkpoint")
    if len(raw_entries) > k:
        raise EvaluationError("leaderboard contains more checkpoints than configured k")
    entries = [
        _leaderboard_entry(value, completed_epochs=completed_epochs, final_step=final_step) for value in raw_entries
    ]
    if len({entry["epoch"] for entry in entries}) != len(entries):
        raise EvaluationError("leaderboard contains duplicate epochs")
    if len({entry["filename"] for entry in entries}) != len(entries):
        raise EvaluationError("leaderboard contains duplicate filenames")
    sort_key = (
        (lambda entry: (entry["metric"], entry["epoch"], entry["filename"]))
        if mode == "min"
        else (lambda entry: (-entry["metric"], entry["epoch"], entry["filename"]))
    )
    if entries != sorted(entries, key=sort_key):
        raise EvaluationError("leaderboard is not in deterministic best-first order")
    if summary.get("best") != raw_entries:
        raise EvaluationError("run summary best entries do not exactly match the leaderboard")
    if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
        raise EvaluationError("checkpoint directory must be a regular directory")
    expected_files = {entry["filename"] for entry in entries}
    actual_files = {
        path.name for path in checkpoint_dir.glob("best_epoch_*.pt") if path.is_file() and not path.is_symlink()
    }
    if actual_files != expected_files:
        raise EvaluationError("best checkpoint files do not exactly match the leaderboard")
    return entries


def _load_best_payload(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise EvaluationError("leaderboard checkpoint must be a regular file")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise EvaluationError("best checkpoint cannot be loaded safely") from error
    if not isinstance(payload, dict):
        raise EvaluationError("best checkpoint payload must be an object")
    return payload


def _validate_and_load_head(
    payload: Mapping[str, Any],
    entry: Mapping[str, Any],
    *,
    model: nn.Module,
    trainable_names: tuple[str, ...],
    group_fingerprint: str,
    actual_base: Mapping[str, Any],
    expected_initial_head: Mapping[str, Any] | None,
    expected_monitor: str,
    resolved_config: Mapping[str, Any],
) -> None:
    if payload.get("format_version") != 1:
        raise EvaluationError("best checkpoint format_version is unsupported")
    if payload.get("kind") != "best":
        raise EvaluationError("best checkpoint kind must be best")
    if payload.get("parameter_state") != "x":
        raise EvaluationError("best checkpoint parameter_state must be x")
    if "optimizer_state" in payload:
        raise EvaluationError("best checkpoint must not contain optimizer state")
    if payload.get("epoch") != entry["epoch"]:
        raise EvaluationError("best checkpoint epoch does not match the leaderboard")
    if payload.get("global_step") != entry["global_step"]:
        raise EvaluationError("best checkpoint global_step does not match the leaderboard")
    payload_metric = _finite_float(payload.get("metric"), "best checkpoint metric")
    if payload_metric != entry["metric"]:
        raise EvaluationError("best checkpoint metric does not match the leaderboard")
    if payload.get("monitor") != expected_monitor:
        raise EvaluationError("best checkpoint monitor does not match the leaderboard")
    if payload.get("group_fingerprint") != group_fingerprint:
        raise EvaluationError("best checkpoint group fingerprint does not match the run summary")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise EvaluationError("best checkpoint metadata must be an object")
    _validate_base_metadata(metadata.get("base_checkpoint"), actual_base, "checkpoint")
    if metadata.get("initial_head_checkpoint") != expected_initial_head:
        raise EvaluationError("checkpoint initial head metadata does not match the run summary")
    if not _configs_match_across_resume(payload.get("config"), resolved_config):
        raise EvaluationError("best checkpoint configuration does not match resolved_config.json")

    model_state = payload.get("model_state")
    if not isinstance(model_state, Mapping):
        raise EvaluationError("best checkpoint model_state must be an object")
    expected_names = set(trainable_names)
    actual_names = set(model_state)
    if actual_names != expected_names:
        raise EvaluationError(
            "best checkpoint does not exactly match the trainable parameter set: "
            f"missing={sorted(expected_names - actual_names)}, unexpected={sorted(actual_names - expected_names)}"
        )
    if any(not isinstance(name, str) or not isinstance(value, torch.Tensor) for name, value in model_state.items()):
        raise EvaluationError("best checkpoint trainable state must map parameter names to tensors")
    try:
        incompatible = model.load_state_dict(dict(model_state), strict=False)
    except RuntimeError as error:
        raise EvaluationError("best checkpoint trainable tensors are incompatible with the base model") from error
    if incompatible.unexpected_keys:
        raise EvaluationError("best checkpoint produced unexpected model keys")


def _configs_match_across_resume(value: object, resolved_config: Mapping[str, Any]) -> bool:
    if not isinstance(value, Mapping):
        return False

    def normalized(config: Mapping[str, Any]) -> dict[str, Any]:
        copied = json.loads(json.dumps(dict(config), allow_nan=False))
        if not isinstance(copied, dict):
            return {}
        trainer = copied.get("trainer")
        if isinstance(trainer, dict):
            trainer["resume_from"] = None
        return copied

    try:
        return normalized(cast(Mapping[str, Any], value)) == normalized(resolved_config)
    except (TypeError, ValueError):
        return False


def _finite_metrics(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise EvaluationError("validation result must be a metric object")
    result: dict[str, float] = {}
    for key, metric in value.items():
        if not isinstance(key, str) or not key:
            raise EvaluationError("validation metric names must be non-empty strings")
        result[key] = _finite_float(metric, f"recomputed validation metric {key}")
    required = {"camera", "depth", "objective"}
    if not result.keys() >= required:
        raise EvaluationError(f"recomputed validation metrics are missing {sorted(required - result.keys())}")
    return result


def _validation_loss_options(config: Mapping[str, Any]) -> dict[str, float] | None:
    loss = config.get("loss")
    if not isinstance(loss, Mapping):
        return None
    validation = loss.get("validation")
    if not isinstance(validation, Mapping):
        raise EvaluationError("loss.validation configuration must be an object")
    options: dict[str, float] = {}
    for key in _LOSS_WEIGHT_KEYS:
        options[key] = _finite_float(validation.get(key), f"loss.validation.{key}")
        if options[key] < 0:
            raise EvaluationError(f"loss.validation.{key} must be non-negative")
    for key in _OPTIONAL_LOSS_DEFAULTS:
        if key in validation:
            options[key] = _finite_float(validation.get(key), f"loss.validation.{key}")
            if options[key] < 0:
                raise EvaluationError(f"loss.validation.{key} must be non-negative")
    max_metric_depth_m = validation.get("max_metric_depth_m")
    if max_metric_depth_m is not None:
        options["max_metric_depth_m"] = _finite_float(
            max_metric_depth_m,
            "loss.validation.max_metric_depth_m",
        )
        if options["max_metric_depth_m"] <= 0:
            raise EvaluationError("loss.validation.max_metric_depth_m must be positive")
    return options


def _validation_dataset(
    config: Mapping[str, Any],
    original_cwd: Path,
    dataset_factory: DatasetFactory,
) -> tuple[EvaluationDataset, dict[str, Any]]:
    data = _mapping(config, "data")
    trainer = _mapping(config, "trainer")
    data_root = _resolve_config_path(data.get("root"), original_cwd, "data.root")
    if data_root.is_symlink() or not data_root.is_dir():
        raise EvaluationError("configured staging root must be a regular directory")
    sequence_frames = trainer.get("sequence_frames")
    if sequence_frames is None:
        min_frames = _positive_int(data.get("min_frames"), "data.min_frames")
        max_frames = _positive_int(data.get("max_frames"), "data.max_frames")
    else:
        min_frames = max_frames = _positive_int(sequence_frames, "trainer.sequence_frames")
    if min_frames > max_frames:
        raise EvaluationError("validation frame range is invalid")
    seed = _nonnegative_int(config.get("seed"), "seed")
    split_key = "smoke_split" if trainer.get("name") == "smoke" else "val_split"
    split = data.get(split_key)
    if not isinstance(split, str) or split not in {"train", "val", "smoke"}:
        raise EvaluationError("validation split must be one of train, val, or smoke")
    min_valid_depth_pixels = _positive_int(
        data.get("min_valid_depth_pixels"),
        "data.min_valid_depth_pixels",
    )
    dataset_options: dict[str, Any] = {
        "split": split,
        "min_frames": min_frames,
        "max_frames": max_frames,
        "filter_short_sequences": min_frames == max_frames,
        "seed": seed,
        "min_valid_depth_pixels": min_valid_depth_pixels,
    }
    dynamic_config = config.get("dynamic_geometry")
    if isinstance(dynamic_config, Mapping) and dynamic_config.get("enabled") is True:
        pseudo_labels = dynamic_config.get("pseudo_labels")
        if not isinstance(pseudo_labels, Mapping):
            raise EvaluationError("dynamic geometry requires pseudo_labels")
        manifest = pseudo_labels.get("teacher_artifact_manifest")
        if (
            not isinstance(manifest, str)
            or not manifest
            or Path(manifest).is_absolute()
            or ".." in Path(manifest).parts
        ):
            raise EvaluationError("dynamic flow teacher manifest must be a safe relative path")
        dataset_options["flow_teacher_manifest"] = manifest
    dataset = dataset_factory(data_root, **dataset_options)
    if len(dataset) < 1:
        raise EvaluationError("validation dataset must not be empty")
    set_epoch = getattr(dataset, "set_epoch", None)
    if not callable(set_epoch):
        raise EvaluationError("validation dataset must support deterministic set_epoch")
    set_epoch(0)
    options = {
        "batch_size": _positive_int(data.get("batch_size"), "data.batch_size"),
        "max_batches": trainer.get("max_val_batches"),
        "min_valid_depth_pixels": min_valid_depth_pixels,
        "num_workers": _nonnegative_int(data.get("num_workers"), "data.num_workers"),
        "pin_memory": data.get("pin_memory"),
        "seed": seed,
        "split": split,
    }
    if not isinstance(options["pin_memory"], bool):
        raise EvaluationError("data.pin_memory must be boolean")
    if options["max_batches"] is not None:
        options["max_batches"] = _positive_int(options["max_batches"], "trainer.max_val_batches")
    return dataset, options


def _make_validation_loader(dataset: EvaluationDataset, options: Mapping[str, Any]) -> DataLoader[Any]:
    dataset.set_epoch(0)
    generator = torch.Generator().manual_seed(int(options["seed"]))
    return DataLoader(
        cast(Dataset[Any], dataset),
        batch_size=int(options["batch_size"]),
        shuffle=False,
        num_workers=int(options["num_workers"]),
        pin_memory=bool(options["pin_memory"]),
        drop_last=False,
        generator=generator,
    )


def _atomic_json(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(dict(payload), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def evaluate_training_checkpoints(
    run_dir: str | os.PathLike[str],
    *,
    output_path: str | os.PathLike[str],
    original_cwd: str | os.PathLike[str],
    device: str | torch.device = "cuda",
    tolerance: float = 1e-4,
    depth_thresholds_m: Iterable[float] = (0.4, 0.8, 1.2),
    checkpoint_limit: int | None = None,
    depth_provided_frames: int | None = None,
    validate_stored_monitor: bool = True,
    evaluation_batch_size: int | None = None,
    model_factory: ModelFactory | None = None,
    dataset_factory: DatasetFactory | None = None,
    validator: Validator | None = None,
) -> dict[str, Any]:
    """Recompute fixed validation metrics for every leaderboard checkpoint.

    The released base checkpoint is loaded strictly exactly once. Each compact
    best checkpoint must then replace the complete trainable state before the
    same deterministic validation dataset is evaluated in BF16.
    """

    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise ValueError("tolerance must be a finite non-negative number")
    tolerance_value = float(tolerance)
    if not math.isfinite(tolerance_value) or tolerance_value < 0:
        raise ValueError("tolerance must be a finite non-negative number")
    if checkpoint_limit is not None and (
        isinstance(checkpoint_limit, bool) or not isinstance(checkpoint_limit, int) or checkpoint_limit < 1
    ):
        raise ValueError("checkpoint_limit must be None or a positive integer")
    if not isinstance(validate_stored_monitor, bool):
        raise ValueError("validate_stored_monitor must be boolean")
    if evaluation_batch_size is not None and (
        isinstance(evaluation_batch_size, bool)
        or not isinstance(evaluation_batch_size, int)
        or evaluation_batch_size < 1
    ):
        raise ValueError("evaluation_batch_size must be None or a positive integer")
    thresholds_m: list[float] = []
    for value in depth_thresholds_m:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("depth thresholds must be finite positive numbers in strictly increasing order")
        threshold = float(value)
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("depth thresholds must be finite positive numbers in strictly increasing order")
        thresholds_m.append(threshold)
    if not thresholds_m or any(left >= right for left, right in pairwise(thresholds_m)):
        raise ValueError("depth thresholds must be finite positive numbers in strictly increasing order")
    validated_thresholds_m = tuple(thresholds_m)

    run_root = Path(run_dir).expanduser().resolve()
    cwd = Path(original_cwd).expanduser().resolve()
    if run_root.is_symlink() or not run_root.is_dir():
        raise EvaluationError("run directory must be a regular directory")
    if cwd.is_symlink() or not cwd.is_dir():
        raise EvaluationError("original cwd must be a regular directory")
    resolved_config = _read_json_object(run_root / "resolved_config.json", "resolved_config.json")
    configured_depth_input = resolved_config.get("depth_input")
    evaluation_depth_input: Mapping[str, object] | None = None
    if isinstance(configured_depth_input, Mapping) and configured_depth_input.get("enabled") is True:
        evaluation_depth_input = cast(Mapping[str, object], configured_depth_input)
    if depth_provided_frames is not None:
        if evaluation_depth_input is None:
            raise EvaluationError("depth_provided_frames requires an enabled depth_input config")
        sequence_frames = _positive_int(
            _mapping(resolved_config, "trainer").get("sequence_frames"), "trainer.sequence_frames"
        )
        if (
            isinstance(depth_provided_frames, bool)
            or not isinstance(depth_provided_frames, int)
            or not 0 <= depth_provided_frames <= sequence_frames
        ):
            raise ValueError("depth_provided_frames must be within the sequence frame range")
        evaluation_depth_input = {**evaluation_depth_input, "validation_provided_frames": depth_provided_frames}
    summary = _read_json_object(run_root / "run_summary.json", "run_summary.json")
    model_config = _mapping(resolved_config, "model")
    if model_config.get("precision") != "bf16":
        raise EvaluationError("final checkpoint evaluation requires model.precision=bf16")
    base_checkpoint = _resolve_config_path(
        model_config.get("pretrained_checkpoint"),
        cwd,
        "model.pretrained_checkpoint",
    )
    actual_base = _actual_base_metadata(base_checkpoint)
    completed_epochs, final_step, group_fingerprint, early_stopping_summary = _validate_completed_summary(
        summary,
        resolved_config,
        actual_base,
    )
    initial_head_metadata = _validated_initial_head_metadata(summary.get("initial_head_checkpoint"))
    configured_initial_head = model_config.get("initial_head_checkpoint")
    if configured_initial_head is None:
        if initial_head_metadata is not None:
            raise EvaluationError("run summary has unexpected initial head metadata")
    elif (
        not isinstance(configured_initial_head, str)
        or not configured_initial_head
        or Path(configured_initial_head).is_absolute()
        or ".." in Path(configured_initial_head).parts
        or initial_head_metadata is None
        or initial_head_metadata["filename"] != Path(configured_initial_head).name
    ):
        raise EvaluationError("configured initial head does not match the run summary")
    else:
        initial_head_path = _resolve_config_path(
            configured_initial_head,
            cwd,
            "model.initial_head_checkpoint",
        )
        if initial_head_path.is_symlink() or not initial_head_path.is_file():
            raise EvaluationError("configured initial head checkpoint must be a regular file")
        assert initial_head_metadata is not None
        if _sha256_file(initial_head_path) != initial_head_metadata["sha256"]:
            raise EvaluationError("configured initial head checkpoint SHA-256 does not match the run summary")
    checkpoint_config = _mapping(resolved_config, "checkpoint")
    monitor = str(checkpoint_config.get("monitor"))
    metric_key = _MONITOR_TO_METRIC.get(monitor)
    if metric_key is None:
        raise EvaluationError("checkpoint monitor is unsupported")
    checkpoint_dir = _safe_run_child(run_root, checkpoint_config.get("directory"), "checkpoint.directory")
    save_last = checkpoint_config.get("save_last")
    if not isinstance(save_last, bool):
        raise EvaluationError("checkpoint.save_last must be boolean")
    last_path = checkpoint_dir / "last.pt"
    if save_last:
        _validate_last_checkpoint(
            last_path,
            completed_epochs=completed_epochs,
            global_step=final_step,
            group_fingerprint=group_fingerprint,
            resolved_config=resolved_config,
            actual_base=actual_base,
            expected_initial_head=initial_head_metadata,
            expected_early_stopping=early_stopping_summary,
        )
    elif last_path.exists():
        raise EvaluationError("last checkpoint exists while checkpoint.save_last is disabled")
    leaderboard = _read_json_object(checkpoint_dir / "leaderboard.json", "leaderboard.json")
    entries = _validate_leaderboard(
        leaderboard,
        summary,
        resolved_config,
        checkpoint_dir,
        completed_epochs=completed_epochs,
        final_step=final_step,
    )
    if checkpoint_limit is not None:
        entries = entries[:checkpoint_limit]

    runtime_device = torch.device(device)
    if runtime_device.type not in {"cpu", "cuda"}:
        raise EvaluationError("evaluation device must be CPU or CUDA")
    if runtime_device.type == "cuda" and not torch.cuda.is_available():
        raise EvaluationError("CUDA evaluation was requested but CUDA is unavailable")
    seed = _nonnegative_int(resolved_config.get("seed"), "seed")
    torch.manual_seed(seed)
    if runtime_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    selected_model_factory = model_factory or build_training_model
    prepared = selected_model_factory(base_checkpoint, device=runtime_device)
    prepared, pixel_config, pixel_enabled = _attach_configured_training_wrappers(
        prepared,
        resolved_config,
        device=runtime_device,
    )
    model = prepared.model
    if not isinstance(model, nn.Module):
        raise EvaluationError("model factory did not return a torch module")
    trainable_names = tuple(prepared.trainable_parameter_names)
    if (
        not trainable_names
        or any(not isinstance(name, str) or not name for name in trainable_names)
        or len(set(trainable_names)) != len(trainable_names)
    ):
        raise EvaluationError("model factory returned an invalid trainable parameter contract")
    if not set(trainable_names) <= set(model.state_dict()):
        raise EvaluationError("model trainable parameter contract is absent from the strict base model")
    model.eval()

    selected_dataset_factory = dataset_factory or ColmapRgbdDataset
    dataset, validation_options = _validation_dataset(resolved_config, cwd, selected_dataset_factory)
    if evaluation_batch_size is not None:
        validation_options = {**validation_options, "batch_size": evaluation_batch_size}
    validation_loss_options = _validation_loss_options(resolved_config)
    renderer_config = resolved_config.get("renderer")
    validation_renderer_options = (
        _renderer_options(cast(Mapping[str, Any], renderer_config)) if isinstance(renderer_config, Mapping) else None
    )
    if (
        validation_loss_options
        and validation_loss_options.get("photometric_weight", 0.0) > 0
        and validation_renderer_options is None
    ):
        raise EvaluationError("renderer configuration is required for photometric validation")
    selected_validator = validator or validate_one_epoch
    checkpoint_reports: list[dict[str, Any]] = []
    for entry in entries:
        checkpoint_path = checkpoint_dir / entry["filename"]
        checkpoint_sha256 = _sha256_file(checkpoint_path)
        payload = _load_best_payload(checkpoint_path)
        _validate_and_load_head(
            payload,
            entry,
            model=model,
            trainable_names=trainable_names,
            group_fingerprint=group_fingerprint,
            actual_base=actual_base,
            expected_initial_head=initial_head_metadata,
            expected_monitor=monitor,
            resolved_config=resolved_config,
        )
        del payload
        gc.collect()
        loader = _make_validation_loader(dataset, validation_options)
        metrics = _finite_metrics(
            selected_validator(
                model=model,
                batches=loader,
                device=runtime_device,
                min_valid_depth_pixels=int(validation_options["min_valid_depth_pixels"]),
                max_batches=validation_options["max_batches"],
                precision="bf16",
                loss_options=validation_loss_options,
                renderer_options=validation_renderer_options,
                depth_thresholds_m=validated_thresholds_m,
                pixel_depth_options=(
                    {
                        **dict(cast(Mapping[str, object], pixel_config["flow"])),
                        "max_depth_m": float(cast(Mapping[str, object], pixel_config["geometry"])["max_depth_m"]),
                    }
                    if pixel_enabled and isinstance(pixel_config, Mapping)
                    else None
                ),
                dynamic_geometry_options=(
                    _dynamic_geometry_runtime_options(
                        cast(Mapping[str, Any], resolved_config["dynamic_geometry"]),
                        epoch=int(entry["epoch"]),
                    )
                    if isinstance(resolved_config.get("dynamic_geometry"), Mapping)
                    and cast(Mapping[str, Any], resolved_config["dynamic_geometry"]).get("enabled") is True
                    else None
                ),
                depth_input_options=evaluation_depth_input,
                flow_generator=(
                    torch.Generator(device=runtime_device).manual_seed(
                        seed + int(cast(Mapping[str, object], pixel_config["flow"])["seed_offset"]) + 1
                    )
                    if pixel_enabled and isinstance(pixel_config, Mapping)
                    else None
                ),
            )
        )
        if metric_key not in metrics:
            raise EvaluationError(f"recomputed validation metrics are missing {metric_key}")
        metric_error = abs(metrics[metric_key] - float(entry["metric"]))
        if validate_stored_monitor and metric_error > tolerance_value:
            raise EvaluationError("recomputed validation monitor exceeds the configured tolerance")
        checkpoint_reports.append(
            {
                "epoch": entry["epoch"],
                "filename": entry["filename"],
                "global_step": entry["global_step"],
                "metric_absolute_error": metric_error,
                "recomputed_metrics": metrics,
                "sha256": checkpoint_sha256,
                "stored_metric": entry["metric"],
            }
        )

    report = {
        "base_checkpoint": actual_base,
        "checkpoints": checkpoint_reports,
        "format_version": 1,
        "group_fingerprint": group_fingerprint,
        "initial_head_checkpoint": initial_head_metadata,
        "monitor": monitor,
        "stored_monitor_validated": validate_stored_monitor,
        "status": "passed",
        "tolerance": tolerance_value,
        "validation": {
            "checkpoint_count": len(checkpoint_reports),
            "depth_thresholds_m": list(validated_thresholds_m),
            "max_batches": validation_options["max_batches"],
            "precision": "bf16",
            "sample_count": len(dataset),
            "split": validation_options["split"],
            "depth_provided_frames": depth_provided_frames,
        },
    }
    _atomic_json(report, Path(output_path).expanduser().resolve())
    return report


@dataclass(frozen=True)
class _ScalarStatistics:
    total: float = 0.0
    count: int = 0

    @property
    def mean(self) -> float | None:
        if self.count == 0:
            return None
        return self.total / self.count


@dataclass(frozen=True)
class _PairedSnapshot:
    identity_digest: str
    sample_count: int
    cases: dict[str, dict[str, DepthSufficientStatistics]]
    pose: dict[str, _ScalarStatistics]
    holdout: DepthSufficientStatistics | None
    holdout_error: str | None


class _BaselinePairedCollector:
    def __init__(self) -> None:
        self.snapshot: _PairedSnapshot | None = None

    def __call__(self, **kwargs: Any) -> dict[str, float]:
        if self.snapshot is not None:
            raise EvaluationError("paired baseline collector must run exactly once")
        self.snapshot = _collect_paired_snapshot(conditioned=False, **kwargs)
        return {"camera": 0.0, "depth": 0.0, "objective": 0.0}


class _CandidatePairedCollector:
    def __init__(self) -> None:
        self.snapshots: list[_PairedSnapshot] = []

    def __call__(self, **kwargs: Any) -> dict[str, float]:
        self.snapshots.append(_collect_paired_snapshot(conditioned=True, **kwargs))
        return {"camera": 0.0, "depth": 0.0, "objective": 0.0}


def evaluate_rgbd_conditioning(
    base_run_dir: str | os.PathLike[str],
    candidate_run_dir: str | os.PathLike[str],
    *,
    output_path: str | os.PathLike[str],
    original_cwd: str | os.PathLike[str],
    device: str | torch.device = "cuda",
    tolerance: float = 1e-4,
    checkpoint_limit: int = 3,
    evaluation_batch_size: int = 2,
    checkpoint_evaluator: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compare RGB-only and RGB-D checkpoints on identical pixels and frame subsets."""

    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise ValueError("tolerance must be a finite non-negative number")
    tolerance_value = float(tolerance)
    if not math.isfinite(tolerance_value) or tolerance_value < 0:
        raise ValueError("tolerance must be a finite non-negative number")
    if isinstance(checkpoint_limit, bool) or not isinstance(checkpoint_limit, int) or checkpoint_limit < 1:
        raise ValueError("checkpoint_limit must be a positive integer")
    if (
        isinstance(evaluation_batch_size, bool)
        or not isinstance(evaluation_batch_size, int)
        or evaluation_batch_size < 1
    ):
        raise ValueError("evaluation_batch_size must be a positive integer")

    base_root = Path(base_run_dir).expanduser().resolve()
    candidate_root = Path(candidate_run_dir).expanduser().resolve()
    cwd = Path(original_cwd).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if base_root == candidate_root:
        raise EvaluationError("paired baseline and candidate run directories must differ")
    for root, name in ((base_root, "baseline"), (candidate_root, "candidate")):
        if root.is_symlink() or not root.is_dir():
            raise EvaluationError(f"paired {name} run directory must be a regular directory")
    if destination.exists() or destination.is_symlink():
        raise EvaluationError("paired output path must not already exist")
    base_config = _read_json_object(base_root / "resolved_config.json", "baseline resolved_config.json")
    candidate_config = _read_json_object(candidate_root / "resolved_config.json", "candidate resolved_config.json")
    _validate_paired_run_configs(base_config, candidate_config)

    destination.parent.mkdir(parents=True, exist_ok=True)
    selected_evaluator = checkpoint_evaluator or evaluate_training_checkpoints
    baseline_collector = _BaselinePairedCollector()
    candidate_collector = _CandidatePairedCollector()
    with tempfile.TemporaryDirectory(prefix=".rgbd-paired-", dir=destination.parent) as temporary_name:
        temporary = Path(temporary_name)
        baseline_validation = selected_evaluator(
            base_root,
            output_path=temporary / "baseline.json",
            original_cwd=cwd,
            device=device,
            tolerance=tolerance_value,
            checkpoint_limit=1,
            validate_stored_monitor=False,
            evaluation_batch_size=evaluation_batch_size,
            validator=baseline_collector,
        )
        candidate_validation = selected_evaluator(
            candidate_root,
            output_path=temporary / "candidate.json",
            original_cwd=cwd,
            device=device,
            tolerance=tolerance_value,
            checkpoint_limit=checkpoint_limit,
            validate_stored_monitor=False,
            evaluation_batch_size=evaluation_batch_size,
            validator=candidate_collector,
        )

    if baseline_collector.snapshot is None:
        raise EvaluationError("paired baseline evaluation produced no snapshot")
    baseline_snapshot = baseline_collector.snapshot
    candidate_checkpoints = candidate_validation.get("checkpoints")
    baseline_checkpoints = baseline_validation.get("checkpoints")
    if not isinstance(baseline_checkpoints, list) or len(baseline_checkpoints) != 1:
        raise EvaluationError("paired baseline evaluation must produce exactly one checkpoint")
    if not isinstance(candidate_checkpoints, list) or not candidate_checkpoints:
        raise EvaluationError("paired candidate evaluation must produce checkpoints")
    if len(candidate_checkpoints) != len(candidate_collector.snapshots):
        raise EvaluationError("paired candidate checkpoint and snapshot counts differ")
    if baseline_validation.get("base_checkpoint") != candidate_validation.get("base_checkpoint"):
        raise EvaluationError("paired runs do not use the same strict base checkpoint")
    baseline_checkpoint = baseline_checkpoints[0]
    if not isinstance(baseline_checkpoint, Mapping):
        raise EvaluationError("paired baseline checkpoint report is invalid")
    initial_head = candidate_validation.get("initial_head_checkpoint")
    if (
        not isinstance(initial_head, Mapping)
        or initial_head.get("sha256") != baseline_checkpoint.get("sha256")
        or initial_head.get("filename") != baseline_checkpoint.get("filename")
    ):
        raise EvaluationError("candidate initial head is not the paired baseline checkpoint")
    for snapshot in candidate_collector.snapshots:
        if (
            snapshot.identity_digest != baseline_snapshot.identity_digest
            or snapshot.sample_count != baseline_snapshot.sample_count
        ):
            raise EvaluationError("paired validation sample identities or ordering differ")

    candidate_reports = [
        _candidate_comparison_report(baseline_snapshot, snapshot, checkpoint)
        for snapshot, checkpoint in zip(candidate_collector.snapshots, candidate_checkpoints, strict=True)
    ]
    selection = _select_paired_candidate(candidate_reports, tolerance_value)
    selected_index = int(selection["candidate_index"])
    selected = candidate_reports[selected_index]
    guardrails = _paired_guardrails(selected)
    partial_passes = [
        _required_improvement(selected["comparisons"][f"V{k}"]["normalized_mae"], tolerance_value) for k in (1, 2, 3)
    ]
    paired_gate_passed = all(partial_passes) and all(
        item["passed"] for item in cast(Mapping[str, Mapping[str, Any]], guardrails["metrics"]).values()
    )
    report = {
        "base_checkpoint": {
            key: baseline_checkpoint[key]
            for key in ("epoch", "filename", "global_step", "sha256")
            if key in baseline_checkpoint
        },
        "candidates": candidate_reports,
        "edge_multiview": {
            "edge_3d_error_proxy": None,
            "multiview_depth_error": None,
            "reason": "not_measured_by_rgbd_paired_v1",
        },
        "format_version": 2,
        "full_training_started": False,
        "guardrails": guardrails,
        "paired_gate": {
            "passed": paired_gate_passed,
            "partial_improvement_by_k": dict(zip(("V1", "V2", "V3"), partial_passes, strict=True)),
            "verdict": "fit_for_next_bounded_stage" if paired_gate_passed else "no_go_for_full",
        },
        "protocol": "rgbd_paired_v1",
        "selection": selection,
        "status": "passed",
        "stored_monitor_validated": False,
        "tolerance": tolerance_value,
        "validation": {
            "availability_case_count": len(all_depth_availability_cases(4)),
            "batch_size": evaluation_batch_size,
            "identity_digest": baseline_snapshot.identity_digest,
            "precision": "bf16_inference_fp32_scoring_float64_accumulation",
            "sample_count": baseline_snapshot.sample_count,
            "sequence_frames": 4,
            "split": "val",
        },
    }
    _atomic_json(report, destination)
    return report


def _validate_paired_run_configs(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> None:
    """Allow only batch size, initial head, and depth-input conditioning to differ."""

    base_data = dict(_mapping(baseline, "data"))
    candidate_data = dict(_mapping(candidate, "data"))
    base_data.pop("batch_size", None)
    candidate_data.pop("batch_size", None)
    if base_data != candidate_data:
        raise EvaluationError("paired run data configurations differ beyond batch_size")
    base_model = dict(_mapping(baseline, "model"))
    candidate_model = dict(_mapping(candidate, "model"))
    base_initial_head = base_model.pop("initial_head_checkpoint", None)
    candidate_initial_head = candidate_model.pop("initial_head_checkpoint", None)
    if base_initial_head is not None or not isinstance(candidate_initial_head, str) or not candidate_initial_head:
        raise EvaluationError("paired runs must compare a raw-head baseline to its conditioned continuation")
    if base_model != candidate_model:
        raise EvaluationError("paired model configurations differ beyond initial_head_checkpoint")
    base_depth = baseline.get("depth_input")
    if isinstance(base_depth, Mapping) and base_depth.get("enabled") is True:
        raise EvaluationError("paired baseline must not enable depth_input")
    candidate_depth = candidate.get("depth_input")
    if not isinstance(candidate_depth, Mapping) or candidate_depth.get("enabled") is not True:
        raise EvaluationError("paired candidate must enable depth_input")
    if candidate_depth.get("patch_size") != 16:
        raise EvaluationError("rgbd_paired_v1 requires depth_input.patch_size=16")
    for key in set(baseline) | set(candidate):
        if key not in {"data", "depth_input", "model"} and baseline.get(key) != candidate.get(key):
            raise EvaluationError(f"paired run configuration differs at {key}")
    if baseline.get("seed") != 42:
        raise EvaluationError("rgbd_paired_v1 requires seed=42")
    trainer = _mapping(baseline, "trainer")
    if trainer.get("sequence_frames") != 4:
        raise EvaluationError("rgbd_paired_v1 requires fixed four-frame sequences")
    if (
        base_model.get("precision") != "bf16"
        or base_model.get("image_height") != 480
        or base_model.get("image_width") != 640
    ):
        raise EvaluationError("rgbd_paired_v1 requires BF16 640x480 models")


def _collect_paired_snapshot(
    *,
    conditioned: bool,
    model: nn.Module,
    batches: Iterable[Mapping[str, Any]],
    device: torch.device,
    precision: str,
    max_batches: int | None,
    **_: Any,
) -> _PairedSnapshot:
    cases = all_depth_availability_cases(4)
    totals = {
        case.case_id: {scope: DepthSufficientStatistics() for scope in ("all", "provided", "unprovided")}
        for case in cases
    }
    pose_totals: dict[str, _ScalarStatistics] = {}
    holdout_total = DepthSufficientStatistics()
    holdout_error: str | None = None
    identity = hashlib.sha256()
    sample_count = 0
    batch_count = 0
    model.eval()
    with torch.inference_mode():
        for raw_batch in batches:
            batch = _move_batch(raw_batch, device)
            batch_size = _update_identity_digest(identity, batch)
            sample_count += batch_size
            images = batch.get("images")
            target = batch.get("depths")
            valid = batch.get("depth_masks")
            scale = batch.get("normalization_scale_m")
            frame_ids = batch.get("frame_ids")
            if (
                not isinstance(images, torch.Tensor)
                or not isinstance(target, torch.Tensor)
                or not isinstance(valid, torch.Tensor)
                or not isinstance(scale, torch.Tensor)
                or not isinstance(frame_ids, torch.Tensor)
                or images.shape[:2] != (batch_size, 4)
            ):
                raise EvaluationError("rgbd_paired_v1 batch contract is invalid")
            mapped_depth = target.unsqueeze(2)
            mapped_mask = valid.unsqueeze(2)
            if conditioned:
                k4_predictions: Mapping[str, torch.Tensor] | None = None
                for case in cases:
                    availability = torch.tensor(case.mask, device=device).expand(batch_size, 4)
                    with _autocast_context(device, precision):
                        raw_predictions = model(
                            images,
                            mapped_depth=mapped_depth,
                            valid_mask=mapped_mask,
                            availability=availability,
                        )
                    predictions = _prediction_mapping(raw_predictions)
                    _accumulate_case_depth(totals[case.case_id], predictions, batch, availability)
                    if case.provided_frames == 4:
                        k4_predictions = predictions
                if k4_predictions is None:
                    raise EvaluationError("rgbd_paired_v1 did not evaluate the k=4 case")
                pose_totals = _merge_pose_statistics(pose_totals, _pose_statistics(k4_predictions, batch))
                if holdout_error is None:
                    try:
                        holdout = build_input_depth_holdout(mapped_depth, mapped_mask, frame_ids, patch_size=16)
                    except ValueError as error:
                        holdout_error = str(error)
                    else:
                        with _autocast_context(device, precision):
                            holdout_predictions = _prediction_mapping(
                                model(
                                    images,
                                    mapped_depth=holdout.depth,
                                    valid_mask=holdout.visible_mask,
                                    availability=torch.ones(batch_size, 4, dtype=torch.bool, device=device),
                                )
                            )
                        holdout_total = merge_depth_statistics(
                            (
                                holdout_total,
                                depth_sufficient_statistics(
                                    holdout_predictions["depth"],
                                    target,
                                    holdout.holdout_mask[:, :, 0],
                                    scale,
                                ),
                            )
                        )
            else:
                with _autocast_context(device, precision):
                    predictions = _prediction_mapping(model(images))
                for case in cases:
                    availability = torch.tensor(case.mask, device=device).expand(batch_size, 4)
                    _accumulate_case_depth(totals[case.case_id], predictions, batch, availability)
                pose_totals = _merge_pose_statistics(pose_totals, _pose_statistics(predictions, batch))
                if holdout_error is None:
                    try:
                        holdout = build_input_depth_holdout(mapped_depth, mapped_mask, frame_ids, patch_size=16)
                    except ValueError as error:
                        holdout_error = str(error)
                    else:
                        holdout_total = merge_depth_statistics(
                            (
                                holdout_total,
                                depth_sufficient_statistics(
                                    predictions["depth"], target, holdout.holdout_mask[:, :, 0], scale
                                ),
                            )
                        )
            batch_count += 1
            if max_batches is not None and batch_count >= max_batches:
                break
    if batch_count == 0:
        raise EvaluationError("rgbd_paired_v1 received no validation batches")
    return _PairedSnapshot(
        identity.hexdigest(),
        sample_count,
        totals,
        pose_totals,
        None if holdout_error is not None else holdout_total,
        holdout_error,
    )


def _prediction_mapping(value: object) -> Mapping[str, torch.Tensor]:
    if not isinstance(value, dict):
        raise EvaluationError("paired model prediction must be an object")
    prediction_mapping = cast(dict[str, object], value)
    depth = prediction_mapping.get("depth")
    pose = prediction_mapping.get("pose_enc")
    if not isinstance(depth, torch.Tensor) or not isinstance(pose, torch.Tensor):
        raise EvaluationError("paired model prediction requires depth and pose_enc tensors")
    return cast(dict[str, torch.Tensor], prediction_mapping)


def _update_identity_digest(digest: Any, batch: Mapping[str, Any]) -> int:
    frame_ids = batch.get("frame_ids")
    sequence_ids = batch.get("sequence_id")
    scene_ids = batch.get("scene_id")
    if not isinstance(frame_ids, torch.Tensor) or frame_ids.ndim != 2:
        raise EvaluationError("paired batch frame_ids must have shape [B,S]")
    batch_size = int(frame_ids.shape[0])
    if not isinstance(sequence_ids, torch.Tensor) or sequence_ids.numel() != batch_size:
        raise EvaluationError("paired batch sequence_id must contain one value per sample")
    if not isinstance(scene_ids, (list, tuple)) or len(scene_ids) != batch_size:
        raise EvaluationError("paired batch scene_id must contain one value per sample")
    for index in range(batch_size):
        scene = scene_ids[index]
        if not isinstance(scene, str) or not scene:
            raise EvaluationError("paired scene identities must be non-empty strings")
        identity = [scene, int(sequence_ids.reshape(-1)[index]), [int(value) for value in frame_ids[index]]]
        digest.update(json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return batch_size


def _accumulate_case_depth(
    totals: dict[str, DepthSufficientStatistics],
    predictions: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    availability: torch.Tensor,
) -> None:
    target = cast(torch.Tensor, batch["depths"])
    valid = cast(torch.Tensor, batch["depth_masks"])
    scale = cast(torch.Tensor, batch["normalization_scale_m"])
    provided = availability[:, :, None, None]
    masks = {"all": valid, "provided": valid & provided, "unprovided": valid & ~provided}
    for scope, mask in masks.items():
        totals[scope] = merge_depth_statistics(
            (totals[scope], depth_sufficient_statistics(predictions["depth"], target, mask, scale))
        )


def _pose_statistics(predictions: Mapping[str, torch.Tensor], batch: Mapping[str, Any]) -> dict[str, _ScalarStatistics]:
    pose = predictions["pose_enc"].float()
    extrinsics = cast(torch.Tensor, batch["extrinsics"]).float()
    intrinsics = cast(torch.Tensor, batch["intrinsics"]).float()
    images = cast(torch.Tensor, batch["images"])
    frame_count = int(pose.shape[1])
    pair_count = frame_count * (frame_count - 1) // 2
    totals: dict[str, _ScalarStatistics] = {}
    for index in range(int(pose.shape[0])):
        camera = compute_camera_loss(
            pose[index : index + 1],
            extrinsics[index : index + 1],
            intrinsics[index : index + 1],
            (int(images.shape[-2]), int(images.shape[-1])),
        )
        pair_metrics = compute_pairwise_pose_loss(
            pose[index : index + 1],
            extrinsics[index : index + 1],
            (int(images.shape[-2]), int(images.shape[-1])),
        )
        valid_pair_float = float(pair_metrics["pairwise_valid_direction_fraction"]) * pair_count
        valid_pair_count = round(valid_pair_float)
        if abs(valid_pair_float - valid_pair_count) > 1e-5:
            raise EvaluationError("pairwise valid-direction count is not integral")
        for name in ("camera", "camera_translation", "camera_rotation", "camera_fov"):
            totals[name] = _add_scalar(totals.get(name), float(camera[name]), frame_count)
        for name in ("pairwise_rotation_degrees", "pairwise_translation_magnitude"):
            totals[name] = _add_scalar(totals.get(name), float(pair_metrics[name]), pair_count)
        for name in ("pairwise_translation_direction_degrees", "rpa_5", "rpa_15", "rpa_30"):
            totals[name] = _add_scalar(totals.get(name), float(pair_metrics[name]), valid_pair_count)
        totals["pairwise_valid_direction_fraction"] = _add_scalar(
            totals.get("pairwise_valid_direction_fraction"), valid_pair_count / pair_count, pair_count
        )
    return totals


def _add_scalar(existing: _ScalarStatistics | None, value: float, count: int) -> _ScalarStatistics:
    if not math.isfinite(value):
        raise EvaluationError("paired scalar metric must be finite")
    current = existing or _ScalarStatistics()
    return _ScalarStatistics(current.total + value * count, current.count + count)


def _merge_pose_statistics(
    left: Mapping[str, _ScalarStatistics], right: Mapping[str, _ScalarStatistics]
) -> dict[str, _ScalarStatistics]:
    merged = dict(left)
    for name, value in right.items():
        current = merged.get(name, _ScalarStatistics())
        merged[name] = _ScalarStatistics(current.total + value.total, current.count + value.count)
    return merged


def _candidate_comparison_report(
    baseline: _PairedSnapshot,
    candidate: _PairedSnapshot,
    checkpoint: object,
) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise EvaluationError("paired candidate checkpoint report is invalid")
    checkpoint_mapping = cast(dict[str, Any], checkpoint)
    cases: dict[str, Any] = {}
    for case in all_depth_availability_cases(4):
        cases[case.case_id] = {
            "provided_frames": case.provided_frames,
            **{
                scope: _depth_comparison(baseline.cases[case.case_id][scope], candidate.cases[case.case_id][scope])
                for scope in ("all", "provided", "unprovided")
            },
        }
    comparisons: dict[str, Any] = {}
    for k in range(5):
        scope = "all" if k in {0, 4} else "unprovided"
        base_total = _merge_case_scope(baseline, k, scope)
        candidate_total = _merge_case_scope(candidate, k, scope)
        comparisons[f"V{k}"] = _depth_comparison(base_total, candidate_total)
    primary_base = sum(cast(float, comparisons[f"V{k}"]["normalized_mae"]["baseline"]["value"]) for k in (1, 2, 3)) / 3
    primary_candidate = (
        sum(cast(float, comparisons[f"V{k}"]["normalized_mae"]["candidate"]["value"]) for k in (1, 2, 3)) / 3
    )
    comparisons["P"] = _value_comparison(primary_base, primary_candidate, count=3)
    holdout: dict[str, Any]
    if baseline.holdout is None or candidate.holdout is None:
        holdout = {
            "reason": candidate.holdout_error or baseline.holdout_error or "holdout_not_available",
            "status": "not_available",
        }
    else:
        holdout = {"status": "measured", **_depth_comparison(baseline.holdout, candidate.holdout)}
    return {
        "cases": cases,
        "checkpoint": {
            key: checkpoint_mapping[key]
            for key in ("epoch", "filename", "global_step", "sha256", "stored_metric")
            if key in checkpoint_mapping
        },
        "comparisons": comparisons,
        "holdout": holdout,
        "pose_k4": _pose_comparison(baseline.pose, candidate.pose),
    }


def _merge_case_scope(snapshot: _PairedSnapshot, k: int, scope: str) -> DepthSufficientStatistics:
    return merge_depth_statistics(
        snapshot.cases[case.case_id][scope] for case in all_depth_availability_cases(4) if case.provided_frames == k
    )


def _depth_comparison(baseline: DepthSufficientStatistics, candidate: DepthSufficientStatistics) -> dict[str, Any]:
    if (
        baseline.valid_pixel_count != candidate.valid_pixel_count
        or baseline.near_valid_pixel_count != candidate.near_valid_pixel_count
    ):
        raise EvaluationError("paired depth comparisons must use identical pixel counts")
    return {
        "metric_mae_m": _sum_comparison(
            baseline.metric_absolute_error_sum_m,
            candidate.metric_absolute_error_sum_m,
            baseline.valid_pixel_count,
        ),
        "near_depth_mae_m": _sum_comparison(
            baseline.near_absolute_error_sum_m,
            candidate.near_absolute_error_sum_m,
            baseline.near_valid_pixel_count,
        ),
        "normalized_mae": _sum_comparison(
            baseline.normalized_absolute_error_sum,
            candidate.normalized_absolute_error_sum,
            baseline.valid_pixel_count,
        ),
    }


def _sum_comparison(baseline_sum: float, candidate_sum: float, count: int) -> dict[str, Any]:
    baseline = metric_result(baseline_sum, count)
    candidate = metric_result(candidate_sum, count)
    baseline["absolute_error_sum"] = baseline_sum
    candidate["absolute_error_sum"] = candidate_sum
    return {"baseline": baseline, "candidate": candidate, **_difference_fields(baseline["value"], candidate["value"])}


def _value_comparison(baseline: float, candidate: float, *, count: int) -> dict[str, Any]:
    return {
        "baseline": {"value": baseline, "count": count},
        "candidate": {"value": candidate, "count": count},
        **_difference_fields(baseline, candidate),
    }


def _difference_fields(baseline: object, candidate: object) -> dict[str, float | None]:
    if baseline is None or candidate is None:
        return {"difference": None, "difference_percent": None}
    baseline_value = float(cast(float, baseline))
    candidate_value = float(cast(float, candidate))
    difference = candidate_value - baseline_value
    return {
        "difference": difference,
        "difference_percent": None if baseline_value == 0 else 100 * difference / baseline_value,
    }


def _pose_comparison(
    baseline: Mapping[str, _ScalarStatistics], candidate: Mapping[str, _ScalarStatistics]
) -> dict[str, Any]:
    if set(baseline) != set(candidate):
        raise EvaluationError("paired pose metric sets differ")
    report: dict[str, Any] = {}
    for name in sorted(baseline):
        if baseline[name].count != candidate[name].count:
            raise EvaluationError("paired pose metric counts differ")
        report[name] = _value_comparison(
            cast(float, baseline[name].mean), cast(float, candidate[name].mean), count=baseline[name].count
        )
    return report


def _select_paired_candidate(candidates: list[dict[str, Any]], tolerance: float) -> dict[str, Any]:
    values = [float(candidate["comparisons"]["P"]["candidate"]["value"]) for candidate in candidates]
    minimum = min(values)
    equivalent = [index for index, value in enumerate(values) if value - minimum <= tolerance]
    selected_index = min(equivalent, key=lambda index: int(candidates[index]["checkpoint"]["epoch"]))
    ranking = sorted(
        range(len(candidates)), key=lambda index: (values[index], candidates[index]["checkpoint"]["epoch"])
    )
    return {
        "candidate_index": selected_index,
        "equivalent_candidate_indices": equivalent,
        "primary": "depth_unprovided_macro",
        "ranking_candidate_indices": ranking,
        "selected_checkpoint": candidates[selected_index]["checkpoint"],
    }


def _paired_guardrails(selected: Mapping[str, Any]) -> dict[str, Any]:
    v4 = cast(Mapping[str, Any], cast(Mapping[str, Any], selected["comparisons"])["V4"])
    pose = cast(Mapping[str, Any], selected["pose_k4"])
    depth_metric = cast(Mapping[str, Any], v4["near_depth_mae_m"])
    camera_metric = cast(Mapping[str, Any], pose["camera_translation"])
    camera = cast(Mapping[str, Any], pose["camera"])
    normalized = cast(Mapping[str, Any], v4["normalized_mae"])
    base_objective = 5 * float(cast(Mapping[str, Any], camera["baseline"])["value"]) + float(
        cast(Mapping[str, Any], normalized["baseline"])["value"]
    )
    candidate_objective = 5 * float(cast(Mapping[str, Any], camera["candidate"])["value"]) + float(
        cast(Mapping[str, Any], normalized["candidate"])["value"]
    )
    metrics = {
        "near_depth_mae_m": _guardrail_metric(depth_metric, absolute_tolerance=0.01),
        "camera_translation": _guardrail_metric(camera_metric, absolute_tolerance=0.01),
        "objective": _guardrail_metric(
            _value_comparison(base_objective, candidate_objective, count=1), absolute_tolerance=0.02
        ),
    }
    return {"metrics": metrics, "relative_tolerance": 0.1}


def _guardrail_metric(comparison: Mapping[str, Any], *, absolute_tolerance: float) -> dict[str, Any]:
    baseline = float(cast(Mapping[str, Any], comparison["baseline"])["value"])
    candidate = float(cast(Mapping[str, Any], comparison["candidate"])["value"])
    limit = baseline + max(abs(baseline) * 0.1, absolute_tolerance)
    return {
        "absolute_tolerance": absolute_tolerance,
        "baseline": baseline,
        "candidate": candidate,
        "limit": limit,
        "passed": candidate <= limit,
    }


def _required_improvement(comparison: Mapping[str, Any], tolerance: float) -> bool:
    difference = comparison.get("difference")
    return isinstance(difference, (int, float)) and not isinstance(difference, bool) and float(difference) < -tolerance


def _attach_configured_training_wrappers(
    prepared: PreparedTrainingModel,
    resolved_config: Mapping[str, Any],
    *,
    device: torch.device,
) -> tuple[PreparedTrainingModel, object, bool]:
    """Rebuild training wrappers in the same base→pixel→dynamic order as the runner."""

    depth_config = resolved_config.get("depth_input")
    depth_enabled = isinstance(depth_config, Mapping) and depth_config.get("enabled") is True
    pixel_config = resolved_config.get("pixel_depth")
    pixel_enabled = isinstance(pixel_config, Mapping) and pixel_config.get("enabled") is True
    dynamic_config = resolved_config.get("dynamic_geometry")
    dynamic_enabled = isinstance(dynamic_config, Mapping) and dynamic_config.get("enabled") is True
    if depth_enabled and (pixel_enabled or dynamic_enabled):
        raise EvaluationError("depth_input cannot be combined with pixel_depth or dynamic_geometry")
    if depth_enabled:
        prepared = attach_depth_input_model(prepared, cast(Mapping[str, object], depth_config), device=device)
    if pixel_enabled:
        prepared = attach_pixel_depth_model(prepared, cast(Mapping[str, object], pixel_config), device=device)
    if dynamic_enabled:
        prepared = attach_dynamic_geometry_model(
            prepared,
            cast(Mapping[str, object], dynamic_config),
            device=device,
        )
    return prepared, pixel_config, pixel_enabled
