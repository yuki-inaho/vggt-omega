"""Strict, deterministic re-evaluation of saved top-K training checkpoints."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol, cast

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from vggt_omega.training.dataset import ColmapRgbdDataset
from vggt_omega.training.model_factory import PreparedTrainingModel, build_training_model
from vggt_omega.training.runner import validate_one_epoch

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
}
_LOSS_WEIGHT_KEYS = (
    "camera_weight",
    "depth_weight",
    "translation_weight",
    "rotation_weight",
    "fov_weight",
)
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
    if (not stopped_early and completed_epochs != configured_epochs) or (
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
    dataset = dataset_factory(
        data_root,
        split=split,
        min_frames=min_frames,
        max_frames=max_frames,
        seed=seed,
        min_valid_depth_pixels=min_valid_depth_pixels,
    )
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

    run_root = Path(run_dir).expanduser().resolve()
    cwd = Path(original_cwd).expanduser().resolve()
    if run_root.is_symlink() or not run_root.is_dir():
        raise EvaluationError("run directory must be a regular directory")
    if cwd.is_symlink() or not cwd.is_dir():
        raise EvaluationError("original cwd must be a regular directory")
    resolved_config = _read_json_object(run_root / "resolved_config.json", "resolved_config.json")
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
    validation_loss_options = _validation_loss_options(resolved_config)
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
            )
        )
        if metric_key not in metrics:
            raise EvaluationError(f"recomputed validation metrics are missing {metric_key}")
        metric_error = abs(metrics[metric_key] - float(entry["metric"]))
        if metric_error > tolerance_value:
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
        "status": "passed",
        "tolerance": tolerance_value,
        "validation": {
            "checkpoint_count": len(checkpoint_reports),
            "max_batches": validation_options["max_batches"],
            "precision": "bf16",
            "sample_count": len(dataset),
            "split": validation_options["split"],
        },
    }
    _atomic_json(report, Path(output_path).expanduser().resolve())
    return report
