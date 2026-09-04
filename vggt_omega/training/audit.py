"""Bounded, privacy-minimal validation of completed training artifacts."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import torch
from tensorboard.backend.event_processing import event_accumulator

from vggt_omega.training.tensorboard import ALLOWED_SCALAR_TAGS

_TEXT_SUFFIXES = frozenset({".csv", ".json", ".log", ".md", ".txt", ".yaml", ".yml"})
_REQUIRED_SCALAR_TAGS = frozenset(
    {
        "optimizer/beta1",
        "optimizer/group_0_lr",
        "train/camera",
        "train/depth",
        "train/grad_norm",
        "train/objective",
        "val/camera",
        "val/depth",
        "val/objective",
    }
)
_CAMERA_COMPONENT_SCALAR_TAGS = frozenset(
    {
        "train/camera_translation",
        "train/camera_rotation",
        "train/camera_fov",
        "val/camera_translation",
        "val/camera_rotation",
        "val/camera_fov",
    }
)
_PAIRWISE_SCALAR_TAGS = frozenset(
    {
        "train/pairwise_pose",
        "train/pairwise_rotation_degrees",
        "train/pairwise_translation_direction_degrees",
        "train/pairwise_translation_magnitude",
        "train/rpa_5",
        "train/rpa_15",
        "train/rpa_30",
        "val/pairwise_pose",
        "val/pairwise_rotation_degrees",
        "val/pairwise_translation_direction_degrees",
        "val/pairwise_translation_magnitude",
        "val/rpa_5",
        "val/rpa_15",
        "val/rpa_30",
    }
)
_PHOTOMETRIC_SCALAR_TAGS = frozenset(
    {
        "train/photometric",
        "train/photometric_visibility",
        "val/photometric",
        "val/photometric_visibility",
    }
)
_HEX_DIGITS = frozenset("0123456789abcdef")
_STANDARD_EARLY_STOPPING_CONFIG = {
    "enabled": False,
    "monitor": "val/objective",
    "mode": "min",
    "patience": 2,
    "min_delta": 0.0,
}


class _AuditState:
    def __init__(self) -> None:
        self.errors: list[dict[str, str]] = []
        self._error_keys: set[tuple[str, str]] = set()

    def error(self, check: str, code: str) -> None:
        key = (check, code)
        if key not in self._error_keys:
            self._error_keys.add(key)
            self.errors.append({"check": check, "code": code})

    def status(self, check: str) -> str:
        return "failed" if any(error["check"] == check for error in self.errors) else "passed"


def _is_nonnegative_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX_DIGITS


def _is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _is_nonnegative_finite_number(value: object) -> bool:
    return _is_finite_number(value) and float(cast(int | float, value)) >= 0


def _read_json_object(path: Path, *, check: str, state: _AuditState) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        state.error(check, "json_unreadable")
        return None
    if not isinstance(value, dict):
        state.error(check, "json_not_object")
        return None
    return value


def _safe_run_child(
    run_dir: Path,
    raw_value: object,
    *,
    fallback: str,
    check: str,
    state: _AuditState,
) -> Path:
    if not isinstance(raw_value, str) or not raw_value:
        state.error(check, "artifact_directory_invalid")
        return run_dir / fallback
    relative = Path(raw_value)
    candidate = (run_dir / relative).resolve(strict=False)
    if relative.is_absolute() or candidate == run_dir or not candidate.is_relative_to(run_dir):
        state.error(check, "artifact_directory_unsafe")
        return run_dir / fallback
    return candidate


def _iter_text_artifacts(run_dir: Path, report_path: Path) -> Iterable[Path]:
    for path in sorted(run_dir.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        if path.resolve(strict=False) == report_path:
            continue
        if path.suffix.lower() in _TEXT_SUFFIXES:
            yield path


def _matching_patterns(path: Path, patterns: tuple[bytes, ...]) -> set[int]:
    matches: set[int] = set()
    if not patterns:
        return matches
    carry_size = max(len(pattern) for pattern in patterns) - 1
    carry = b""
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            window = carry + chunk
            matches.update(index for index, pattern in enumerate(patterns) if pattern in window)
            carry = window[-carry_size:] if carry_size > 0 else b""
    return matches


def _audit_privacy(
    run_dir: Path,
    report_path: Path,
    deny_tokens: tuple[str, ...],
    state: _AuditState,
) -> dict[str, Any]:
    text_file_count = 0
    patterns = (b"/home/", *(token.encode("utf-8") for token in deny_tokens))
    for path in sorted(run_dir.rglob("*")):
        if path.is_symlink():
            state.error("privacy", "symlink_artifact")
    for path in _iter_text_artifacts(run_dir, report_path):
        text_file_count += 1
        try:
            matches = _matching_patterns(path, patterns)
        except OSError:
            state.error("privacy", "text_artifact_unreadable")
            continue
        if 0 in matches:
            state.error("privacy", "absolute_home_path")
        if any(index > 0 for index in matches):
            state.error("privacy", "private_token")
    return {"status": state.status("privacy"), "text_file_count": text_file_count}


def _nested_mapping(value: object, key: str) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    nested = cast(Mapping[str, Any], value).get(key)
    return nested if isinstance(nested, Mapping) else None


def _audit_summary(
    run_dir: Path,
    state: _AuditState,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any]]:
    summary = _read_json_object(run_dir / "run_summary.json", check="summary", state=state)
    config = _read_json_object(run_dir / "resolved_config.json", check="summary", state=state)
    if summary is not None:
        if summary.get("status") != "complete":
            state.error("summary", "run_not_complete")
        if not _is_nonnegative_integer(summary.get("epochs_completed")):
            state.error("summary", "epochs_completed_invalid")
        if not _is_nonnegative_integer(summary.get("global_step")):
            state.error("summary", "global_step_invalid")
        if not _is_digest(summary.get("group_fingerprint")):
            state.error("summary", "group_fingerprint_invalid")
        for metric_group in ("train", "validation"):
            metrics = _nested_mapping(summary, metric_group)
            if not metrics or any(
                not isinstance(key, str) or not key or not _is_finite_number(value) for key, value in metrics.items()
            ):
                state.error("summary", f"{metric_group}_metrics_invalid")
        base = summary.get("base_checkpoint")
        if not _valid_base_checkpoint(base):
            state.error("summary", "base_checkpoint_invalid")
        model_config = _nested_mapping(config, "model")
        configured_initial = model_config.get("initial_head_checkpoint") if model_config else None
        initial_head = summary.get("initial_head_checkpoint")
        if configured_initial is None:
            if initial_head is not None:
                state.error("summary", "initial_head_unexpected")
        elif (
            not isinstance(configured_initial, str)
            or not configured_initial
            or Path(configured_initial).is_absolute()
            or ".." in Path(configured_initial).parts
            or not _valid_initial_head_checkpoint(initial_head)
            or cast(Mapping[str, Any], initial_head).get("filename") != Path(configured_initial).name
        ):
            state.error("summary", "initial_head_invalid")
        trainer_config = _nested_mapping(config, "trainer")
        expected_epochs = trainer_config.get("epochs") if trainer_config else None
        completed_epochs = summary.get("epochs_completed")
        stopped_early = summary.get("stopped_early", False)
        if (
            not _is_nonnegative_integer(expected_epochs)
            or not isinstance(stopped_early, bool)
            or (not stopped_early and completed_epochs != expected_epochs)
            or (
                stopped_early
                and (
                    not _is_nonnegative_integer(completed_epochs)
                    or cast(int, completed_epochs) < 1
                    or cast(int, completed_epochs) > cast(int, expected_epochs)
                )
            )
        ):
            state.error("summary", "run_epoch_count_mismatch")
        configured_early = _nested_mapping(trainer_config, "early_stopping")
        early_config = configured_early or _STANDARD_EARLY_STOPPING_CONFIG
        early_summary = _nested_mapping(summary, "early_stopping")
        if early_summary is None and not bool(early_config.get("enabled")) and not stopped_early:
            early_summary = {**early_config, "best": None, "bad_epochs": 0, "stopped": False}
        if early_summary is None:
            state.error("summary", "early_stopping_state_missing")
        else:
            expected_keys = ("enabled", "monitor", "mode", "patience", "min_delta")
            if any(early_summary.get(key) != early_config.get(key) for key in expected_keys):
                state.error("summary", "early_stopping_config_mismatch")
            enabled = early_config.get("enabled")
            patience = early_config.get("patience")
            best = early_summary.get("best")
            bad_epochs = early_summary.get("bad_epochs")
            stopped = early_summary.get("stopped")
            if (
                not isinstance(enabled, bool)
                or not _is_nonnegative_integer(patience)
                or cast(int, patience) < 1
                or not _is_nonnegative_finite_number(early_config.get("min_delta"))
                or early_config.get("mode") not in {"min", "max"}
                or not isinstance(early_config.get("monitor"), str)
                or not early_config.get("monitor")
                or (best is not None and not _is_finite_number(best))
                or not _is_nonnegative_integer(bad_epochs)
                or not isinstance(stopped, bool)
            ):
                state.error("summary", "early_stopping_state_invalid")
            elif stopped != stopped_early:
                state.error("summary", "early_stopping_flag_mismatch")
            elif enabled:
                checkpoint_config = _nested_mapping(config, "checkpoint")
                if (
                    checkpoint_config is None
                    or early_config.get("monitor") != checkpoint_config.get("monitor")
                    or early_config.get("mode") != checkpoint_config.get("mode")
                ):
                    state.error("summary", "early_stopping_checkpoint_monitor_mismatch")
                if _is_nonnegative_integer(completed_epochs) and cast(int, completed_epochs) > 0 and best is None:
                    state.error("summary", "early_stopping_best_missing")
                if stopped != (cast(int, bad_epochs) >= cast(int, patience)):
                    state.error("summary", "early_stopping_patience_inconsistent")
            elif best is not None or bad_epochs != 0 or stopped:
                state.error("summary", "disabled_early_stopping_has_state")
    return summary, config, {"status": state.status("summary")}


def _valid_base_checkpoint(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    checkpoint = cast(Mapping[str, Any], value)
    filename = checkpoint.get("filename")
    size_bytes = checkpoint.get("size_bytes")
    return (
        isinstance(filename, str)
        and bool(filename)
        and Path(filename).name == filename
        and _is_nonnegative_integer(size_bytes)
        and _is_digest(checkpoint.get("sha256"))
    )


def _valid_initial_head_checkpoint(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    checkpoint = cast(Mapping[str, Any], value)
    filename = checkpoint.get("filename")
    return (
        isinstance(filename, str)
        and bool(filename)
        and Path(filename).name == filename
        and _is_digest(checkpoint.get("sha256"))
        and checkpoint.get("kind") in {"best", "resume"}
        and _is_nonnegative_integer(checkpoint.get("epoch"))
        and _is_nonnegative_integer(checkpoint.get("global_step"))
        and checkpoint.get("parameter_state") == "x"
    )


def _audit_tensorboard(
    run_dir: Path,
    config: Mapping[str, Any] | None,
    summary: Mapping[str, Any] | None,
    state: _AuditState,
) -> dict[str, Any]:
    logging_config = _nested_mapping(config, "logging")
    raw_directory = logging_config.get("directory", "tensorboard") if logging_config else "tensorboard"
    log_dir = _safe_run_child(
        run_dir,
        raw_directory,
        fallback="tensorboard",
        check="tensorboard",
        state=state,
    )
    event_files = sorted(log_dir.glob("events.out.tfevents.*")) if log_dir.is_dir() else []
    scalar_tags: set[str] = set()
    if not event_files:
        state.error("tensorboard", "tensorboard_events_missing")
    else:
        size_guidance = {
            event_accumulator.COMPRESSED_HISTOGRAMS: 1,
            event_accumulator.HISTOGRAMS: 1,
            event_accumulator.IMAGES: 1,
            event_accumulator.AUDIO: 1,
            event_accumulator.SCALARS: 0,
            event_accumulator.TENSORS: 1,
        }
        try:
            events = event_accumulator.EventAccumulator(str(log_dir), size_guidance=size_guidance)
            events.Reload()
            tags = events.Tags()
            scalar_tags = set(tags.get("scalars", []))
            forbidden_lists = ("images", "audio", "histograms", "distributions", "tensors", "run_metadata")
            if any(tags.get(name) for name in forbidden_lists) or tags.get("graph") or tags.get("meta_graph"):
                state.error("tensorboard", "tensorboard_non_scalar_content")
            required_tags = _REQUIRED_SCALAR_TAGS
            loss_config = _nested_mapping(config, "loss")
            if loss_config and loss_config.get("name") in {
                "camera_motion_curriculum",
                "camera_translation_focus",
            }:
                required_tags = required_tags | _CAMERA_COMPONENT_SCALAR_TAGS
            if loss_config:
                training_loss = _nested_mapping(loss_config, "training")
                validation_loss = _nested_mapping(loss_config, "validation")
                training_pairwise = bool(training_loss and float(training_loss.get("relative_pose_weight", 0.0)) > 0)
                validation_pairwise = bool(
                    validation_loss and float(validation_loss.get("relative_pose_weight", 0.0)) > 0
                )
                if training_pairwise or validation_pairwise:
                    required_tags = required_tags | _PAIRWISE_SCALAR_TAGS
                training_photometric = bool(training_loss and float(training_loss.get("photometric_weight", 0.0)) > 0)
                validation_photometric = bool(
                    validation_loss and float(validation_loss.get("photometric_weight", 0.0)) > 0
                )
                if training_photometric or validation_photometric:
                    required_tags = required_tags | _PHOTOMETRIC_SCALAR_TAGS
            if not scalar_tags >= required_tags:
                state.error("tensorboard", "tensorboard_missing_scalar_tags")
            if not scalar_tags <= ALLOWED_SCALAR_TAGS:
                state.error("tensorboard", "tensorboard_unknown_scalar_tag")
            trainer_config = _nested_mapping(config, "trainer")
            device = str(trainer_config.get("device", "")) if trainer_config else ""
            if device.startswith("cuda") and "system/max_cuda_memory_gib" not in scalar_tags:
                state.error("tensorboard", "tensorboard_missing_cuda_memory_tag")
            epochs_completed = summary.get("epochs_completed") if summary else None
            validate_every = trainer_config.get("validate_every_epochs") if trainer_config else None
            if _is_nonnegative_integer(epochs_completed) and _is_nonnegative_integer(validate_every):
                completed_count = cast(int, epochs_completed)
                validation_interval = cast(int, validate_every)
                expected_validations = completed_count // validation_interval if validation_interval else -1
                validation_steps = {sample.step for sample in events.Scalars("val/objective")}
                if len(validation_steps) != expected_validations:
                    state.error("tensorboard", "tensorboard_validation_count_mismatch")
            early_config = _nested_mapping(trainer_config, "early_stopping")
            if early_config and early_config.get("enabled") is True and summary is not None:
                monitor = early_config.get("monitor")
                mode = early_config.get("mode")
                patience = early_config.get("patience")
                min_delta = early_config.get("min_delta")
                early_summary = _nested_mapping(summary, "early_stopping")
                if (
                    not isinstance(monitor, str)
                    or monitor not in scalar_tags
                    or mode not in {"min", "max"}
                    or not _is_nonnegative_integer(patience)
                    or cast(int, patience) < 1
                    or not _is_nonnegative_finite_number(min_delta)
                    or early_summary is None
                ):
                    state.error("tensorboard", "early_stopping_history_unverifiable")
                else:
                    reconstructed_best: float | None = None
                    reconstructed_bad_epochs = 0
                    reconstructed_stopped = False
                    for sample in events.Scalars(monitor):
                        value = float(sample.value)
                        improved = reconstructed_best is None or (
                            value < reconstructed_best - float(cast(int | float, min_delta))
                            if mode == "min"
                            else value > reconstructed_best + float(cast(int | float, min_delta))
                        )
                        if improved:
                            reconstructed_best = value
                            reconstructed_bad_epochs = 0
                        else:
                            reconstructed_bad_epochs += 1
                            reconstructed_stopped = reconstructed_bad_epochs >= cast(int, patience)
                    summary_best = early_summary.get("best")
                    if (
                        reconstructed_best is None
                        or not _is_finite_number(summary_best)
                        or not math.isclose(
                            reconstructed_best,
                            float(cast(int | float, summary_best)),
                            rel_tol=1e-6,
                            abs_tol=1e-8,
                        )
                        or early_summary.get("bad_epochs") != reconstructed_bad_epochs
                        or early_summary.get("stopped") != reconstructed_stopped
                    ):
                        state.error("tensorboard", "early_stopping_history_mismatch")
            for tag in scalar_tags:
                samples = events.Scalars(tag)
                if not samples or samples[-1].step < 0 or not math.isfinite(float(samples[-1].value)):
                    state.error("tensorboard", "tensorboard_scalar_invalid")
                    break
        except (KeyError, OSError, RuntimeError, ValueError):
            state.error("tensorboard", "tensorboard_events_unreadable")
    return {
        "event_file_count": len(event_files),
        "scalar_tags": sorted(scalar_tags),
        "status": state.status("tensorboard"),
    }


def _valid_leaderboard_entry(value: object, state: _AuditState) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        state.error("leaderboard", "leaderboard_entry_invalid")
        return None
    raw_entry = cast(dict[str, Any], value)
    epoch = raw_entry.get("epoch")
    global_step = raw_entry.get("global_step")
    metric = raw_entry.get("metric")
    filename = raw_entry.get("filename")
    if not _is_nonnegative_integer(epoch) or not _is_nonnegative_integer(global_step):
        state.error("leaderboard", "leaderboard_entry_invalid")
        return None
    if isinstance(metric, bool) or not isinstance(metric, (int, float)) or not math.isfinite(float(metric)):
        state.error("leaderboard", "leaderboard_entry_invalid")
        return None
    if (
        not isinstance(filename, str)
        or not filename
        or Path(filename).name != filename
        or not filename.startswith("best_epoch_")
        or not filename.endswith(".pt")
    ):
        state.error("leaderboard", "leaderboard_entry_invalid")
        return None
    return {"epoch": epoch, "filename": filename, "global_step": global_step, "metric": float(metric)}


def _leaderboard_sort_key(entry: Mapping[str, Any], mode: str) -> tuple[float, int, str]:
    metric = float(entry["metric"])
    return (metric if mode == "min" else -metric, int(entry["epoch"]), str(entry["filename"]))


def _audit_leaderboard(
    run_dir: Path,
    config: Mapping[str, Any] | None,
    summary: Mapping[str, Any] | None,
    state: _AuditState,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    checkpoint_config = _nested_mapping(config, "checkpoint")
    raw_directory = checkpoint_config.get("directory", "checkpoints") if checkpoint_config else "checkpoints"
    checkpoint_dir = _safe_run_child(
        run_dir,
        raw_directory,
        fallback="checkpoints",
        check="leaderboard",
        state=state,
    )
    leaderboard = _read_json_object(checkpoint_dir / "leaderboard.json", check="leaderboard", state=state)
    entries: list[dict[str, Any]] = []
    k = 0
    mode = ""
    if leaderboard is not None:
        k_value = leaderboard.get("k")
        mode_value = leaderboard.get("mode")
        monitor = leaderboard.get("monitor")
        if leaderboard.get("format_version") != 1:
            state.error("leaderboard", "leaderboard_header_invalid")
        if not isinstance(k_value, int) or isinstance(k_value, bool) or k_value < 1:
            state.error("leaderboard", "leaderboard_header_invalid")
        else:
            k = k_value
        if mode_value not in {"min", "max"}:
            state.error("leaderboard", "leaderboard_header_invalid")
        else:
            mode = str(mode_value)
        if not isinstance(monitor, str) or not monitor:
            state.error("leaderboard", "leaderboard_header_invalid")
        raw_entries = leaderboard.get("entries")
        if not isinstance(raw_entries, list):
            state.error("leaderboard", "leaderboard_entries_invalid")
        else:
            entries = [entry for value in raw_entries if (entry := _valid_leaderboard_entry(value, state))]
            if len(entries) != len(raw_entries):
                state.error("leaderboard", "leaderboard_entries_invalid")
            if not raw_entries:
                state.error("leaderboard", "leaderboard_empty")
        if k and len(entries) > k:
            state.error("leaderboard", "leaderboard_exceeds_k")
        if len({entry["epoch"] for entry in entries}) != len(entries):
            state.error("leaderboard", "leaderboard_duplicate_epoch")
        if len({entry["filename"] for entry in entries}) != len(entries):
            state.error("leaderboard", "leaderboard_duplicate_filename")
        if mode and entries != sorted(entries, key=lambda entry: _leaderboard_sort_key(entry, mode)):
            state.error("leaderboard", "leaderboard_order")

        if checkpoint_config:
            expected_header = {
                "k": checkpoint_config.get("k"),
                "mode": checkpoint_config.get("mode"),
                "monitor": checkpoint_config.get("monitor"),
            }
            if any(leaderboard.get(key) != value for key, value in expected_header.items()):
                state.error("leaderboard", "leaderboard_config_mismatch")

    best_files = {path.name for path in checkpoint_dir.glob("best_epoch_*.pt") if path.is_file()}
    referenced_files = {str(entry["filename"]) for entry in entries}
    if best_files != referenced_files:
        state.error("leaderboard", "leaderboard_file_set_mismatch")
    all_pt_files = {path.name for path in checkpoint_dir.glob("*.pt") if path.is_file()}
    if all_pt_files - referenced_files - {"last.pt"}:
        state.error("leaderboard", "untracked_checkpoint")
    if len([name for name in all_pt_files if name.startswith("last")]) > 1:
        state.error("leaderboard", "multiple_last_checkpoints")
    if list(run_dir.rglob("*.tmp")):
        state.error("leaderboard", "temporary_artifact")
    if summary is not None and summary.get("best") != entries:
        state.error("leaderboard", "summary_leaderboard_mismatch")
    return (
        checkpoint_dir,
        entries,
        {
            "best_checkpoint_count": len(best_files),
            "k": k,
            "status": state.status("leaderboard"),
        },
    )


def _contains_forbidden_model_key(model_state: Mapping[object, object]) -> bool:
    for raw_name in model_state:
        if not isinstance(raw_name, str):
            return True
        parts = raw_name.split(".")
        if "aggregator" in parts:
            return True
        if any(left == "dense_head" and right == "proj_conf" for left, right in pairwise(parts)):
            return True
    return False


def _payload_privacy_codes(value: object, deny_tokens: tuple[str, ...]) -> set[str]:
    codes: set[str] = set()
    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, str):
            if "/home/" in current:
                codes.add("checkpoint_absolute_home_path")
            if any(token in current for token in deny_tokens):
                codes.add("checkpoint_private_token")
        elif isinstance(current, Mapping):
            pending.extend(current.keys())
            pending.extend(current.values())
        elif isinstance(current, (list, tuple, set)):
            pending.extend(current)
    return codes


def _audit_one_checkpoint(
    path: Path,
    *,
    role: str,
    entry: Mapping[str, Any] | None,
    summary: Mapping[str, Any] | None,
    config: Mapping[str, Any] | None,
    deny_tokens: tuple[str, ...],
    state: _AuditState,
) -> bool:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except (OSError, RuntimeError, TypeError, ValueError):
        state.error("checkpoints", "checkpoint_unreadable")
        return False
    if not isinstance(payload, dict):
        state.error("checkpoints", "checkpoint_payload_invalid")
        return False
    for code in _payload_privacy_codes(payload, deny_tokens):
        state.error("checkpoints", code)
    if payload.get("format_version") != 1 or payload.get("parameter_state") != "x":
        state.error("checkpoints", "checkpoint_header_invalid")
    model_state = payload.get("model_state")
    if not isinstance(model_state, Mapping) or not model_state:
        state.error("checkpoints", "checkpoint_model_state_invalid")
    elif _contains_forbidden_model_key(model_state):
        state.error("checkpoints", "checkpoint_forbidden_model_state")
    if summary is not None:
        if payload.get("group_fingerprint") != summary.get("group_fingerprint"):
            state.error("checkpoints", "checkpoint_group_fingerprint_mismatch")
        metadata = payload.get("metadata")
        base_checkpoint = metadata.get("base_checkpoint") if isinstance(metadata, Mapping) else None
        if base_checkpoint != summary.get("base_checkpoint"):
            state.error("checkpoints", "checkpoint_base_sha_mismatch")
        initial_head = metadata.get("initial_head_checkpoint") if isinstance(metadata, Mapping) else None
        if initial_head != summary.get("initial_head_checkpoint"):
            state.error("checkpoints", "checkpoint_initial_head_mismatch")
    if role == "best":
        if payload.get("kind") != "best":
            state.error("checkpoints", "best_checkpoint_kind_invalid")
        if "optimizer_state" in payload:
            state.error("checkpoints", "best_checkpoint_has_optimizer")
        if entry is not None and (
            payload.get("epoch") != entry.get("epoch")
            or payload.get("global_step") != entry.get("global_step")
            or payload.get("metric") != entry.get("metric")
        ):
            state.error("checkpoints", "best_checkpoint_entry_mismatch")
    else:
        if payload.get("kind") != "resume" or payload.get("checkpoint_role") != "last":
            state.error("checkpoints", "last_checkpoint_kind_invalid")
        if not isinstance(payload.get("optimizer_state"), Mapping):
            state.error("checkpoints", "last_checkpoint_missing_optimizer")
        if summary is not None:
            completed_epochs = summary.get("epochs_completed")
            if (
                not _is_nonnegative_integer(completed_epochs)
                or payload.get("epoch") != cast(int, completed_epochs) - 1
                or payload.get("global_step") != summary.get("global_step")
            ):
                state.error("checkpoints", "last_checkpoint_position_mismatch")
            training_state = payload.get("training_state")
            summary_early = summary.get("early_stopping")
            trainer_config = _nested_mapping(config, "trainer")
            configured_early = _nested_mapping(trainer_config, "early_stopping")
            early_enabled = bool(configured_early.get("enabled")) if configured_early else False
            legacy_disabled_state = training_state is None and summary_early is None and not early_enabled
            if not legacy_disabled_state and (
                not isinstance(training_state, Mapping) or training_state.get("early_stopping") != summary_early
            ):
                state.error("checkpoints", "last_checkpoint_early_stopping_mismatch")
    return True


def _audit_checkpoints(
    checkpoint_dir: Path,
    entries: list[dict[str, Any]],
    summary: Mapping[str, Any] | None,
    config: Mapping[str, Any] | None,
    deny_tokens: tuple[str, ...],
    state: _AuditState,
) -> dict[str, Any]:
    audited_count = 0
    for entry in entries:
        path = checkpoint_dir / str(entry["filename"])
        if path.is_file() and _audit_one_checkpoint(
            path,
            role="best",
            entry=entry,
            summary=summary,
            config=config,
            deny_tokens=deny_tokens,
            state=state,
        ):
            audited_count += 1
        elif not path.is_file():
            state.error("checkpoints", "checkpoint_missing")
    last_path = checkpoint_dir / "last.pt"
    has_last = last_path.is_file()
    checkpoint_config = _nested_mapping(config, "checkpoint")
    save_last = checkpoint_config.get("save_last") if checkpoint_config else None
    if save_last is True and not has_last:
        state.error("checkpoints", "configured_last_checkpoint_missing")
    if save_last is False and has_last:
        state.error("checkpoints", "disabled_last_checkpoint_present")
    if has_last and _audit_one_checkpoint(
        last_path,
        role="last",
        entry=None,
        summary=summary,
        config=config,
        deny_tokens=deny_tokens,
        state=state,
    ):
        audited_count += 1
    return {
        "audited_checkpoint_count": audited_count,
        "has_last": has_last,
        "status": state.status("checkpoints"),
    }


def _atomic_json_report(report: Mapping[str, Any], destination: Path) -> None:
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
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def audit_training_artifacts(
    run_dir: str | os.PathLike[str],
    *,
    report_path: str | os.PathLike[str],
    deny_tokens: Iterable[str] = (),
) -> dict[str, Any]:
    """Audit a run and atomically write a generic report without echoing input paths or tokens."""

    run_root = Path(run_dir).resolve(strict=False)
    report_destination = Path(report_path).resolve(strict=False)
    tokens = tuple(dict.fromkeys(token for token in deny_tokens if token))
    state = _AuditState()
    checks: dict[str, dict[str, Any]] = {}
    if not run_root.is_dir():
        state.error("summary", "run_directory_invalid")
        summary = None
        config = None
        checks["summary"] = {"status": state.status("summary")}
        checks["privacy"] = {"status": "skipped", "text_file_count": 0}
        checks["tensorboard"] = {"event_file_count": 0, "scalar_tags": [], "status": "skipped"}
        checks["leaderboard"] = {"best_checkpoint_count": 0, "k": 0, "status": "skipped"}
        checks["checkpoints"] = {"audited_checkpoint_count": 0, "has_last": False, "status": "skipped"}
    else:
        checks["privacy"] = _audit_privacy(run_root, report_destination, tokens, state)
        summary, config, checks["summary"] = _audit_summary(run_root, state)
        checks["tensorboard"] = _audit_tensorboard(run_root, config, summary, state)
        checkpoint_dir, entries, checks["leaderboard"] = _audit_leaderboard(
            run_root,
            config,
            summary,
            state,
        )
        checks["checkpoints"] = _audit_checkpoints(checkpoint_dir, entries, summary, config, tokens, state)
    report = {
        "checks": checks,
        "errors": sorted(state.errors, key=lambda error: (error["check"], error["code"])),
        "format_version": 1,
        "status": "failed" if state.errors else "passed",
    }
    _atomic_json_report(report, report_destination)
    return report
