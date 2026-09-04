from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import struct
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import torch

StateSelector = Callable[[torch.nn.Module], Mapping[str, torch.Tensor]]
StateLoader = Callable[[torch.nn.Module, Mapping[str, torch.Tensor]], None]


class _EvaluationWeightOptimizer(Protocol):
    train_mode: bool

    def eval(self) -> None: ...

    def train(self) -> None: ...


@dataclass(frozen=True)
class _CheckpointEntry:
    epoch: int
    global_step: int
    metric: float
    filename: str

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            "epoch": self.epoch,
            "filename": self.filename,
            "global_step": self.global_step,
            "metric": self.metric,
        }


@dataclass(frozen=True)
class ResumeState:
    epoch: int
    global_step: int
    config: dict[str, Any]
    metadata: dict[str, Any]
    group_fingerprint: str
    training_state: dict[str, Any] | None


def _validate_nonnegative_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _validate_group_fingerprint(group_fingerprint: str) -> None:
    if len(group_fingerprint) != 64 or any(character not in "0123456789abcdef" for character in group_fingerprint):
        raise ValueError("group_fingerprint must be a lowercase SHA-256 hexadecimal digest")


def _validate_metadata(metadata: Mapping[str, Any]) -> None:
    base_checkpoint = metadata.get("base_checkpoint")
    if not isinstance(base_checkpoint, Mapping):
        raise ValueError("metadata.base_checkpoint is required")
    required = {"filename", "size_bytes", "sha256"}
    missing = required - set(base_checkpoint)
    if missing:
        raise ValueError(f"metadata.base_checkpoint is missing fields: {sorted(missing)}")

    filename = base_checkpoint["filename"]
    if not isinstance(filename, str) or not filename or Path(filename).name != filename:
        raise ValueError("base checkpoint filename must be a basename without a directory")
    size_bytes = base_checkpoint["size_bytes"]
    if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
        raise ValueError("base checkpoint size_bytes must be a non-negative integer")
    sha256 = base_checkpoint["sha256"]
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
    ):
        raise ValueError("base checkpoint sha256 must be a lowercase hexadecimal digest")


def _snapshot_selected_state(model: torch.nn.Module, state_selector: StateSelector) -> dict[str, torch.Tensor]:
    selected_state = state_selector(model)
    if not isinstance(selected_state, Mapping) or not selected_state:
        raise ValueError("state_selector must return a non-empty mapping")

    snapshot: dict[str, torch.Tensor] = {}
    for name, value in selected_state.items():
        if not isinstance(name, str) or not name:
            raise ValueError("state_selector keys must be non-empty strings")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"state_selector value for {name!r} is not a tensor")
        snapshot[name] = value.detach().cpu().clone()
    return snapshot


def _atomic_torch_save(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary_path)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_json_save(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _optimizer_has_evaluation_weights(optimizer: torch.optim.Optimizer) -> bool:
    return (
        hasattr(optimizer, "train_mode")
        and callable(getattr(optimizer, "train", None))
        and callable(getattr(optimizer, "eval", None))
    )


@contextmanager
def optimizer_evaluation_state(optimizer: torch.optim.Optimizer) -> Iterator[None]:
    """Temporarily expose AMUSE's averaged x weights and restore its prior mode."""

    if not _optimizer_has_evaluation_weights(optimizer):
        yield
        return

    evaluation_optimizer = cast(_EvaluationWeightOptimizer, optimizer)
    was_training = bool(evaluation_optimizer.train_mode)
    evaluation_optimizer.eval()
    try:
        yield
    finally:
        if was_training:
            evaluation_optimizer.train()


def _checkpoint_payload(
    *,
    kind: str,
    epoch: int,
    global_step: int,
    model_state: Mapping[str, torch.Tensor],
    group_fingerprint: str,
    config: Mapping[str, Any],
    metadata: Mapping[str, Any],
    monitor: str | None = None,
    metric: float | None = None,
    optimizer_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "config": copy.deepcopy(dict(config)),
        "epoch": epoch,
        "format_version": 1,
        "global_step": global_step,
        "group_fingerprint": group_fingerprint,
        "kind": kind,
        "metadata": copy.deepcopy(dict(metadata)),
        "model_state": dict(model_state),
        "parameter_state": "x",
    }
    if monitor is not None:
        payload["monitor"] = monitor
    if metric is not None:
        payload["metric"] = metric
    if optimizer_state is not None:
        payload["optimizer_state"] = dict(optimizer_state)
    return payload


def save_resume_checkpoint(
    destination: str | os.PathLike[str],
    *,
    epoch: int,
    global_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    state_selector: StateSelector,
    group_fingerprint: str,
    config: Mapping[str, Any],
    metadata: Mapping[str, Any],
    training_state: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save model and optimizer from evaluation-weight state x."""

    _validate_nonnegative_integer("epoch", epoch)
    _validate_nonnegative_integer("global_step", global_step)
    _validate_group_fingerprint(group_fingerprint)
    _validate_metadata(metadata)
    destination_path = Path(destination)

    with optimizer_evaluation_state(optimizer):
        model_state = _snapshot_selected_state(model, state_selector)
        optimizer_state = optimizer.state_dict()
        payload = _checkpoint_payload(
            kind="resume",
            epoch=epoch,
            global_step=global_step,
            model_state=model_state,
            optimizer_state=optimizer_state,
            group_fingerprint=group_fingerprint,
            config=config,
            metadata=metadata,
        )
        if training_state is not None:
            payload["training_state"] = copy.deepcopy(dict(training_state))
        _atomic_torch_save(payload, destination_path)
    return destination_path


def _load_payload(path: Path, *, map_location: str | torch.device) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dictionary")
    if payload.get("format_version") != 1:
        raise ValueError(f"unsupported checkpoint format_version: {payload.get('format_version')!r}")
    if payload.get("parameter_state") != "x":
        raise ValueError("resume requires parameter_state=x")
    return payload


def load_resume_checkpoint(
    checkpoint_path: str | os.PathLike[str],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_group_fingerprint: str,
    state_loader: StateLoader | None = None,
    map_location: str | torch.device = "cpu",
) -> ResumeState:
    """Load an x-state checkpoint and enter AMUSE training mode exactly once."""

    _validate_group_fingerprint(expected_group_fingerprint)
    payload = _load_payload(Path(checkpoint_path), map_location=map_location)
    if payload.get("kind") != "resume":
        raise ValueError(f"checkpoint is not resumable: kind={payload.get('kind')!r}")
    actual_fingerprint = payload.get("group_fingerprint")
    if actual_fingerprint != expected_group_fingerprint:
        raise ValueError(
            "optimizer parameter group fingerprint mismatch: "
            f"checkpoint={actual_fingerprint!r}, expected={expected_group_fingerprint!r}"
        )

    epoch = _validate_nonnegative_integer("checkpoint epoch", payload.get("epoch"))
    global_step = _validate_nonnegative_integer("checkpoint global_step", payload.get("global_step"))
    config = payload.get("config")
    metadata = payload.get("metadata")
    if not isinstance(config, dict) or not isinstance(metadata, dict):
        raise ValueError("checkpoint config/metadata must be dictionaries")
    _validate_metadata(metadata)
    training_state = payload.get("training_state")
    if training_state is not None and not isinstance(training_state, dict):
        raise ValueError("checkpoint training_state must be a dictionary when present")

    if _optimizer_has_evaluation_weights(optimizer):
        evaluation_optimizer = cast(_EvaluationWeightOptimizer, optimizer)
        if evaluation_optimizer.train_mode:
            raise ValueError("optimizer must be in evaluation mode before loading an x-state checkpoint")

    model_state = payload.get("model_state")
    optimizer_state = payload.get("optimizer_state")
    if not isinstance(model_state, Mapping):
        raise ValueError("checkpoint model_state is missing or invalid")
    if not isinstance(optimizer_state, Mapping):
        raise ValueError("checkpoint optimizer_state is missing or invalid")

    if state_loader is None:
        model.load_state_dict(model_state, strict=True)
    else:
        state_loader(model, model_state)
    optimizer.load_state_dict(optimizer_state)
    if _optimizer_has_evaluation_weights(optimizer):
        cast(_EvaluationWeightOptimizer, optimizer).train()

    return ResumeState(
        epoch=epoch,
        global_step=global_step,
        config=config,
        metadata=metadata,
        group_fingerprint=expected_group_fingerprint,
        training_state=training_state,
    )


class TopKCheckpointManager:
    """Maintain deterministic top-K head checkpoints and an optional last.pt."""

    def __init__(
        self,
        directory: str | os.PathLike[str],
        *,
        k: int,
        monitor: str,
        mode: str,
        save_last: bool = False,
    ) -> None:
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be >= 1")
        if not monitor:
            raise ValueError("monitor must be non-empty")
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.k = k
        self.monitor = monitor
        self.mode = mode
        self.save_last = save_last
        self.leaderboard_path = self.directory / "leaderboard.json"
        self.last_path = self.directory / "last.pt"
        self._entries = self._load_entries()

    @property
    def entries(self) -> tuple[dict[str, int | float | str], ...]:
        return tuple(entry.to_dict() for entry in self._entries)

    def _sort_key(self, entry: _CheckpointEntry) -> tuple[float, int, str]:
        primary = entry.metric if self.mode == "min" else -entry.metric
        return primary, entry.epoch, entry.filename

    def _sorted_topk(self, entries: list[_CheckpointEntry]) -> list[_CheckpointEntry]:
        return sorted(entries, key=self._sort_key)[: self.k]

    def _leaderboard_payload(self, entries: list[_CheckpointEntry]) -> dict[str, Any]:
        return {
            "entries": [entry.to_dict() for entry in entries],
            "format_version": 1,
            "k": self.k,
            "mode": self.mode,
            "monitor": self.monitor,
        }

    def _load_entries(self) -> list[_CheckpointEntry]:
        best_files = {path.name for path in self.directory.glob("best_epoch_*.pt") if path.is_file()}
        if not self.leaderboard_path.exists():
            if best_files:
                raise ValueError(f"checkpoint directory contains untracked best checkpoints: {sorted(best_files)}")
            return []
        payload = json.loads(self.leaderboard_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("leaderboard payload must be a dictionary")
        expected_header = {"format_version": 1, "k": self.k, "mode": self.mode, "monitor": self.monitor}
        for key, expected in expected_header.items():
            if payload.get(key) != expected:
                raise ValueError(
                    f"leaderboard {key} does not match manager configuration: "
                    f"stored={payload.get(key)!r}, expected={expected!r}"
                )

        entries: list[_CheckpointEntry] = []
        raw_entries = payload.get("entries")
        if not isinstance(raw_entries, list):
            raise ValueError("leaderboard entries must be a list")
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, dict):
                raise ValueError("each leaderboard entry must be a dictionary")
            entry = _CheckpointEntry(
                epoch=raw_entry.get("epoch"),
                global_step=raw_entry.get("global_step"),
                metric=float(raw_entry["metric"]),
                filename=str(raw_entry["filename"]),
            )
            _validate_nonnegative_integer("leaderboard epoch", entry.epoch)
            _validate_nonnegative_integer("leaderboard global_step", entry.global_step)
            if not math.isfinite(entry.metric):
                raise ValueError("leaderboard contains a non-finite metric")
            if not entry.filename or Path(entry.filename).name != entry.filename:
                raise ValueError("leaderboard filename must be a basename")
            if not (self.directory / entry.filename).is_file():
                raise ValueError(f"leaderboard references a missing checkpoint: {entry.filename}")
            entries.append(entry)
        if len(entries) > self.k:
            raise ValueError("leaderboard contains more than k entries")
        if len({entry.epoch for entry in entries}) != len(entries):
            raise ValueError("leaderboard contains duplicate epochs")
        if len({entry.filename for entry in entries}) != len(entries):
            raise ValueError("leaderboard contains duplicate filenames")
        if entries != self._sorted_topk(entries):
            raise ValueError("leaderboard entries are not in deterministic rank order")
        referenced_files = {entry.filename for entry in entries}
        if best_files != referenced_files:
            untracked = sorted(best_files - referenced_files)
            raise ValueError(f"checkpoint directory contains untracked best checkpoints: {untracked}")
        return entries

    def _candidate_filename(self, epoch: int, metric: float) -> str:
        identity = struct.pack(">qd", epoch, metric)
        suffix = hashlib.sha256(identity).hexdigest()[:12]
        return f"best_epoch_{epoch:06d}_{suffix}.pt"

    def _validate_metric(self, metrics: Mapping[str, float]) -> float:
        if self.monitor not in metrics:
            raise KeyError(f"missing monitored metric: {self.monitor}")
        metric = float(metrics[self.monitor])
        if not math.isfinite(metric):
            raise ValueError(f"monitored metric {self.monitor} must be finite")
        return metric

    def _save_best_payload(
        self,
        *,
        destination: Path,
        entry: _CheckpointEntry,
        model_state: Mapping[str, torch.Tensor],
        group_fingerprint: str,
        config: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> None:
        payload = _checkpoint_payload(
            kind="best",
            epoch=entry.epoch,
            global_step=entry.global_step,
            model_state=model_state,
            group_fingerprint=group_fingerprint,
            config=config,
            metadata=metadata,
            monitor=self.monitor,
            metric=entry.metric,
        )
        _atomic_torch_save(payload, destination)

    def _save_last_payload(
        self,
        *,
        epoch: int,
        global_step: int,
        model_state: Mapping[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        group_fingerprint: str,
        config: Mapping[str, Any],
        metadata: Mapping[str, Any],
        training_state: Mapping[str, Any] | None,
    ) -> None:
        payload = _checkpoint_payload(
            kind="resume",
            epoch=epoch,
            global_step=global_step,
            model_state=model_state,
            optimizer_state=optimizer.state_dict(),
            group_fingerprint=group_fingerprint,
            config=config,
            metadata=metadata,
        )
        payload["checkpoint_role"] = "last"
        if training_state is not None:
            payload["training_state"] = copy.deepcopy(dict(training_state))
        _atomic_torch_save(payload, self.last_path)

    def update(
        self,
        *,
        epoch: int,
        global_step: int,
        metrics: Mapping[str, float],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        state_selector: StateSelector,
        group_fingerprint: str,
        config: Mapping[str, Any],
        metadata: Mapping[str, Any],
        training_state: Mapping[str, Any] | None = None,
    ) -> dict[str, int | float | str] | None:
        _validate_nonnegative_integer("epoch", epoch)
        _validate_nonnegative_integer("global_step", global_step)
        _validate_group_fingerprint(group_fingerprint)
        _validate_metadata(metadata)
        metric = self._validate_metric(metrics)

        matching_epoch = [entry for entry in self._entries if entry.epoch == epoch]
        if matching_epoch:
            previous = matching_epoch[0]
            if previous.metric != metric or previous.global_step != global_step:
                raise ValueError(f"epoch {epoch} already has a different checkpoint entry")
            candidate = previous
            proposed_entries = list(self._entries)
            candidate_is_new = False
        else:
            candidate = _CheckpointEntry(
                epoch=epoch,
                global_step=global_step,
                metric=metric,
                filename=self._candidate_filename(epoch, metric),
            )
            proposed_entries = self._sorted_topk([*self._entries, candidate])
            candidate_is_new = candidate in proposed_entries

        needs_model_state = candidate_is_new or self.save_last
        if not needs_model_state:
            return None

        with optimizer_evaluation_state(optimizer):
            model_state = _snapshot_selected_state(model, state_selector)
            if candidate_is_new:
                candidate_path = self.directory / candidate.filename
                candidate_existed = candidate_path.exists()
                try:
                    self._save_best_payload(
                        destination=candidate_path,
                        entry=candidate,
                        model_state=model_state,
                        group_fingerprint=group_fingerprint,
                        config=config,
                        metadata=metadata,
                    )
                    _atomic_json_save(self._leaderboard_payload(proposed_entries), self.leaderboard_path)
                except Exception:
                    if not candidate_existed:
                        candidate_path.unlink(missing_ok=True)
                    raise

                obsolete_filenames = {entry.filename for entry in self._entries} - {
                    entry.filename for entry in proposed_entries
                }
                self._entries = proposed_entries
                for filename in sorted(obsolete_filenames):
                    (self.directory / filename).unlink(missing_ok=True)

            if self.save_last:
                self._save_last_payload(
                    epoch=epoch,
                    global_step=global_step,
                    model_state=model_state,
                    optimizer=optimizer,
                    group_fingerprint=group_fingerprint,
                    config=config,
                    metadata=metadata,
                    training_state=training_state,
                )

        return candidate.to_dict() if candidate_is_new else None
