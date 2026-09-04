from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import pytest
import torch

import vggt_omega.training.checkpointing as checkpointing
from vggt_omega.training.checkpointing import (
    TopKCheckpointManager,
    load_resume_checkpoint,
    save_resume_checkpoint,
)
from vggt_omega.training.optim.amuse import AMUSE

MONITOR = "val/objective"
GROUP_FINGERPRINT = "a" * 64


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = torch.nn.Linear(2, 2)
        self.head = torch.nn.Linear(2, 1)
        self.base.requires_grad_(False)


def _base_metadata() -> dict[str, Any]:
    return {
        "base_checkpoint": {
            "filename": "base_model.pt",
            "size_bytes": 123,
            "sha256": "b" * 64,
        }
    }


def _head_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    assert isinstance(model, _TinyModel)
    return {f"head.{name}": value for name, value in model.head.state_dict().items()}


def _update(
    manager: TopKCheckpointManager,
    *,
    epoch: int,
    metric: float,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> Any:
    model = model or _TinyModel()
    optimizer = optimizer or torch.optim.SGD([parameter for parameter in model.parameters() if parameter.requires_grad])
    return manager.update(
        epoch=epoch,
        global_step=(epoch + 1) * 10,
        metrics={MONITOR: metric},
        model=model,
        optimizer=optimizer,
        state_selector=_head_state,
        group_fingerprint=GROUP_FINGERPRINT,
        config={"profile": "test"},
        metadata=_base_metadata(),
    )


def _leaderboard(directory: Path) -> dict[str, Any]:
    return json.loads((directory / "leaderboard.json").read_text(encoding="utf-8"))


def _best_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("best_epoch_*.pt"))


def test_topk_min_prunes_to_k_and_keeps_leaderboard_consistent(tmp_path: Path) -> None:
    manager = TopKCheckpointManager(tmp_path, k=2, monitor=MONITOR, mode="min")

    for epoch, metric in enumerate((0.5, 0.4, 0.6, 0.3)):
        _update(manager, epoch=epoch, metric=metric)
        current_leaderboard = _leaderboard(tmp_path)
        assert len(_best_files(tmp_path)) <= 2
        assert {entry["filename"] for entry in current_leaderboard["entries"]} == {
            path.name for path in _best_files(tmp_path)
        }

    leaderboard = _leaderboard(tmp_path)
    assert [(entry["metric"], entry["epoch"]) for entry in leaderboard["entries"]] == [(0.3, 3), (0.4, 1)]
    assert {entry["filename"] for entry in leaderboard["entries"]} == {path.name for path in _best_files(tmp_path)}
    assert len(_best_files(tmp_path)) == 2
    assert not list(tmp_path.glob("*.tmp"))


def test_topk_ties_prefer_earlier_epoch_deterministically(tmp_path: Path) -> None:
    manager = TopKCheckpointManager(tmp_path, k=2, monitor=MONITOR, mode="min")

    _update(manager, epoch=2, metric=0.4)
    _update(manager, epoch=1, metric=0.4)
    _update(manager, epoch=3, metric=0.4)

    assert [(entry["metric"], entry["epoch"]) for entry in manager.entries] == [(0.4, 1), (0.4, 2)]


def test_topk_max_keeps_largest_metrics(tmp_path: Path) -> None:
    manager = TopKCheckpointManager(tmp_path, k=2, monitor=MONITOR, mode="max")
    for epoch, metric in enumerate((0.1, 0.3, 0.2)):
        _update(manager, epoch=epoch, metric=metric)

    assert [(entry["metric"], entry["epoch"]) for entry in manager.entries] == [(0.3, 1), (0.2, 2)]


@pytest.mark.parametrize("metric", [math.nan, math.inf, -math.inf])
def test_topk_rejects_nonfinite_metric_without_saving(tmp_path: Path, metric: float) -> None:
    manager = TopKCheckpointManager(tmp_path, k=2, monitor=MONITOR, mode="min", save_last=True)

    with pytest.raises(ValueError, match="finite"):
        _update(manager, epoch=0, metric=metric)

    assert not _best_files(tmp_path)
    assert not (tmp_path / "leaderboard.json").exists()
    assert not (tmp_path / "last.pt").exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_topk_rejects_missing_monitored_metric(tmp_path: Path) -> None:
    model = _TinyModel()
    optimizer = torch.optim.SGD(model.head.parameters(), lr=0.1)
    manager = TopKCheckpointManager(tmp_path, k=2, monitor=MONITOR, mode="min")

    with pytest.raises(KeyError, match=MONITOR):
        manager.update(
            epoch=0,
            global_step=0,
            metrics={},
            model=model,
            optimizer=optimizer,
            state_selector=_head_state,
            group_fingerprint=GROUP_FINGERPRINT,
            config={},
            metadata=_base_metadata(),
        )


def test_near_edge_topk_orders_three_best_and_rejects_missing_metric(tmp_path: Path) -> None:
    monitor = "val/near_edge_objective"
    manager = TopKCheckpointManager(tmp_path, k=3, monitor=monitor, mode="min")
    model = _TinyModel()
    optimizer = torch.optim.SGD(model.head.parameters(), lr=0.1)
    for epoch, metric in enumerate((0.5, 0.2, 0.4, 0.1)):
        manager.update(
            epoch=epoch,
            global_step=epoch + 1,
            metrics={monitor: metric},
            model=model,
            optimizer=optimizer,
            state_selector=_head_state,
            group_fingerprint=GROUP_FINGERPRINT,
            config={"checkpoint": {"monitor": monitor}},
            metadata=_base_metadata(),
        )

    assert [entry["metric"] for entry in manager.entries] == [0.1, 0.2, 0.4]
    with pytest.raises(KeyError, match=monitor):
        manager.update(
            epoch=4,
            global_step=5,
            metrics={},
            model=model,
            optimizer=optimizer,
            state_selector=_head_state,
            group_fingerprint=GROUP_FINGERPRINT,
            config={},
            metadata=_base_metadata(),
        )


def test_topk_atomic_replacements_use_same_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    replacements: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def recording_replace(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        replacements.append((source_path, destination_path))
        real_replace(source_path, destination_path)

    monkeypatch.setattr(checkpointing.os, "replace", recording_replace)
    manager = TopKCheckpointManager(tmp_path, k=2, monitor=MONITOR, mode="min")
    _update(manager, epoch=0, metric=0.5)

    assert len(replacements) == 2
    assert all(source.parent == destination.parent == tmp_path for source, destination in replacements)
    assert all(source.suffix == ".tmp" for source, _ in replacements)


def test_topk_keeps_previous_best_if_leaderboard_update_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = TopKCheckpointManager(tmp_path, k=1, monitor=MONITOR, mode="min")
    _update(manager, epoch=0, metric=0.5)
    previous_leaderboard = (tmp_path / "leaderboard.json").read_bytes()
    previous_best = _best_files(tmp_path)

    def fail_leaderboard(*args: Any, **kwargs: Any) -> None:
        raise OSError("synthetic leaderboard failure")

    monkeypatch.setattr(checkpointing, "_atomic_json_save", fail_leaderboard)
    with pytest.raises(OSError, match="synthetic leaderboard"):
        _update(manager, epoch=1, metric=0.4)

    assert (tmp_path / "leaderboard.json").read_bytes() == previous_leaderboard
    assert _best_files(tmp_path) == previous_best
    assert [(entry["metric"], entry["epoch"]) for entry in manager.entries] == [(0.5, 0)]
    assert not list(tmp_path.glob("*.tmp"))


def test_topk_rejects_untracked_best_file_without_leaderboard(tmp_path: Path) -> None:
    (tmp_path / "best_epoch_000000_orphan.pt").touch()

    with pytest.raises(ValueError, match="untracked best checkpoints"):
        TopKCheckpointManager(tmp_path, k=1, monitor=MONITOR, mode="min")


def test_best_checkpoint_uses_only_caller_selected_head_state(tmp_path: Path) -> None:
    model = _TinyModel()
    optimizer = torch.optim.SGD(model.head.parameters(), lr=0.1)
    manager = TopKCheckpointManager(tmp_path, k=1, monitor=MONITOR, mode="min")
    _update(manager, epoch=0, metric=0.5, model=model, optimizer=optimizer)

    payload = torch.load(_best_files(tmp_path)[0], map_location="cpu", weights_only=True)
    assert set(payload["model_state"]) == {"head.weight", "head.bias"}
    assert "optimizer_state" not in payload
    assert payload["parameter_state"] == "x"
    assert payload["metadata"] == _base_metadata()


@pytest.mark.parametrize("save_last", [False, True])
def test_last_checkpoint_is_optional_and_overwritten_in_place(tmp_path: Path, save_last: bool) -> None:
    model = _TinyModel()
    optimizer = torch.optim.SGD(model.head.parameters(), lr=0.1)
    manager = TopKCheckpointManager(tmp_path, k=1, monitor=MONITOR, mode="min", save_last=save_last)

    _update(manager, epoch=0, metric=0.5, model=model, optimizer=optimizer)
    _update(manager, epoch=1, metric=0.4, model=model, optimizer=optimizer)

    last_path = tmp_path / "last.pt"
    assert last_path.exists() is save_last
    assert not list(tmp_path.glob("last_epoch_*.pt"))
    if save_last:
        payload = torch.load(last_path, map_location="cpu", weights_only=True)
        assert payload["epoch"] == 1
        assert payload["global_step"] == 20
        assert payload["parameter_state"] == "x"
        assert "optimizer_state" in payload


def _make_resume_pair() -> tuple[torch.nn.Linear, AMUSE]:
    model = torch.nn.Linear(2, 2)
    optimizer = AMUSE(
        [
            {
                "params": [model.weight],
                "use_muon": True,
                "aux_update_type": "adamw",
                "lr": 1e-3,
                "momentum": 0.95,
                "weight_decay": 0.01,
            },
            {
                "params": [model.bias],
                "use_muon": False,
                "update_type": "adamw",
                "lr": 1e-4,
                "beta2": 0.999,
                "eps": 1e-10,
                "weight_decay": 0.01,
            },
        ],
        beta1=0.4,
        warmup_steps=2,
        rho=0.3,
    )
    return model, optimizer


def _resume_step(model: torch.nn.Linear, optimizer: AMUSE, step: int) -> None:
    features = torch.tensor([[1.0 + step, -0.5], [0.25, 0.75 + step]])
    target = torch.tensor([[0.1, -0.2], [0.3, 0.4]])
    loss = torch.nn.functional.mse_loss(model(features), target)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _assert_nested_close(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0.0, atol=1e-7)
    elif isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            _assert_nested_close(left[key], right[key])
    elif isinstance(left, list):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_close(left_item, right_item)
    else:
        assert left == right


def test_amuse_x_checkpoint_resume_matches_uninterrupted_next_step(tmp_path: Path) -> None:
    torch.manual_seed(7)
    uninterrupted_model, uninterrupted_optimizer = _make_resume_pair()
    uninterrupted_optimizer.train()
    _resume_step(uninterrupted_model, uninterrupted_optimizer, 0)
    _resume_step(uninterrupted_model, uninterrupted_optimizer, 1)
    checkpoint_path = tmp_path / "resume.pt"

    save_resume_checkpoint(
        checkpoint_path,
        epoch=1,
        global_step=2,
        model=uninterrupted_model,
        optimizer=uninterrupted_optimizer,
        state_selector=lambda model: model.state_dict(),
        group_fingerprint=GROUP_FINGERPRINT,
        config={"profile": "test"},
        metadata=_base_metadata(),
    )
    assert uninterrupted_optimizer.train_mode is True
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert payload["parameter_state"] == "x"

    _resume_step(uninterrupted_model, uninterrupted_optimizer, 2)
    uninterrupted_model_state = uninterrupted_model.state_dict()
    uninterrupted_optimizer_state = uninterrupted_optimizer.state_dict()

    resumed_model, resumed_optimizer = _make_resume_pair()
    resume_state = load_resume_checkpoint(
        checkpoint_path,
        model=resumed_model,
        optimizer=resumed_optimizer,
        expected_group_fingerprint=GROUP_FINGERPRINT,
    )
    assert resumed_optimizer.train_mode is True
    assert resume_state.epoch == 1
    assert resume_state.global_step == 2
    _resume_step(resumed_model, resumed_optimizer, 2)

    _assert_nested_close(uninterrupted_model_state, resumed_model.state_dict())
    _assert_nested_close(uninterrupted_optimizer_state, resumed_optimizer.state_dict())


def test_resume_rejects_group_fingerprint_mismatch(tmp_path: Path) -> None:
    model, optimizer = _make_resume_pair()
    optimizer.train()
    _resume_step(model, optimizer, 0)
    checkpoint_path = tmp_path / "resume.pt"
    save_resume_checkpoint(
        checkpoint_path,
        epoch=0,
        global_step=1,
        model=model,
        optimizer=optimizer,
        state_selector=lambda model: model.state_dict(),
        group_fingerprint=GROUP_FINGERPRINT,
        config={},
        metadata=_base_metadata(),
    )

    resumed_model, resumed_optimizer = _make_resume_pair()
    with pytest.raises(ValueError, match="fingerprint"):
        load_resume_checkpoint(
            checkpoint_path,
            model=resumed_model,
            optimizer=resumed_optimizer,
            expected_group_fingerprint="c" * 64,
        )
    assert resumed_optimizer.train_mode is False


def test_resume_validates_metadata_before_mutating_model_or_optimizer(tmp_path: Path) -> None:
    saved_model, saved_optimizer = _make_resume_pair()
    saved_optimizer.train()
    _resume_step(saved_model, saved_optimizer, 0)
    checkpoint_path = tmp_path / "resume.pt"
    save_resume_checkpoint(
        checkpoint_path,
        epoch=0,
        global_step=1,
        model=saved_model,
        optimizer=saved_optimizer,
        state_selector=lambda model: model.state_dict(),
        group_fingerprint=GROUP_FINGERPRINT,
        config={},
        metadata=_base_metadata(),
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    payload["metadata"] = {}
    torch.save(payload, checkpoint_path)

    resumed_model, resumed_optimizer = _make_resume_pair()
    model_before = {name: value.detach().clone() for name, value in resumed_model.state_dict().items()}
    with pytest.raises(ValueError, match=r"metadata\.base_checkpoint"):
        load_resume_checkpoint(
            checkpoint_path,
            model=resumed_model,
            optimizer=resumed_optimizer,
            expected_group_fingerprint=GROUP_FINGERPRINT,
        )

    _assert_nested_close(model_before, resumed_model.state_dict())
    assert resumed_optimizer.train_mode is False


def test_atomic_save_failure_restores_amuse_train_mode_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, optimizer = _make_resume_pair()
    optimizer.train()
    _resume_step(model, optimizer, 0)

    def fail_save(*args: Any, **kwargs: Any) -> None:
        raise OSError("synthetic save failure")

    monkeypatch.setattr(checkpointing.torch, "save", fail_save)
    with pytest.raises(OSError, match="synthetic"):
        save_resume_checkpoint(
            tmp_path / "resume.pt",
            epoch=0,
            global_step=1,
            model=model,
            optimizer=optimizer,
            state_selector=lambda model: model.state_dict(),
            group_fingerprint=GROUP_FINGERPRINT,
            config={},
            metadata=_base_metadata(),
        )

    assert optimizer.train_mode is True
    assert not (tmp_path / "resume.pt").exists()
    assert not list(tmp_path.glob("*.tmp"))
