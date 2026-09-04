from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch

import vggt_omega.training.evaluation as evaluation_module
from scripts.evaluate_training_checkpoints import main as evaluation_cli_main
from vggt_omega.training.evaluation import (
    EvaluationError,
    _validation_loss_options,
    evaluate_training_checkpoints,
)
from vggt_omega.training.model_factory import PreparedTrainingModel

GROUP_FINGERPRINT = "a" * 64


class _TinyHeadModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frozen = torch.nn.Parameter(torch.tensor(-1.0), requires_grad=False)
        self.head = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        raise AssertionError("the injected validator owns tiny-model evaluation")


class _TinyDataset(torch.utils.data.Dataset[dict[str, torch.Tensor]]):
    def __init__(self) -> None:
        self.epoch = -1

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"sample": torch.tensor(index)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_validation_loss_options_are_read_from_resolved_config() -> None:
    config = {
        "loss": {
            "validation": {
                "camera_weight": 4.0,
                "depth_weight": 0.75,
                "translation_weight": 2.0,
                "rotation_weight": 3.0,
                "fov_weight": 0.25,
            }
        }
    }

    assert _validation_loss_options(config) == {
        "camera_weight": 4.0,
        "depth_weight": 0.75,
        "translation_weight": 2.0,
        "rotation_weight": 3.0,
        "fov_weight": 0.25,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolved_config() -> dict[str, Any]:
    return {
        "checkpoint": {
            "directory": "checkpoints",
            "k": 2,
            "mode": "min",
            "monitor": "val/objective",
            "save_last": True,
        },
        "data": {
            "batch_size": 1,
            "max_frames": 4,
            "min_frames": 2,
            "min_valid_depth_pixels": 1,
            "num_workers": 0,
            "pin_memory": False,
            "root": "../private-staging",
            "smoke_split": "smoke",
            "train_split": "train",
            "val_split": "val",
        },
        "model": {
            "precision": "bf16",
            "pretrained_checkpoint": "models/base.pt",
        },
        "seed": 17,
        "trainer": {
            "deterministic": True,
            "early_stopping": {
                "enabled": False,
                "min_delta": 0.0,
                "mode": "min",
                "monitor": "val/objective",
                "patience": 2,
            },
            "epochs": 2,
            "max_val_batches": None,
            "name": "finetune",
            "resume_from": None,
            "sequence_frames": 2,
        },
    }


def _make_run(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    original_cwd = tmp_path / "original-private-root"
    run_dir = tmp_path / "private-run-name"
    (original_cwd / "models").mkdir(parents=True)
    (tmp_path / "private-staging").mkdir()
    (run_dir / "checkpoints").mkdir(parents=True)
    base_path = original_cwd / "models" / "base.pt"
    base_path.write_bytes(b"strict-base")
    base_metadata = {
        "filename": base_path.name,
        "sha256": _sha256(base_path),
        "size_bytes": base_path.stat().st_size,
    }
    config = _resolved_config()
    entries: list[dict[str, Any]] = []
    for epoch, step, metric, value in ((0, 10, 1.25, 1.25), (1, 20, 1.5, 1.5)):
        filename = f"best_epoch_{epoch:06d}_generic.pt"
        entry = {"epoch": epoch, "filename": filename, "global_step": step, "metric": metric}
        entries.append(entry)
        torch.save(
            {
                "config": config,
                "epoch": epoch,
                "format_version": 1,
                "global_step": step,
                "group_fingerprint": GROUP_FINGERPRINT,
                "kind": "best",
                "metadata": {"base_checkpoint": base_metadata},
                "metric": metric,
                "model_state": {"head": torch.tensor(value)},
                "monitor": "val/objective",
                "parameter_state": "x",
            },
            run_dir / "checkpoints" / filename,
        )
    _write_json(
        run_dir / "checkpoints" / "leaderboard.json",
        {
            "entries": entries,
            "format_version": 1,
            "k": 2,
            "mode": "min",
            "monitor": "val/objective",
        },
    )
    _write_json(run_dir / "resolved_config.json", config)
    early_stopping = {
        "bad_epochs": 0,
        "best": None,
        "enabled": False,
        "min_delta": 0.0,
        "mode": "min",
        "monitor": "val/objective",
        "patience": 2,
        "stopped": False,
    }
    summary = {
        "base_checkpoint": base_metadata,
        "best": entries,
        "early_stopping": early_stopping,
        "epochs_completed": 2,
        "global_step": 20,
        "group_fingerprint": GROUP_FINGERPRINT,
        "status": "complete",
        "stopped_early": False,
        "train": {"objective": 2.0},
        "validation": {"objective": 1.5},
    }
    _write_json(run_dir / "run_summary.json", summary)
    torch.save(
        {
            "checkpoint_role": "last",
            "config": config,
            "epoch": 1,
            "format_version": 1,
            "global_step": 20,
            "group_fingerprint": GROUP_FINGERPRINT,
            "kind": "resume",
            "metadata": {"base_checkpoint": base_metadata},
            "model_state": {"head": torch.tensor(1.5)},
            "optimizer_state": {},
            "parameter_state": "x",
            "training_state": {"early_stopping": early_stopping},
        },
        run_dir / "checkpoints" / "last.pt",
    )
    return run_dir, original_cwd, config


def _evaluation_dependencies():
    builds: list[tuple[Path, torch.device]] = []
    datasets: list[tuple[Path, str, dict[str, Any], _TinyDataset]] = []
    precisions: list[str] = []

    def model_factory(checkpoint: Path, *, device: torch.device) -> PreparedTrainingModel:
        builds.append((checkpoint, device))
        return PreparedTrainingModel(model=_TinyHeadModel().to(device), trainable_parameter_names=("head",))

    def dataset_factory(root: Path, *, split: str, **options: Any) -> _TinyDataset:
        dataset = _TinyDataset()
        datasets.append((root, split, options, dataset))
        return dataset

    def validator(*, model: torch.nn.Module, batches: Any, precision: str, **kwargs: Any) -> dict[str, float]:
        assert next(iter(batches))["sample"].tolist() == [0]
        precisions.append(precision)
        value = float(model.get_parameter("head").detach())
        return {"camera": value / 5, "depth": 0.0, "objective": value}

    return builds, datasets, precisions, model_factory, dataset_factory, validator


def test_evaluator_strictly_checks_and_recomputes_every_best_without_paths(tmp_path: Path) -> None:
    run_dir, original_cwd, _ = _make_run(tmp_path)
    output = tmp_path / "reports" / "final_evaluation.json"
    builds, datasets, precisions, model_factory, dataset_factory, validator = _evaluation_dependencies()

    report = evaluate_training_checkpoints(
        run_dir,
        output_path=output,
        original_cwd=original_cwd,
        device="cpu",
        tolerance=0.0,
        model_factory=model_factory,
        dataset_factory=dataset_factory,
        validator=validator,
    )

    assert report["status"] == "passed"
    assert len(builds) == 1
    assert builds[0][0] == original_cwd / "models" / "base.pt"
    assert len(datasets) == 1
    assert datasets[0][0] == tmp_path / "private-staging"
    assert datasets[0][1] == "val"
    assert datasets[0][2] == {
        "max_frames": 2,
        "min_frames": 2,
        "min_valid_depth_pixels": 1,
        "seed": 17,
    }
    assert datasets[0][3].epoch == 0
    assert precisions == ["bf16", "bf16"]
    assert [item["stored_metric"] for item in report["checkpoints"]] == [1.25, 1.5]
    assert [item["recomputed_metrics"]["objective"] for item in report["checkpoints"]] == [1.25, 1.5]
    assert all(len(item["sha256"]) == 64 for item in report["checkpoints"])
    assert json.loads(output.read_text(encoding="utf-8")) == report
    rendered = output.read_text(encoding="utf-8")
    assert str(run_dir) not in rendered
    assert str(original_cwd) not in rendered
    assert "private-run-name" not in rendered
    assert "private-staging" not in rendered
    assert not list(output.parent.glob(".*.tmp"))


def test_evaluator_accepts_legacy_disabled_early_stopping_artifacts(tmp_path: Path) -> None:
    run_dir, original_cwd, config = _make_run(tmp_path)
    del config["trainer"]["early_stopping"]
    _write_json(run_dir / "resolved_config.json", config)
    summary_path = run_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text())
    del summary["early_stopping"]
    _write_json(summary_path, summary)
    for checkpoint in (run_dir / "checkpoints").glob("best_epoch_*.pt"):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        payload["config"] = config
        torch.save(payload, checkpoint)
    last_path = run_dir / "checkpoints" / "last.pt"
    last = torch.load(last_path, map_location="cpu", weights_only=True)
    last["config"] = config
    del last["training_state"]
    torch.save(last, last_path)
    *_, model_factory, dataset_factory, validator = _evaluation_dependencies()

    report = evaluate_training_checkpoints(
        run_dir,
        output_path=tmp_path / "report.json",
        original_cwd=original_cwd,
        device="cpu",
        model_factory=model_factory,
        dataset_factory=dataset_factory,
        validator=validator,
    )

    assert report["status"] == "passed"


def test_evaluator_recomputes_configured_translation_monitor(tmp_path: Path) -> None:
    run_dir, original_cwd, config = _make_run(tmp_path)
    config["checkpoint"]["monitor"] = "val/camera_translation"
    _write_json(run_dir / "resolved_config.json", config)
    leaderboard_path = run_dir / "checkpoints" / "leaderboard.json"
    leaderboard = json.loads(leaderboard_path.read_text())
    leaderboard["monitor"] = "val/camera_translation"
    _write_json(leaderboard_path, leaderboard)
    for checkpoint in (run_dir / "checkpoints").glob("best_epoch_*.pt"):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        payload["config"] = config
        payload["monitor"] = "val/camera_translation"
        torch.save(payload, checkpoint)
    last_checkpoint = run_dir / "checkpoints" / "last.pt"
    last_payload = torch.load(last_checkpoint, map_location="cpu", weights_only=True)
    last_payload["config"] = config
    torch.save(last_payload, last_checkpoint)
    *_, model_factory, dataset_factory, _ = _evaluation_dependencies()

    def validator(*, model: torch.nn.Module, **kwargs: Any) -> dict[str, float]:
        value = float(model.get_parameter("head").detach())
        return {
            "camera": value,
            "camera_translation": value,
            "camera_rotation": 0.0,
            "camera_fov": 0.0,
            "depth": 0.0,
            "objective": value * 5.0,
        }

    report = evaluate_training_checkpoints(
        run_dir,
        output_path=tmp_path / "report.json",
        original_cwd=original_cwd,
        device="cpu",
        model_factory=model_factory,
        dataset_factory=dataset_factory,
        validator=validator,
    )

    assert report["status"] == "passed"
    assert report["monitor"] == "val/camera_translation"
    assert all(item["metric_absolute_error"] == 0.0 for item in report["checkpoints"])


def test_evaluator_rejects_false_early_stop_claim(tmp_path: Path) -> None:
    run_dir, original_cwd, _ = _make_run(tmp_path)
    summary_path = run_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["stopped_early"] = True
    summary["epochs_completed"] = 1
    summary["early_stopping"]["stopped"] = True
    summary["early_stopping"]["bad_epochs"] = 2
    _write_json(summary_path, summary)
    *_, model_factory, dataset_factory, validator = _evaluation_dependencies()

    with pytest.raises(EvaluationError, match="disabled early stopping"):
        evaluate_training_checkpoints(
            run_dir,
            output_path=tmp_path / "report.json",
            original_cwd=original_cwd,
            device="cpu",
            model_factory=model_factory,
            dataset_factory=dataset_factory,
            validator=validator,
        )


def test_evaluator_rejects_last_checkpoint_position_mismatch(tmp_path: Path) -> None:
    run_dir, original_cwd, _ = _make_run(tmp_path)
    last = run_dir / "checkpoints" / "last.pt"
    payload = torch.load(last, map_location="cpu", weights_only=True)
    payload["global_step"] = 19
    torch.save(payload, last)
    *_, model_factory, dataset_factory, validator = _evaluation_dependencies()

    with pytest.raises(EvaluationError, match="position"):
        evaluate_training_checkpoints(
            run_dir,
            output_path=tmp_path / "report.json",
            original_cwd=original_cwd,
            device="cpu",
            model_factory=model_factory,
            dataset_factory=dataset_factory,
            validator=validator,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("kind", "kind"),
        ("parameter_state", "parameter_state"),
        ("head_keys", "trainable"),
        ("base_sha", "base"),
        ("group", "fingerprint"),
        ("epoch", "epoch"),
        ("step", "global_step"),
        ("metric", "metric"),
        ("monitor", "monitor"),
        ("optimizer", "optimizer"),
    ],
)
def test_evaluator_rejects_checkpoint_contract_mismatches(tmp_path: Path, mutation: str, message: str) -> None:
    run_dir, original_cwd, _ = _make_run(tmp_path)
    checkpoint = sorted((run_dir / "checkpoints").glob("best_epoch_*.pt"))[0]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if mutation == "kind":
        payload["kind"] = "resume"
    elif mutation == "parameter_state":
        payload["parameter_state"] = "y"
    elif mutation == "head_keys":
        payload["model_state"] = {"unexpected": torch.tensor(1.25)}
    elif mutation == "base_sha":
        payload["metadata"]["base_checkpoint"]["sha256"] = "b" * 64
    elif mutation == "group":
        payload["group_fingerprint"] = "b" * 64
    elif mutation == "epoch":
        payload["epoch"] = 9
    elif mutation == "step":
        payload["global_step"] = 9
    elif mutation == "metric":
        payload["metric"] = 9.0
    elif mutation == "monitor":
        payload["monitor"] = "val/depth"
    elif mutation == "optimizer":
        payload["optimizer_state"] = {}
    torch.save(payload, checkpoint)
    *_, model_factory, dataset_factory, validator = _evaluation_dependencies()

    with pytest.raises(EvaluationError, match=message):
        evaluate_training_checkpoints(
            run_dir,
            output_path=tmp_path / "report.json",
            original_cwd=original_cwd,
            device="cpu",
            model_factory=model_factory,
            dataset_factory=dataset_factory,
            validator=validator,
        )

    assert not (tmp_path / "report.json").exists()


def test_evaluator_rejects_metric_outside_absolute_tolerance(tmp_path: Path) -> None:
    run_dir, original_cwd, _ = _make_run(tmp_path)
    *_, model_factory, dataset_factory, _ = _evaluation_dependencies()

    def validator(**kwargs: Any) -> dict[str, float]:
        return {"camera": 0.0, "depth": 0.0, "objective": 1.2511}

    with pytest.raises(EvaluationError, match="tolerance"):
        evaluate_training_checkpoints(
            run_dir,
            output_path=tmp_path / "report.json",
            original_cwd=original_cwd,
            device="cpu",
            tolerance=1e-3,
            model_factory=model_factory,
            dataset_factory=dataset_factory,
            validator=validator,
        )


@pytest.mark.parametrize("value", [-1.0, math.inf, math.nan])
def test_evaluator_rejects_invalid_tolerance(tmp_path: Path, value: float) -> None:
    run_dir, original_cwd, _ = _make_run(tmp_path)

    with pytest.raises(ValueError, match="tolerance"):
        evaluate_training_checkpoints(
            run_dir,
            output_path=tmp_path / "report.json",
            original_cwd=original_cwd,
            device="cpu",
            tolerance=value,
        )


def test_evaluator_loads_best_checkpoints_with_safe_torch_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, original_cwd, _ = _make_run(tmp_path)
    *_, model_factory, dataset_factory, validator = _evaluation_dependencies()
    real_load = torch.load
    calls: list[dict[str, Any]] = []

    def recording_load(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(evaluation_module.torch, "load", recording_load)
    evaluate_training_checkpoints(
        run_dir,
        output_path=tmp_path / "report.json",
        original_cwd=original_cwd,
        device="cpu",
        model_factory=model_factory,
        dataset_factory=dataset_factory,
        validator=validator,
    )

    assert calls
    assert all(call["weights_only"] is True and call["mmap"] is True for call in calls)
    assert all(call["map_location"] == "cpu" for call in calls)


def test_evaluator_accepts_only_resume_path_as_a_checkpoint_config_difference(tmp_path: Path) -> None:
    run_dir, original_cwd, config = _make_run(tmp_path)
    config["trainer"]["resume_from"] = "previous-run/checkpoints/last.pt"
    _write_json(run_dir / "resolved_config.json", config)
    *_, model_factory, dataset_factory, validator = _evaluation_dependencies()

    report = evaluate_training_checkpoints(
        run_dir,
        output_path=tmp_path / "report.json",
        original_cwd=original_cwd,
        device="cpu",
        model_factory=model_factory,
        dataset_factory=dataset_factory,
        validator=validator,
    )

    assert report["status"] == "passed"


def test_evaluation_cli_passes_all_arguments_and_prints_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    expected = {"format_version": 1, "status": "passed"}
    captured: dict[str, Any] = {}

    def fake_evaluate(run_dir: str, **kwargs: Any) -> dict[str, Any]:
        captured["run_dir"] = run_dir
        captured.update(kwargs)
        return expected

    monkeypatch.setattr("scripts.evaluate_training_checkpoints.evaluate_training_checkpoints", fake_evaluate)
    exit_code = evaluation_cli_main(
        [
            "--run-dir",
            str(tmp_path / "run"),
            "--output",
            str(tmp_path / "result.json"),
            "--original-cwd",
            str(tmp_path / "repo"),
            "--device",
            "cpu",
            "--tolerance",
            "0.125",
        ]
    )

    assert exit_code == 0
    assert captured == {
        "device": "cpu",
        "original_cwd": str(tmp_path / "repo"),
        "output_path": str(tmp_path / "result.json"),
        "run_dir": str(tmp_path / "run"),
        "tolerance": 0.125,
    }
    assert json.loads(capsys.readouterr().out) == expected


def test_evaluation_cli_passes_explicit_depth_thresholds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_evaluate(run_dir: str, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"format_version": 1, "status": "passed"}

    monkeypatch.setattr("scripts.evaluate_training_checkpoints.evaluate_training_checkpoints", fake_evaluate)
    assert (
        evaluation_cli_main(
            [
                "--run-dir",
                str(tmp_path / "run"),
                "--output",
                str(tmp_path / "result.json"),
                "--original-cwd",
                str(tmp_path / "repo"),
                "--depth-threshold-m",
                "0.6",
                "--depth-threshold-m",
                "1.2",
            ]
        )
        == 0
    )
    assert captured["depth_thresholds_m"] == (0.6, 1.2)
