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
    _validate_completed_summary,
    _validation_loss_options,
    evaluate_rgbd_conditioning,
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


def test_completed_summary_accepts_exact_global_step_limit() -> None:
    config = _resolved_config()
    config["trainer"]["epochs"] = 6
    config["trainer"]["max_train_steps"] = 1
    base = {"filename": "base.pt", "sha256": "a" * 64, "size_bytes": 1}
    summary = {
        "base_checkpoint": base,
        "best": [],
        "early_stopping": {
            "bad_epochs": 0,
            "best": None,
            "enabled": False,
            "min_delta": 0.0,
            "mode": "min",
            "monitor": "val/objective",
            "patience": 2,
            "stopped": False,
        },
        "epochs_completed": 2,
        "global_step": 1,
        "group_fingerprint": GROUP_FINGERPRINT,
        "status": "complete",
        "stopped_early": False,
        "train": {"objective": 2.0},
        "validation": {"objective": 1.5},
    }

    completed_epochs, global_step, _, _ = _validate_completed_summary(summary, config, base)

    assert (completed_epochs, global_step) == (2, 1)


def test_completed_summary_rejects_incomplete_run_below_global_step_limit() -> None:
    config = _resolved_config()
    config["trainer"]["epochs"] = 6
    config["trainer"]["max_train_steps"] = 2
    base = {"filename": "base.pt", "sha256": "a" * 64, "size_bytes": 1}
    summary = {
        "base_checkpoint": base,
        "best": [],
        "early_stopping": {
            "bad_epochs": 0,
            "best": None,
            "enabled": False,
            "min_delta": 0.0,
            "mode": "min",
            "monitor": "val/objective",
            "patience": 2,
            "stopped": False,
        },
        "epochs_completed": 2,
        "global_step": 1,
        "group_fingerprint": GROUP_FINGERPRINT,
        "status": "complete",
        "stopped_early": False,
        "train": {"objective": 2.0},
        "validation": {"objective": 1.5},
    }

    with pytest.raises(EvaluationError, match="expected configured epochs"):
        _validate_completed_summary(summary, config, base)


def test_evaluator_attaches_enabled_dynamic_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    prepared = PreparedTrainingModel(model=_TinyHeadModel(), trainable_parameter_names=("head",))
    dynamic_config = {"enabled": True, "contract_version": 1}
    captured: list[tuple[PreparedTrainingModel, dict[str, Any], torch.device]] = []

    def fake_attach(
        value: PreparedTrainingModel,
        config: dict[str, Any],
        *,
        device: torch.device,
    ) -> PreparedTrainingModel:
        captured.append((value, config, device))
        return value

    monkeypatch.setattr(evaluation_module, "attach_dynamic_geometry_model", fake_attach, raising=False)

    actual, _, pixel_enabled = evaluation_module._attach_configured_training_wrappers(
        prepared,
        {"pixel_depth": {"enabled": False}, "dynamic_geometry": dynamic_config},
        device=torch.device("cpu"),
    )

    assert actual is prepared
    assert pixel_enabled is False
    assert captured == [(prepared, dynamic_config, torch.device("cpu"))]


def test_evaluator_attaches_enabled_depth_input_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    prepared = PreparedTrainingModel(model=_TinyHeadModel(), trainable_parameter_names=("head",))
    depth_config = {"enabled": True, "patch_size": 16, "embed_dim": 1024}
    captured: list[tuple[PreparedTrainingModel, dict[str, Any], torch.device]] = []

    def fake_attach(
        value: PreparedTrainingModel,
        config: dict[str, Any],
        *,
        device: torch.device,
    ) -> PreparedTrainingModel:
        captured.append((value, config, device))
        return value

    monkeypatch.setattr(evaluation_module, "attach_depth_input_model", fake_attach)
    actual, _, pixel_enabled = evaluation_module._attach_configured_training_wrappers(
        prepared,
        {
            "depth_input": depth_config,
            "pixel_depth": {"enabled": False},
            "dynamic_geometry": {"enabled": False},
        },
        device=torch.device("cpu"),
    )

    assert actual is prepared
    assert pixel_enabled is False
    assert captured == [(prepared, depth_config, torch.device("cpu"))]


def test_validation_dataset_receives_dynamic_flow_teacher_manifest(tmp_path: Path) -> None:
    config = _resolved_config()
    config["dynamic_geometry"] = {
        "enabled": True,
        "pseudo_labels": {"teacher_artifact_manifest": "flow_teacher/manifest.json"},
    }
    (tmp_path / "private-staging").mkdir()
    captured: list[dict[str, Any]] = []

    def factory(_root: Path, **options: Any) -> _TinyDataset:
        captured.append(options)
        return _TinyDataset()

    evaluation_module._validation_dataset(config, tmp_path / "original", factory)

    assert captured[0]["flow_teacher_manifest"] == "flow_teacher/manifest.json"


def test_pairwise_validation_options_and_monitor_names_are_stable() -> None:
    config = {
        "loss": {
            "validation": {
                "camera_weight": 5.0,
                "depth_weight": 1.0,
                "translation_weight": 1.0,
                "rotation_weight": 1.0,
                "fov_weight": 0.5,
                "relative_pose_weight": 0.1,
                "relative_rotation_weight": 1.0,
                "relative_translation_direction_weight": 1.0,
                "relative_translation_magnitude_weight": 1.0,
            }
        }
    }

    options = _validation_loss_options(config)

    assert options is not None and options["relative_pose_weight"] == pytest.approx(0.1)
    for metric in (
        "pairwise_pose",
        "pairwise_rotation_degrees",
        "pairwise_translation_direction_degrees",
        "pairwise_translation_magnitude",
        "rpa_5",
        "rpa_15",
        "rpa_30",
    ):
        assert evaluation_module._MONITOR_TO_METRIC[f"val/{metric}"] == metric


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
        "filter_short_sequences": True,
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


def test_evaluator_recomputes_configured_dynamic_classification_monitor(tmp_path: Path) -> None:
    run_dir, original_cwd, config = _make_run(tmp_path)
    config["checkpoint"]["monitor"] = "val/dynamic_classification"
    _write_json(run_dir / "resolved_config.json", config)
    leaderboard_path = run_dir / "checkpoints" / "leaderboard.json"
    leaderboard = json.loads(leaderboard_path.read_text())
    leaderboard["monitor"] = "val/dynamic_classification"
    _write_json(leaderboard_path, leaderboard)
    for checkpoint in (run_dir / "checkpoints").glob("best_epoch_*.pt"):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        payload["config"] = config
        payload["monitor"] = "val/dynamic_classification"
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
            "dynamic_classification": value,
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
    assert report["monitor"] == "val/dynamic_classification"
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


def test_paired_protocol_preserves_legacy_checkpoint_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    expected = {"format_version": 2, "protocol": "rgbd_paired_v1", "status": "passed"}
    captured: dict[str, Any] = {}

    def fake_paired(base_run_dir: str, candidate_run_dir: str, **kwargs: Any) -> dict[str, Any]:
        captured.update({"base_run_dir": base_run_dir, "candidate_run_dir": candidate_run_dir, **kwargs})
        return expected

    monkeypatch.setattr("scripts.evaluate_training_checkpoints.evaluate_rgbd_conditioning", fake_paired)
    assert (
        evaluation_cli_main(
            [
                "--protocol",
                "rgbd_paired_v1",
                "--paired-baseline-run",
                str(tmp_path / "base"),
                "--run-dir",
                str(tmp_path / "candidate"),
                "--output",
                str(tmp_path / "comparison.json"),
                "--original-cwd",
                str(tmp_path / "repo"),
                "--device",
                "cpu",
                "--eval-batch-size",
                "2",
                "--checkpoint-limit",
                "3",
            ]
        )
        == 0
    )
    assert captured == {
        "base_run_dir": str(tmp_path / "base"),
        "candidate_run_dir": str(tmp_path / "candidate"),
        "checkpoint_limit": 3,
        "device": "cpu",
        "evaluation_batch_size": 2,
        "original_cwd": str(tmp_path / "repo"),
        "output_path": str(tmp_path / "comparison.json"),
        "tolerance": 1e-4,
    }
    assert json.loads(capsys.readouterr().out) == expected

    with pytest.raises(SystemExit):
        evaluation_cli_main(
            [
                "--run-dir",
                str(tmp_path / "candidate"),
                "--output",
                str(tmp_path / "legacy.json"),
                "--original-cwd",
                str(tmp_path / "repo"),
                "--paired-baseline-run",
                str(tmp_path / "base"),
            ]
        )

    assert callable(evaluate_rgbd_conditioning)


class _TinyPairedModel(torch.nn.Module):
    def __init__(self, depth_value: float) -> None:
        super().__init__()
        self.depth_value = depth_value

    def forward(
        self,
        images: torch.Tensor,
        *,
        mapped_depth: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        availability: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size, frames, _, height, width = images.shape
        pose = torch.zeros(batch_size, frames, 9, device=images.device)
        pose[..., 6] = 1
        return {
            "depth": torch.full(
                (batch_size, frames, height, width, 1),
                self.depth_value,
                device=images.device,
            ),
            "pose_enc": pose,
        }


def test_rgbd_paired_evaluator_reuses_strict_evaluator_and_writes_new_report(tmp_path: Path) -> None:
    base_run = tmp_path / "base"
    candidate_run = tmp_path / "candidate"
    base_run.mkdir()
    candidate_run.mkdir()
    common = {
        "data": {"batch_size": 8, "root": "/data", "val_split": "val"},
        "model": {
            "image_height": 480,
            "image_width": 640,
            "initial_head_checkpoint": None,
            "precision": "bf16",
        },
        "seed": 42,
        "trainer": {"sequence_frames": 4},
    }
    _write_json(base_run / "resolved_config.json", {**common, "depth_input": None})
    candidate_config = json.loads(json.dumps(common))
    candidate_config["data"]["batch_size"] = 2
    candidate_config["model"]["initial_head_checkpoint"] = "base/checkpoints/best_epoch_000009_test.pt"
    candidate_config["depth_input"] = {"enabled": True, "patch_size": 16}
    _write_json(candidate_run / "resolved_config.json", candidate_config)
    calls: list[dict[str, Any]] = []
    candidate_identity = {"sequence_id": 7}

    def fake_strict_evaluator(run_dir: Path, **kwargs: Any) -> dict[str, Any]:
        is_candidate = Path(run_dir) == candidate_run
        calls.append({"candidate": is_candidate, **kwargs})
        batch_size = 1
        intrinsics = torch.eye(3).reshape(1, 1, 3, 3).expand(batch_size, 4, 3, 3).clone()
        intrinsics[..., 0, 0] = 10
        intrinsics[..., 1, 1] = 10
        intrinsics[..., 0, 2] = 16
        intrinsics[..., 1, 2] = 16
        batch = {
            "depth_masks": torch.ones(batch_size, 4, 32, 32, dtype=torch.bool),
            "depths": torch.ones(batch_size, 4, 32, 32),
            "extrinsics": torch.eye(4)[:3].reshape(1, 1, 3, 4).expand(batch_size, 4, 3, 4).clone(),
            "frame_ids": torch.tensor([[0, 2, 3, 4]]),
            "images": torch.zeros(batch_size, 4, 3, 32, 32),
            "intrinsics": intrinsics,
            "normalization_scale_m": torch.ones(batch_size),
            "scene_id": ["scene"],
            "sequence_id": torch.tensor([candidate_identity["sequence_id"] if is_candidate else 7]),
        }
        count = 2 if is_candidate else 1
        checkpoints = []
        for index in range(count):
            kwargs["validator"](
                model=_TinyPairedModel(0.5 if is_candidate else 1.0),
                batches=[batch],
                device=torch.device("cpu"),
                precision="bf16",
                max_batches=None,
            )
            epoch = 2 + index
            checkpoints.append(
                {
                    "epoch": epoch if is_candidate else 9,
                    "filename": f"best_epoch_{epoch:06d}_test.pt" if is_candidate else "best_epoch_000009_test.pt",
                    "global_step": 10 + index,
                    "sha256": chr(ord("b") + index) * 64 if is_candidate else "a" * 64,
                    "stored_metric": 1.0,
                }
            )
        return {
            "base_checkpoint": {"filename": "base.pt", "sha256": "f" * 64, "size_bytes": 1},
            "checkpoints": checkpoints,
            "initial_head_checkpoint": (
                {"filename": "best_epoch_000009_test.pt", "sha256": "a" * 64} if is_candidate else None
            ),
        }

    output = tmp_path / "comparison.json"
    report = evaluate_rgbd_conditioning(
        base_run,
        candidate_run,
        output_path=output,
        original_cwd=tmp_path,
        device="cpu",
        checkpoint_limit=2,
        evaluation_batch_size=1,
        checkpoint_evaluator=fake_strict_evaluator,
    )

    assert report["status"] == "passed"
    assert report["protocol"] == "rgbd_paired_v1"
    assert report["validation"]["availability_case_count"] == 16
    assert report["validation"]["sample_count"] == 1
    assert len(report["candidates"]) == 2
    assert report["selection"]["selected_checkpoint"]["epoch"] == 2
    assert report["candidates"][0]["comparisons"]["V2"]["normalized_mae"]["baseline"]["count"] > 0
    assert report["candidates"][0]["comparisons"]["V2"]["normalized_mae"]["candidate"]["count"] > 0
    assert report["candidates"][0]["holdout"]["status"] == "measured"
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert [call["candidate"] for call in calls] == [False, True]
    assert all(call["validate_stored_monitor"] is False for call in calls)
    assert all(call["evaluation_batch_size"] == 1 for call in calls)
    with pytest.raises(EvaluationError, match="must not already exist"):
        evaluate_rgbd_conditioning(
            base_run,
            candidate_run,
            output_path=output,
            original_cwd=tmp_path,
            device="cpu",
            checkpoint_evaluator=fake_strict_evaluator,
        )
    candidate_identity["sequence_id"] = 8
    mismatched_output = tmp_path / "mismatched.json"
    with pytest.raises(EvaluationError, match="identities or ordering"):
        evaluate_rgbd_conditioning(
            base_run,
            candidate_run,
            output_path=mismatched_output,
            original_cwd=tmp_path,
            device="cpu",
            checkpoint_limit=2,
            evaluation_batch_size=1,
            checkpoint_evaluator=fake_strict_evaluator,
        )
    assert not mismatched_output.exists()
