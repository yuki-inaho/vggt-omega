from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch.utils.tensorboard import SummaryWriter

from scripts.audit_training_artifacts import main as audit_cli_main
from vggt_omega.training.audit import audit_training_artifacts

BASE_CHECKPOINT = {
    "filename": "base_model.pt",
    "sha256": "b" * 64,
    "size_bytes": 123,
}
GROUP_FINGERPRINT = "a" * 64
BEST_FILENAME = "best_epoch_000000_deadbeef0000.pt"
EARLY_STOPPING_DISABLED = {
    "bad_epochs": 0,
    "best": None,
    "enabled": False,
    "min_delta": 0.0,
    "mode": "min",
    "monitor": "val/objective",
    "patience": 2,
    "stopped": False,
}


def _checkpoint_payload(*, kind: str, include_optimizer: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "config": {"profile": "test"},
        "epoch": 0,
        "format_version": 1,
        "global_step": 1,
        "group_fingerprint": GROUP_FINGERPRINT,
        "kind": kind,
        "metadata": {"base_checkpoint": BASE_CHECKPOINT},
        "model_state": {
            "camera_head.weight": torch.ones(1),
            "dense_head.depth.weight": torch.ones(1),
        },
        "parameter_state": "x",
    }
    if kind == "best":
        payload.update({"metric": 1.0, "monitor": "val/objective"})
    if include_optimizer:
        payload["optimizer_state"] = {"param_groups": [], "state": {}}
        payload["checkpoint_role"] = "last"
        payload["training_state"] = {"early_stopping": EARLY_STOPPING_DISABLED}
    return payload


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_tensorboard(log_dir: Path, *, forbidden_image: bool = False) -> None:
    writer = SummaryWriter(log_dir=str(log_dir))
    scalars = {
        "train/objective": 2.0,
        "train/camera": 1.0,
        "train/depth": 1.0,
        "train/grad_norm": 0.5,
        "optimizer/group_0_lr": 1e-4,
        "optimizer/beta1": 0.4,
        "val/objective": 1.0,
        "val/camera": 0.4,
        "val/depth": 0.6,
    }
    for tag, value in scalars.items():
        writer.add_scalar(tag, value, 1)
    if forbidden_image:
        writer.add_image("forbidden/image", torch.zeros(3, 2, 2), 1)
    writer.close()


def _make_valid_run(tmp_path: Path, *, save_last: bool = True) -> Path:
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    leaderboard_entry = {
        "epoch": 0,
        "filename": BEST_FILENAME,
        "global_step": 1,
        "metric": 1.0,
    }
    _write_json(
        checkpoint_dir / "leaderboard.json",
        {
            "entries": [leaderboard_entry],
            "format_version": 1,
            "k": 2,
            "mode": "min",
            "monitor": "val/objective",
        },
    )
    torch.save(_checkpoint_payload(kind="best", include_optimizer=False), checkpoint_dir / BEST_FILENAME)
    if save_last:
        torch.save(_checkpoint_payload(kind="resume", include_optimizer=True), checkpoint_dir / "last.pt")
    _write_json(
        run_dir / "run_summary.json",
        {
            "base_checkpoint": BASE_CHECKPOINT,
            "best": [leaderboard_entry],
            "early_stopping": EARLY_STOPPING_DISABLED,
            "epochs_completed": 1,
            "global_step": 1,
            "group_fingerprint": GROUP_FINGERPRINT,
            "status": "complete",
            "stopped_early": False,
            "train": {"camera": 1.0, "depth": 1.0, "objective": 2.0},
            "validation": {"camera": 0.4, "depth": 0.6, "objective": 1.0},
        },
    )
    _write_json(
        run_dir / "resolved_config.json",
        {
            "checkpoint": {
                "directory": "checkpoints",
                "k": 2,
                "mode": "min",
                "monitor": "val/objective",
                "save_last": save_last,
            },
            "logging": {"directory": "tensorboard", "enabled": True},
            "trainer": {
                "device": "cpu",
                "early_stopping": {
                    "enabled": False,
                    "min_delta": 0.0,
                    "mode": "min",
                    "monitor": "val/objective",
                    "patience": 2,
                },
                "epochs": 1,
                "validate_every_epochs": 1,
            },
        },
    )
    (run_dir / "train.log").write_text("training complete\n", encoding="utf-8")
    _write_tensorboard(run_dir / "tensorboard")
    return run_dir


def _error_codes(report: dict[str, Any]) -> set[str]:
    return {error["code"] for error in report["errors"]}


def test_audit_accepts_complete_generic_run_and_writes_atomic_report(tmp_path: Path) -> None:
    run_dir = _make_valid_run(tmp_path)
    report_path = tmp_path / "reports" / "audit.json"

    report = audit_training_artifacts(run_dir, report_path=report_path, deny_tokens=("secret-session",))

    assert report["status"] == "passed"
    assert report["errors"] == []
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    assert report["checks"]["leaderboard"]["best_checkpoint_count"] == 1
    assert report["checks"]["checkpoints"]["has_last"] is True
    assert set(report["checks"]["tensorboard"]["scalar_tags"]) == {
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
    rendered = report_path.read_text(encoding="utf-8")
    assert str(run_dir) not in rendered
    assert "secret-session" not in rendered
    assert not list(report_path.parent.glob(".*.tmp"))


def test_audit_accepts_legacy_disabled_early_stopping_artifacts(tmp_path: Path) -> None:
    run_dir = _make_valid_run(tmp_path)
    config_path = run_dir / "resolved_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    del config["trainer"]["early_stopping"]
    _write_json(config_path, config)
    summary_path = run_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    del summary["early_stopping"]
    _write_json(summary_path, summary)
    last_path = run_dir / "checkpoints" / "last.pt"
    last = torch.load(last_path, map_location="cpu", weights_only=True)
    del last["training_state"]
    torch.save(last, last_path)

    report = audit_training_artifacts(run_dir, report_path=tmp_path / "audit.json")

    assert report["status"] == "passed"


def test_audit_allows_run_without_optional_last_checkpoint(tmp_path: Path) -> None:
    run_dir = _make_valid_run(tmp_path, save_last=False)

    report = audit_training_artifacts(run_dir, report_path=tmp_path / "audit.json")

    assert report["status"] == "passed"
    assert report["checks"]["checkpoints"]["has_last"] is False


def test_audit_requires_last_checkpoint_when_configured(tmp_path: Path) -> None:
    run_dir = _make_valid_run(tmp_path)
    (run_dir / "checkpoints" / "last.pt").unlink()

    report = audit_training_artifacts(run_dir, report_path=tmp_path / "audit.json")

    assert report["status"] == "failed"
    assert "configured_last_checkpoint_missing" in _error_codes(report)


def test_audit_rejects_run_that_did_not_finish_configured_epochs(tmp_path: Path) -> None:
    run_dir = _make_valid_run(tmp_path)
    config_path = run_dir / "resolved_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["trainer"]["epochs"] = 2
    _write_json(config_path, config)

    report = audit_training_artifacts(run_dir, report_path=tmp_path / "audit.json")

    assert report["status"] == "failed"
    assert "run_epoch_count_mismatch" in _error_codes(report)


def test_audit_accepts_valid_early_stopped_run(tmp_path: Path) -> None:
    run_dir = _make_valid_run(tmp_path)
    config_path = run_dir / "resolved_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["trainer"]["epochs"] = 5
    config["trainer"]["early_stopping"].update({"enabled": True, "patience": 1})
    _write_json(config_path, config)
    summary_path = run_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["epochs_completed"] = 2
    summary["global_step"] = 2
    summary["stopped_early"] = True
    summary["early_stopping"].update({"bad_epochs": 1, "best": 1.0, "enabled": True, "patience": 1, "stopped": True})
    _write_json(summary_path, summary)
    last_path = run_dir / "checkpoints" / "last.pt"
    last = torch.load(last_path, map_location="cpu", weights_only=True)
    last["epoch"] = 1
    last["global_step"] = 2
    last["training_state"] = {"early_stopping": summary["early_stopping"]}
    torch.save(last, last_path)
    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
    writer.add_scalar("val/objective", 1.1, 2)
    writer.close()

    report = audit_training_artifacts(run_dir, report_path=tmp_path / "audit.json")

    assert report["status"] == "passed"


def test_audit_rejects_false_early_stop_claim(tmp_path: Path) -> None:
    run_dir = _make_valid_run(tmp_path)
    summary_path = run_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["stopped_early"] = True
    summary["early_stopping"].update({"bad_epochs": 2, "best": 1.0, "stopped": True})
    _write_json(summary_path, summary)

    report = audit_training_artifacts(run_dir, report_path=tmp_path / "audit.json")

    assert report["status"] == "failed"
    assert "disabled_early_stopping_has_state" in _error_codes(report)


def test_audit_rejects_early_stopping_state_inconsistent_with_tensorboard(tmp_path: Path) -> None:
    run_dir = _make_valid_run(tmp_path)
    config_path = run_dir / "resolved_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["trainer"]["early_stopping"]["enabled"] = True
    _write_json(config_path, config)
    summary_path = run_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["early_stopping"].update({"best": 0.5, "enabled": True})
    _write_json(summary_path, summary)
    last_path = run_dir / "checkpoints" / "last.pt"
    last = torch.load(last_path, map_location="cpu", weights_only=True)
    last["training_state"] = {"early_stopping": summary["early_stopping"]}
    torch.save(last, last_path)

    report = audit_training_artifacts(run_dir, report_path=tmp_path / "audit.json")

    assert report["status"] == "failed"
    assert "early_stopping_history_mismatch" in _error_codes(report)


def test_audit_requires_one_validation_event_per_validation_epoch(tmp_path: Path) -> None:
    run_dir = _make_valid_run(tmp_path)
    config_path = run_dir / "resolved_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["trainer"]["epochs"] = 2
    _write_json(config_path, config)
    summary_path = run_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["epochs_completed"] = 2
    _write_json(summary_path, summary)

    report = audit_training_artifacts(run_dir, report_path=tmp_path / "audit.json")

    assert report["status"] == "failed"
    assert "tensorboard_validation_count_mismatch" in _error_codes(report)


def test_audit_cli_writes_report_and_returns_success(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_dir = _make_valid_run(tmp_path)
    report_path = tmp_path / "cli-audit.json"

    exit_code = audit_cli_main(
        [
            "--run-dir",
            str(run_dir),
            "--report",
            str(report_path),
            "--deny-token",
            "generic-secret",
        ]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["status"] == "passed"
    assert json.loads(report_path.read_text(encoding="utf-8"))["status"] == "passed"


def test_audit_reports_home_path_and_custom_token_without_leaking_values(tmp_path: Path) -> None:
    run_dir = _make_valid_run(tmp_path)
    private_token = "private-token-value"
    (run_dir / "train.log").write_text(
        f"source=/home/example/private/session token={private_token}\n",
        encoding="utf-8",
    )

    report = audit_training_artifacts(run_dir, report_path=tmp_path / "audit.json", deny_tokens=(private_token,))

    assert report["status"] == "failed"
    assert {"absolute_home_path", "private_token"} <= _error_codes(report)
    rendered = json.dumps(report)
    assert "/home/example" not in rendered
    assert private_token not in rendered


def test_audit_rejects_non_scalar_tensorboard_content(tmp_path: Path) -> None:
    run_dir = _make_valid_run(tmp_path)
    _write_tensorboard(run_dir / "tensorboard", forbidden_image=True)

    report = audit_training_artifacts(run_dir, report_path=tmp_path / "audit.json")

    assert report["status"] == "failed"
    assert "tensorboard_non_scalar_content" in _error_codes(report)


@pytest.mark.parametrize("profile", ["camera_motion_curriculum", "camera_translation_focus"])
def test_camera_profile_audit_requires_component_scalars(tmp_path: Path, profile: str) -> None:
    run_dir = _make_valid_run(tmp_path)
    config_path = run_dir / "resolved_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["loss"] = {"name": profile}
    _write_json(config_path, config)

    report = audit_training_artifacts(run_dir, report_path=tmp_path / "audit.json")

    assert report["status"] == "failed"
    assert "tensorboard_missing_scalar_tags" in _error_codes(report)


def test_audit_cross_checks_initial_head_provenance(tmp_path: Path) -> None:
    run_dir = _make_valid_run(tmp_path)
    initial = {
        "epoch": 49,
        "filename": "last.pt",
        "global_step": 36200,
        "kind": "resume",
        "parameter_state": "x",
        "sha256": "d" * 64,
    }
    config_path = run_dir / "resolved_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["model"] = {"initial_head_checkpoint": "previous/checkpoints/last.pt"}
    _write_json(config_path, config)
    summary_path = run_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["initial_head_checkpoint"] = initial
    _write_json(summary_path, summary)
    for checkpoint in (run_dir / "checkpoints").glob("*.pt"):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        payload["metadata"]["initial_head_checkpoint"] = initial
        torch.save(payload, checkpoint)

    passing = audit_training_artifacts(run_dir, report_path=tmp_path / "passing.json")
    assert passing["status"] == "passed"

    best_path = run_dir / "checkpoints" / BEST_FILENAME
    payload = torch.load(best_path, map_location="cpu", weights_only=True)
    payload["metadata"]["initial_head_checkpoint"]["sha256"] = "e" * 64
    torch.save(payload, best_path)
    failing = audit_training_artifacts(run_dir, report_path=tmp_path / "failing.json")

    assert failing["status"] == "failed"
    assert "checkpoint_initial_head_mismatch" in _error_codes(failing)


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("aggregator", "checkpoint_forbidden_model_state"),
        ("confidence", "checkpoint_forbidden_model_state"),
        ("best_optimizer", "best_checkpoint_has_optimizer"),
        ("last_no_optimizer", "last_checkpoint_missing_optimizer"),
        ("group_mismatch", "checkpoint_group_fingerprint_mismatch"),
        ("base_mismatch", "checkpoint_base_sha_mismatch"),
    ],
)
def test_audit_rejects_invalid_checkpoint_payloads(tmp_path: Path, mutation: str, expected_code: str) -> None:
    run_dir = _make_valid_run(tmp_path)
    checkpoint_dir = run_dir / "checkpoints"
    best_path = checkpoint_dir / BEST_FILENAME
    target = best_path
    payload = torch.load(best_path, map_location="cpu", weights_only=True)
    if mutation == "aggregator":
        payload["model_state"]["aggregator.block.weight"] = torch.ones(1)
    elif mutation == "confidence":
        payload["model_state"]["dense_head.proj_conf.weight"] = torch.ones(1)
    elif mutation == "best_optimizer":
        payload["optimizer_state"] = {"state": {}, "param_groups": []}
    elif mutation == "last_no_optimizer":
        target = checkpoint_dir / "last.pt"
        payload = torch.load(target, map_location="cpu", weights_only=True)
        del payload["optimizer_state"]
    elif mutation == "group_mismatch":
        payload["group_fingerprint"] = "c" * 64
    elif mutation == "base_mismatch":
        payload["metadata"]["base_checkpoint"]["sha256"] = "c" * 64
    torch.save(payload, target)

    report = audit_training_artifacts(run_dir, report_path=tmp_path / "audit.json")

    assert report["status"] == "failed"
    assert expected_code in _error_codes(report)


def test_audit_rejects_leaderboard_header_and_file_count_over_k(tmp_path: Path) -> None:
    run_dir = _make_valid_run(tmp_path)
    leaderboard_path = run_dir / "checkpoints" / "leaderboard.json"
    leaderboard = json.loads(leaderboard_path.read_text(encoding="utf-8"))
    leaderboard["format_version"] = 2
    leaderboard["k"] = 0
    _write_json(leaderboard_path, leaderboard)

    report = audit_training_artifacts(run_dir, report_path=tmp_path / "audit.json")

    assert report["status"] == "failed"
    assert {"leaderboard_header_invalid", "leaderboard_config_mismatch"} <= _error_codes(report)


def test_audit_rejects_unordered_leaderboard_untracked_checkpoint_and_tmp(tmp_path: Path) -> None:
    run_dir = _make_valid_run(tmp_path)
    checkpoint_dir = run_dir / "checkpoints"
    leaderboard_path = checkpoint_dir / "leaderboard.json"
    leaderboard = json.loads(leaderboard_path.read_text(encoding="utf-8"))
    second_filename = "best_epoch_000001_feedface0000.pt"
    second_entry = {"epoch": 1, "filename": second_filename, "global_step": 2, "metric": 0.5}
    leaderboard["entries"].append(second_entry)
    _write_json(leaderboard_path, leaderboard)
    second = _checkpoint_payload(kind="best", include_optimizer=False)
    second.update({"epoch": 1, "global_step": 2, "metric": 0.5})
    torch.save(second, checkpoint_dir / second_filename)
    torch.save(second, checkpoint_dir / "untracked.pt")
    (checkpoint_dir / ".interrupted.tmp").write_text("partial", encoding="utf-8")

    report = audit_training_artifacts(run_dir, report_path=tmp_path / "audit.json")

    assert report["status"] == "failed"
    assert {
        "leaderboard_order",
        "untracked_checkpoint",
        "temporary_artifact",
    } <= _error_codes(report)
