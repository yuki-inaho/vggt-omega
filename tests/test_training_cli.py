from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path

import vggt_omega.training.runner as training_runner

REPO_ROOT = Path(__file__).parents[1]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_colmap_rgbd.py"


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT), *arguments],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_training_cli_help_composes_without_loading_data_or_model() -> None:
    result = _run("--help")

    assert result.returncode == 0, result.stderr
    assert "optimizer" in result.stdout
    assert "trainer" in result.stdout


def test_training_cli_can_print_resolved_job_config() -> None:
    result = _run("--cfg", "job", "optimizer=adamw", "trainer=finetune", "checkpoint.k=2")

    assert result.returncode == 0, result.stderr
    assert "name: adamw" in result.stdout
    assert "name: finetune" in result.stdout
    assert "k: 2" in result.stdout
    assert "pretrained checkpoint not found" not in result.stderr


def test_training_cli_executes_hydra_multirun_without_allocating_model(monkeypatch, tmp_path: Path) -> None:
    observed_k: list[int] = []

    def fake_run_training(cfg, *, output_dir: str, original_cwd: str) -> dict[str, object]:
        observed_k.append(int(cfg.checkpoint.k))
        return {"status": "complete", "output_dir_name": Path(output_dir).name, "cwd": original_cwd}

    monkeypatch.setattr(training_runner, "run_training", fake_run_training)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(TRAIN_SCRIPT),
            "--multirun",
            f"hydra.sweep.dir={tmp_path.as_posix()}",
            "hydra.sweep.subdir=job_${hydra.job.num}",
            "checkpoint.k=1,2",
        ],
    )

    runpy.run_path(str(TRAIN_SCRIPT), run_name="__main__")

    assert observed_k == [1, 2]
