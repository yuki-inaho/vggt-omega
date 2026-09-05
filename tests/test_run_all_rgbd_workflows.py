from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts.run_all_rgbd_workflows import aggregate_summaries, run_all


def test_aggregate_summaries_requires_every_dataset(tmp_path: Path) -> None:
    names = ["segment_a", "segment_b"]
    first = tmp_path / names[0]
    first.mkdir()
    (first / "workflow_summary.json").write_text(json.dumps({"workflow_complete": True, "frame_count": 10}))

    partial = aggregate_summaries(tmp_path, names)

    assert partial["all_workflows_complete"] is False
    assert partial["datasets"]["segment_b"]["reason"] == "summary_missing"
    second = tmp_path / names[1]
    second.mkdir()
    (second / "workflow_summary.json").write_text(json.dumps({"workflow_complete": True, "frame_count": 20}))
    complete = aggregate_summaries(tmp_path, names)
    assert complete["all_workflows_complete"] is True
    assert json.loads((tmp_path / "all_datasets_summary.json").read_text()) == complete


def test_run_all_records_failing_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    (source / "segment_a").mkdir(parents=True)

    def fail(*args: object, **kwargs: object) -> None:
        raise subprocess.CalledProcessError(9, ["runner"])

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        run_all(source, output, ["segment_a"])

    state = json.loads((output / "run_all_state.json").read_text())
    assert state == {
        "status": "failed",
        "current_dataset": "segment_a",
        "completed_datasets": [],
        "returncode": 9,
    }
