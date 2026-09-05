"""Run named RGB-D dataset splits serially and aggregate their summaries."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def aggregate_summaries(output_root: Path, names: Sequence[str]) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for name in names:
        path = output_root / name / "workflow_summary.json"
        if not path.is_file():
            datasets[name] = {"workflow_complete": False, "reason": "summary_missing"}
        else:
            datasets[name] = json.loads(path.read_text())
    result = {
        "schema_version": 1,
        "dataset_order": list(names),
        "all_workflows_complete": all(item.get("workflow_complete", False) for item in datasets.values()),
        "datasets": datasets,
    }
    _write_json(output_root / "all_datasets_summary.json", result)
    return result


def run_all(source_root: Path, output_root: Path, names: Sequence[str]) -> dict[str, Any]:
    runner = Path(__file__).with_name("run_full_rgbd_workflow.py")
    completed: list[str] = []
    for name in names:
        source = source_root / name
        output = output_root / name
        if not source.is_dir():
            raise FileNotFoundError(f"dataset split does not exist: {source}")
        command = [
            sys.executable,
            str(runner),
            "--source",
            str(source),
            "--output",
            str(output),
            "--chunk-size",
            "16",
            "--chunk-overlap",
            "4",
            "--sequential-overlap",
            "10",
        ]
        state_path = output_root / "run_all_state.json"
        _write_json(
            state_path,
            {"status": "running", "current_dataset": name, "completed_datasets": completed},
        )
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as error:
            aggregate_summaries(output_root, names)
            _write_json(
                state_path,
                {
                    "status": "failed",
                    "current_dataset": name,
                    "completed_datasets": completed,
                    "returncode": error.returncode,
                },
            )
            raise
        completed.append(name)
        aggregate_summaries(output_root, names)
    result = aggregate_summaries(output_root, names)
    _write_json(
        output_root / "run_all_state.json",
        {"status": "complete", "current_dataset": None, "completed_datasets": completed},
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    args = parser.parse_args(argv)
    summary = run_all(args.source_root, args.output_root, args.datasets)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
