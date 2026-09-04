from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.run_full_rgbd_workflow import (
    _chunk_count,
    _complete_stage,
    _existing_dynamic_summary,
    _existing_manifest_frame_count,
    _load_state,
    _model_complete,
    _opend4rt_chunk_count,
    _rail_model_metrics,
    _run_logged,
    _stage_complete,
)


def test_stage_state_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "workflow"
    state = _load_state(output)

    _complete_stage(output, state, "prepare", {"frame_count": 8})
    loaded = _load_state(output)

    assert _stage_complete(loaded, "prepare")
    assert not _stage_complete(loaded, "vggt4d_chunks")
    assert loaded["stages"]["prepare"]["evidence"] == {"frame_count": 8}


def test_run_logged_appends_retry_evidence(tmp_path: Path) -> None:
    log = tmp_path / "stage.log"
    _run_logged([sys.executable, "-c", "print('first')"], tmp_path, log)
    _run_logged([sys.executable, "-c", "print('second')"], tmp_path, log)

    content = log.read_text()
    assert "first" in content and "second" in content
    assert content.count("# command:") == 2


def test_chunk_count_requires_metadata_and_all_masks(tmp_path: Path) -> None:
    run = tmp_path / "chunks"
    run.mkdir()
    manifest = {
        "chunk_count": 2,
        "chunks": [
            {"chunk_index": 0, "global_indices": [0, 1]},
            {"chunk_index": 1, "global_indices": [1, 2]},
        ],
    }
    (run / "chunks_manifest.json").write_text(json.dumps(manifest))
    first = run / "chunk_000000"
    first.mkdir()
    (first / "metadata.json").write_text(json.dumps({"num_frames": 2}))
    for name in ("pred_traj.txt", "pred_intrinsics.txt", "frames.txt"):
        (first / name).write_text("x")
    for pattern in ("frame", "conf", "dynamic_mask"):
        for index in range(2):
            suffix = ".npy" if pattern in {"frame", "conf"} else ".png"
            (first / f"{pattern}_{index:04d}{suffix}").write_text("x")
    for index in range(2):
        (first / f"frame_{index:04d}.png").write_text("x")
    second = run / "chunk_000001"
    second.mkdir()
    (second / "metadata.json").write_text("{}")
    (second / "dynamic_mask_0000.png").write_text("x")

    assert _chunk_count(run) == (1, 2)


def test_opend4rt_chunk_count_requires_scene_and_metadata(tmp_path: Path) -> None:
    run = tmp_path / "chunks"
    run.mkdir()
    manifest = {
        "chunk_count": 2,
        "chunks": [{"chunk_index": 0}, {"chunk_index": 1}],
    }
    (run / "chunks_manifest.json").write_text(json.dumps(manifest))
    first = run / "chunk_000000"
    first.mkdir()
    (first / "metadata.json").write_text("{}")
    (first / "dense_scene.npz").write_bytes(b"x")
    second = run / "chunk_000001"
    second.mkdir()
    (second / "metadata.json").write_text("{}")

    assert _opend4rt_chunk_count(run) == (1, 2)


def test_model_complete_requires_sparse_binary_contract(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (model / name).write_bytes(b"x")

    assert _model_complete(model)
    (model / "images.bin").unlink()
    assert not _model_complete(model)


def test_existing_manifest_must_match_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    (output / "manifest.json").write_text(json.dumps({"source_dataset": str(source), "frames": [{"frame_index": 0}]}))

    assert _existing_manifest_frame_count(source, output) == 1
    assert _existing_manifest_frame_count(tmp_path / "other", output) is None


def test_existing_dynamic_summary_requires_nonempty_cloud(tmp_path: Path) -> None:
    dynamic = tmp_path / "dynamic_map"
    dynamic.mkdir()
    (dynamic / "dynamic_rgbd_map.json").write_text(json.dumps({"point_count_after_voxel": 4}))
    cloud = dynamic / "dynamic_rgbd_map.ply"
    cloud.write_bytes(b"ply")

    assert _existing_dynamic_summary(tmp_path) == {"point_count_after_voxel": 4}
    cloud.write_bytes(b"")
    assert _existing_dynamic_summary(tmp_path) is None


def test_rail_model_metrics_parses_colmap_two_line_records(tmp_path: Path) -> None:
    trajectory = tmp_path / "trajectory.json"
    trajectory.write_text(json.dumps({"rail": {"rail_axis": [1, 0, 0], "rail_centroid_m": [0, 0, 0]}}))
    images = tmp_path / "images.txt"
    images.write_text("# header\n1 1 0 0 0 0 -0.01 0 1 a.png\n\n2 1 0 0 0 -1 -0.02 0 1 b.png\n10 10 -1\n")

    metrics = _rail_model_metrics(images, trajectory)

    assert metrics["registered_images"] == 2
    assert metrics["rail_orthogonal_rms_m"] == pytest.approx(np.sqrt((0.01**2 + 0.02**2) / 2))
