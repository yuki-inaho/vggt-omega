"""Opt-in real-data smoke test for VGGT RGB-D chunk alignment."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.rgbd_smoke
def test_two_overlapping_chunks_create_aligned_and_fused_clouds(tmp_path: Path) -> None:
    session_value = os.environ.get("VGGT_RGBD_SMOKE_SESSION")
    checkpoint_value = os.environ.get("VGGT_RGBD_SMOKE_CHECKPOINT")
    if not session_value or not checkpoint_value:
        pytest.skip("Set VGGT_RGBD_SMOKE_SESSION and VGGT_RGBD_SMOKE_CHECKPOINT to run the real-data smoke test")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    session = Path(session_value).resolve()
    checkpoint = Path(checkpoint_value).resolve()
    if not session.is_dir() or not checkpoint.is_file():
        pytest.skip("The configured RGB-D session or checkpoint does not exist")

    output = tmp_path / "alignment"
    environment = os.environ.copy()
    environment.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "run_vggt_rgbd_chunk_alignment.py"),
            "--session-dir",
            str(session),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output),
            "--chunk-size",
            "6",
            "--stride",
            "3",
            "--max-chunks",
            "2",
            "--width",
            "640",
            "--height",
            "480",
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        timeout=300,
    )

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    aligned_clouds = list((output / "aligned_masked_clouds").glob("*.ply"))
    assert summary["mode"] == "vggt_initial_pose_shared_frame_chain"
    assert summary["chunk_count"] == 2
    assert summary["unique_frame_count"] == 9
    assert len(aligned_clouds) == 9
    assert summary["input_point_count"] > summary["fused_point_count"] > 0
    assert summary["edge_residual_summary"]["translation_residual_m_max_worst"] >= 0
    assert (output / "fused_masked_vggt_initial_pose.ply").stat().st_size > 0
    assert (output / "vggt_initial_pose_alignment.npz").stat().st_size > 0
