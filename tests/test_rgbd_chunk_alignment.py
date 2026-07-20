"""CPU tests for shared-frame VGGT chunk alignment geometry."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_vggt_rgbd_chunk_alignment import (
    ChunkResult,
    chunk_start_indices,
    estimate_adjacent_chunk_transform,
    global_frame_poses,
    summarize_edge_residuals,
    transform_points,
)


def _pose(translation: tuple[float, float, float]) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = translation
    return result


def test_shared_camera_poses_recover_chunk_transform_and_global_poses() -> None:
    previous_poses = np.stack([_pose((0.0, 0.0, 0.0)), _pose((0.1, 0.0, 0.0)), _pose((0.2, 0.0, 0.0))])
    target_from_source = _pose((0.5, -0.2, 0.1))
    current_poses = np.stack([target_from_source @ pose for pose in previous_poses])
    previous = ChunkResult(0, ("f0", "f1", "f2"), 1.0, 1_000, previous_poses)
    current = ChunkResult(1, ("f0", "f1", "f2"), 1.0, 1_000, current_poses)

    estimated, metrics = estimate_adjacent_chunk_transform(previous, current)
    np.testing.assert_allclose(estimated, target_from_source, atol=1e-12)
    assert metrics["shared_frames"] == ["f0", "f1", "f2"]
    assert cast(float, metrics["translation_residual_m_max"]) < 1e-12

    stems, global_poses, counts = global_frame_poses(
        [previous, current],
        [np.eye(4), np.linalg.inv(estimated)],
    )
    assert stems == ["f0", "f1", "f2"]
    assert counts == {"f0": 2, "f1": 2, "f2": 2}
    np.testing.assert_allclose(global_poses, previous_poses, atol=1e-12)


def test_transform_points_uses_camera_to_global_pose() -> None:
    points = np.asarray([[0.0, 0.0, 1.0], [1.0, 2.0, 3.0]])
    transformed = transform_points(points, _pose((0.5, -0.25, 1.0)))
    np.testing.assert_allclose(
        transformed,
        np.asarray([[0.5, -0.25, 2.0], [1.5, 1.75, 4.0]]),
    )


def test_chunk_starts_cover_the_session_tail() -> None:
    starts = chunk_start_indices(frame_count=62, chunk_size=6, stride=3)
    assert starts[:3] == [0, 3, 6]
    assert starts[-1] == 56
    assert starts[-1] + 6 == 62


def test_edge_residual_summary_reports_percentiles_and_worst_value() -> None:
    edges: list[dict[str, object]] = [
        {
            "translation_residual_m_median": value,
            "translation_residual_m_max": value * 2,
            "rotation_residual_deg_median": value * 10,
            "rotation_residual_deg_max": value * 20,
        }
        for value in (0.001, 0.002, 0.003)
    ]
    summary = summarize_edge_residuals(edges)
    assert summary["translation_residual_m_median_p50"] == 0.002
    assert summary["translation_residual_m_median_worst"] == 0.003
