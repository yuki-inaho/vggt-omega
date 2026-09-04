from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.report_overlap_distribution import OverlapReportError, summarize_overlap_distribution


def _fixture_arrays() -> tuple[np.ndarray, ...]:
    sequences = np.asarray([[0, 1, 2], [3, 4, -1], [5, 6, -1]], dtype=np.int64)
    lengths = np.asarray([3, 2, 2], dtype=np.int64)
    split_ids = np.asarray([0, 1, 2], dtype=np.int8)
    all_depth = np.zeros((3, 3, 3), dtype=np.float32)
    near_depth = np.zeros_like(all_depth)
    values = ((0, 0, 1, 0.9), (0, 0, 2, 0.6), (0, 1, 2, 0.3), (1, 0, 1, 0.5), (2, 0, 1, 0.1))
    for sequence_id, first, second, value in values:
        all_depth[sequence_id, first, second] = all_depth[sequence_id, second, first] = value
        near_depth[sequence_id, first, second] = near_depth[sequence_id, second, first] = value + 0.05
    extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None, :3], 7, axis=0)
    extrinsics[:, 0, 3] = -np.arange(7, dtype=np.float32) * 0.01
    return sequences, lengths, split_ids, all_depth, near_depth, extrinsics


def test_overlap_report_has_finite_split_distributions_buckets_and_no_paths() -> None:
    report = summarize_overlap_distribution(*_fixture_arrays())

    assert report["schema_version"] == 1
    assert report["pair_count"] == 5
    assert report["pair_leakage_count"] == 0
    assert report["splits"]["train"]["pair_count"] == 3
    assert report["splits"]["val"]["pair_count"] == 1
    assert report["splits"]["smoke"]["pair_count"] == 1
    for split in report["splits"].values():
        assert sum(split["near_depth_buckets"].values()) == split["pair_count"]
        assert len(split["near_depth"]["quantiles"]) == 7
    serialized = json.dumps(report, allow_nan=False, sort_keys=True)
    assert "/home/" not in serialized
    assert "scene_" not in serialized


def test_overlap_report_rejects_nonfinite_scores() -> None:
    arrays = list(_fixture_arrays())
    arrays[3][0, 0, 1] = np.nan

    with pytest.raises(OverlapReportError, match="finite"):
        summarize_overlap_distribution(*arrays)
