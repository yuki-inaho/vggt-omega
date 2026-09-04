from __future__ import annotations

import pytest
import torch

from vggt_omega.training.multiview_metrics import (
    directional_depth_consistency,
    sequence_multiview_consistency,
)


def _intrinsics(height: int, width: int, focal: float = 8.0) -> torch.Tensor:
    return torch.tensor(
        [[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )


def _pose(tx: float = 0.0) -> torch.Tensor:
    pose = torch.eye(4, dtype=torch.float32)[:3]
    pose[0, 3] = tx
    return pose


def test_directional_depth_consistency_identity_is_zero_with_full_coverage() -> None:
    depth = torch.ones(5, 7)
    result = directional_depth_consistency(
        depth,
        depth,
        _intrinsics(5, 7),
        _intrinsics(5, 7),
        _pose(),
        _pose(),
    )

    assert result["depth_error"] == pytest.approx(0)
    assert result["relative_error"] == pytest.approx(0)
    assert result["coverage"] == pytest.approx(1)
    assert result["visible_points"] == 35


def test_directional_depth_consistency_known_baseline_preserves_plane_depth() -> None:
    depth = torch.ones(5, 9)
    result = directional_depth_consistency(
        depth,
        depth,
        _intrinsics(5, 9, focal=4),
        _intrinsics(5, 9, focal=4),
        _pose(),
        _pose(tx=0.25),
    )

    assert result["depth_error"] == pytest.approx(0)
    assert 0 < result["coverage"] < 1


def test_directional_depth_consistency_rejects_occlusion_and_non_overlap() -> None:
    source = torch.ones(5, 7)
    occluder = torch.full_like(source, 0.5)
    intrinsics = _intrinsics(5, 7)

    occluded = directional_depth_consistency(source, occluder, intrinsics, intrinsics, _pose(), _pose())
    non_overlap = directional_depth_consistency(source, source, intrinsics, intrinsics, _pose(), _pose(tx=100))

    assert occluded["coverage"] == pytest.approx(0)
    assert occluded["depth_error"] == pytest.approx(0)
    assert non_overlap["coverage"] == pytest.approx(0)
    assert non_overlap["visible_points"] == 0


def test_sequence_multiview_consistency_reports_symmetric_coverage_and_pair_counts() -> None:
    depth = torch.ones(1, 3, 5, 7)
    intrinsics = _intrinsics(5, 7).reshape(1, 1, 3, 3).repeat(1, 3, 1, 1)
    poses = torch.stack((_pose(), _pose(tx=0.125), _pose(tx=100))).reshape(1, 3, 3, 4)
    valid = torch.ones_like(depth, dtype=torch.bool)

    result = sequence_multiview_consistency(depth, intrinsics, poses, valid_mask=valid, max_depth_m=1.2)

    assert result["pair_count"] == 3
    assert result["direction_count"] == 6
    assert result["visible_direction_count"] == 2
    assert result["symmetric_depth_error"] == pytest.approx(0)
    assert 0 < result["symmetric_coverage"] < 1


def test_multiview_consistency_rejects_invalid_pose() -> None:
    depth = torch.ones(3, 4)
    intrinsics = _intrinsics(3, 4)
    invalid = _pose()
    invalid[0, 0] = torch.nan

    with pytest.raises(ValueError, match="extrinsics"):
        directional_depth_consistency(depth, depth, intrinsics, intrinsics, invalid, _pose())
