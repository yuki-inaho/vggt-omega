from __future__ import annotations

import numpy as np
import pytest

from scripts.align_vggt4d_chunks import (
    align_local_chunk,
    average_rigid_transforms,
    constrain_positions_near_rail,
    matrix_to_quaternion_wxyz,
    project_positions_to_rail,
)
from scripts.rgbd_sfm_pilot import quaternion_wxyz_to_matrix


def _pose(x: float, y: float = 0.0, yaw_deg: float = 0.0) -> np.ndarray:
    angle = np.deg2rad(yaw_deg)
    cosine, sine = np.cos(angle), np.sin(angle)
    result = np.eye(4)
    result[:3, :3] = [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    result[:3, 3] = [x, y, 0.0]
    return result


def test_matrix_quaternion_round_trip() -> None:
    rotation = _pose(0.0, yaw_deg=37.0)[:3, :3]

    quaternion = matrix_to_quaternion_wxyz(rotation)

    np.testing.assert_allclose(quaternion_wxyz_to_matrix(quaternion), rotation, atol=1e-7)
    assert quaternion[0] >= 0.0


def test_average_rigid_transforms_recovers_exact_transform() -> None:
    expected = _pose(2.0, y=-0.3, yaw_deg=15.0)

    averaged = average_rigid_transforms([expected, expected.copy()])

    np.testing.assert_allclose(averaged, expected, atol=1e-9)


def test_align_local_chunk_uses_all_shared_frames() -> None:
    global_poses = {2: _pose(2.0), 3: _pose(3.0)}
    local_poses = {2: _pose(0.0), 3: _pose(1.0), 4: _pose(2.0), 5: _pose(3.0)}

    aligned, edge = align_local_chunk(local_poses, global_poses)

    np.testing.assert_allclose(aligned[4], _pose(4.0), atol=1e-9)
    np.testing.assert_allclose(aligned[5], _pose(5.0), atol=1e-9)
    assert edge["shared_global_indices"] == [2, 3]
    assert edge["translation_rms_m"] == pytest.approx(0.0, abs=1e-12)
    assert edge["rotation_rms_deg"] == pytest.approx(0.0, abs=1e-12)


def test_align_local_chunk_rejects_disconnected_chunk() -> None:
    with pytest.raises(ValueError, match="no shared frame"):
        align_local_chunk({4: _pose(0.0), 5: _pose(1.0)}, {0: _pose(0.0)})


def test_project_positions_to_rail_removes_orthogonal_drift() -> None:
    positions = np.array([[0.0, 3.0, -2.0], [2.0, -4.0, 5.0]])
    rail = {"rail_axis": [1.0, 0.0, 0.0], "rail_centroid_m": [1.0, 2.0, 3.0]}

    projected = project_positions_to_rail(positions, rail)

    np.testing.assert_allclose(projected, [[0.0, 2.0, 3.0], [2.0, 2.0, 3.0]])


def test_near_rail_constraint_keeps_small_non_collinear_residual() -> None:
    positions = np.array([[0.0, 1.0, 0.0], [2.0, -1.0, 0.0]])
    rail = {
        "rail_axis": [1.0, 0.0, 0.0],
        "rail_centroid_m": [1.0, 0.0, 0.0],
        "orthogonal_rms_m": 1.0,
    }

    constrained, retained = constrain_positions_near_rail(positions, rail)

    assert retained == pytest.approx(0.005)
    np.testing.assert_allclose(constrained[:, 1], [0.005, -0.005])
