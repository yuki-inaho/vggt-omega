from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.reconstruct_training_comparison import (
    _depth_diagnostics,
    _depth_preview,
    _first_camera_identity_error,
    _select_points,
    _validate_reconstruction_contract,
    _write_binary_ply,
)
from vggt_omega.pipeline import SceneResult


def _scene() -> SceneResult:
    height, width = 2, 3
    images = torch.arange(2 * 3 * height * width, dtype=torch.float32).reshape(2, 3, height, width)
    images /= images.max()
    depth = np.array(
        [
            [[0.2, 0.6, 1.4], [0.5, 1.0, 0.8]],
            [[0.3, 0.7, 1.5], [0.4, 1.1, 0.9]],
        ],
        dtype=np.float32,
    )[..., None]
    world_points = np.zeros((2, height, width, 3), dtype=np.float32)
    world_points[..., 0] = np.arange(2 * height * width).reshape(2, height, width)
    world_points[..., 2] = depth[..., 0]
    extrinsic = np.repeat(np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)[None], 2, axis=0)
    intrinsic = np.repeat(np.eye(3)[None], 2, axis=0)
    return SceneResult(
        images=images,
        pose_enc=np.zeros((2, 9), dtype=np.float32),
        extrinsic=extrinsic.astype(np.float32),
        intrinsic=intrinsic.astype(np.float32),
        depth=depth,
        depth_conf=np.ones((2, height, width), dtype=np.float32),
        world_points=world_points,
    )


def test_point_selection_is_deterministic_and_respects_near_depth() -> None:
    scene = _scene()

    first_points, first_colors = _select_points(scene, scale_m=1.0, near_depth_m=1.2, max_points=4)
    second_points, second_colors = _select_points(scene, scale_m=1.0, near_depth_m=1.2, max_points=4)

    assert first_points.shape == first_colors.shape == (4, 3)
    np.testing.assert_array_equal(first_points, second_points)
    np.testing.assert_array_equal(first_colors, second_colors)
    assert first_colors.dtype == np.uint8


def test_binary_ply_schema_and_payload_count_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "anonymous.ply"
    points = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, 0.5]], dtype=np.float32)
    colors = np.array([[1, 2, 3], [253, 254, 255]], dtype=np.uint8)

    _write_binary_ply(path, points, colors)

    raw = path.read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:end].decode("ascii")
    vertex_dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    vertices = np.frombuffer(raw, dtype=vertex_dtype, offset=end)
    assert "format binary_little_endian 1.0" in header
    assert "element vertex 2" in header
    assert len(vertices) == 2
    np.testing.assert_allclose(np.column_stack((vertices["x"], vertices["y"], vertices["z"])), points)


def test_binary_ply_rejects_nonfinite_points(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="finite"):
        _write_binary_ply(
            tmp_path / "invalid.ply",
            np.array([[np.nan, 0.0, 1.0]], dtype=np.float32),
            np.zeros((1, 3), dtype=np.uint8),
        )


def test_rectangular_depth_diagnostics_and_camera_identity_are_finite() -> None:
    depth = _scene().depth[..., 0]

    diagnostics = _depth_diagnostics(depth)
    preview = _depth_preview(depth, near_depth_m=1.2)

    assert preview.shape == (2, 3, 3)
    assert preview.dtype == np.uint8
    assert diagnostics["finite_depth_fraction"] == 1.0
    assert diagnostics["spatial_edge_l1_m"] > 0
    assert diagnostics["temporal_depth_change_proxy_m"] > 0
    assert _first_camera_identity_error(_scene().extrinsic) == 0.0


def test_reconstruction_contract_accepts_matching_finite_scenes() -> None:
    _validate_reconstruction_contract(_scene(), _scene(), scale_m=0.75)


@pytest.mark.parametrize("scale_m", [0.0, -1.0, np.nan])
def test_reconstruction_contract_rejects_invalid_scene_scale(scale_m: float) -> None:
    with pytest.raises(ValueError, match="scale"):
        _validate_reconstruction_contract(_scene(), _scene(), scale_m=scale_m)


def test_reconstruction_contract_rejects_nonfinite_geometry() -> None:
    invalid = _scene()
    assert invalid.world_points is not None
    invalid.world_points[0, 0, 0, 0] = np.nan

    with pytest.raises(ValueError, match="non-finite geometry"):
        _validate_reconstruction_contract(_scene(), invalid, scale_m=1.0)


def test_reconstruction_contract_rejects_nonpositive_depth() -> None:
    invalid = _scene()
    invalid.depth[0, 0, 0, 0] = 0.0

    with pytest.raises(ValueError, match="non-positive depth"):
        _validate_reconstruction_contract(_scene(), invalid, scale_m=1.0)


def test_reconstruction_contract_rejects_noncanonical_first_camera() -> None:
    invalid = _scene()
    invalid.extrinsic[0, 0, 3] = 0.02

    with pytest.raises(ValueError, match="first camera"):
        _validate_reconstruction_contract(_scene(), invalid, scale_m=1.0)
