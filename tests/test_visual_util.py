from __future__ import annotations

import numpy as np

from visual_util import (
    _limit_points,
    compute_camera_faces,
    depth_edge,
    get_opengl_conversion_matrix,
    transform_points,
)


def test_depth_edge_flags_jumps() -> None:
    depth = np.ones((1, 8, 8), dtype=np.float32)
    depth[0, 3:, :] = 10.0
    edges = depth_edge(depth, rtol=0.5)
    assert edges.shape == depth.shape
    assert edges[0, 2:4, :].any()
    assert not edges[0, 0, 0]


def test_depth_edge_no_jump() -> None:
    depth = np.ones((2, 4, 4), dtype=np.float32)
    edges = depth_edge(depth, rtol=0.5)
    assert not edges.any()


def test_limit_points_uniform_subsample() -> None:
    verts = np.arange(30).reshape(10, 3).astype(np.float32)
    colors = (verts * 10).clip(0, 255).astype(np.uint8)
    sub_v, sub_c = _limit_points(verts, colors, max_points=4)
    assert sub_v.shape == (4, 3)
    assert sub_c.shape == (4, 3)
    assert sub_v[0].tolist() == verts[0].tolist()
    assert sub_v[-1].tolist() == verts[-1].tolist()


def test_limit_points_below_budget_is_noop() -> None:
    verts = np.zeros((3, 3), dtype=np.float32)
    colors = np.zeros((3, 3), dtype=np.uint8)
    sub_v, sub_c = _limit_points(verts, colors, max_points=10)
    assert sub_v is verts
    assert sub_c is colors


def test_opengl_conversion_matrix_flips_y_and_z() -> None:
    m = get_opengl_conversion_matrix()
    assert m.shape == (4, 4)
    assert m[1, 1] == -1
    assert m[2, 2] == -1
    assert m[0, 0] == 1
    assert m[3, 3] == 1


def test_transform_points_identity_preserves_points() -> None:
    points = np.random.default_rng(0).normal(size=(7, 3)).astype(np.float32)
    out = transform_points(np.eye(4), points)
    np.testing.assert_allclose(out, points, atol=1e-6)


def test_transform_points_translation() -> None:
    points = np.zeros((4, 3), dtype=np.float32)
    transform = np.eye(4)
    transform[:3, 3] = [1.0, 2.0, 3.0]
    out = transform_points(transform, points)
    np.testing.assert_allclose(out, np.broadcast_to([1, 2, 3], (4, 3)))


def test_compute_camera_faces_runs() -> None:
    import trimesh

    cone = trimesh.creation.cone(0.1, 0.2, sections=4)
    faces = compute_camera_faces(cone)
    assert faces.ndim == 2
    assert faces.shape[1] == 3
    assert faces.shape[0] > 0
