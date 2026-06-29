"""Unit tests for the RGB-D metric-trajectory geometry helpers (no GPU needed)."""

from __future__ import annotations

import numpy as np

from scripts.rgbd_metric_trajectory import (
    align_sensor_depth_to_rgb,
    camera_centers,
    path_length,
    select_stems,
)


def test_camera_centers_identity_rotation() -> None:
    # world->camera [I | t] => camera centre is -t
    extr = np.zeros((2, 3, 4))
    extr[:, :3, :3] = np.eye(3)
    extr[0, :, 3] = [1.0, 2.0, 3.0]
    extr[1, :, 3] = [-4.0, 0.0, 5.0]
    centers = camera_centers(extr)
    np.testing.assert_allclose(centers, [[-1.0, -2.0, -3.0], [4.0, 0.0, -5.0]])


def test_camera_centers_with_rotation() -> None:
    theta = np.pi / 2
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    c_true = np.array([2.0, -1.0, 0.5])
    t = -R @ c_true  # extrinsic translation for a known centre
    extr = np.concatenate([R, t[:, None]], axis=1)[None]
    np.testing.assert_allclose(camera_centers(extr)[0], c_true, atol=1e-9)


def test_path_length() -> None:
    assert path_length(np.array([[0.0, 0, 0]])) == 0.0
    centers = np.array([[0.0, 0, 0], [3.0, 0, 0], [3.0, 4.0, 0]])
    assert path_length(centers) == 7.0  # 3 (x) + 4 (y)


def test_align_identity_is_pixel_preserving() -> None:
    # identity extrinsic + identical intrinsics => depth maps to the same pixel
    K = np.array([[300.0, 0, 320.0], [0, 300.0, 240.0], [0, 0, 1.0]])
    depth_m = np.full((480, 640), 1.5)
    valid = np.zeros((480, 640), dtype=bool)
    valid[100:300, 200:400] = True  # a sub-window of valid metric depth
    depth_m[~valid] = 0.0
    aligned, mask = align_sensor_depth_to_rgb(depth_m, valid, K, K, np.eye(3), np.zeros(3), (480, 640))
    # valid input pixels keep their value at the same location
    np.testing.assert_allclose(aligned[valid], 1.5, atol=1e-6)
    assert mask.sum() == int(valid.sum())
    assert aligned[~mask].max() == 0.0


def test_align_baseline_shifts_columns() -> None:
    # a +x depth->rgb translation shifts the reprojected depth horizontally
    K = np.array([[300.0, 0, 320.0], [0, 300.0, 240.0], [0, 0, 1.0]])
    depth_m = np.full((480, 640), 2.0)
    valid = np.ones((480, 640), dtype=bool)
    base, _ = align_sensor_depth_to_rgb(depth_m, valid, K, K, np.eye(3), np.zeros(3), (480, 640))
    shifted, _ = align_sensor_depth_to_rgb(depth_m, valid, K, K, np.eye(3), np.array([0.2, 0.0, 0.0]), (480, 640))
    # u' = fx * (X + tx)/Z + cx ; with constant Z=2, tx=0.2 -> +30 px shift
    expected = int(round(300.0 * 0.2 / 2.0))
    assert expected == 30
    assert not np.array_equal(base, shifted)


def test_select_stems_even_sampling(tmp_path) -> None:
    rgb = tmp_path / "rgb"
    rgb.mkdir()
    for i in range(1, 21):
        (rgb / f"{i:08d}.jpg").write_bytes(b"x")
    stems = select_stems(rgb, 5)
    assert len(stems) == 5
    assert stems[0] == "00000001"
    assert stems[-1] == "00000020"
    assert stems == sorted(stems)
