from __future__ import annotations

import pytest
import torch

from vggt_omega.training.overlap import bidirectional_rgbd_overlap, directional_rgbd_overlap


def _intrinsics(height: int, width: int, focal: float = 8.0) -> torch.Tensor:
    return torch.tensor(
        [[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )


def _w2c(tx: float = 0.0) -> torch.Tensor:
    transform = torch.eye(4, dtype=torch.float32)[:3].clone()
    transform[0, 3] = tx
    return transform


def test_directional_overlap_identity_is_one_and_ignores_invalid_source_depth() -> None:
    depth = torch.ones((5, 7), dtype=torch.float32)
    depth[0, 0] = 0.0
    depth[0, 1] = float("nan")
    intrinsics = _intrinsics(*depth.shape)

    score = directional_rgbd_overlap(depth, depth, intrinsics, intrinsics, _w2c(), _w2c())

    assert score.item() == pytest.approx(1.0)
    assert torch.isfinite(score)


def test_directional_overlap_large_translation_has_no_overlap() -> None:
    depth = torch.ones((5, 7), dtype=torch.float32)
    intrinsics = _intrinsics(*depth.shape)

    score = directional_rgbd_overlap(depth, depth, intrinsics, intrinsics, _w2c(), _w2c(tx=100.0))

    assert score.item() == pytest.approx(0.0)


def test_directional_overlap_rejects_occluded_or_depth_inconsistent_projection() -> None:
    source = torch.ones((5, 7), dtype=torch.float32)
    target = torch.full_like(source, 0.5)
    intrinsics = _intrinsics(*source.shape)

    score = directional_rgbd_overlap(
        source,
        target,
        intrinsics,
        intrinsics,
        _w2c(),
        _w2c(),
        relative_depth_tolerance=0.01,
    )

    assert score.item() == pytest.approx(0.0)


def test_directional_overlap_empty_source_is_defined_as_zero() -> None:
    source = torch.zeros((5, 7), dtype=torch.float32)
    target = torch.ones_like(source)
    intrinsics = _intrinsics(*source.shape)

    score = directional_rgbd_overlap(source, target, intrinsics, intrinsics, _w2c(), _w2c())

    assert score.item() == pytest.approx(0.0)
    assert torch.isfinite(score)


def test_directional_overlap_all_nan_source_is_defined_as_zero() -> None:
    source = torch.full((3, 4), float("nan"), dtype=torch.float32)
    target = torch.ones_like(source)
    intrinsics = _intrinsics(*source.shape)

    score = directional_rgbd_overlap(source, target, intrinsics, intrinsics, _w2c(), _w2c())

    assert score.item() == pytest.approx(0.0)
    assert torch.isfinite(score)


def test_overlap_rejects_nonfinite_camera_and_implicit_batch_shape() -> None:
    depth = torch.ones((3, 4), dtype=torch.float32)
    intrinsics = _intrinsics(*depth.shape)
    nonfinite_pose = _w2c()
    nonfinite_pose[0, 0] = float("nan")

    with pytest.raises(ValueError, match="finite 3x4"):
        directional_rgbd_overlap(depth, depth, intrinsics, intrinsics, nonfinite_pose, _w2c())
    with pytest.raises(ValueError, match="rank-two"):
        directional_rgbd_overlap(depth[None], depth[None], intrinsics, intrinsics, _w2c(), _w2c())


def test_near_depth_threshold_excludes_far_inconsistent_pixels() -> None:
    source = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    target = torch.tensor([[1.0, 3.0]], dtype=torch.float32)
    intrinsics = _intrinsics(*source.shape, focal=4.0)

    all_depth = directional_rgbd_overlap(source, target, intrinsics, intrinsics, _w2c(), _w2c())
    near_depth = directional_rgbd_overlap(
        source,
        target,
        intrinsics,
        intrinsics,
        _w2c(),
        _w2c(),
        max_depth_m=1.2,
    )

    assert all_depth.item() == pytest.approx(0.5)
    assert near_depth.item() == pytest.approx(1.0)


def test_bidirectional_overlap_is_mean_of_both_directions() -> None:
    first = torch.ones((5, 7), dtype=torch.float32)
    second = first.clone()
    second[:, :2] = 0.0
    intrinsics = _intrinsics(*first.shape)

    forward = directional_rgbd_overlap(first, second, intrinsics, intrinsics, _w2c(), _w2c())
    backward = directional_rgbd_overlap(second, first, intrinsics, intrinsics, _w2c(), _w2c())
    symmetric = bidirectional_rgbd_overlap(first, second, intrinsics, intrinsics, _w2c(), _w2c())

    assert symmetric.item() == pytest.approx(((forward + backward) / 2).item())
    assert 0.0 <= symmetric.item() <= 1.0


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("relative_depth_tolerance", 0.0, "relative_depth_tolerance"),
        ("pixel_stride", 0, "pixel_stride"),
        ("max_depth_m", 0.0, "max_depth_m"),
    ],
)
def test_overlap_rejects_invalid_configuration(keyword: str, value: float, message: str) -> None:
    depth = torch.ones((2, 2), dtype=torch.float32)
    intrinsics = _intrinsics(*depth.shape)

    with pytest.raises(ValueError, match=message):
        directional_rgbd_overlap(depth, depth, intrinsics, intrinsics, _w2c(), _w2c(), **{keyword: value})
