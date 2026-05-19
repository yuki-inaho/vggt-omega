from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vggt_omega.inference import InferenceResult
from vggt_omega.visualize import (
    _build_keep_mask,
    _depth_edge_mask,
    _frame_color,
    _subsample,
    make_blueprint,
    save_results_to_rrd,
)


def _fake_inference_result(h: int = 32, w: int = 48, seed: int = 0) -> InferenceResult:
    rng = np.random.default_rng(seed)
    return InferenceResult(
        image=rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8),
        width=w,
        height=h,
        extrinsic=np.array([[1.0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32),
        intrinsic=np.array([[w * 1.0, 0, w / 2], [0, h * 1.0, h / 2], [0, 0, 1]], dtype=np.float32),
        depth_map=rng.uniform(0.5, 5.0, size=(h, w)).astype(np.float32),
        depth_conf=rng.uniform(1.0, 5.0, size=(h, w)).astype(np.float32),
        point_map_by_unprojection=rng.normal(size=(h, w, 3)).astype(np.float32),
    )


def test_depth_edge_mask_flags_jumps() -> None:
    depth = np.ones((6, 6), dtype=np.float32)
    depth[3:, :] = 10.0
    mask = _depth_edge_mask(depth, rtol=0.5)
    assert mask.shape == depth.shape
    assert mask[2:4, :].any()


def test_build_keep_mask_respects_conf_threshold() -> None:
    res = _fake_inference_result()
    keep = _build_keep_mask(
        conf=res.depth_conf,
        colors=res.image,
        depth=res.depth_map,
        conf_percent=50.0,
        mask_black_bg=False,
        mask_white_bg=False,
        filter_depth_edges=False,
        depth_edge_rtol=0.03,
    )
    assert keep.shape == res.depth_conf.shape
    assert keep.sum() <= res.depth_conf.size  # at most ~half remain
    assert keep.sum() > 0


def test_build_keep_mask_black_white_filters() -> None:
    res = _fake_inference_result()
    res.image[:, :, :] = 0  # all black
    keep = _build_keep_mask(
        conf=res.depth_conf,
        colors=res.image,
        depth=None,
        conf_percent=0.0,
        mask_black_bg=True,
        mask_white_bg=False,
        filter_depth_edges=False,
        depth_edge_rtol=0.03,
    )
    assert keep.sum() == 0


def test_subsample_below_budget_is_noop() -> None:
    pts = np.zeros((5, 3), dtype=np.float32)
    rgb = np.zeros((5, 3), dtype=np.uint8)
    out_p, out_c = _subsample(pts, rgb, max_points=100)
    assert out_p is pts
    assert out_c is rgb


def test_subsample_above_budget_returns_uniform() -> None:
    pts = np.arange(30, dtype=np.float32).reshape(10, 3)
    rgb = np.zeros((10, 3), dtype=np.uint8)
    out_p, out_c = _subsample(pts, rgb, max_points=4)
    assert out_p.shape == (4, 3)
    assert out_c.shape == (4, 3)


def test_frame_color_distinct_across_frames() -> None:
    c0 = _frame_color(0, 4)
    c1 = _frame_color(2, 4)
    assert c0 != c1


def test_make_blueprint_constructs() -> None:
    bp = make_blueprint()
    assert bp is not None


def test_save_results_to_rrd_writes_file(tmp_path: Path) -> None:
    results = [_fake_inference_result(seed=i) for i in range(2)]
    out = tmp_path / "scene.rrd"
    saved = save_results_to_rrd(results, out, max_points=128)
    assert saved == out
    assert out.is_file()
    assert out.stat().st_size > 0


def test_save_results_to_rrd_accepts_scene_result(tmp_path: Path) -> None:
    pytest.importorskip("rerun")
    import torch

    from vggt_omega.pipeline import SceneResult

    h, w, n = 16, 24, 2
    images = torch.rand(n, 3, h, w)
    scene = SceneResult(
        images=images,
        pose_enc=np.zeros((n, 9), dtype=np.float32),
        extrinsic=np.tile(np.eye(3, 4, dtype=np.float32)[None], (n, 1, 1)),
        intrinsic=np.tile(
            np.array([[w * 1.0, 0, w / 2], [0, h * 1.0, h / 2], [0, 0, 1]], dtype=np.float32)[None],
            (n, 1, 1),
        ),
        depth=np.ones((n, h, w, 1), dtype=np.float32),
        depth_conf=np.ones((n, h, w), dtype=np.float32),
    ).with_world_points()
    out = tmp_path / "scene.rrd"
    save_results_to_rrd(scene, out, max_points=64)
    assert out.is_file()
