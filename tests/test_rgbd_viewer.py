from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vggt_omega.rgbd_viewer import (
    RgbdDatasetIndex,
    RgbdLoaderConfig,
    RgbdViewerError,
    load_rgbd_frame,
    render_rgbd_gallery,
)


def _write_rgb(path: Path, value: int = 20, *, shape: tuple[int, int] = (4, 6)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((*shape, 3), value, dtype=np.uint8), mode="RGB").save(path)


def _write_depth(path: Path, *, shape: tuple[int, int] = (4, 6)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    depth = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape) * 10
    Image.fromarray(depth).save(path)


def _write_mask(path: Path, *, shape: tuple[int, int] = (4, 6), valid: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.full(shape, 255 if valid else 0, dtype=np.uint8)
    mask[0, 0] = 0
    Image.fromarray(mask, mode="L").save(path)


def _write_staging_frame(root: Path, scene: str, name: str, *, rgb_value: int = 20) -> None:
    scene_root = root / "scenes" / scene
    _write_rgb(scene_root / "rgb" / name, rgb_value)
    _write_depth(scene_root / "depth" / name)
    _write_mask(scene_root / "valid_mask" / name)


def test_staging_layout_indexes_multiple_scenes_and_loads_metric_depth(tmp_path: Path) -> None:
    _write_staging_frame(tmp_path, "scene_000000", "frame_000000.png", rgb_value=10)
    _write_staging_frame(tmp_path, "scene_000001", "frame_000001.png", rgb_value=30)

    index = RgbdDatasetIndex.discover(tmp_path)

    assert index.frame_ids == (
        "scenes/scene_000000/rgb/frame_000000.png",
        "scenes/scene_000001/rgb/frame_000001.png",
    )
    pair = index.resolve(index.frame_ids[1])
    loaded = load_rgbd_frame(pair, index.config)
    assert loaded.rgb.shape == (4, 6, 3)
    assert loaded.depth_m.shape == (4, 6)
    assert loaded.depth_m[0, 1] == pytest.approx(0.01)
    assert loaded.valid_mask.dtype == np.bool_
    assert loaded.valid_mask.sum() == 23


def test_generic_suffix_layout_can_derive_mask_explicitly(tmp_path: Path) -> None:
    _write_rgb(tmp_path / "capture" / "images" / "shot_rgb.png")
    _write_depth(tmp_path / "capture" / "mapped_depth" / "shot_depth.png")
    config = RgbdLoaderConfig(
        rgb_directory="images",
        depth_directory="mapped_depth",
        mask_directory=None,
        rgb_key_suffix="_rgb",
        depth_key_suffix="_depth",
        derive_mask_when_missing=True,
        depth_scale_to_m=0.002,
    )

    index = RgbdDatasetIndex.discover(tmp_path, config)
    loaded = load_rgbd_frame(index.pairs[0], config)

    assert index.frame_ids == ("capture/images/shot_rgb.png",)
    assert loaded.depth_m[0, 1] == pytest.approx(0.02)
    assert np.array_equal(loaded.valid_mask, loaded.depth_m > 0)
    assert loaded.mask_source == "derived_from_depth"


def test_uploaded_duplicate_basename_is_resolved_by_content_digest(tmp_path: Path) -> None:
    _write_staging_frame(tmp_path, "scene_000000", "frame.png", rgb_value=10)
    _write_staging_frame(tmp_path, "scene_000001", "frame.png", rgb_value=30)
    uploaded = tmp_path / "upload" / "frame.png"
    _write_rgb(uploaded, value=30)
    index = RgbdDatasetIndex.discover(tmp_path / "scenes")

    pair = index.resolve(uploaded)

    assert "scene_000001" in pair.frame_id


def test_ambiguous_or_missing_pairs_fail_closed(tmp_path: Path) -> None:
    _write_staging_frame(tmp_path, "scene_000000", "frame.png", rgb_value=10)
    _write_staging_frame(tmp_path, "scene_000001", "frame.png", rgb_value=10)
    index = RgbdDatasetIndex.discover(tmp_path)

    with pytest.raises(RgbdViewerError, match="ambiguous"):
        index.resolve("frame.png")

    (tmp_path / "scenes" / "scene_000001" / "depth" / "frame.png").unlink()
    with pytest.raises(RgbdViewerError, match="missing depth"):
        RgbdDatasetIndex.discover(tmp_path)


def test_shape_and_stored_mask_mismatches_are_rejected(tmp_path: Path) -> None:
    _write_staging_frame(tmp_path, "scene_000000", "frame.png")
    index = RgbdDatasetIndex.discover(tmp_path)
    pair = index.pairs[0]
    assert pair.mask_path is not None
    _write_mask(pair.mask_path, shape=(3, 6))

    with pytest.raises(RgbdViewerError, match="shape"):
        load_rgbd_frame(pair, index.config)

    _write_mask(pair.mask_path, valid=False)
    with pytest.raises(RgbdViewerError, match="does not match"):
        load_rgbd_frame(pair, index.config)


def test_render_gallery_returns_four_views_and_metric_statistics(tmp_path: Path) -> None:
    _write_staging_frame(tmp_path, "scene_000000", "frame.png")
    index = RgbdDatasetIndex.discover(tmp_path)
    loaded = load_rgbd_frame(index.pairs[0], index.config)

    rendered = render_rgbd_gallery([loaded], depth_ceiling_m=0.2, overlay_alpha=0.5)

    assert len(rendered.gallery) == 4
    assert [caption.rsplit(" / ", 1)[-1] for _, caption in rendered.gallery] == [
        "RGB",
        "Depth (0-0.200 m)",
        "Valid mask",
        "RGB + depth",
    ]
    assert all(image.size == (6, 4) for image, _ in rendered.gallery)
    assert rendered.depth_ceiling_m == pytest.approx(0.2)
    assert rendered.statistics[0][0] == index.frame_ids[0]
    assert rendered.statistics[0][2] == 23
    assert rendered.statistics[0][3] == pytest.approx(100 * 23 / 24)


def test_render_gallery_can_choose_a_shared_auto_depth_ceiling(tmp_path: Path) -> None:
    _write_staging_frame(tmp_path, "scene_000000", "frame.png")
    index = RgbdDatasetIndex.discover(tmp_path)
    loaded = load_rgbd_frame(index.pairs[0], index.config)

    rendered = render_rgbd_gallery([loaded], depth_ceiling_m=None, overlay_alpha=0.25)

    valid_values = loaded.depth_m[loaded.valid_mask]
    assert rendered.depth_ceiling_m == pytest.approx(float(np.quantile(valid_values, 0.98)))
    with pytest.raises(RgbdViewerError, match="overlay_alpha"):
        render_rgbd_gallery([loaded], depth_ceiling_m=1.0, overlay_alpha=2.0)
