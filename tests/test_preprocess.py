from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from vggt_omega.preprocess import (
    load_images_from_paths,
    preprocess_images,
    read_images_from_video,
)


def _make_frame(rng: np.random.Generator, h: int = 64, w: int = 96) -> np.ndarray:
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def test_preprocess_images_balanced_returns_bchw(rng: np.random.Generator) -> None:
    frames = [_make_frame(rng) for _ in range(3)]
    out = preprocess_images(frames, image_resolution=128, mode="balanced", patch_size=16)
    assert isinstance(out, torch.Tensor)
    assert out.ndim == 4
    assert out.shape[0] == 3
    assert out.shape[1] == 3
    assert out.shape[2] % 16 == 0
    assert out.shape[3] % 16 == 0


def test_preprocess_images_max_size_caps_longest_side(rng: np.random.Generator) -> None:
    frames = [_make_frame(rng, h=48, w=128)]
    out = preprocess_images(frames, image_resolution=128, mode="max_size", patch_size=16)
    assert max(out.shape[2], out.shape[3]) == 128


def test_preprocess_images_empty_raises() -> None:
    with pytest.raises(ValueError):
        preprocess_images([], image_resolution=128)


def test_preprocess_images_validates_shape(rng: np.random.Generator) -> None:
    with pytest.raises(ValueError):
        preprocess_images([rng.integers(0, 256, size=(32, 32), dtype=np.uint8)], image_resolution=128)


def test_preprocess_images_validates_shape_via_disk(rng: np.random.Generator, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        preprocess_images(
            [rng.integers(0, 256, size=(32, 32), dtype=np.uint8)],
            image_resolution=128,
            tmp_dir=tmp_path / "frames",
        )


def test_preprocess_images_scales_unit_float_input(rng: np.random.Generator) -> None:
    frame = rng.random((32, 32, 3), dtype=np.float32)
    out = preprocess_images([frame], image_resolution=64)
    assert out.max() > 0.5
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_preprocess_images_disk_path_matches_in_memory(rng: np.random.Generator, tmp_path: Path) -> None:
    frames = [_make_frame(rng, h=64, w=96) for _ in range(2)]
    in_mem = preprocess_images(frames, image_resolution=128, mode="balanced")
    via_disk = preprocess_images(frames, image_resolution=128, mode="balanced", tmp_dir=tmp_path / "frames")
    # Should agree up to PNG/JPEG rounding.
    assert in_mem.shape == via_disk.shape
    assert torch.allclose(in_mem, via_disk, atol=1e-3)


def test_load_images_from_paths_calls_legacy_loader(synthetic_image_dir: Path) -> None:
    out = load_images_from_paths(
        sorted(synthetic_image_dir.glob("*.png")),
        image_resolution=128,
    )
    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == len(list(synthetic_image_dir.glob("*.png")))


def test_read_images_from_video_yields_frames() -> None:
    video = Path("examples/forest_road.mp4")
    if not video.is_file():
        pytest.skip("example video missing")
    frames = read_images_from_video(video, sample_fps=0.5)
    assert len(frames) >= 1
    assert frames[0].ndim == 3 and frames[0].shape[-1] == 3


def test_read_images_from_video_respects_max_frames() -> None:
    video = Path("examples/forest_road.mp4")
    if not video.is_file():
        pytest.skip("example video missing")
    frames = read_images_from_video(video, sample_fps=1.0, max_frames=1)
    assert len(frames) == 1


def test_read_images_from_video_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        read_images_from_video(tmp_path / "nope.mp4")
