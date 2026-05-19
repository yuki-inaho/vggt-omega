from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vggt_omega.utils.load_fn import load_and_preprocess_images


def test_balanced_mode_keeps_token_budget(synthetic_image_dir: Path) -> None:
    paths = sorted(str(p) for p in synthetic_image_dir.glob("*.png"))
    images = load_and_preprocess_images(paths, mode="balanced", image_resolution=256, patch_size=16)
    assert images.ndim == 4
    assert images.shape[0] == len(paths)
    assert images.shape[1] == 3
    # patch-aligned
    assert images.shape[2] % 16 == 0
    assert images.shape[3] % 16 == 0
    # token count budget should match (image_resolution/patch_size)**2
    tokens = (images.shape[2] // 16) * (images.shape[3] // 16)
    assert abs(tokens - (256 // 16) ** 2) <= 4


def test_max_size_mode_caps_longest_side(synthetic_image_dir: Path) -> None:
    paths = sorted(str(p) for p in synthetic_image_dir.glob("*.png"))
    images = load_and_preprocess_images(paths, mode="max_size", image_resolution=256, patch_size=16)
    longest = max(images.shape[2], images.shape[3])
    assert longest == 256


def test_load_returns_float_tensor_in_unit_range(synthetic_image_dir: Path) -> None:
    paths = sorted(str(p) for p in synthetic_image_dir.glob("*.png"))
    images = load_and_preprocess_images(paths, image_resolution=128, patch_size=16)
    assert isinstance(images, torch.Tensor)
    assert images.dtype == torch.float32
    assert images.min().item() >= 0.0
    assert images.max().item() <= 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"image_resolution": 0},
        {"image_resolution": 100, "patch_size": 16},  # not divisible
        {"mode": "weird"},
        {"patch_size": 0},
    ],
)
def test_invalid_arguments_raise(synthetic_image_dir: Path, kwargs: dict) -> None:
    paths = [str(next(synthetic_image_dir.glob("*.png")))]
    with pytest.raises(ValueError):
        load_and_preprocess_images(paths, **kwargs)


def test_empty_list_raises() -> None:
    with pytest.raises(ValueError):
        load_and_preprocess_images([])
