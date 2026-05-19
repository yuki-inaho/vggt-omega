from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def torch_rng() -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(0)
    return g


@pytest.fixture
def synthetic_image_dir(tmp_path: Path, rng: np.random.Generator) -> Path:
    out = tmp_path / "images"
    out.mkdir()
    for i in range(3):
        arr = rng.integers(0, 256, size=(96, 160, 3), dtype=np.uint8)
        Image.fromarray(arr).save(out / f"frame_{i:03d}.png")
    return out


def has_cuda() -> bool:
    return torch.cuda.is_available()


def has_checkpoint() -> bool:
    return Path("checkpoints/vggt_omega_1b_512.pt").is_file()


needs_cuda = pytest.mark.skipif(not has_cuda(), reason="CUDA not available")
needs_ckpt = pytest.mark.skipif(not has_checkpoint(), reason="VGGT-Omega 512 checkpoint not found")
