from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_RENDERING_PATH = Path(__file__).parents[1] / "vggt_omega/training/rendering.py"
_SPEC = importlib.util.spec_from_file_location("vggt_omega_training_rendering_standalone", _RENDERING_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_RENDERING = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RENDERING)
reproject_rgbd = _RENDERING.reproject_rgbd
soft_zbuffer_reproject = _RENDERING.soft_zbuffer_reproject


def _inputs(device: torch.device) -> dict[str, torch.Tensor]:
    height = width = 16
    rgb = torch.rand((1, 3, height, width), device=device, dtype=torch.float32)
    depth = torch.ones((1, height, width), device=device, dtype=torch.float32, requires_grad=True)
    intrinsics = torch.tensor(
        [[[20.0, 0.0, width / 2], [0.0, 20.0, height / 2], [0.0, 0.0, 1.0]]],
        device=device,
        requires_grad=True,
    )
    extrinsics = torch.eye(4, device=device, dtype=torch.float32)[None, :3]
    return {
        "source_rgb": rgb,
        "source_depth": depth,
        "source_intrinsics": intrinsics,
        "target_intrinsics": intrinsics,
        "source_extrinsics_w2c": extrinsics,
        "target_extrinsics_w2c": extrinsics,
    }


def test_renderer_dispatch_rejects_unknown_backend_without_importing_gsplat() -> None:
    sys.modules.pop("gsplat", None)

    with pytest.raises(ValueError, match="renderer backend"):
        reproject_rgbd("implicit-fallback", **_inputs(torch.device("cpu")))

    assert "gsplat" not in sys.modules


def test_explicit_gsplat_backend_fails_instead_of_falling_back_when_unavailable() -> None:
    if importlib.util.find_spec("gsplat") is not None:
        pytest.skip("gsplat is installed in this interpreter")

    with pytest.raises(RuntimeError, match="gsplat backend"):
        reproject_rgbd("gsplat", **_inputs(torch.device("cpu")))


@pytest.mark.gpu
def test_gsplat_renderer_has_finite_depth_and_intrinsics_gradients() -> None:
    pytest.importorskip("gsplat")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    inputs = _inputs(torch.device("cuda"))

    rendered = reproject_rgbd("gsplat", **inputs)
    soft = soft_zbuffer_reproject(**inputs)
    loss = rendered["rgb"].square().mean() + rendered["weight"].mean()
    loss.backward()

    depth_gradient = inputs["source_depth"].grad
    intrinsics_gradient = inputs["target_intrinsics"].grad
    assert rendered["rgb"].shape == inputs["source_rgb"].shape
    assert rendered["visibility"].shape == inputs["source_depth"].shape
    assert depth_gradient is not None and torch.isfinite(depth_gradient).all()
    assert intrinsics_gradient is not None and torch.isfinite(intrinsics_gradient).all()
    assert torch.count_nonzero(depth_gradient) > 0
    assert torch.count_nonzero(intrinsics_gradient) > 0
    assert torch.isfinite((rendered["rgb"] - soft["rgb"]).abs().mean())
