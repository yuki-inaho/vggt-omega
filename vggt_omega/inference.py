"""Official-VGGT-style inference API for VGGT-Omega.

This mirrors the surface area of
``vggt-pytorch-inference/vggt_inference/vggt_inference.py``
(``VGGTInference`` + ``InferenceResult``) so call sites can move between the
projects without rewriting glue code. The pipeline (checkpoint load, model
forward, post-processing) lives in :mod:`vggt_omega.pipeline`; this layer
focuses on the public, frame-major API.

Typical use::

    from vggt_omega.inference import VGGTOmegaInference
    from vggt_omega.preprocess import read_images_from_video

    model = VGGTOmegaInference(checkpoint_path="checkpoints/vggt_omega_1b_512.pt")
    frames = read_images_from_video("examples/forest_road.mp4", sample_fps=1.0)
    results = model(frames)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .pipeline import DEFAULT_CHECKPOINT_512, SceneResult, VGGTOmegaPipeline, autodetect_device
from .preprocess import PreprocessMode, preprocess_images


@dataclass
class InferenceResult:
    """Per-frame VGGT-Omega output (mirrors the official VGGT signature)."""

    image: np.ndarray  # (H, W, 3) uint8
    width: int
    height: int
    extrinsic: np.ndarray  # (3, 4)
    intrinsic: np.ndarray  # (3, 3)
    depth_map: np.ndarray  # (H, W)
    depth_conf: np.ndarray  # (H, W)
    point_map_by_unprojection: np.ndarray  # (H, W, 3)


class VGGTOmegaInference(nn.Module):
    """``nn.Module`` wrapper presenting the official VGGT API for VGGT-Omega."""

    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT_512,
        image_resolution: int = 512,
        device: torch.device | str | None = None,
        enable_alignment: bool = False,
    ) -> None:
        super().__init__()
        self.image_resolution = image_resolution
        self.device_obj = torch.device(device) if device is not None else autodetect_device()
        self.pipeline = VGGTOmegaPipeline(
            checkpoint_path=checkpoint_path,
            device=self.device_obj,
            enable_alignment=enable_alignment,
        )

    @property
    def model(self) -> nn.Module:
        """Underlying VGGT-Omega backbone (for parity with VGGTInference.model)."""
        return self.pipeline.model

    def forward(
        self,
        input_images: list[np.ndarray],
        mode: PreprocessMode = "balanced",
    ) -> list[InferenceResult]:
        tensor = preprocess_images(
            input_images,
            image_resolution=self.image_resolution,
            mode=mode,
        )
        scene = self.pipeline.run(tensor)
        return scene_result_to_inference_results(scene)


def scene_result_to_inference_results(scene: SceneResult) -> list[InferenceResult]:
    scene = scene.with_world_points()
    images = scene.images.detach().cpu().numpy() if isinstance(scene.images, torch.Tensor) else scene.images
    rgb_uint8 = (np.transpose(images, (0, 2, 3, 1)) * 255.0).clip(0, 255).astype(np.uint8)
    num_frames, height, width, _ = rgb_uint8.shape

    depth = scene.depth[..., 0] if scene.depth.ndim == 4 else scene.depth

    return [
        InferenceResult(
            image=rgb_uint8[i],
            width=int(width),
            height=int(height),
            extrinsic=scene.extrinsic[i],
            intrinsic=scene.intrinsic[i],
            depth_map=depth[i],
            depth_conf=scene.depth_conf[i],
            point_map_by_unprojection=scene.world_points[i],
        )
        for i in range(num_frames)
    ]
