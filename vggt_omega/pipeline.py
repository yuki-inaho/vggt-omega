"""Reusable VGGT-Omega inference pipeline.

The pipeline encapsulates checkpoint loading, device selection and the
``model.forward`` + post-processing dance that ``demo_gradio.py`` had inlined.
Importing this module has no side effects (no implicit weight loading); it
is intended to be used directly from notebooks, tests and CLIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .models import VGGTOmega
from .utils.geometry import unproject_depth_map_to_point_map
from .utils.pose_enc import encoding_to_camera

DEFAULT_CHECKPOINT_512 = Path("checkpoints/vggt_omega_1b_512.pt")
DEFAULT_CHECKPOINT_256_TEXT = Path("checkpoints/vggt_omega_1b_256_text.pt")


@dataclass
class SceneResult:
    """Numpy-side outputs for one VGGT-Omega scene.

    Shape conventions (``N`` = number of frames, ``H/W`` = image height/width):

    - ``images``: ``(N, 3, H, W)`` torch tensor on inference device.
    - ``pose_enc``: ``(N, 9)`` per-frame 9D camera encoding.
    - ``extrinsic``: ``(N, 3, 4)`` world-to-camera matrices (OpenCV convention).
    - ``intrinsic``: ``(N, 3, 3)`` pinhole intrinsics.
    - ``depth``: ``(N, H, W, 1)`` predicted depth.
    - ``depth_conf``: ``(N, H, W)`` per-pixel depth confidence.
    - ``world_points``: ``(N, H, W, 3)`` depth back-projected to world coords
      (filled lazily by :meth:`with_world_points`).
    """

    images: torch.Tensor
    pose_enc: np.ndarray
    extrinsic: np.ndarray
    intrinsic: np.ndarray
    depth: np.ndarray
    depth_conf: np.ndarray
    camera_tokens: np.ndarray | None = None
    register_tokens: np.ndarray | None = None
    text_alignment_embedding: np.ndarray | None = None
    world_points: np.ndarray | None = None

    def with_world_points(self) -> SceneResult:
        if self.world_points is None:
            self.world_points = unproject_depth_map_to_point_map(
                self.depth,
                self.extrinsic,
                self.intrinsic,
            )
        return self

    def as_npz_dict(self) -> dict[str, Any]:
        d = self.with_world_points()
        return {
            "images": d.images.detach().cpu().numpy() if isinstance(d.images, torch.Tensor) else d.images,
            "pose_enc": d.pose_enc,
            "extrinsic": d.extrinsic,
            "intrinsic": d.intrinsic,
            "depth": d.depth,
            "depth_conf": d.depth_conf,
            "world_points_from_depth": d.world_points,
            **(
                {"text_alignment_embedding": d.text_alignment_embedding}
                if d.text_alignment_embedding is not None
                else {}
            ),
        }


def autodetect_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class VGGTOmegaPipeline:
    """High-level wrapper bundling checkpoint loading and inference."""

    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT_512,
        device: torch.device | str | None = None,
        enable_alignment: bool = False,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device) if device is not None else autodetect_device()
        self.model = self._build_model(self.checkpoint_path, self.device, enable_alignment)

    @staticmethod
    def _build_model(checkpoint_path: Path, device: torch.device, enable_alignment: bool) -> nn.Module:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        model = VGGTOmega(enable_alignment=enable_alignment).eval()
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)
        return model.to(device)

    @torch.inference_mode()
    def run(self, images: torch.Tensor) -> SceneResult:
        """Run inference on a preprocessed BCHW tensor and return a SceneResult."""
        images = images.to(self.device)
        predictions = self.model(images)
        return _predictions_to_scene_result(predictions)


def _predictions_to_scene_result(predictions: dict[str, torch.Tensor]) -> SceneResult:
    images = predictions["images"]
    extrinsic, intrinsic = encoding_to_camera(predictions["pose_enc"], images.shape[-2:])

    def to_np(tensor: torch.Tensor) -> np.ndarray:
        array = tensor.detach().float().cpu().numpy()
        return array[0] if array.shape[0] == 1 else array

    camera_and_register = predictions.get("camera_and_register_tokens")
    camera_tokens = register_tokens = None
    if camera_and_register is not None:
        cr_np = to_np(camera_and_register)
        camera_tokens = cr_np[:, :1]
        register_tokens = cr_np[:, 1:]

    return SceneResult(
        images=images[0] if images.shape[0] == 1 else images,
        pose_enc=to_np(predictions["pose_enc"]),
        extrinsic=to_np(extrinsic),
        intrinsic=to_np(intrinsic),
        depth=to_np(predictions["depth"]),
        depth_conf=to_np(predictions["depth_conf"]),
        camera_tokens=camera_tokens,
        register_tokens=register_tokens,
        text_alignment_embedding=(
            to_np(predictions["text_alignment_embedding"]) if "text_alignment_embedding" in predictions else None
        ),
    )
