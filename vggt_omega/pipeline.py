"""Reusable VGGT-Omega inference pipeline.

The pipeline encapsulates checkpoint loading, device selection and the
``model.forward`` + post-processing dance that ``demo_gradio.py`` had inlined.
Importing this module has no side effects (no implicit weight loading); it
is intended to be used directly from notebooks, tests and CLIs.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from jaxtyping import Float
from torch import nn

from .models import VGGTOmega
from .utils.geometry import unproject_depth_map_to_point_map
from .utils.load_fn import load_checkpoint_state_dict
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

    images: Float[torch.Tensor, "n_img 3 h w"]
    pose_enc: Float[np.ndarray, "n_img 9"]
    extrinsic: Float[np.ndarray, "n_img 3 4"]
    intrinsic: Float[np.ndarray, "n_img 3 3"]
    depth: Float[np.ndarray, "n_img h w 1"]
    depth_conf: Float[np.ndarray, "n_img h w"]
    camera_tokens: Float[np.ndarray, "n_img 1 embed"] | None = None
    register_tokens: Float[np.ndarray, "n_img reg_tokens embed"] | None = None
    text_alignment_embedding: Float[np.ndarray, "embed"] | None = None
    world_points: Float[np.ndarray, "n_img h w 3"] | None = None

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
        head_checkpoint_path: str | Path | None = None,
        device: torch.device | str | None = None,
        enable_alignment: bool = False,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.head_checkpoint_path = Path(head_checkpoint_path) if head_checkpoint_path is not None else None
        self.device = torch.device(device) if device is not None else autodetect_device()
        self.model = self._build_model(self.checkpoint_path, self.device, enable_alignment)
        self.recommended_input_shape: tuple[int, int] | None = None
        if self.head_checkpoint_path is not None:
            self.recommended_input_shape = self._apply_head_checkpoint(
                self.model,
                self.head_checkpoint_path,
                self.checkpoint_path,
            )
        self.model.eval()

    @staticmethod
    def _build_model(checkpoint_path: Path, device: torch.device, enable_alignment: bool) -> nn.Module:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        model = VGGTOmega(enable_alignment=enable_alignment).eval()
        state_dict = load_checkpoint_state_dict(checkpoint_path)
        model.load_state_dict(state_dict, strict=True)
        return model.to(device)

    @staticmethod
    def _apply_head_checkpoint(model: nn.Module, head_path: Path, base_path: Path) -> tuple[int, int]:
        """Strictly overlay a training best/resume head on its exact released base."""
        if head_path.suffix == ".zst":
            raise ValueError("decompress .zst checkpoints before loading them")
        if head_path.is_symlink() or not head_path.is_file():
            raise FileNotFoundError(f"Head checkpoint not found or not a regular file: {head_path}")
        payload = torch.load(head_path, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(payload, Mapping):
            raise ValueError("head checkpoint payload must be a mapping")
        if payload.get("format_version") != 1 or payload.get("kind") not in {"best", "resume"}:
            raise ValueError("head checkpoint is not a supported VGGT-Omega training artifact")
        if payload.get("parameter_state") != "x":
            raise ValueError("head checkpoint must contain AMUSE evaluation weights (parameter_state=x)")

        metadata = payload.get("metadata")
        base_metadata = metadata.get("base_checkpoint") if isinstance(metadata, Mapping) else None
        if not isinstance(base_metadata, Mapping):
            raise ValueError("head checkpoint does not identify its base checkpoint")
        expected_size = base_metadata.get("size_bytes")
        expected_sha = base_metadata.get("sha256")
        if expected_size != base_path.stat().st_size or expected_sha != _sha256_file(base_path):
            raise ValueError("head checkpoint was trained from a different base checkpoint")

        model_state = payload.get("model_state")
        if not isinstance(model_state, Mapping):
            raise ValueError("head checkpoint model_state must be a mapping")
        expected_names = {
            name
            for name, _ in model.named_parameters()
            if (name.startswith("camera_head.") or name.startswith("dense_head."))
            and not name.startswith("dense_head.proj_conf.")
        }
        actual_names = set(model_state)
        if actual_names != expected_names:
            raise ValueError(
                "head checkpoint does not exactly match the trainable camera/depth state: "
                f"missing={sorted(expected_names - actual_names)}, unexpected={sorted(actual_names - expected_names)}"
            )
        if any(not isinstance(name, str) or not isinstance(value, torch.Tensor) for name, value in model_state.items()):
            raise ValueError("head checkpoint model_state must map names to tensors")
        if any(not torch.isfinite(value).all() for value in model_state.values()):
            raise ValueError("head checkpoint contains non-finite tensors")
        try:
            incompatible = model.load_state_dict(dict(model_state), strict=False)
        except RuntimeError as error:
            raise ValueError("head checkpoint tensors are incompatible with the base model") from error
        if incompatible.unexpected_keys:
            raise ValueError(f"head checkpoint produced unexpected keys: {incompatible.unexpected_keys}")

        config = payload.get("config")
        model_config = config.get("model") if isinstance(config, Mapping) else None
        if not isinstance(model_config, Mapping):
            raise ValueError("head checkpoint has no model input-shape configuration")
        height = model_config.get("image_height")
        width = model_config.get("image_width")
        if (
            isinstance(height, bool)
            or isinstance(width, bool)
            or not isinstance(height, int)
            or not isinstance(width, int)
        ):
            raise ValueError("head checkpoint image dimensions must be integers")
        if height <= 0 or width <= 0 or height % 16 or width % 16:
            raise ValueError("head checkpoint image dimensions must be positive multiples of 16")
        return height, width

    @torch.inference_mode()
    def run(self, images: Float[torch.Tensor, "n_img 3 h w"]) -> SceneResult:
        """Run inference on a preprocessed BCHW tensor and return a SceneResult."""
        images = images.to(self.device)
        predictions = self.model(images)
        return _predictions_to_scene_result(predictions)


def _predictions_to_scene_result(predictions: dict[str, torch.Tensor]) -> SceneResult:
    images = predictions["images"]
    extrinsic, intrinsic = encoding_to_camera(predictions["pose_enc"], images.shape[-2:])
    if intrinsic is None:
        raise ValueError("encoding_to_camera unexpectedly returned no intrinsics")

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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
