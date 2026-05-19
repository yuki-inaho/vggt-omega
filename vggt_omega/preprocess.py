"""Numpy-array centric image preprocessing for VGGT-Omega.

Mirrors the entrypoint shape of the official VGGT inference reference so
callers can move between projects without rewriting glue code. Path-based
loading is retained in :mod:`vggt_omega.utils.load_fn`; this module focuses on
images already decoded as numpy arrays (e.g. video frames) and converts them
to the BCHW float tensor the model expects.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from jaxtyping import Float, UInt8
from PIL import Image

from .utils.load_fn import load_and_preprocess_images

PreprocessMode = Literal["balanced", "max_size"]


def load_images_from_paths(
    image_paths: Iterable[str | Path],
    image_resolution: int = 512,
    mode: PreprocessMode = "balanced",
    patch_size: int = 16,
) -> Float[torch.Tensor, "n_img 3 h w"]:
    """Thin wrapper over the legacy path-based loader for API parity."""
    paths = [str(p) for p in image_paths]
    return load_and_preprocess_images(
        paths,
        mode=mode,
        image_resolution=image_resolution,
        patch_size=patch_size,
    )


def read_images_from_video(
    video_path: str | Path,
    sample_fps: float = 1.0,
) -> list[UInt8[np.ndarray, "h w 3"]]:
    """Decode a video to RGB numpy frames sampled at ``sample_fps``.

    Returns BGR-to-RGB converted frames so the rest of the preprocessing
    pipeline can treat them like images read from disk via PIL.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video at {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    sample_fps = max(float(sample_fps), 0.1)
    frame_interval = max(int(round(fps / sample_fps)), 1)

    frames: list[np.ndarray] = []
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % frame_interval == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_idx += 1
    finally:
        cap.release()

    if not frames:
        raise ValueError(f"No frames decoded from {video_path}")
    return frames


def preprocess_images(
    images: list[UInt8[np.ndarray, "h w 3"]],
    image_resolution: int = 512,
    mode: PreprocessMode = "balanced",
    patch_size: int = 16,
    tmp_dir: str | Path | None = None,
) -> Float[torch.Tensor, "n_img 3 h w"]:
    """Preprocess a list of HxWx3 uint8/float RGB arrays into the model tensor.

    Routes through the path-based loader so the preprocessing is bit-identical
    to disk-loaded images. Frames are dumped to a temporary directory only
    when one is provided; otherwise an in-memory PIL pipeline is used.
    """
    if not images:
        raise ValueError("preprocess_images requires at least one image")

    if tmp_dir is not None:
        return _preprocess_via_disk(images, image_resolution, mode, patch_size, Path(tmp_dir))
    return _preprocess_in_memory(images, image_resolution, mode, patch_size)


def _preprocess_in_memory(
    images: list[np.ndarray],
    image_resolution: int,
    mode: PreprocessMode,
    patch_size: int,
) -> torch.Tensor:
    """Run the same resize logic as :mod:`load_fn` without touching disk."""
    from torchvision import transforms as TF

    from .utils.load_fn import (
        _balanced_target_shape,
        _crop_to_supported_aspect_ratio,
        _max_size_target_shape,
        _pad_images_to_common_size,
    )

    if image_resolution <= 0:
        raise ValueError("image_resolution must be positive")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if image_resolution % patch_size != 0:
        raise ValueError("image_resolution must be divisible by patch_size")
    if mode not in ("balanced", "max_size"):
        raise ValueError("Mode must be either 'balanced' or 'max_size'")

    to_tensor = TF.ToTensor()
    tensors: list[torch.Tensor] = []
    shapes: set[tuple[int, int]] = set()
    for frame in images:
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(f"Expected HxWx3 RGB array, got shape {frame.shape}")
        pil = Image.fromarray(frame.astype(np.uint8) if frame.dtype != np.uint8 else frame, mode="RGB")
        pil = _crop_to_supported_aspect_ratio(pil)
        width, height = pil.size
        aspect_ratio = height / max(width, 1)
        target_shape = (
            _balanced_target_shape(aspect_ratio, image_resolution, patch_size)
            if mode == "balanced"
            else _max_size_target_shape(aspect_ratio, image_resolution, patch_size)
        )
        target_h, target_w = target_shape
        pil = pil.resize((target_w, target_h), Image.Resampling.BICUBIC)
        tensor = to_tensor(pil)
        shapes.add((tensor.shape[1], tensor.shape[2]))
        tensors.append(tensor)

    if len(shapes) > 1:
        tensors = _pad_images_to_common_size(tensors, shapes)
    return torch.stack(tensors)


def _preprocess_via_disk(
    images: list[np.ndarray],
    image_resolution: int,
    mode: PreprocessMode,
    patch_size: int,
    tmp_dir: Path,
) -> torch.Tensor:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for idx, frame in enumerate(images):
        out = tmp_dir / f"frame_{idx:06d}.png"
        Image.fromarray(frame.astype(np.uint8) if frame.dtype != np.uint8 else frame).save(out)
        paths.append(str(out))
    return load_and_preprocess_images(paths, mode=mode, image_resolution=image_resolution, patch_size=patch_size)
