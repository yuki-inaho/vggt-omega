"""Privacy-minimal scalar-only TensorBoard logging."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

_DEPTH_EVALUATION_PREFIXES = (
    "depth_all",
    "depth_lt_0p4m",
    "depth_lt_0p8m",
    "depth_lt_1p2m",
)
_DEPTH_EVALUATION_SUFFIXES = (
    "abs_rel",
    "coverage",
    "mae_m",
    "normalized_l1",
    "rmse_m",
    "valid_pixels",
)

ALLOWED_SCALAR_TAGS = frozenset(
    {
        "train/objective",
        "train/camera",
        "train/camera_translation",
        "train/camera_rotation",
        "train/camera_fov",
        "train/depth",
        "train/sample_overlap",
        "train/overlap_target",
        "train/overlap_fallback",
        "train/pairwise_pose",
        "train/pairwise_rotation",
        "train/pairwise_translation_direction",
        "train/pairwise_translation_magnitude",
        "train/pairwise_valid_direction_fraction",
        "train/pairwise_rotation_degrees",
        "train/pairwise_translation_direction_degrees",
        "train/rpa_5",
        "train/rpa_15",
        "train/rpa_30",
        "train/photometric",
        "train/photometric_visibility",
        "train/grad_norm",
        "optimizer/group_0_lr",
        "optimizer/group_1_lr",
        "optimizer/beta1",
        "val/objective",
        "val/camera",
        "val/camera_translation",
        "val/camera_rotation",
        "val/camera_fov",
        "val/depth",
        "val/pairwise_pose",
        "val/pairwise_rotation",
        "val/pairwise_translation_direction",
        "val/pairwise_translation_magnitude",
        "val/pairwise_valid_direction_fraction",
        "val/pairwise_rotation_degrees",
        "val/pairwise_translation_direction_degrees",
        "val/rpa_5",
        "val/rpa_15",
        "val/rpa_30",
        "val/photometric",
        "val/photometric_visibility",
        "system/max_cuda_memory_gib",
    }
    | {f"val/{prefix}_{suffix}" for prefix in _DEPTH_EVALUATION_PREFIXES for suffix in _DEPTH_EVALUATION_SUFFIXES}
)


class TensorBoardScalarLogger:
    """A deliberately small TensorBoard API that cannot log images or text."""

    def __init__(self, log_dir: str | Path, *, enabled: bool, rank: int = 0) -> None:
        self._active = bool(enabled) and int(rank) == 0
        self._writer = SummaryWriter(log_dir=str(log_dir)) if self._active else None

    @property
    def active(self) -> bool:
        return self._active

    def log_scalars(self, scalars: Mapping[str, float], *, step: int) -> None:
        if not self._active:
            return
        if step < 0:
            raise ValueError(f"TensorBoard step must be non-negative, got {step}")
        assert self._writer is not None
        for tag, raw_value in scalars.items():
            if tag not in ALLOWED_SCALAR_TAGS:
                raise ValueError(f"Unsupported TensorBoard scalar tag: {tag!r}")
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError(f"TensorBoard scalar {tag!r} must be finite, got {value}")
            self._writer.add_scalar(tag, value, global_step=step)

    def flush(self) -> None:
        if self._writer is not None:
            self._writer.flush()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()

    def __enter__(self) -> TensorBoardScalarLogger:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
