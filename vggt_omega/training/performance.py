"""Small, explicit runtime contracts and anonymous training timing metrics."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from typing import Any

import torch
from beartype import beartype
from jaxtyping import Bool, Float, jaxtyped


@jaxtyped(typechecker=beartype)
def _typed_training_batch_contract(
    images: Float[torch.Tensor, "batch frames 3 height width"],
    depths: Float[torch.Tensor, "batch frames height width"],
    depth_masks: Bool[torch.Tensor, "batch frames height width"],
    intrinsics: Float[torch.Tensor, "batch frames 3 3"],
    extrinsics: Float[torch.Tensor, "batch frames 3 4"],
) -> None:
    del images, depths, depth_masks, intrinsics, extrinsics


def validate_training_batch_contract(batch: Mapping[str, Any]) -> None:
    """Validate the public RGB-D tensor boundary without logging identifiers."""

    required = ("images", "depths", "depth_masks", "intrinsics", "extrinsics")
    missing = [name for name in required if not isinstance(batch.get(name), torch.Tensor)]
    if missing:
        raise TypeError(f"training batch is missing tensor fields: {missing}")
    _typed_training_batch_contract(
        batch["images"],
        batch["depths"],
        batch["depth_masks"],
        batch["intrinsics"],
        batch["extrinsics"],
    )
    images = batch["images"]
    for optional_name in ("dynamic_masks", "frame_mask"):
        optional = batch.get(optional_name)
        if optional is None:
            continue
        if not isinstance(optional, torch.Tensor) or optional.dtype is not torch.bool:
            raise TypeError(f"{optional_name} must be a boolean tensor")
        expected = (
            images.shape[:2]
            if optional_name == "frame_mask"
            else (images.shape[0], images.shape[1], *images.shape[-2:])
        )
        if tuple(optional.shape) != tuple(expected):
            raise ValueError(f"{optional_name} shape does not match the RGB-D batch")


class StepProfiler:
    """Measure warmup, compute, throughput, and loader wait with scalar-only output."""

    def __init__(
        self,
        *,
        enabled: bool,
        warmup_steps: int,
        active_steps: int,
        synchronize: Callable[[], None],
    ) -> None:
        if warmup_steps < 0 or active_steps < 1:
            raise ValueError("profiler warmup_steps must be non-negative and active_steps positive")
        self.enabled = enabled
        self.warmup_steps = warmup_steps
        self.active_steps = active_steps
        self.synchronize = synchronize
        self._index = 0
        self._ready_at: float | None = None
        self._last_complete_at = time.perf_counter()
        self._data_wait_seconds = 0.0
        self._compute_seconds = 0.0
        self._warmup_seconds = 0.0
        self._samples = 0
        self._recorded_steps = 0
        self._current_wait = 0.0
        self._current_samples = 0

    def batch_ready(self, *, sample_count: int) -> None:
        if not self.enabled:
            return
        if self._ready_at is not None:
            raise RuntimeError("profiler batch_ready called before batch_complete")
        if sample_count < 1:
            raise ValueError("profiled sample_count must be positive")
        self.synchronize()
        now = time.perf_counter()
        self._current_wait = now - self._last_complete_at
        self._current_samples = sample_count
        self._ready_at = now

    def batch_complete(self) -> None:
        if not self.enabled:
            return
        if self._ready_at is None:
            raise RuntimeError("profiler batch_complete called without batch_ready")
        self.synchronize()
        now = time.perf_counter()
        compute = now - self._ready_at
        if self._index < self.warmup_steps:
            self._warmup_seconds += compute
        elif self._recorded_steps < self.active_steps:
            self._data_wait_seconds += self._current_wait
            self._compute_seconds += compute
            self._samples += self._current_samples
            self._recorded_steps += 1
        self._index += 1
        self._ready_at = None
        self._last_complete_at = now

    def metrics(self) -> dict[str, float]:
        if not self.enabled:
            return {}
        total = self._data_wait_seconds + self._compute_seconds
        metrics = {
            "profile_warmup_seconds": self._warmup_seconds,
            "profile_step_time_seconds": self._compute_seconds / max(1, self._recorded_steps),
            "profile_samples_per_second": self._samples / max(self._compute_seconds, torch.finfo(torch.float64).eps),
            "profile_data_wait_fraction": self._data_wait_seconds / max(total, torch.finfo(torch.float64).eps),
            "profile_active_steps": float(self._recorded_steps),
        }
        if not all(math.isfinite(value) and value >= 0 for value in metrics.values()):
            raise ValueError("runtime profiler produced a non-finite or negative scalar")
        return metrics
