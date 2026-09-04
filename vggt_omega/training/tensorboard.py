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
_DYNAMIC_TRAINING_SUFFIXES = (
    "dynamic_area_prior",
    "dynamic_classification",
    "dynamic_curriculum_stage_index",
    "dynamic_cycle",
    "dynamic_f1",
    "dynamic_iou",
    "dynamic_known_coverage",
    "dynamic_near_coverage",
    "dynamic_objective",
    "dynamic_positive_count",
    "dynamic_precision",
    "dynamic_recall",
    "dynamic_reprojection",
    "dynamic_scene_flow",
    "dynamic_scene_flow_epe",
    "dynamic_spatial",
    "dynamic_static_count",
    "dynamic_static_false_positive_rate",
    "dynamic_teacher_coverage",
    "dynamic_temporal_depth",
    "dynamic_temporal_mask",
    "dynamic_visibility",
    "dynamic_visibility_f1",
    "dynamic_visibility_iou",
    "dynamic_visibility_known_coverage",
    "dynamic_visibility_negative_count",
    "dynamic_visibility_positive_count",
    "dynamic_visibility_precision",
    "dynamic_visibility_recall",
)

ALLOWED_SCALAR_TAGS = frozenset(
    {
        "train/objective",
        "train/camera",
        "train/camera_translation",
        "train/camera_rotation",
        "train/camera_fov",
        "train/depth",
        "train/flow",
        "train/flow_gradient",
        "train/flow_objective",
        "train/ode_steps",
        "train/temporal_enabled",
        "train/residual_gate",
        "train/curriculum_stage_index",
        "train/gpa_objective",
        "train/gpa_physical",
        "train/gpa_photometric",
        "train/gpa_structural",
        "train/gpa_smoothness",
        "train/gpa_valid_fraction",
        "train/correspondence_objective",
        "train/correspondence_covisibility",
        "train/correspondence_pair_count",
        "train/profile_warmup_seconds",
        "train/profile_step_time_seconds",
        "train/profile_samples_per_second",
        "train/profile_data_wait_fraction",
        "train/profile_active_steps",
        "train/multiframe_frame_count",
        "train/multiframe_reference_index",
        "train/multiframe_padding_fraction",
        "train/multiframe_valid_fraction",
        "train/multiframe_dynamic_fraction",
        "train/multiframe_warped_visibility",
        "train/multiframe_preserve_frame_order",
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
        "val/near_depth_mae_m",
        "val/edge_3d_error_proxy",
        "val/near_edge_3d_error_proxy",
        "val/edge_coverage",
        "val/near_edge_coverage",
        "val/multiview_depth_error",
        "val/multiview_relative_error",
        "val/multiview_coverage",
        "val/multiview_pair_count",
        "val/multiview_visible_direction_count",
        "val/near_edge_objective",
        "val/ode_steps",
        "val/residual_gate",
        "val/curriculum_stage_index",
        "val/gpa_objective",
        "val/gpa_physical",
        "val/gpa_photometric",
        "val/gpa_structural",
        "val/gpa_smoothness",
        "val/gpa_valid_fraction",
        "val/correspondence_objective",
        "val/correspondence_covisibility",
        "val/correspondence_pair_count",
        "val/guardrail_triggered",
        "val/guardrail_near_depth_mae_m_excess",
        "val/guardrail_camera_translation_excess",
        "val/guardrail_objective_excess",
        "val/multiframe_frame_count",
        "val/multiframe_reference_index",
        "val/multiframe_padding_fraction",
        "val/multiframe_valid_fraction",
        "val/multiframe_dynamic_fraction",
        "val/multiframe_warped_visibility",
        "val/multiframe_preserve_frame_order",
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
    | {f"{split}/{suffix}" for split in ("train", "val") for suffix in _DYNAMIC_TRAINING_SUFFIXES}
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
