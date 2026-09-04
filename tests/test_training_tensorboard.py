from pathlib import Path

import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from vggt_omega.training.tensorboard import TensorBoardScalarLogger


def _event_files(root: Path) -> list[Path]:
    return sorted(root.glob("events.out.tfevents.*"))


def test_scalar_logger_writes_readable_required_tags(tmp_path: Path) -> None:
    log_dir = tmp_path / "tensorboard"
    logger = TensorBoardScalarLogger(log_dir=log_dir, enabled=True, rank=0)
    logger.log_scalars(
        {
            "train/objective": 1.25,
            "train/camera": 0.5,
            "train/camera_translation": 0.2,
            "train/camera_rotation": 0.1,
            "train/camera_fov": 0.4,
            "train/depth": 0.75,
            "train/flow": 1.1,
            "train/flow_gradient": 0.2,
            "train/flow_objective": 1.14,
            "train/ode_steps": 4.0,
            "train/temporal_enabled": 1.0,
            "train/residual_gate": 0.0,
            "train/curriculum_stage_index": 4.0,
            "train/gpa_objective": 0.2,
            "train/gpa_physical": 0.19,
            "train/gpa_photometric": 0.15,
            "train/gpa_structural": 0.4,
            "train/gpa_smoothness": 0.01,
            "train/gpa_valid_fraction": 0.7,
            "train/correspondence_objective": 0.3,
            "train/correspondence_covisibility": 0.6,
            "train/correspondence_pair_count": 12.0,
            "train/profile_warmup_seconds": 1.5,
            "train/profile_step_time_seconds": 0.25,
            "train/profile_samples_per_second": 8.0,
            "train/profile_data_wait_fraction": 0.1,
            "train/profile_active_steps": 4.0,
            "train/multiframe_frame_count": 3.0,
            "train/multiframe_reference_index": 1.0,
            "train/multiframe_padding_fraction": 0.25,
            "train/multiframe_valid_fraction": 0.75,
            "train/multiframe_dynamic_fraction": 0.1,
            "train/multiframe_warped_visibility": 0.5,
            "train/multiframe_preserve_frame_order": 1.0,
            "train/grad_norm": 2.0,
            "optimizer/group_0_lr": 1e-4,
            "optimizer/group_1_lr": 1e-5,
            "optimizer/beta1": 0.4,
            "system/max_cuda_memory_gib": 7.5,
        },
        step=3,
    )
    logger.log_scalars(
        {
            "val/objective": 1.0,
            "val/camera": 0.4,
            "val/camera_translation": 0.2,
            "val/camera_rotation": 0.1,
            "val/camera_fov": 0.2,
            "val/depth": 0.6,
            "val/near_depth_mae_m": 0.05,
            "val/edge_3d_error_proxy": 0.08,
            "val/near_edge_3d_error_proxy": 0.06,
            "val/edge_coverage": 0.2,
            "val/near_edge_coverage": 0.3,
            "val/multiview_depth_error": 0.04,
            "val/multiview_relative_error": 0.03,
            "val/multiview_coverage": 0.7,
            "val/multiview_pair_count": 3.0,
            "val/multiview_visible_direction_count": 6.0,
            "val/near_edge_objective": 0.1,
            "val/ode_steps": 4.0,
            "val/residual_gate": 0.1,
            "val/curriculum_stage_index": 4.0,
            "val/gpa_objective": 0.2,
            "val/gpa_physical": 0.19,
            "val/gpa_photometric": 0.15,
            "val/gpa_structural": 0.4,
            "val/gpa_smoothness": 0.01,
            "val/gpa_valid_fraction": 0.7,
            "val/correspondence_objective": 0.3,
            "val/correspondence_covisibility": 0.6,
            "val/correspondence_pair_count": 12.0,
            "val/pairwise_pose": 0.3,
            "val/pairwise_rotation": 0.1,
            "val/pairwise_translation_direction": 0.15,
            "val/pairwise_translation_magnitude": 0.05,
            "val/pairwise_valid_direction_fraction": 1.0,
            "val/pairwise_rotation_degrees": 5.73,
            "val/pairwise_translation_direction_degrees": 8.59,
            "val/rpa_5": 0.25,
            "val/rpa_15": 0.75,
            "val/rpa_30": 0.9,
        },
        step=4,
    )
    logger.close()

    assert len(_event_files(log_dir)) == 1
    events = EventAccumulator(str(log_dir))
    events.Reload()
    tags = set(events.Tags()["scalars"])
    assert tags == {
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
        "system/max_cuda_memory_gib",
    }
    assert events.Scalars("train/objective")[0].step == 3
    assert events.Scalars("val/objective")[0].value == pytest.approx(1.0)


def test_logger_rejects_unknown_or_nonfinite_scalars(tmp_path: Path) -> None:
    logger = TensorBoardScalarLogger(log_dir=tmp_path, enabled=True, rank=0)

    with pytest.raises(ValueError, match="Unsupported TensorBoard scalar tag"):
        logger.log_scalars({"private/source_path": 1.0}, step=0)
    with pytest.raises(ValueError, match="finite"):
        logger.log_scalars({"train/objective": float("nan")}, step=0)

    logger.close()


def test_scalar_logger_writes_all_dynamic_training_tags(tmp_path: Path) -> None:
    dynamic_tags = {
        "train/dynamic_area_prior",
        "train/dynamic_classification",
        "train/dynamic_curriculum_stage_index",
        "train/dynamic_cycle",
        "train/dynamic_known_coverage",
        "train/dynamic_near_coverage",
        "train/dynamic_objective",
        "train/dynamic_positive_count",
        "train/dynamic_precision",
        "train/dynamic_recall",
        "train/dynamic_reprojection",
        "train/dynamic_scene_flow",
        "train/dynamic_spatial",
        "train/dynamic_static_count",
        "train/dynamic_teacher_coverage",
        "train/dynamic_temporal_depth",
        "train/dynamic_temporal_mask",
        "train/dynamic_visibility",
        "train/dynamic_visibility_known_coverage",
        "train/dynamic_visibility_negative_count",
        "train/dynamic_visibility_positive_count",
        "train/dynamic_visibility_precision",
    }
    logger = TensorBoardScalarLogger(log_dir=tmp_path, enabled=True, rank=0)
    logger.log_scalars(dict.fromkeys(dynamic_tags, 0.25), step=1)
    logger.close()

    events = EventAccumulator(str(tmp_path))
    events.Reload()
    assert set(events.Tags()["scalars"]) == dynamic_tags


def test_nonzero_rank_and_disabled_logger_create_no_events(tmp_path: Path) -> None:
    rank_logger = TensorBoardScalarLogger(log_dir=tmp_path / "rank", enabled=True, rank=1)
    rank_logger.log_scalars({"train/objective": 1.0}, step=0)
    rank_logger.close()

    disabled_logger = TensorBoardScalarLogger(log_dir=tmp_path / "disabled", enabled=False, rank=0)
    disabled_logger.log_scalars({"train/objective": 1.0}, step=0)
    disabled_logger.close()

    assert not list(tmp_path.rglob("events.out.tfevents.*"))
