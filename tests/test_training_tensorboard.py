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


def test_nonzero_rank_and_disabled_logger_create_no_events(tmp_path: Path) -> None:
    rank_logger = TensorBoardScalarLogger(log_dir=tmp_path / "rank", enabled=True, rank=1)
    rank_logger.log_scalars({"train/objective": 1.0}, step=0)
    rank_logger.close()

    disabled_logger = TensorBoardScalarLogger(log_dir=tmp_path / "disabled", enabled=False, rank=0)
    disabled_logger.log_scalars({"train/objective": 1.0}, step=0)
    disabled_logger.close()

    assert not list(tmp_path.rglob("events.out.tfevents.*"))
