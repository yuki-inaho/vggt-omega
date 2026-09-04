from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from vggt_omega.training.config import validate_training_config

CONFIG_DIR = Path(__file__).parents[1] / "configs" / "training"


def _compose(*overrides: str, return_hydra_config: bool = False) -> DictConfig:
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        return compose(
            config_name="config",
            overrides=list(overrides),
            return_hydra_config=return_hydra_config,
        )


def test_default_training_config_composes_and_validates() -> None:
    cfg = _compose()

    validate_training_config(cfg)

    assert cfg.optimizer.name == "amuse"
    assert cfg.loss.name == "standard"
    assert cfg.trainer.name == "smoke"
    assert cfg.checkpoint.k == 3
    assert cfg.checkpoint.monitor == "val/objective"
    assert cfg.logging.name == "tensorboard"


def test_camera_motion_curriculum_composes_and_validates() -> None:
    cfg = _compose("loss=camera_motion_curriculum", "trainer=finetune", "trainer.epochs=9")

    validate_training_config(cfg)

    assert [stage.start_epoch for stage in cfg.loss.curriculum] == [0, 3, 6]
    assert cfg.loss.curriculum[0].translation_weight == pytest.approx(4.0)
    assert cfg.loss.curriculum[0].rotation_weight == pytest.approx(2.0)
    assert cfg.loss.validation.translation_weight == pytest.approx(1.0)


def test_translation_focus_early_stopping_config_validates() -> None:
    cfg = _compose(
        "loss=camera_translation_focus",
        "trainer=finetune",
        "trainer.epochs=5",
        "checkpoint.monitor=val/camera_translation",
        "trainer.early_stopping.enabled=true",
        "trainer.early_stopping.monitor=val/camera_translation",
    )

    validate_training_config(cfg)

    assert cfg.loss.training.translation_weight == pytest.approx(4.0)
    assert cfg.trainer.early_stopping.patience == 2


def test_near_depth_focus_config_validates() -> None:
    cfg = _compose("loss=near_depth_focus", "trainer=finetune", "trainer.epochs=5")

    validate_training_config(cfg)

    assert cfg.loss.training.max_metric_depth_m == pytest.approx(1.2)
    assert cfg.loss.validation.max_metric_depth_m == pytest.approx(1.2)
    assert cfg.loss.training.depth_weight == pytest.approx(4.0)


def test_early_stopping_monitor_must_match_checkpoint_monitor() -> None:
    cfg = _compose("trainer=finetune", "trainer.early_stopping.enabled=true")
    cfg.trainer.early_stopping.monitor = "val/camera_translation"

    with pytest.raises(ValueError, match="same monitor"):
        validate_training_config(cfg)


def test_curriculum_resume_keeps_initial_head_provenance() -> None:
    cfg = _compose("loss=camera_motion_curriculum", "trainer=finetune", "trainer.epochs=9")
    cfg.model.initial_head_checkpoint = "previous/checkpoints/last.pt"
    cfg.trainer.resume_from = "current/checkpoints/last.pt"

    validate_training_config(cfg)


@pytest.mark.parametrize(
    "path",
    ["/private/last.pt", "../outside/last.pt", ""],
)
def test_initial_head_checkpoint_must_be_private_safe_relative_path(path: str) -> None:
    cfg = _compose()
    cfg.model.initial_head_checkpoint = path

    with pytest.raises(ValueError, match="initial_head_checkpoint"):
        validate_training_config(cfg)


def test_adamw_finetune_override_composes() -> None:
    cfg = _compose("optimizer=adamw", "trainer=finetune", "checkpoint.k=2")

    validate_training_config(cfg)

    assert cfg.optimizer.name == "adamw"
    assert cfg.trainer.name == "finetune"
    assert cfg.checkpoint.k == 2


def test_tensorboard_can_only_be_disabled_with_an_explicit_profile() -> None:
    cfg = _compose("logging=tensorboard_disabled")

    validate_training_config(cfg)

    assert cfg.logging.name == "tensorboard_disabled"
    assert cfg.logging.enabled is False


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (("optimizer=amuse", "trainer.strategy=fsdp"), "FSDP"),
        (("optimizer=amuse", "optimizer.scheduler=cosine"), "scheduler"),
        (("checkpoint.k=0",), "checkpoint.k"),
        (("model.image_height=385",), "patch size"),
    ],
)
def test_invalid_training_config_is_rejected(overrides: tuple[str, ...], message: str) -> None:
    cfg = _compose(*overrides)

    with pytest.raises(ValueError, match=message):
        validate_training_config(cfg)


def test_hydra_keeps_the_repository_working_directory() -> None:
    cfg = _compose(return_hydra_config=True)

    assert cfg.hydra.job.chdir is False
    assert cfg.hydra.output_subdir is None
    assert "outputs/training/" in cfg.hydra.run.dir
