from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import vggt_omega.training.runner as runner_module
from vggt_omega.training.model_factory import PreparedTrainingModel
from vggt_omega.training.runner import (
    _initialize_head_from_checkpoint,
    _training_loss_options,
    _validate_resume_config,
    run_training,
    train_one_epoch,
    validate_one_epoch,
)


class _TinyCameraDepthModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pose = torch.nn.Parameter(torch.zeros(9))
        self.log_depth = torch.nn.Parameter(torch.tensor(0.25))

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, frames, _, height, width = images.shape
        pose = self.pose.view(1, 1, 9).expand(batch, frames, -1)
        depth = self.log_depth.exp().expand(batch, frames, height, width, 1)
        return {"pose_enc": pose, "depth": depth}


class _RecordingLogger:
    def __init__(self) -> None:
        self.records: list[tuple[int, dict[str, float]]] = []

    def log_scalars(self, scalars: Mapping[str, float], *, step: int) -> None:
        self.records.append((step, dict(scalars)))


def _batch() -> dict[str, torch.Tensor]:
    batch, frames, height, width = 1, 2, 16, 16
    images = torch.full((batch, frames, 3, height, width), 0.5)
    depths = torch.ones((batch, frames, height, width))
    masks = torch.ones_like(depths, dtype=torch.bool)
    extrinsics = torch.eye(4)[:3].view(1, 1, 3, 4).repeat(batch, frames, 1, 1)
    extrinsics[:, 1, 0, 3] = -0.1
    intrinsics = (
        torch.tensor([[8.0, 0.0, width / 2], [0.0, 8.0, height / 2], [0.0, 0.0, 1.0]])
        .view(1, 1, 3, 3)
        .repeat(batch, frames, 1, 1)
    )
    return {
        "images": images,
        "depths": depths,
        "depth_masks": masks,
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
        "frame_ids": torch.arange(frames).view(1, frames),
    }


def test_train_one_epoch_updates_model_logs_scalars_and_clips_gradients() -> None:
    model = _TinyCameraDepthModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    logger = _RecordingLogger()
    before = model.pose.detach().clone()

    result = train_one_epoch(
        model=model,
        batches=[_batch(), _batch()],
        optimizer=optimizer,
        device=torch.device("cpu"),
        gradient_clip_norm=1.0,
        gradient_accumulation_steps=1,
        min_valid_depth_pixels=1,
        global_step=0,
        logger=logger,
        log_every_steps=1,
    )

    assert result.global_step == 2
    assert result.optimizer_steps == 2
    assert math.isfinite(result.metrics["objective"])
    assert not torch.equal(model.pose, before)
    assert [step for step, _ in logger.records] == [1, 2]
    assert all("train/objective" in scalars for _, scalars in logger.records)
    assert all("train/grad_norm" in scalars for _, scalars in logger.records)


def test_train_one_epoch_honors_accumulation_and_step_limit() -> None:
    model = _TinyCameraDepthModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    result = train_one_epoch(
        model=model,
        batches=[_batch(), _batch(), _batch(), _batch()],
        optimizer=optimizer,
        device=torch.device("cpu"),
        gradient_clip_norm=1.0,
        gradient_accumulation_steps=2,
        min_valid_depth_pixels=1,
        global_step=4,
        logger=None,
        log_every_steps=1,
        max_optimizer_steps=1,
    )

    assert result.global_step == 5
    assert result.optimizer_steps == 1


def test_validate_one_epoch_is_finite_and_does_not_create_gradients() -> None:
    model = _TinyCameraDepthModel()

    metrics = validate_one_epoch(
        model=model,
        batches=[_batch(), _batch()],
        device=torch.device("cpu"),
        min_valid_depth_pixels=1,
        max_batches=1,
    )

    assert set(metrics) >= {"objective", "camera", "depth"}
    assert all(math.isfinite(value) for value in metrics.values())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_validate_one_epoch_reports_metric_depth_threshold_metrics() -> None:
    model = _TinyCameraDepthModel()
    batch = _batch()
    batch["normalization_scale_m"] = torch.tensor(1.0)
    batch["depths"][..., :8, :] = 0.5
    batch["depths"][..., 8:, :] = 1.25

    metrics = validate_one_epoch(
        model=model,
        batches=[batch],
        device=torch.device("cpu"),
        min_valid_depth_pixels=1,
        depth_thresholds_m=(1.2,),
    )

    expected_prediction = math.exp(0.25)
    assert metrics["depth_lt_1p2m_valid_pixels"] == 256
    assert metrics["depth_lt_1p2m_coverage"] == pytest.approx(0.5)
    assert metrics["depth_lt_1p2m_mae_m"] == pytest.approx(expected_prediction - 0.5)
    assert metrics["depth_lt_1p2m_rmse_m"] == pytest.approx(expected_prediction - 0.5)
    assert metrics["depth_lt_1p2m_abs_rel"] == pytest.approx((expected_prediction - 0.5) / 0.5)
    assert metrics["depth_lt_1p2m_normalized_l1"] == pytest.approx(expected_prediction - 0.5)
    assert metrics["depth_all_valid_pixels"] == 512
    assert metrics["depth_all_coverage"] == 1.0


def test_validate_one_epoch_rejects_invalid_depth_thresholds() -> None:
    with pytest.raises(ValueError, match="depth thresholds"):
        validate_one_epoch(
            model=_TinyCameraDepthModel(),
            batches=[_batch()],
            device=torch.device("cpu"),
            min_valid_depth_pixels=1,
            depth_thresholds_m=(1.2, 0.8),
        )


def test_train_one_epoch_rejects_nonfinite_loss_without_stepping() -> None:
    model = _TinyCameraDepthModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    with torch.no_grad():
        model.pose[0] = torch.nan
    before_steps = [state.get("step") for state in optimizer.state.values()]

    with pytest.raises(ValueError, match=r"NaN|non-finite"):
        train_one_epoch(
            model=model,
            batches=[_batch()],
            optimizer=optimizer,
            device=torch.device("cpu"),
            gradient_clip_norm=1.0,
            gradient_accumulation_steps=1,
            min_valid_depth_pixels=1,
            global_step=0,
            logger=None,
            log_every_steps=1,
        )

    assert [state.get("step") for state in optimizer.state.values()] == before_steps


class _TinyDataset(torch.utils.data.Dataset[dict[str, torch.Tensor]]):
    def __init__(self) -> None:
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        batch = _batch()
        return {key: value[0] for key, value in batch.items()}


class _InterruptingDataset(_TinyDataset):
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if self.epoch == 1:
            raise RuntimeError("synthetic interruption")
        return super().__getitem__(index)


def _training_config():
    config_dir = Path(__file__).parents[1] / "configs" / "training"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name="config")
    cfg.trainer.device = "cpu"
    cfg.trainer.epochs = 2
    cfg.trainer.max_train_steps = None
    cfg.trainer.max_val_batches = 1
    cfg.model.precision = "fp32"
    cfg.data.num_workers = 0
    cfg.data.pin_memory = False
    cfg.data.min_frames = 2
    cfg.data.max_frames = 2
    cfg.checkpoint.k = 1
    return cfg


def test_camera_motion_curriculum_selects_weights_by_epoch() -> None:
    config_dir = Path(__file__).parents[1] / "configs" / "training"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(
            config_name="config",
            overrides=["loss=camera_motion_curriculum", "trainer=finetune", "trainer.epochs=9"],
        )

    assert _training_loss_options(cfg, 0)["translation_weight"] == pytest.approx(4.0)
    assert _training_loss_options(cfg, 3)["translation_weight"] == pytest.approx(3.0)
    assert _training_loss_options(cfg, 8)["translation_weight"] == pytest.approx(2.0)


def test_initial_head_checkpoint_loads_x_state_without_optimizer(tmp_path: Path) -> None:
    model = _TinyCameraDepthModel()
    names = tuple(name for name, _ in model.named_parameters())
    base_path = tmp_path / "base.pt"
    base_path.write_bytes(b"base")
    base = {
        "filename": base_path.name,
        "size_bytes": base_path.stat().st_size,
        "sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
    }
    initial = tmp_path / "last.pt"
    torch.save(
        {
            "format_version": 1,
            "kind": "resume",
            "parameter_state": "x",
            "epoch": 50,
            "global_step": 123,
            "metadata": {"base_checkpoint": base},
            "model_state": {"pose": torch.ones(9), "log_depth": torch.tensor(2.0)},
            "optimizer_state": {"ignored": True},
        },
        initial,
    )

    metadata = _initialize_head_from_checkpoint(
        initial,
        model=model,
        trainable_parameter_names=names,
        expected_base_checkpoint=base,
    )

    assert torch.equal(model.pose, torch.ones(9))
    assert model.log_depth.item() == pytest.approx(2.0)
    assert metadata["kind"] == "resume"
    assert metadata["epoch"] == 50


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("parameter_state", "evaluation weights x"),
        ("base", "base metadata"),
        ("model_state", "exactly match trainable"),
    ],
)
def test_initial_head_checkpoint_rejects_incompatible_payload(tmp_path: Path, mutation: str, message: str) -> None:
    model = _TinyCameraDepthModel()
    names = tuple(name for name, _ in model.named_parameters())
    base = {"filename": "base.pt", "size_bytes": 4, "sha256": "a" * 64}
    payload: dict[str, object] = {
        "format_version": 1,
        "kind": "resume",
        "parameter_state": "x",
        "epoch": 49,
        "global_step": 36200,
        "metadata": {"base_checkpoint": base},
        "model_state": {"pose": torch.ones(9), "log_depth": torch.tensor(2.0)},
    }
    if mutation == "parameter_state":
        payload["parameter_state"] = "y"
    elif mutation == "base":
        payload["metadata"] = {"base_checkpoint": {**base, "sha256": "b" * 64}}
    else:
        payload["model_state"] = {"pose": torch.ones(9)}
    checkpoint = tmp_path / "initial.pt"
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match=message):
        _initialize_head_from_checkpoint(
            checkpoint,
            model=model,
            trainable_parameter_names=names,
            expected_base_checkpoint=base,
        )


def test_resume_config_migrates_legacy_missing_standard_loss_fields() -> None:
    cfg = _training_config()
    current = json.loads(json.dumps(OmegaConf.to_container(cfg, resolve=True)))
    legacy = json.loads(json.dumps(current))
    del legacy["loss"]
    del legacy["model"]["initial_head_checkpoint"]

    _validate_resume_config(legacy, current)


def test_run_training_connects_epochs_tensorboard_topk_and_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _training_config()
    base_checkpoint = tmp_path / "base.pt"
    base_checkpoint.write_bytes(b"base-checkpoint")
    cfg.model.pretrained_checkpoint = str(base_checkpoint)
    cfg.data.root = str(tmp_path / "staging")
    model = _TinyCameraDepthModel()

    monkeypatch.setattr(runner_module, "ColmapRgbdDataset", lambda *args, **kwargs: _TinyDataset())
    monkeypatch.setattr(
        runner_module,
        "build_training_model",
        lambda *args, **kwargs: PreparedTrainingModel(
            model=model,
            trainable_parameter_names=tuple(name for name, _ in model.named_parameters()),
        ),
    )

    def build_optimizer(*args, **kwargs):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        return SimpleNamespace(optimizer=optimizer, scheduler=None, group_fingerprint="d" * 64)

    monkeypatch.setattr(runner_module, "_build_optimizer_from_config", build_optimizer)

    summary = run_training(cfg, output_dir=tmp_path / "run", original_cwd=tmp_path)

    assert summary["status"] == "complete"
    assert summary["epochs_completed"] == 2
    assert summary["global_step"] == 4
    checkpoint_dir = tmp_path / "run" / "checkpoints"
    assert len(list(checkpoint_dir.glob("best_epoch_*.pt"))) == 1
    assert (checkpoint_dir / "last.pt").is_file()
    assert (checkpoint_dir / "leaderboard.json").is_file()
    assert (tmp_path / "run" / "run_summary.json").is_file()
    assert (tmp_path / "run" / "resolved_config.json").is_file()
    progress = json.loads((tmp_path / "run" / "progress.json").read_text())
    assert progress["status"] == "complete"
    assert progress["epochs_completed"] == 2
    assert progress["global_step"] == 4
    events = EventAccumulator(str(tmp_path / "run" / "tensorboard"))
    events.Reload()
    tags = set(events.Tags()["scalars"])
    assert {"train/objective", "val/objective", "val/camera", "val/depth"} <= tags


def test_run_training_early_stops_after_patience(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _training_config()
    cfg.trainer.epochs = 5
    cfg.trainer.early_stopping.enabled = True
    cfg.trainer.early_stopping.monitor = "val/camera_translation"
    cfg.trainer.early_stopping.patience = 2
    cfg.checkpoint.monitor = "val/camera_translation"
    base_checkpoint = tmp_path / "base.pt"
    base_checkpoint.write_bytes(b"base-checkpoint")
    cfg.model.pretrained_checkpoint = str(base_checkpoint)
    cfg.data.root = str(tmp_path / "staging")
    model = _TinyCameraDepthModel()
    validation_values = iter((1.0, 1.1, 1.2, 0.9, 0.8))

    monkeypatch.setattr(runner_module, "ColmapRgbdDataset", lambda *args, **kwargs: _TinyDataset())
    monkeypatch.setattr(
        runner_module,
        "build_training_model",
        lambda *args, **kwargs: PreparedTrainingModel(
            model=model,
            trainable_parameter_names=tuple(name for name, _ in model.named_parameters()),
        ),
    )

    def build_optimizer(*args, **kwargs):
        return SimpleNamespace(
            optimizer=torch.optim.AdamW(model.parameters(), lr=1e-2),
            scheduler=None,
            group_fingerprint="d" * 64,
        )

    def validation(**kwargs):
        translation = next(validation_values)
        return {
            "camera": translation,
            "camera_translation": translation,
            "camera_rotation": 0.0,
            "camera_fov": 0.0,
            "depth": 0.5,
            "objective": 5.0 * translation + 0.5,
        }

    monkeypatch.setattr(runner_module, "_build_optimizer_from_config", build_optimizer)
    monkeypatch.setattr(runner_module, "validate_one_epoch", validation)

    summary = run_training(cfg, output_dir=tmp_path / "run", original_cwd=tmp_path)

    assert summary["status"] == "complete"
    assert summary["stopped_early"] is True
    assert summary["epochs_completed"] == 3
    assert summary["global_step"] == 6
    assert summary["early_stopping"]["best"] == pytest.approx(1.0)
    assert summary["early_stopping"]["bad_epochs"] == 2
    assert summary["best"][0]["metric"] == pytest.approx(1.0)

    expected_train = summary["train"]
    expected_validation = summary["validation"]
    (tmp_path / "run" / "run_summary.json").unlink()
    cfg.trainer.resume_from = str(tmp_path / "run" / "checkpoints" / "last.pt")
    resumed = run_training(cfg, output_dir=tmp_path / "run", original_cwd=tmp_path)

    assert resumed["stopped_early"] is True
    assert resumed["epochs_completed"] == 3
    assert resumed["global_step"] == 6
    assert resumed["train"] == expected_train
    assert resumed["validation"] == expected_validation


def test_early_stopping_patience_is_preserved_across_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _training_config()
    cfg.trainer.epochs = 5
    cfg.trainer.early_stopping.enabled = True
    cfg.trainer.early_stopping.monitor = "val/camera_translation"
    cfg.trainer.early_stopping.patience = 2
    cfg.checkpoint.monitor = "val/camera_translation"
    base_checkpoint = tmp_path / "base.pt"
    base_checkpoint.write_bytes(b"base-checkpoint")
    cfg.model.pretrained_checkpoint = str(base_checkpoint)
    cfg.data.root = str(tmp_path / "staging")
    interrupted = True
    validation_values = iter((1.0, 1.1, 1.2, 0.9))

    class InterruptAtThirdEpoch(_TinyDataset):
        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            if self.epoch == 2:
                raise RuntimeError("synthetic interruption")
            return super().__getitem__(index)

    def dataset_factory(*args, **kwargs):
        return InterruptAtThirdEpoch() if interrupted else _TinyDataset()

    def model_factory(*args, **kwargs):
        model = _TinyCameraDepthModel()
        return PreparedTrainingModel(
            model=model,
            trainable_parameter_names=tuple(name for name, _ in model.named_parameters()),
        )

    def optimizer_factory(config, model, **kwargs):
        return SimpleNamespace(
            optimizer=torch.optim.AdamW(model.parameters(), lr=1e-2),
            scheduler=None,
            group_fingerprint="d" * 64,
        )

    def validation(**kwargs):
        translation = next(validation_values)
        return {
            "camera": translation,
            "camera_translation": translation,
            "camera_rotation": 0.0,
            "camera_fov": 0.0,
            "depth": 0.5,
            "objective": 5.0 * translation + 0.5,
        }

    monkeypatch.setattr(runner_module, "ColmapRgbdDataset", dataset_factory)
    monkeypatch.setattr(runner_module, "build_training_model", model_factory)
    monkeypatch.setattr(runner_module, "_build_optimizer_from_config", optimizer_factory)
    monkeypatch.setattr(runner_module, "validate_one_epoch", validation)
    output = tmp_path / "run"

    with pytest.raises(RuntimeError, match="synthetic interruption"):
        run_training(cfg, output_dir=output, original_cwd=tmp_path)
    last_checkpoint = output / "checkpoints" / "last.pt"
    payload = torch.load(last_checkpoint, map_location="cpu", weights_only=True)
    assert payload["training_state"]["early_stopping"]["best"] == pytest.approx(1.0)
    assert payload["training_state"]["early_stopping"]["bad_epochs"] == 1

    interrupted = False
    cfg.trainer.resume_from = str(last_checkpoint)
    summary = run_training(cfg, output_dir=output, original_cwd=tmp_path)

    assert summary["stopped_early"] is True
    assert summary["epochs_completed"] == 3
    assert summary["early_stopping"]["best"] == pytest.approx(1.0)
    assert summary["early_stopping"]["bad_epochs"] == 2


def test_completed_run_noop_resume_preserves_latest_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _training_config()
    base_checkpoint = tmp_path / "base.pt"
    base_checkpoint.write_bytes(b"base-checkpoint")
    cfg.model.pretrained_checkpoint = str(base_checkpoint)
    cfg.data.root = str(tmp_path / "staging")

    def model_factory(*args, **kwargs):
        model = _TinyCameraDepthModel()
        return PreparedTrainingModel(
            model=model,
            trainable_parameter_names=tuple(name for name, _ in model.named_parameters()),
        )

    def optimizer_factory(config, model, **kwargs):
        return SimpleNamespace(
            optimizer=torch.optim.AdamW(model.parameters(), lr=1e-2),
            scheduler=None,
            group_fingerprint="d" * 64,
        )

    monkeypatch.setattr(runner_module, "ColmapRgbdDataset", lambda *args, **kwargs: _TinyDataset())
    monkeypatch.setattr(runner_module, "build_training_model", model_factory)
    monkeypatch.setattr(runner_module, "_build_optimizer_from_config", optimizer_factory)
    output = tmp_path / "run"

    initial = run_training(cfg, output_dir=output, original_cwd=tmp_path)
    cfg.trainer.resume_from = str(output / "checkpoints" / "last.pt")
    resumed = run_training(cfg, output_dir=output, original_cwd=tmp_path)

    assert resumed["global_step"] == initial["global_step"]
    assert resumed["epochs_completed"] == initial["epochs_completed"]
    assert resumed["train"] == initial["train"]
    assert resumed["validation"] == initial["validation"]
    persisted_config = json.loads((output / "resolved_config.json").read_text())
    assert persisted_config["trainer"]["resume_from"] is None
    last_payload = torch.load(output / "checkpoints" / "last.pt", map_location="cpu", weights_only=True)
    assert last_payload["config"]["trainer"]["resume_from"] is None


def test_rejected_resume_does_not_overwrite_existing_resolved_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _training_config()
    base_checkpoint = tmp_path / "base.pt"
    base_checkpoint.write_bytes(b"base-checkpoint")
    cfg.model.pretrained_checkpoint = str(base_checkpoint)
    cfg.data.root = str(tmp_path / "staging")

    def model_factory(*args, **kwargs):
        model = _TinyCameraDepthModel()
        return PreparedTrainingModel(
            model=model,
            trainable_parameter_names=tuple(name for name, _ in model.named_parameters()),
        )

    def optimizer_factory(config, model, **kwargs):
        return SimpleNamespace(
            optimizer=torch.optim.AdamW(model.parameters(), lr=1e-2),
            scheduler=None,
            group_fingerprint="d" * 64,
        )

    monkeypatch.setattr(runner_module, "ColmapRgbdDataset", lambda *args, **kwargs: _TinyDataset())
    monkeypatch.setattr(runner_module, "build_training_model", model_factory)
    monkeypatch.setattr(runner_module, "_build_optimizer_from_config", optimizer_factory)
    output = tmp_path / "run"
    run_training(cfg, output_dir=output, original_cwd=tmp_path)
    config_path = output / "resolved_config.json"
    before = config_path.read_bytes()

    cfg.trainer.epochs = 3
    cfg.trainer.resume_from = str(output / "checkpoints" / "last.pt")
    with pytest.raises(ValueError, match="configuration does not match"):
        run_training(cfg, output_dir=output, original_cwd=tmp_path)

    assert config_path.read_bytes() == before


def test_run_training_resumes_from_epoch_boundary_last_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _training_config()
    base_checkpoint = tmp_path / "base.pt"
    base_checkpoint.write_bytes(b"base-checkpoint")
    cfg.model.pretrained_checkpoint = str(base_checkpoint)
    cfg.data.root = str(tmp_path / "staging")
    interrupt = True

    def dataset_factory(*args, **kwargs):
        return _InterruptingDataset() if interrupt else _TinyDataset()

    def model_factory(*args, **kwargs):
        model = _TinyCameraDepthModel()
        return PreparedTrainingModel(
            model=model,
            trainable_parameter_names=tuple(name for name, _ in model.named_parameters()),
        )

    def optimizer_factory(config, model, **kwargs):
        return SimpleNamespace(
            optimizer=torch.optim.AdamW(model.parameters(), lr=1e-2),
            scheduler=None,
            group_fingerprint="d" * 64,
        )

    monkeypatch.setattr(runner_module, "ColmapRgbdDataset", dataset_factory)
    monkeypatch.setattr(runner_module, "build_training_model", model_factory)
    monkeypatch.setattr(runner_module, "_build_optimizer_from_config", optimizer_factory)
    output = tmp_path / "run"

    with pytest.raises(RuntimeError, match="synthetic interruption"):
        run_training(cfg, output_dir=output, original_cwd=tmp_path)
    failed_progress = json.loads((output / "progress.json").read_text())
    assert failed_progress["status"] == "failed"
    assert failed_progress["exception_type"] == "RuntimeError"
    assert failed_progress["epochs_completed"] == 1
    assert failed_progress["global_step"] == 2
    last_checkpoint = output / "checkpoints" / "last.pt"
    assert last_checkpoint.is_file()

    interrupt = False
    cfg.trainer.resume_from = str(last_checkpoint)
    summary = run_training(cfg, output_dir=output, original_cwd=tmp_path)

    assert summary["status"] == "complete"
    assert summary["epochs_completed"] == 2
    assert summary["global_step"] == 4


def test_initial_head_run_resumes_with_same_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _training_config()
    base_checkpoint = tmp_path / "base.pt"
    base_checkpoint.write_bytes(b"base-checkpoint")
    cfg.model.pretrained_checkpoint = str(base_checkpoint)
    cfg.model.initial_head_checkpoint = "initial/last.pt"
    cfg.data.root = str(tmp_path / "staging")
    base = {
        "filename": base_checkpoint.name,
        "size_bytes": base_checkpoint.stat().st_size,
        "sha256": hashlib.sha256(base_checkpoint.read_bytes()).hexdigest(),
    }
    initial = tmp_path / "initial" / "last.pt"
    initial.parent.mkdir()
    torch.save(
        {
            "format_version": 1,
            "kind": "resume",
            "parameter_state": "x",
            "epoch": 49,
            "global_step": 36200,
            "metadata": {"base_checkpoint": base},
            "model_state": {"pose": torch.ones(9), "log_depth": torch.tensor(0.0)},
        },
        initial,
    )
    interrupted = True

    def dataset_factory(*args, **kwargs):
        return _InterruptingDataset() if interrupted else _TinyDataset()

    def model_factory(*args, **kwargs):
        model = _TinyCameraDepthModel()
        return PreparedTrainingModel(
            model=model,
            trainable_parameter_names=tuple(name for name, _ in model.named_parameters()),
        )

    def optimizer_factory(config, model, **kwargs):
        return SimpleNamespace(
            optimizer=torch.optim.AdamW(model.parameters(), lr=1e-2),
            scheduler=None,
            group_fingerprint="d" * 64,
        )

    monkeypatch.setattr(runner_module, "ColmapRgbdDataset", dataset_factory)
    monkeypatch.setattr(runner_module, "build_training_model", model_factory)
    monkeypatch.setattr(runner_module, "_build_optimizer_from_config", optimizer_factory)
    output = tmp_path / "run"

    with pytest.raises(RuntimeError, match="synthetic interruption"):
        run_training(cfg, output_dir=output, original_cwd=tmp_path)
    last_checkpoint = output / "checkpoints" / "last.pt"
    assert last_checkpoint.is_file()

    interrupted = False
    cfg.trainer.resume_from = str(last_checkpoint)
    summary = run_training(cfg, output_dir=output, original_cwd=tmp_path)

    assert summary["status"] == "complete"
    assert summary["epochs_completed"] == 2
    assert summary["global_step"] == 4
    assert summary["initial_head_checkpoint"]["sha256"] == hashlib.sha256(initial.read_bytes()).hexdigest()
