from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import pytest
import torch
from torch.utils.data import default_collate

from vggt_omega.training.checkpointing import optimizer_evaluation_state
from vggt_omega.training.dataset import ColmapRgbdDataset, DataContractError
from vggt_omega.training.losses import compute_camera_depth_loss
from vggt_omega.training.model_factory import build_training_model
from vggt_omega.training.optimizer_factory import build_amuse_optimizer


def _required_path(environment_name: str) -> Path:
    raw_value = os.environ.get(environment_name)
    if not raw_value:
        pytest.skip(f"{environment_name} is required for the real-data GPU smoke test")
    path = Path(raw_value).expanduser().resolve()
    if not path.exists():
        pytest.fail(f"{environment_name} does not exist: {path}")
    return path


def _finite_scalars(losses: dict[str, torch.Tensor]) -> dict[str, float]:
    scalars = {name: float(value.detach()) for name, value in losses.items()}
    assert all(math.isfinite(value) for value in scalars.values())
    return scalars


@pytest.mark.gpu
@pytest.mark.slow
def test_real_checkpoint_rgbd_amuse_step_is_finite_and_updates_only_heads() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the real-data GPU smoke test")

    checkpoint = _required_path("VGGT_OMEGA_TRAINING_CHECKPOINT")
    staging = _required_path("VGGT_OMEGA_TRAINING_STAGING")
    frames = int(os.environ.get("VGGT_OMEGA_SMOKE_FRAMES", "2"))
    if not 2 <= frames <= 4:
        pytest.fail("VGGT_OMEGA_SMOKE_FRAMES must be between 2 and 4")

    dataset = ColmapRgbdDataset(
        staging,
        split="smoke",
        min_frames=frames,
        max_frames=frames,
        seed=42,
        min_valid_depth_pixels=1024,
    )
    sample = None
    for index in range(len(dataset)):
        try:
            sample = dataset[index]
        except DataContractError:
            continue
        break
    if sample is None:
        pytest.fail(f"smoke split contains no valid {frames}-frame sequence")
    batch = {
        key: value.cuda(non_blocking=False) if isinstance(value, torch.Tensor) else value
        for key, value in default_collate([sample]).items()
    }

    prepared = build_training_model(checkpoint, device="cuda")
    model = prepared.model
    optimizer_bundle = build_amuse_optimizer(model, total_optimizer_steps=1)
    optimizer = optimizer_bundle.optimizer
    optimizer.train()

    named_parameters = dict(model.named_parameters())
    watched_name = "camera_head.camera_branch.2.weight"
    watched_before = named_parameters[watched_name].detach().clone()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        predictions = model(batch["images"])
        losses = compute_camera_depth_loss(predictions, batch, min_valid_depth_pixels=1024)
    train_scalars = _finite_scalars(losses)
    losses["objective"].backward()

    aggregator_gradients = [parameter.grad for parameter in model.aggregator.parameters()]
    confidence_gradients = [parameter.grad for parameter in model.dense_head.proj_conf.parameters()]
    trainable_gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert aggregator_gradients and all(gradient is None for gradient in aggregator_gradients)
    assert confidence_gradients and all(gradient is None for gradient in confidence_gradients)
    assert trainable_gradients and all(gradient is not None for gradient in trainable_gradients)
    assert all(torch.isfinite(gradient).all() for gradient in trainable_gradients if gradient is not None)

    preclip_norm = float(
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            max_norm=1.0,
            error_if_nonfinite=True,
        )
    )
    clipped_norm = math.sqrt(
        sum(float(gradient.detach().float().square().sum()) for gradient in trainable_gradients if gradient is not None)
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    step_seconds = time.perf_counter() - started

    assert math.isfinite(preclip_norm)
    assert clipped_norm <= 1.001
    assert not torch.equal(watched_before, named_parameters[watched_name].detach())

    with optimizer_evaluation_state(optimizer), torch.no_grad():
        model.eval()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            validation_predictions = model(batch["images"])
            validation_losses = compute_camera_depth_loss(
                validation_predictions,
                batch,
                min_valid_depth_pixels=1024,
            )
    validation_scalars = _finite_scalars(validation_losses)

    report = {
        "format_version": 1,
        "status": "pass",
        "frames": frames,
        "optimizer": "amuse",
        "parameter_state": "x",
        "group_fingerprint": optimizer_bundle.group_fingerprint,
        "train": train_scalars,
        "validation": validation_scalars,
        "gradient_norm_before_clip": preclip_norm,
        "gradient_norm_after_clip": clipped_norm,
        "step_seconds": step_seconds,
        "max_cuda_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
    }
    report_path = os.environ.get("VGGT_OMEGA_SMOKE_REPORT")
    if report_path:
        destination = Path(report_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
