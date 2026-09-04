from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from vggt_omega.training.model_factory import build_training_model, set_training_mode


class _TinyDenseHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Linear(4, 4)
        self.proj = nn.Linear(4, 1)
        self.proj_conf = nn.Linear(4, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = torch.tanh(self.trunk(features))
        return self.proj(hidden), self.proj_conf(hidden)


class _TinyOmega(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.aggregator = nn.Linear(3, 4)
        self.aggregator.register_buffer("bias_mask", torch.zeros(4))
        self.camera_head = nn.Linear(4, 9)
        self.dense_head = _TinyDenseHead()

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.aggregator(images)
        depth, depth_conf = self.dense_head(features)
        return {
            "pose_enc": self.camera_head(features),
            "depth": depth,
            "depth_conf": depth_conf,
        }


def _write_checkpoint(path: Path, *, state: dict[str, torch.Tensor] | None = None) -> Path:
    if state is None:
        torch.manual_seed(0)
        state = _TinyOmega().state_dict()
    torch.save({"model": state}, path)
    return path


def test_build_training_model_strictly_loads_and_freezes_expected_modules(tmp_path: Path) -> None:
    checkpoint = _write_checkpoint(tmp_path / "model.pt")

    prepared = build_training_model(checkpoint, model_builder=_TinyOmega)
    model = prepared.model

    assert prepared.trainable_parameter_names
    assert all(not parameter.requires_grad for parameter in model.aggregator.parameters())
    assert all(not parameter.requires_grad for parameter in model.dense_head.proj_conf.parameters())
    assert all(parameter.requires_grad for parameter in model.camera_head.parameters())
    assert all(parameter.requires_grad for parameter in model.dense_head.trunk.parameters())
    assert all(parameter.requires_grad for parameter in model.dense_head.proj.parameters())
    assert set(prepared.trainable_parameter_names) == {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }


def test_set_training_mode_keeps_frozen_aggregator_in_eval_and_gradients_out(tmp_path: Path) -> None:
    prepared = build_training_model(_write_checkpoint(tmp_path / "model.pt"), model_builder=_TinyOmega)
    model = set_training_mode(prepared.model)

    assert model.training
    assert not model.aggregator.training

    predictions = model(torch.ones(2, 3))
    loss = predictions["pose_enc"].square().mean() + predictions["depth"].square().mean()
    loss.backward()

    assert all(parameter.grad is None for parameter in model.aggregator.parameters())
    assert all(parameter.grad is None for parameter in model.dense_head.proj_conf.parameters())
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert trainable
    assert all(parameter.grad is not None for parameter in trainable)
    assert all(torch.isfinite(parameter.grad).all() for parameter in trainable if parameter.grad is not None)


def test_build_training_model_rejects_missing_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        build_training_model(tmp_path / "missing.pt", model_builder=_TinyOmega)


def test_build_training_model_rejects_state_dict_mismatch_without_fallback(tmp_path: Path) -> None:
    state = _TinyOmega().state_dict()
    state.pop("camera_head.bias")
    checkpoint = _write_checkpoint(tmp_path / "incomplete.pt", state=state)

    with pytest.raises(RuntimeError, match=r"camera_head\.bias"):
        build_training_model(checkpoint, model_builder=_TinyOmega)


def test_build_training_model_rejects_checkpoint_shape_mismatch(tmp_path: Path) -> None:
    state = _TinyOmega().state_dict()
    state["camera_head.weight"] = torch.zeros(8, 4)
    checkpoint = _write_checkpoint(tmp_path / "wrong-shape.pt", state=state)

    with pytest.raises(RuntimeError, match="size mismatch"):
        build_training_model(checkpoint, model_builder=_TinyOmega)


def test_build_training_model_rejects_nonfinite_attention_bias_mask(tmp_path: Path) -> None:
    state = _TinyOmega().state_dict()
    state["aggregator.bias_mask"] = torch.full_like(state["aggregator.bias_mask"], torch.nan)
    checkpoint = _write_checkpoint(tmp_path / "nonfinite.pt", state=state)

    with pytest.raises(ValueError, match="non-finite attention bias mask"):
        build_training_model(checkpoint, model_builder=_TinyOmega)
