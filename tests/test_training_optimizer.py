from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import cast

import pytest
import torch

from vggt_omega.training.optim.amuse import AMUSE
from vggt_omega.training.optimizer_factory import (
    AMUSE_TERMINAL_PARAMETER_NAMES,
    ParameterGroupingError,
    build_adamw_optimizer,
    build_amuse_optimizer,
    classify_amuse_parameters,
    validate_parameter_partition,
)
from vggt_omega.training.runner import _set_optimizer_stage_learning_rate_scale

REPO_ROOT = Path(__file__).resolve().parents[1]
AMUSE_SOURCE = REPO_ROOT / "vggt_omega" / "training" / "optim" / "amuse.py"
AMUSE_UPSTREAM = REPO_ROOT / "third_party" / "amuse" / "UPSTREAM.md"
AMUSE_LICENSE = REPO_ROOT / "third_party" / "amuse" / "LICENSE"
UPSTREAM_COMMIT = "48922743b32f33f919ab54edde3dbad0d0ce2dc7"
UPSTREAM_BLOB = "144361bf100d0a3a07172fb007a6fb27ff58f046"


def _git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data, usedforsecurity=False).hexdigest()


def _make_small_optimizer() -> tuple[torch.nn.Parameter, torch.nn.Parameter, AMUSE]:
    matrix = torch.nn.Parameter(torch.tensor([[0.4, -0.2], [0.1, 0.3]], dtype=torch.float32))
    bias = torch.nn.Parameter(torch.tensor([0.2, -0.1], dtype=torch.float32))
    optimizer = AMUSE(
        [
            {
                "params": [matrix],
                "use_muon": True,
                "aux_update_type": "adamw",
                "lr": 1e-3,
                "momentum": 0.95,
                "weight_decay": 0.01,
            },
            {
                "params": [bias],
                "use_muon": False,
                "update_type": "adamw",
                "lr": 1e-4,
                "beta2": 0.999,
                "eps": 1e-10,
                "weight_decay": 0.01,
            },
        ],
        beta1=0.4,
        warmup_steps=2,
        rho=0.3,
    )
    return matrix, bias, optimizer


def _step_small_model(matrix: torch.Tensor, bias: torch.Tensor, optimizer: AMUSE) -> float:
    features = torch.tensor([[2.0, -1.0], [0.5, 1.5]], dtype=torch.float32)
    target = torch.tensor([[0.1, -0.2], [0.3, 0.4]], dtype=torch.float32)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        prediction = features @ matrix + bias
        loss = torch.nn.functional.mse_loss(prediction, target)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_([matrix, bias], max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    assert math.isfinite(float(loss.detach()))
    assert math.isfinite(float(grad_norm))
    return float(loss.detach())


def test_vendored_amuse_matches_pinned_upstream_blob() -> None:
    assert _git_blob_sha1(AMUSE_SOURCE) == UPSTREAM_BLOB

    provenance = AMUSE_UPSTREAM.read_text(encoding="utf-8")
    assert "https://github.com/kjeiun/amuse" in provenance
    assert UPSTREAM_COMMIT in provenance
    assert UPSTREAM_BLOB in provenance
    assert "Local modifications: none" in provenance
    assert "Apache License" in AMUSE_LICENSE.read_text(encoding="utf-8")


def test_amuse_rejects_zero_warmup() -> None:
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    with pytest.raises(ValueError, match="warmup_steps > 0"):
        AMUSE([{"params": [parameter], "use_muon": True}], warmup_steps=0)


def test_amuse_requires_train_mode_before_step() -> None:
    matrix, bias, optimizer = _make_small_optimizer()
    (matrix.sum() + bias.sum()).backward()
    with pytest.raises(Exception, match="not in train mode"):
        optimizer.step()


def test_amuse_matrix_and_bias_two_step_bfloat16_update_is_finite() -> None:
    matrix, bias, optimizer = _make_small_optimizer()
    matrix_before = matrix.detach().clone()
    bias_before = bias.detach().clone()
    optimizer.train()

    losses = [_step_small_model(matrix, bias, optimizer) for _ in range(2)]

    assert all(math.isfinite(loss) for loss in losses)
    assert torch.isfinite(matrix).all()
    assert torch.isfinite(bias).all()
    assert not torch.equal(matrix, matrix_before)
    assert not torch.equal(bias, bias_before)
    assert [group["k"] for group in optimizer.param_groups] == [2, 2]


def test_amuse_eval_train_round_trip_restores_training_weights() -> None:
    matrix, bias, optimizer = _make_small_optimizer()
    optimizer.train()
    _step_small_model(matrix, bias, optimizer)
    _step_small_model(matrix, bias, optimizer)
    training_matrix = matrix.detach().clone()
    training_bias = bias.detach().clone()

    optimizer.eval()
    evaluation_matrix = matrix.detach().clone()
    optimizer.train()

    assert not torch.equal(evaluation_matrix, training_matrix)
    torch.testing.assert_close(matrix, training_matrix, rtol=0.0, atol=1e-7)
    torch.testing.assert_close(bias, training_bias, rtol=0.0, atol=1e-7)


class _TinyOmegaHeads(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.aggregator = torch.nn.Linear(4, 4)
        self.camera_head = torch.nn.Module()
        self.camera_head.token_norm = torch.nn.LayerNorm(4)
        self.camera_head.trunk = torch.nn.Linear(4, 4)
        self.camera_head.camera_branch = torch.nn.Sequential(
            torch.nn.Linear(4, 2),
            torch.nn.GELU(),
            torch.nn.Linear(2, 9),
        )
        self.dense_head = torch.nn.Module()
        self.dense_head.decoder = torch.nn.Conv2d(4, 4, kernel_size=3, padding=1)
        self.dense_head.proj = torch.nn.Conv2d(4, 1, kernel_size=1)
        self.dense_head.proj_conf = torch.nn.Conv2d(4, 1, kernel_size=1)

        self.aggregator.requires_grad_(False)
        self.dense_head.proj_conf.requires_grad_(False)


class _TinyDepthInputOmega(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adapter = torch.nn.Conv2d(2, 4, kernel_size=2, stride=2)
        self.base_model = _TinyOmegaHeads()


def test_amuse_parameter_groups_are_complete_disjoint_and_terminal_safe() -> None:
    model = _TinyOmegaHeads()
    grouping = classify_amuse_parameters(model)
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}

    assert set(grouping.muon_names).isdisjoint(grouping.fallback_names)
    assert set(grouping.muon_names) | set(grouping.fallback_names) == trainable_names
    assert set(AMUSE_TERMINAL_PARAMETER_NAMES) <= set(grouping.fallback_names)
    assert set(AMUSE_TERMINAL_PARAMETER_NAMES).isdisjoint(grouping.muon_names)
    assert "camera_head.trunk.weight" in grouping.muon_names
    assert "dense_head.decoder.weight" in grouping.muon_names
    assert "camera_head.token_norm.weight" in grouping.fallback_names
    assert "aggregator.weight" in grouping.frozen_names
    assert "dense_head.proj_conf.weight" in grouping.frozen_names
    assert len(grouping.fingerprint) == 64
    assert grouping.fingerprint == classify_amuse_parameters(model).fingerprint


def test_depth_input_adapter_and_wrapped_heads_have_complete_optimizer_coverage() -> None:
    model = _TinyDepthInputOmega()

    grouping = classify_amuse_parameters(model)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}

    assert set(grouping.muon_names) | set(grouping.fallback_names) == trainable
    assert any(name.startswith("adapter.") for name in grouping.muon_names)
    assert "base_model.dense_head.proj_conf.weight" in grouping.frozen_names


@pytest.mark.parametrize(
    ("muon_names", "fallback_names", "match"),
    [
        (("camera_head.trunk.weight",), ("camera_head.trunk.weight",), "overlap"),
        ((), (), "missing"),
        (("not.a.parameter",), (), "unknown"),
    ],
)
def test_parameter_partition_rejects_overlap_missing_and_unknown_names(
    muon_names: tuple[str, ...], fallback_names: tuple[str, ...], match: str
) -> None:
    model = _TinyOmegaHeads()
    with pytest.raises(ParameterGroupingError, match=match):
        validate_parameter_partition(model, muon_names=muon_names, fallback_names=fallback_names)


def test_amuse_parameter_groups_reject_unexpected_trainable_parameter() -> None:
    model = _TinyOmegaHeads()
    model.aggregator.weight.requires_grad_(True)

    with pytest.raises(ParameterGroupingError, match="outside camera_head/dense_head"):
        classify_amuse_parameters(model)


def test_amuse_parameter_groups_reject_missing_terminal_projection() -> None:
    model = _TinyOmegaHeads()
    projection = cast(torch.nn.Conv2d, model.dense_head.proj)
    projection.weight.requires_grad_(False)

    with pytest.raises(ParameterGroupingError, match="terminal prediction weights must be trainable"):
        classify_amuse_parameters(model)


def test_build_amuse_optimizer_computes_warmup_and_preserves_explicit_mode() -> None:
    model = _TinyOmegaHeads()
    result = build_amuse_optimizer(model, total_optimizer_steps=21, warmup_ratio=0.05)

    assert isinstance(result.optimizer, AMUSE)
    assert result.scheduler is None
    assert result.warmup_steps == 2
    assert result.optimizer.train_mode is False
    assert [group["group_name"] for group in result.optimizer.param_groups] == ["muon", "fallback"]
    assert result.optimizer.param_groups[0]["use_muon"] is True
    assert result.optimizer.param_groups[1]["use_muon"] is False
    assert result.optimizer.param_groups[1]["update_type"] == "adamw"
    assert result.grouping.fingerprint == result.group_fingerprint


def test_amuse_preserves_explicit_stage_learning_rate_scale_after_step() -> None:
    matrix, bias, optimizer = _make_small_optimizer()
    base_learning_rates = tuple(float(group["lr"]) for group in optimizer.param_groups)
    _set_optimizer_stage_learning_rate_scale(optimizer, base_learning_rates, 0.5)
    matrix.grad = torch.ones_like(matrix)
    bias.grad = torch.ones_like(bias)

    optimizer.train()
    optimizer.step()

    # The first of two warmup steps applies sched=0.5 in addition to the
    # curriculum's persistent stage scale=0.5.
    assert optimizer.param_groups[0]["base_lr"] == pytest.approx(1e-3 * 0.5)
    assert optimizer.param_groups[1]["base_lr"] == pytest.approx(1e-4 * 0.5)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3 * 0.5 * 0.5)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(1e-4 * 0.5 * 0.5)


def test_build_amuse_optimizer_rejects_external_scheduler() -> None:
    with pytest.raises(ValueError, match="external scheduler"):
        build_amuse_optimizer(_TinyOmegaHeads(), total_optimizer_steps=10, external_scheduler="cosine")


def test_build_amuse_optimizer_rejects_fsdp() -> None:
    with pytest.raises(ValueError, match="FSDP"):
        build_amuse_optimizer(_TinyOmegaHeads(), total_optimizer_steps=10, distributed_strategy="fsdp")


@pytest.mark.parametrize("total_optimizer_steps", [True, 0, -1, 1.5])
def test_build_amuse_optimizer_rejects_nonpositive_or_noninteger_steps(
    total_optimizer_steps: int,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        build_amuse_optimizer(_TinyOmegaHeads(), total_optimizer_steps=total_optimizer_steps)


def test_build_adamw_optimizer_has_complete_trainable_coverage() -> None:
    model = _TinyOmegaHeads()
    result = build_adamw_optimizer(
        model,
        lr=2e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )
    grouped_parameters = [parameter for group in result.optimizer.param_groups for parameter in group["params"]]
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]

    assert isinstance(result.optimizer, torch.optim.AdamW)
    assert result.scheduler is None
    assert result.warmup_steps == 0
    assert {id(parameter) for parameter in grouped_parameters} == {id(parameter) for parameter in trainable_parameters}
    assert len({id(parameter) for parameter in grouped_parameters}) == len(grouped_parameters)
    assert [group["group_name"] for group in result.optimizer.param_groups] == [
        "adamw_matrix",
        "adamw_fallback",
    ]
    assert all(group["lr"] == pytest.approx(2e-5) for group in result.optimizer.param_groups)
