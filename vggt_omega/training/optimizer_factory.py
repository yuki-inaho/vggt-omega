from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from vggt_omega.training.optim.amuse import AMUSE

AMUSE_TERMINAL_PARAMETER_NAMES = (
    "camera_head.camera_branch.2.weight",
    "dense_head.proj.weight",
)
_TRAINABLE_PREFIXES = ("camera_head.", "dense_head.")
_CONFIDENCE_PREFIX = "dense_head.proj_conf."
_SUPPORTED_STRATEGIES = {"single", "ddp"}


class ParameterGroupingError(ValueError):
    """Raised when trainable parameters cannot be grouped without guessing."""


@dataclass(frozen=True)
class AmuseParameterGrouping:
    muon_names: tuple[str, ...]
    fallback_names: tuple[str, ...]
    frozen_names: tuple[str, ...]
    fingerprint: str


@dataclass(frozen=True)
class OptimizerBuildResult:
    optimizer: torch.optim.Optimizer
    scheduler: None
    grouping: AmuseParameterGrouping
    warmup_steps: int
    group_fingerprint: str


def _named_parameters(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    return dict(model.named_parameters())


def _ordered_muon_names(names: Sequence[str], named_parameters: Mapping[str, torch.nn.Parameter]) -> tuple[str, ...]:
    # AMUSE applies the same stable size sort in its constructor. Starting from
    # lexical order makes equal-size tensors deterministic across processes.
    lexical_names = sorted(names)
    return tuple(sorted(lexical_names, key=lambda name: named_parameters[name].size(), reverse=True))


def _group_fingerprint(
    named_parameters: Mapping[str, torch.nn.Parameter],
    *,
    muon_names: Sequence[str],
    fallback_names: Sequence[str],
) -> str:
    def describe(name: str) -> dict[str, Any]:
        parameter = named_parameters[name]
        return {
            "dtype": str(parameter.dtype),
            "name": name,
            "shape": list(parameter.shape),
        }

    payload = {
        "format_version": 1,
        "fallback": [describe(name) for name in fallback_names],
        "muon": [describe(name) for name in muon_names],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_parameter_partition(
    model: torch.nn.Module,
    *,
    muon_names: Sequence[str],
    fallback_names: Sequence[str],
) -> None:
    """Validate an exact, disjoint partition of all trainable parameters."""

    named_parameters = _named_parameters(model)
    muon_list = list(muon_names)
    fallback_list = list(fallback_names)
    duplicate_muon = {name for name in muon_list if muon_list.count(name) > 1}
    duplicate_fallback = {name for name in fallback_list if fallback_list.count(name) > 1}
    if duplicate_muon or duplicate_fallback:
        duplicates = sorted(duplicate_muon | duplicate_fallback)
        raise ParameterGroupingError(f"duplicate parameter names in a group: {duplicates}")

    grouped = set(muon_list) | set(fallback_list)
    unknown = grouped - set(named_parameters)
    if unknown:
        raise ParameterGroupingError(f"unknown parameter names: {sorted(unknown)}")

    overlap = set(muon_list) & set(fallback_list)
    if overlap:
        raise ParameterGroupingError(f"parameter group overlap: {sorted(overlap)}")

    trainable = {name for name, parameter in named_parameters.items() if parameter.requires_grad}
    missing = trainable - grouped
    if missing:
        raise ParameterGroupingError(f"missing trainable parameters: {sorted(missing)}")

    frozen_in_groups = grouped - trainable
    if frozen_in_groups:
        raise ParameterGroupingError(f"frozen parameters included in optimizer groups: {sorted(frozen_in_groups)}")


def classify_amuse_parameters(model: torch.nn.Module) -> AmuseParameterGrouping:
    """Classify the frozen-backbone VGGT-Omega heads for AMUSE."""

    named_parameters = _named_parameters(model)
    absent_terminal = set(AMUSE_TERMINAL_PARAMETER_NAMES) - set(named_parameters)
    if absent_terminal:
        raise ParameterGroupingError(f"expected terminal prediction weights are absent: {sorted(absent_terminal)}")

    frozen_terminal = [name for name in AMUSE_TERMINAL_PARAMETER_NAMES if not named_parameters[name].requires_grad]
    if frozen_terminal:
        raise ParameterGroupingError(f"terminal prediction weights must be trainable: {frozen_terminal}")

    muon_names: list[str] = []
    fallback_names: list[str] = []
    frozen_names: list[str] = []
    for name, parameter in sorted(named_parameters.items()):
        if not parameter.requires_grad:
            frozen_names.append(name)
            continue
        if not name.startswith(_TRAINABLE_PREFIXES):
            raise ParameterGroupingError(f"trainable parameter outside camera_head/dense_head is not supported: {name}")
        if name.startswith(_CONFIDENCE_PREFIX):
            raise ParameterGroupingError(f"confidence projection must remain frozen: {name}")

        if name in AMUSE_TERMINAL_PARAMETER_NAMES or parameter.ndim < 2:
            fallback_names.append(name)
        elif parameter.ndim in {2, 4}:
            muon_names.append(name)
        else:
            raise ParameterGroupingError(
                f"unsupported trainable parameter rank for AMUSE grouping: {name} has ndim={parameter.ndim}"
            )

    if not muon_names:
        raise ParameterGroupingError("Muon parameter group is empty")
    if not fallback_names:
        raise ParameterGroupingError("fallback parameter group is empty")

    ordered_muon_names = _ordered_muon_names(muon_names, named_parameters)
    ordered_fallback_names = tuple(sorted(fallback_names))
    validate_parameter_partition(
        model,
        muon_names=ordered_muon_names,
        fallback_names=ordered_fallback_names,
    )
    fingerprint = _group_fingerprint(
        named_parameters,
        muon_names=ordered_muon_names,
        fallback_names=ordered_fallback_names,
    )
    return AmuseParameterGrouping(
        muon_names=ordered_muon_names,
        fallback_names=ordered_fallback_names,
        frozen_names=tuple(sorted(frozen_names)),
        fingerprint=fingerprint,
    )


def _validate_finite(name: str, value: float, *, positive: bool = False, nonnegative: bool = False) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be > 0")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be >= 0")


def build_amuse_optimizer(
    model: torch.nn.Module,
    *,
    total_optimizer_steps: int,
    muon_lr: float = 1e-4,
    aux_lr: float = 1e-5,
    aux_update_type: str = "adamw",
    beta1: float = 0.4,
    beta2: float = 0.999,
    momentum: float = 0.95,
    rho: float = 0.3,
    r: float = 0.0,
    weight_lr_power: float = 2.0,
    warmup_ratio: float = 0.05,
    weight_decay: float = 0.01,
    weight_decay_at_y: float = 0.0,
    external_scheduler: Any = None,
    distributed_strategy: str = "single",
) -> OptimizerBuildResult:
    """Build AMUSE without entering its stateful training mode."""

    if external_scheduler is not None:
        raise ValueError("AMUSE does not support an external scheduler")
    normalized_strategy = distributed_strategy.lower()
    if normalized_strategy == "fsdp":
        raise ValueError("AMUSE with FSDP is not supported")
    if normalized_strategy not in _SUPPORTED_STRATEGIES:
        raise ValueError(f"unsupported distributed strategy for AMUSE: {distributed_strategy}")
    if (
        isinstance(total_optimizer_steps, bool)
        or not isinstance(total_optimizer_steps, int)
        or total_optimizer_steps < 1
    ):
        raise ValueError("total_optimizer_steps must be a positive integer")
    if not 0.0 <= warmup_ratio <= 1.0:
        raise ValueError("warmup_ratio must be in [0, 1]")
    if not 0.0 < beta1 < 1.0:
        raise ValueError("beta1 must be in (0, 1)")
    if not 0.0 <= beta2 < 1.0:
        raise ValueError("beta2 must be in [0, 1)")
    if not 0.0 <= momentum < 1.0:
        raise ValueError("momentum must be in [0, 1)")
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    if aux_update_type != "adamw":
        raise ValueError("VGGT-Omega AMUSE requires aux_update_type='adamw'")

    for name, value, options in (
        ("muon_lr", muon_lr, {"positive": True}),
        ("aux_lr", aux_lr, {"positive": True}),
        ("r", r, {}),
        ("weight_lr_power", weight_lr_power, {}),
        ("weight_decay", weight_decay, {"nonnegative": True}),
        ("weight_decay_at_y", weight_decay_at_y, {"nonnegative": True}),
    ):
        _validate_finite(name, float(value), **options)

    grouping = classify_amuse_parameters(model)
    named_parameters = _named_parameters(model)
    warmup_steps = max(1, math.ceil(total_optimizer_steps * warmup_ratio))
    parameter_groups = [
        {
            "params": [named_parameters[name] for name in grouping.muon_names],
            "param_names": list(grouping.muon_names),
            "group_name": "muon",
            "use_muon": True,
            "aux_update_type": aux_update_type,
            "lr": muon_lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
        },
        {
            "params": [named_parameters[name] for name in grouping.fallback_names],
            "param_names": list(grouping.fallback_names),
            "group_name": "fallback",
            "use_muon": False,
            "update_type": aux_update_type,
            "lr": aux_lr,
            "beta2": beta2,
            "eps": 1e-10,
            "weight_decay": weight_decay,
        },
    ]
    optimizer = AMUSE(
        parameter_groups,
        weight_decay_at_y=weight_decay_at_y,
        beta1=beta1,
        weight_lr_power=weight_lr_power,
        warmup_steps=warmup_steps,
        rho=rho,
        r=r,
    )
    return OptimizerBuildResult(
        optimizer=optimizer,
        scheduler=None,
        grouping=grouping,
        warmup_steps=warmup_steps,
        group_fingerprint=grouping.fingerprint,
    )


def build_adamw_optimizer(
    model: torch.nn.Module,
    *,
    lr: float,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.01,
) -> OptimizerBuildResult:
    """Build the explicit AdamW comparison profile over the same trainable set."""

    _validate_finite("lr", float(lr), positive=True)
    _validate_finite("eps", float(eps), positive=True)
    _validate_finite("weight_decay", float(weight_decay), nonnegative=True)
    if len(betas) != 2 or any(not 0.0 <= beta < 1.0 for beta in betas):
        raise ValueError("betas must contain two values in [0, 1)")

    grouping = classify_amuse_parameters(model)
    named_parameters = _named_parameters(model)
    parameter_groups = [
        {
            "params": [named_parameters[name] for name in grouping.muon_names],
            "param_names": list(grouping.muon_names),
            "group_name": "adamw_matrix",
            "lr": lr,
            "weight_decay": weight_decay,
        },
        {
            "params": [named_parameters[name] for name in grouping.fallback_names],
            "param_names": list(grouping.fallback_names),
            "group_name": "adamw_fallback",
            "lr": lr,
            "weight_decay": weight_decay,
        },
    ]
    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )
    return OptimizerBuildResult(
        optimizer=optimizer,
        scheduler=None,
        grouping=grouping,
        warmup_steps=0,
        group_fingerprint=grouping.fingerprint,
    )
