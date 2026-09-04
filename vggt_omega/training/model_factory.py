"""Build a pretrained VGGT-Omega model for camera/depth fine-tuning."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_checkpoint_state_dict


@dataclass(frozen=True)
class PreparedTrainingModel:
    """A strictly loaded model and its exact trainable-parameter contract."""

    model: nn.Module
    trainable_parameter_names: tuple[str, ...]


def build_training_model(
    checkpoint_path: str | Path,
    *,
    model_builder: Callable[[], nn.Module] = VGGTOmega,
    device: torch.device | str | None = None,
) -> PreparedTrainingModel:
    """Strictly load pretrained weights and expose only camera/depth parameters.

    A checkpoint is mandatory. Any missing key, unexpected key, shape mismatch,
    or non-finite attention bias mask aborts construction; this function never
    falls back to random initialization.
    """

    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model = model_builder()
    state_dict = load_checkpoint_state_dict(checkpoint)
    try:
        incompatible = model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise RuntimeError(f"Strict pretrained checkpoint load failed for {checkpoint.name}: {error}") from error

    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Strict pretrained checkpoint load returned incompatible keys: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    _validate_attention_bias_masks(model)
    _configure_trainable_parameters(model)
    set_training_mode(model)

    if device is not None:
        model = model.to(device)

    trainable_names = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    if not trainable_names:
        raise RuntimeError("Camera/depth training configuration produced no trainable parameters")
    _validate_trainable_names(trainable_names)
    return PreparedTrainingModel(model=model, trainable_parameter_names=trainable_names)


def set_training_mode(model: nn.Module) -> nn.Module:
    """Enable head training while keeping the frozen encoder in evaluation mode."""

    aggregator = _require_module(model, "aggregator")
    dense_head = _require_module(model, "dense_head")
    confidence_projection = _require_module(dense_head, "proj_conf", prefix="dense_head")

    model.train()
    aggregator.requires_grad_(False)
    aggregator.eval()
    confidence_projection.requires_grad_(False)
    confidence_projection.eval()
    return model


def _configure_trainable_parameters(model: nn.Module) -> None:
    camera_head = _require_module(model, "camera_head")
    dense_head = _require_module(model, "dense_head")
    confidence_projection = _require_module(dense_head, "proj_conf", prefix="dense_head")

    # Start from an explicit deny-all state so optional heads cannot silently
    # enter the optimizer when the model grows new inference capabilities.
    model.requires_grad_(False)
    camera_head.requires_grad_(True)
    dense_head.requires_grad_(True)
    confidence_projection.requires_grad_(False)


def _require_module(module: nn.Module, name: str, *, prefix: str = "model") -> nn.Module:
    value = getattr(module, name, None)
    if not isinstance(value, nn.Module):
        raise TypeError(f"{prefix}.{name} must be an initialized torch.nn.Module")
    return value


def _validate_attention_bias_masks(model: nn.Module) -> None:
    nonfinite: list[str] = []
    for name, buffer in model.named_buffers():
        if name.endswith("bias_mask") and not torch.isfinite(buffer).all():
            nonfinite.append(name)
    if nonfinite:
        raise ValueError(f"Checkpoint contains non-finite attention bias mask buffers: {nonfinite}")


def _validate_trainable_names(trainable_names: tuple[str, ...]) -> None:
    invalid = [
        name
        for name in trainable_names
        if not (name.startswith("camera_head.") or name.startswith("dense_head."))
        or name.startswith("dense_head.proj_conf.")
    ]
    if invalid:
        raise RuntimeError(f"Unexpected trainable parameters outside camera/depth heads: {invalid}")
