"""Build a pretrained VGGT-Omega model for camera/depth fine-tuning."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from vggt_omega.models import VGGTOmega
from vggt_omega.training.dynamic_model import DynamicGeometryTrainingModel
from vggt_omega.training.pixel_depth_model import PixelPerfectDepthTrainingModel
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


def attach_pixel_depth_model(
    prepared: PreparedTrainingModel,
    config: Mapping[str, object],
    *,
    device: torch.device | str | None = None,
) -> PreparedTrainingModel:
    """Attach the optional refiner only after strict base/head initialization."""

    refiner = _mapping(config, "refiner")
    temporal = _mapping(config, "temporal")
    geometry = _mapping(config, "geometry")
    flow = _mapping(config, "flow")
    optimization = _mapping(config, "optimization")
    self_supervised = config.get("self_supervised")
    correspondence: Mapping[str, object] = {}
    if self_supervised is not None:
        if not isinstance(self_supervised, Mapping):
            raise ValueError("pixel-depth config 'self_supervised' must be a mapping")
        correspondence = _mapping(self_supervised, "correspondence")
    correspondence_enabled = bool(correspondence.get("enabled", False))
    model = PixelPerfectDepthTrainingModel(
        prepared.model,
        semantic_input_dim=int(config["semantic_input_dim"]),
        hidden_dim=int(refiner["hidden_dim"]),
        refiner_depth=int(refiner["depth"]),
        num_heads=int(refiner["num_heads"]),
        coarse_patch_size=int(refiner["coarse_patch_size"]),
        fine_patch_size=int(refiner["fine_patch_size"]),
        temporal_depth=int(temporal["depth"]),
        log_residual_scale=float(flow["log_residual_scale"]),
        gradient_loss_weight=float(flow["gradient_loss_weight"]),
        ode_steps=int(flow["ode_steps"]),
        max_depth_m=float(geometry["max_depth_m"]),
        geometry_enabled=bool(geometry["enabled"]),
        reference_mode=str(temporal["reference_mode"]),
        residual_gate_initial=float(flow.get("residual_gate_initial", 1.0)),
        correspondence_enabled=correspondence_enabled,
        correspondence_hidden_dim=int(correspondence.get("hidden_dim", 256)),
        correspondence_pair_chunk_size=int(correspondence.get("pair_chunk_size", 2)),
    )
    if not bool(optimization["train_base_heads"]):
        model.base_model.requires_grad_(False)
    model.semantic_adapter.requires_grad_(True)
    model.temporal_mixer.requires_grad_(True)
    model.refiner.requires_grad_(True)
    model.residual_gate.requires_grad_(True)
    if model.correspondence_head is not None:
        model.correspondence_head.requires_grad_(True)
    if device is not None:
        model = model.to(device)
    model.train()
    trainable_names = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    required_prefixes = ("semantic_adapter.", "temporal_mixer.", "refiner.", "residual_gate.")
    if correspondence_enabled:
        required_prefixes = (*required_prefixes, "correspondence_head.")
    missing_groups = [
        prefix for prefix in required_prefixes if not any(name.startswith(prefix) for name in trainable_names)
    ]
    if missing_groups:
        raise RuntimeError(f"pixel-depth trainable state is missing groups: {missing_groups}")
    return PreparedTrainingModel(model=model, trainable_parameter_names=trainable_names)


def attach_dynamic_geometry_model(
    prepared: PreparedTrainingModel,
    config: Mapping[str, object],
    *,
    device: torch.device | str | None = None,
) -> PreparedTrainingModel:
    """Attach dynamic geometry only when explicitly enabled."""

    if not bool(config.get("enabled", False)):
        return prepared
    feature_dim = getattr(prepared.model, "patch_feature_dim", None)
    if isinstance(feature_dim, bool) or not isinstance(feature_dim, int) or feature_dim < 1:
        raise ValueError("dynamic geometry requires a positive model patch_feature_dim")
    prefixes = config.get("joint_base_parameter_prefixes", [])
    if not isinstance(prefixes, Sequence) or isinstance(prefixes, (str, bytes)):
        raise ValueError("joint_base_parameter_prefixes must be a sequence")
    model = DynamicGeometryTrainingModel(
        prepared.model,
        feature_dim=feature_dim,
        hidden_dim=int(config["hidden_dim"]),
        relative_camera_dim=int(config["relative_camera_dim"]),
        pair_chunk_size=int(config["pair_chunk_size"]),
        refinement_seed=int(config["refinement_seed"]),
        visibility_threshold=float(config["visibility_threshold"]),
        static_probability_max=float(config["static_probability_max"]),
        dynamic_probability_min=float(config["dynamic_probability_min"]),
        joint_base_parameter_prefixes=tuple(str(prefix) for prefix in prefixes),
    )
    if device is not None:
        model = model.to(device)
    model.train()
    trainable_names = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    if not trainable_names:
        raise RuntimeError("dynamic geometry produced no optimizer candidates")
    return PreparedTrainingModel(
        model=model,
        trainable_parameter_names=(*trainable_names, "dynamic_geometry_ready"),
    )


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


def _mapping(owner: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = owner.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"pixel-depth config {key!r} must be a mapping")
    return value
