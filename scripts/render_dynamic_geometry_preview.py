"""Render privacy-minimal 4D diagnostics from a completed dynamic training run."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from vggt_omega.training.dataset import ColmapRgbdDataset
from vggt_omega.training.evaluation import (
    _attach_configured_training_wrappers,
    _validation_dataset,
)
from vggt_omega.training.model_factory import build_training_model
from vggt_omega.training.runner import (
    _dynamic_geometry_losses,
    _dynamic_geometry_runtime_options,
    _load_trainable_state,
    _move_batch,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--original-cwd", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--trajectory-stride", type=int, default=8)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _heatmap(values: np.ndarray) -> np.ndarray:
    finite = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    clipped = np.clip(finite, 0.0, 1.0)
    red = np.rint(clipped * 255).astype(np.uint8)
    blue = 255 - red
    return np.stack((red, np.zeros_like(red), blue), axis=-1)


def _normalized_magnitude(flow: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, float]:
    magnitude = np.linalg.norm(flow.astype(np.float32), axis=-1)
    finite = magnitude[valid & np.isfinite(magnitude)]
    scale = float(np.quantile(finite, 0.99)) if finite.size else 0.0
    normalized = np.zeros_like(magnitude, dtype=np.float32)
    if scale > 0:
        normalized[valid] = np.clip(magnitude[valid] / scale, 0.0, 1.0)
    return normalized, scale


def _atomic_json(payload: dict[str, Any], destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(destination)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.sample_index < 0 or args.trajectory_stride < 1:
        raise ValueError("sample-index must be non-negative and trajectory-stride must be positive")
    run_dir = Path(args.run_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    original_cwd = Path(args.original_cwd).expanduser().resolve()
    config = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    leaderboard = json.loads((run_dir / "checkpoints" / "leaderboard.json").read_text(encoding="utf-8"))
    entries = leaderboard.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("completed run has no best checkpoint")
    checkpoint_name = entries[0].get("filename")
    if not isinstance(checkpoint_name, str) or Path(checkpoint_name).name != checkpoint_name:
        raise ValueError("best checkpoint filename is invalid")
    checkpoint_path = run_dir / "checkpoints" / checkpoint_name
    model_config = cast(dict[str, Any], config["model"])
    base_value = model_config["pretrained_checkpoint"]
    if not isinstance(base_value, str) or Path(base_value).is_absolute():
        raise ValueError("preview requires a privacy-safe relative pretrained checkpoint")
    device = torch.device(args.device)
    prepared = build_training_model((original_cwd / base_value).resolve(), device=device)
    prepared, _, _ = _attach_configured_training_wrappers(prepared, config, device=device)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    _load_trainable_state(prepared.model, payload["model_state"], prepared.trainable_parameter_names)
    prepared.model.eval()

    dataset, _ = _validation_dataset(config, original_cwd, ColmapRgbdDataset)
    if args.sample_index >= len(dataset):
        raise IndexError("sample-index is outside the anonymous validation split")
    batch = next(iter(DataLoader(torch.utils.data.Subset(cast(Any, dataset), [args.sample_index]), batch_size=1)))
    device_batch = _move_batch(batch, device)
    images = cast(torch.Tensor, device_batch["images"])
    frame_ids = cast(torch.Tensor, device_batch["frame_ids"])
    frame_mask_value = device_batch.get("frame_mask")
    frame_mask = (
        torch.ones(frame_ids.shape, dtype=torch.bool, device=device)
        if frame_mask_value is None
        else cast(torch.Tensor, frame_mask_value).to(device=device, dtype=torch.bool)
    )
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        prediction = prepared.model.forward_dynamic(images, frame_ids=frame_ids, frame_mask=frame_mask)
    dynamic_options = _dynamic_geometry_runtime_options(cast(dict[str, Any], config["dynamic_geometry"]), epoch=3)
    if dynamic_options is None:
        raise ValueError("preview requires enabled dynamic geometry")
    with torch.no_grad():
        diagnostics = _dynamic_geometry_losses(prediction, device_batch, dynamic_options)

    pair_valid = prediction["motion_pair_valid_mask"][0].bool().cpu().numpy()
    valid_indices = np.flatnonzero(pair_valid)
    if valid_indices.size == 0:
        raise ValueError("selected anonymous sequence has no valid temporal pair")
    pair_index = int(valid_indices[0])
    domain = prediction["motion_domain_mask"][0, pair_index].bool().cpu().numpy()
    probability = prediction["dynamic_probability"][0, pair_index].float().cpu().numpy()
    public_mask = prediction["dynamic_mask"][0, pair_index].bool().cpu().numpy()
    unknown = prediction["dynamic_unknown_mask"][0, pair_index].bool().cpu().numpy()
    flow = prediction["canonical_scene_flow"][0, pair_index].float().cpu().numpy()
    magnitude, magnitude_scale = _normalized_magnitude(flow, domain)
    source_position = int(prediction["motion_pair_indices"][0, pair_index, 0])
    source_rgb = images[0, source_position].float().cpu().permute(1, 2, 0).numpy()
    source_rgb = np.rint(np.clip(source_rgb, 0, 1) * 255).astype(np.uint8)
    overlay = source_rgb.astype(np.float32)
    overlay[..., 0] = np.maximum(overlay[..., 0], probability * 255)
    overlay = np.rint(np.clip(overlay, 0, 255)).astype(np.uint8)

    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_heatmap(probability)).save(output_dir / "dynamic_probability.png")
    Image.fromarray(_heatmap(magnitude)).save(output_dir / "scene_flow_magnitude.png")
    Image.fromarray(public_mask.astype(np.uint8) * 255).save(output_dir / "dynamic_mask_public.png")
    Image.fromarray(unknown.astype(np.uint8) * 255).save(output_dir / "dynamic_unknown.png")
    Image.fromarray(overlay).save(output_dir / "dynamic_probability_overlay.png")

    current = prediction["canonical_points_current"][0, source_position].float().cpu().numpy()
    target = prediction["canonical_points_at_target_time"][0, pair_index].float().cpu().numpy()
    sampled = np.zeros_like(domain)
    sampled[:: args.trajectory_stride, :: args.trajectory_stride] = True
    sampled &= domain & np.isfinite(current).all(axis=-1) & np.isfinite(target).all(axis=-1)
    np.savez_compressed(
        output_dir / "trajectory.npz",
        current_points=current[sampled].astype(np.float32),
        target_points=target[sampled].astype(np.float32),
    )
    ready = bool(prediction["dynamic_geometry_ready"].item())
    _atomic_json(
        {
            "checkpoint_sha256": _sha256(checkpoint_path),
            "dynamic_geometry_ready": ready,
            "dynamic_f1": float(diagnostics["dynamic_f1"]),
            "dynamic_iou": float(diagnostics["dynamic_iou"]),
            "dynamic_probability_mean": float(probability[domain].mean()) if domain.any() else 0.0,
            "dynamic_precision": float(diagnostics["dynamic_precision"]),
            "dynamic_recall": float(diagnostics["dynamic_recall"]),
            "dynamic_static_false_positive_rate": float(diagnostics["dynamic_static_false_positive_rate"]),
            "format_version": 1,
            "motion_domain_pixels": int(domain.sum()),
            "public_dynamic_pixels": int(public_mask.sum()),
            "sample_index": args.sample_index,
            "scene_flow_magnitude_p99_scene_units": magnitude_scale,
            "scene_flow_epe_scene_units": float(diagnostics["dynamic_scene_flow_epe"]),
            "temporal_flicker_proxy": float(diagnostics["dynamic_temporal_mask"]),
            "trajectory_count": int(sampled.sum()),
            "unknown_pixels": int(unknown.sum()),
            "visibility_f1": float(diagnostics["dynamic_visibility_f1"]),
            "visibility_iou": float(diagnostics["dynamic_visibility_iou"]),
            "visibility_known_coverage": float(diagnostics["dynamic_visibility_known_coverage"]),
            "visibility_precision": float(diagnostics["dynamic_visibility_precision"]),
            "visibility_recall": float(diagnostics["dynamic_visibility_recall"]),
        },
        output_dir / "summary.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
