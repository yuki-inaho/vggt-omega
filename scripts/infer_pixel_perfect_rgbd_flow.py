#!/usr/bin/env python3
"""Export RGB-D and correspondence flow from a trained Pixel-Perfect wrapper."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from reconstruct_rgbd_video import (
    _chunk_windows,
    _depth_color,
    _label,
    _load_dataset_arrays,
    _metric_scale,
    _open_writer,
    _read_rgb_depth,
    _sha256,
)
from vggt_omega.training.correspondence import build_rgbd_correspondence_targets
from vggt_omega.training.evaluation import _attach_configured_training_wrappers
from vggt_omega.training.model_factory import build_training_model
from vggt_omega.training.runner import _load_trainable_state


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-windows", type=int, default=0, help="0 processes every trajectory chunk")
    parser.add_argument("--video-width", type=int, default=320)
    parser.add_argument("--video-height", type=int, default=240)
    parser.add_argument("--video-fps", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=1042)
    return parser


def _load_trained_model(
    run_dir: Path,
    checkpoint: Path,
    repo_root: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    config = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    base_value = cast(Mapping[str, Any], config["model"])["pretrained_checkpoint"]
    if not isinstance(base_value, str):
        raise ValueError("resolved model.pretrained_checkpoint must be a string")
    base_path = (repo_root / base_value).resolve() if not Path(base_value).is_absolute() else Path(base_value)
    prepared = build_training_model(base_path, device=device)
    prepared, pixel_config, pixel_enabled = _attach_configured_training_wrappers(
        prepared,
        config,
        device=device,
    )
    if not pixel_enabled or not isinstance(pixel_config, Mapping):
        raise ValueError("run does not contain an enabled Pixel-Perfect depth wrapper")
    self_supervised = pixel_config.get("self_supervised")
    correspondence = self_supervised.get("correspondence") if isinstance(self_supervised, Mapping) else None
    if not isinstance(correspondence, Mapping) or correspondence.get("enabled") is not True:
        raise ValueError("run does not contain an enabled correspondence-flow head")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, Mapping) or payload.get("format_version") != 1:
        raise ValueError("checkpoint does not follow the training artifact contract")
    if payload.get("config") != config:
        raise ValueError("checkpoint and run resolved configurations differ")
    model_state = payload.get("model_state")
    if not isinstance(model_state, Mapping):
        raise ValueError("checkpoint has no model_state")
    _load_trainable_state(
        prepared.model,
        cast(Mapping[str, torch.Tensor], model_state),
        prepared.trainable_parameter_names,
    )
    prepared.model.eval()
    checkpoint_info = {
        "epoch": int(payload["epoch"]),
        "global_step": int(payload["global_step"]),
        "kind": str(payload["kind"]),
        "sha256": _sha256(checkpoint),
    }
    training_state = payload.get("training_state")
    latest_validation = training_state.get("latest_validation") if isinstance(training_state, Mapping) else None
    if isinstance(latest_validation, Mapping):
        checkpoint_info["validation"] = {
            name: float(latest_validation[name])
            for name in (
                "correspondence_epe_px",
                "camera_translation",
                "near_depth_mae_m",
                "depth_all_mae_m",
                "objective",
            )
            if name in latest_validation
        }
    return prepared.model, config, checkpoint_info


def _checkpoint_validation_from_tensorboard(
    run_dir: Path,
    *,
    global_step: int,
) -> dict[str, float]:
    accumulator = EventAccumulator(
        str(run_dir / "tensorboard"),
        size_guidance={"scalars": 0},
    )
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", []))
    result: dict[str, float] = {}
    for name in (
        "correspondence_epe_px",
        "camera_translation",
        "near_depth_mae_m",
        "depth_all_mae_m",
        "objective",
    ):
        tag = f"val/{name}"
        if tag not in available:
            continue
        matches = [event for event in accumulator.Scalars(tag) if event.step == global_step]
        if matches:
            result[name] = float(matches[-1].value)
    return result


def _source_to_target_pair(prediction: Mapping[str, torch.Tensor]) -> tuple[int, torch.Tensor]:
    pair_indices = prediction.get("correspondence_pair_indices")
    flow = prediction.get("correspondence_flow_pixels")
    if not isinstance(pair_indices, torch.Tensor) or not isinstance(flow, torch.Tensor):
        raise ValueError("Pixel-Perfect inference did not return correspondence flow")
    pairs = pair_indices[0].detach().cpu().numpy()
    matches = np.flatnonzero((pairs[:, 0] == 0) & (pairs[:, 1] == 1))
    if len(matches) != 1:
        raise ValueError("inference did not contain exactly one frame-0 to frame-1 flow")
    return int(matches[0]), pair_indices[:, matches[:1]]


def _flow_resize(flow: np.ndarray, width: int, height: int) -> np.ndarray:
    source_height, source_width = flow.shape[:2]
    resized = cv2.resize(flow.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    resized[..., 0] *= width / source_width
    resized[..., 1] *= height / source_height
    return resized


def _mask_resize(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)


def _flow_color(flow: np.ndarray, scale_px: float, mask: np.ndarray | None = None) -> np.ndarray:
    finite = np.isfinite(flow).all(axis=-1)
    if mask is not None:
        finite &= mask
    magnitude = np.linalg.norm(np.nan_to_num(flow, nan=0.0), axis=-1)
    angle = np.arctan2(flow[..., 1], flow[..., 0])
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = np.rint((angle + math.pi) * (179.0 / (2 * math.pi))).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.rint(np.clip(magnitude / max(scale_px, 1e-6), 0, 1) * 255).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    bgr[~finite] = 0
    return bgr


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = _parser().parse_args()
    if args.max_windows < 0 or args.video_width < 16 or args.video_height < 16 or args.video_fps <= 0:
        raise ValueError("window limit, video dimensions, or FPS are invalid")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = args.run_dir.expanduser().resolve()
    checkpoint = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else run_dir / "checkpoints" / "last.pt"
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model, config, checkpoint_info = _load_trained_model(run_dir, checkpoint, repo_root, device)

    metadata, arrays = _load_dataset_arrays(args.data_root.expanduser().resolve())
    windows = _chunk_windows(arrays, 2)
    if args.max_windows:
        windows = windows[: args.max_windows]
    scene_root = Path(arrays["scene_root"])
    model_config = cast(Mapping[str, Any], config["model"])
    height = int(model_config["image_height"])
    width = int(model_config["image_width"])
    if (height, width) != (480, 640):
        raise ValueError("this output workflow expects the trained 640x480 profile")

    source_rgb_frames: list[np.ndarray] = []
    base_depth_frames: list[np.ndarray] = []
    refined_depth_frames: list[np.ndarray] = []
    measured_depth_frames: list[np.ndarray] = []
    predicted_flow_frames: list[np.ndarray] = []
    geometric_flow_frames: list[np.ndarray] = []
    residual_flow_frames: list[np.ndarray] = []
    teacher_flow_frames: list[np.ndarray] = []
    teacher_masks: list[np.ndarray] = []
    frame_pairs: list[list[int]] = []
    base_scales: list[float] = []
    refined_scales: list[float] = []
    epe_sum = 0.0
    geometric_epe_sum = 0.0
    epe_pixels = 0
    generator = torch.Generator(device=device).manual_seed(args.seed)
    started_at = time.perf_counter()

    for window_index, frame_ids in enumerate(windows):
        rgb_frames: list[np.ndarray] = []
        metric_depths: list[np.ndarray] = []
        for frame_id in frame_ids:
            rgb, depth = _read_rgb_depth(scene_root, int(frame_id), width, height)
            rgb_frames.append(rgb)
            metric_depths.append(depth)
        rgb_array = np.stack(rgb_frames)
        metric_depth = np.stack(metric_depths).astype(np.float32)
        images = torch.from_numpy(rgb_array).permute(0, 3, 1, 2).float().div_(255).unsqueeze(0).to(device)
        valid_mask = torch.from_numpy(metric_depth > 0).unsqueeze(0).to(device)
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            prediction = model.forward_refine(
                images,
                generator=generator,
                valid_mask=valid_mask,
            )
        pair_offset, requested_pair = _source_to_target_pair(prediction)
        base_depth = prediction["base_depth"][0, ..., 0].float().cpu().numpy()
        refined_depth = prediction["depth"][0, ..., 0].float().cpu().numpy()
        base_scale, _ = _metric_scale(base_depth, metric_depth)
        refined_scale, _ = _metric_scale(refined_depth, metric_depth)
        predicted_flow = prediction["correspondence_flow_pixels"][0, pair_offset].float().cpu().numpy()
        geometric_flow = prediction["correspondence_geometric_flow_pixels"][0, pair_offset].float().cpu().numpy()
        residual_flow = prediction["correspondence_residual_flow_pixels"][0, pair_offset].float().cpu().numpy()

        frame_positions = [int(np.flatnonzero(arrays["frame_ids"] == int(frame_id))[0]) for frame_id in frame_ids]
        intrinsics = torch.from_numpy(arrays["intrinsics"][frame_positions]).unsqueeze(0).to(device)
        extrinsics = torch.from_numpy(arrays["extrinsics_w2c"][frame_positions]).unsqueeze(0).to(device)
        depth_tensor = torch.from_numpy(metric_depth).unsqueeze(0).to(device)
        near_valid = valid_mask & (depth_tensor <= 1.2)
        teacher = build_rgbd_correspondence_targets(
            depth_tensor,
            intrinsics,
            extrinsics,
            requested_pair,
            valid_mask=near_valid,
            relative_depth_tolerance=0.03,
        )
        teacher_flow = teacher["flow_pixels"][0, 0].float().cpu().numpy()
        teacher_mask = teacher["covisibility_mask"][0, 0].cpu().numpy().astype(bool)
        finite_mask = teacher_mask & np.isfinite(predicted_flow).all(axis=-1) & np.isfinite(teacher_flow).all(axis=-1)
        if finite_mask.any():
            error = np.linalg.norm(predicted_flow[finite_mask] - teacher_flow[finite_mask], axis=-1)
            geometric_error = np.linalg.norm(geometric_flow[finite_mask] - teacher_flow[finite_mask], axis=-1)
            epe_sum += float(error.sum(dtype=np.float64))
            geometric_epe_sum += float(geometric_error.sum(dtype=np.float64))
            epe_pixels += int(error.size)

        source_rgb_frames.append(cv2.resize(rgb_array[0], (args.video_width, args.video_height), cv2.INTER_AREA))
        base_depth_frames.append(
            cv2.resize(base_depth[0] * base_scale, (args.video_width, args.video_height), cv2.INTER_LINEAR)
        )
        refined_depth_frames.append(
            cv2.resize(refined_depth[0] * refined_scale, (args.video_width, args.video_height), cv2.INTER_LINEAR)
        )
        measured_depth_frames.append(
            cv2.resize(metric_depth[0], (args.video_width, args.video_height), cv2.INTER_NEAREST)
        )
        predicted_flow_frames.append(_flow_resize(predicted_flow, args.video_width, args.video_height))
        geometric_flow_frames.append(_flow_resize(geometric_flow, args.video_width, args.video_height))
        residual_flow_frames.append(_flow_resize(residual_flow, args.video_width, args.video_height))
        teacher_flow_frames.append(_flow_resize(teacher_flow, args.video_width, args.video_height))
        teacher_masks.append(_mask_resize(teacher_mask, args.video_width, args.video_height))
        frame_pairs.append([int(frame_ids[0]), int(frame_ids[1])])
        base_scales.append(base_scale)
        refined_scales.append(refined_scale)
        if (window_index + 1) % 10 == 0 or window_index + 1 == len(windows):
            print(
                f"processed {window_index + 1}/{len(windows)} RGB-D pairs "
                f"({time.perf_counter() - started_at:.1f}s)",
                flush=True,
            )

    rgb_array = np.stack(source_rgb_frames)
    base_depth_array = np.stack(base_depth_frames).astype(np.float32)
    refined_depth_array = np.stack(refined_depth_frames).astype(np.float32)
    measured_depth_array = np.stack(measured_depth_frames).astype(np.float32)
    predicted_flow_array = np.stack(predicted_flow_frames).astype(np.float32)
    geometric_flow_array = np.stack(geometric_flow_frames).astype(np.float32)
    residual_flow_array = np.stack(residual_flow_frames).astype(np.float32)
    teacher_flow_array = np.stack(teacher_flow_frames).astype(np.float32)
    teacher_mask_array = np.stack(teacher_masks)
    flow_magnitudes = np.concatenate(
        (
            np.linalg.norm(predicted_flow_array, axis=-1).reshape(-1),
            np.linalg.norm(geometric_flow_array, axis=-1).reshape(-1),
            np.linalg.norm(teacher_flow_array[teacher_mask_array], axis=-1),
        )
    )
    finite_magnitudes = flow_magnitudes[np.isfinite(flow_magnitudes)]
    flow_scale = max(float(np.quantile(finite_magnitudes, 0.99)) if finite_magnitudes.size else 1.0, 1e-3)

    rgbd_writer = _open_writer(
        output_dir / "pixel_perfect_rgbd.mp4",
        args.video_width * 4,
        args.video_height,
        args.video_fps,
    )
    flow_writer = _open_writer(
        output_dir / "pixel_perfect_flow.mp4",
        args.video_width * 4,
        args.video_height,
        args.video_fps,
    )
    combined_writer = _open_writer(
        output_dir / "pixel_perfect_rgbd_flow.mp4",
        args.video_width * 7,
        args.video_height,
        args.video_fps,
    )
    first_combined: np.ndarray | None = None
    try:
        for index in range(len(frame_pairs)):
            rgb = _label(cv2.cvtColor(rgb_array[index], cv2.COLOR_RGB2BGR), "input RGB")
            base_depth = _label(_depth_color(base_depth_array[index]), "VGGT base depth [m]")
            refined_depth = _label(_depth_color(refined_depth_array[index]), "Pixel-Perfect depth [m]")
            measured_depth = _label(_depth_color(measured_depth_array[index]), "RGB-D measured depth [m]")
            geometric_flow = _label(_flow_color(geometric_flow_array[index], flow_scale), "geometry flow 0->1")
            learned_flow = _label(_flow_color(predicted_flow_array[index], flow_scale), "learned total flow 0->1")
            teacher_flow = _label(
                _flow_color(teacher_flow_array[index], flow_scale, teacher_mask_array[index]),
                "RGB-D teacher flow 0->1",
            )
            rgbd_writer.write(np.concatenate((rgb, base_depth, refined_depth, measured_depth), axis=1))
            flow_writer.write(np.concatenate((rgb, geometric_flow, learned_flow, teacher_flow), axis=1))
            combined = np.concatenate(
                (rgb, base_depth, refined_depth, measured_depth, geometric_flow, learned_flow, teacher_flow),
                axis=1,
            )
            combined_writer.write(combined)
            if first_combined is None:
                first_combined = combined.copy()
    finally:
        rgbd_writer.release()
        flow_writer.release()
        combined_writer.release()
    assert first_combined is not None
    cv2.imwrite(str(output_dir / "preview_rgbd_flow.png"), first_combined)

    np.savez_compressed(
        output_dir / "pixel_perfect_rgbd_flow_predictions.npz",
        frame_pairs=np.asarray(frame_pairs, dtype=np.int64),
        base_depth_m=base_depth_array.astype(np.float16),
        refined_depth_m=refined_depth_array.astype(np.float16),
        measured_depth_m=measured_depth_array.astype(np.float16),
        predicted_flow_xy=np.clip(predicted_flow_array, -65504, 65504).astype(np.float16),
        geometric_flow_xy=np.clip(geometric_flow_array, -65504, 65504).astype(np.float16),
        learned_residual_flow_xy=np.clip(residual_flow_array, -65504, 65504).astype(np.float16),
        rgbd_teacher_flow_xy=np.clip(teacher_flow_array, -65504, 65504).astype(np.float16),
        rgbd_teacher_covisibility=teacher_mask_array.astype(np.uint8),
    )
    run_summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    checkpoint_validation = checkpoint_info.get("validation", {})
    if not isinstance(checkpoint_validation, Mapping):
        checkpoint_validation = {}
    validation_source = "checkpoint_training_state"
    if not checkpoint_validation:
        checkpoint_validation = _checkpoint_validation_from_tensorboard(
            run_dir,
            global_step=int(checkpoint_info["global_step"]),
        )
        validation_source = "tensorboard_at_checkpoint_step"
    fallback_validation = cast(Mapping[str, Any], run_summary["validation"])
    if not checkpoint_validation:
        validation_source = "final_run_summary_fallback"

    def validation_metric(name: str) -> float:
        return float(checkpoint_validation.get(name, fallback_validation[name]))

    learned_epe = epe_sum / max(epe_pixels, 1)
    geometric_epe = geometric_epe_sum / max(epe_pixels, 1)
    summary = {
        "status": "passed",
        "format_version": 1,
        "checkpoint": checkpoint_info,
        "source_frame_count": int(metadata["frame_count"]),
        "processed_pair_count": len(frame_pairs),
        "model_grid_hw": [height, width],
        "saved_grid_hw": [args.video_height, args.video_width],
        "video_fps": float(args.video_fps),
        "metric_scale": {
            "base_median": float(np.median(base_scales)),
            "refined_median": float(np.median(refined_scales)),
            "refined_min": float(np.min(refined_scales)),
            "refined_max": float(np.max(refined_scales)),
        },
        "flow": {
            "coordinate_space": "pixel_displacement_xy",
            "visualization_p99_scale_px_at_saved_grid": flow_scale,
            "predicted_finite_fraction": float(np.isfinite(predicted_flow_array).mean()),
            "teacher_covisibility_fraction": float(teacher_mask_array.mean()),
            "rgbd_teacher_epe_px_at_model_grid": learned_epe,
            "geometric_rgbd_teacher_epe_px_at_model_grid": geometric_epe,
            "learned_epe_improvement_px": geometric_epe - learned_epe,
            "learned_epe_improvement_fraction": (
                (geometric_epe - learned_epe) / geometric_epe if geometric_epe > 0 else 0.0
            ),
            "rgbd_teacher_epe_pixel_count": epe_pixels,
        },
        "training_validation": {
            "source": validation_source,
            "near_depth_mae_m": validation_metric("near_depth_mae_m"),
            "all_depth_mae_m": validation_metric("depth_all_mae_m"),
            "camera_translation": validation_metric("camera_translation"),
            "correspondence_epe_px": validation_metric("correspondence_epe_px"),
            "objective": validation_metric("objective"),
            "max_cuda_memory_gib": float(run_summary["max_cuda_memory_gib"]),
            "global_step": int(checkpoint_info["global_step"]),
        },
        "outputs": {
            "rgbd_video": "pixel_perfect_rgbd.mp4",
            "flow_video": "pixel_perfect_flow.mp4",
            "combined_video": "pixel_perfect_rgbd_flow.mp4",
            "prediction_archive": "pixel_perfect_rgbd_flow_predictions.npz",
            "preview": "preview_rgbd_flow.png",
        },
    }
    _atomic_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
