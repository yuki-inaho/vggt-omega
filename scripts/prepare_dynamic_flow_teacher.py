#!/usr/bin/env python3
"""Prepare anonymous bidirectional OpenCV DIS flow-teacher artifacts."""

from __future__ import annotations

import argparse
import json
import os
from itertools import pairwise
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class FlowTeacherError(ValueError):
    """Raised when anonymous input or generated teacher data is invalid."""


MANIFEST = {
    "coordinate_space": "pixel_displacement_xy",
    "file_template": "scenes/{scene_id}/flow_teacher/pair_{source_id:06d}_{target_id:06d}.npz",
    "flow_dtype": "float32",
    "format": "dynamic_flow_teacher_v1",
    "occlusion_dtype": "int8_tri_state",
    "schema_version": 1,
}


def _sample(array: np.ndarray, flow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = flow.shape[:2]
    columns, rows = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    map_x = columns + flow[..., 0]
    map_y = rows + flow[..., 1]
    sampled = cv2.remap(array, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    in_bounds = (map_x >= 0) & (map_x <= width - 1) & (map_y >= 0) & (map_y <= height - 1)
    return sampled, in_bounds


def _directed_confidence(
    source: np.ndarray,
    target: np.ndarray,
    flow: np.ndarray,
    reverse: np.ndarray,
    *,
    photo_sigma: float,
    texture_scale: float,
    coherence_sigma: float,
    cycle_threshold: float,
    max_flow_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    source_float = source.astype(np.float32) / 255.0
    target_float = target.astype(np.float32) / 255.0
    gray = cv2.cvtColor(source_float, cv2.COLOR_RGB2GRAY)
    gradient = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    texture = np.clip(gradient / texture_scale, 0.0, 1.0)
    smoothed = cv2.GaussianBlur(flow, (3, 3), 0)
    coherence = np.exp(-np.linalg.norm(flow - smoothed, axis=-1) / coherence_sigma)
    warped_target, in_bounds = _sample(target_float, flow)
    photo = np.exp(-np.mean(np.abs(source_float - warped_target), axis=-1) / photo_sigma)
    sampled_reverse, reverse_in_bounds = _sample(reverse, flow)
    cycle_error = np.linalg.norm(flow + sampled_reverse, axis=-1)
    cycle = np.exp(-cycle_error / max(cycle_threshold, 1e-6))
    height, width = flow.shape[:2]
    magnitude_valid = np.linalg.norm(flow, axis=-1) <= max(height, width) * max_flow_fraction
    finite_flow = np.isfinite(flow).all(axis=-1)
    valid = in_bounds & reverse_in_bounds & magnitude_valid & finite_flow
    occlusion = np.full((height, width), -1, dtype=np.int8)
    occlusion[~in_bounds & magnitude_valid & finite_flow] = 0
    occlusion[valid & (cycle_error <= cycle_threshold)] = 1
    occlusion[valid & (cycle_error > cycle_threshold)] = 0
    visible_confidence = texture * coherence * photo * cycle
    occlusion_confidence = texture * coherence
    confidence = np.where(
        occlusion == 1,
        visible_confidence,
        np.where(occlusion == 0, occlusion_confidence, 0.0),
    ).astype(np.float32)
    return confidence, occlusion


def compute_bidirectional_dis_teacher(
    first_rgb: np.ndarray,
    second_rgb: np.ndarray,
    *,
    photo_sigma: float = 0.10,
    texture_scale: float = 0.05,
    coherence_sigma: float = 1.0,
    cycle_threshold: float = 1.0,
    max_flow_fraction: float = 0.5,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return deterministic forward/reverse flow teacher payloads."""

    if first_rgb.shape != second_rgb.shape or first_rgb.ndim != 3 or first_rgb.shape[-1] != 3:
        raise FlowTeacherError("RGB inputs must have matching HxWx3 shapes")
    if first_rgb.dtype != np.uint8 or second_rgb.dtype != np.uint8:
        raise FlowTeacherError("RGB inputs must be uint8")
    if min(photo_sigma, texture_scale, coherence_sigma, cycle_threshold, max_flow_fraction) <= 0:
        raise FlowTeacherError("teacher thresholds must be positive")
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    first_gray = cv2.cvtColor(first_rgb, cv2.COLOR_RGB2GRAY)
    second_gray = cv2.cvtColor(second_rgb, cv2.COLOR_RGB2GRAY)
    forward = dis.calc(first_gray, second_gray, None).astype(np.float32)
    reverse = dis.calc(second_gray, first_gray, None).astype(np.float32)
    forward_confidence, forward_occlusion = _directed_confidence(
        first_rgb,
        second_rgb,
        forward,
        reverse,
        photo_sigma=photo_sigma,
        texture_scale=texture_scale,
        coherence_sigma=coherence_sigma,
        cycle_threshold=cycle_threshold,
        max_flow_fraction=max_flow_fraction,
    )
    reverse_confidence, reverse_occlusion = _directed_confidence(
        second_rgb,
        first_rgb,
        reverse,
        forward,
        photo_sigma=photo_sigma,
        texture_scale=texture_scale,
        coherence_sigma=coherence_sigma,
        cycle_threshold=cycle_threshold,
        max_flow_fraction=max_flow_fraction,
    )
    return (
        {"flow_xy": forward, "confidence": forward_confidence, "occlusion_label": forward_occlusion},
        {"flow_xy": reverse, "confidence": reverse_confidence, "occlusion_label": reverse_occlusion},
    )


def _required_pairs(root: Path, split: str) -> list[tuple[int, int]]:
    try:
        entries = [line.strip() for line in (root / "splits" / f"{split}.txt").read_text().splitlines() if line.strip()]
        with np.load(root / "scenes/scene_000000/sequences.npz", allow_pickle=False) as payload:
            sequences = payload["sequences"]
            lengths = payload["lengths"]
    except (OSError, ValueError, KeyError) as error:
        raise FlowTeacherError("anonymous split or sequence arrays are invalid") from error
    result: set[tuple[int, int]] = set()
    for entry in entries:
        sequence_id = int(entry.rsplit("_", 1)[1])
        frames = sorted(int(value) for value in sequences[sequence_id, : int(lengths[sequence_id])])
        result.update(pairwise(frames))
    return sorted(result)


def prepare_flow_teacher(root: Path, *, split: str, overwrite: bool = False, **options: Any) -> dict[str, Any]:
    root = Path(root)
    try:
        metadata = json.loads((root / "dataset.json").read_text())
    except (OSError, ValueError) as error:
        raise FlowTeacherError("dataset.json is missing or invalid") from error
    if metadata.get("format") != "colmap_rgbd_v2":
        raise FlowTeacherError("flow teacher requires anonymous staging schema v2")
    pairs = _required_pairs(root, split)
    if not pairs:
        raise FlowTeacherError("selected split contains no temporal pair")
    output_root = root / "scenes/scene_000000/flow_teacher"
    output_root.mkdir(parents=True, exist_ok=True)
    completed = 0
    for first_id, second_id in pairs:
        paths = (
            output_root / f"pair_{first_id:06d}_{second_id:06d}.npz",
            output_root / f"pair_{second_id:06d}_{first_id:06d}.npz",
        )
        if not overwrite and all(path.is_file() for path in paths):
            completed += 1
            continue
        first = cv2.cvtColor(
            cv2.imread(str(root / f"scenes/scene_000000/rgb/frame_{first_id:06d}.png")),
            cv2.COLOR_BGR2RGB,
        )
        second = cv2.cvtColor(
            cv2.imread(str(root / f"scenes/scene_000000/rgb/frame_{second_id:06d}.png")),
            cv2.COLOR_BGR2RGB,
        )
        forward, reverse = compute_bidirectional_dis_teacher(first, second, **options)
        for path, payload in zip(paths, (forward, reverse), strict=True):
            temporary = path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, **payload)
            os.replace(temporary, path)
        completed += 1
    manifest_root = root / "flow_teacher"
    manifest_root.mkdir(parents=True, exist_ok=True)
    (manifest_root / "manifest.json").write_text(json.dumps(MANIFEST, indent=2, sort_keys=True) + "\n")
    return {"format": MANIFEST["format"], "split": split, "pair_count": len(pairs), "completed": completed}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "smoke"), default="smoke")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        report = prepare_flow_teacher(args.root, split=args.split, overwrite=args.overwrite)
    except (FlowTeacherError, OSError, ValueError) as error:
        print(f"error: {error}")
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
