#!/usr/bin/env python3
"""Evaluate timestamp burst boundaries with six-frame VGGT RGB-D windows."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from run_vggt_rgbd_chunk_alignment import AlignmentConfig, infer_chunk, rotation_angle_deg
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_checkpoint_state_dict


@dataclass(frozen=True)
class Frame:
    session_id: str
    timestamp: float
    basename: str
    rgb: Path
    depth: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--odometry-csv", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera-side", default="camera_l")
    parser.add_argument("--burst-gap-s", type=float, default=2.0)
    parser.add_argument("--context-frames", type=int, default=3)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--min-valid-scale-pixels", type=int, default=100_000)
    parser.add_argument("--max-jump-ratio", type=float, default=2.5)
    parser.add_argument("--max-rotation-deg", type=float, default=10.0)
    parser.add_argument("--max-odometry-error-m", type=float, default=0.30)
    parser.add_argument("--max-boundaries", type=int, default=None)
    return parser.parse_args()


def load_frames(args: argparse.Namespace) -> list[Frame]:
    result = []
    with args.metadata_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["camera_side"] != args.camera_side or row["has_rgb"] != "True" or row["has_depth"] != "True":
                continue
            prefix = f"{row['session_id']}__{row['image_basename']}_{args.camera_side}"
            rgb = args.session_dir / "rgb" / f"{prefix}_rgb.png"
            depth = args.session_dir / "mapped_depth_dense" / f"{prefix}_depth.png"
            if rgb.is_file() and depth.is_file():
                result.append(Frame(row["session_id"], float(row["timestamp_sec"]), prefix, rgb, depth))
    return sorted(result, key=lambda frame: (frame.session_id, frame.timestamp))


def split_bursts(frames: list[Frame], threshold: float) -> list[list[Frame]]:
    bursts: list[list[Frame]] = []
    for frame in frames:
        if (
            not bursts
            or frame.session_id != bursts[-1][-1].session_id
            or frame.timestamp - bursts[-1][-1].timestamp > threshold
        ):
            bursts.append([frame])
        else:
            bursts[-1].append(frame)
    return bursts


def load_odometry(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    grouped: dict[str, list[tuple[float, float]]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                grouped.setdefault(row["session_id"], []).append((float(row["timestamp_sec"]), float(row["odo"])))
            except ValueError:
                continue
    return {
        session: (np.asarray([item[0] for item in sorted(values)]), np.asarray([item[1] for item in sorted(values)]))
        for session, values in grouped.items()
    }


def interpolate_odometry(data: dict[str, tuple[np.ndarray, np.ndarray]], session: str, timestamp: float) -> float:
    if session not in data:
        return math.nan
    times, values = data[session]
    if timestamp < times[0] or timestamp > times[-1]:
        return math.nan
    return float(np.interp(timestamp, times, values))


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    frames = load_frames(args)
    bursts = split_bursts(frames, args.burst_gap_s)
    boundaries = [
        (index, bursts[index], bursts[index + 1])
        for index in range(len(bursts) - 1)
        if bursts[index][-1].session_id == bursts[index + 1][0].session_id
    ]
    if args.max_boundaries is not None:
        boundaries = boundaries[: args.max_boundaries]
    odometry = load_odometry(args.odometry_csv)
    result_path = args.output_dir / "boundary_results.json"
    completed = {}
    if result_path.is_file():
        completed = {item["boundary_id"]: item for item in json.loads(result_path.read_text())}

    model = VGGTOmega().eval().to("cuda")
    model.load_state_dict(load_checkpoint_state_dict(args.checkpoint))
    config = AlignmentConfig(
        args.session_dir,
        args.checkpoint,
        args.output_dir,
        2 * args.context_frames,
        args.context_frames,
        None,
        args.width,
        args.height,
        0.10,
        5.00,
        0.005,
    )

    for number, (burst_index, previous, current) in enumerate(boundaries, 1):
        boundary_id = f"{previous[-1].session_id}:{burst_index:04d}"
        if boundary_id in completed:
            continue
        before = previous[-args.context_frames :]
        after = current[: args.context_frames]
        selected = before + after
        if len(before) < 2 or len(after) < 2:
            completed[boundary_id] = {
                "boundary_id": boundary_id,
                "accepted": False,
                "reason": "insufficient_context",
                "previous_burst_frames": len(previous),
                "next_burst_frames": len(current),
            }
            atomic_json(result_path, list(completed.values()))
            continue
        pairs = [(frame.rgb, frame.depth) for frame in selected]
        chunk = infer_chunk(model, pairs, 0, config)
        poses = chunk.camera_to_local_world
        centres = poses[:, :3, 3]
        steps = np.linalg.norm(np.diff(centres, axis=0), axis=1)
        boundary_step_index = len(before) - 1
        jump = float(steps[boundary_step_index])
        within = np.delete(steps, boundary_step_index)
        within_median = float(np.median(within))
        jump_ratio = jump / within_median if within_median > 1e-8 else math.inf
        relative_rotation = poses[boundary_step_index + 1, :3, :3].T @ poses[boundary_step_index, :3, :3]
        rotation_deg = rotation_angle_deg(relative_rotation)
        odo_before = interpolate_odometry(odometry, previous[-1].session_id, previous[-1].timestamp)
        odo_after = interpolate_odometry(odometry, current[0].session_id, current[0].timestamp)
        odo_delta = abs(odo_after - odo_before) if np.isfinite([odo_before, odo_after]).all() else math.nan
        odo_error = abs(jump - odo_delta) if math.isfinite(odo_delta) else math.nan
        accepted = (
            chunk.scale_valid_pixels >= args.min_valid_scale_pixels
            and jump_ratio <= args.max_jump_ratio
            and rotation_deg <= args.max_rotation_deg
            and (not math.isfinite(odo_error) or odo_error <= args.max_odometry_error_m)
        )
        completed[boundary_id] = {
            "boundary_id": boundary_id,
            "session_id": previous[-1].session_id,
            "previous_burst_index": burst_index,
            "next_burst_index": burst_index + 1,
            "previous_burst_frames": len(previous),
            "next_burst_frames": len(current),
            "previous_frame": previous[-1].basename,
            "next_frame": current[0].basename,
            "gap_s": current[0].timestamp - previous[-1].timestamp,
            "window_frames": [frame.basename for frame in selected],
            "rgbd_scale": chunk.scale,
            "scale_valid_pixels": chunk.scale_valid_pixels,
            "vggt_jump_m": jump,
            "within_step_median_m": within_median,
            "jump_ratio": jump_ratio,
            "boundary_rotation_deg": rotation_deg,
            "odometry_delta_m": odo_delta,
            "odometry_error_m": odo_error,
            "accepted": accepted,
        }
        atomic_json(result_path, list(completed.values()))
        print(
            f"boundary {number}/{len(boundaries)} accepted={accepted} gap={completed[boundary_id]['gap_s']:.3f}s ratio={jump_ratio:.3f}",
            flush=True,
        )

    results = list(completed.values())
    accepted_by_index = {item["previous_burst_index"] for item in results if item.get("accepted")}
    macro_segments = []
    start = 0
    for index in range(len(bursts) - 1):
        same_session = bursts[index][-1].session_id == bursts[index + 1][0].session_id
        if not same_session or index not in accepted_by_index:
            segment_bursts = bursts[start : index + 1]
            macro_segments.append(segment_bursts)
            start = index + 1
    macro_segments.append(bursts[start:])
    summary_segments = []
    for index, segment in enumerate(macro_segments):
        flat = [frame for burst in segment for frame in burst]
        summary_segments.append(
            {
                "macro_segment_id": index,
                "session_id": flat[0].session_id,
                "frame_count": len(flat),
                "burst_count": len(segment),
                "first_frame": flat[0].basename,
                "last_frame": flat[-1].basename,
                "duration_s": flat[-1].timestamp - flat[0].timestamp,
                "frames": [frame.basename for frame in flat],
            }
        )
    summary_segments.sort(key=lambda item: item["frame_count"], reverse=True)
    atomic_json(args.output_dir / "macro_segments.json", summary_segments)
    atomic_json(
        args.output_dir / "summary.json",
        {
            "frame_count": len(frames),
            "burst_count": len(bursts),
            "evaluated_boundaries": len(results),
            "accepted_boundaries": sum(bool(item.get("accepted")) for item in results),
            "macro_segment_count": len(summary_segments),
            "largest_macro_segment": summary_segments[0],
        },
    )
    print(
        json.dumps(
            {
                "largest_frame_count": summary_segments[0]["frame_count"],
                "accepted_boundaries": sum(bool(item.get("accepted")) for item in results),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
