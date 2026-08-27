#!/usr/bin/env python3
"""Find long RGB-D macro segments by scoring VGGT poses across timestamp gaps.

The input is first split into timestamp-contiguous bursts.  For every burst
boundary, a six-frame window (three frames on either side) is inferred once
with VGGT.  A conservative geometric/odometry score determines whether the
two bursts are candidates for the same longer reconstruction region.

Results are appended to JSONL after every boundary, so an interrupted GPU run
can resume without repeating completed inferences.
"""

from __future__ import annotations

import argparse
import csv
import json
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from run_vggt_rgbd_pose_workflow import estimate_scale, load_inputs
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_checkpoint_state_dict
from vggt_omega.utils.pose_enc import encoding_to_camera


@dataclass(frozen=True)
class Frame:
    session_id: str
    timestamp: float
    stem: str


@dataclass(frozen=True)
class Burst:
    identifier: str
    session_id: str
    frames: tuple[Frame, ...]


class DisjointSet:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--odometry-csv", type=Path, default=None)
    parser.add_argument("--camera-side", default="camera_l")
    parser.add_argument("--gap-threshold-sec", type=float, default=2.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--min-valid-pixels", type=int, default=100_000)
    parser.add_argument("--max-rotation-deg", type=float, default=5.0)
    parser.add_argument("--max-direction-change-deg", type=float, default=45.0)
    parser.add_argument("--min-linearity", type=float, default=0.95)
    parser.add_argument("--min-view-perpendicular-deg", type=float, default=65.0)
    parser.add_argument("--min-odometry-ratio", type=float, default=0.25)
    parser.add_argument("--max-odometry-ratio", type=float, default=4.0)
    parser.add_argument("--max-boundaries", type=int, default=None)
    args = parser.parse_args()
    if args.width % 16 or args.height % 16:
        raise ValueError("--width and --height must be divisible by 16")
    return args


def load_bursts(args: argparse.Namespace) -> list[Burst]:
    grouped: dict[str, list[Frame]] = defaultdict(list)
    with args.metadata_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["camera_side"] != args.camera_side:
                continue
            if row["has_rgb"] != "True" or row["has_depth"] != "True":
                continue
            stem = Path(row["rgb_path"]).name.removesuffix("_rgb.png")
            rgb = args.dataset_dir / "rgb" / f"{stem}_rgb.png"
            depth = args.dataset_dir / "mapped_depth_dense" / f"{stem}_depth.png"
            if rgb.is_file() and depth.is_file():
                grouped[row["session_id"]].append(Frame(row["session_id"], float(row["timestamp_sec"]), stem))
    bursts: list[Burst] = []
    for session_id, frames in sorted(grouped.items()):
        frames.sort(key=lambda frame: frame.timestamp)
        current = [frames[0]]
        number = 0
        for frame in frames[1:]:
            if frame.timestamp - current[-1].timestamp > args.gap_threshold_sec:
                bursts.append(Burst(f"{session_id}:burst:{number:04d}", session_id, tuple(current)))
                number += 1
                current = [frame]
            else:
                current.append(frame)
        bursts.append(Burst(f"{session_id}:burst:{number:04d}", session_id, tuple(current)))
    return bursts


def load_odometry(path: Path | None) -> dict[str, list[tuple[float, float]]]:
    result: dict[str, list[tuple[float, float]]] = defaultdict(list)
    if path is None:
        return result
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            result[row["session_id"]].append((float(row["timestamp_sec"]), float(row["odo"])))
    for values in result.values():
        values.sort()
    return result


def interpolated_odometry(rows: list[tuple[float, float]], timestamp: float) -> float | None:
    index = bisect_right([item[0] for item in rows], timestamp)
    if index == 0 or index == len(rows):
        return None
    left_time, left_value = rows[index - 1]
    right_time, right_value = rows[index]
    return left_value + (right_value - left_value) * (timestamp - left_time) / (right_time - left_time)


def homogeneous_camera_to_global(extrinsics: np.ndarray) -> np.ndarray:
    transforms = np.broadcast_to(np.eye(4, dtype=np.float64), (len(extrinsics), 4, 4)).copy()
    transforms[:, :3, :4] = extrinsics
    return np.linalg.inv(transforms)


def angle_degrees(left: np.ndarray, right: np.ndarray) -> float:
    left_norm, right_norm = np.linalg.norm(left), np.linalg.norm(right)
    if left_norm < 1e-8 or right_norm < 1e-8:
        return 0.0
    cosine = np.clip(np.dot(left, right) / (left_norm * right_norm), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def rotation_degrees(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def infer_boundary(
    model: VGGTOmega,
    frames: list[Frame],
    args: argparse.Namespace,
    odometry: dict[str, list[tuple[float, float]]],
) -> dict[str, Any]:
    pairs = [
        (
            args.dataset_dir / "rgb" / f"{frame.stem}_rgb.png",
            args.dataset_dir / "mapped_depth_dense" / f"{frame.stem}_depth.png",
        )
        for frame in frames
    ]
    images, metric_depth = load_inputs(pairs, args.width, args.height)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        prediction = model(images.to("cuda"))
        pose_encoding = prediction["pose_enc"].float().cpu()
        depth = prediction["depth"].float().cpu().numpy()[0, ..., 0]
        extrinsics, _ = encoding_to_camera(pose_encoding, (args.height, args.width))
    scale, valid_pixels = estimate_scale(depth, metric_depth, 0.10, 5.00)
    scaled = extrinsics.numpy()[0]
    scaled[:, :3, 3] *= scale
    poses = homogeneous_camera_to_global(scaled)
    centres = poses[:, :3, 3]
    central = centres[3] - centres[2]
    before = centres[2] - centres[0]
    after = centres[5] - centres[3]
    direction_reference = before + after
    direction_change = angle_degrees(central, direction_reference)
    relative_rotation = poses[3, :3, :3].T @ poses[2, :3, :3]
    central_rotation = rotation_degrees(relative_rotation)
    centred = centres - centres.mean(axis=0)
    singular_values, vectors = np.linalg.eigh(centred.T @ centred)
    travel = vectors[:, int(np.argmax(singular_values))]
    linearity = float(np.max(singular_values) / np.sum(singular_values)) if np.sum(singular_values) else 1.0
    views = np.einsum("nij,j->ni", poses[:, :3, :3], np.array([0.0, 0.0, 1.0]))
    view_angles = [angle_degrees(view, travel) for view in views]
    view_perpendicular = float(np.median([min(angle, 180.0 - angle) for angle in view_angles]))

    odo_delta = None
    odo_ratio = None
    rows = odometry.get(frames[0].session_id, [])
    if rows:
        left = interpolated_odometry(rows, frames[2].timestamp)
        right = interpolated_odometry(rows, frames[3].timestamp)
        if left is not None and right is not None:
            odo_delta = abs(right - left)
            if odo_delta > 0.01:
                odo_ratio = float(np.linalg.norm(central) / odo_delta)
    metric_ok = odo_ratio is None or args.min_odometry_ratio <= odo_ratio <= args.max_odometry_ratio
    passed = (
        valid_pixels >= args.min_valid_pixels
        and central_rotation <= args.max_rotation_deg
        and direction_change <= args.max_direction_change_deg
        and linearity >= args.min_linearity
        and view_perpendicular >= args.min_view_perpendicular_deg
        and metric_ok
    )
    return {
        "scale": scale,
        "valid_pixels": valid_pixels,
        "central_translation_m": float(np.linalg.norm(central)),
        "central_rotation_deg": central_rotation,
        "direction_change_deg": direction_change,
        "linearity": linearity,
        "view_perpendicular_deg": view_perpendicular,
        "odometry_delta": odo_delta,
        "odometry_ratio": odo_ratio,
        "pass": passed,
    }


def load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            completed.add(json.loads(line)["boundary_id"])
    return completed


def summarize(args: argparse.Namespace, bursts: list[Burst], results_path: Path) -> dict[str, Any]:
    records = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line]
    by_id = {record["boundary_id"]: record for record in records}
    dsu = DisjointSet([burst.identifier for burst in bursts])
    for index in range(len(bursts) - 1):
        left, right = bursts[index], bursts[index + 1]
        if left.session_id != right.session_id:
            continue
        record = by_id.get(f"{left.identifier}->{right.identifier}")
        if record and record.get("pass"):
            dsu.union(left.identifier, right.identifier)
    groups: dict[str, list[Burst]] = defaultdict(list)
    for burst in bursts:
        groups[dsu.find(burst.identifier)].append(burst)
    components = []
    for group in groups.values():
        frames = [frame for burst in group for frame in burst.frames]
        components.append(
            {
                "session_id": group[0].session_id,
                "burst_ids": [burst.identifier for burst in group],
                "frame_count": len(frames),
                "start_timestamp": min(frame.timestamp for frame in frames),
                "end_timestamp": max(frame.timestamp for frame in frames),
                "frame_stems": [frame.stem for frame in sorted(frames, key=lambda item: item.timestamp)],
            }
        )
    components.sort(key=lambda component: component["frame_count"], reverse=True)
    return {
        "burst_count": len(bursts),
        "boundary_count": len(records),
        "passed_boundary_count": sum(record.get("pass", False) for record in records),
        "components": components,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "boundary_results.jsonl"
    bursts = load_bursts(args)
    odometry = load_odometry(args.odometry_csv)
    completed = load_completed(results_path)
    boundaries = [
        (index, bursts[index], bursts[index + 1])
        for index in range(len(bursts) - 1)
        if bursts[index].session_id == bursts[index + 1].session_id
    ]
    pending = [item for item in boundaries if f"{item[1].identifier}->{item[2].identifier}" not in completed]
    if args.max_boundaries is not None:
        pending = pending[: args.max_boundaries]
    session_frames: dict[str, list[Frame]] = defaultdict(list)
    for burst in bursts:
        session_frames[burst.session_id].extend(burst.frames)
    frame_positions = {
        session_id: {frame.stem: index for index, frame in enumerate(frames)}
        for session_id, frames in session_frames.items()
    }
    print(f"bursts={len(bursts)} boundaries={len(boundaries)} pending={len(pending)}")
    model = VGGTOmega().eval().to("cuda")
    model.load_state_dict(load_checkpoint_state_dict(args.checkpoint))
    with results_path.open("a", encoding="utf-8") as handle:
        for count, (_index, left, right) in enumerate(pending, start=1):
            all_session_frames = session_frames[left.session_id]
            boundary_index = frame_positions[left.session_id][right.frames[0].stem]
            preceding = all_session_frames[max(0, boundary_index - 3) : boundary_index]
            following = all_session_frames[boundary_index : boundary_index + 3]
            boundary_id = f"{left.identifier}->{right.identifier}"
            record: dict[str, Any] = {
                "boundary_id": boundary_id,
                "left_burst": left.identifier,
                "right_burst": right.identifier,
                "left_frame_count": len(left.frames),
                "right_frame_count": len(right.frames),
                "gap_sec": right.frames[0].timestamp - left.frames[-1].timestamp,
                "frames": [frame.stem for frame in preceding + following],
            }
            if len(preceding) != 3 or len(following) != 3:
                record.update({"pass": False, "reason": "insufficient_boundary_context"})
            else:
                try:
                    record.update(infer_boundary(model, preceding + following, args, odometry))
                except Exception as error:  # preserve the scan and classify the failed boundary
                    record.update({"pass": False, "reason": f"{type(error).__name__}: {error}"})
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(f"{count}/{len(pending)} pass={record['pass']} {boundary_id}")
    summary = summarize(args, bursts, results_path)
    summary_path = args.output_dir / "macro_segments.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if summary["components"]:
        largest = summary["components"][0]
        with (args.output_dir / "largest_macro_segment_frames.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["frame_stem"])
            writer.writeheader()
            writer.writerows({"frame_stem": stem} for stem in largest["frame_stems"])
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "largest_frame_count": summary["components"][0]["frame_count"] if summary["components"] else 0,
                "component_count": len(summary["components"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
