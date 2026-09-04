"""Summarize anonymous RGB-D overlap profiles without recording input paths."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

SPLIT_NAMES = ("train", "val", "smoke")
QUANTILES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)


class OverlapReportError(ValueError):
    """Raised when an anonymous overlap profile violates its numeric contract."""


def _distribution(values: np.ndarray) -> dict[str, Any]:
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise OverlapReportError("distribution values must be non-empty and finite")
    quantile_values = np.quantile(values, QUANTILES)
    return {
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "standard_deviation": float(values.std()),
        "quantiles": {f"q{int(q * 100):02d}": float(value) for q, value in zip(QUANTILES, quantile_values)},
    }


def _correlation(first: np.ndarray, second: np.ndarray) -> dict[str, float | int]:
    if len(first) != len(second) or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise OverlapReportError("correlation inputs must have matching finite values")
    correlation = 0.0
    if len(first) >= 2 and float(first.std()) > 0 and float(second.std()) > 0:
        correlation = float(np.corrcoef(first, second)[0, 1])
    return {"pearson": correlation, "sample_count": int(len(first))}


def _camera_centers(extrinsics_w2c: np.ndarray) -> np.ndarray:
    rotations = extrinsics_w2c[:, :3, :3]
    translations = extrinsics_w2c[:, :3, 3]
    return -np.einsum("nij,nj->ni", rotations.transpose(0, 2, 1), translations)


def _rotation_degrees(first: np.ndarray, second: np.ndarray) -> float:
    relative = first @ second.T
    cosine = float(np.clip((np.trace(relative) - 1) / 2, -1, 1))
    return math.degrees(math.acos(cosine))


def summarize_overlap_distribution(
    sequences: np.ndarray,
    lengths: np.ndarray,
    split_ids: np.ndarray,
    all_depth: np.ndarray,
    near_depth: np.ndarray,
    extrinsics_w2c: np.ndarray,
) -> dict[str, Any]:
    """Return a finite JSON-compatible split report from numeric-only arrays."""

    sequences = np.asarray(sequences)
    lengths = np.asarray(lengths)
    split_ids = np.asarray(split_ids)
    all_depth = np.asarray(all_depth)
    near_depth = np.asarray(near_depth)
    extrinsics_w2c = np.asarray(extrinsics_w2c)
    if sequences.ndim != 2:
        raise OverlapReportError("sequences must be rank two")
    sequence_count, max_frames = sequences.shape
    expected_profile_shape = (sequence_count, max_frames, max_frames)
    if lengths.shape != (sequence_count,) or split_ids.shape != (sequence_count,):
        raise OverlapReportError("sequence metadata shapes are inconsistent")
    if all_depth.shape != expected_profile_shape or near_depth.shape != expected_profile_shape:
        raise OverlapReportError("overlap profile shapes are inconsistent")
    if not np.isfinite(all_depth).all() or not np.isfinite(near_depth).all():
        raise OverlapReportError("overlap scores must be finite")
    if ((all_depth < 0) | (all_depth > 1)).any() or ((near_depth < 0) | (near_depth > 1)).any():
        raise OverlapReportError("overlap scores must be within [0, 1]")
    if extrinsics_w2c.ndim != 3 or extrinsics_w2c.shape[1:] != (3, 4) or not np.isfinite(extrinsics_w2c).all():
        raise OverlapReportError("extrinsics must be finite Nx3x4 values")
    if not np.isin(split_ids, (0, 1, 2)).all():
        raise OverlapReportError("split IDs must be train, val, or smoke")

    centers = _camera_centers(extrinsics_w2c)
    pair_records: list[dict[tuple[int, int], tuple[float, float, int, float, float]]] = [{}, {}, {}]
    sequence_counts = [0, 0, 0]
    for sequence_id, (row, raw_length, raw_split) in enumerate(zip(sequences, lengths, split_ids, strict=True)):
        length = int(raw_length)
        split = int(raw_split)
        if not 2 <= length <= max_frames:
            raise OverlapReportError("sequence length is outside the profile shape")
        active = row[:length]
        if (active < 0).any() or (active >= len(extrinsics_w2c)).any() or len(set(active.tolist())) != length:
            raise OverlapReportError("sequence contains invalid generic frame IDs")
        sequence_counts[split] += 1
        for first_offset, second_offset in combinations(range(length), 2):
            first_id = int(active[first_offset])
            second_id = int(active[second_offset])
            key = (min(first_id, second_id), max(first_id, second_id))
            values = (
                float(all_depth[sequence_id, first_offset, second_offset]),
                float(near_depth[sequence_id, first_offset, second_offset]),
                abs(first_id - second_id),
                float(np.linalg.norm(centers[first_id] - centers[second_id])),
                _rotation_degrees(extrinsics_w2c[first_id, :3, :3], extrinsics_w2c[second_id, :3, :3]),
            )
            previous = pair_records[split].get(key)
            if previous is not None and not np.allclose(previous, values, atol=1e-5):
                raise OverlapReportError("repeated generic frame pair has inconsistent profile values")
            pair_records[split][key] = values
    if any(not records for records in pair_records):
        raise OverlapReportError("every split must contain at least one unique pair")

    pair_sets = [set(records) for records in pair_records]
    pair_leakage = sum(len(pair_sets[a] & pair_sets[b]) for a, b in combinations(range(3), 2))
    train_near = np.asarray([record[1] for record in pair_records[0].values()], dtype=np.float64)
    low_boundary, end_target, start_target = (float(value) for value in np.quantile(train_near, (0.25, 0.5, 0.75)))
    split_reports: dict[str, Any] = {}
    for split, split_name in enumerate(SPLIT_NAMES):
        records = list(pair_records[split].values())
        columns = [np.asarray(values, dtype=np.float64) for values in zip(*records, strict=True)]
        overlap_all, overlap_near, frame_interval, translation, rotation = columns
        bucket_counts = {
            "low": int(np.sum(overlap_near < low_boundary)),
            "medium": int(np.sum((overlap_near >= low_boundary) & (overlap_near < start_target))),
            "high": int(np.sum(overlap_near >= start_target)),
        }
        split_reports[split_name] = {
            "sequence_count": sequence_counts[split],
            "pair_count": len(records),
            "all_depth": _distribution(overlap_all),
            "near_depth": _distribution(overlap_near),
            "frame_interval": _distribution(frame_interval),
            "translation_m": _distribution(translation),
            "rotation_degrees": _distribution(rotation),
            "near_depth_buckets": bucket_counts,
            "correlations": {
                "near_vs_frame_interval": _correlation(overlap_near, frame_interval),
                "near_vs_translation": _correlation(overlap_near, translation),
                "near_vs_rotation": _correlation(overlap_near, rotation),
                "all_vs_near": _correlation(overlap_all, overlap_near),
            },
        }
    return {
        "schema_version": 1,
        "pair_count": sum(len(records) for records in pair_records),
        "pair_leakage_count": pair_leakage,
        "recommended_curriculum": {
            "source": "train_near_depth_quantiles",
            "low_boundary": low_boundary,
            "end_target": end_target,
            "start_target": start_target,
            "fallback": "nearest_score",
        },
        "splits": split_reports,
    }


def build_overlap_distribution_report(staging_root: Path) -> dict[str, Any]:
    """Load one anonymous staging set and return its path-free report."""

    scene_root = Path(staging_root) / "scenes/scene_000000"
    try:
        with np.load(scene_root / "sequences.npz", allow_pickle=False) as sequences:
            rows = sequences["sequences"].copy()
            lengths = sequences["lengths"].copy()
            split_ids = sequences["split_ids"].copy()
        with np.load(scene_root / "overlap.npz", allow_pickle=False) as overlap:
            all_depth = overlap["all_depth"].copy()
            near_depth = overlap["near_depth"].copy()
        with np.load(scene_root / "cameras.npz", allow_pickle=False) as cameras:
            extrinsics = cameras["extrinsics_w2c"].copy()
    except (OSError, ValueError, KeyError) as error:
        raise OverlapReportError("anonymous overlap staging arrays are missing or invalid") from error
    return summarize_overlap_distribution(rows, lengths, split_ids, all_depth, near_depth, extrinsics)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = build_overlap_distribution_report(args.staging_root)
        serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    except (OSError, OverlapReportError, ValueError) as error:
        print(f"error: {error}")
        return 2
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
