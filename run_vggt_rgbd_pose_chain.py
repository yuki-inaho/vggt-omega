#!/usr/bin/env python3
"""Infer a metric VGGT pose chain from overlapping RGB-D chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from run_vggt_rgbd_chunk_alignment import (
    AlignmentConfig,
    chunk_start_indices,
    collect_all_pairs,
    estimate_adjacent_chunk_transform,
    global_frame_poses,
    infer_chunk,
    summarize_edge_residuals,
)
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_checkpoint_state_dict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=6)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--min-depth-m", type=float, default=0.10)
    parser.add_argument("--max-depth-m", type=float, default=5.00)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    if args.chunk_size < 2 or not 0 < args.stride < args.chunk_size:
        raise ValueError("require chunk-size >= 2 and 0 < stride < chunk-size")
    if not 0 < args.min_depth_m < args.max_depth_m:
        raise ValueError("require 0 < min-depth-m < max-depth-m")

    config = AlignmentConfig(
        args.session_dir,
        args.checkpoint,
        args.output_dir,
        args.chunk_size,
        args.stride,
        None,
        args.width,
        args.height,
        args.min_depth_m,
        args.max_depth_m,
        0.005,
    )
    pairs = collect_all_pairs(args.session_dir)
    starts = chunk_start_indices(len(pairs), args.chunk_size, args.stride)
    if len(starts) < 2:
        raise RuntimeError("at least two chunks are required")

    model = VGGTOmega().eval().to("cuda")
    model.load_state_dict(load_checkpoint_state_dict(args.checkpoint))
    chunks = []
    for index, start in enumerate(starts):
        chunks.append(infer_chunk(model, pairs[start : start + args.chunk_size], start, config))
        print(f"chunk {index + 1}/{len(starts)} start={start}", flush=True)

    chunk_to_global = [np.eye(4, dtype=np.float64)]
    edges = []
    for index in range(1, len(chunks)):
        previous_to_current, metrics = estimate_adjacent_chunk_transform(chunks[index - 1], chunks[index])
        chunk_to_global.append(chunk_to_global[-1] @ np.linalg.inv(previous_to_current))
        edges.append(
            {
                "source_chunk": index - 1,
                "target_chunk": index,
                **metrics,
                "transform_target_from_source": previous_to_current.tolist(),
            }
        )
    stems, camera_to_global, observation_counts = global_frame_poses(chunks, chunk_to_global)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "vggt_rgbd_global_poses.npz",
        frame_stems=np.asarray(stems),
        camera_to_global=camera_to_global,
        chunk_to_global=np.stack(chunk_to_global),
        chunk_scales=np.asarray([chunk.scale for chunk in chunks]),
    )
    summary = {
        "frame_count": len(stems),
        "chunk_count": len(chunks),
        "chunk_size": args.chunk_size,
        "stride": args.stride,
        "metric_depth_range_m": [args.min_depth_m, args.max_depth_m],
        "frame_observation_counts": observation_counts,
        "edge_residual_summary": summarize_edge_residuals(edges),
        "edges": edges,
        "poses": str(args.output_dir / "vggt_rgbd_global_poses.npz"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ("frame_count", "chunk_count", "edge_residual_summary")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
