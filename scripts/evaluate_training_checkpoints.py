"""Recompute validation metrics for every saved top-K VGGT-Omega checkpoint."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from vggt_omega.training.evaluation import evaluate_rgbd_conditioning, evaluate_training_checkpoints


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        choices=("legacy", "rgbd_paired_v1"),
        default="legacy",
        help="Evaluation protocol (default: legacy)",
    )
    parser.add_argument("--run-dir", required=True, help="Completed training run directory")
    parser.add_argument("--output", required=True, help="Atomic final_evaluation.json destination")
    parser.add_argument(
        "--original-cwd",
        required=True,
        help="Original project cwd used to resolve generic relative data/model paths",
    )
    parser.add_argument("--device", default="cuda", help="Evaluation device (default: cuda)")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Maximum absolute difference between stored and recomputed val/objective",
    )
    parser.add_argument(
        "--depth-threshold-m",
        action="append",
        type=float,
        default=None,
        help="Repeatable metric-depth upper bound in meters (default: 0.4, 0.8, 1.2)",
    )
    parser.add_argument("--checkpoint-limit", type=int, default=None)
    parser.add_argument("--depth-provided-frames", type=int, default=None)
    parser.add_argument("--skip-stored-monitor-check", action="store_true")
    parser.add_argument("--paired-baseline-run", default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.protocol == "rgbd_paired_v1":
        if args.paired_baseline_run is None:
            parser.error("rgbd_paired_v1 requires --paired-baseline-run")
        if args.eval_batch_size is None:
            parser.error("rgbd_paired_v1 requires --eval-batch-size")
        if args.depth_threshold_m is not None or args.depth_provided_frames is not None:
            parser.error("rgbd_paired_v1 fixes depth thresholds and all availability cases")
        if args.skip_stored_monitor_check:
            parser.error("rgbd_paired_v1 manages stored-monitor validation internally")
        report = evaluate_rgbd_conditioning(
            args.paired_baseline_run,
            args.run_dir,
            output_path=args.output,
            original_cwd=args.original_cwd,
            device=args.device,
            tolerance=args.tolerance,
            checkpoint_limit=args.checkpoint_limit or 3,
            evaluation_batch_size=args.eval_batch_size,
        )
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 0
    if args.paired_baseline_run is not None or args.eval_batch_size is not None:
        parser.error("paired-only arguments require --protocol rgbd_paired_v1")
    evaluation_options = {}
    if args.depth_threshold_m is not None:
        evaluation_options["depth_thresholds_m"] = tuple(args.depth_threshold_m)
    if args.checkpoint_limit is not None:
        evaluation_options["checkpoint_limit"] = args.checkpoint_limit
    if args.depth_provided_frames is not None:
        evaluation_options["depth_provided_frames"] = args.depth_provided_frames
    if args.skip_stored_monitor_check:
        evaluation_options["validate_stored_monitor"] = False
    report = evaluate_training_checkpoints(
        args.run_dir,
        output_path=args.output,
        original_cwd=args.original_cwd,
        device=args.device,
        tolerance=args.tolerance,
        **evaluation_options,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
