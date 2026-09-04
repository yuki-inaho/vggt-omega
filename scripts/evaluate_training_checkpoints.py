"""Recompute validation metrics for every saved top-K VGGT-Omega checkpoint."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from vggt_omega.training.evaluation import evaluate_training_checkpoints


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    evaluation_options = {}
    if args.depth_threshold_m is not None:
        evaluation_options["depth_thresholds_m"] = tuple(args.depth_threshold_m)
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
