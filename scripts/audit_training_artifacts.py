"""Audit VGGT-Omega training outputs without exposing source identifiers."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from vggt_omega.training.audit import audit_training_artifacts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Completed training run directory")
    parser.add_argument("--report", required=True, help="Atomic JSON report destination")
    parser.add_argument(
        "--deny-token",
        action="append",
        default=[],
        help="Private token to detect without reproducing it in the report; repeat as needed",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = audit_training_artifacts(args.run_dir, report_path=args.report, deny_tokens=args.deny_token)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
