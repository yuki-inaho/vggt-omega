"""Private-safe metric comparison rows for local training reports."""

from __future__ import annotations

import math
from collections.abc import Mapping


def compare_metric_summaries(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    *,
    modes: Mapping[str, str],
) -> list[dict[str, float | bool | str]]:
    """Return deterministic baseline/candidate rows without run identifiers."""

    if set(modes) - (set(baseline) & set(candidate)):
        raise ValueError("every comparison mode must exist in both metric summaries")
    rows: list[dict[str, float | bool | str]] = []
    for metric in sorted(modes):
        mode = modes[metric]
        if mode not in {"min", "max"}:
            raise ValueError(f"comparison mode for {metric} must be min or max")
        before = float(baseline[metric])
        after = float(candidate[metric])
        if not math.isfinite(before) or not math.isfinite(after):
            raise ValueError(f"comparison metric {metric} must be finite")
        delta = after - before
        rows.append(
            {
                "metric": metric,
                "baseline": before,
                "candidate": after,
                "delta": delta,
                "improved": delta < 0 if mode == "min" else delta > 0,
            }
        )
    return rows
