from __future__ import annotations

import pytest

from vggt_omega.training.comparison import compare_metric_summaries


def test_metric_comparison_has_explicit_before_after_delta_and_direction() -> None:
    rows = compare_metric_summaries(
        {"near_mae": 0.05, "coverage": 0.6},
        {"near_mae": 0.04, "coverage": 0.7},
        modes={"near_mae": "min", "coverage": "max"},
    )

    assert rows == [
        {
            "metric": "coverage",
            "baseline": 0.6,
            "candidate": 0.7,
            "delta": pytest.approx(0.1),
            "improved": True,
        },
        {
            "metric": "near_mae",
            "baseline": 0.05,
            "candidate": 0.04,
            "delta": pytest.approx(-0.01),
            "improved": True,
        },
    ]


def test_metric_comparison_rejects_missing_and_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="both"):
        compare_metric_summaries({}, {"loss": 1.0}, modes={"loss": "min"})
    with pytest.raises(ValueError, match="finite"):
        compare_metric_summaries({"loss": 1.0}, {"loss": float("nan")}, modes={"loss": "min"})
