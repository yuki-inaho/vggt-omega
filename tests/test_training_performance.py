import time
from pathlib import Path

import pytest
import torch
from radon.complexity import cc_visit
from radon.metrics import mi_visit

from vggt_omega.training.performance import StepProfiler, validate_training_batch_contract


def _batch() -> dict[str, torch.Tensor]:
    return {
        "images": torch.rand(2, 4, 3, 16, 24),
        "depths": torch.rand(2, 4, 16, 24),
        "depth_masks": torch.ones(2, 4, 16, 24, dtype=torch.bool),
        "intrinsics": torch.eye(3).reshape(1, 1, 3, 3).repeat(2, 4, 1, 1),
        "extrinsics": torch.eye(4)[:3].reshape(1, 1, 3, 4).repeat(2, 4, 1, 1),
    }


def test_training_batch_contract_accepts_shared_axes_and_rejects_mismatch() -> None:
    validate_training_batch_contract(_batch())
    invalid = _batch()
    invalid["depths"] = invalid["depths"][:, :3]

    with pytest.raises(Exception, match=r"Type-check error|Expected type"):
        validate_training_batch_contract(invalid)


def test_step_profiler_separates_warmup_and_reports_finite_anonymous_scalars() -> None:
    profiler = StepProfiler(enabled=True, warmup_steps=1, active_steps=2, synchronize=lambda: None)
    profiler.batch_ready(sample_count=2)
    time.sleep(0.001)
    profiler.batch_complete()
    for _ in range(2):
        profiler.batch_ready(sample_count=2)
        time.sleep(0.001)
        profiler.batch_complete()

    metrics = profiler.metrics()

    assert metrics["profile_warmup_seconds"] > 0
    assert metrics["profile_step_time_seconds"] > 0
    assert metrics["profile_samples_per_second"] > 0
    assert 0 <= metrics["profile_data_wait_fraction"] <= 1
    assert metrics["profile_active_steps"] == 2


def test_disabled_step_profiler_has_no_metrics() -> None:
    profiler = StepProfiler(enabled=False, warmup_steps=1, active_steps=2, synchronize=lambda: None)
    profiler.batch_ready(sample_count=2)
    profiler.batch_complete()

    assert profiler.metrics() == {}


def test_performance_module_meets_complexity_and_maintainability_budget() -> None:
    source_path = Path(__file__).parents[1] / "vggt_omega" / "training" / "performance.py"
    source = source_path.read_text(encoding="utf-8")

    offenders = [(block.name, block.complexity) for block in cc_visit(source) if block.complexity > 10]

    assert offenders == []
    assert mi_visit(source, multi=True) >= 20.0
