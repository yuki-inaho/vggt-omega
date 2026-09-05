from __future__ import annotations

from collections import Counter

import pytest
import torch

from vggt_omega.training.depth_input_evaluation import (
    all_depth_availability_cases,
    build_input_depth_holdout,
    depth_sufficient_statistics,
    merge_depth_statistics,
    metric_result,
    paired_depth_statistics,
)
from vggt_omega.training.depth_input_model import MappedDepthTokenAdapter


def _depth_prediction(frame_errors: list[float]) -> torch.Tensor:
    return torch.tensor(frame_errors, dtype=torch.float32).reshape(1, len(frame_errors), 1, 1, 1)


def test_paired_masks_do_not_create_a_false_regression() -> None:
    prediction = _depth_prediction([0.0, 0.0, 2.0, 2.0])
    target = torch.zeros(1, 4, 1, 1)
    valid = torch.ones_like(target, dtype=torch.bool)
    scale = torch.ones(1)

    for case in all_depth_availability_cases(4):
        selected = torch.tensor(case.mask).reshape(1, 4, 1, 1)
        paired = paired_depth_statistics(
            prediction,
            prediction.clone(),
            target,
            valid & ~selected,
            scale,
        )
        assert paired.baseline.normalized_absolute_error_sum == paired.candidate.normalized_absolute_error_sum
        assert paired.baseline.valid_pixel_count == paired.candidate.valid_pixel_count

    whole_baseline = depth_sufficient_statistics(prediction, target, valid, scale)
    trailing_candidate = depth_sufficient_statistics(
        prediction,
        target,
        valid & torch.tensor([False, False, True, True]).reshape(1, 4, 1, 1),
        scale,
    )
    assert whole_baseline.normalized_mae == 1.0
    assert trailing_candidate.normalized_mae == 2.0


def test_pixel_reduction_is_batch_partition_invariant() -> None:
    prediction = torch.tensor([0.0, 0.0, 9.0]).reshape(3, 1, 1, 1, 1)
    target = torch.zeros(3, 1, 1, 1)
    valid = torch.ones_like(target, dtype=torch.bool)
    scale = torch.ones(3)

    expected = depth_sufficient_statistics(prediction, target, valid, scale)
    by_one = merge_depth_statistics(
        depth_sufficient_statistics(
            prediction[index : index + 1], target[index : index + 1], valid[index : index + 1], scale[index : index + 1]
        )
        for index in range(3)
    )
    by_two = merge_depth_statistics(
        (
            depth_sufficient_statistics(prediction[:2], target[:2], valid[:2], scale[:2]),
            depth_sufficient_statistics(prediction[2:], target[2:], valid[2:], scale[2:]),
        )
    )

    assert expected.normalized_mae == by_one.normalized_mae == by_two.normalized_mae == 3.0
    assert (0.0 + 9.0) / 2 == 4.5  # Old mean-of-batch-means counterexample.


def test_all_fixed4_availability_masks_are_balanced() -> None:
    cases = all_depth_availability_cases(4)

    assert len(cases) == 16
    assert len({case.mask for case in cases}) == 16
    assert Counter(case.provided_frames for case in cases) == {0: 1, 1: 4, 2: 6, 3: 4, 4: 1}
    for provided_frames in range(1, 4):
        exposure = [
            sum(case.mask[index] for case in cases if case.provided_frames == provided_frames) for index in range(4)
        ]
        assert len(set(exposure)) == 1


def test_input_holdout_preserves_targets_and_excludes_hidden_values() -> None:
    depth = torch.arange(1, 33, dtype=torch.float32).reshape(1, 2, 1, 4, 4)
    valid = torch.ones_like(depth, dtype=torch.bool)
    frame_ids = torch.tensor([[3, 4]])
    target_depth = depth.clone()
    target_mask = valid.clone()

    first = build_input_depth_holdout(depth, valid, frame_ids, patch_size=2)
    changed = depth.clone()
    changed[first.holdout_mask] = 10_000
    second = build_input_depth_holdout(changed, valid, frame_ids, patch_size=2)

    assert torch.equal(depth, target_depth)
    assert torch.equal(valid, target_mask)
    assert first.holdout_mask.any()
    assert first.visible_mask.any()
    assert torch.equal(first.depth, second.depth)
    assert torch.equal(first.visible_mask, second.visible_mask)
    adapter = MappedDepthTokenAdapter(patch_size=2, embed_dim=3)
    availability = torch.ones(1, 2, dtype=torch.bool)
    torch.testing.assert_close(
        adapter(first.depth, first.visible_mask, availability),
        adapter(second.depth, second.visible_mask, availability),
    )

    with pytest.raises(ValueError, match="visible valid depth"):
        build_input_depth_holdout(depth, torch.zeros_like(valid), frame_ids, patch_size=2)


def test_empty_subsets_and_metric_units_are_explicit() -> None:
    prediction = torch.tensor([1.0, 1.0, 4.0, 4.0]).reshape(2, 1, 1, 2, 1)
    target = torch.ones(2, 1, 1, 2)
    valid = torch.ones_like(target, dtype=torch.bool)
    scale = torch.tensor([0.5, 2.0])

    statistics = depth_sufficient_statistics(prediction, target, valid, scale)

    assert statistics.normalized_absolute_error_sum == 6.0
    assert statistics.normalized_mae == 1.5
    assert statistics.metric_absolute_error_sum_m == 12.0
    assert statistics.metric_mae_m == 3.0
    assert statistics.near_absolute_error_sum_m == 0.0
    assert statistics.near_valid_pixel_count == 2
    assert statistics.near_mae_m == 0.0
    assert metric_result(0.0, 0) == {"value": None, "count": 0, "reason": "not_applicable"}
