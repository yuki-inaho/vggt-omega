from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch


def _api() -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any], Callable[..., Any], type[Any]]:
    from vggt_omega.training.dynamic_geometry import (
        CanonicalMotionHead,
        build_rgbd_motion_targets,
        build_temporal_pairs,
        canonical_points_from_depth,
        partition_dynamic_probability,
    )

    return (
        build_temporal_pairs,
        canonical_points_from_depth,
        build_rgbd_motion_targets,
        partition_dynamic_probability,
        CanonicalMotionHead,
    )


def _geometry(
    *,
    batch: int = 1,
    frames: int = 2,
    height: int = 4,
    width: int = 5,
) -> dict[str, torch.Tensor]:
    depths = torch.ones(batch, frames, height, width)
    intrinsics = torch.eye(3).expand(batch, frames, 3, 3).clone()
    intrinsics[..., 0, 2] = (width - 1) / 2
    intrinsics[..., 1, 2] = (height - 1) / 2
    extrinsics = torch.eye(4).expand(batch, frames, 4, 4).clone()[..., :3, :]
    return {
        "depths": depths,
        "depth_masks": torch.ones_like(depths, dtype=torch.bool),
        "original_depth_observed_mask": torch.ones_like(depths, dtype=torch.bool),
        "intrinsics": intrinsics,
        "extrinsics_w2c": extrinsics,
        "normalization_scale_m": torch.ones(batch),
        "frame_ids": torch.arange(frames).expand(batch, frames).clone(),
        "frame_mask": torch.ones(batch, frames, dtype=torch.bool),
    }


def _motion_inputs(geometry: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    build_pairs, _, _, _, _ = _api()
    pairs = build_pairs(geometry["frame_ids"], geometry["frame_mask"])
    batch, pair_count = pairs["motion_pair_indices"].shape[:2]
    height, width = geometry["depths"].shape[-2:]
    return {
        **geometry,
        "motion_pair_indices": pairs["motion_pair_indices"],
        "pixel_flow_xy": torch.zeros(batch, pair_count, height, width, 2),
        "flow_confidence": torch.ones(batch, pair_count, height, width),
    }


def test_temporal_pairs_follow_frame_ids_and_pad_short_samples() -> None:
    build_pairs, _, _, _, _ = _api()
    frame_ids = torch.tensor([[12, 10, 11, -1], [20, 22, -1, -1]])
    frame_mask = torch.tensor([[True, True, True, False], [True, True, False, False]])

    result = build_pairs(frame_ids, frame_mask)

    torch.testing.assert_close(
        result["motion_pair_indices"][0],
        torch.tensor([[1, 2], [2, 1], [2, 0], [0, 2]]),
    )
    torch.testing.assert_close(
        result["motion_pair_indices"][1],
        torch.tensor([[0, 1], [1, 0], [-1, -1], [-1, -1]]),
    )
    torch.testing.assert_close(
        result["motion_pair_valid_mask"],
        torch.tensor([[True, True, True, True], [True, True, False, False]]),
    )
    torch.testing.assert_close(result["motion_time_delta_frames"][0], torch.tensor([1, -1, 1, -1]))


def test_temporal_pairs_reject_empty_duplicate_and_custom_graph() -> None:
    build_pairs, _, _, _, _ = _api()
    with pytest.raises(ValueError, match="at least two"):
        build_pairs(torch.tensor([[0, -1]]), torch.tensor([[True, False]]))
    with pytest.raises(ValueError, match="duplicate"):
        build_pairs(torch.tensor([[3, 3]]), torch.tensor([[True, True]]))
    with pytest.raises(ValueError, match="adjacent_bidirectional"):
        build_pairs(
            torch.tensor([[0, 1, 2]]),
            torch.ones(1, 3, dtype=torch.bool),
            motion_pair_indices=torch.tensor([[[0, 2], [2, 0], [-1, -1], [-1, -1]]]),
        )


def test_canonical_points_are_rectangular_and_first_camera_rebased() -> None:
    _, canonical_points, _, _, _ = _api()
    geometry = _geometry(height=3, width=7)
    geometry["extrinsics_w2c"][:, 0, :, 3] = torch.tensor([0.5, -0.25, 0.0])
    geometry["extrinsics_w2c"][:, 1, :, 3] = torch.tensor([1.0, -0.25, 0.0])

    result = canonical_points(
        geometry["depths"],
        geometry["depth_masks"],
        geometry["intrinsics"],
        geometry["extrinsics_w2c"],
        geometry["frame_mask"],
    )

    assert result["canonical_points_current"].shape == (1, 2, 3, 7, 3)
    assert result["canonical_points_valid_mask"].shape == (1, 2, 3, 7)
    torch.testing.assert_close(result["rebased_extrinsics_w2c"][0, 0], torch.eye(4)[:3], atol=1e-6, rtol=0)


def test_rgbd_motion_static_identity_is_known_zero() -> None:
    _, _, build_targets, _, _ = _api()
    inputs = _motion_inputs(_geometry())

    result = build_targets(**inputs)

    center = (0, 0, 1, 2)
    torch.testing.assert_close(result["target_canonical_scene_flow"][center], torch.zeros(3), atol=1e-6, rtol=0)
    assert result["target_visibility_label"][center].item() == 1
    assert result["target_visibility_known_mask"][center].item() is True
    assert result["target_dynamic_label"][center].item() == 0


def test_rgbd_motion_compensates_known_rigid_camera_translation() -> None:
    _, _, build_targets, _, _ = _api()
    geometry = _geometry(height=4, width=5)
    geometry["extrinsics_w2c"][:, 1, 0, 3] = 1.0
    inputs = _motion_inputs(geometry)
    inputs["pixel_flow_xy"][:, 0, :, :, 0] = 1.0
    inputs["pixel_flow_xy"][:, 1, :, :, 0] = -1.0

    result = build_targets(**inputs)

    center = (0, 0, 1, 1)
    torch.testing.assert_close(result["target_canonical_scene_flow"][center], torch.zeros(3), atol=1e-6, rtol=0)
    assert result["target_dynamic_label"][center].item() == 0


def test_rgbd_motion_marks_known_moving_point_dynamic() -> None:
    _, _, build_targets, _, _ = _api()
    geometry = _geometry(height=4, width=5)
    geometry["depths"][:, 1] = 1.1
    inputs = _motion_inputs(geometry)

    result = build_targets(**inputs, static_off_m=0.01, dynamic_on_m=0.05)

    center = (0, 0, 1, 2)
    torch.testing.assert_close(
        result["target_canonical_scene_flow"][center],
        torch.tensor([0.0, -0.05, 0.1]),
        atol=1e-5,
        rtol=0,
    )
    assert result["target_dynamic_label"][center].item() == 1


def test_explicit_occlusion_is_visibility_negative_but_motion_unknown() -> None:
    _, _, build_targets, _, _ = _api()
    inputs = _motion_inputs(_geometry())
    occlusion = torch.full(inputs["flow_confidence"].shape, -1, dtype=torch.int8)
    occlusion[0, 0, 1, 2] = 0

    result = build_targets(**inputs, flow_occlusion_label=occlusion)

    center = (0, 0, 1, 2)
    assert result["target_visibility_label"][center].item() == 0
    assert result["target_visibility_known_mask"][center].item() is True
    assert result["target_visibility_confidence"][center].item() > 0
    assert result["target_dynamic_label"][center].item() == -1
    assert result["target_confidence"][center].item() == 0
    torch.testing.assert_close(result["target_canonical_scene_flow"][center], torch.zeros(3))


def test_depth_boundary_is_unknown_instead_of_dynamic() -> None:
    _, _, build_targets, _, _ = _api()
    geometry = _geometry(height=5, width=5)
    geometry["depths"][0, 1, 2, 2] = 10.0
    inputs = _motion_inputs(geometry)

    result = build_targets(**inputs, depth_discontinuity_relative=0.03)

    center = (0, 0, 2, 2)
    assert result["target_visibility_label"][center].item() == -1
    assert result["target_visibility_known_mask"][center].item() is False
    assert result["target_dynamic_label"][center].item() == -1
    assert result["target_confidence"][center].item() == 0


def test_missing_flow_teacher_returns_finite_graph_zero_targets() -> None:
    build_pairs, _, build_targets, _, _ = _api()
    geometry = _geometry()
    pairs = build_pairs(geometry["frame_ids"], geometry["frame_mask"])

    result = build_targets(
        **geometry,
        motion_pair_indices=pairs["motion_pair_indices"],
        pixel_flow_xy=None,
        flow_confidence=None,
    )

    assert not result["target_visibility_known_mask"].any()
    assert (result["target_visibility_label"] == -1).all()
    assert (result["target_dynamic_label"] == -1).all()
    assert not result["target_confidence"].any()
    assert torch.isfinite(result["target_canonical_scene_flow"]).all()
    assert not result["target_canonical_scene_flow"].any()


def test_rgbd_motion_targets_use_float32_under_mixed_precision() -> None:
    _, _, build_targets, _, _ = _api()
    inputs = _motion_inputs(_geometry())
    for key in (
        "depths",
        "intrinsics",
        "extrinsics_w2c",
        "normalization_scale_m",
        "pixel_flow_xy",
        "flow_confidence",
    ):
        inputs[key] = inputs[key].to(torch.bfloat16)

    result = build_targets(**inputs)

    for key in (
        "target_canonical_scene_flow",
        "target_scene_flow_m",
        "target_visibility_confidence",
        "target_confidence",
    ):
        assert result[key].dtype is torch.float32
        assert torch.isfinite(result[key]).all()


def test_dynamic_probability_partition_handles_extremes_readiness_and_padding() -> None:
    _, _, _, partition, _ = _api()
    probability = torch.tensor([[[[0.10, 0.90, 0.50, 0.90]]]])
    visibility = torch.ones_like(probability)
    domain = torch.tensor([[[[True, True, True, False]]]])

    ready = partition(
        probability,
        visibility,
        domain,
        ready=True,
        visibility_threshold=0.5,
        static_probability_max=0.25,
        dynamic_probability_min=0.75,
    )
    torch.testing.assert_close(ready["dynamic_mask"], torch.tensor([[[[False, True, False, False]]]]))
    torch.testing.assert_close(ready["dynamic_unknown_mask"], torch.tensor([[[[False, False, True, False]]]]))

    unready = partition(
        probability,
        visibility,
        domain,
        ready=False,
        visibility_threshold=0.5,
        static_probability_max=0.25,
        dynamic_probability_min=0.75,
    )
    assert not unready["dynamic_mask"].any()
    torch.testing.assert_close(unready["dynamic_unknown_mask"], domain)


def test_motion_head_rectangular_output_has_finite_backward() -> None:
    _, _, _, _, motion_head = _api()
    head = motion_head(feature_dim=8, hidden_dim=16, relative_camera_dim=12)
    features = torch.randn(1, 3, 6, 8, requires_grad=True)
    pair_indices = torch.tensor([[[0, 1], [1, 0], [1, 2], [2, 1]]])
    pair_valid = torch.ones(1, 4, dtype=torch.bool)
    relative_camera = torch.randn(1, 4, 12)
    delta_frames = torch.tensor([[1, -1, 1, -1]])

    result = head(
        features,
        relative_camera,
        delta_frames,
        pair_indices,
        pair_valid,
        patch_grid_hw=(2, 3),
        output_hw=(5, 7),
    )
    assert result["canonical_scene_flow"].shape == (1, 4, 5, 7, 3)
    assert result["motion_visibility_probability"].shape == (1, 4, 5, 7)
    assert result["dynamic_probability"].shape == (1, 4, 5, 7)
    loss = (
        result["canonical_scene_flow"].square().mean()
        + result["motion_visibility_probability"].mean()
        + result["dynamic_probability"].mean()
    )
    loss.backward()
    gradients = [parameter.grad for parameter in head.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
