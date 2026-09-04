from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from vggt_omega.training.multiframe import (
    TemporalSemanticMixer,
    build_warped_neighbor_condition,
    multiframe_tracking_scalars,
)

CONFIG_DIR = Path(__file__).parents[1] / "configs" / "training"


def _cameras(frame_count: int, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    intrinsics = (
        torch.tensor(
            [[4.0, 0.0, width / 2], [0.0, 4.0, height / 2], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        .reshape(1, 1, 3, 3)
        .repeat(1, frame_count, 1, 1)
    )
    extrinsics = torch.eye(4, dtype=torch.float32)[:3].reshape(1, 1, 3, 4).repeat(1, frame_count, 1, 1)
    return intrinsics, extrinsics


def _condition_inputs(frame_count: int = 2, height: int = 4, width: int = 5) -> dict[str, torch.Tensor]:
    image = torch.linspace(0, 1, height * width).reshape(1, 1, 1, height, width).repeat(1, frame_count, 3, 1, 1)
    depth = torch.ones(1, frame_count, height, width)
    intrinsics, extrinsics = _cameras(frame_count, height, width)
    return {
        "images": image,
        "depths": depth,
        "intrinsics": intrinsics,
        "extrinsics_w2c": extrinsics,
        "valid_mask": torch.ones_like(depth, dtype=torch.bool),
    }


def test_warped_neighbor_condition_identical_frames_is_identity() -> None:
    inputs = _condition_inputs()

    result = build_warped_neighbor_condition(**inputs, max_depth_m=1.2)

    torch.testing.assert_close(result["warped_rgb"], inputs["images"], atol=1e-6, rtol=1e-6)
    assert result["visibility"].all()
    assert result["condition"].shape == (1, 2, 4, 4, 5)


def test_warped_neighbor_condition_known_translation_and_occlusion() -> None:
    inputs = _condition_inputs(height=5, width=7)
    inputs["images"].zero_()
    inputs["images"][:, 0, :, 2, 3] = 1
    inputs["extrinsics_w2c"][:, 1, 0, 3] = 0.25

    translated = build_warped_neighbor_condition(**inputs, max_depth_m=1.2)
    torch.testing.assert_close(translated["warped_rgb"][0, 1, :, 2, 4], torch.ones(3), atol=1e-5, rtol=1e-5)

    inputs["depths"][:, 1] = 0.5
    occluded = build_warped_neighbor_condition(
        **inputs,
        max_depth_m=1.2,
        relative_depth_tolerance=0.01,
    )
    assert not occluded["visibility"][:, 1].any()


def test_warped_neighbor_condition_is_permutation_equivariant() -> None:
    inputs = _condition_inputs(frame_count=3)
    inputs["images"] = torch.rand_like(inputs["images"])
    permutation = torch.tensor([2, 0, 1])

    original = build_warped_neighbor_condition(**inputs, max_depth_m=1.2)
    permuted_inputs = {key: value[:, permutation] if value.ndim >= 2 else value for key, value in inputs.items()}
    permuted = build_warped_neighbor_condition(**permuted_inputs, max_depth_m=1.2)

    torch.testing.assert_close(permuted["condition"], original["condition"][:, permutation])


def test_warped_neighbor_condition_ignores_padded_frames_and_single_frame_is_zero() -> None:
    inputs = _condition_inputs(frame_count=3)
    inputs["images"][:, 2] = 99
    frame_mask = torch.tensor([[True, True, False]])
    padded = build_warped_neighbor_condition(**inputs, frame_mask=frame_mask, max_depth_m=1.2)
    cropped = build_warped_neighbor_condition(
        **{key: value[:, :2] for key, value in inputs.items()},
        max_depth_m=1.2,
    )
    torch.testing.assert_close(padded["condition"][:, :2], cropped["condition"])
    assert padded["condition"][:, 2].eq(0).all()

    single = build_warped_neighbor_condition(
        **{key: value[:, :1] for key, value in inputs.items()},
        max_depth_m=1.2,
    )
    assert single["condition"].eq(0).all()
    assert not single["visibility"].any()


def test_temporal_semantic_mixer_tracks_reference_permutation_and_padding() -> None:
    torch.manual_seed(7)
    mixer = TemporalSemanticMixer(hidden_dim=8, num_heads=2, depth=2)
    features = torch.randn(2, 3, 4, 8)
    token_mask = torch.ones(2, 3, 4, dtype=torch.bool)
    frame_mask = torch.tensor([[True, True, False], [True, True, True]])
    references = torch.tensor([1, 2])
    permutation = torch.tensor([2, 0, 1])
    inverse = torch.argsort(permutation)
    remapped_references = inverse[references]

    original = mixer(features, token_mask, frame_mask=frame_mask, reference_indices=references)
    permuted = mixer(
        features[:, permutation],
        token_mask[:, permutation],
        frame_mask=frame_mask[:, permutation],
        reference_indices=remapped_references,
    )

    torch.testing.assert_close(permuted, original[:, permutation], atol=1e-6, rtol=1e-6)
    assert original[0, 2].eq(0).all()


def test_temporal_semantic_mixer_single_frame_degenerates_to_masked_identity() -> None:
    mixer = TemporalSemanticMixer(hidden_dim=8, num_heads=2, depth=2)
    features = torch.randn(2, 1, 3, 8)
    token_mask = torch.tensor([[[True, False, True]], [[True, True, True]]])

    mixed = mixer(features, token_mask, reference_indices=torch.zeros(2, dtype=torch.long))

    expected = torch.where(token_mask.unsqueeze(-1), features, torch.zeros_like(features))
    torch.testing.assert_close(mixed, expected)


def test_multiframe_hydra_profile_tracks_explicit_mask_and_reference_contract() -> None:
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="config", overrides=["pixel_depth=pixel_perfect_multiframe"])

    assert cfg.pixel_depth.enabled is True
    assert cfg.pixel_depth.temporal.reference_mode == "random_valid"
    assert cfg.pixel_depth.temporal.preserve_frame_order is True
    assert cfg.pixel_depth.geometry.occlusion_mask is True
    assert cfg.pixel_depth.geometry.dynamic_mask is True
    assert cfg.pixel_depth.geometry.invalid_mask is True
    assert cfg.pixel_depth.geometry.max_depth_m == 1.2


def test_multiframe_tracking_scalars_report_reference_padding_and_masks() -> None:
    frame_mask = torch.tensor([[True, True, False]])
    valid = torch.tensor([[[[True, True]], [[True, False]], [[False, False]]]])
    dynamic = torch.tensor([[[[False, True]], [[False, False]], [[False, False]]]])
    visible = torch.tensor([[[[True, False]], [[False, True]], [[False, False]]]])

    scalars = multiframe_tracking_scalars(
        frame_mask=frame_mask,
        valid_mask=valid,
        dynamic_mask=dynamic,
        warped_visibility=visible,
        reference_indices=torch.tensor([1]),
        preserve_frame_order=True,
    )

    expected = {
        "multiframe_frame_count": 2.0,
        "multiframe_reference_index": 1.0,
        "multiframe_padding_fraction": 1 / 3,
        "multiframe_valid_fraction": 3 / 4,
        "multiframe_dynamic_fraction": 1 / 4,
        "multiframe_warped_visibility": 2 / 4,
        "multiframe_preserve_frame_order": 1.0,
    }
    assert scalars.keys() == expected.keys()
    for name, value in expected.items():
        assert scalars[name] == pytest.approx(value)
