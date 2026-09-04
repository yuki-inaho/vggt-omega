from __future__ import annotations

import cv2
import numpy as np

from scripts.prepare_dynamic_flow_teacher import _directed_confidence, compute_bidirectional_dis_teacher


def _textured_image(height: int = 32, width: int = 48) -> np.ndarray:
    rows, columns = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    return np.stack(
        (
            (columns * 13 + rows * 3) % 256,
            (columns * 5 + rows * 17) % 256,
            (columns * 19 + rows * 7) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)


def test_dis_teacher_is_deterministic_finite_and_rectangular() -> None:
    first = _textured_image()
    transform = np.float32([[1, 0, 1], [0, 1, 0]])
    second = cv2.warpAffine(first, transform, (first.shape[1], first.shape[0]))

    first_forward, first_reverse = compute_bidirectional_dis_teacher(first, second)
    second_forward, second_reverse = compute_bidirectional_dis_teacher(first, second)

    for left, right in ((first_forward, second_forward), (first_reverse, second_reverse)):
        assert set(left) == {"confidence", "flow_xy", "occlusion_label"}
        assert left["flow_xy"].shape == (32, 48, 2)
        assert left["flow_xy"].dtype == np.float32
        assert left["confidence"].shape == (32, 48)
        assert left["confidence"].dtype == np.float32
        assert left["occlusion_label"].dtype == np.int8
        assert np.isfinite(left["flow_xy"]).all()
        assert np.isfinite(left["confidence"]).all()
        assert ((left["confidence"] >= 0) & (left["confidence"] <= 1)).all()
        assert np.isin(left["occlusion_label"], (-1, 0, 1)).all()
        np.testing.assert_array_equal(left["flow_xy"], right["flow_xy"])
        np.testing.assert_array_equal(left["confidence"], right["confidence"])
        np.testing.assert_array_equal(left["occlusion_label"], right["occlusion_label"])


def test_dis_teacher_identity_has_near_zero_flow() -> None:
    image = _textured_image()

    forward, reverse = compute_bidirectional_dis_teacher(image, image)

    assert np.quantile(np.linalg.norm(forward["flow_xy"], axis=-1), 0.95) < 1e-3
    assert np.quantile(np.linalg.norm(reverse["flow_xy"], axis=-1), 0.95) < 1e-3


def test_occluded_out_of_bounds_pixels_keep_source_evidence_confidence() -> None:
    image = _textured_image(height=16, width=24)
    flow = np.zeros((16, 24, 2), dtype=np.float32)
    flow[..., 0] = 4.0
    reverse = -flow

    confidence, occlusion = _directed_confidence(
        image,
        image,
        flow,
        reverse,
        photo_sigma=0.1,
        texture_scale=0.05,
        coherence_sigma=1.0,
        cycle_threshold=1.0,
        max_flow_fraction=0.5,
    )

    out_of_bounds = np.zeros((16, 24), dtype=bool)
    out_of_bounds[:, -4:] = True
    assert (occlusion[out_of_bounds] == 0).all()
    assert (confidence[out_of_bounds] > 0).any()
