from __future__ import annotations

import numpy as np

from scripts.render_dynamic_geometry_preview import _heatmap, _normalized_magnitude


def test_dynamic_preview_heatmap_preserves_rectangular_shape_and_bounds() -> None:
    values = np.array([[0.0, 0.5, 1.0], [np.nan, -1.0, 2.0]], dtype=np.float32)

    image = _heatmap(values)

    assert image.shape == (2, 3, 3)
    assert image.dtype == np.uint8
    np.testing.assert_array_equal(image[0, 0], (0, 0, 255))
    np.testing.assert_array_equal(image[0, 2], (255, 0, 0))


def test_dynamic_preview_magnitude_ignores_invalid_pixels_and_handles_zero() -> None:
    flow = np.zeros((2, 3, 3), dtype=np.float32)
    flow[0, 0, 0] = 2.0
    flow[1, 2] = np.nan
    valid = np.ones((2, 3), dtype=bool)
    valid[1, 2] = False

    normalized, scale = _normalized_magnitude(flow, valid)

    assert scale > 0
    assert np.isfinite(normalized).all()
    assert normalized[0, 0] == 1.0
    assert normalized[1, 2] == 0.0

    zeros, zero_scale = _normalized_magnitude(np.zeros_like(flow), valid)
    assert zero_scale == 0.0
    assert not zeros.any()
