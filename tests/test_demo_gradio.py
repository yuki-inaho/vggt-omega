from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import torch

import demo_gradio
from demo_gradio import parse_args, resolve_preprocess_options


def test_auto_preprocessing_uses_fine_tuned_training_shape() -> None:
    assert resolve_preprocess_options("auto", None, None, (384, 512)) == ("fixed", 384, 512)
    assert resolve_preprocess_options("auto", None, None, None) == ("balanced", None, None)


def test_preprocessing_modes_are_selectable_and_fixed_is_validated() -> None:
    assert resolve_preprocess_options("balanced", 384, 512, (384, 512)) == ("balanced", None, None)
    assert resolve_preprocess_options("max_size", None, None, (384, 512)) == ("max_size", None, None)
    assert resolve_preprocess_options("fixed", 256, 384, None) == ("fixed", 256, 384)
    with pytest.raises(ValueError, match="requires target"):
        resolve_preprocess_options("fixed", None, None, None)
    with pytest.raises(ValueError, match="multiples of 16"):
        resolve_preprocess_options("fixed", 385, 512, None)


def test_demo_cli_accepts_base_head_and_preprocessing_controls() -> None:
    args = parse_args(
        [
            "--checkpoint",
            "base.pt",
            "--head-checkpoint",
            "best.pt",
            "--preprocess-mode",
            "fixed",
            "--target-height",
            "384",
            "--target-width",
            "512",
        ]
    )

    assert args.checkpoint == "base.pt"
    assert args.head_checkpoint == "best.pt"
    assert args.preprocess_mode == "fixed"
    assert (args.target_height, args.target_width) == (384, 512)


def test_run_model_auto_forwards_recorded_head_shape(monkeypatch, tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "000000.png").write_bytes(b"fixture")
    observed: dict[str, object] = {}

    def fake_load(image_names, **kwargs):
        observed["names"] = image_names
        observed.update(kwargs)
        return torch.zeros(1, 3, 384, 512)

    class FakeScene:
        def with_world_points(self):
            return self

        def as_npz_dict(self):
            return {"depth": torch.zeros(1, 384, 512, 1).numpy()}

    class FakePipeline:
        recommended_input_shape = (384, 512)

        def run(self, images):
            assert tuple(images.shape) == (1, 3, 384, 512)
            return FakeScene()

    monkeypatch.setattr(demo_gradio, "load_images_from_paths", fake_load)

    result = demo_gradio.run_model(str(tmp_path), cast(Any, FakePipeline()), 512, "auto", None, None)

    assert result["depth"].shape == (1, 384, 512, 1)
    assert observed["mode"] == "fixed"
    assert observed["target_height"] == 384
    assert observed["target_width"] == 512
