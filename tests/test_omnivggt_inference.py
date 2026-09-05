# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import trimesh

from vggt_omega.omnivggt_inference import (
    prepare_omnivggt_input,
    render_omnivggt_predictions,
    run_omnivggt_model,
)
from vggt_omega.rgbd_viewer import LoadedRgbdFrame, RgbdFramePair


def _frame(tmp_path: Path, index: int, *, height: int = 480, width: int = 640) -> LoadedRgbdFrame:
    pair = RgbdFramePair(
        frame_id=f"scene/rgb/frame_{index:06d}.png",
        pair_key=f"frame_{index:06d}",
        rgb_path=tmp_path / f"rgb_{index}.png",
        depth_path=tmp_path / f"depth_{index}.png",
        mask_path=tmp_path / f"mask_{index}.png",
    )
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[..., 0] = 32 + index
    rgb[..., 1] = np.arange(width, dtype=np.uint16)[None, :] % 256
    depth = np.full((height, width), 0.8 + index * 0.1, dtype=np.float32)
    mask = np.ones((height, width), dtype=np.bool_)
    mask[: height // 4] = False
    depth[~mask] = 0
    return LoadedRgbdFrame(pair, rgb, depth, mask, f"mask_{index}.png")


def test_prepare_omnivggt_input_matches_official_tensor_contract(tmp_path: Path) -> None:
    prepared = prepare_omnivggt_input([_frame(tmp_path, 0), _frame(tmp_path, 1)], target_size=518)

    assert prepared.images.shape == (2, 3, 392, 518)
    assert prepared.depth.shape == (1, 2, 392, 518, 1)
    assert prepared.mask.shape == (1, 2, 392, 518)
    assert prepared.extrinsics.shape == (1, 2, 3, 4)
    assert prepared.intrinsics.shape == (1, 2, 3, 3)
    assert prepared.depth_gt_index == (0, 1)
    assert prepared.camera_gt_index == ()
    assert prepared.images.dtype == torch.float32
    assert prepared.depth.dtype == torch.float32
    assert prepared.mask.dtype == torch.float32
    assert prepared.images.min() >= 0 and prepared.images.max() <= 1
    assert torch.all(prepared.depth.squeeze(-1)[prepared.mask == 0] == 0)


def test_run_omnivggt_model_passes_explicit_depth_indices_and_returns_cpu(tmp_path: Path) -> None:
    prepared = prepare_omnivggt_input([_frame(tmp_path, 0)], target_size=28)

    class FakeModel:
        def inference(self, **kwargs):
            assert kwargs["images"].device.type == "cpu"
            assert kwargs["depth_gt_index"] == [0]
            assert kwargs["camera_gt_index"] == []
            batch, frames, height, width, _ = kwargs["depth"].shape
            return {
                "depth": torch.ones(batch, frames, height, width, 1),
                "depth_conf": torch.full((batch, frames, height, width), 2.0),
                "world_points": torch.ones(batch, frames, height, width, 3),
                "world_points_conf": torch.full((batch, frames, height, width), 3.0),
                "pose_enc": torch.zeros(batch, frames, 9),
            }

    predictions = run_omnivggt_model(FakeModel(), prepared, device=torch.device("cpu"))

    assert set(predictions) >= {"depth", "depth_conf", "world_points", "pose_enc"}
    assert all(not value.is_cuda for value in predictions.values())


def test_render_omnivggt_predictions_exports_2d_results_camera_table_and_glb(tmp_path: Path) -> None:
    prepared = prepare_omnivggt_input([_frame(tmp_path, 0), _frame(tmp_path, 1)], target_size=28)
    _, frames, height, width, _ = prepared.depth.shape
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    xyz = torch.stack((x.float(), y.float(), torch.ones_like(x).float()), dim=-1)
    world_points = xyz[None, None].repeat(1, frames, 1, 1, 1)
    predictions = {
        "depth": torch.linspace(0.5, 1.5, height * width).reshape(1, 1, height, width, 1).repeat(1, frames, 1, 1, 1),
        "depth_conf": torch.linspace(1, 4, height * width).reshape(1, 1, height, width).repeat(1, frames, 1, 1),
        "world_points": world_points,
        "world_points_conf": torch.ones(1, frames, height, width) * 4,
        "pose_enc": torch.zeros(1, frames, 9),
    }

    def decode_pose(_pose: torch.Tensor, _shape: tuple[int, int]):
        extrinsics = torch.eye(4)[:3].reshape(1, 1, 3, 4).repeat(1, frames, 1, 1)
        intrinsics = torch.eye(3).reshape(1, 1, 3, 3).repeat(1, frames, 1, 1)
        intrinsics[..., 0, 0] = 100
        intrinsics[..., 1, 1] = 101
        intrinsics[..., 0, 2] = width / 2
        intrinsics[..., 1, 2] = height / 2
        return extrinsics, intrinsics

    result = render_omnivggt_predictions(
        prepared,
        predictions,
        pose_decoder=decode_pose,
        output_directory=tmp_path,
        confidence_percentile=0,
        max_points=500,
    )

    assert len(result.gallery) == frames * 4
    assert len(result.frame_statistics) == frames
    assert len(result.camera_statistics) == frames
    assert result.camera_statistics[0][4:6] == (100.0, 101.0)
    assert result.glb_path.is_file()
    assert result.glb_path.suffix == ".glb"
    assert 0 < result.exported_points <= 500
    scene = cast(Any, trimesh.load(result.glb_path, force="scene"))
    assert len(scene.geometry) >= frames * 2
    assert all(isinstance(geometry, trimesh.Trimesh) for geometry in scene.geometry.values())
    assert all(len(geometry.faces) > 0 for geometry in scene.geometry.values())
