from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from vggt_omega.colmap_export import (
    PinholeCamera,
    export_scene_outputs,
    load_camera_from_dataset_info,
    rotmat_to_colmap_qvec,
    validate_scene_pose,
)
from vggt_omega.pipeline import SceneResult


def _fake_scene(num_frames: int = 2) -> SceneResult:
    images = torch.zeros(num_frames, 3, 8, 8)
    extrinsic = np.tile(np.eye(3, 4, dtype=np.float32)[None], (num_frames, 1, 1))
    extrinsic[1, 0, 3] = 0.25
    intrinsic = np.tile(np.eye(3, dtype=np.float32)[None], (num_frames, 1, 1))
    return SceneResult(
        images=images,
        pose_enc=np.zeros((num_frames, 9), dtype=np.float32),
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        depth=np.ones((num_frames, 8, 8, 1), dtype=np.float32),
        depth_conf=np.ones((num_frames, 8, 8), dtype=np.float32),
        world_points=np.zeros((num_frames, 8, 8, 3), dtype=np.float32),
    )


def test_rotmat_to_colmap_qvec_identity_is_wxyz() -> None:
    qvec = rotmat_to_colmap_qvec(np.eye(3, dtype=np.float32))
    assert qvec == (1.0, 0.0, 0.0, 0.0)


def test_validate_scene_pose_detects_non_identity_translation() -> None:
    summary = validate_scene_pose(_fake_scene(), expected_count=2)
    assert summary["num_poses"] == 2
    assert summary["extrinsic_finite"] is True
    assert summary["all_identity_pose"] is False
    assert summary["quaternion_norm_min"] == 1.0


def test_load_camera_from_dataset_info(tmp_path: Path) -> None:
    dataset_info = tmp_path / "dataset_info.json"
    dataset_info.write_text(
        json.dumps({"camera": {"width": 800, "height": 600, "fx": 1.0, "fy": 2.0, "cx": 3.0, "cy": 4.0}}),
        encoding="utf-8",
    )
    camera = load_camera_from_dataset_info(dataset_info)
    assert camera == PinholeCamera(width=800, height=600, fx=1.0, fy=2.0, cx=3.0, cy=4.0)


def test_export_scene_outputs_writes_colmap_text(tmp_path: Path) -> None:
    image_paths = []
    for name in ("frame_00001.jpg", "frame_00002.jpg"):
        path = tmp_path / name
        path.write_bytes(b"dummy")
        image_paths.append(path)

    output = tmp_path / "export"
    summary = export_scene_outputs(
        _fake_scene(),
        image_paths,
        output,
        PinholeCamera(width=800, height=600, fx=554.0, fy=561.0, cx=401.0, cy=288.0),
        copy_input_images=True,
    )

    assert (output / "predictions.npz").is_file()
    assert (output / "sparse" / "0" / "cameras.txt").is_file()
    assert (output / "sparse" / "0" / "images.txt").is_file()
    assert (output / "sparse" / "0" / "points3D.txt").is_file()
    assert (output / "images" / "frame_00001.jpg").is_file()
    assert summary["num_images"] == 2

    images_txt = (output / "sparse" / "0" / "images.txt").read_text(encoding="utf-8")
    assert "1 1 0 0 0 0 0 0 1 frame_00001.jpg" in images_txt
    assert "frame_00002.jpg" in images_txt
