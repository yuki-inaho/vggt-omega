from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from scripts.export_opend4rt_dynamic_map import export_opend4rt_dynamic_map


def test_export_uses_dynamic_visible_tracks_and_covers_overlap_once(tmp_path: Path) -> None:
    root = tmp_path / "workflow"
    (root / "images").mkdir(parents=True)
    (root / "mapped_depth").mkdir()
    frames = []
    for index in range(3):
        image = np.full((4, 4, 3), 10 + index, np.uint8)
        depth = np.full((4, 4), 1000, np.uint16)
        image_name = f"rgb_{index}.png"
        depth_name = f"depth_{index}.png"
        cv2.imwrite(str(root / "images" / image_name), image)
        cv2.imwrite(str(root / "mapped_depth" / depth_name), depth)
        frames.append({"frame_index": index, "image_name": image_name, "depth_name": depth_name})
    (root / "manifest.json").write_text(json.dumps({"frames": frames}))
    trajectory = root / "trajectory.json"
    trajectory.write_text(
        json.dumps({"frames": [{"frame_index": index, "camera_to_world": np.eye(4).tolist()} for index in range(3)]})
    )
    chunks = root / "opend4rt_chunks"
    chunk_dir = chunks / "chunk_000000"
    chunk_dir.mkdir(parents=True)
    chunk_manifest = {
        "chunk_count": 1,
        "chunks": [{"chunk_index": 0, "global_indices": [0, 1, 2]}],
    }
    (chunks / "chunks_manifest.json").write_text(json.dumps(chunk_manifest))
    uv = np.array([[[1, 1], [2, 2]], [[1, 1], [2, 2]], [[1, 1], [2, 2]]], np.float32)
    np.savez(
        chunk_dir / "dense_scene.npz",
        point_uv_px=uv,
        point_visibility=np.ones((3, 2), bool),
        point_is_dynamic=np.array([True, False]),
        global_indices=np.array([0, 1, 2]),
    )

    summary = export_opend4rt_dynamic_map(
        root,
        chunks,
        trajectory,
        root / "dynamic_map" / "opend4rt_dynamic_rgbd_map.ply",
        fx=1,
        fy=1,
        cx=0,
        cy=0,
        voxel_size_m=0.001,
    )

    assert summary["covered_frame_count"] == 3
    assert summary["point_count_before_voxel"] == 3
    assert summary["point_count_after_voxel"] == 1
    assert Path(summary["output_path"]).is_file()
