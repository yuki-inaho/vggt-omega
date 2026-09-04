from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.rgbd_sfm_pilot import (
    analyze_rail_keyframes,
    export_dynamic_rgbd_map,
    filter_colmap_matches,
    image_ids_to_pair_id,
    pair_id_to_image_ids,
    prepare_pilot,
    robust_metric_scale,
    summarize_workflow,
)


def _write_png(path: Path, value: int, shape: tuple[int, int] = (6, 8)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full(shape, value, dtype=np.uint16)
    assert cv2.imwrite(str(path), image)


def _source_dataset(root: Path, frame_count: int = 8) -> Path:
    for index in range(frame_count):
        stem = f"capture_{index:04d}_camera_r"
        _write_png(root / "rgb" / f"{stem}_rgb.png", index, shape=(6, 8))
        _write_png(root / "mapped_depth" / f"{stem}_depth.png", 500 + index)
    return root


def test_prepare_pilot_selects_stride_and_is_idempotent(tmp_path: Path) -> None:
    source = _source_dataset(tmp_path / "source", frame_count=8)
    output = tmp_path / "pilot"

    first = prepare_pilot(source, output, start=1, stride=2, count=3)
    second = prepare_pilot(source, output, start=1, stride=2, count=3)

    assert first == second
    assert [frame["source_index"] for frame in first["frames"]] == [1, 3, 5]
    assert len(list((output / "images").iterdir())) == 3
    assert len(list((output / "mapped_depth").iterdir())) == 3
    for frame in first["frames"]:
        image = output / "images" / frame["image_name"]
        depth = output / "mapped_depth" / frame["depth_name"]
        assert image.is_symlink() and image.resolve().is_file()
        assert depth.is_symlink() and depth.resolve().is_file()
        assert frame["image_sha256"] == hashlib.sha256(image.read_bytes()).hexdigest()
        assert frame["depth_sha256"] == hashlib.sha256(depth.read_bytes()).hexdigest()


def test_prepare_pilot_rejects_missing_depth(tmp_path: Path) -> None:
    source = _source_dataset(tmp_path / "source", frame_count=2)
    next((source / "mapped_depth").glob("*.png")).unlink()

    with pytest.raises(FileNotFoundError, match="mapped depth"):
        prepare_pilot(source, tmp_path / "pilot", start=0, stride=1, count=2)


def test_robust_metric_scale_ignores_ratio_outlier() -> None:
    predicted = [np.full((10, 10), 0.5, dtype=np.float32)]
    measured = [np.full((10, 10), 1.0, dtype=np.float32)]
    measured[0][0, 0] = 100.0

    result = robust_metric_scale(measured, predicted, min_depth_m=0.1, max_depth_m=200.0)

    assert result["scale"] == pytest.approx(2.0)
    assert result["trimmed_sample_count"] > 0
    assert result["raw_sample_count"] == 100


def test_robust_metric_scale_rejects_empty_valid_pixels() -> None:
    with pytest.raises(ValueError, match="valid depth ratios"):
        robust_metric_scale(
            [np.zeros((2, 2), dtype=np.float32)],
            [np.ones((2, 2), dtype=np.float32)],
            min_depth_m=0.1,
            max_depth_m=1.3,
        )


def test_analyze_rail_keyframes_reports_residual_and_endpoints() -> None:
    positions = np.array(
        [[0.0, 0.00, 0.0], [0.04, 0.01, 0.0], [0.09, -0.01, 0.0], [0.15, 0.00, 0.0]],
        dtype=np.float64,
    )
    rotations = np.repeat(np.eye(3)[None], 4, axis=0)

    result = analyze_rail_keyframes(
        positions,
        rotations,
        min_translation_m=0.08,
        min_rotation_deg=3.0,
        max_gap=3,
    )

    assert result["keyframe_indices"] == [0, 2, 3]
    assert result["orthogonal_rms_m"] > 0.0
    assert result["orthogonal_max_m"] >= result["orthogonal_rms_m"]
    assert len(result["rail_coordinates_m"]) == 4
    assert np.dot(result["rail_axis"], positions[-1] - positions[0]) >= 0.0


def test_pair_id_round_trip() -> None:
    pair_id = image_ids_to_pair_id(19, 3)
    assert pair_id_to_image_ids(pair_id) == (3, 19)


def _make_filter_fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    pilot = root / "pilot"
    (pilot / "images").mkdir(parents=True)
    (pilot / "mapped_depth").mkdir()
    vggt = pilot / "vggt4d"
    vggt.mkdir()

    frames = []
    for index in range(2):
        image_name = f"capture_{index:04d}_camera_r_rgb.png"
        depth_name = image_name.replace("_rgb.png", "_depth.png")
        _write_png(pilot / "images" / image_name, index, shape=(6, 8))
        depth = np.full((6, 8), 1000, dtype=np.uint16)
        if index == 0:
            depth[1, 4] = 0
        assert cv2.imwrite(str(pilot / "mapped_depth" / depth_name), depth)
        mask = np.zeros((6, 8), dtype=np.uint8)
        if index == 0:
            mask[1, 2] = 255
        assert cv2.imwrite(str(vggt / f"dynamic_mask_{index:04d}.png"), mask)
        frames.append({"frame_index": index, "image_name": image_name, "depth_name": depth_name})

    manifest = {"schema_version": 1, "image_width": 8, "image_height": 6, "frames": frames}
    manifest_path = pilot / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    trajectory = {
        "frames": [
            {
                "frame_index": 0,
                "camera_to_world": np.eye(4).tolist(),
            },
            {
                "frame_index": 1,
                "camera_to_world": np.eye(4).tolist(),
            },
        ]
    }
    trajectory_path = pilot / "trajectory.json"
    trajectory_path.write_text(json.dumps(trajectory))

    database = pilot / "raw.db"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE images(image_id INTEGER PRIMARY KEY, name TEXT NOT NULL, camera_id INTEGER NOT NULL);
        CREATE TABLE keypoints(image_id INTEGER PRIMARY KEY, rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB);
        CREATE TABLE matches(pair_id INTEGER PRIMARY KEY, rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB);
        CREATE TABLE two_view_geometries(pair_id INTEGER PRIMARY KEY, rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB);
        """
    )
    connection.executemany(
        "INSERT INTO images(image_id, name, camera_id) VALUES (?, ?, 1)",
        [(1, frames[0]["image_name"]), (2, frames[1]["image_name"])],
    )
    # Matches: valid, dynamic in image 1, invalid depth in image 1, inconsistent geometry.
    keypoints = np.array([[1.5, 1.5], [2.5, 1.5], [4.5, 1.5], [6.5, 1.5]], np.float32)
    connection.executemany(
        "INSERT INTO keypoints(image_id, rows, cols, data) VALUES (?, ?, ?, ?)",
        [(1, 4, 2, keypoints.tobytes()), (2, 4, 2, keypoints.tobytes())],
    )
    matches = np.array([[0, 0], [1, 1], [2, 2], [3, 0]], np.uint32)
    connection.execute(
        "INSERT INTO matches(pair_id, rows, cols, data) VALUES (?, ?, 2, ?)",
        (image_ids_to_pair_id(1, 2), len(matches), matches.tobytes()),
    )
    connection.execute(
        "INSERT INTO two_view_geometries(pair_id, rows, cols, data) VALUES (?, 1, 2, ?)",
        (image_ids_to_pair_id(1, 2), np.array([[0, 0]], np.uint32).tobytes()),
    )
    connection.commit()
    connection.close()
    return database, manifest_path, vggt, trajectory_path


def test_filter_colmap_matches_is_non_destructive_and_counts_reasons(tmp_path: Path) -> None:
    raw_db, manifest, vggt_dir, trajectory = _make_filter_fixture(tmp_path)
    filtered_db = raw_db.with_name("filtered.db")
    raw_before = hashlib.sha256(raw_db.read_bytes()).hexdigest()

    summary = filter_colmap_matches(
        raw_db=raw_db,
        filtered_db=filtered_db,
        manifest_path=manifest,
        vggt_dir=vggt_dir,
        trajectory_path=trajectory,
        fx=4.0,
        fy=4.0,
        cx=4.0,
        cy=3.0,
        min_depth_m=0.1,
        max_depth_m=1.3,
        absolute_tolerance_m=0.05,
        relative_tolerance=0.05,
    )

    assert hashlib.sha256(raw_db.read_bytes()).hexdigest() == raw_before
    assert summary["totals"] == {
        "before": 4,
        "kept": 2,
        "kept_depth_consistent": 1,
        "kept_depth_unavailable": 1,
        "invalid_coordinate": 0,
        "dynamic": 1,
        "invalid_depth": 0,
        "inconsistent_3d": 1,
    }
    connection = sqlite3.connect(filtered_db)
    assert connection.execute("SELECT rows FROM matches").fetchone()[0] == 2
    assert connection.execute("SELECT COUNT(*) FROM two_view_geometries").fetchone()[0] == 0
    connection.close()


def test_filter_colmap_matches_refuses_existing_output(tmp_path: Path) -> None:
    raw_db, manifest, vggt_dir, trajectory = _make_filter_fixture(tmp_path)
    filtered_db = raw_db.with_name("filtered.db")
    filtered_db.write_bytes(b"do not replace")

    with pytest.raises(FileExistsError, match="filtered database"):
        filter_colmap_matches(
            raw_db=raw_db,
            filtered_db=filtered_db,
            manifest_path=manifest,
            vggt_dir=vggt_dir,
            trajectory_path=trajectory,
            fx=4.0,
            fy=4.0,
            cx=4.0,
            cy=3.0,
        )


def test_export_dynamic_rgbd_map_writes_metric_ply(tmp_path: Path) -> None:
    _, _, vggt_dir, trajectory = _make_filter_fixture(tmp_path)
    pilot = tmp_path / "pilot"
    output = pilot / "dynamic_map" / "dynamic_rgbd_map.ply"

    summary = export_dynamic_rgbd_map(
        pilot_root=pilot,
        vggt_dir=vggt_dir,
        trajectory_path=trajectory,
        output_path=output,
        fx=4.0,
        fy=4.0,
        cx=4.0,
        cy=3.0,
        pixel_stride=1,
        voxel_size_m=0.001,
    )

    assert summary["point_count_before_voxel"] == 1
    assert summary["point_count_after_voxel"] == 1
    assert output.read_bytes().startswith(b"ply\nformat binary_little_endian 1.0\nelement vertex 1\n")
    assert output.with_suffix(".json").is_file()


def test_workflow_summary_contract_records_explicit_blockers(tmp_path: Path) -> None:
    pilot = tmp_path / "pilot"
    pilot.mkdir()

    summary = summarize_workflow(
        pilot,
        vggt4d_checkpoint=tmp_path / "missing-vggt4d.pt",
        vggt_omega_checkpoint=tmp_path / "missing-vggt-omega.pt",
        opend4rt_checkpoint=tmp_path / "missing-opend4rt.ckpt",
    )

    assert set(summary["definition_of_done"]) == {
        "DoD-1_manifest",
        "DoD-2_vggt4d",
        "DoD-3_metric_rail_keyframes",
        "DoD-4_raw_aliked_lightglue",
        "DoD-5_rgbd_filtered_db",
        "DoD-6_glomap_bundle_adjustment",
        "DoD-7_quality_logs",
        "DoD-8_component_inventory",
    }
    assert summary["definition_of_done"]["DoD-8_component_inventory"] is True
    assert summary["core_workflow_success"] is False
    assert all(summary["components"][name]["status"] == "blocked" for name in ("vggt4d", "vggt_omega", "opend4rt"))
    assert (pilot / "workflow_summary.json").is_file()
