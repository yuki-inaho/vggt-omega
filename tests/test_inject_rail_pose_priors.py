from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from scripts.inject_rail_pose_priors import inject_rail_priors, project_to_rail, rail_covariance


def test_projection_and_covariance_follow_rail() -> None:
    axis = np.array([1.0, 0.0, 0.0])
    projected = project_to_rail(np.array([2.0, 3.0, 4.0]), np.zeros(3), axis)
    covariance = rail_covariance(axis, along_std_m=0.5, cross_std_m=0.03)

    assert np.allclose(projected, [2.0, 0.0, 0.0])
    assert np.allclose(np.diag(covariance), [0.25, 0.0009, 0.0009])


def test_inject_copies_database_and_writes_one_prior_per_image(tmp_path: Path) -> None:
    input_db = tmp_path / "filtered.db"
    with sqlite3.connect(input_db) as connection:
        connection.execute("CREATE TABLE images (image_id INTEGER, name TEXT, camera_id INTEGER)")
        connection.execute(
            """CREATE TABLE pose_priors (
            pose_prior_id INTEGER PRIMARY KEY, corr_data_id INTEGER, corr_sensor_id INTEGER,
            corr_sensor_type INTEGER, position BLOB, position_covariance BLOB,
            gravity BLOB, coordinate_system INTEGER)"""
        )
        connection.executemany("INSERT INTO images VALUES (?, ?, ?)", [(1, "a.png", 7), (2, "b.png", 7)])
    trajectory = tmp_path / "trajectory.json"
    trajectory.write_text(
        json.dumps(
            {
                "rail": {"rail_axis": [1, 0, 0], "rail_centroid_m": [0, 0, 0]},
                "frames": [
                    {"image_name": "a.png", "camera_to_world": np.eye(4).tolist()},
                    {
                        "image_name": "b.png",
                        "camera_to_world": [
                            [1, 0, 0, 1],
                            [0, 1, 0, 2],
                            [0, 0, 1, 3],
                            [0, 0, 0, 1],
                        ],
                    },
                ],
            }
        )
    )
    output_db = tmp_path / "rail.db"

    summary = inject_rail_priors(input_db, output_db, trajectory)

    with sqlite3.connect(output_db) as connection:
        rows = connection.execute(
            "SELECT corr_data_id, corr_sensor_id, corr_sensor_type, position, coordinate_system "
            "FROM pose_priors ORDER BY corr_data_id"
        ).fetchall()
    assert summary["pose_prior_count"] == 2
    assert np.allclose(np.frombuffer(rows[1][3], dtype="<f8"), [1, 0.2, 0.3])
    assert rows[1][:3] == (2, 7, 0)
    assert rows[1][4] == 1
    assert input_db.read_bytes() != output_db.read_bytes()
    assert summary["input_database_sha256"] != summary["output_database_sha256"]

    with pytest.raises(FileExistsError):
        inject_rail_priors(input_db, output_db, trajectory)

    fresh_output = tmp_path / "invalid.db"
    with pytest.raises(ValueError, match="degenerate"):
        inject_rail_priors(input_db, fresh_output, trajectory, projection_strength=1.0)
