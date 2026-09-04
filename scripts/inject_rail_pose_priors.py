"""Copy a COLMAP database and add rail-projected Cartesian pose priors."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rail_covariance(axis: np.ndarray, along_std_m: float, cross_std_m: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    along = np.outer(axis, axis)
    return along_std_m**2 * along + cross_std_m**2 * (np.eye(3) - along)


def project_to_rail(position: np.ndarray, centroid: np.ndarray, axis: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    offset = np.asarray(position, dtype=np.float64) - centroid
    return np.asarray(centroid, dtype=np.float64) + np.dot(offset, axis) * axis


def inject_rail_priors(
    input_db: Path,
    output_db: Path,
    trajectory_path: Path,
    along_std_m: float = 0.50,
    cross_std_m: float = 0.001,
    projection_strength: float = 0.90,
) -> dict[str, Any]:
    if output_db.exists():
        raise FileExistsError(f"Refusing to overwrite existing output database: {output_db}")
    if along_std_m <= 0 or cross_std_m <= 0:
        raise ValueError("prior standard deviations must be positive")
    if not 0 <= projection_strength < 1:
        raise ValueError("projection_strength must be in [0, 1) to avoid a degenerate collinear Sim(3)")

    trajectory = json.loads(trajectory_path.read_text())
    rail = trajectory["rail"]
    axis = np.asarray(rail["rail_axis"], dtype=np.float64)
    centroid = np.asarray(rail["rail_centroid_m"], dtype=np.float64)
    covariance = rail_covariance(axis, along_std_m, cross_std_m)
    frames = {frame["image_name"]: frame for frame in trajectory["frames"]}

    output_db.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_db, output_db)
    try:
        with sqlite3.connect(output_db) as connection:
            images = connection.execute("SELECT image_id, name, camera_id FROM images ORDER BY image_id").fetchall()
            connection.execute("DELETE FROM pose_priors")
            positions = []
            for image_id, name, camera_id in images:
                frame = frames.get(name)
                if frame is None:
                    raise ValueError(f"trajectory has no frame for database image: {name}")
                position = np.asarray(frame["camera_to_world"], dtype=np.float64)[:3, 3]
                rail_position = project_to_rail(position, centroid, axis)
                prior_position = position + projection_strength * (rail_position - position)
                positions.append(prior_position)
                connection.execute(
                    """
                    INSERT INTO pose_priors (
                        corr_data_id, corr_sensor_id, corr_sensor_type,
                        position, position_covariance, gravity, coordinate_system
                    ) VALUES (?, ?, 0, ?, ?, ?, 1)
                    """,
                    (
                        image_id,
                        camera_id,
                        prior_position.astype("<f8").tobytes(),
                        np.asfortranarray(covariance, dtype="<f8").tobytes(order="F"),
                        np.full(3, np.nan, dtype="<f8").tobytes(),
                    ),
                )
            count = connection.execute("SELECT COUNT(*) FROM pose_priors").fetchone()[0]
            connection.commit()
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    except Exception:
        output_db.unlink(missing_ok=True)
        raise

    projected_positions = np.asarray(positions)
    summary = {
        "schema_version": 1,
        "input_database": str(input_db.resolve()),
        "input_database_sha256": _sha256(input_db),
        "output_database": str(output_db.resolve()),
        "output_database_sha256": _sha256(output_db),
        "trajectory": str(trajectory_path.resolve()),
        "pose_prior_count": int(count),
        "rail_axis": axis.tolist(),
        "rail_centroid_m": centroid.tolist(),
        "along_std_m": float(along_std_m),
        "cross_std_m": float(cross_std_m),
        "projection_strength": float(projection_strength),
        "prior_orthogonal_rms_m": float(
            np.sqrt(np.mean(np.sum(np.cross(projected_positions - centroid, axis) ** 2, axis=1)))
        ),
    }
    summary_path = output_db.with_suffix(".rail_priors.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-db", type=Path, required=True)
    parser.add_argument("--output-db", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--along-std-m", type=float, default=0.50)
    parser.add_argument("--cross-std-m", type=float, default=0.001)
    parser.add_argument("--projection-strength", type=float, default=0.90)
    args = parser.parse_args(argv)
    summary = inject_rail_priors(
        args.input_db,
        args.output_db,
        args.trajectory,
        args.along_std_m,
        args.cross_std_m,
        args.projection_strength,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
