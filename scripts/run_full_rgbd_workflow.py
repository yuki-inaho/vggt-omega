"""Run the resumable full RGB-D/VGGT4D/ALIKED/GLOMAP workflow for one split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sqlite3
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.align_vggt4d_chunks import align_chunk_run  # noqa: E402
from scripts.export_opend4rt_dynamic_map import export_opend4rt_dynamic_map  # noqa: E402
from scripts.inject_rail_pose_priors import inject_rail_priors  # noqa: E402
from scripts.rgbd_sfm_pilot import (  # noqa: E402
    DEFAULT_INTRINSICS,
    _database_metrics,
    _parse_model_analyzer,
    export_dynamic_rgbd_map,
    filter_colmap_matches,
    prepare_pilot,
)

# Sibling repositories are resolved from the environment; the defaults assume
# they sit next to this checkout.
SIBLING_ROOT = Path(os.environ.get("RGBD_WORKFLOW_REPO_ROOT", REPO_ROOT.parent))
VGGT4D_REPO = Path(os.environ.get("VGGT4D_REPO", SIBLING_ROOT / "VGGT4D"))
VGGT4D_CHECKPOINT = VGGT4D_REPO / "ckpts/model_tracker_fixed_e20.pt"
COLMAP_REPO = Path(os.environ.get("COLMAP_REPO", SIBLING_ROOT / "colmap"))
OPEND4RT_REPO = Path(os.environ.get("OPEND4RT_REPO", SIBLING_ROOT / "Open-d4rt"))
OPEND4RT_CONFIG = OPEND4RT_REPO / "checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml"
OPEND4RT_CHECKPOINT = OPEND4RT_REPO / "checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_state(output: Path) -> dict[str, Any]:
    path = output / "workflow_state.json"
    if path.is_file():
        return json.loads(path.read_text())
    return {"schema_version": 1, "stages": {}}


def _stage_complete(state: dict[str, Any], name: str) -> bool:
    return state["stages"].get(name, {}).get("status") == "complete"


def _complete_stage(output: Path, state: dict[str, Any], name: str, evidence: Any) -> None:
    state["stages"][name] = {"status": "complete", "evidence": evidence}
    _write_json(output / "workflow_state.json", state)


def _run_logged(command: Sequence[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        log.write(f"\n# cwd: {cwd}\n# command: {shlex.join(command)}\n")
        log.flush()
        subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=True)


def _chunk_count(run_dir: Path) -> tuple[int, int]:
    manifest_path = run_dir / "chunks_manifest.json"
    if not manifest_path.is_file():
        return 0, 0
    manifest = json.loads(manifest_path.read_text())
    expected = int(manifest["chunk_count"])
    complete = 0
    for chunk in manifest["chunks"]:
        chunk_dir = run_dir / f"chunk_{int(chunk['chunk_index']):06d}"
        metadata = chunk_dir / "metadata.json"
        expected_frames = len(chunk["global_indices"])
        fixed = ("pred_traj.txt", "pred_intrinsics.txt", "frames.txt")
        patterns = ("frame_*.png", "frame_*.npy", "conf_*.npy", "dynamic_mask_*.png")
        try:
            metadata_payload = json.loads(metadata.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if (
            metadata_payload.get("num_frames") == expected_frames
            and all((chunk_dir / name).is_file() for name in fixed)
            and all(len(list(chunk_dir.glob(pattern))) == expected_frames for pattern in patterns)
        ):
            complete += 1
    return complete, expected


def _opend4rt_chunk_count(run_dir: Path) -> tuple[int, int]:
    manifest_path = run_dir / "chunks_manifest.json"
    if not manifest_path.is_file():
        return 0, 0
    manifest = json.loads(manifest_path.read_text())
    expected = int(manifest["chunk_count"])
    complete = sum(
        (run_dir / f"chunk_{int(chunk['chunk_index']):06d}" / "metadata.json").is_file()
        and (run_dir / f"chunk_{int(chunk['chunk_index']):06d}" / "dense_scene.npz").is_file()
        for chunk in manifest["chunks"]
    )
    return int(complete), expected


def _model_complete(path: Path) -> bool:
    return all((path / name).is_file() for name in ("cameras.bin", "images.bin", "points3D.bin"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _existing_manifest_frame_count(source: Path, output: Path) -> int | None:
    path = output / "manifest.json"
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text())
    if Path(manifest.get("source_dataset", "")).resolve() != source.resolve():
        return None
    return len(manifest.get("frames", []))


def _existing_filter_summary(raw_db: Path, filtered_db: Path, summary_path: Path) -> dict[str, Any] | None:
    if not raw_db.is_file() or not filtered_db.is_file() or not summary_path.is_file():
        return None
    summary = json.loads(summary_path.read_text())
    raw_hash = _sha256(raw_db)
    if summary.get("raw_database_sha256_before") != raw_hash:
        return None
    if summary.get("raw_database_sha256_after") != raw_hash:
        return None
    metrics = _database_metrics(filtered_db)
    if not metrics or metrics.get("matches_rows") != summary.get("totals", {}).get("kept"):
        return None
    return summary


def _existing_dynamic_summary(output: Path) -> dict[str, Any] | None:
    metadata = output / "dynamic_map" / "dynamic_rgbd_map.json"
    point_cloud = output / "dynamic_map" / "dynamic_rgbd_map.ply"
    if not metadata.is_file() or not point_cloud.is_file() or point_cloud.stat().st_size == 0:
        return None
    summary = json.loads(metadata.read_text())
    if summary.get("point_count_after_voxel", 0) <= 0:
        return None
    return summary


def _pose_prior_count(database_path: Path) -> int:
    if not database_path.is_file():
        return 0
    with sqlite3.connect(database_path) as connection:
        return int(connection.execute("SELECT COUNT(*) FROM pose_priors").fetchone()[0])


def _rail_model_metrics(images_txt: Path, trajectory_path: Path) -> dict[str, Any]:
    trajectory = json.loads(trajectory_path.read_text())
    axis = np.asarray(trajectory["rail"]["rail_axis"], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    centroid = np.asarray(trajectory["rail"]["rail_centroid_m"], dtype=np.float64)
    centers = []
    # COLMAP stores exactly two lines per image. The POINTS2D line may be
    # empty, so it must still consume its slot; skipping blank lines shifts
    # the parser and can make numeric POINTS2D data look like a camera pose.
    record_lines = [line for line in images_txt.read_text().splitlines() if not line.startswith("#")]
    if len(record_lines) % 2:
        raise ValueError(f"incomplete COLMAP two-line image record in {images_txt}")
    for line in record_lines[::2]:
        fields = line.split(maxsplit=9)
        if len(fields) < 10:
            raise ValueError(f"invalid COLMAP image record: {line}")
        qw, qx, qy, qz = map(float, fields[1:5])
        translation = np.asarray(list(map(float, fields[5:8])))
        rotation = np.asarray(
            [
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
            ]
        )
        centers.append(-rotation.T @ translation)
    if not centers:
        raise ValueError(f"no camera centers parsed from {images_txt}")
    centers_array = np.asarray(centers)
    residuals = np.linalg.norm(np.cross(centers_array - centroid, axis), axis=1)
    return {
        "registered_images": len(centers),
        "rail_orthogonal_rms_m": float(np.sqrt(np.mean(residuals**2))),
        "rail_orthogonal_max_m": float(residuals.max()),
        "rail_orthogonal_median_m": float(np.median(residuals)),
    }


def _write_full_summary(source: Path, output: Path) -> dict[str, Any]:
    manifest = json.loads((output / "manifest.json").read_text())
    frame_count = len(manifest["frames"])
    source_frame_count = len(list((source / "rgb").glob("*.png")))
    manifest_files_exist = all(
        (output / "images" / frame["image_name"]).is_file()
        and (output / "mapped_depth" / frame["depth_name"]).is_file()
        for frame in manifest["frames"]
    )
    complete_chunks, expected_chunks = _chunk_count(output / "vggt4d_chunks")
    d4rt_complete_chunks, d4rt_expected_chunks = _opend4rt_chunk_count(output / "opend4rt_chunks")
    trajectory_path = output / "trajectory" / "aligned_trajectory.json"
    trajectory = json.loads(trajectory_path.read_text()) if trajectory_path.is_file() else None
    raw = _database_metrics(output / "colmap" / "raw.db")
    filtered = _database_metrics(output / "colmap" / "filtered.db")
    filter_path = output / "colmap" / "match_filter_summary.json"
    filter_summary = json.loads(filter_path.read_text()) if filter_path.is_file() else None
    model = _parse_model_analyzer(output / "logs" / "model_after_ba.log")
    rail_model = _parse_model_analyzer(output / "logs" / "model_after_rail_ba.log")
    rail_prior_path = output / "colmap" / "rail_prior.rail_priors.json"
    rail_prior = json.loads(rail_prior_path.read_text()) if rail_prior_path.is_file() else None
    rail_metrics_path = output / "colmap" / "rail_ba_metrics.json"
    rail_metrics = json.loads(rail_metrics_path.read_text()) if rail_metrics_path.is_file() else None
    dynamic_path = output / "dynamic_map" / "dynamic_rgbd_map.json"
    dynamic = json.loads(dynamic_path.read_text()) if dynamic_path.is_file() else None
    d4rt_dynamic_path = output / "dynamic_map" / "opend4rt_dynamic_rgbd_map.json"
    d4rt_dynamic = json.loads(d4rt_dynamic_path.read_text()) if d4rt_dynamic_path.is_file() else None
    checks = {
        "manifest_all_frames": frame_count > 0 and frame_count == source_frame_count and manifest_files_exist,
        "vggt4d_all_chunks": expected_chunks > 0 and complete_chunks == expected_chunks,
        "opend4rt_all_chunks": d4rt_expected_chunks > 0 and d4rt_complete_chunks == d4rt_expected_chunks,
        "aligned_pose_all_frames": bool(trajectory and trajectory["frame_count"] == frame_count),
        "sequential_features_all_frames": bool(
            raw
            and raw.get("images") == frame_count
            and raw.get("keypoints") == frame_count
            and raw.get("descriptors") == frame_count
            and raw.get("matches_rows", 0) > 0
        ),
        "rgbd_filtered_matches": bool(
            filter_summary
            and filtered
            and filtered.get("matches_rows") == filter_summary["totals"]["kept"]
            and filter_summary["raw_database_sha256_before"] == filter_summary["raw_database_sha256_after"]
        ),
        "glomap_bundle_adjustment": bool(
            model and model.get("registered_images", 0) >= 4 and model.get("points3d", 0) > 0
        ),
        "rail_prior_bundle_adjustment": bool(
            rail_prior
            and rail_prior.get("pose_prior_count") == frame_count
            and rail_model
            and rail_model.get("registered_images", 0) >= 4
            and rail_model.get("points3d", 0) > 0
            and rail_metrics
            and rail_metrics.get("rail_orthogonal_rms_m", float("inf")) <= 0.05
        ),
        "dynamic_map": bool(dynamic and dynamic.get("point_count_after_voxel", 0) > 0),
        "opend4rt_dynamic_map": bool(
            d4rt_dynamic
            and d4rt_dynamic.get("covered_frame_count") == frame_count
            and d4rt_dynamic.get("point_count_after_voxel", 0) > 0
        ),
    }
    summary = {
        "schema_version": 1,
        "source_dataset": str(source.resolve()),
        "output": str(output.resolve()),
        "frame_count": frame_count,
        "workflow_complete": all(checks.values()),
        "checks": checks,
        "metrics": {
            "complete_chunks": complete_chunks,
            "expected_chunks": expected_chunks,
            "opend4rt_complete_chunks": d4rt_complete_chunks,
            "opend4rt_expected_chunks": d4rt_expected_chunks,
            "keyframe_count": len(trajectory["keyframe_indices"]) if trajectory else None,
            "rail": trajectory["rail"] if trajectory else None,
            "alignment_edges": trajectory["alignment_edges"] if trajectory else None,
            "raw_database": raw,
            "filtered_database": filtered,
            "match_filter": filter_summary["totals"] if filter_summary else None,
            "bundle_adjusted_model": model,
            "rail_prior": rail_prior,
            "rail_bundle_adjusted_model": rail_model,
            "rail_bundle_adjustment_metrics": rail_metrics,
            "dynamic_map": dynamic,
            "opend4rt_dynamic_map": d4rt_dynamic,
        },
    }
    _write_json(output / "workflow_summary.json", summary)
    return summary


def run_workflow(
    source: Path,
    output: Path,
    chunk_size: int = 16,
    chunk_overlap: int = 4,
    sequential_overlap: int = 10,
) -> dict[str, Any]:
    state = _load_state(output)
    frame_count = len(list((source / "rgb").glob("*.png")))
    logs = output / "logs"

    if not _stage_complete(state, "prepare"):
        existing_count = _existing_manifest_frame_count(source, output)
        if existing_count == frame_count:
            _complete_stage(output, state, "prepare", {"frame_count": existing_count, "reconciled": True})
        else:
            manifest = prepare_pilot(source, output, start=0, stride=1, count=frame_count)
            _complete_stage(output, state, "prepare", {"frame_count": len(manifest["frames"])})

    if not _stage_complete(state, "vggt4d_chunks"):
        _run_logged(
            [
                "uv",
                "run",
                "--no-sync",
                "python",
                "-m",
                "scripts.infer_chunks",
                "--input",
                str(output / "images"),
                "--output",
                str(output / "vggt4d_chunks"),
                "--checkpoint",
                str(VGGT4D_CHECKPOINT),
                "--mode",
                "crop",
                "--chunk-size",
                str(chunk_size),
                "--overlap",
                str(chunk_overlap),
            ],
            VGGT4D_REPO,
            logs / "vggt4d_chunks.log",
        )
        complete, expected = _chunk_count(output / "vggt4d_chunks")
        if complete != expected:
            raise RuntimeError(f"VGGT4D chunk contract failed: {complete}/{expected}")
        _complete_stage(output, state, "vggt4d_chunks", {"complete": complete, "expected": expected})

    if not _stage_complete(state, "opend4rt_chunks"):
        _run_logged(
            [
                "uv",
                "run",
                "--no-sync",
                "python",
                "-m",
                "scripts.dump_dense_scene_chunks",
                "--config",
                str(OPEND4RT_CONFIG),
                "--ckpt-path",
                str(OPEND4RT_CHECKPOINT),
                "--image-dir",
                str(output / "images"),
                "--output",
                str(output / "opend4rt_chunks"),
                "--chunk-size",
                "16",
                "--overlap",
                "4",
                "--grid-cols",
                "32",
                "--grid-rows",
                "24",
                "--query-chunk-size",
                "96",
            ],
            OPEND4RT_REPO,
            logs / "opend4rt_chunks.log",
        )
        complete, expected = _opend4rt_chunk_count(output / "opend4rt_chunks")
        if complete != expected:
            raise RuntimeError(f"OpenD4RT chunk contract failed: {complete}/{expected}")
        _complete_stage(output, state, "opend4rt_chunks", {"complete": complete, "expected": expected})

    if not _stage_complete(state, "align_trajectory"):
        trajectory_path = output / "trajectory" / "aligned_trajectory.json"
        trajectory = json.loads(trajectory_path.read_text()) if trajectory_path.is_file() else None
        if not trajectory or trajectory.get("frame_count") != frame_count:
            trajectory = align_chunk_run(
                chunk_run_dir=output / "vggt4d_chunks",
                mapped_depth_dir=output / "mapped_depth",
                output_dir=output / "trajectory",
                min_depth_m=0.1,
                max_depth_m=1.3,
                min_translation_m=0.08,
                min_rotation_deg=3.0,
                max_gap=12,
            )
        _complete_stage(
            output,
            state,
            "align_trajectory",
            {"frame_count": trajectory["frame_count"], "keyframe_count": len(trajectory["keyframe_indices"])},
        )

    raw_db = output / "colmap" / "raw.db"
    if not _stage_complete(state, "feature_extraction"):
        metrics = _database_metrics(raw_db)
        if not metrics or metrics.get("keypoints") != frame_count or metrics.get("descriptors") != frame_count:
            _run_logged(
                [
                    "pixi",
                    "run",
                    "colmap",
                    "feature_extractor",
                    "--database_path",
                    str(raw_db),
                    "--image_path",
                    str(output / "images"),
                    "--ImageReader.camera_model",
                    "PINHOLE",
                    "--ImageReader.camera_params",
                    ",".join(map(str, DEFAULT_INTRINSICS)),
                    "--ImageReader.single_camera",
                    "1",
                    "--FeatureExtraction.type",
                    "ALIKED_N16ROT",
                ],
                COLMAP_REPO,
                logs / "feature_extractor.log",
            )
            metrics = _database_metrics(raw_db)
        if not metrics or metrics.get("keypoints") != frame_count:
            raise RuntimeError(f"feature extraction contract failed: {metrics}")
        _complete_stage(output, state, "feature_extraction", metrics)

    if not _stage_complete(state, "sequential_matching"):
        metrics = _database_metrics(raw_db)
        if not metrics or metrics.get("matches_rows", 0) <= 0:
            _run_logged(
                [
                    "pixi",
                    "run",
                    "colmap",
                    "sequential_matcher",
                    "--database_path",
                    str(raw_db),
                    "--FeatureMatching.type",
                    "ALIKED_LIGHTGLUE",
                    "--SequentialMatching.overlap",
                    str(sequential_overlap),
                ],
                COLMAP_REPO,
                logs / "sequential_matcher.log",
            )
            metrics = _database_metrics(raw_db)
        if not metrics or metrics.get("matches_rows", 0) <= 0:
            raise RuntimeError(f"sequential matching contract failed: {metrics}")
        _complete_stage(output, state, "sequential_matching", metrics)

    filtered_db = output / "colmap" / "filtered.db"
    if not _stage_complete(state, "rgbd_filter"):
        filter_summary_path = output / "colmap" / "match_filter_summary.json"
        summary = _existing_filter_summary(raw_db, filtered_db, filter_summary_path)
        if summary is None:
            if filtered_db.exists():
                raise RuntimeError(
                    f"partial or stale filtered database exists; preserving it for inspection: {filtered_db}"
                )
            summary = filter_colmap_matches(
                raw_db=raw_db,
                filtered_db=filtered_db,
                manifest_path=output / "manifest.json",
                vggt_dir=output / "vggt4d_chunks",
                trajectory_path=output / "trajectory" / "aligned_trajectory.json",
                min_depth_m=0.1,
                max_depth_m=1.3,
                absolute_tolerance_m=0.05,
                relative_tolerance=0.05,
                missing_depth_policy="keep",
            )
        _complete_stage(output, state, "rgbd_filter", summary["totals"])

    if not _stage_complete(state, "geometric_verification"):
        metrics = _database_metrics(filtered_db)
        if not metrics or metrics.get("two_view_geometries_rows", 0) <= 0:
            _run_logged(
                ["pixi", "run", "colmap", "geometric_verifier", "--database_path", str(filtered_db)],
                COLMAP_REPO,
                logs / "geometric_verifier.log",
            )
            metrics = _database_metrics(filtered_db)
        if not metrics or metrics.get("two_view_geometries_rows", 0) <= 0:
            raise RuntimeError(f"geometric verification contract failed: {metrics}")
        _complete_stage(output, state, "geometric_verification", metrics)

    rail_db = output / "colmap" / "rail_prior.db"
    if not _stage_complete(state, "rail_prior_database"):
        rail_summary_path = rail_db.with_suffix(".rail_priors.json")
        rail_summary = json.loads(rail_summary_path.read_text()) if rail_summary_path.is_file() else None
        valid_existing = bool(
            rail_summary
            and _pose_prior_count(rail_db) == frame_count
            and rail_summary.get("input_database_sha256") == _sha256(filtered_db)
            and rail_summary.get("output_database_sha256") == _sha256(rail_db)
        )
        if not valid_existing:
            if rail_db.exists() or rail_summary_path.exists():
                raise RuntimeError(
                    f"partial or stale rail-prior database exists; preserving it for inspection: {rail_db}"
                )
            rail_summary = inject_rail_priors(
                filtered_db,
                rail_db,
                output / "trajectory" / "aligned_trajectory.json",
                along_std_m=0.50,
                cross_std_m=0.001,
                projection_strength=0.90,
            )
        _complete_stage(output, state, "rail_prior_database", rail_summary)

    sparse_dir = output / "colmap" / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    if not _stage_complete(state, "global_mapper"):
        if not _model_complete(sparse_dir / "0"):
            if any(sparse_dir.iterdir()):
                raise RuntimeError(f"partial GLOMAP output exists; preserving it: {sparse_dir}")
            _run_logged(
                [
                    "pixi",
                    "run",
                    "colmap",
                    "global_mapper",
                    "--database_path",
                    str(rail_db),
                    "--image_path",
                    str(output / "images"),
                    "--output_path",
                    str(sparse_dir),
                    "--GlobalMapper.gp_use_gpu",
                    "0",
                    "--GlobalMapper.ba_ceres_use_gpu",
                    "0",
                    "--GlobalMapper.ba_refine_focal_length",
                    "0",
                    "--GlobalMapper.ba_refine_principal_point",
                    "0",
                    "--GlobalMapper.ba_refine_extra_params",
                    "0",
                ],
                COLMAP_REPO,
                logs / "global_mapper.log",
            )
        if not _model_complete(sparse_dir / "0"):
            raise RuntimeError("global_mapper did not produce sparse/0")
        _complete_stage(output, state, "global_mapper", str(sparse_dir / "0"))

    ba_dir = output / "colmap" / "ba"
    ba_dir.mkdir(parents=True, exist_ok=True)
    if not _stage_complete(state, "bundle_adjustment"):
        if not _model_complete(ba_dir):
            if any(ba_dir.iterdir()):
                raise RuntimeError(f"partial bundle-adjustment output exists; preserving it: {ba_dir}")
            _run_logged(
                [
                    "pixi",
                    "run",
                    "colmap",
                    "bundle_adjuster",
                    "--input_path",
                    str(sparse_dir / "0"),
                    "--output_path",
                    str(ba_dir),
                    "--BundleAdjustment.refine_focal_length",
                    "0",
                    "--BundleAdjustment.refine_principal_point",
                    "0",
                    "--BundleAdjustment.refine_extra_params",
                    "0",
                    "--BundleAdjustmentCeres.use_gpu",
                    "0",
                ],
                COLMAP_REPO,
                logs / "bundle_adjuster.log",
            )
        if not _model_complete(ba_dir):
            raise RuntimeError("bundle_adjuster did not produce a readable model")
        _run_logged(
            ["pixi", "run", "colmap", "model_analyzer", "--path", str(ba_dir)],
            COLMAP_REPO,
            logs / "model_after_ba.log",
        )
        _complete_stage(output, state, "bundle_adjustment", _parse_model_analyzer(logs / "model_after_ba.log"))

    rail_ba_dir = output / "colmap" / "rail_ba"
    rail_ba_dir.mkdir(parents=True, exist_ok=True)
    if not _stage_complete(state, "rail_prior_bundle_adjustment"):
        rail_ba_log = logs / "rail_prior_mapper.log"
        valid_existing = _model_complete(rail_ba_dir) and rail_ba_log.is_file()
        if valid_existing:
            valid_existing = "Alignment w.r.t. prior positions failed" not in rail_ba_log.read_text()
        if not valid_existing:
            if any(rail_ba_dir.iterdir()):
                raise RuntimeError(f"partial or invalid rail-prior BA output exists; preserving it: {rail_ba_dir}")
            _run_logged(
                [
                    "pixi",
                    "run",
                    "colmap",
                    "pose_prior_mapper",
                    "--database_path",
                    str(rail_db),
                    "--image_path",
                    str(output / "images"),
                    "--input_path",
                    str(ba_dir),
                    "--output_path",
                    str(rail_ba_dir),
                    "--Mapper.ba_refine_focal_length",
                    "0",
                    "--Mapper.ba_refine_principal_point",
                    "0",
                    "--Mapper.ba_refine_extra_params",
                    "0",
                    "--Mapper.ba_use_gpu",
                    "0",
                    "--use_robust_loss_on_prior_position",
                    "0",
                    "--prior_position_loss_scale",
                    "7.82",
                ],
                COLMAP_REPO,
                rail_ba_log,
            )
        if not _model_complete(rail_ba_dir):
            raise RuntimeError("pose_prior_mapper did not produce a readable model")
        if "Alignment w.r.t. prior positions failed" in rail_ba_log.read_text():
            raise RuntimeError("pose_prior_mapper produced a model but failed to apply rail position priors")
        _run_logged(
            ["pixi", "run", "colmap", "model_analyzer", "--path", str(rail_ba_dir)],
            COLMAP_REPO,
            logs / "model_after_rail_ba.log",
        )
        rail_text_dir = output / "colmap" / "rail_ba_text"
        rail_text_dir.mkdir(parents=True, exist_ok=True)
        _run_logged(
            [
                "pixi",
                "run",
                "colmap",
                "model_converter",
                "--input_path",
                str(rail_ba_dir),
                "--output_path",
                str(rail_text_dir),
                "--output_type",
                "TXT",
            ],
            COLMAP_REPO,
            logs / "rail_ba_model_converter.log",
        )
        rail_metrics = _rail_model_metrics(
            rail_text_dir / "images.txt", output / "trajectory" / "aligned_trajectory.json"
        )
        if rail_metrics["rail_orthogonal_rms_m"] > 0.05:
            raise RuntimeError(f"rail-prior BA exceeds 5 cm orthogonal RMS: {rail_metrics}")
        _write_json(output / "colmap" / "rail_ba_metrics.json", rail_metrics)
        _complete_stage(
            output,
            state,
            "rail_prior_bundle_adjustment",
            {
                "model": _parse_model_analyzer(logs / "model_after_rail_ba.log"),
                "rail": rail_metrics,
            },
        )

    if not _stage_complete(state, "dynamic_map"):
        summary = _existing_dynamic_summary(output)
        if summary is None:
            dynamic_output = output / "dynamic_map" / "dynamic_rgbd_map.ply"
            if dynamic_output.exists() or dynamic_output.with_suffix(".json").exists():
                raise RuntimeError(f"partial VGGT4D dynamic map exists; preserving it: {dynamic_output}")
            summary = export_dynamic_rgbd_map(
                pilot_root=output,
                vggt_dir=output / "vggt4d_chunks",
                trajectory_path=output / "trajectory" / "aligned_trajectory.json",
                output_path=output / "dynamic_map" / "dynamic_rgbd_map.ply",
                min_depth_m=0.1,
                max_depth_m=1.3,
                pixel_stride=8,
                voxel_size_m=0.005,
            )
        _complete_stage(output, state, "dynamic_map", summary)

    if not _stage_complete(state, "opend4rt_dynamic_map"):
        d4rt_output = output / "dynamic_map" / "opend4rt_dynamic_rgbd_map.ply"
        metadata_path = d4rt_output.with_suffix(".json")
        summary = json.loads(metadata_path.read_text()) if metadata_path.is_file() else None
        if not summary or summary.get("covered_frame_count") != frame_count or not d4rt_output.is_file():
            if d4rt_output.exists() or metadata_path.exists():
                raise RuntimeError(f"partial OpenD4RT dynamic map exists; preserving it: {d4rt_output}")
            summary = export_opend4rt_dynamic_map(
                workflow_root=output,
                chunks_dir=output / "opend4rt_chunks",
                trajectory_path=output / "trajectory" / "aligned_trajectory.json",
                output_path=d4rt_output,
                min_depth_m=0.1,
                max_depth_m=1.3,
                voxel_size_m=0.005,
            )
        _complete_stage(output, state, "opend4rt_dynamic_map", summary)

    return _write_full_summary(source, output)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--chunk-overlap", type=int, default=4)
    parser.add_argument("--sequential-overlap", type=int, default=10)
    args = parser.parse_args(argv)
    summary = run_workflow(
        source=args.source,
        output=args.output,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        sequential_overlap=args.sequential_overlap,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
