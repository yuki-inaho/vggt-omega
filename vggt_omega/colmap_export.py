"""COLMAP text export helpers for VGGT-Omega scene results."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .pipeline import SceneResult
from .utils.rotation import mat_to_quat


@dataclass(frozen=True)
class PinholeCamera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def collect_image_paths(images_dir: str | Path, max_images: int | None = None) -> list[Path]:
    root = Path(images_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {root}")
    image_paths = sorted(p for p in root.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if max_images is not None and max_images > 0:
        image_paths = image_paths[:max_images]
    if not image_paths:
        raise FileNotFoundError(f"No images found under {root}")
    return image_paths


def load_camera_from_dataset_info(path: str | Path) -> PinholeCamera:
    dataset_info_path = Path(path)
    with dataset_info_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    camera = payload.get("camera")
    if not isinstance(camera, dict):
        raise ValueError(f"dataset_info.json does not contain a camera mapping: {dataset_info_path}")
    return PinholeCamera(
        width=int(camera["width"]),
        height=int(camera["height"]),
        fx=float(camera["fx"]),
        fy=float(camera["fy"]),
        cx=float(camera["cx"]),
        cy=float(camera["cy"]),
    )


def camera_from_intrinsics(intrinsic: np.ndarray, width: int, height: int) -> PinholeCamera:
    return PinholeCamera(
        width=int(width),
        height=int(height),
        fx=float(intrinsic[0, 0]),
        fy=float(intrinsic[1, 1]),
        cx=float(intrinsic[0, 2]),
        cy=float(intrinsic[1, 2]),
    )


def rotmat_to_colmap_qvec(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to COLMAP Hamilton (qw, qx, qy, qz)."""
    tensor = torch.as_tensor(rotation, dtype=torch.float32).reshape(1, 3, 3)
    q_xyzw = mat_to_quat(tensor)[0].detach().cpu().numpy().astype(np.float64)
    qx, qy, qz, qw = (float(v) for v in q_xyzw)
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm == 0 or not math.isfinite(norm):
        raise ValueError("Invalid zero or non-finite quaternion")
    return qw / norm, qx / norm, qy / norm, qz / norm


def validate_scene_pose(scene: SceneResult, expected_count: int) -> dict[str, Any]:
    extrinsic = np.asarray(scene.extrinsic)
    if extrinsic.shape != (expected_count, 3, 4):
        raise ValueError(f"Expected extrinsic shape {(expected_count, 3, 4)}, got {extrinsic.shape}")
    finite = bool(np.isfinite(extrinsic).all())
    rotations = extrinsic[:, :3, :3]
    translations = extrinsic[:, :3, 3]
    identity = np.eye(3, dtype=np.float32)
    all_identity_rot = bool(np.allclose(rotations, identity[None], atol=1e-5))
    all_zero_t = bool(np.allclose(translations, 0.0, atol=1e-7))
    q_norms = []
    for rotation in rotations:
        qvec = rotmat_to_colmap_qvec(rotation)
        q_norms.append(float(np.linalg.norm(np.asarray(qvec, dtype=np.float64))))
    return {
        "num_poses": int(expected_count),
        "extrinsic_finite": finite,
        "all_identity_rotation": all_identity_rot,
        "all_zero_translation": all_zero_t,
        "all_identity_pose": bool(all_identity_rot and all_zero_t),
        "quaternion_norm_min": float(min(q_norms)),
        "quaternion_norm_max": float(max(q_norms)),
        "translation_norm_min": float(np.linalg.norm(translations, axis=1).min()),
        "translation_norm_max": float(np.linalg.norm(translations, axis=1).max()),
    }


def write_colmap_text(
    scene: SceneResult,
    image_names: Sequence[str],
    output_dir: str | Path,
    camera: PinholeCamera,
) -> None:
    if len(image_names) != int(scene.extrinsic.shape[0]):
        raise ValueError(f"image_names count {len(image_names)} does not match scene poses {scene.extrinsic.shape[0]}")

    sparse_dir = Path(output_dir)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    with (sparse_dir / "cameras.txt").open("w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        f.write(
            "1 PINHOLE "
            f"{camera.width} {camera.height} "
            f"{camera.fx:.12g} {camera.fy:.12g} {camera.cx:.12g} {camera.cy:.12g}\n"
        )

    with (sparse_dir / "images.txt").open("w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(image_names)}, mean observations per image: 0\n")
        for image_id, (name, extrinsic) in enumerate(zip(image_names, scene.extrinsic, strict=True), start=1):
            qvec = rotmat_to_colmap_qvec(extrinsic[:3, :3])
            tvec = [float(v) for v in extrinsic[:3, 3]]
            f.write(
                f"{image_id} "
                f"{qvec[0]:.17g} {qvec[1]:.17g} {qvec[2]:.17g} {qvec[3]:.17g} "
                f"{tvec[0]:.17g} {tvec[1]:.17g} {tvec[2]:.17g} "
                f"1 {name}\n"
            )
            f.write("\n")

    with (sparse_dir / "points3D.txt").open("w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: 0, mean track length: 0\n")


def copy_images(image_paths: Sequence[Path], output_images_dir: str | Path) -> None:
    dst_root = Path(output_images_dir)
    dst_root.mkdir(parents=True, exist_ok=True)
    for path in image_paths:
        shutil.copy2(path, dst_root / path.name)


def export_scene_outputs(
    scene: SceneResult,
    image_paths: Sequence[Path],
    output_root: str | Path,
    camera: PinholeCamera,
    copy_input_images: bool = False,
    run_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output = Path(output_root)
    sparse_dir = output / "sparse" / "0"
    output.mkdir(parents=True, exist_ok=True)

    npz_path = output / "predictions.npz"
    np.savez_compressed(npz_path, **scene.as_npz_dict())
    write_colmap_text(scene, [p.name for p in image_paths], sparse_dir, camera)
    if copy_input_images:
        copy_images(image_paths, output / "images")

    pose_summary = validate_scene_pose(scene, len(image_paths))
    summary = {
        "created_at": utc_now(),
        "num_images": len(image_paths),
        "image_names": [p.name for p in image_paths],
        "camera": camera.__dict__,
        "run_settings": run_settings or {},
        "paths": {
            "output_root": str(output),
            "predictions_npz": str(npz_path),
            "sparse_dir": str(sparse_dir),
            "images_dir": str(output / "images") if copy_input_images else None,
        },
        "pose_summary": pose_summary,
    }
    with (output / "export_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary
