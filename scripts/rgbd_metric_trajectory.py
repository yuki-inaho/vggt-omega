"""Estimate a metric camera trajectory from an RGB-D sequence with VGGT-Omega.

VGGT-Omega is a monocular (RGB-only) feed-forward model: it predicts depth and
camera poses in an arbitrary, internally-consistent scale. This script recovers
the *metric* (meter) scale of that trajectory using the sensor depth and the
real camera parameters that ship alongside the RGB frames.

Two phases:
  Phase 1  Run VGGT-Omega on an RGB sequence and read the camera trajectory in
           VGGT's own (arbitrary) scale.
  Phase 2  Reproject the metric sensor depth into the RGB field of view using
           the real intrinsics (K_depth, K_rgb) and the depth->rgb extrinsic,
           compare it to VGGT's predicted depth per valid pixel, and derive a
           single global scale factor. Apply it to the trajectory.

Expected on-disk layout (pass --data-root and --frame-name)::

    <data-root>/<frame-name>/
        rgb/<stem>.jpg|png         RGB frames (HxWx3, uint8)
        depth/<stem>.png           sensor depth (uint16); value * 1/depth-scale = meters
        camera_parameters/
            rgb_camera_param.yaml    contains K (3x3, row-major list)
            depth_camera_param.yaml  contains K and P (3x4 [R|t], depth->rgb)

The depth->rgb extrinsic is read from the depth camera's ``P`` (3x4 [R|t],
mapping a point in the depth optical frame to the rgb optical frame). Override
with --identity-extrinsic if your depth is already registered to the RGB FoV.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

RGB_EXTS = (".jpg", ".jpeg", ".png")


# --------------------------------------------------------------------------- #
# Camera-parameter IO
# --------------------------------------------------------------------------- #
def load_intrinsics(yaml_path: Path) -> np.ndarray:
    """Read a 3x3 intrinsic matrix K from a ROS-style camera_param YAML."""
    import yaml  # lazy: only needed when actually running on data

    data = yaml.safe_load(yaml_path.read_text())
    return np.asarray(data["K"], dtype=float).reshape(3, 3)


def load_depth_to_rgb(depth_yaml_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read depth->rgb extrinsic [R|t] from the depth camera's ``P`` (3x4)."""
    import yaml

    data = yaml.safe_load(depth_yaml_path.read_text())
    P = np.asarray(data["P"], dtype=float).reshape(3, 4)
    return P[:, :3], P[:, 3]


# --------------------------------------------------------------------------- #
# Geometry (pure numpy, unit-testable without a GPU / checkpoint)
# --------------------------------------------------------------------------- #
def align_sensor_depth_to_rgb(
    depth_m: np.ndarray,
    valid: np.ndarray,
    K_depth: np.ndarray,
    K_rgb: np.ndarray,
    R_d2r: np.ndarray,
    t_d2r: np.ndarray,
    rgb_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Reproject metric depth (depth FoV) onto the RGB image grid.

    Returns ``(depth_on_rgb_grid, mask)`` in meters, using a nearest-surface
    z-buffer so the closest sample wins per RGB pixel.
    """
    vs, us = np.nonzero(valid)
    zz = depth_m[valid]
    fx_d, fy_d, cx_d, cy_d = K_depth[0, 0], K_depth[1, 1], K_depth[0, 2], K_depth[1, 2]
    pts = np.stack([(us - cx_d) / fx_d * zz, (vs - cy_d) / fy_d * zz, zz], axis=0)
    pts_r = R_d2r @ pts + t_d2r[:, None]
    zr = pts_r[2]
    front = zr > 1e-6
    pts_r, zr = pts_r[:, front], zr[front]

    fx_r, fy_r, cx_r, cy_r = K_rgb[0, 0], K_rgb[1, 1], K_rgb[0, 2], K_rgb[1, 2]
    ur = np.round(fx_r * pts_r[0] / zr + cx_r).astype(np.int64)
    vr = np.round(fy_r * pts_r[1] / zr + cy_r).astype(np.int64)

    h, w = rgb_hw
    inb = (ur >= 0) & (ur < w) & (vr >= 0) & (vr < h)
    ur, vr, zr = ur[inb], vr[inb], zr[inb]

    buf = np.full(h * w, np.inf)
    np.minimum.at(buf, vr * w + ur, zr)
    buf = buf.reshape(h, w)
    mask = np.isfinite(buf)
    buf[~mask] = 0.0
    return buf, mask


def camera_centers(extrinsic: np.ndarray) -> np.ndarray:
    """world->camera [R|t] (N,3,4) -> camera centers in world coords (N,3)."""
    R = extrinsic[:, :, :3]
    t = extrinsic[:, :, 3]
    return -np.einsum("nij,nj->ni", np.transpose(R, (0, 2, 1)), t)


def path_length(centers: np.ndarray) -> float:
    """Summed Euclidean length of a polyline through the camera centers."""
    if len(centers) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(centers, axis=0), axis=1).sum())


# --------------------------------------------------------------------------- #
# Pipeline glue
# --------------------------------------------------------------------------- #
def select_stems(rgb_dir: Path, num_frames: int) -> list[str]:
    stems = sorted({p.stem for p in rgb_dir.iterdir() if p.suffix.lower() in RGB_EXTS})
    if not stems:
        raise SystemExit(f"No RGB frames found under {rgb_dir}")
    if num_frames <= 0 or num_frames >= len(stems):
        return stems
    idx = np.unique(np.linspace(0, len(stems) - 1, num_frames).round().astype(int))
    return [stems[i] for i in idx]


def read_rgb(rgb_dir: Path, stem: str) -> np.ndarray:
    for ext in RGB_EXTS:
        p = rgb_dir / f"{stem}{ext}"
        if p.exists():
            return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
    raise FileNotFoundError(f"No RGB image for stem {stem} under {rgb_dir}")


def run_vggt(frames_rgb: list[np.ndarray], checkpoint: str, image_resolution: int):
    from vggt_omega.pipeline import VGGTOmegaPipeline  # lazy: heavy / GPU deps
    from vggt_omega.preprocess import preprocess_images

    images = preprocess_images(frames_rgb, image_resolution=image_resolution)
    scene = VGGTOmegaPipeline(checkpoint).run(images)
    return images, scene


def plot_trajectory(centers: np.ndarray, scale: float, out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    ax.plot(centers[:, 0], centers[:, 1], centers[:, 2], "-o", ms=3)
    ax.scatter(*centers[0], c="g", s=40, label="start")
    ax.scatter(*centers[-1], c="r", s=40, label="end")
    ax.set_title(f"3D camera trajectory (metric, s={scale:.3f})")
    ax.set_xlabel("X[m]")
    ax.set_ylabel("Y[m]")
    ax.set_zlabel("Z[m]")
    ax.legend()
    for k, (a, b, la, lb) in enumerate([(0, 2, "X[m]", "Z[m]"), (0, 1, "X[m]", "Y[m]"), (2, 1, "Z[m]", "Y[m]")]):
        axp = fig.add_subplot(2, 2, k + 2)
        axp.plot(centers[:, a], centers[:, b], "-o", ms=3)
        axp.scatter(centers[0, a], centers[0, b], c="g", s=40)
        axp.scatter(centers[-1, a], centers[-1, b], c="r", s=40)
        axp.set_xlabel(la)
        axp.set_ylabel(lb)
        axp.set_aspect("equal", "datalim")
        axp.grid(True, alpha=0.3)
        axp.set_title(f"{la[0]}-{lb[0]} projection")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, help="Directory containing <frame-name>/")
    ap.add_argument("--frame-name", required=True, help="Sequence directory under --data-root")
    ap.add_argument("--checkpoint", default="checkpoints/vggt_omega_1b_512.pt")
    ap.add_argument("--num-frames", type=int, default=8, help="Evenly sampled frame count (<=0 = all)")
    ap.add_argument("--image-resolution", type=int, default=512)
    ap.add_argument("--out-dir", default="outputs/rgbd_trajectory")
    ap.add_argument("--depth-scale", type=float, default=1000.0, help="sensor units per meter (uint16 mm -> 1000)")
    ap.add_argument("--depth-invalid", type=int, default=65535, help="sensor value treated as invalid")
    ap.add_argument("--zmin", type=float, default=0.1)
    ap.add_argument("--zmax", type=float, default=5.0)
    ap.add_argument("--conf-percentile", type=float, default=50.0, help="keep VGGT depth above this confidence pct")
    ap.add_argument("--identity-extrinsic", action="store_true", help="depth already registered to RGB FoV")
    args = ap.parse_args(argv)

    base = Path(args.data_root) / args.frame_name
    rgb_dir, depth_dir, cam = base / "rgb", base / "depth", base / "camera_parameters"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stems = select_stems(rgb_dir, args.num_frames)
    print(f"[setup] {len(stems)} frames: {stems}")

    K_rgb = load_intrinsics(cam / "rgb_camera_param.yaml")
    K_depth = load_intrinsics(cam / "depth_camera_param.yaml")
    if args.identity_extrinsic:
        R_d2r, t_d2r = np.eye(3), np.zeros(3)
    else:
        R_d2r, t_d2r = load_depth_to_rgb(cam / "depth_camera_param.yaml")
    print(f"[setup] depth->rgb |t|={np.linalg.norm(t_d2r) * 1000:.1f}mm")

    # ---- Phase 1: VGGT trajectory (RGB only, arbitrary scale) ----
    frames_rgb = [read_rgb(rgb_dir, s) for s in stems]
    rgb_hw = frames_rgb[0].shape[:2]
    images, scene = run_vggt(frames_rgb, args.checkpoint, args.image_resolution)
    hv, wv = scene.depth.shape[1], scene.depth.shape[2]
    centers_raw = camera_centers(scene.extrinsic)
    print(
        f"[phase1] vggt input {tuple(images.shape)} depth grid {hv}x{wv} "
        f"raw path={path_length(centers_raw):.4f} (vggt units)"
    )

    # ---- Phase 2: metric scale from aligned sensor depth ----
    pred_depth, pred_conf = scene.depth[..., 0], scene.depth_conf
    scales: list[float] = []
    valid_px: list[int] = []
    for i, stem in enumerate(stems):
        raw = cv2.imread(str(depth_dir / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        depth_m = raw.astype(np.float64) / args.depth_scale
        valid = (raw > 0) & (raw != args.depth_invalid) & (depth_m >= args.zmin) & (depth_m <= args.zmax)
        aligned, mask = align_sensor_depth_to_rgb(depth_m, valid, K_depth, K_rgb, R_d2r, t_d2r, rgb_hw)
        a = cv2.resize(aligned.astype(np.float32), (wv, hv), interpolation=cv2.INTER_NEAREST)
        m = cv2.resize(mask.astype(np.uint8), (wv, hv), interpolation=cv2.INTER_NEAREST).astype(bool)
        conf_thr = np.percentile(pred_conf[i], args.conf_percentile)
        good = m & (a > 0) & (pred_depth[i] > 1e-6) & (pred_conf[i] >= conf_thr)
        if good.sum() < 200:
            scales.append(np.nan)
            valid_px.append(int(good.sum()))
            continue
        s = float(np.median(a[good] / pred_depth[i][good]))
        scales.append(s)
        valid_px.append(int(good.sum()))
        print(f"[phase2] {stem}: valid_px={good.sum():6d} scale={s:.4f}")

    arr = np.array(scales, dtype=float)
    ok = arr[np.isfinite(arr)]
    if ok.size == 0:
        raise SystemExit("No frame produced a usable scale; check depth units / masking / camera params.")
    s_global = float(np.median(ok))
    mad = float(np.median(np.abs(ok - s_global)))
    centers_metric = centers_raw * s_global

    print("\n=== SCALE SUMMARY ===")
    print(f"global scale s = {s_global:.4f} m/unit  (MAD {mad:.4f} = {mad / s_global * 100:.1f}% of s)")
    print(f"raw path  = {path_length(centers_raw):.4f} vggt-units")
    print(f"metric path = {path_length(centers_metric):.4f} m")
    print(f"metric bbox (m) = {np.round(centers_metric.max(0) - centers_metric.min(0), 3).tolist()}")

    np.savez(
        out_dir / "raw_trajectory.npz",
        stems=stems,
        extrinsic=scene.extrinsic,
        intrinsic=scene.intrinsic,
        centers=centers_raw,
    )
    np.savez(
        out_dir / "metric_trajectory.npz",
        stems=stems,
        centers_metric=centers_metric,
        scale=s_global,
        per_frame_scale=arr,
    )
    (out_dir / "scale_report.json").write_text(
        json.dumps(
            {
                "frames": stems,
                "vggt_depth_grid": [int(hv), int(wv)],
                "scale_global": s_global,
                "scale_mad": mad,
                "scale_rel_spread": mad / s_global,
                "per_frame_scale": [None if not np.isfinite(x) else x for x in arr],
                "per_frame_valid_px": valid_px,
                "raw_path_len_vggt": path_length(centers_raw),
                "metric_path_len_m": path_length(centers_metric),
                "depth_to_rgb_baseline_mm": float(np.linalg.norm(t_d2r) * 1000),
            },
            indent=2,
        )
    )
    plot_trajectory(centers_metric, s_global, out_dir / "trajectory_plot.png")
    print(f"\n[done] artifacts -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
