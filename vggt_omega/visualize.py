"""Rerun visualization for VGGT-Omega inference results.

The blueprint follows the same idea as the Gradio demo (point cloud +
camera frustums + per-frame RGB and depth panels) but uses Rerun so the
result can be streamed live, saved to a ``.rrd`` file, or screenshotted
headlessly via the web viewer.

Available filters mirror :func:`visual_util.predictions_to_glb`:

* Confidence percentile threshold.
* Optional black-/white-background masks.
* Optional depth-edge suppression (relative depth jump test).
* Max-points uniform subsample to keep the viewer responsive.

The helpers accept either a :class:`vggt_omega.SceneResult` or a list of
:class:`vggt_omega.InferenceResult` so callers can plug in whichever
representation they happen to have.
"""

from __future__ import annotations

import colorsys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Bool, Float, UInt8

from .inference import InferenceResult, scene_result_to_inference_results
from .pipeline import SceneResult

DEFAULT_APP_ID = "vggt-omega"


# ---------------------------------------------------------------------------
# Point filtering — matches visual_util.predictions_to_glb behaviour.
# ---------------------------------------------------------------------------


def _depth_edge_mask(depth: Float[np.ndarray, "h w"], rtol: float, kernel: int = 3) -> Bool[np.ndarray, "h w"]:
    pad = kernel // 2
    padded = np.pad(depth, ((pad, pad), (pad, pad)), mode="edge")
    d_max = np.full_like(depth, -np.inf)
    d_min = np.full_like(depth, np.inf)
    for y in range(kernel):
        for x in range(kernel):
            window = padded[y : y + depth.shape[0], x : x + depth.shape[1]]
            d_max = np.maximum(d_max, window)
            d_min = np.minimum(d_min, window)
    return ((d_max - d_min) / np.maximum(np.abs(depth), 1e-6)) > rtol


def _build_keep_mask(
    conf: Float[np.ndarray, "h w"],
    colors: UInt8[np.ndarray, "h w 3"],
    depth: Float[np.ndarray, "h w"] | None,
    conf_percent: float,
    mask_black_bg: bool,
    mask_white_bg: bool,
    filter_depth_edges: bool,
    depth_edge_rtol: float,
) -> Bool[np.ndarray, "h w"]:
    conf_percent = _normalize_conf_percent(conf_percent)
    valid = np.isfinite(conf)
    threshold = np.percentile(conf[valid], conf_percent) if conf_percent > 0 and valid.any() else 0.0
    keep = valid & (conf >= threshold) & (conf > 1e-5)
    if mask_black_bg:
        keep &= colors.sum(axis=-1) >= 16
    if mask_white_bg:
        keep &= ~((colors[..., 0] > 240) & (colors[..., 1] > 240) & (colors[..., 2] > 240))
    if filter_depth_edges and depth is not None:
        keep &= ~_depth_edge_mask(depth, rtol=depth_edge_rtol)
    return keep


def _normalize_conf_percent(conf_percent: float) -> float:
    if not np.isfinite(conf_percent):
        raise ValueError(f"conf_percent must be finite, got {conf_percent!r}")
    return float(np.clip(conf_percent, 0.0, 100.0))


def _subsample(
    points: Float[np.ndarray, "n 3"],
    colors: UInt8[np.ndarray, "n 3"],
    max_points: int,
) -> tuple[Float[np.ndarray, "k 3"], UInt8[np.ndarray, "k 3"]]:
    if max_points <= 0 or len(points) <= max_points:
        return points, colors
    indices = np.linspace(0, len(points) - 1, max_points).astype(np.int64)
    return points[indices], colors[indices]


# ---------------------------------------------------------------------------
# Rerun blueprint + per-frame logging.
# ---------------------------------------------------------------------------


def make_blueprint() -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="world", contents=["world/**"], name="3D"),
            rrb.Vertical(
                rrb.Spatial2DView(origin="frame/image", name="image"),
                rrb.Spatial2DView(origin="frame/depth", name="depth"),
                rrb.Spatial2DView(origin="frame/confidence", name="confidence"),
            ),
        ),
        collapse_panels=True,
    )


def _log_static_world() -> None:
    rr.log(
        "world",
        rr.Transform3D(),
        static=True,
    )


def _log_camera_frustum(idx: int, result: InferenceResult, color: tuple[int, int, int]) -> None:
    extrinsic = np.eye(4)
    extrinsic[:3, :4] = result.extrinsic
    cam_to_world = np.linalg.inv(extrinsic)
    rr.log(
        f"world/camera_{idx:03d}",
        rr.Transform3D(translation=cam_to_world[:3, 3], mat3x3=cam_to_world[:3, :3]),
        static=True,
    )
    rr.log(
        f"world/camera_{idx:03d}",
        rr.Pinhole(
            image_from_camera=result.intrinsic,
            width=result.width,
            height=result.height,
            image_plane_distance=0.05,
        ),
        static=True,
    )
    rr.log(
        f"world/camera_{idx:03d}/cone",
        rr.Arrows3D(
            origins=[[0, 0, 0]],
            vectors=[[0, 0, 0.05]],
            colors=[color],
            radii=[0.001],
        ),
        static=True,
    )


def _frame_color(idx: int, total: int) -> tuple[int, int, int]:
    # Map frame index to a distinct hue without pulling matplotlib into this module.
    r, g, b = colorsys.hsv_to_rgb((idx / max(total, 1)) % 1.0, 0.85, 1.0)
    return int(255 * r), int(255 * g), int(255 * b)


def _log_frame(
    idx: int,
    result: InferenceResult,
    color: tuple[int, int, int],
    conf_percent: float,
    mask_black_bg: bool,
    mask_white_bg: bool,
    filter_depth_edges: bool,
    depth_edge_rtol: float,
    max_points: int,
    accumulate_points: bool,
) -> None:
    rr.set_time("frame", sequence=idx)

    _log_camera_frustum(idx, result, color)

    keep = _build_keep_mask(
        conf=result.depth_conf,
        colors=result.image,
        depth=result.depth_map,
        conf_percent=conf_percent,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        filter_depth_edges=filter_depth_edges,
        depth_edge_rtol=depth_edge_rtol,
    )
    pts = result.point_map_by_unprojection.reshape(-1, 3)
    rgb = result.image.reshape(-1, 3)
    keep_flat = keep.reshape(-1) & np.isfinite(pts).all(axis=1)
    pts_kept, rgb_kept = _subsample(pts[keep_flat], rgb[keep_flat], max_points)

    rr.log(
        f"world/points_{idx:03d}" if accumulate_points else "world/points",
        rr.Points3D(pts_kept, colors=rgb_kept, radii=0.0005),
        static=accumulate_points,
    )

    rr.log("frame/image", rr.Image(result.image))
    depth_viz = result.depth_map.copy()
    depth_viz[~keep] = 0.0
    rr.log("frame/depth", rr.DepthImage(depth_viz))
    rr.log("frame/confidence", rr.DepthImage(result.depth_conf.astype(np.float32)))


def _coerce_results(results: SceneResult | Sequence[InferenceResult]) -> list[InferenceResult]:
    if isinstance(results, SceneResult):
        coerced = scene_result_to_inference_results(results)
    else:
        coerced = list(results)
    if not coerced:
        raise ValueError("At least one inference result is required for visualization")
    return coerced


def _log_all_frames(
    results: list[InferenceResult],
    conf_percent: float,
    mask_black_bg: bool,
    mask_white_bg: bool,
    filter_depth_edges: bool,
    depth_edge_rtol: float,
    max_points: int,
    accumulate_points: bool,
    frame_offset: int,
) -> None:
    _log_static_world()
    for local_idx, result in enumerate(results):
        frame_idx = frame_offset + local_idx
        _log_frame(
            frame_idx,
            result,
            color=_frame_color(frame_idx, max(frame_offset + len(results), len(results))),
            conf_percent=conf_percent,
            mask_black_bg=mask_black_bg,
            mask_white_bg=mask_white_bg,
            filter_depth_edges=filter_depth_edges,
            depth_edge_rtol=depth_edge_rtol,
            max_points=max_points,
            accumulate_points=accumulate_points,
        )


# ---------------------------------------------------------------------------
# Public entrypoints.
# ---------------------------------------------------------------------------


def visualize_results(
    results: SceneResult | Sequence[InferenceResult],
    *,
    conf_percent: float = 50.0,
    mask_black_bg: bool = False,
    mask_white_bg: bool = False,
    filter_depth_edges: bool = True,
    depth_edge_rtol: float = 0.03,
    max_points: int = 1_000_000,
    accumulate_points: bool = True,
    frame_offset: int = 0,
    app_id: str = DEFAULT_APP_ID,
    spawn: bool = True,
) -> None:
    """Stream the visualization to a (locally-spawned) Rerun viewer."""
    rr.init(app_id, spawn=spawn)
    rr.send_blueprint(make_blueprint())
    _log_all_frames(
        _coerce_results(results),
        conf_percent=conf_percent,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        filter_depth_edges=filter_depth_edges,
        depth_edge_rtol=depth_edge_rtol,
        max_points=max_points,
        accumulate_points=accumulate_points,
        frame_offset=frame_offset,
    )


def save_results_to_rrd(
    results: SceneResult | Sequence[InferenceResult],
    rrd_path: str | Path,
    *,
    conf_percent: float = 50.0,
    mask_black_bg: bool = False,
    mask_white_bg: bool = False,
    filter_depth_edges: bool = True,
    depth_edge_rtol: float = 0.03,
    max_points: int = 1_000_000,
    accumulate_points: bool = True,
    frame_offset: int = 0,
    app_id: str = DEFAULT_APP_ID,
) -> Path:
    """Write the same visualization to an ``.rrd`` file without opening a viewer."""
    rrd_path = Path(rrd_path)
    rrd_path.parent.mkdir(parents=True, exist_ok=True)
    rr.init(app_id, spawn=False)
    rr.save(str(rrd_path), default_blueprint=make_blueprint())
    _log_all_frames(
        _coerce_results(results),
        conf_percent=conf_percent,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        filter_depth_edges=filter_depth_edges,
        depth_edge_rtol=depth_edge_rtol,
        max_points=max_points,
        accumulate_points=accumulate_points,
        frame_offset=frame_offset,
    )
    rr.disconnect()
    return rrd_path
