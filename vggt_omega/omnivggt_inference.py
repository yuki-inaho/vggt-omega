# SPDX-License-Identifier: Apache-2.0
"""Thin, reusable adapter around an external official OmniVGGT checkout."""

from __future__ import annotations

import hashlib
import importlib
import math
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import torch
from PIL import Image

from .rgbd_viewer import LoadedRgbdFrame

OFFICIAL_CHECKPOINT_SHA256 = "c9c3772b9bbfc648fa95ebffb3d9ff856f21e9b9712685eb0f98add53897969d"


class OmniVggtInferenceError(RuntimeError):
    """Raised when the external runtime or an inference result violates its contract."""


@dataclass(frozen=True)
class OmniVggtRuntimeConfig:
    """Locations and device selection for the external official implementation."""

    official_repository: Path
    checkpoint: Path
    device: str = "cuda"
    expected_checkpoint_sha256: str | None = OFFICIAL_CHECKPOINT_SHA256


@dataclass(frozen=True)
class PreparedOmniVggtInput:
    """Official OmniVGGT inference tensors plus stable frame metadata."""

    frame_ids: tuple[str, ...]
    images: torch.Tensor
    depth: torch.Tensor
    mask: torch.Tensor
    extrinsics: torch.Tensor
    intrinsics: torch.Tensor
    depth_gt_index: tuple[int, ...]
    camera_gt_index: tuple[int, ...]

    @property
    def image_size_hw(self) -> tuple[int, int]:
        return int(self.images.shape[-2]), int(self.images.shape[-1])


@dataclass(frozen=True)
class OmniVggtInferenceResult:
    """UI-neutral OmniVGGT outputs and an exported point-cloud scene."""

    gallery: tuple[tuple[Image.Image, str], ...]
    frame_statistics: tuple[tuple[object, ...], ...]
    camera_statistics: tuple[tuple[object, ...], ...]
    glb_path: Path
    exported_points: int
    inference_seconds: float = 0.0


def prepare_omnivggt_input(
    frames: Sequence[LoadedRgbdFrame],
    *,
    target_size: int = 518,
    patch_size: int = 14,
) -> PreparedOmniVggtInput:
    """Resize paired RGB-D frames using the official width/crop convention."""

    if not frames:
        raise OmniVggtInferenceError("at least one RGB-D frame is required for OmniVGGT inference")
    if target_size <= 0 or patch_size <= 0 or target_size % patch_size:
        raise OmniVggtInferenceError("target_size must be a positive multiple of patch_size")
    source_shape = frames[0].rgb.shape[:2]
    if any(frame.rgb.shape[:2] != source_shape for frame in frames):
        raise OmniVggtInferenceError("all selected RGB-D frames must share one source resolution")

    source_height, source_width = source_shape
    resized_height = max(
        patch_size,
        round(source_height * (target_size / source_width) / patch_size) * patch_size,
    )
    crop_start = max(0, (resized_height - target_size) // 2)
    final_height = min(resized_height, target_size)

    image_tensors: list[torch.Tensor] = []
    depths: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for frame in frames:
        resized_rgb = cv2.resize(frame.rgb, (target_size, resized_height), interpolation=cv2.INTER_CUBIC)
        resized_depth = cv2.resize(frame.depth_m, (target_size, resized_height), interpolation=cv2.INTER_NEAREST)
        resized_mask = cv2.resize(
            frame.valid_mask.astype(np.uint8),
            (target_size, resized_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.bool_)
        if resized_height > target_size:
            row_slice = slice(crop_start, crop_start + target_size)
            resized_rgb = resized_rgb[row_slice]
            resized_depth = resized_depth[row_slice]
            resized_mask = resized_mask[row_slice]
        if resized_rgb.shape[:2] != (final_height, target_size):
            raise OmniVggtInferenceError("RGB-D preprocessing produced an unexpected image shape")
        resized_depth = np.where(resized_mask, resized_depth, 0).astype(np.float32, copy=False)
        image_tensors.append(torch.from_numpy(resized_rgb.copy()).permute(2, 0, 1).float().div_(255))
        depths.append(resized_depth)
        masks.append(resized_mask.astype(np.float32))

    images = torch.stack(image_tensors)
    depth = torch.from_numpy(np.stack(depths))[None, ..., None]
    mask = torch.from_numpy(np.stack(masks))[None]
    frame_count = len(frames)
    return PreparedOmniVggtInput(
        frame_ids=tuple(frame.pair.frame_id for frame in frames),
        images=images,
        depth=depth,
        mask=mask,
        extrinsics=torch.zeros(1, frame_count, 3, 4),
        intrinsics=torch.zeros(1, frame_count, 3, 3),
        depth_gt_index=tuple(range(frame_count)),
        camera_gt_index=(),
    )


def run_omnivggt_model(
    model: Any,
    prepared: PreparedOmniVggtInput,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Run the official inference contract and detach tensor outputs to CPU."""

    inputs = {
        "images": prepared.images.to(device),
        "extrinsics": prepared.extrinsics.to(device),
        "intrinsics": prepared.intrinsics.to(device),
        "depth": prepared.depth.to(device),
        "mask": prepared.mask.to(device),
        "depth_gt_index": list(prepared.depth_gt_index),
        "camera_gt_index": list(prepared.camera_gt_index),
    }
    with torch.inference_mode():
        predictions = model.inference(**inputs)
    if not isinstance(predictions, Mapping):
        raise OmniVggtInferenceError("OmniVGGT inference must return a prediction mapping")
    return {
        str(key): value.detach().float().cpu() for key, value in predictions.items() if isinstance(value, torch.Tensor)
    }


def render_omnivggt_predictions(
    prepared: PreparedOmniVggtInput,
    predictions: Mapping[str, torch.Tensor],
    *,
    pose_decoder: Callable[[torch.Tensor, tuple[int, int]], tuple[torch.Tensor, torch.Tensor]],
    output_directory: str | Path,
    confidence_percentile: float,
    max_points: int,
    inference_seconds: float = 0.0,
) -> OmniVggtInferenceResult:
    """Render model outputs, compute scale-aware comparison metrics, and export GLB."""

    if not 0 <= confidence_percentile <= 100:
        raise OmniVggtInferenceError("confidence_percentile must be between 0 and 100")
    if max_points <= 0:
        raise OmniVggtInferenceError("max_points must be positive")
    required = {"depth", "depth_conf", "world_points", "pose_enc"}
    missing = sorted(required.difference(predictions))
    if missing:
        raise OmniVggtInferenceError(f"OmniVGGT predictions are missing: {', '.join(missing)}")

    frame_count = len(prepared.frame_ids)
    height, width = prepared.image_size_hw
    predicted_depth = _scalar_frames(predictions["depth"], frame_count, height, width, "depth")
    depth_confidence = _scalar_frames(predictions["depth_conf"], frame_count, height, width, "depth_conf")
    world_points = _vector_frames(predictions["world_points"], frame_count, height, width, "world_points")
    point_confidence = _scalar_frames(
        predictions.get("world_points_conf", predictions["depth_conf"]),
        frame_count,
        height,
        width,
        "world_points_conf",
    )
    extrinsics, intrinsics = pose_decoder(predictions["pose_enc"], (height, width))
    extrinsics_np = _camera_array(extrinsics, frame_count, (3, 4), "extrinsics")
    intrinsics_np = _camera_array(intrinsics, frame_count, (3, 3), "intrinsics")

    input_depth = prepared.depth.numpy()[0, ..., 0]
    input_mask = prepared.mask.numpy()[0] > 0
    rgb = prepared.images.permute(0, 2, 3, 1).numpy()
    gallery: list[tuple[Image.Image, str]] = []
    frame_statistics: list[tuple[object, ...]] = []
    for index, frame_id in enumerate(prepared.frame_ids):
        valid = input_mask[index] & np.isfinite(predicted_depth[index]) & (predicted_depth[index] > 0)
        input_values = input_depth[index][valid]
        predicted_values = predicted_depth[index][valid]
        if input_values.size:
            input_median = float(np.median(input_values))
            predicted_median = float(np.median(predicted_values))
            scale = input_median / max(predicted_median, 1e-8)
            aligned_error = np.abs(predicted_depth[index] * scale - input_depth[index])
            aligned_rmse = float(np.sqrt(np.mean(np.square(aligned_error[valid]))))
        else:
            input_median = predicted_median = scale = aligned_rmse = math.nan
            aligned_error = np.zeros((height, width), dtype=np.float32)

        predicted_color = _colorize(predicted_depth[index], predicted_depth[index] > 0)
        confidence_color = _colorize(depth_confidence[index], np.isfinite(depth_confidence[index]))
        overlay = _overlay(rgb[index], predicted_color, 0.48)
        error_color = _colorize(aligned_error, valid)
        label = Path(frame_id).name
        gallery.extend(
            (
                (Image.fromarray(predicted_color), f"{label} · OmniVGGT predicted depth"),
                (Image.fromarray(confidence_color), f"{label} · predicted confidence"),
                (Image.fromarray(overlay), f"{label} · RGB + predicted depth"),
                (Image.fromarray(error_color), f"{label} · scale-aligned |prediction - mapped depth|"),
            )
        )
        frame_statistics.append(
            (
                frame_id,
                round(float(valid.mean() * 100), 3),
                round(input_median, 6),
                round(predicted_median, 6),
                round(scale, 6),
                round(aligned_rmse, 6),
                round(float(np.median(depth_confidence[index][np.isfinite(depth_confidence[index])])), 6),
            )
        )

    camera_statistics = tuple(
        (
            frame_id,
            round(float(extrinsics_np[index, 0, 3]), 6),
            round(float(extrinsics_np[index, 1, 3]), 6),
            round(float(extrinsics_np[index, 2, 3]), 6),
            round(float(intrinsics_np[index, 0, 0]), 6),
            round(float(intrinsics_np[index, 1, 1]), 6),
            round(float(intrinsics_np[index, 0, 2]), 6),
            round(float(intrinsics_np[index, 1, 2]), 6),
        )
        for index, frame_id in enumerate(prepared.frame_ids)
    )
    glb_path, exported_points = _export_glb(
        output_directory,
        world_points,
        point_confidence,
        rgb,
        extrinsics_np,
        confidence_percentile,
        max_points,
    )
    return OmniVggtInferenceResult(
        gallery=tuple(gallery),
        frame_statistics=tuple(frame_statistics),
        camera_statistics=camera_statistics,
        glb_path=glb_path,
        exported_points=exported_points,
        inference_seconds=float(inference_seconds),
    )


def load_official_omnivggt(
    config: OmniVggtRuntimeConfig,
) -> tuple[Any, Callable[[torch.Tensor, tuple[int, int]], tuple[torch.Tensor, torch.Tensor]]]:
    """Load a pinned external official checkout without copying its MIT code."""

    repository = config.official_repository.expanduser().resolve()
    checkpoint = config.checkpoint.expanduser().resolve()
    if not (repository / "omnivggt" / "models" / "omnivggt.py").is_file():
        raise OmniVggtInferenceError(f"official OmniVGGT repository is invalid: {repository}")
    if not checkpoint.is_file():
        raise OmniVggtInferenceError(f"official OmniVGGT checkpoint is missing: {checkpoint}")
    if config.expected_checkpoint_sha256 and _sha256(checkpoint) != config.expected_checkpoint_sha256.lower():
        raise OmniVggtInferenceError("official OmniVGGT checkpoint SHA-256 does not match")
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise OmniVggtInferenceError("CUDA was requested but is not available")

    with _repository_import_path(repository), _without_redundant_torch_hub_download():
        model_module = importlib.import_module("omnivggt.models.omnivggt")
        pose_module = importlib.import_module("omnivggt.utils.pose_enc")
        model = model_module.OmniVGGT()
    from safetensors.torch import load_file

    state_dict = load_file(str(checkpoint), device="cpu")
    model.load_state_dict(state_dict, strict=True)
    del state_dict
    model.to(device).eval()
    return model, pose_module.pose_encoding_to_extri_intri


def infer_and_render(
    model: Any,
    pose_decoder: Callable[[torch.Tensor, tuple[int, int]], tuple[torch.Tensor, torch.Tensor]],
    prepared: PreparedOmniVggtInput,
    *,
    device: torch.device,
    output_directory: str | Path,
    confidence_percentile: float,
    max_points: int,
) -> OmniVggtInferenceResult:
    """Time one model invocation and produce all viewer-neutral outputs."""

    started = time.perf_counter()
    predictions = run_omnivggt_model(model, prepared, device=device)
    elapsed = time.perf_counter() - started
    return render_omnivggt_predictions(
        prepared,
        predictions,
        pose_decoder=pose_decoder,
        output_directory=output_directory,
        confidence_percentile=confidence_percentile,
        max_points=max_points,
        inference_seconds=elapsed,
    )


def _scalar_frames(tensor: torch.Tensor, frames: int, height: int, width: int, name: str) -> np.ndarray:
    array = tensor.detach().float().cpu().numpy()
    if array.ndim == 5 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim == 3:
        array = array[None]
    if array.shape != (1, frames, height, width):
        raise OmniVggtInferenceError(f"{name} has unexpected shape {array.shape}")
    return array[0]


def _vector_frames(tensor: torch.Tensor, frames: int, height: int, width: int, name: str) -> np.ndarray:
    array = tensor.detach().float().cpu().numpy()
    if array.ndim == 4:
        array = array[None]
    if array.shape != (1, frames, height, width, 3):
        raise OmniVggtInferenceError(f"{name} has unexpected shape {array.shape}")
    return array[0]


def _camera_array(tensor: torch.Tensor, frames: int, tail: tuple[int, int], name: str) -> np.ndarray:
    array = tensor.detach().float().cpu().numpy()
    if array.ndim == 3:
        array = array[None]
    expected = (1, frames, *tail)
    if array.shape != expected:
        raise OmniVggtInferenceError(f"{name} has unexpected shape {array.shape}")
    return array[0]


def _colorize(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    finite = valid & np.isfinite(values)
    normalized = np.zeros(values.shape, dtype=np.uint8)
    if finite.any():
        low, high = np.percentile(values[finite], [2, 98])
        high = max(float(high), float(low) + 1e-8)
        normalized[finite] = np.clip((values[finite] - low) / (high - low) * 255, 0, 255).astype(np.uint8)
    colored = cv2.cvtColor(cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    colored[~finite] = 0
    return colored


def _overlay(rgb: np.ndarray, color: np.ndarray, alpha: float) -> np.ndarray:
    rgb_u8 = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    return cv2.addWeighted(rgb_u8, 1 - alpha, color, alpha, 0)


def _export_glb(
    output_directory: str | Path,
    world_points: np.ndarray,
    confidence: np.ndarray,
    rgb: np.ndarray,
    extrinsics: np.ndarray,
    confidence_percentile: float,
    max_points: int,
) -> tuple[Path, int]:
    import trimesh

    frame_count, height, width, _ = world_points.shape
    finite = np.isfinite(world_points).all(axis=-1) & np.isfinite(confidence)
    if not finite.any():
        raise OmniVggtInferenceError("OmniVGGT produced no finite 3D points")
    threshold = float(np.percentile(confidence[finite], confidence_percentile))
    stride = max(1, math.ceil(math.sqrt(frame_count * height * width / max_points)))
    scene = trimesh.Scene()
    exported_points = 0
    scene_points: list[np.ndarray] = []
    for frame_index in range(frame_count):
        sampled_points = world_points[frame_index, ::stride, ::stride]
        sampled_scores = confidence[frame_index, ::stride, ::stride]
        sampled_colors = np.clip(rgb[frame_index, ::stride, ::stride] * 255, 0, 255).astype(np.uint8)
        sampled_valid = np.isfinite(sampled_points).all(axis=-1) & np.isfinite(sampled_scores)
        confidence_valid = sampled_valid & (sampled_scores >= threshold)
        mesh = _grid_surface_mesh(sampled_points, sampled_colors, confidence_valid, trimesh)
        if mesh is None:
            mesh = _grid_surface_mesh(sampled_points, sampled_colors, sampled_valid, trimesh)
        if mesh is None:
            continue
        exported_points += len(mesh.vertices)
        scene_points.append(np.asarray(mesh.vertices))
        scene.add_geometry(mesh, node_name=f"omnivggt_surface_{frame_index:03d}")
    if not scene_points:
        raise OmniVggtInferenceError("OmniVGGT points could not form a browser-visible surface")
    all_scene_points = np.concatenate(scene_points)
    scene_scale = max(float(np.linalg.norm(np.ptp(all_scene_points, axis=0))), 1e-3)
    for index, world_to_camera in enumerate(extrinsics):
        transform = np.eye(4, dtype=np.float64)
        transform[:3] = world_to_camera
        camera_to_world = np.linalg.inv(transform)
        axis = trimesh.creation.axis(
            transform=camera_to_world,
            origin_size=scene_scale * 0.004,
            axis_length=scene_scale * 0.04,
        )
        scene.add_geometry(axis, node_name=f"camera_{index:03d}")
    directory = Path(output_directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    glb_path = directory / f"omnivggt_scene_{uuid.uuid4().hex[:12]}.glb"
    scene.export(glb_path)
    return glb_path, exported_points


def _grid_surface_mesh(points: np.ndarray, colors: np.ndarray, valid: np.ndarray, trimesh_module: Any):
    rows, columns = valid.shape
    vertex_ids = np.full((rows, columns), -1, dtype=np.int64)
    vertex_ids[valid] = np.arange(int(valid.sum()))
    upper_left = vertex_ids[:-1, :-1]
    upper_right = vertex_ids[:-1, 1:]
    lower_left = vertex_ids[1:, :-1]
    lower_right = vertex_ids[1:, 1:]
    first_valid = (upper_left >= 0) & (lower_left >= 0) & (upper_right >= 0)
    second_valid = (upper_right >= 0) & (lower_left >= 0) & (lower_right >= 0)
    first_faces = np.stack((upper_left[first_valid], lower_left[first_valid], upper_right[first_valid]), axis=-1)
    second_faces = np.stack((upper_right[second_valid], lower_left[second_valid], lower_right[second_valid]), axis=-1)
    faces = np.concatenate((first_faces, second_faces), axis=0)
    if not len(faces):
        return None
    return trimesh_module.Trimesh(
        vertices=points[valid],
        faces=faces,
        vertex_colors=colors[valid],
        process=False,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _repository_import_path(repository: Path):
    value = str(repository)
    sys.path.insert(0, value)
    try:
        yield
    finally:
        if sys.path and sys.path[0] == value:
            sys.path.pop(0)


@contextmanager
def _without_redundant_torch_hub_download():
    original_load = torch.hub.load

    class EmptyPretrained:
        @staticmethod
        def state_dict() -> dict[str, torch.Tensor]:
            return {}

    def empty_pretrained(*_args: Any, **_kwargs: Any) -> EmptyPretrained:
        return EmptyPretrained()

    torch.hub.load = cast(Any, empty_pretrained)
    try:
        yield
    finally:
        torch.hub.load = original_load
