# SPDX-License-Identifier: Apache-2.0
"""Reusable RGB-D pairing and loading primitives for local visualization."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


class RgbdViewerError(ValueError):
    """Raised when an RGB-D layout or frame violates the explicit viewer contract."""


@dataclass(frozen=True)
class RgbdLoaderConfig:
    """Describe one sibling-directory RGB-D layout.

    Pair keys are filename stems after removing the configured suffix.  The
    default matches ``rgb/depth/valid_mask/frame_*.png`` staging directories;
    ``_rgb``/``_depth`` datasets are supported by changing the suffix fields.
    """

    rgb_directory: str = "rgb"
    depth_directory: str = "depth"
    mask_directory: str | None = "valid_mask"
    rgb_key_suffix: str = ""
    depth_key_suffix: str = ""
    mask_key_suffix: str = ""
    depth_scale_to_m: float = 0.001
    derive_mask_when_missing: bool = False
    require_mask_matches_depth: bool = True
    image_extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

    def __post_init__(self) -> None:
        for name, value in (
            ("rgb_directory", self.rgb_directory),
            ("depth_directory", self.depth_directory),
        ):
            if not value or Path(value).name != value:
                raise RgbdViewerError(f"{name} must be one non-empty directory name")
        if self.mask_directory is not None and (
            not self.mask_directory or Path(self.mask_directory).name != self.mask_directory
        ):
            raise RgbdViewerError("mask_directory must be one directory name or None")
        if self.mask_directory is None and not self.derive_mask_when_missing:
            raise RgbdViewerError("mask_directory=None requires derive_mask_when_missing=True")
        if not math.isfinite(self.depth_scale_to_m) or self.depth_scale_to_m <= 0:
            raise RgbdViewerError("depth_scale_to_m must be finite and positive")
        normalized_extensions = tuple(extension.lower() for extension in self.image_extensions)
        if not normalized_extensions or any(not extension.startswith(".") for extension in normalized_extensions):
            raise RgbdViewerError("image_extensions must contain dot-prefixed suffixes")
        object.__setattr__(self, "image_extensions", normalized_extensions)


@dataclass(frozen=True)
class RgbdFramePair:
    """Resolved read-only source paths for one RGB-D frame."""

    frame_id: str
    pair_key: str
    rgb_path: Path
    depth_path: Path
    mask_path: Path | None


@dataclass(frozen=True)
class LoadedRgbdFrame:
    """Decoded arrays in RGB uint8, metric-depth float32, and bool-mask form."""

    pair: RgbdFramePair
    rgb: np.ndarray
    depth_m: np.ndarray
    valid_mask: np.ndarray
    mask_source: str


@dataclass(frozen=True)
class RenderedRgbdGallery:
    """UI-neutral gallery images and per-frame metric-depth statistics."""

    gallery: tuple[tuple[Image.Image, str], ...]
    statistics: tuple[tuple[object, ...], ...]
    depth_ceiling_m: float


class RgbdDatasetIndex:
    """Lightweight immutable index that resolves selected or uploaded RGB files."""

    def __init__(self, root: Path, config: RgbdLoaderConfig, pairs: tuple[RgbdFramePair, ...]) -> None:
        self.root = root
        self.config = config
        self.pairs = pairs
        self._by_id = {pair.frame_id: pair for pair in pairs}
        self._by_resolved_path = {pair.rgb_path.resolve(): pair for pair in pairs}
        by_basename: defaultdict[str, list[RgbdFramePair]] = defaultdict(list)
        for pair in pairs:
            by_basename[pair.rgb_path.name].append(pair)
        self._by_basename = {name: tuple(values) for name, values in by_basename.items()}

    @property
    def frame_ids(self) -> tuple[str, ...]:
        """Return stable dataset-relative RGB identifiers suitable for a UI."""

        return tuple(pair.frame_id for pair in self.pairs)

    @classmethod
    def discover(
        cls,
        root: str | Path,
        config: RgbdLoaderConfig | None = None,
    ) -> RgbdDatasetIndex:
        """Discover sibling RGB/depth/mask directories recursively under ``root``."""

        resolved_root = Path(root).expanduser().resolve()
        if not resolved_root.is_dir():
            raise RgbdViewerError(f"dataset root is not a directory: {resolved_root}")
        selected_config = config or RgbdLoaderConfig()
        rgb_directories = _find_named_directories(resolved_root, selected_config.rgb_directory)
        if not rgb_directories:
            raise RgbdViewerError(
                f"no {selected_config.rgb_directory!r} directory exists under dataset root: {resolved_root}"
            )

        pairs: list[RgbdFramePair] = []
        for rgb_directory in rgb_directories:
            pairs.extend(_discover_directory_pairs(resolved_root, rgb_directory, selected_config))
        if not pairs:
            raise RgbdViewerError("RGB directories contain no supported image files")
        ordered = tuple(sorted(pairs, key=lambda pair: pair.frame_id))
        frame_ids = [pair.frame_id for pair in ordered]
        if len(frame_ids) != len(set(frame_ids)):
            raise RgbdViewerError("dataset produces duplicate relative RGB frame identifiers")
        return cls(resolved_root, selected_config, ordered)

    def resolve(self, reference: str | Path) -> RgbdFramePair:
        """Resolve a dataset-relative ID, source path, or Gradio-uploaded RGB file."""

        value = str(reference)
        if value in self._by_id:
            return self._by_id[value]
        candidate_path = Path(value).expanduser()
        if candidate_path.exists():
            resolved = candidate_path.resolve()
            direct = self._by_resolved_path.get(resolved)
            if direct is not None:
                return direct
        basename = candidate_path.name
        candidates = self._by_basename.get(basename, ())
        if not candidates:
            raise RgbdViewerError(f"RGB selection is not present in the dataset index: {reference}")
        if len(candidates) == 1:
            return candidates[0]
        if candidate_path.is_file():
            selected_digest = _file_digest(candidate_path)
            digest_matches = [pair for pair in candidates if _file_digest(pair.rgb_path) == selected_digest]
            if len(digest_matches) == 1:
                return digest_matches[0]
        locations = ", ".join(pair.frame_id for pair in candidates[:5])
        raise RgbdViewerError(f"RGB basename is ambiguous; select a relative path instead: {basename} ({locations})")

    def resolve_many(self, references: list[str | Path] | tuple[str | Path, ...]) -> tuple[RgbdFramePair, ...]:
        """Resolve references in order and reject duplicate selections."""

        resolved = tuple(self.resolve(reference) for reference in references)
        frame_ids = [pair.frame_id for pair in resolved]
        if len(frame_ids) != len(set(frame_ids)):
            raise RgbdViewerError("RGB selections resolve to duplicate dataset frames")
        return resolved


def load_rgbd_frame(pair: RgbdFramePair, config: RgbdLoaderConfig) -> LoadedRgbdFrame:
    """Decode one pair without modifying any source file."""

    try:
        with Image.open(pair.rgb_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        with Image.open(pair.depth_path) as image:
            depth_raw = np.asarray(image).copy()
    except OSError as error:
        raise RgbdViewerError(f"could not decode RGB-D pair {pair.frame_id}: {error}") from error
    if depth_raw.ndim != 2 or not np.issubdtype(depth_raw.dtype, np.integer):
        raise RgbdViewerError(f"depth must be a single-channel integer image: {pair.depth_path}")
    if rgb.shape[:2] != depth_raw.shape:
        raise RgbdViewerError(f"RGB and depth shape differ for {pair.frame_id}: {rgb.shape[:2]} != {depth_raw.shape}")
    if np.issubdtype(depth_raw.dtype, np.signedinteger) and np.any(depth_raw < 0):
        raise RgbdViewerError(f"depth contains negative integer values: {pair.depth_path}")

    mask_source: str
    if pair.mask_path is None:
        if not config.derive_mask_when_missing:
            raise RgbdViewerError(f"valid mask is missing for {pair.frame_id}")
        valid_mask = depth_raw > 0
        mask_source = "derived_from_depth"
    else:
        try:
            with Image.open(pair.mask_path) as image:
                mask_raw = np.asarray(image).copy()
        except OSError as error:
            raise RgbdViewerError(f"could not decode valid mask for {pair.frame_id}: {error}") from error
        if mask_raw.ndim != 2 or mask_raw.shape != depth_raw.shape:
            raise RgbdViewerError(
                f"valid mask shape differs for {pair.frame_id}: {mask_raw.shape} != {depth_raw.shape}"
            )
        valid_mask = mask_raw != 0
        mask_source = pair.mask_path.name
        if config.require_mask_matches_depth and not np.array_equal(valid_mask, depth_raw > 0):
            raise RgbdViewerError(f"stored valid mask does not match depth>0 for {pair.frame_id}")

    depth_m = depth_raw.astype(np.float32) * np.float32(config.depth_scale_to_m)
    if not np.isfinite(depth_m[valid_mask]).all() or np.any(depth_m[valid_mask] <= 0):
        raise RgbdViewerError(f"valid metric depth is non-finite or non-positive for {pair.frame_id}")
    return LoadedRgbdFrame(
        pair=pair,
        rgb=rgb,
        depth_m=depth_m,
        valid_mask=valid_mask.astype(np.bool_, copy=False),
        mask_source=mask_source,
    )


def render_rgbd_gallery(
    frames: Sequence[LoadedRgbdFrame],
    *,
    depth_ceiling_m: float | None,
    overlay_alpha: float,
) -> RenderedRgbdGallery:
    """Render RGB, metric depth, valid mask, and overlay for selected frames."""

    if not frames:
        raise RgbdViewerError("at least one RGB-D frame must be selected")
    if not math.isfinite(overlay_alpha) or not 0 <= overlay_alpha <= 1:
        raise RgbdViewerError("overlay_alpha must be between 0 and 1")
    if depth_ceiling_m is None:
        valid_depth = [frame.depth_m[frame.valid_mask] for frame in frames if frame.valid_mask.any()]
        if not valid_depth:
            raise RgbdViewerError("automatic depth ceiling requires at least one valid depth pixel")
        ceiling = float(np.quantile(np.concatenate(valid_depth), 0.98))
    else:
        ceiling = float(depth_ceiling_m)
    if not math.isfinite(ceiling) or ceiling <= 0:
        raise RgbdViewerError("depth_ceiling_m must be finite and positive")

    gallery: list[tuple[Image.Image, str]] = []
    statistics: list[tuple[object, ...]] = []
    for frame in frames:
        depth_color = _colorize_depth(frame.depth_m, frame.valid_mask, ceiling)
        mask_rgb = np.repeat((frame.valid_mask.astype(np.uint8) * 255)[..., None], 3, axis=2)
        overlay = frame.rgb.copy()
        blended = np.rint(
            frame.rgb.astype(np.float32) * (1 - overlay_alpha) + depth_color.astype(np.float32) * overlay_alpha
        ).astype(np.uint8)
        overlay[frame.valid_mask] = blended[frame.valid_mask]
        prefix = frame.pair.frame_id
        gallery.extend(
            (
                (Image.fromarray(frame.rgb), f"{prefix} / RGB"),
                (Image.fromarray(depth_color), f"{prefix} / Depth (0-{ceiling:.3f} m)"),
                (Image.fromarray(mask_rgb), f"{prefix} / Valid mask"),
                (Image.fromarray(overlay), f"{prefix} / RGB + depth"),
            )
        )
        values = frame.depth_m[frame.valid_mask]
        if values.size:
            minimum, median, percentile95, maximum = (
                float(np.min(values)),
                float(np.median(values)),
                float(np.quantile(values, 0.95)),
                float(np.max(values)),
            )
        else:
            minimum = median = percentile95 = maximum = None
        statistics.append(
            (
                prefix,
                frame.mask_source,
                int(values.size),
                100.0 * float(values.size) / float(frame.valid_mask.size),
                minimum,
                median,
                percentile95,
                maximum,
            )
        )
    return RenderedRgbdGallery(tuple(gallery), tuple(statistics), ceiling)


def _colorize_depth(depth_m: np.ndarray, valid_mask: np.ndarray, ceiling_m: float) -> np.ndarray:
    normalized = np.clip(depth_m / np.float32(ceiling_m), 0, 1)
    quantized = np.rint(normalized * 255).astype(np.uint8)
    bgr = cv2.applyColorMap(quantized, cv2.COLORMAP_TURBO)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb[~valid_mask] = 0
    return rgb


def _find_named_directories(root: Path, name: str) -> tuple[Path, ...]:
    directories = [root] if root.name == name else []
    directories.extend(path for path in root.rglob(name) if path.is_dir())
    return tuple(sorted(set(directories)))


def _discover_directory_pairs(root: Path, rgb_directory: Path, config: RgbdLoaderConfig) -> list[RgbdFramePair]:
    container = rgb_directory.parent
    depth_directory = container / config.depth_directory
    if not depth_directory.is_dir():
        raise RgbdViewerError(f"missing depth directory beside {rgb_directory}: {depth_directory}")
    rgb_by_key = _files_by_key(rgb_directory, config.rgb_key_suffix, config, "RGB")
    depth_by_key = _files_by_key(depth_directory, config.depth_key_suffix, config, "depth")
    missing_depth = sorted(set(rgb_by_key) - set(depth_by_key))
    extra_depth = sorted(set(depth_by_key) - set(rgb_by_key))
    if missing_depth:
        raise RgbdViewerError(f"missing depth for RGB pair keys in {container}: {missing_depth[:5]}")
    if extra_depth:
        raise RgbdViewerError(f"depth files have no RGB pair in {container}: {extra_depth[:5]}")

    mask_by_key: dict[str, Path] | None = None
    if config.mask_directory is not None:
        mask_directory = container / config.mask_directory
        if mask_directory.is_dir():
            mask_by_key = _files_by_key(mask_directory, config.mask_key_suffix, config, "valid mask")
            missing_masks = sorted(set(rgb_by_key) - set(mask_by_key))
            extra_masks = sorted(set(mask_by_key) - set(rgb_by_key))
            if missing_masks:
                raise RgbdViewerError(f"missing valid mask for RGB pair keys in {container}: {missing_masks[:5]}")
            if extra_masks:
                raise RgbdViewerError(f"valid masks have no RGB pair in {container}: {extra_masks[:5]}")
        elif not config.derive_mask_when_missing:
            raise RgbdViewerError(f"missing valid-mask directory beside {rgb_directory}: {mask_directory}")

    pairs: list[RgbdFramePair] = []
    for key, rgb_path in rgb_by_key.items():
        pairs.append(
            RgbdFramePair(
                frame_id=rgb_path.relative_to(root).as_posix(),
                pair_key=key,
                rgb_path=rgb_path,
                depth_path=depth_by_key[key],
                mask_path=None if mask_by_key is None else mask_by_key[key],
            )
        )
    return pairs


def _files_by_key(
    directory: Path,
    key_suffix: str,
    config: RgbdLoaderConfig,
    kind: str,
) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in config.image_extensions:
            continue
        key = _pair_key(path, key_suffix)
        if key in result:
            raise RgbdViewerError(f"duplicate pair key {key!r} in {directory}")
        result[key] = path
    if not result:
        raise RgbdViewerError(f"missing {kind} image files in {directory}")
    return result


def _pair_key(path: Path, suffix: str) -> str:
    if suffix and not path.stem.endswith(suffix):
        raise RgbdViewerError(f"filename does not end with configured key suffix {suffix!r}: {path.name}")
    key = path.stem[: -len(suffix)] if suffix else path.stem
    if not key:
        raise RgbdViewerError(f"filename has an empty pair key: {path.name}")
    return key


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
