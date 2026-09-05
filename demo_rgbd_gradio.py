# SPDX-License-Identifier: Apache-2.0
"""Read-only Gradio viewer for paired RGB, mapped depth, and valid masks."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from tempfile import gettempdir
from typing import Any

import gradio as gr
import torch

from vggt_omega.omnivggt_inference import (
    OmniVggtInferenceError,
    OmniVggtRuntimeConfig,
    infer_and_render,
    load_official_omnivggt,
    prepare_omnivggt_input,
)
from vggt_omega.rgbd_viewer import (
    RgbdDatasetIndex,
    RgbdLoaderConfig,
    RgbdViewerError,
    load_rgbd_frame,
    render_rgbd_gallery,
)

DEFAULT_DATASET_ROOT = "/workspace/data/vggt_omega/colmap_rgbd_640x480_v1"
DEFAULT_OMNIVGGT_REPOSITORY = "/workspace/external/OmniVGGT-official"
DEFAULT_OMNIVGGT_CHECKPOINT = "/workspace/models/OmniVGGT/OmniVGGT.safetensors"
MAX_SELECTED_FRAMES = 64
MAX_OMNIVGGT_FRAMES = 8
STAT_HEADERS = [
    "frame_id",
    "mask_source",
    "valid_pixels",
    "valid_percent",
    "min_depth_m",
    "median_depth_m",
    "p95_depth_m",
    "max_depth_m",
]

VIEWER_CSS = """
.rgbd-status { padding: 0.6rem 0.8rem; border-left: 4px solid #0ea5e9; }
.rgbd-gallery { background: linear-gradient(145deg, rgba(14,165,233,.05), rgba(34,197,94,.04)); }
"""

OMNI_FRAME_HEADERS = [
    "frame_id",
    "compared_valid_percent",
    "mapped_depth_median_m",
    "predicted_depth_median",
    "alignment_scale",
    "aligned_rmse_m",
    "predicted_confidence_median",
]
OMNI_CAMERA_HEADERS = ["frame_id", "tx", "ty", "tz", "fx", "fy", "cx", "cy"]


def loader_config(
    rgb_directory: str,
    depth_directory: str,
    mask_directory: str,
    rgb_key_suffix: str,
    depth_key_suffix: str,
    mask_key_suffix: str,
    depth_scale_to_m: float,
    derive_mask_when_missing: bool,
    require_mask_matches_depth: bool,
) -> RgbdLoaderConfig:
    """Build the reusable loader config from CLI or UI scalar values."""

    normalized_mask_directory = mask_directory.strip() or None
    return RgbdLoaderConfig(
        rgb_directory=rgb_directory.strip(),
        depth_directory=depth_directory.strip(),
        mask_directory=normalized_mask_directory,
        rgb_key_suffix=rgb_key_suffix,
        depth_key_suffix=depth_key_suffix,
        mask_key_suffix=mask_key_suffix,
        depth_scale_to_m=float(depth_scale_to_m),
        derive_mask_when_missing=bool(derive_mask_when_missing),
        require_mask_matches_depth=bool(require_mask_matches_depth),
    )


def list_frame_choices(root: str | Path, config: RgbdLoaderConfig) -> tuple[tuple[str, ...], str]:
    """Return stable dataset-relative RGB choices without decoding image pixels."""

    index = RgbdDatasetIndex.discover(root, config)
    scene_count = len({Path(frame_id).parent.parent.as_posix() for frame_id in index.frame_ids})
    return index.frame_ids, f"Indexed **{len(index.pairs):,} RGB-D pairs** across **{scene_count} scene(s)**."


def visualize_selection(
    root: str | Path,
    selected_frame_ids: Sequence[str] | None,
    uploaded_rgb: Sequence[Any] | None,
    config: RgbdLoaderConfig,
    *,
    depth_ceiling_m: float,
    auto_depth_ceiling: bool,
    overlay_alpha: float,
) -> tuple[list[tuple[Any, str]], list[list[object]], str]:
    """Resolve selected RGB references and return Gradio-ready images and rows."""

    loaded, source = _resolve_loaded_frames(
        root,
        selected_frame_ids,
        uploaded_rgb,
        config,
        max_frames=MAX_SELECTED_FRAMES,
    )
    rendered = render_rgbd_gallery(
        loaded,
        depth_ceiling_m=None if auto_depth_ceiling else float(depth_ceiling_m),
        overlay_alpha=float(overlay_alpha),
    )
    status = (
        f"Rendered **{len(loaded)} frame(s)** from **{source}** / **{len(rendered.gallery)} views**. "
        f"Shared depth ceiling: **{rendered.depth_ceiling_m:.3f} m**. Sources were not modified."
    )
    return list(rendered.gallery), [list(row) for row in rendered.statistics], status


def run_omnivggt_selection(
    root: str | Path,
    selected_frame_ids: Sequence[str] | None,
    uploaded_rgb: Sequence[Any] | None,
    config: RgbdLoaderConfig,
    *,
    official_repository: str,
    checkpoint: str,
    device: str,
    target_size: int,
    confidence_percentile: float,
    max_points: int,
) -> tuple[list[tuple[Any, str]], list[list[object]], list[list[object]], str, str, str]:
    """Resolve RGB-D input, run official OmniVGGT, and return Gradio-ready outputs."""

    loaded, source = _resolve_loaded_frames(
        root,
        selected_frame_ids,
        uploaded_rgb,
        config,
        max_frames=MAX_OMNIVGGT_FRAMES,
    )
    prepared = prepare_omnivggt_input(loaded, target_size=int(target_size))
    model, pose_decoder = _get_omnivggt_runtime(official_repository, checkpoint, device)
    result = infer_and_render(
        model,
        pose_decoder,
        prepared,
        device=torch.device(device),
        output_directory=Path(gettempdir()) / "vggt_omega_omnivggt_outputs",
        confidence_percentile=float(confidence_percentile),
        max_points=int(max_points),
    )
    status = (
        f"OmniVGGT rendered **{len(loaded)} frame(s)** from **{source}** in "
        f"**{result.inference_seconds:.3f} s** on **{device}**; exported "
        f"**{result.exported_points:,} points**. Mapped depth was supplied to every selected frame."
    )
    glb_path = str(result.glb_path)
    return (
        list(result.gallery),
        [list(row) for row in result.frame_statistics],
        [list(row) for row in result.camera_statistics],
        glb_path,
        glb_path,
        status,
    )


def _resolve_loaded_frames(
    root: str | Path,
    selected_frame_ids: Sequence[str] | None,
    uploaded_rgb: Sequence[Any] | None,
    config: RgbdLoaderConfig,
    *,
    max_frames: int,
):
    uploaded_references = [_uploaded_path(item) for item in uploaded_rgb or ()]
    references: list[str | Path] = [
        str(reference) for reference in (uploaded_references or list(selected_frame_ids or ()))
    ]
    if not references:
        raise RgbdViewerError("select or upload at least one RGB image")
    if len(references) > max_frames:
        raise RgbdViewerError(f"select at most {max_frames} frames for this operation")
    index = RgbdDatasetIndex.discover(root, config)
    pairs = index.resolve_many(references)
    loaded = [load_rgbd_frame(pair, config) for pair in pairs]
    source = "uploaded RGB" if uploaded_references else "dataset selection"
    return loaded, source


@lru_cache(maxsize=2)
def _get_omnivggt_runtime(official_repository: str, checkpoint: str, device: str):
    return load_official_omnivggt(
        OmniVggtRuntimeConfig(
            official_repository=Path(official_repository),
            checkpoint=Path(checkpoint),
            device=device,
        )
    )


def build_ui(
    default_root: str | Path,
    config: RgbdLoaderConfig,
    *,
    official_repository: str = DEFAULT_OMNIVGGT_REPOSITORY,
    checkpoint: str = DEFAULT_OMNIVGGT_CHECKPOINT,
    device: str = "cuda",
) -> gr.Blocks:
    """Build the paired-input inspector and official OmniVGGT inference viewer."""

    try:
        initial_choices, initial_status = list_frame_choices(default_root, config)
        initial_selection: list[str] = []
    except RgbdViewerError as error:
        initial_choices, initial_selection = (), []
        initial_status = f"Dataset is not indexed yet: {error}"

    with gr.Blocks(title="OmniVGGT RGB-D Inference Viewer") as demo:
        gr.Markdown(
            "# OmniVGGT RGB-D Inference Viewer\n"
            "Select dataset RGB frames or upload their copies. Matching mapped depth and valid masks are resolved "
            "from the read-only dataset index. Inspect the inputs first, then run the official OmniVGGT model."
        )
        with gr.Row():
            dataset_root = gr.Textbox(label="Dataset root", value=str(default_root), scale=5)
            reload_button = gr.Button("Reload index", variant="secondary", scale=1)
        status = gr.Markdown(initial_status, elem_classes=["rgbd-status"])
        with gr.Accordion("Layout settings", open=False):
            with gr.Row():
                rgb_directory = gr.Textbox(label="RGB directory", value=config.rgb_directory)
                depth_directory = gr.Textbox(label="Depth directory", value=config.depth_directory)
                mask_directory = gr.Textbox(
                    label="Mask directory",
                    value=config.mask_directory or "",
                    info="Leave empty only when derive-mask is enabled.",
                )
            with gr.Row():
                rgb_key_suffix = gr.Textbox(label="RGB key suffix", value=config.rgb_key_suffix)
                depth_key_suffix = gr.Textbox(label="Depth key suffix", value=config.depth_key_suffix)
                mask_key_suffix = gr.Textbox(label="Mask key suffix", value=config.mask_key_suffix)
            with gr.Row():
                depth_scale = gr.Number(
                    label="Stored depth → meters scale",
                    value=config.depth_scale_to_m,
                    precision=6,
                )
                derive_mask = gr.Checkbox(
                    label="Derive mask from depth>0 when mask directory is absent",
                    value=config.derive_mask_when_missing,
                )
                strict_mask = gr.Checkbox(
                    label="Require stored mask == depth>0",
                    value=config.require_mask_matches_depth,
                )
        with gr.Row():
            selected_frames = gr.Dropdown(
                choices=initial_choices,
                value=initial_selection,
                multiselect=True,
                max_choices=MAX_SELECTED_FRAMES,
                label="RGB frames",
                info=f"Dataset-relative RGB paths; choose up to {MAX_SELECTED_FRAMES}.",
                scale=3,
            )
            uploaded_rgb = gr.File(
                file_count="multiple",
                file_types=["image"],
                type="filepath",
                label="RGB uploads",
                scale=2,
            )
        gr.Markdown("Uploaded copies are matched by basename, then by file digest only when the basename is ambiguous.")
        with gr.Row():
            auto_ceiling = gr.Checkbox(label="Auto depth ceiling (shared p98)", value=False)
            depth_ceiling = gr.Number(label="Depth ceiling (m)", value=1.3, precision=3)
            overlay_alpha = gr.Slider(
                minimum=0,
                maximum=1,
                value=0.45,
                step=0.05,
                label="Depth overlay opacity",
            )
            render_button = gr.Button("Inspect paired RGB-D", variant="secondary")
        gallery = gr.Gallery(
            label="RGB-D views",
            columns=4,
            object_fit="contain",
            type="pil",
            height="auto",
            elem_classes=["rgbd-gallery"],
            buttons=["download", "fullscreen"],
        )
        statistics = gr.Dataframe(
            headers=STAT_HEADERS,
            datatype=["str", "str", "number", "number", "number", "number", "number", "number"],
            type="array",
            label="Frame statistics",
            interactive=False,
            wrap=True,
            show_search="search",
        )

        gr.Markdown("## OmniVGGT model results")
        with gr.Accordion("Official OmniVGGT runtime", open=False):
            official_repository_input = gr.Textbox(
                label="Official OmniVGGT repository",
                value=official_repository,
            )
            checkpoint_input = gr.Textbox(label="OmniVGGT checkpoint", value=checkpoint)
            with gr.Row():
                device_input = gr.Dropdown(label="Inference device", choices=["cuda", "cpu"], value=device)
                target_size_input = gr.Number(label="Model target width", value=518, precision=0)
                confidence_percentile_input = gr.Slider(
                    minimum=0,
                    maximum=99,
                    value=25,
                    step=1,
                    label="Discard lowest confidence percentile",
                )
                max_points_input = gr.Number(label="Maximum exported 3D points", value=200000, precision=0)
        run_omnivggt_button = gr.Button("Run OmniVGGT inference", variant="primary")
        omnivggt_status = gr.Markdown(
            "Select or upload 1-8 RGB frames, then run inference. First use loads the official checkpoint."
        )
        omnivggt_gallery = gr.Gallery(
            label="OmniVGGT predictions",
            columns=4,
            object_fit="contain",
            type="pil",
            height="auto",
            elem_classes=["rgbd-gallery"],
            buttons=["download", "fullscreen"],
        )
        omnivggt_frame_statistics = gr.Dataframe(
            headers=OMNI_FRAME_HEADERS,
            datatype=["str", "number", "number", "number", "number", "number", "number"],
            type="array",
            label="OmniVGGT frame comparison",
            interactive=False,
            wrap=True,
        )
        omnivggt_camera_statistics = gr.Dataframe(
            headers=OMNI_CAMERA_HEADERS,
            datatype=["str", "number", "number", "number", "number", "number", "number", "number"],
            type="array",
            label="Predicted cameras",
            interactive=False,
            wrap=True,
        )
        omnivggt_model = gr.Model3D(
            label="OmniVGGT 3D reconstruction",
            display_mode="point_cloud",
            height=560,
            clear_color=(0.02, 0.025, 0.04, 1.0),
        )
        omnivggt_download = gr.File(label="Download OmniVGGT GLB", interactive=False)
        gr.ClearButton(
            [
                selected_frames,
                uploaded_rgb,
                gallery,
                statistics,
                omnivggt_gallery,
                omnivggt_frame_statistics,
                omnivggt_camera_statistics,
                omnivggt_model,
                omnivggt_download,
            ],
            value="Clear selection/results",
        )

        layout_inputs = [
            rgb_directory,
            depth_directory,
            mask_directory,
            rgb_key_suffix,
            depth_key_suffix,
            mask_key_suffix,
            depth_scale,
            derive_mask,
            strict_mask,
        ]

        def reload_index(root: str, *layout_values: Any):
            try:
                current_config = loader_config(*layout_values)
                choices, message = list_frame_choices(root, current_config)
            except RgbdViewerError as error:
                raise gr.Error(str(error)) from error
            return gr.update(choices=choices, value=[]), message

        def render(
            root: str,
            frame_ids: Sequence[str] | None,
            uploads: Sequence[Any] | None,
            auto: bool,
            ceiling: float,
            alpha: float,
            *layout_values: Any,
        ):
            try:
                current_config = loader_config(*layout_values)
                return visualize_selection(
                    root,
                    frame_ids,
                    uploads,
                    current_config,
                    depth_ceiling_m=ceiling,
                    auto_depth_ceiling=auto,
                    overlay_alpha=alpha,
                )
            except RgbdViewerError as error:
                raise gr.Error(str(error)) from error

        reload_button.click(
            reload_index,
            inputs=[dataset_root, *layout_inputs],
            outputs=[selected_frames, status],
            api_name="reload_rgbd_index",
        )
        render_inputs = [
            dataset_root,
            selected_frames,
            uploaded_rgb,
            auto_ceiling,
            depth_ceiling,
            overlay_alpha,
            *layout_inputs,
        ]
        render_button.click(
            render,
            inputs=render_inputs,
            outputs=[gallery, statistics, status],
            api_name="visualize_rgbd",
        )
        uploaded_rgb.change(
            render,
            inputs=render_inputs,
            outputs=[gallery, statistics, status],
            api_name="visualize_uploaded_rgb",
        )

        def run_omnivggt(
            root: str,
            frame_ids: Sequence[str] | None,
            uploads: Sequence[Any] | None,
            repository_value: str,
            checkpoint_value: str,
            device_value: str,
            target_size_value: int,
            confidence_percentile_value: float,
            max_points_value: int,
            *layout_values: Any,
        ):
            try:
                current_config = loader_config(*layout_values)
                return run_omnivggt_selection(
                    root,
                    frame_ids,
                    uploads,
                    current_config,
                    official_repository=repository_value,
                    checkpoint=checkpoint_value,
                    device=device_value,
                    target_size=int(target_size_value),
                    confidence_percentile=float(confidence_percentile_value),
                    max_points=int(max_points_value),
                )
            except (RgbdViewerError, OmniVggtInferenceError, RuntimeError, OSError) as error:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise gr.Error(str(error)) from error

        run_omnivggt_button.click(
            run_omnivggt,
            inputs=[
                dataset_root,
                selected_frames,
                uploaded_rgb,
                official_repository_input,
                checkpoint_input,
                device_input,
                target_size_input,
                confidence_percentile_input,
                max_points_input,
                *layout_inputs,
            ],
            outputs=[
                omnivggt_gallery,
                omnivggt_frame_statistics,
                omnivggt_camera_statistics,
                omnivggt_model,
                omnivggt_download,
                omnivggt_status,
            ],
            api_name="run_omnivggt",
        )
    return demo


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View paired RGB, mapped depth, and valid masks in Gradio")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--rgb-directory", default="rgb")
    parser.add_argument("--depth-directory", default="depth")
    parser.add_argument("--mask-directory", default="valid_mask", help="Use an empty string to derive masks")
    parser.add_argument("--rgb-key-suffix", default="")
    parser.add_argument("--depth-key-suffix", default="")
    parser.add_argument("--mask-key-suffix", default="")
    parser.add_argument("--depth-scale-to-m", type=float, default=0.001)
    parser.add_argument("--derive-mask-when-missing", action="store_true")
    parser.add_argument("--allow-mask-depth-mismatch", action="store_true")
    parser.add_argument("--omnivggt-repository", default=DEFAULT_OMNIVGGT_REPOSITORY)
    parser.add_argument("--omnivggt-checkpoint", default=DEFAULT_OMNIVGGT_CHECKPOINT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> RgbdLoaderConfig:
    return loader_config(
        args.rgb_directory,
        args.depth_directory,
        args.mask_directory,
        args.rgb_key_suffix,
        args.depth_key_suffix,
        args.mask_key_suffix,
        args.depth_scale_to_m,
        args.derive_mask_when_missing,
        not args.allow_mask_depth_mismatch,
    )


def _uploaded_path(item: Any) -> Path:
    if isinstance(item, (str, Path)):
        return Path(item)
    if isinstance(item, dict):
        value = item.get("path", item.get("name"))
        if value is not None:
            return Path(value)
    value = getattr(item, "name", None)
    if value is not None:
        return Path(value)
    raise RgbdViewerError(f"unsupported Gradio upload value: {type(item).__name__}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = config_from_args(args)
    demo = build_ui(
        args.dataset_root,
        config,
        official_repository=args.omnivggt_repository,
        checkpoint=args.omnivggt_checkpoint,
        device=args.device,
    )
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        css=VIEWER_CSS,
    )


if __name__ == "__main__":
    main()
