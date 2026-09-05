# SPDX-License-Identifier: Apache-2.0
"""Read-only Gradio viewer for paired RGB, mapped depth, and valid masks."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import gradio as gr

from vggt_omega.rgbd_viewer import (
    RgbdDatasetIndex,
    RgbdLoaderConfig,
    RgbdViewerError,
    load_rgbd_frame,
    render_rgbd_gallery,
)

DEFAULT_DATASET_ROOT = "/workspace/data/vggt_omega/colmap_rgbd_640x480_v1"
MAX_SELECTED_FRAMES = 64
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

    uploaded_references = [_uploaded_path(item) for item in uploaded_rgb or ()]
    references: list[str | Path] = uploaded_references or list(selected_frame_ids or ())
    if not references:
        raise RgbdViewerError("select or upload at least one RGB image")
    if len(references) > MAX_SELECTED_FRAMES:
        raise RgbdViewerError(f"select at most {MAX_SELECTED_FRAMES} frames per render")
    index = RgbdDatasetIndex.discover(root, config)
    pairs = index.resolve_many(references)
    loaded = [load_rgbd_frame(pair, config) for pair in pairs]
    rendered = render_rgbd_gallery(
        loaded,
        depth_ceiling_m=None if auto_depth_ceiling else float(depth_ceiling_m),
        overlay_alpha=float(overlay_alpha),
    )
    source = "uploaded RGB" if uploaded_references else "dataset selection"
    status = (
        f"Rendered **{len(loaded)} frame(s)** from **{source}** / **{len(rendered.gallery)} views**. "
        f"Shared depth ceiling: **{rendered.depth_ceiling_m:.3f} m**. Sources were not modified."
    )
    return list(rendered.gallery), [list(row) for row in rendered.statistics], status


def build_ui(default_root: str | Path, config: RgbdLoaderConfig) -> gr.Blocks:
    """Build the checkpoint-free local RGB-D viewer."""

    try:
        initial_choices, initial_status = list_frame_choices(default_root, config)
        initial_selection: list[str] = []
    except RgbdViewerError as error:
        initial_choices, initial_selection = (), []
        initial_status = f"Dataset is not indexed yet: {error}"

    with gr.Blocks(title="VGGT-Ω RGB-D Input Viewer") as demo:
        gr.Markdown(
            "# VGGT-Ω / OmniVGGT RGB-D Input Viewer\n"
            "Select dataset RGB frames or upload their copies. Matching mapped depth and valid masks are resolved "
            "from the read-only dataset index; no checkpoint or CUDA is required."
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
            render_button = gr.Button("Visualize RGB-D", variant="primary")
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
        gr.ClearButton([selected_frames, uploaded_rgb, gallery, statistics], value="Clear selection/results")

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
    demo = build_ui(args.dataset_root, config)
    demo.queue(default_concurrency_limit=2).launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        css=VIEWER_CSS,
    )


if __name__ == "__main__":
    main()
