set shell := ["bash", "-cu"]

# ---------------------------------------------------------------------------
# VGGT-Omega utility commands.
# Run `just` (no args) to list everything.
# ---------------------------------------------------------------------------

uv := "uv run --extra demo --extra viz"
uv_rgbd := "uv run --extra rgbd"
uv_training := "uv run --extra training"
ckpt_512 := env_var_or_default("VGGT_OMEGA_CKPT", "checkpoints/vggt_omega_1b_512.pt")
ckpt_256 := env_var_or_default("VGGT_OMEGA_CKPT_256", "checkpoints/vggt_omega_1b_256_text.pt")

default:
    @just --list

# Install / refresh the dev environment from pyproject.toml.
sync:
    uv sync --extra demo --extra viz

# Install the supervised-training dependencies without importing AMUSE's
# upstream environment pins.
sync-training:
    uv sync --extra demo --extra viz --extra rgbd --extra training

# Run the focused CPU training-pipeline tests.
test-training *ARGS:
    {{uv_training}} pytest tests/test_training_*.py -m "not gpu" {{ARGS}}

# Export one private RGB-D source into the anonymous staging contract.
prepare-training *ARGS:
    {{uv_training}} python scripts/prepare_colmap_rgbd_training.py {{ARGS}}

# Strict-load the pretrained model and execute the bounded real-data smoke run.
train-smoke *ARGS:
    {{uv_training}} python scripts/train_colmap_rgbd.py trainer=smoke optimizer=amuse {{ARGS}}

# Run the configured 50-epoch domain fine-tune after the smoke gate passes.
train *ARGS:
    {{uv_training}} python scripts/train_colmap_rgbd.py trainer=finetune optimizer=amuse {{ARGS}}

# Inspect scalar-only training progress.
tensorboard logdir="outputs/training" host="127.0.0.1" port="6006":
    {{uv_training}} tensorboard --logdir "{{logdir}}" --host "{{host}}" --port "{{port}}"

# Format Python files in place.
format:
    {{uv}} ruff format .

# Check formatting without writing.
format-check:
    {{uv}} ruff format --check .

# Lint with ruff (auto-fix safe issues).
lint:
    {{uv}} ruff check --fix .

# Lint without fixes.
lint-check:
    {{uv}} ruff check .

# Static type check.
typecheck:
    {{uv}} ty check vggt_omega tests

# Cyclomatic complexity & maintainability index.
complexity:
    {{uv}} radon cc vggt_omega -a -s
    {{uv}} radon mi vggt_omega -s

# Run the regression test suite (CPU only by default).
test *ARGS:
    {{uv}} pytest -m "not gpu" {{ARGS}}

# Run GPU-marked tests (requires CUDA + a checkpoint).
test-gpu *ARGS:
    {{uv}} pytest -m gpu {{ARGS}}

# Aggregate quality gate.
check: format-check lint-check typecheck test

# Launch the Gradio demo with the 512-px checkpoint.
demo *ARGS:
    {{uv}} python demo_gradio.py --checkpoint {{ckpt_512}} --image-resolution 512 {{ARGS}}

# Launch the Gradio demo with the text-aligned 256-px checkpoint.
demo-text *ARGS:
    {{uv}} python demo_gradio.py --checkpoint {{ckpt_256}} --image-resolution 256 --enable-alignment {{ARGS}}

# Inspect paired RGB-D inputs and run the official OmniVGGT checkpoint.
omnivggt-viewer root="/workspace/data/vggt_omega/colmap_rgbd_640x480_v1" *ARGS:
    {{uv}} python demo_rgbd_gradio.py --dataset-root "{{root}}" {{ARGS}}

# Backward-compatible entry point for the combined input/inference viewer.
rgbd-viewer root="/workspace/data/vggt_omega/colmap_rgbd_640x480_v1" *ARGS:
    {{uv}} python demo_rgbd_gradio.py --dataset-root "{{root}}" {{ARGS}}

# Quick CLI smoke test: extract frames from an example video and run inference.
smoke ckpt=ckpt_512 video="examples/forest_road.mp4" frames="4":
    {{uv}} python -m vggt_omega.cli smoke --checkpoint {{ckpt}} --video {{video}} --num-frames {{frames}}

# Run one RGB-D chunk with metric scale correction and pose plots.
rgbd-pose session output ckpt=ckpt_512 *ARGS:
    {{uv_rgbd}} python run_vggt_rgbd_pose_workflow.py --session-dir {{session}} --checkpoint {{ckpt}} --output-dir {{output}} {{ARGS}}

# Align overlapping RGB-D pose chunks and fuse their masked point clouds.
rgbd-align session output ckpt=ckpt_512 *ARGS:
    {{uv_rgbd}} python run_vggt_rgbd_chunk_alignment.py --session-dir {{session}} --checkpoint {{ckpt}} --output-dir {{output}} {{ARGS}}

# ---------------------------------------------------------------------------
# Rerun visualization recipes.
# ---------------------------------------------------------------------------

# Write the reconstruction to a .rrd file (no display required).
viz-rrd video="examples/forest_road.mp4" output="outputs/scene.rrd" frames="6":
    {{uv}} python scripts/visualize.py --checkpoint {{ckpt_512}} --video {{video}} \
        --num-frames {{frames}} --image-resolution 512 \
        --mode rrd --output {{output}}

# Write a smoother playback .rrd: sample by FPS and show only the current frame's point cloud.
viz-rrd-fast video="examples/forest_road.mp4" output="outputs/scene_fast.rrd" fps="8.0" max_points="15000":
    {{uv}} python scripts/visualize.py --checkpoint {{ckpt_512}} --video "{{video}}" \
        --sample-fps {{fps}} --image-resolution 512 \
        --mode rrd --output "{{output}}" --max-points {{max_points}} --no-accumulate-points

# Launch the local Rerun viewer (requires a display).
viz-viewer video="examples/forest_road.mp4" frames="6":
    {{uv}} python scripts/visualize.py --checkpoint {{ckpt_512}} --video {{video}} \
        --num-frames {{frames}} --image-resolution 512 \
        --mode viewer

# Screenshot the web viewer via Playwright (headless ok). Re-uses an existing .rrd when present.
viz-screenshot video="examples/forest_road.mp4" rrd="outputs/scene.rrd" png="outputs/scene.png" frames="6":
    if [ -f "{{rrd}}" ]; then \
        {{uv}} python scripts/visualize.py --mode screenshot --rrd-input "{{rrd}}" --output "{{png}}"; \
    else \
        {{uv}} python scripts/visualize.py --checkpoint {{ckpt_512}} --video "{{video}}" \
            --num-frames {{frames}} --image-resolution 512 \
            --mode screenshot --rrd-output "{{rrd}}" --output "{{png}}"; \
    fi

# Install Playwright browsers (needed once for `just viz-screenshot`).
viz-browsers:
    {{uv}} playwright install chromium

# Remove caches & build artefacts.
clean:
    rm -rf .pytest_cache .ruff_cache .ty_cache build dist *.egg-info
    find . -type d -name __pycache__ -prune -exec rm -rf {} +
