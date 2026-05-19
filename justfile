set shell := ["bash", "-cu"]

# ---------------------------------------------------------------------------
# VGGT-Omega utility commands.
# Run `just` (no args) to list everything.
# ---------------------------------------------------------------------------

uv := "uv run --extra demo --extra viz"
ckpt_512 := env_var_or_default("VGGT_OMEGA_CKPT", "checkpoints/vggt_omega_1b_512.pt")
ckpt_256 := env_var_or_default("VGGT_OMEGA_CKPT_256", "checkpoints/vggt_omega_1b_256_text.pt")

default:
    @just --list

# Install / refresh the dev environment from pyproject.toml.
sync:
    uv sync --extra demo --extra viz

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

# Quick CLI smoke test: extract frames from an example video and run inference.
smoke ckpt=ckpt_512 video="examples/forest_road.mp4" frames="4":
    {{uv}} python -m vggt_omega.cli smoke --checkpoint {{ckpt}} --video {{video}} --num-frames {{frames}}

# ---------------------------------------------------------------------------
# Rerun visualization recipes.
# ---------------------------------------------------------------------------

# Write the reconstruction to a .rrd file (no display required).
viz-rrd video="examples/forest_road.mp4" output="outputs/scene.rrd" frames="6":
    {{uv}} python scripts/visualize.py --checkpoint {{ckpt_512}} --video {{video}} \
        --num-frames {{frames}} --image-resolution 512 \
        --mode rrd --output {{output}}

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
