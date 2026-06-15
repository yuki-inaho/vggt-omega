# VGGT-Omega — Verified setup on RTX PRO 4000 Blackwell (cu130)

Machine-specific, reproducibility-verified setup record for this box. Branch
name encodes the confirmed CUDA build: **`blackwell-cu130`** (Blackwell GPU +
torch CUDA 13.0 build).

## Confirmed hardware / driver
| Item | Confirmed value |
|---|---|
| GPU | NVIDIA RTX PRO 4000 Blackwell, 24467 MiB (~24 GB) |
| Compute capability | **sm_120** (`torch.cuda.get_device_capability` → `(12, 0)`) |
| Driver | 580.159.04 |
| Driver CUDA version | **13.0** |

## Environment (uv)
- **Python 3.11.15** — NOT 3.10. The locked `onnxruntime==1.24.3` ships wheels
  only for cp311+, so `uv sync` on the default 3.10 fails.
- Create the venv (at `<repo>/.venv`):
  ```bash
  UV_PROJECT_ENVIRONMENT="$PWD/.venv" uv sync --extra demo --extra viz --python 3.11
  ```
- Confirmed key versions: `torch 2.12.0+cu130`, `torchvision 0.27.0`,
  `triton 3.7.0`. `torch.cuda.is_available()` → `True`, device reported as
  `NVIDIA RTX PRO 4000 Blackwell`. cu130 runs natively on the CUDA-13 driver.

## Checkpoint (gated)
- `facebook/VGGT-Omega` is gated on Hugging Face; export `HF_TOKEN` (with
  approved access) before downloading.
  ```bash
  HF_TOKEN=... uv run --no-project --with huggingface_hub \
    python -c "from huggingface_hub import hf_hub_download as d; \
    print(d('facebook/VGGT-Omega','vggt_omega_1b_512.pt', local_dir='checkpoints'))"
  ```
- Downloaded `checkpoints/vggt_omega_1b_512.pt` → 4.3 GB, loads as
  **1411 tensors / 1.14B params** (CPU `torch.load` check passed).

## Verified run + memory profile
- Smoke test:
  ```bash
  just smoke   # forest_road.mp4, 4 frames
  ```
  Result: `smoke ok: images=(4,3,384,688) depth=(4,384,688,1) world_points=(4,384,688,3)` (rc=0).
- Measured peak memory for this 4-frame run (1 s sampling):
  | Resource | Peak |
  |---|---|
  | GPU | **5,716 MiB (~5.6 GB)** — min free VRAM ~18.3 GB of 24 GB |
  | System RAM | **31,727 MiB (~31 GB)** (during checkpoint load) |
  - In line with upstream README table (1f≈6.0 GB, 10f≈6.7 GB); ample headroom
    on 24 GB for the larger frame counts (100f≈13.4 GB, 200f≈20.8 GB).
