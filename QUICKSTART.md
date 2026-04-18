# Quickstart — 4-GPU Tensor-Parallel FLUX.1-dev on AMD ROCm

## Requirements

| Component | Minimum | Tested |
|---|---|---|
| GPU | 4 × AMD RDNA3, ≥ 16 GB VRAM each | 4 × RX 7800 XT (gfx1101) |
| ROCm | 6.2+ | 7.1.52802 |
| PyTorch | 2.7+rocm | 2.9.1+rocm7.1.1 |
| diffusers | 0.37+ | 0.37.1 |
| Python | 3.10+ | 3.12 |
| RAM | 48 GB+ | 503 GB (T5 encoder spikes to ~12 GB) |
| Disk | 30 GB free | — |

> **Why 4 GPUs and not 5?** FLUX.1-dev has 24 attention heads and inner_dim=3072 (24 × 128).
> Both are divisible by 4 (6 heads/rank, 768 dims/rank) but **not by 5**.
> tp=4 is the only valid world size for this architecture.

---

## Setup

### 1. Install ROCm PyTorch (mandatory — system python3 is CUDA-only on most dual-driver setups)

```bash
python3 -m venv ~/rocm-venv
source ~/rocm-venv/bin/activate

# ROCm wheel — must come from AMD's index, not PyPI
pip install --pre torch==2.9.1 --index-url https://download.pytorch.org/whl/rocm7.1

# Verify ROCm is active (should print True)
python3 -c "import torch; print(torch.cuda.is_available(), torch.version.hip)"
```

### 2. Clone and install dependencies

```bash
git clone https://github.com/Dev-next-gen/diffusers-rocm-parallel
cd diffusers-rocm-parallel
pip install -r requirements.txt   # torch already installed above — this adds diffusers etc.
```

Download FLUX.1-dev (requires HuggingFace account + license acceptance):
```bash
huggingface-cli download black-forest-labs/FLUX.1-dev --local-dir /path/to/FLUX.1-dev
export FLUX_MODEL=/path/to/FLUX.1-dev
```

---

## Run

> **Python path**: the launchers respect `$PYTHON`. If your ROCm venv python
> isn't the system default, set it explicitly:
> ```bash
> export PYTHON=~/rocm-venv/bin/python3   # or wherever your ROCm venv lives
> ```

```bash
# Basic generation — 1024² × 28 steps (~51 s on 4× RX 7800 XT)
./examples/4gpu_flux_tp.sh --prompt "your prompt here" --out output.png

# Custom seed and size
./examples/4gpu_flux_tp.sh \
  --prompt "a red apple on a white table" \
  --size 1024 --steps 28 --seed 42 --out apple.png

# Different model (any FLUX.1-dev fine-tune with same architecture)
FLUX_MODEL=/path/to/your/flux-finetune ./examples/4gpu_flux_tp.sh \
  --prompt "your prompt" --out result.png
```

### Environment variable
```bash
export FLUX_MODEL=/path/to/FLUX.1-dev   # default: /mnt/DATA1/MODELS/FLUX.1-dev
export HF_TOKEN=hf_...                  # only needed if model is gated on HF Hub
```

---

## What it does

The script (`bench/flux_tensor_parallel.py`) runs a **Megatron-style tensor-parallel**
inference pipeline:

```
Step 1  rank 0 only   — encode text with T5 + CLIP  (~10 GB peak, freed after)
Step 2  all ranks     — broadcast embeddings + latents
Step 3  all ranks     — load transformer, apply TP sharding
Step 4  all ranks     — denoising loop (28 steps, all GPUs active simultaneously)
Step 5  rank 0 only   — VAE decode + save image
```

Each GPU holds **~6 GB of transformer weights** (1/4 of QKV + FFN projections).
All 4 GPUs compute in parallel; a single `dist.all_reduce` per RowwiseLinear
synchronises partial sums. Peak VRAM is ~11.2 GB/GPU at 1024².

---

## Expected output

```
[TP-4] 4 GPUs | 28 steps | 1024²
  model: /path/to/FLUX.1-dev
[rank0] loading T5/CLIP for encoding (before transformer)...
[rank0] encoded, GPU after free = 0.08 GB
[rank0] loading transformer (all 4 ranks in parallel)...
[rank0] transformer TP'd, VRAM = 10.97 GB
[rank0] warmup (4 steps)...
[rank0] timed run (28 steps)...
  total=51.3s  step=1.83s
  per-GPU peak: ['11.18', '11.18', '11.18', '11.18'] GB
  total VRAM across GPUs: 44.74 GB
[rank0] saved → output.png
```

---

## Troubleshooting

**`No module named 'diffusers'`**  
Wrong Python — system `python3` is CUDA-only on AMD machines with dual drivers.  
Fix: `export PYTHON=~/rocm-venv/bin/python3` then re-run the launcher.  
Or activate the venv: `source ~/rocm-venv/bin/activate`

**Output is pure noise (not a coherent image)**  
Two known silent bugs — both fixed in the current version:
- TP-1: `proj_out` column slicing in single blocks → see [`docs/tp_bugs.md`](docs/tp_bugs.md#bug-tp-1)
- TP-2: timestep broadcast via temporary tensor → see [`docs/tp_bugs.md`](docs/tp_bugs.md#bug-tp-2)

If you're implementing your own TP and see noise: check that all ranks receive
the **same timestep** at each denoising step. Use `dist.broadcast` on a
pre-allocated CUDA buffer, not on `.cuda()` temporaries.

**OOM at 1024²**  
Peak is ~11.2 GB/GPU. If you're hitting OOM, check that no other process is
using the GPUs: `rocm-smi` or `watch -n1 rocm-smi`.

**Slow first run**  
AOTriton kernels compile on first use (~30–60 s). Subsequent runs use the cache.

---

## Compatible models

Any **FLUX.1-dev fine-tune** with the same architecture (24 heads, hidden_dim=4096,
19 double + 38 single transformer blocks) works without any code change:

```bash
FLUX_MODEL=/path/to/your-flux-finetune ./examples/4gpu_flux_tp.sh --prompt "..."
```

Tested:
- `black-forest-labs/FLUX.1-dev` (reference)
- `FHDR_Uncensored` (fine-tune, same architecture, same timing)
