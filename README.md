# diffusers-rocm-parallel

**Multi-GPU tensor / context parallel diffusion on AMD ROCm — with the patch that makes it actually work.**

> **Companion repo:** For the *single-GPU* AMD stack (5 torchao + diffusers backport patches that bring FLUX.1-dev to NVIDIA-class latency on one RX 7800 XT at 72.5 s / 6.4 GB), see [**`flux-amd-rocm`**](https://github.com/Dev-next-gen/flux-amd-rocm). This repo is the **multi-GPU** extension: true Megatron-style tensor parallelism (QKV + FFN sharded) AND context parallelism (ring attention / Ulysses) on AMD ROCm.

Diffusers 0.37 introduced native [context parallelism](https://github.com/huggingface/diffusers/pull/12693) (ring attention, Ulysses) via `model.enable_parallelism()`. It works out of the box on CUDA. On AMD ROCm (torch 2.9+), it crashes on the first denoising step with:

```
RuntimeError: The size of tensor a (24) must match the size of tensor b (128)
at non-singleton dimension 3
```

This repo is the fix, the benches that prove it, and a set of plug-and-play launchers for common multi-GPU diffusion workloads on AMD cards.

---

## TL;DR

```bash
# 1. Install ROCm PyTorch (must be from the AMD wheel index, not PyPI)
python3 -m venv ~/rocm-venv && source ~/rocm-venv/bin/activate
pip install --pre torch==2.9.1 --index-url https://download.pytorch.org/whl/rocm7.1

# 2. Clone + install deps
git clone https://github.com/Dev-next-gen/diffusers-rocm-parallel
cd diffusers-rocm-parallel
pip install -r requirements.txt

# 3. Run (set PYTHON so the launchers use your ROCm venv)
export PYTHON=~/rocm-venv/bin/python3

# 4× RX 7800 XT, FLUX.1-dev bf16, Megatron-style tensor parallelism
./examples/4gpu_flux_tp.sh

# 2× RX 7800 XT, FLUX.1-dev, ring attention
./examples/2gpu_flux_ring.sh
```

You get:
- **4-GPU tensor parallelism**: FLUX.1-dev bf16 in **~51 s**, **11.18 GB per GPU** (all 4 active simultaneously) — no quantization, full bf16 precision
- **Ring attention**: same single-GPU VRAM envelope (6.4 GB) spread across N cards
- No xfuser, no custom transformer wrappers beyond the sharding logic
- Compatible with any FLUX.1-dev fine-tune (same architecture)

→ **[QUICKSTART.md](QUICKSTART.md)** for step-by-step reproduction

---

## The bug

Diffusers' `_templated_context_parallel_attention` ring merge step expects the attention log-sum-exp (`LSE`) tensor to be 4-dimensional — shape `[B, H, S, 1]`. Older torch versions returned LSE as 3D, so diffusers has:

```python
# diffusers/models/attention_dispatch.py
if is_torch_version("<", "2.9.0"):
    lse = lse.unsqueeze(-1)
```

The assumption is that on torch 2.9+, native SDPA already returns LSE as 4D. **This is true on CUDA. It is NOT true on ROCm** — `torch.ops.aten._scaled_dot_product_flash_attention` on ROCm 7.1 / AOTriton still returns LSE as `[B, H, S]`. Without the unsqueeze, the ring merge:

```python
out = prev_out - torch.nn.functional.sigmoid(lse - prev_lse) * (prev_out - out)
```

tries to broadcast a 3D LSE against 4D `out`, and fails.

The patch wraps `_native_flash_attention_forward_op` so that LSE is always 4D, regardless of backend. See [docs/bug.md](docs/bug.md) for the full write-up and reproducer.

---

## Benchmarks

Measured on **RX 7800 XT (gfx1101, 16 GB)**, ROCm 7.1, torch 2.9.1, diffusers 0.37.1. FLUX.1-dev 1024² × 28 steps.

### Tensor Parallelism (Megatron-style QKV + FFN sharding) — bf16, no quantization

<!-- TP_BENCHMARKS_START -->

| Config | Latency | Step | Peak VRAM / GPU | Total VRAM | All GPUs active? |
|---|---|---|---|---|---|
| 1× 7800 XT, single GPU bf16 | ~144 s | ~5.1 s | ~24 GB | 24 GB | — |
| **4× 7800 XT, tp=4 bf16** | **51.5 s** | **1.84 s** | **11.18 GB** | **44.74 GB** | **✅ yes** |

<!-- TP_BENCHMARKS_END -->

**How it works:** QKV projections are column-parallel (each rank holds 6/24 heads), FFN projections are column/row-parallel. All 4 ranks run the full forward pass simultaneously; `dist.all_reduce` at each RowwiseLinear synchronises partial sums. AdaLN norm linears are replicated (small, output must be full-dim on every rank). See [`bench/flux_tensor_parallel.py`](bench/flux_tensor_parallel.py) for the full implementation.

**Key constraint:** FLUX.1-dev has 24 attention heads and inner_dim=3072 (24 × 128). Both are exactly divisible by 4 (6 heads/rank, 768 dims/rank) but NOT by 5 — tp=4 is the natural world size for this architecture.

### Context Parallelism (ring attention) — int8 + group_offload

<!-- CP_BENCHMARKS_START -->

| Config | Latency | Peak VRAM / GPU | Total VRAM |
|---|---|---|---|
| 1× 7800 XT baseline (reference, int8) | 72.5 s | 6.39 GB | 6.39 GB |
| **2× 7800 XT, ring_degree=2** | **102.9 s** | **6.39 GB** | **12.78 GB** |
| *4× 7800 XT, ring_degree=4* | *pending* | *pending* | *pending* |

<!-- CP_BENCHMARKS_END -->

**Reading these numbers:** ring attention with group_offload does NOT speed up 1024² generation on this hardware — PCIe KV-gather communication dominates. The win is that it works at all on AMD (previously impossible), and VRAM per GPU stays flat, so you could fit a larger model or resolution.

---

## What's in this repo

| File | Purpose |
|---|---|
| [`bench/flux_tensor_parallel.py`](bench/flux_tensor_parallel.py) | **4-GPU Megatron-style TP** — FLUX.1-dev bf16, all ranks active simultaneously |
| [`bench/flux_ring_attention.py`](bench/flux_ring_attention.py) | Ring attention bench — FLUX.1-dev + ring attention + group_offload |
| [`bench/flux_device_map_balanced.py`](bench/flux_device_map_balanced.py) | Weight-sharded pipeline via `device_map="balanced"` (single process, sequential) |
| [`examples/_common.sh`](examples/_common.sh) | Shared ROCm env vars sourced by all launchers |
| [`examples/4gpu_flux_tp.sh`](examples/4gpu_flux_tp.sh) | Launcher for TP-4 |
| [`examples/2gpu_flux_ring.sh`](examples/2gpu_flux_ring.sh) | Launcher for ring attention, 2 GPUs |
| [`examples/4gpu_flux_ring.sh`](examples/4gpu_flux_ring.sh) | Launcher for ring attention, 4 GPUs |
| [`examples/5gpu_flux_ring.sh`](examples/5gpu_flux_ring.sh) | Launcher for ring attention, 5 GPUs — 1008² (not yet validated) |
| [`examples/5gpu_flux_device_map.sh`](examples/5gpu_flux_device_map.sh) | `device_map="balanced"` across 5 GPUs — sequential, not recommended for latency |
| [`patches/diffusers_rocm_lse_shape.py`](patches/diffusers_rocm_lse_shape.py) | Monkey-patch fixing the LSE shape bug for ring attention on ROCm |
| [`QUICKSTART.md`](QUICKSTART.md) | Step-by-step reproduction guide |
| [`docs/bug.md`](docs/bug.md) | LSE shape bug write-up, reproducer, proposed upstream fix |
| [`docs/tp_bugs.md`](docs/tp_bugs.md) | The 2 silent TP bugs found during development + root-cause + fix |
| [`docs/performance.md`](docs/performance.md) | When TP / CP helps vs hurts on consumer AMD |
| [`tests/test_lse_shape.py`](tests/test_lse_shape.py) | Regression test for the LSE patch |

---

## Requirements

| Component | Version |
|---|---|
| ROCm | 7.1+ |
| PyTorch | 2.9.1+rocm7.1.1 |
| diffusers | 0.37+ |
| torchao | 0.13 – 0.14.1 (for int8 benches) |
| GPUs | ≥2 RDNA3 (gfx1100 / gfx1101) or CDNA2/3 |

For the full torchao + group_offload stack on AMD (5 other patches), see the companion repo [`flux-amd-rocm`](https://github.com/Dev-next-gen/flux-amd-rocm).

---

## Tensor Parallelism — how it works

The TP implementation in `bench/flux_tensor_parallel.py` is a from-scratch Megatron-style sharding applied as a post-load monkey-patch. No custom model class, no framework dependency beyond PyTorch distributed.

```
                   rank 0          rank 1          rank 2          rank 3
to_q/to_k/to_v    out[0:768]      out[768:1536]   out[1536:2304]  out[2304:3072]
to_out[0]         in[0:768]       in[768:1536]    in[1536:2304]   in[2304:3072]
                                     ← all_reduce →
ff.net[0].proj    out[0:3072]     out[3072:6144]  out[6144:9216]  out[9216:12288]
ff.net[2]         in[0:3072]      in[3072:6144]   in[6144:9216]   in[9216:12288]
                                     ← all_reduce →
```

`out[a:b]` = rank holds those output-dimension rows of the weight matrix (ColwiseParallel).  
`in[a:b]`  = rank holds those input-dimension columns of the weight matrix (RowwiseParallel), followed by `dist.all_reduce`.

- **ColwiseParallel** (`_Col`): rank `i` holds output rows `[i*s:(i+1)*s]`; bias is also sliced. Output is sharded along the output dimension.
- **RowwiseParallel** (`_Row`): rank `i` holds input columns `[i*s:(i+1)*s]`; `dist.all_reduce` after local matmul produces the full replicated output.
- **AdaLN (norm linears)**: replicated across all ranks — their output (shift/scale/gate) must be full-dim everywhere. Post-all_reduce activations are also full-dim, so the element-wise multiply works.
- **Head patching**: `attn.heads` is set to `24 // 4 = 6` per rank so that `unflatten(-1, (heads, -1))` gives the correct local shape `(B, S, 6, 128)`.

The load sequence minimises peak VRAM: encode text (T5, 10 GB) on rank 0 alone → broadcast embeddings → free T5 → all ranks load transformer in parallel → apply TP → each rank retains only its ~11 GB shard.

## Implementation bugs found (and fixed)

Two **silent correctness bugs** were discovered while implementing the TP sharding.
Both cause the model to produce pure noise without any error or crash — the only
diagnostic is visual inspection of the output.

| Bug | Root cause | Symptom |
|---|---|---|
| [TP-1](docs/tp_bugs.md#bug-tp-1) | `proj_out` in single blocks slices contiguous columns instead of the correct non-contiguous `[attn_cols \| mlp_cols]` split | Garbage output from all 38 single-stream blocks |
| [TP-2](docs/tp_bugs.md#bug-tp-2) | `dist.broadcast(tensor.cuda(), src=0)` writes into a temporary — non-rank-0 GPUs keep `timestep=0.0` for all denoising steps | All non-rank-0 ranks compute with wrong time embedding → incoherent all_reduce |

Full root-cause analysis and minimal reproducers: **[docs/tp_bugs.md](docs/tp_bugs.md)**

---

## Upstream status

The LSE shape fix will be filed as a PR against [huggingface/diffusers](https://github.com/huggingface/diffusers). The proper fix is to test `lse.ndim < out.ndim` (or the active backend), not the torch version — the torch version check conflates CUDA and ROCm backend behaviour.

Until then, this monkey-patch is drop-in.

---

## License

MIT.

FLUX.1-dev weights are released by Black Forest Labs under their own [non-commercial license](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md). This repo does not redistribute any model weights.

---

## Credits

- [@Sayak Paul](https://github.com/sayakpaul) and the HuggingFace / PyTorch / TorchAO teams for the upstream diffusers parallelism work
- The ROCm and AOTriton teams at AMD
- Leo Camus — Megatron-style TP implementation, LSE shape backport, AMD-specific patches and reference benchmarks
