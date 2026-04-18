# Tensor-Parallel FLUX bugs — root causes and fixes

Three bugs were found while implementing Megatron-style TP for FLUX.1-dev on AMD ROCm.
All three cause **silent incorrect output** (plausible-looking noise, no crash) rather than errors.

---

## Bug TP-1 — `proj_out` non-contiguous column slicing in single transformer blocks

### Where
`FluxSingleTransformerBlock.proj_out` — the output projection of the 38 single-stream blocks.

### What happens without the fix
`proj_out` takes as input the **concatenation** of two partial outputs:
```
input = cat([attn_partial, mlp_partial], dim=-1)   # (B, S, 768 + 3072) = (B, S, 3840)
```

The first part comes from the TP'd attention (6 heads × 128 = 768 dims per rank),
the second from the TP'd MLP (12288 / 4 = 3072 dims per rank).

In the **original full model**, `proj_out.weight` has shape `(3072, 15360)` where the
15360 input is `[attn_full(3072) | mlp_full(12288)]`.

The naïve row-parallel slicing takes contiguous columns `[rank*3840:(rank+1)*3840]`, but
the correct columns for rank `i` are **non-contiguous**:
- attn columns: `[i*768 : (i+1)*768]`
- mlp columns: `[3072 + i*3072 : 3072 + (i+1)*3072]`

Taking the wrong (contiguous) columns means each rank multiplies its local input
against weights that belong to a different rank's portion — the all_reduce sums
up incompatible partial products → garbage output.

### Symptom
All 38 single blocks produce wrong activations. The final image looks like high-frequency
noise rather than a coherent scene.

### Fix
```python
def _row_split(lin, rank, ws, split_at):
    """Row-parallel for proj_out whose input = [attn_partial | mlp_partial] per rank."""
    a_full = split_at                        # 3072 (attn full dim)
    b_full = lin.weight.shape[1] - split_at  # 12288 (mlp full dim)
    a_s = a_full // ws                       # 768
    b_s = b_full // ws                       # 3072
    w_a = lin.weight.data[:, rank*a_s:(rank+1)*a_s]
    w_b = lin.weight.data[:, split_at + rank*b_s : split_at + (rank+1)*b_s]
    w = torch.cat([w_a, w_b], dim=1).contiguous().cuda()
    b = lin.bias.data.cuda() if lin.bias is not None else None
    return _Row(w, b)

# In apply_tp for single blocks:
blk.proj_out = _row_split(blk.proj_out, rank, ws, split_at=3072)
```

### Why `split_at=3072`
FLUX attention: 24 heads × 128 head_dim = 3072. MLP hidden: 4 × 3072 = 12288.
Total `proj_out` input = 3072 + 12288 = 15360. The boundary is always at 3072.

---

## Bug TP-2 — Timestep broadcast modifies a temporary CUDA tensor

### Where
The distributed setup code that synchronises `timesteps` across ranks before the
denoising loop.

### What happens without the fix
```python
# BROKEN — this is what was written originally:
if rank != 0:
    timesteps_cpu = torch.zeros(ts_len.item(), dtype=torch.float32)
dist.broadcast(timesteps_cpu.cuda(), src=0)   # ← .cuda() creates a TEMPORARY
timesteps = timesteps_cpu.cuda()              # ← ranks 1-3: still all-zeros!
```

`.cuda()` creates a **new tensor** each call. `dist.broadcast` writes into the
first temporary; that temporary is immediately garbage-collected.  
`timesteps_cpu` on non-rank-0 ranks is never updated — it remains all-zeros.  
`timesteps = timesteps_cpu.cuda()` creates yet another zeros tensor.

**Result**: ranks 1, 2, 3 enter the denoising loop with `timestep = 0.0` for
every step, while rank 0 uses the real timestep (~0.86 → ~0.04 over 28 steps).

### Why this produces noise and not a crash
The transformer is fully differentiable at `timestep=0`; it returns plausible-shaped
tensors. But the time embedding (`temb`) is completely different on rank 0 vs ranks 1-3:
- adaLN gates (shift/scale/gate from `norm1`) diverge across ranks
- Each `_Row.all_reduce` sums incompatible partial activations
- The resulting `noise_pred` is random garbage
- The denoising loop accumulates 28 steps of garbage → final latent ≈ noise

The diagnostic that confirmed this: a single-step debug run where we **manually**
passed the correct timestep to all 4 ranks produced a clean `noise_pred`
(std ≈ 1.1, no NaN). The same run with the pipeline's `timesteps` variable
would have used 0.0 on ranks 1-3.

### Fix
```python
# FIXED — allocate the CUDA buffer first, then broadcast in-place:
if rank == 0:
    ts_buf = timesteps.float().contiguous().cuda()
else:
    ts_buf = torch.zeros(ts_len.item(), dtype=torch.float32, device="cuda")
dist.broadcast(ts_buf, src=0)   # in-place on ts_buf — all ranks now have real timesteps
timesteps = ts_buf
```

The rule: **never pass `tensor.cuda()` to `dist.broadcast`** — the result is
discarded. Allocate the buffer, then broadcast.

---

## Bug TP-3 — LSE shape mismatch in ring attention on ROCm (pre-existing)

Documented separately in [`bug.md`](bug.md). Affects ring/context-parallel attention,
not tensor parallelism. Fixed by the patch in `patches/diffusers_rocm_lse_shape.py`.

---

## Summary table

| # | Bug | Symptom | Scope |
|---|---|---|---|
| TP-1 | `proj_out` contiguous column slicing | noise output, no crash | 38 single transformer blocks |
| TP-2 | Timestep broadcast via temporary tensor | noise output, no crash | all non-rank-0 GPUs |
| TP-3 | LSE shape mismatch on ROCm | crash on first step | ring attention only |

Both TP-1 and TP-2 are **silent correctness bugs** — the model runs to completion
at the expected speed with normal VRAM usage, but produces pure noise.
The only way to detect them is to visually inspect the output or compare
`noise_pred` statistics against a reference single-GPU run.
