# The LSE-shape bug

## Symptom

Running any diffusers pipeline with `enable_parallelism(config=ContextParallelConfig(ring_degree=N))` on AMD ROCm (tested on ROCm 7.1, torch 2.9.1) crashes on the first denoising step with:

```
RuntimeError: The size of tensor a (24) must match the size of tensor b (128)
at non-singleton dimension 3
```

Traceback points inside `diffusers/models/attention_dispatch.py`:

```
File "/.../diffusers/models/attention_dispatch.py", line 1909, in forward
    out = prev_out - torch.nn.functional.sigmoid(lse - prev_lse) * (prev_out - out)
```

## Root cause

`TemplatedRingAttention.forward` merges per-iteration attention outputs using the log-sum-exp trick:

```python
out = prev_out - torch.nn.functional.sigmoid(lse - prev_lse) * (prev_out - out)
```

For this broadcast to work, `out` and `lse` must align — on `attention_dispatch.py` the code assumes `out.shape == [B, H, S, D]` and `lse.shape == [B, H, S, 1]` (or something that broadcasts).

A few lines above:

```python
# Refer to:
# https://github.com/huggingface/diffusers/pull/12693#issuecomment-3627519544
if is_torch_version("<", "2.9.0"):
    lse = lse.unsqueeze(-1)
```

The author's assumption: **on torch ≥ 2.9.0, the native SDPA flash backend returns LSE as 4D already**. This is true on CUDA (verified), where `torch.ops.aten._scaled_dot_product_flash_attention` was updated to return LSE with a trailing singleton.

On ROCm 7.1 / AOTriton, the same op still returns LSE as 3D:

```python
>>> import torch
>>> q = torch.randn(1, 24, 512, 128, device='cuda', dtype=torch.bfloat16)
>>> k = torch.randn(1, 24, 512, 128, device='cuda', dtype=torch.bfloat16)
>>> v = torch.randn(1, 24, 512, 128, device='cuda', dtype=torch.bfloat16)
>>> out, lse, *_ = torch.ops.aten._scaled_dot_product_flash_attention(
...     query=q, key=k, value=v, dropout_p=0.0, is_causal=False, return_debug_mask=False)
>>> out.shape
torch.Size([1, 24, 512, 128])
>>> lse.shape
torch.Size([1, 24, 512])        # 3D — CUDA on 2.9+ returns 4D here
```

So on ROCm + torch 2.9+, the `is_torch_version` check skips the unsqueeze, `lse` stays 3D, and the ring-merge broadcast blows up.

## Why 24 vs 128

The error message `size of tensor a (24) must match the size of tensor b (128) at non-singleton dimension 3` reflects the Flux transformer's attention layout after diffusers internal transposes:

- `out` is laid out as `[B, S, H=24, D=128]` at the merge site
- `lse` is laid out as `[B, H=24, S, D?]` from SDPA

With mismatched positional semantics plus the missing unsqueeze, dim 3 collides: 24 (`lse`'s H) vs 128 (`out`'s D).

## Fix

### This repo's monkey-patch

[`patches/diffusers_rocm_lse_shape.py`](../patches/diffusers_rocm_lse_shape.py) wraps `_native_flash_attention_forward_op` so its LSE return is always 4D, regardless of backend:

```python
def patched(ctx, query, key, value, *args, **kwargs):
    result = original(ctx, query, key, value, *args, **kwargs)
    if isinstance(result, tuple) and len(result) >= 2:
        out, lse = result[0], result[1]
        if lse is not None and lse.ndim < out.ndim:
            lse = lse.unsqueeze(-1)
            result = (out, lse) + result[2:]
    return result
```

Applied via `from patches import apply_all; apply_all()` before any diffusers pipeline is loaded.

### Proposed upstream fix

In `diffusers/models/attention_dispatch.py`, replace the torch-version branch with a shape-aware check:

```python
# Before:
if is_torch_version("<", "2.9.0"):
    lse = lse.unsqueeze(-1)

# After:
if lse.ndim < out.ndim:
    lse = lse.unsqueeze(-1)
```

This is backend-agnostic: it works on any platform (CUDA, ROCm, XLA, NPU) regardless of what SDPA returned — the invariant is "LSE must broadcast against out", not "torch version is X".

## Reproducer

```python
import os, sys, torch
import torch.distributed as dist
from diffusers import FluxPipeline, ContextParallelConfig

# Launch with: torchrun --nproc_per_node=2 repro.py
world_size = int(os.environ["WORLD_SIZE"])
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl")

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
).to(f"cuda:{local_rank}")

pipe.transformer.set_attention_backend("_native_flash")
pipe.transformer.enable_parallelism(
    config=ContextParallelConfig(ring_degree=world_size),
)

pipe("a cat", num_inference_steps=4, height=1024, width=1024).images
# → RuntimeError: The size of tensor a (24) must match the size of tensor b (128) ...
```

Apply the patch once at the top:

```python
from patches import apply_all; apply_all()
```

and the same script completes normally.
