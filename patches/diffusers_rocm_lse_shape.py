"""
Patch diffusers `_native_flash_attention_forward_op` so its LSE return has the
4D shape expected by `TemplatedRingAttention` on ROCm.

Why: Diffusers 0.37's ring attention merge step expects LSE as [B, H, S, 1].
On torch <2.9 it explicitly `.unsqueeze(-1)`; on torch >=2.9 it assumes the
native SDPA already returns 4D. That assumption holds on CUDA but NOT on ROCm:
`torch.ops.aten._scaled_dot_product_flash_attention` on ROCm 7.1 / AOTriton
returns LSE as [B, H, S] (3D), causing:

    RuntimeError: The size of tensor a (24) must match the size of tensor b
    (128) at non-singleton dimension 3

inside the ring merge. Fix: wrap `_native_flash_attention_forward_op` to
unsqueeze the LSE when it is 3D, regardless of torch version.

Upstream status: needs PR to `huggingface/diffusers` — the is_torch_version
check in `diffusers/models/attention_dispatch.py` conflates CUDA and ROCm
behaviour for torch 2.9+. Proper fix is to test `lse.ndim` or backend, not
torch version.
"""
from __future__ import annotations

import os


def apply() -> None:
    if os.environ.get("DIFFUSERS_ROCM_LSE_SHAPE_PATCH", "1") != "1":
        return

    try:
        from diffusers.models import attention_dispatch as ad
    except Exception:
        return

    original = getattr(ad, "_native_flash_attention_forward_op", None)
    if original is None or getattr(original, "_rocm_lse_patched", False):
        return

    def patched(ctx, query, key, value, *args, **kwargs):
        result = original(ctx, query, key, value, *args, **kwargs)
        # forward_op contract: returns (out, lse) when return_lse, else out
        if isinstance(result, tuple) and len(result) >= 2:
            out, lse = result[0], result[1]
            if lse is not None and lse.ndim < out.ndim:
                lse = lse.unsqueeze(-1)
                result = (out, lse) + result[2:]
        return result

    patched._rocm_lse_patched = True  # type: ignore[attr-defined]
    ad._native_flash_attention_forward_op = patched
    print("[diffusers-rocm-lse-patch] _native_flash_attention_forward_op wrapped")
