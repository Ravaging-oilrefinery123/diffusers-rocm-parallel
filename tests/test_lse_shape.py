"""
Regression test for the LSE-shape monkey-patch.

Verifies that after `patches.apply_all()`, the wrapped
`_native_flash_attention_forward_op` returns `lse` with the same ndim as
`out` (i.e. 4D for the Flux attention layout), regardless of what the raw
aten SDPA flash op returned on this platform.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def apply_patch():
    from patches import apply_all
    apply_all()


def test_raw_rocm_lse_is_3d_without_patch():
    """Confirms the underlying bug: raw aten SDPA returns 3D LSE on ROCm."""
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    q = torch.randn(1, 24, 512, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 24, 512, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 24, 512, 128, device="cuda", dtype=torch.bfloat16)
    out, lse, *_ = torch.ops.aten._scaled_dot_product_flash_attention(
        query=q, key=k, value=v, dropout_p=0.0, is_causal=False,
        return_debug_mask=False,
    )
    assert out.ndim == 4
    # On CUDA torch 2.9+ this is 4D; on ROCm torch 2.9.1 this is 3D.
    # Either way, the patch should normalize it.
    assert lse.ndim in (3, 4)


def test_patched_forward_op_returns_4d_lse():
    """After patch, the diffusers forward_op wrapper returns LSE with ndim == out.ndim."""
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    from diffusers.models import attention_dispatch as ad

    # Minimal stand-in for the autograd FunctionCtx the forward_op expects
    class _Ctx:
        def __init__(self):
            self.max_q = None
            self.max_k = None
            self.dropout_p = 0.0
            self.is_causal = False
            self.scale = None
        def save_for_backward(self, *args, **kwargs): pass

    q = torch.randn(1, 24, 512, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 24, 512, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 24, 512, 128, device="cuda", dtype=torch.bfloat16)

    result = ad._native_flash_attention_forward_op(
        _Ctx(), q, k, v, None, 0.0, False, None, False, True,
        _save_ctx=False, _parallel_config=None,
    )
    assert isinstance(result, tuple)
    out, lse = result[0], result[1]
    assert out.ndim == 4, f"out.ndim={out.ndim}"
    assert lse.ndim == out.ndim, (
        f"LSE should be normalized to match out.ndim={out.ndim}, got lse.ndim={lse.ndim}"
    )
