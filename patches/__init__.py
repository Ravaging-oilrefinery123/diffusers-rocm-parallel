"""
Monkey-patches that make Diffusers 0.37+ native context parallelism work on
AMD ROCm.

Usage:

    from patches import apply_all
    apply_all()

    # ... now load your diffusion pipeline as usual and call
    # `model.enable_parallelism(config=ContextParallelConfig(ring_degree=N))`
"""
from .diffusers_rocm_lse_shape import apply as _apply_rocm_lse_shape


def apply_all() -> None:
    """Apply the full patch set. Idempotent, safe to call multiple times."""
    _apply_rocm_lse_shape()
