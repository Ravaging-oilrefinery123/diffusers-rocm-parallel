"""
Multi-GPU ring-attention bench for FLUX.1-dev on AMD ROCm.

Uses:
 - diffusers 0.37+ native context parallelism (`enable_parallelism` +
   `ContextParallelConfig(ring_degree=N)`) — NO xfuser wrapper
 - torchao Int8WeightOnlyConfig quantization
 - diffusers group_offload (block_level=8, stream, record_stream)
 - The single LSE shape patch from `patches/`

Launch via torchrun (see `examples/*gpu_flux_ring.sh`).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

# The patch must be applied on every rank before diffusers' context parallel
# code path runs. Keep this at the top, before any diffusers import that
# triggers attention_dispatch initialization.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from patches import apply_all
apply_all()

from diffusers import FluxPipeline, ContextParallelConfig
from diffusers.hooks import apply_group_offloading
from torchao.quantization import Int8WeightOnlyConfig, quantize_


DEFAULT_MODEL = "/mnt/DATA1/MODELS/FLUX.1-dev"
DEFAULT_PROMPT = (
    "cinematic film still of a cat sipping a margarita in a pool in Palm Springs, "
    "highly detailed, cinematic"
)


def build_pipeline(model_path: str, world_size: int, local_rank: int) -> FluxPipeline:
    device = f"cuda:{local_rank}"

    pipe = FluxPipeline.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )

    quantize_(pipe.transformer, Int8WeightOnlyConfig())
    quantize_(pipe.text_encoder_2, Int8WeightOnlyConfig())
    gc.collect()

    # `native` is CP-compatible but returns no LSE; `_native_flash` uses torch
    # SDPA flash kernel (AOTriton on ROCm) which does — but returns LSE 3D, so
    # our patch wraps the forward op to unsqueeze.
    if world_size > 1:
        pipe.transformer.set_attention_backend("_native_flash")
        pipe.transformer.enable_parallelism(
            config=ContextParallelConfig(ring_degree=world_size),
        )

    pipe.transformer.enable_group_offload(
        onload_device=torch.device(device), offload_device=torch.device("cpu"),
        offload_type="block_level", num_blocks_per_group=8,
        use_stream=True, non_blocking=True, record_stream=True,
    )
    pipe.vae.enable_group_offload(
        onload_device=torch.device(device), offload_device=torch.device("cpu"),
        offload_type="leaf_level",
        use_stream=True, non_blocking=True, record_stream=True,
    )
    for enc in (pipe.text_encoder, pipe.text_encoder_2):
        apply_group_offloading(
            enc, onload_device=torch.device(device), offload_device=torch.device("cpu"),
            offload_type="leaf_level",
            use_stream=True, non_blocking=True, record_stream=True,
        )
    return pipe


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default="bench_outputs")
    ap.add_argument("--results-file", default="bench_results.json")
    args = ap.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    if rank == 0:
        print(f"torch={torch.__version__}  HIP={torch.version.hip}")
        print(f"world_size={world_size}  model={args.model}")
        print(f"ring_degree={world_size}  size={args.size}  steps={args.steps}")

    t0 = time.time()
    pipe = build_pipeline(args.model, world_size, local_rank)
    load_dt = time.time() - t0
    if rank == 0:
        print(f"[rank0] pipeline loaded in {load_dt:.1f}s")

    gen = torch.Generator(device="cpu").manual_seed(args.seed)

    if rank == 0:
        print("[rank0] warmup (4 steps)...")
    _ = pipe(
        prompt=args.prompt, num_inference_steps=4,
        guidance_scale=3.5, height=args.size, width=args.size,
        generator=gen, max_sequence_length=256,
    ).images
    torch.cuda.synchronize(local_rank)
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        print("[rank0] warmup done, running timed bench...")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = pipe(
        prompt=args.prompt, num_inference_steps=args.steps,
        guidance_scale=3.5, height=args.size, width=args.size,
        generator=gen, max_sequence_length=256,
    ).images
    torch.cuda.synchronize(local_rank)
    if world_size > 1:
        dist.barrier()
    dt = time.time() - t0

    peak_gb = torch.cuda.max_memory_allocated(local_rank) / 1e9
    peaks_tensor = torch.tensor([peak_gb], device=f"cuda:{local_rank}", dtype=torch.float32)
    if world_size > 1:
        all_peaks = [torch.zeros_like(peaks_tensor) for _ in range(world_size)]
        dist.all_gather(all_peaks, peaks_tensor)
        all_peaks_list = [float(x.item()) for x in all_peaks]
    else:
        all_peaks_list = [peak_gb]

    if rank == 0:
        result = {
            "world_size": world_size,
            "ring_degree": world_size,
            "steps": args.steps,
            "size": args.size,
            "total_s": round(dt, 2),
            "step_s": round(dt / args.steps, 3),
            "peak_vram_per_rank_gb": [round(p, 2) for p in all_peaks_list],
            "peak_vram_max_single_gpu_gb": round(max(all_peaks_list), 2),
            "peak_vram_total_gb": round(sum(all_peaks_list), 2),
            "load_s": round(load_dt, 1),
        }
        print(f"[rank0] total={dt:.1f}s  step={dt/args.steps:.2f}s  "
              f"peak max_gpu={max(all_peaks_list):.2f}GB  total={sum(all_peaks_list):.2f}GB")
        print(f"[rank0] per-rank peak: {[round(p, 2) for p in all_peaks_list]}")

        out_dir = Path(args.output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        if len(out) > 0:
            out[0].save(out_dir / f"flux_ring_ws{world_size}.png")
        Path(args.results_file).write_text(json.dumps(result, indent=2))
        print(f"[rank0] results written to {args.results_file}")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
