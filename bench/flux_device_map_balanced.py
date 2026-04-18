"""
Multi-GPU *weight-sharded* inference via `device_map="balanced"`.

This is the second axis of multi-GPU diffusion on AMD: instead of splitting
activations across ranks (ring / ulysses attention), we shard the MODEL
WEIGHTS across all visible GPUs. One pipeline, one forward pass at a time,
but the model itself is physically distributed — so you can run a model that
would not fit on any single GPU.

Unlike the ring-attention bench in this folder, this uses a SINGLE process
(no torchrun). Accelerate routes tensors between GPUs over PCIe as the
forward pass progresses.

Typical use: running FLUX.1-dev in bf16 (~33 GB of weights) on 5× 16 GB cards,
or running SD3/Cogvideo-style models that exceed any consumer GPU's VRAM.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

# Consistency with the CP bench — the LSE patch is a no-op here (no ring
# attention), but importing it makes `apply_all()` idempotent and keeps the
# stack uniform across repos.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from patches import apply_all
apply_all()

import torch
from diffusers import FluxPipeline


DEFAULT_MODEL = "/mnt/DATA1/MODELS/FLUX.1-dev"
DEFAULT_PROMPT = (
    "cinematic film still of a cat sipping a margarita in a pool in Palm Springs, "
    "highly detailed, cinematic"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default="bench_outputs")
    ap.add_argument("--results-file", default="bench_results_device_map.json")
    args = ap.parse_args()

    n_gpu = torch.cuda.device_count()
    print(f"torch={torch.__version__}  HIP={torch.version.hip}")
    print(f"visible GPUs: {n_gpu}")
    for i in range(n_gpu):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name}  {p.total_memory/1e9:.1f} GB")
    print(f"model={args.model}  size={args.size}²  steps={args.steps}")

    t0 = time.time()
    pipe = FluxPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
    )
    load_dt = time.time() - t0
    print(f"pipeline loaded in {load_dt:.1f}s")

    gen = torch.Generator(device="cpu").manual_seed(args.seed)

    print("warmup (4 steps)...")
    _ = pipe(
        prompt=args.prompt, num_inference_steps=4,
        guidance_scale=3.5, height=args.size, width=args.size,
        generator=gen, max_sequence_length=256,
    ).images
    for i in range(n_gpu):
        torch.cuda.synchronize(i)

    for i in range(n_gpu):
        torch.cuda.reset_peak_memory_stats(i)
    gc.collect()

    t0 = time.time()
    out = pipe(
        prompt=args.prompt, num_inference_steps=args.steps,
        guidance_scale=3.5, height=args.size, width=args.size,
        generator=gen, max_sequence_length=256,
    ).images
    for i in range(n_gpu):
        torch.cuda.synchronize(i)
    dt = time.time() - t0

    peaks = []
    for i in range(n_gpu):
        with torch.cuda.device(i):
            peaks.append(torch.cuda.max_memory_allocated() / 1e9)

    result = {
        "mode": "device_map_balanced_bf16",
        "num_gpus": n_gpu,
        "steps": args.steps,
        "size": args.size,
        "total_s": round(dt, 2),
        "step_s": round(dt / args.steps, 3),
        "peak_vram_per_gpu_gb": [round(p, 2) for p in peaks],
        "peak_vram_max_single_gpu_gb": round(max(peaks), 2),
        "peak_vram_total_gb": round(sum(peaks), 2),
        "load_s": round(load_dt, 1),
    }
    print(f"total={dt:.1f}s  step={dt/args.steps:.2f}s  "
          f"peak max_gpu={max(peaks):.2f}GB  total={sum(peaks):.2f}GB")
    print(f"per-GPU peak: {[round(p, 2) for p in peaks]}")

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out[0].save(out_dir / f"flux_device_map_n{n_gpu}.png")
    Path(args.results_file).write_text(json.dumps(result, indent=2))
    print(f"results written to {args.results_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
