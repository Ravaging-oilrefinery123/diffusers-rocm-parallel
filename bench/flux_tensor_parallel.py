#!/usr/bin/env python3
"""
4-GPU Tensor-Parallel FLUX.1-dev bf16.
Megatron-style: QKV + FFN sharded column/row-wise across 4 ranks.
Each GPU holds ~6 GB of transformer weights instead of 24 GB.

Usage:
    PYTHON=/path/to/rocm-venv/bin/python3 ./examples/4gpu_flux_tp.sh [--steps N] [--size S]

Environment:
    FLUX_MODEL   path to FLUX.1-dev (default /mnt/DATA1/MODELS/FLUX.1-dev)
    HF_TOKEN     required if model is gated
"""
import os, time, argparse
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from diffusers import FluxPipeline, FluxTransformer2DModel

MODEL   = os.environ.get("FLUX_MODEL", "/mnt/DATA1/MODELS/FLUX.1-dev")
PROMPT  = "a dragon coiled around a medieval tower at sunset, fantasy art"


# ─── Column / Row parallel linears ────────────────────────────────────────────

class _Col(nn.Module):
    """Column-parallel linear: rank i holds output rows [i*s:(i+1)*s]."""
    def __init__(self, w, b=None):
        super().__init__()
        self.weight = nn.Parameter(w)
        self.bias   = nn.Parameter(b) if b is not None else None
    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

class _Row(nn.Module):
    """Row-parallel linear: rank i holds input cols [i*s:(i+1)*s]; all_reduce after."""
    def __init__(self, w, full_bias=None):
        super().__init__()
        self.weight    = nn.Parameter(w)
        self._full_bias = full_bias          # replicated, added post-reduce

    def forward(self, x):
        out = F.linear(x, self.weight)
        dist.all_reduce(out)
        if self._full_bias is not None:
            out = out + self._full_bias
        return out

def _col(lin, rank, ws):
    s = lin.weight.shape[0] // ws
    w = lin.weight.data[rank*s:(rank+1)*s].contiguous().cuda()
    b = lin.bias.data[rank*s:(rank+1)*s].contiguous().cuda() if lin.bias is not None else None
    return _Col(w, b)

def _row(lin, rank, ws):
    s = lin.weight.shape[1] // ws
    w = lin.weight.data[:, rank*s:(rank+1)*s].contiguous().cuda()
    b = lin.bias.data.cuda() if lin.bias is not None else None
    return _Row(w, b)

def _row_split(lin, rank, ws, split_at):
    """Row-parallel for proj_out in single blocks whose input = [attn_partial | mlp_partial].
    The two parts are non-contiguous in the original weight — slice them separately.
    split_at: the boundary between attn and mlp in the full input dim (= attn_full_dim = 3072).
    """
    a_full = split_at                       # 3072
    b_full = lin.weight.shape[1] - split_at  # 12288
    a_s = a_full // ws                      # 768
    b_s = b_full // ws                      # 3072
    w_a = lin.weight.data[:, rank*a_s:(rank+1)*a_s]
    w_b = lin.weight.data[:, split_at + rank*b_s:split_at + (rank+1)*b_s]
    w = torch.cat([w_a, w_b], dim=1).contiguous().cuda()
    b = lin.bias.data.cuda() if lin.bias is not None else None
    return _Row(w, b)


# ─── TP sharding ──────────────────────────────────────────────────────────────

def apply_tp(transformer, rank, ws):
    """
    Replace attention QKV/out and FFN projections with TP-aware modules.
    Norm linears (adaLN) stay replicated — they are small and their output
    (shift/scale/gate) must be full-dim on every rank.
    """
    hpr = transformer.config.num_attention_heads // ws  # 24 // 4 = 6

    # ── Double transformer blocks (MM-DiT) ────────────────────────────────
    for blk in transformer.transformer_blocks:
        a = blk.attn

        # Image-stream QKV
        a.to_q = _col(a.to_q, rank, ws)
        a.to_k = _col(a.to_k, rank, ws)
        a.to_v = _col(a.to_v, rank, ws)
        a.to_out[0] = _row(a.to_out[0], rank, ws)

        # Text-stream QKV
        a.add_q_proj = _col(a.add_q_proj, rank, ws)
        a.add_k_proj = _col(a.add_k_proj, rank, ws)
        a.add_v_proj = _col(a.add_v_proj, rank, ws)
        a.to_add_out = _row(a.to_add_out, rank, ws)

        # Patch head count so unflatten(-1, (heads, -1)) uses local count
        a.heads = hpr

        # FFN - image stream  (ff.net[0] is GELU wrapper with .proj child)
        blk.ff.net[0].proj = _col(blk.ff.net[0].proj, rank, ws)
        blk.ff.net[2]      = _row(blk.ff.net[2],      rank, ws)

        # FFN - text stream
        blk.ff_context.net[0].proj = _col(blk.ff_context.net[0].proj, rank, ws)
        blk.ff_context.net[2]      = _row(blk.ff_context.net[2],      rank, ws)


    # ── Single transformer blocks ─────────────────────────────────────────
    for blk in transformer.single_transformer_blocks:
        a = blk.attn

        a.to_q = _col(a.to_q, rank, ws)
        a.to_k = _col(a.to_k, rank, ws)
        a.to_v = _col(a.to_v, rank, ws)
        a.heads = hpr

        # proj_mlp: up-proj (3072→12288), ColwiseParallel → 3072/rank
        blk.proj_mlp = _col(blk.proj_mlp, rank, ws)

        # proj_out input = [attn_out(768) | mlp_out(3072)] per rank — non-contiguous cols
        blk.proj_out = _row_split(blk.proj_out, rank, ws, split_at=3072)

    return transformer


# ─── Broadcast helpers ────────────────────────────────────────────────────────

def _bcast(t, src=0):
    dist.broadcast(t, src=src)
    return t

def broadcast_tensor(t, rank, src=0):
    """Broadcast tensor from rank src to all others. Non-src ranks pass t=None."""
    # meta: [ndim, d0, d1, ..., dtype_id]  — max 8 dims → buffer of 10
    meta = torch.zeros(10, dtype=torch.long, device="cuda")
    if rank == src:
        ndim = t.ndim
        dtype_id = {torch.bfloat16: 0, torch.float16: 1, torch.float32: 2}[t.dtype]
        meta[0] = ndim
        meta[1:1+ndim] = torch.tensor(list(t.shape), dtype=torch.long)
        meta[9] = dtype_id
    dist.broadcast(meta, src=src)
    ndim     = meta[0].item()
    shape    = meta[1:1+ndim].tolist()
    dtype    = [torch.bfloat16, torch.float16, torch.float32][meta[9].item()]
    if rank == src:
        out = t.contiguous().cuda()
    else:
        out = torch.empty(shape, dtype=dtype, device="cuda")
    dist.broadcast(out, src=src)
    return out


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default=MODEL)
    parser.add_argument("--prompt",  default=PROMPT)
    parser.add_argument("--steps",   type=int, default=28)
    parser.add_argument("--size",    type=int, default=1024)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--out",     default="tp4_output.png")
    parser.add_argument("--warmup",  type=int, default=4)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    ws   = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(rank)

    if rank == 0:
        print(f"[TP-4] {ws} GPUs | {args.steps} steps | {args.size}²")
        print(f"  model: {args.model}")

    # ── Step 1: rank 0 encodes text FIRST (T5 ~10 GB alone on GPU, fits) ──
    # Non-rank-0 ranks wait. GPU freed before transformer loads (step 2).
    pipe = None
    if rank == 0:
        print("[rank0] loading T5/CLIP for encoding (before transformer)...")
        pipe = FluxPipeline.from_pretrained(
            args.model, transformer=None, torch_dtype=torch.bfloat16,
        )
        pipe.text_encoder   = pipe.text_encoder.to(device)    # CLIP  ~0.5 GB
        pipe.text_encoder_2 = pipe.text_encoder_2.to(device)  # T5   ~10.0 GB

        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
                prompt=args.prompt, prompt_2=None, max_sequence_length=256,
            )

        # Free text encoders immediately — GPU back to ~0.5 GB
        pipe.text_encoder   = pipe.text_encoder.cpu()
        pipe.text_encoder_2 = pipe.text_encoder_2.cpu()
        torch.cuda.empty_cache()
        print(f"[rank0] encoded, GPU after free = {torch.cuda.memory_allocated(device)/1e9:.2f} GB")

        # Prepare latents + scheduler while GPU is still light
        pipe.vae = pipe.vae.to(device)
        latents, latent_image_ids = pipe.prepare_latents(
            batch_size=1,
            num_channels_latents=16,   # FLUX VAE latent channels pre-pack (64 // 4)
            height=args.size,
            width=args.size,
            dtype=torch.bfloat16,
            device=device,
            generator=torch.Generator(device=device).manual_seed(args.seed),
        )
        # FLUX scheduler uses dynamic shifting — requires mu from image seq len
        import numpy as np
        from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
        image_seq_len = (args.size // pipe.vae_scale_factor // 2) ** 2
        mu = calculate_shift(
            image_seq_len,
            pipe.scheduler.config.get("base_image_seq_len", 256),
            pipe.scheduler.config.get("max_image_seq_len", 4096),
            pipe.scheduler.config.get("base_shift", 0.5),
            pipe.scheduler.config.get("max_shift", 1.16),
        )
        sigmas = np.linspace(1.0, 1 / args.steps, args.steps)
        timesteps, _ = retrieve_timesteps(
            pipe.scheduler, args.steps, device, sigmas=sigmas, mu=mu,
        )
        timesteps_cpu = timesteps.cpu()
        pipe.vae = pipe.vae.cpu()
        torch.cuda.empty_cache()

    # ── Step 2: broadcast embeddings + latents BEFORE loading transformer ─
    dist.barrier()
    prompt_embeds        = broadcast_tensor(prompt_embeds        if rank == 0 else None, rank)
    pooled_prompt_embeds = broadcast_tensor(pooled_prompt_embeds if rank == 0 else None, rank)
    text_ids             = broadcast_tensor(text_ids             if rank == 0 else None, rank)
    latents              = broadcast_tensor(latents              if rank == 0 else None, rank)
    latent_image_ids     = broadcast_tensor(latent_image_ids     if rank == 0 else None, rank)

    ts_len = torch.tensor([args.steps], dtype=torch.long, device="cuda")
    dist.broadcast(ts_len, src=0)
    # Allocate a CUDA buffer first, then broadcast in-place so all ranks get the real values.
    # (the old `dist.broadcast(cpu.cuda(), src=0)` pattern modifies a temporary — the original
    # stays at zeros on non-rank-0 ranks, giving timestep=0 in the denoising loop.)
    if rank == 0:
        ts_buf = timesteps.float().contiguous().cuda()
    else:
        ts_buf = torch.zeros(ts_len.item(), dtype=torch.float32, device="cuda")
    dist.broadcast(ts_buf, src=0)
    timesteps = ts_buf
    dist.barrier()

    # ── Step 3: all ranks load transformer in parallel (GPU now clear) ────
    if rank == 0:
        print("[rank0] loading transformer (all 4 ranks in parallel)...")
    transformer = FluxTransformer2DModel.from_pretrained(
        args.model, subfolder="transformer",
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    transformer = apply_tp(transformer, rank, ws)
    transformer = transformer.to(device, dtype=torch.bfloat16)
    transformer.eval()
    dist.barrier()

    vram_after_tp = torch.cuda.memory_allocated(device) / 1e9
    if rank == 0:
        print(f"[rank{rank}] transformer TP'd, VRAM = {vram_after_tp:.2f} GB")

    # ── Step 4: denoising loop ─────────────────────────────────────────────
    guidance = torch.full([1], 3.5, device=device, dtype=torch.bfloat16)

    def run_denoising(latents, steps_override=None):
        ts = timesteps if steps_override is None else timesteps[:steps_override]
        lat = latents.clone()
        for i, t in enumerate(ts):
            t_tensor = t.expand(lat.shape[0]).to(device)
            with torch.no_grad():
                noise_pred = transformer(
                    hidden_states=lat,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    timestep=t_tensor / 1000.0,
                    img_ids=latent_image_ids,
                    txt_ids=text_ids,
                    guidance=guidance,
                    return_dict=False,
                )[0]
            # Euler step (FlowMatch)
            sigma = t.to(device, dtype=torch.float32) / 1000.0
            sigma_next = (timesteps[i+1].to(device, dtype=torch.float32) / 1000.0
                          if i + 1 < len(ts) else torch.zeros_like(sigma))
            dt = sigma_next - sigma
            lat = lat + noise_pred.to(torch.float32) * dt
            lat = lat.to(torch.bfloat16)
        return lat

    # Warmup
    if rank == 0:
        print(f"[rank0] warmup ({args.warmup} steps)...")
    _ = run_denoising(latents, steps_override=args.warmup)
    torch.cuda.synchronize()
    dist.barrier()

    # Timed run
    if rank == 0:
        print(f"[rank0] timed run ({args.steps} steps)...")
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    t0 = time.time()

    final_latents = run_denoising(latents)

    torch.cuda.synchronize()
    dist.barrier()
    elapsed = time.time() - t0
    peak_vram = torch.cuda.max_memory_allocated(device) / 1e9

    if rank == 0:
        print(f"\n{'='*55}")
        print(f"  total={elapsed:.1f}s  step={elapsed/args.steps:.2f}s")
        print(f"  peak VRAM rank0={peak_vram:.2f} GB")

    # Gather peak VRAM from all ranks
    peak_t = torch.tensor([peak_vram], device=device)
    all_peaks = [torch.zeros(1, device=device) for _ in range(ws)]
    dist.all_gather(all_peaks, peak_t)
    if rank == 0:
        peaks = [p.item() for p in all_peaks]
        print(f"  per-GPU peak: {[f'{p:.2f}' for p in peaks]} GB")
        print(f"  total VRAM across GPUs: {sum(peaks):.2f} GB")
        print(f"{'='*55}")

    # ── Step 5: VAE decode on rank 0 ──────────────────────────────────────
    if rank == 0:
        print("[rank0] VAE decode...")
        # Free transformer from GPU 0 — no more distributed ops needed
        del transformer
        torch.cuda.empty_cache()
        pipe.vae = pipe.vae.to(device)
        pipe.vae.enable_tiling()   # tile decode to stay within VRAM
        with torch.no_grad():
            final_latents_unpacked = pipe._unpack_latents(
                final_latents, height=args.size, width=args.size,
                vae_scale_factor=pipe.vae_scale_factor,
            )
            final_latents_unpacked = (
                final_latents_unpacked / pipe.vae.config.scaling_factor
                + pipe.vae.config.shift_factor
            )
            image = pipe.vae.decode(final_latents_unpacked, return_dict=False)[0]
        image = pipe.image_processor.postprocess(image, output_type="pil")
        image[0].save(args.out)
        print(f"[rank0] saved → {args.out}")

        import json
        result = {
            "config": "tp4_bf16",
            "gpus": ws,
            "steps": args.steps,
            "size": args.size,
            "total_s": round(elapsed, 2),
            "step_s":  round(elapsed / args.steps, 3),
            "peak_vram_per_gpu_gb": [round(p, 3) for p in peaks],
            "peak_max_gpu_gb": round(max(peaks), 3),
            "total_vram_gb": round(sum(peaks), 3),
        }
        out_json = args.out.replace(".png", "_results.json")
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[rank0] results → {out_json}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
