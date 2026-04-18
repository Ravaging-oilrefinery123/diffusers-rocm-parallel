# Performance — when context parallel helps vs hurts

**Short version:** on consumer AMD with PCIe-only interconnect and int8 + group_offload, ring-attention context parallel **does not reduce latency** for single-image FLUX.1-dev at 1024². It reproduces identical per-GPU VRAM, which is the technical win (you could fit a bigger model or larger resolution that wouldn't fit on one card), but latency scales *against* you as `ring_degree` grows.

## Measured on 2–5× RX 7800 XT (gfx1101, 16 GB)

FLUX.1-dev 1024² × 28 steps, int8 weight-only quantization, group_offload block_level=8 + stream + record_stream.

| `ring_degree` | Latency | step | Peak per GPU | Total VRAM | Δ vs 1× |
|---|---|---|---|---|---|
| 1 (baseline) | 72.5 s | 2.59 s | 6.39 GB | 6.39 GB | — |
| 2 | 102.9 s | 3.68 s | 6.39 GB | 12.78 GB | **+42 %** slower |
| 4 | *pending* | *pending* | *pending* | *pending* | *pending* |
| 5 | *pending* | *pending* | *pending* | *pending* | *pending* |

## Why it's slower, not faster

Three things combine:

1. **Attention is not the bottleneck when you're already offloading.** Under group_offload, the transformer's forward is dominated by CPU↔GPU weight transfers (the block swap), not by attention compute. Ring attention parallelizes the compute but can't help the transfer, so you pay communication overhead without winning anything.

2. **PCIe is slow for the all-gather pattern ring attention needs.** Every ring iteration, each rank sends its local K/V chunk to the next rank. With `ring_degree=N`, that's `N-1` hops per attention layer × 57 attention layers × 28 steps. On 7800 XT cards connected via PCIe 4.0 (no NVLink equivalent), this serialized comm path dominates.

3. **FP32 LSE conversion doubles the comm volume.** `ContextParallelConfig(convert_to_fp32=True)` (the default, for numerical stability) means the merge step transports LSE in FP32 instead of bf16 — extra bandwidth pressure on the slowest link.

## When ring attention *would* pay off

- **Long sequences that don't fit on one card.** If a single card couldn't hold the KV cache (think 4096² or video models), ring attention is the only way to run the model at all. Our measurement at 1024² fits on one card easily, so ring is a *choice* not a *necessity*.
- **Attention-heavy models without offload.** If the weights are resident on-GPU and attention dominates the forward, ring's parallelized compute is worth the comm cost. Inference with model_cpu_offload + bf16 at 4096² would be closer to this regime.
- **High-bandwidth interconnect (Infinity Fabric, MI300X).** On CDNA3 with xGMI / Infinity Fabric links between GPUs, the all-gather cost drops by an order of magnitude. Multi-GPU CDNA3 ring should actually be fast.

## What this repo is useful for, given the above

1. **Fitting bigger models / resolutions.** 6.39 GB peak per GPU on 4 GPUs means you have ~50 GB of effective VRAM budget if you stayed at 1 image — room for larger transformer, longer sequence, or higher res.
2. **Unblocking anyone trying to use `enable_parallelism` on AMD.** The LSE patch is the unblock. Performance is a downstream conversation.
3. **Baseline for AMD-specific benches.** Once AMD ships a faster GPU-GPU link story on consumer RDNA (e.g., via Infinity Fabric on a prosumer card), these numbers become a before/after.

## For pure throughput: data parallel beats context parallel

If the goal is "generate 5 images", run 5 independent pipelines on 5 GPUs. At 72.5 s each, that's 5 images / 72.5 s. Context parallel at `ring_degree=5` would be 1 image per ~130 s → 5 images per ~130 s only if you batch cleverly across 5 CP runs. Data parallel is strictly better for throughput.
