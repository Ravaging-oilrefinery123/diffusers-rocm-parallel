#!/usr/bin/env bash
# Shared environment for ROCm multi-GPU diffusion runs. Sourced by the
# launchers in this folder; not meant to run standalone.

# ---- Core ROCm env (tune PYTORCH_ROCM_ARCH for your card) ----
: "${HSA_OVERRIDE_GFX_VERSION:=11.0.1}"
: "${PYTORCH_ROCM_ARCH:=gfx1101}"
export HSA_OVERRIDE_GFX_VERSION PYTORCH_ROCM_ARCH
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export TOKENIZERS_PARALLELISM=false

# ---- RCCL / NCCL heartbeat fixes ----
# Without these, xfuser / native CP commonly hit TCPStore broken-pipe on
# shutdown because the heartbeat monitor keeps polling after process-group
# teardown. Disabling the monitor + extending timeouts avoids the crash.
export TORCH_NCCL_HEARTBEAT_MONITOR_ENABLED=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=9999
export TORCH_NCCL_DUMP_ON_TIMEOUT=0
export TORCH_NCCL_DESYNC_DEBUG=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=0
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=0
export RCCL_LAZY_INIT=0

# ---- Torchao backport patches (if the companion flux-amd-rocm patches are used) ----
export TORCHAO_PIN_MEMORY_PATCH=1
export TORCHAO_STREAM_SYNC_PATCH=1

# ---- Our LSE-shape patch toggle ----
export DIFFUSERS_ROCM_LSE_SHAPE_PATCH=1
