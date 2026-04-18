#!/usr/bin/env bash
# 4-GPU Tensor-Parallel FLUX.1-dev bf16
# Each GPU holds 1/4 of QKV+FFN weights (~6 GB) — all 4 active in parallel.
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"
source "$SCRIPT_DIR/_common.sh"

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3}"
PYTHON="${PYTHON:-python3}"

echo "[TP-4] launching on HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
echo "[TP-4] python = $PYTHON"

"$PYTHON" -m torch.distributed.run \
  --nproc_per_node=4 \
  --nnodes=1 \
  --master_addr=127.0.0.1 \
  --master_port=29600 \
  bench/flux_tensor_parallel.py "$@"
