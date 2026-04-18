#!/usr/bin/env bash
# 2× AMD GPU ring-attention bench, FLUX.1-dev 1024² × 28 steps.
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

# shellcheck source=./_common.sh
source "$SCRIPT_DIR/_common.sh"

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1}"
NPROC=2
PYTHON="${PYTHON:-python3}"

echo "[2gpu] launching ring_degree=$NPROC on HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
echo "[2gpu] python = $PYTHON"

exec "$PYTHON" -m torch.distributed.run \
    --nproc_per_node="$NPROC" \
    --nnodes=1 \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    bench/flux_ring_attention.py \
    "$@"
