#!/usr/bin/env bash
# 5× AMD GPU ring-attention bench, FLUX.1-dev at 1008×1008 (28 steps).
#
# IMPORTANT: ring_degree must divide the transformer sequence length
# (h/16 × w/16 + max_sequence_length). At 1024² with 256-token text:
# seq = 4096 + 256 = 4352 → NOT divisible by 5.
# At 1008² with 256-token text: seq = 3969 + 256 = 4225 → divisible by 5 (=845).
# Pick any (h,w) multiple of 16 such that h*w/256 + 256 ≡ 0 (mod 5).
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

# shellcheck source=./_common.sh
source "$SCRIPT_DIR/_common.sh"

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3,4}"
NPROC=5
SIZE="${SIZE:-1008}"
PYTHON="${PYTHON:-python3}"

echo "[5gpu] launching ring_degree=$NPROC at ${SIZE}² on HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
echo "[5gpu] python = $PYTHON"

exec "$PYTHON" -m torch.distributed.run \
    --nproc_per_node="$NPROC" \
    --nnodes=1 \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    bench/flux_ring_attention.py \
    --size "$SIZE" \
    "$@"
