#!/usr/bin/env bash
# 5× AMD GPU with FLUX.1-dev bf16 sharded across cards via device_map="balanced".
#
# Purpose: demonstrate running a model larger than any single GPU (~33 GB bf16)
# by spreading the weights across 5× 16 GB cards. No quantization, no offload.
#
# This does NOT use torchrun — a single process drives all GPUs via accelerate's
# device_map. Intra-model activations flow between GPUs over PCIe on demand.
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

# shellcheck source=./_common.sh
source "$SCRIPT_DIR/_common.sh"

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3,4}"

echo "[device_map] running FLUX.1-dev bf16 sharded across HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"

PYTHON="${PYTHON:-python3}"
exec "$PYTHON" bench/flux_device_map_balanced.py "$@"
