#!/usr/bin/env bash
# Start GME-Qwen2-VL-2B-Instruct vLLM embedding server (port 8014)
# Used by: AUGUSTUSMemory, UniversalRAGMemory
#
# Required env vars (set in eval_framework/.env):
#   GME_MODEL_PATH            — path to downloaded gme-Qwen2-VL-2B-Instruct snapshot
#   VLLM_PYTHON               — Python interpreter with vLLM installed
#
# Optional env vars (defaults shown):
#   GME_HOST                  — 0.0.0.0
#   GME_PORT                  — 8014
#   GME_CUDA_VISIBLE_DEVICES  — 0
#   GME_TP                    — 1
#   GME_GPU_MEMORY_UTILIZATION — 0.40
#   GME_MAX_NUM_SEQS          — 8
#   GME_MODEL                 — gme-Qwen2-VL-2B-Instruct

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Prefer .env at the repo root; fall back to eval_framework/.env.
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/../../.env}"
[ -f "$ENV_FILE" ] || ENV_FILE="$SCRIPT_DIR/../.env"
[ -f "$ENV_FILE" ] && { set -a; . "$ENV_FILE"; set +a; }

: "${GME_MODEL_PATH:?Set GME_MODEL_PATH in eval_framework/.env}"
: "${VLLM_PYTHON:?Set VLLM_PYTHON in eval_framework/.env}"

echo "[run_gme_vllm] Starting GME server on port ${GME_PORT:-8014} ..."
echo "[run_gme_vllm] Model: $GME_MODEL_PATH"

CUDA_VISIBLE_DEVICES="${GME_CUDA_VISIBLE_DEVICES:-0}" \
"$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
  --host "${GME_HOST:-0.0.0.0}" \
  --port "${GME_PORT:-8014}" \
  --model "$GME_MODEL_PATH" \
  --served-model-name "${GME_MODEL:-gme-Qwen2-VL-2B-Instruct}" \
  --runner pooling --convert embed \
  --tensor-parallel-size "${GME_TP:-1}" \
  --max-model-len 8192 \
  --gpu-memory-utilization "${GME_GPU_MEMORY_UTILIZATION:-0.40}" \
  --max-num-seqs "${GME_MAX_NUM_SEQS:-8}" \
  --hf-overrides '{"architectures":["Qwen2VLForConditionalGeneration"]}' \
  --limit-mm-per-prompt '{"image":1}'
