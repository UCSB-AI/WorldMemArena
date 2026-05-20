#!/usr/bin/env bash
# Start Qwen3-VL-Embedding-8B vLLM embedding server (port 8013)
# Used by: Qwen3-VL-Embedding-8B baseline
#
# Required env vars (set in eval_framework/.env):
#   QWEN_VL_EMBED_MODEL_PATH          — path to downloaded Qwen3-VL-Embedding-8B snapshot
#   VLLM_PYTHON                       — Python interpreter with vLLM installed
#
# Optional env vars (defaults shown):
#   QWEN_VL_EMBED_HOST                — 0.0.0.0
#   QWEN_VL_EMBED_PORT                — 8013
#   QWEN_VL_EMBED_CUDA_VISIBLE_DEVICES — 1
#   QWEN_VL_EMBED_TP                  — 1
#   QWEN_VL_EMBED_GPU_MEMORY_UTILIZATION — 0.45
#   QWEN_VL_EMBED_MAX_NUM_SEQS        — 8
#   QWEN_VL_EMBED_MODEL               — Qwen3-VL-Embedding-8B

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Prefer .env at the repo root; fall back to eval_framework/.env.
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/../../.env}"
[ -f "$ENV_FILE" ] || ENV_FILE="$SCRIPT_DIR/../.env"
[ -f "$ENV_FILE" ] && { set -a; . "$ENV_FILE"; set +a; }

: "${QWEN_VL_EMBED_MODEL_PATH:?Set QWEN_VL_EMBED_MODEL_PATH in eval_framework/.env}"
: "${VLLM_PYTHON:?Set VLLM_PYTHON in eval_framework/.env}"

echo "[run_qwen_vl_embed_vllm] Starting Qwen3-VL-Embedding server on port ${QWEN_VL_EMBED_PORT:-8013} ..."
echo "[run_qwen_vl_embed_vllm] Model: $QWEN_VL_EMBED_MODEL_PATH"

CUDA_VISIBLE_DEVICES="${QWEN_VL_EMBED_CUDA_VISIBLE_DEVICES:-1}" \
"$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
  --host "${QWEN_VL_EMBED_HOST:-0.0.0.0}" \
  --port "${QWEN_VL_EMBED_PORT:-8013}" \
  --model "$QWEN_VL_EMBED_MODEL_PATH" \
  --served-model-name "${QWEN_VL_EMBED_MODEL:-Qwen3-VL-Embedding-8B}" \
  --runner pooling --convert embed \
  --tensor-parallel-size "${QWEN_VL_EMBED_TP:-1}" \
  --max-model-len 8192 \
  --gpu-memory-utilization "${QWEN_VL_EMBED_GPU_MEMORY_UTILIZATION:-0.45}" \
  --max-num-seqs "${QWEN_VL_EMBED_MAX_NUM_SEQS:-8}" \
  --limit-mm-per-prompt '{"image":1}' \
  --trust-remote-code
