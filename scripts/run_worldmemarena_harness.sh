#!/usr/bin/env bash
# Run Harness baselines (Harness-openclaw-gpt, Harness-openclaw-deepseek,
# Harness-codex) over the full WorldMemArena dataset (461 samples).
#
# Usage:
#   cd <repo-root>
#   bash scripts/run_worldmemarena_harness.sh
#
# Override env vars:
#   DATASET_DIR           path to WorldMemArena/ (default: <repo>/WorldMemArena)
#   OUTPUT_DIR            output dir relative to REPO_ROOT
#   HARNESS_KEYS          space-separated harness keys (default: openclaw_gpt openclaw_deepseek codex)
#   DATA150_COUNT         total sample count to consider (default: 461)
#   PER_BASELINE_WORKERS  concurrent samples per baseline (default: 3)
#   MAX_BASELINE_WORKERS  max concurrent baselines (default: 4)

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PACKAGE_DIR="${PACKAGE_DIR:-$REPO_ROOT/eval_framework}"
OUTPUT_DIR="${DATA150_HARNESS_OUTPUT_DIR:-exp_log/worldmemarena_harness}"
LOG_DIR="$REPO_ROOT/$OUTPUT_DIR"
RUN_LOG="$LOG_DIR/run_worldmemarena_harness.log"
PID_FILE="$LOG_DIR/run_worldmemarena_harness.pid"

mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

# ---- Load .env ----
# Prefer .env at the repo root; fall back to eval_framework/.env.
ENV_FILE="${ENV_FILE:-$REPO_ROOT/.env}"
[ -f "$ENV_FILE" ] || ENV_FILE="$PACKAGE_DIR/.env"
[ -f "$ENV_FILE" ] && { set -a; . "$ENV_FILE"; set +a; }

# ---- Harness CLI paths ----
export QWEN3_VL_8B_BASE_URL="${QWEN3_VL_8B_BASE_URL:-http://127.0.0.1:8020/v1}"
export QWEN3_VL_8B_MODEL="${QWEN3_VL_8B_MODEL:-Qwen3-VL-8B-Instruct}"

# If openclaw is not on PATH, set OPENCLAW_ENV_BIN to its bin directory:
#   OPENCLAW_ENV_BIN=/path/to/envs/openclaw/bin
if [[ -n "${OPENCLAW_ENV_BIN:-}" && -x "$OPENCLAW_ENV_BIN/openclaw" ]]; then
  export PATH="$OPENCLAW_ENV_BIN:$PATH"
fi

# If codex is not on PATH, set CODEX_BIN_DIR to its bin directory:
#   CODEX_BIN_DIR=~/.npm-global/bin
if [[ -n "${CODEX_BIN_DIR:-}" && -x "$CODEX_BIN_DIR/codex" ]]; then
  export PATH="$CODEX_BIN_DIR:$PATH"
fi

HARNESS_KEYS_DEFAULT=(openclaw_gpt openclaw_deepseek codex)
if [[ -n "${HARNESS_KEYS:-}" ]]; then
  # shellcheck disable=SC2206
  HARNESS_KEYS_ARR=($HARNESS_KEYS)
else
  HARNESS_KEYS_ARR=("${HARNESS_KEYS_DEFAULT[@]}")
fi

DATASET_DIR="${DATASET_DIR:-$REPO_ROOT/WorldMemArena}"

PY_BASE="${PY_BASE:-python3}"

cmd=(
  "$PY_BASE" -m eval_framework.cli
  --run-data150-openclaw-harness
  --dataset "${DATASET_DIR}"
  --dataset-type worldmemarena
  --output-dir "$OUTPUT_DIR"
  --data150-count "${DATA150_COUNT:-461}"
  --harness-keys "${HARNESS_KEYS_ARR[@]}"
  --per-baseline-workers "${PER_BASELINE_WORKERS:-3}"
  --max-baseline-workers "${MAX_BASELINE_WORKERS:-4}"
)

if [[ -n "${DATA150_SAMPLE_INDICES:-}" ]]; then
  # shellcheck disable=SC2206
  IDS_ARR=($DATA150_SAMPLE_INDICES)
  cmd+=(--data150-sample-indices "${IDS_ARR[@]}")
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  cmd+=(--dry-run)
  "${cmd[@]}"
  exit 0
fi

nohup "${cmd[@]}" >> "$RUN_LOG" 2>&1 &
echo $! > "$PID_FILE"
echo "worldmemarena_harness pid=$(cat "$PID_FILE") log=$RUN_LOG"
