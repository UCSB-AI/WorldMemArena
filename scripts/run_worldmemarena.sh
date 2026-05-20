#!/usr/bin/env bash
# Run the 11 memory baselines on a 400-sample subset of WorldMemArena
# with Qwen3-VL (local vLLM :8020) as the system-under-test.
# Judge stays on GPT (OPENAI_*_JUDGE from .env).
#
# Baselines (11 total):
#   A-Mem, MGMemory, SimpleMem, Omni-SimpleMem, M2A, ViLoMem, MIRIX,
#   AUGUSTUSMemory, Qwen3-VL-Embedding-8B, UniversalRAGMemory, MMFU_Single
#
# Sample quota (400 total, mirrors original all_qwen_vllm selection):
#   lifelong/project/{academic,education,finance,health,software,startup}: 2 each
#   lifelong/personal: 10
#   agent/gui and agent/embodied subcategories: see QUOTA below
#
# Usage:
#   cd <repo-root>
#   screen -dmS wma bash -lc 'cd <repo-root> && bash scripts/run_worldmemarena.sh'
#
# Override env vars:
#   DATASET_DIR    path to WorldMemArena/ (default: <repo>/WorldMemArena)
#   OUTPUT_DIR_REL output dir relative to REPO_ROOT (default: exp_log/worldmemarena)
#   BASELINES      space-separated baseline list (default: all 11)
#   RUN_TAG        fast|slow — appended to log/csv filenames to allow two parallel drivers
#   MAX_RETRIES    retry count on failures (default: 3)

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PACKAGE_DIR="${PACKAGE_DIR:-$REPO_ROOT/eval_framework}"
OUTPUT_DIR_REL="${OUTPUT_DIR_REL:-exp_log/worldmemarena}"
OUTPUT_DIR_ABS="$REPO_ROOT/$OUTPUT_DIR_REL"

RUN_TAG="${RUN_TAG:-}"
RUN_SUFFIX=""
[ -n "$RUN_TAG" ] && RUN_SUFFIX="_${RUN_TAG}"
PROGRESS_FILENAME="${PROGRESS_FILENAME:-log${RUN_SUFFIX}.csv}"
SELECTED_BASENAME="${SELECTED_BASENAME:-selected_indices${RUN_SUFFIX}.txt}"
RUN_LOG="$OUTPUT_DIR_ABS/run_worldmemarena${RUN_SUFFIX}.log"
PID_FILE="$OUTPUT_DIR_ABS/run_worldmemarena${RUN_SUFFIX}.pid"
SELECTED_FILE="$OUTPUT_DIR_ABS/$SELECTED_BASENAME"
MAX_RETRIES="${MAX_RETRIES:-3}"

if [ "${RUN_TAG:-}" = "slow" ]; then
  PER_BASELINE_WORKERS="${SLOW_PER_BASELINE_WORKERS:-3}"
  MAX_BASELINE_WORKERS="${SLOW_MAX_BASELINE_WORKERS:-2}"
  export PER_BASELINE_WORKERS MAX_BASELINE_WORKERS
fi

mkdir -p "$OUTPUT_DIR_ABS"
cd "$REPO_ROOT"

# ---- Load .env ----
# Prefer .env at the repo root; fall back to eval_framework/.env.
ENV_FILE="${ENV_FILE:-$REPO_ROOT/.env}"
[ -f "$ENV_FILE" ] || ENV_FILE="$PACKAGE_DIR/.env"
[ -f "$ENV_FILE" ] && { set -a; . "$ENV_FILE"; set +a; }

# ---- System-under-test = local Qwen vLLM ----
export OPENAI_MODEL="${OPENAI_MODEL_QWEN_VLLM:-Qwen3-VL-8B-Instruct}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL_QWEN_VLLM:-http://127.0.0.1:8020/v1}"
export OPENAI_MAX_TOKENS="${OPENAI_MAX_TOKENS:-4096}"

# Text embedding stays on real OpenAI so adapter vector schemas remain compatible.
export OPENAI_EMBEDDING_MODEL="${OPENAI_EMBEDDING_MODEL:-text-embedding-3-small}"
export OPENAI_EMBEDDING_BASE_URL="${OPENAI_EMBEDDING_BASE_URL:-https://api.openai.com/v1}"
export OPENAI_EMBEDDING_DIMS="${OPENAI_EMBEDDING_DIMS:-1536}"
echo "[run_worldmemarena] OPENAI_MODEL=$OPENAI_MODEL @ $OPENAI_BASE_URL"
echo "[run_worldmemarena] OPENAI_EMBEDDING_MODEL=$OPENAI_EMBEDDING_MODEL @ $OPENAI_EMBEDDING_BASE_URL"

# ---- Judge must remain GPT ----
: "${OPENAI_MODEL_JUDGE:?set OPENAI_MODEL_JUDGE in .env}"
: "${OPENAI_BASE_URL_JUDGE:?set OPENAI_BASE_URL_JUDGE in .env}"
: "${OPENAI_API_KEY_JUDGE:?set OPENAI_API_KEY_JUDGE in .env}"

# ---- Embedding / multimodal endpoints ----
export QWEN_VL_EMBED_BASE_URL="${QWEN_VL_EMBED_BASE_URL:-http://127.0.0.1:8013/v1}"
export GME_BASE_URL="${GME_BASE_URL:-http://127.0.0.1:8014/v1}"

DATASET_DIR="${DATASET_DIR:-$REPO_ROOT/WorldMemArena}"

# ---- Build the 400-sample 1-based index list ----
python3 - "$SELECTED_FILE" "$DATASET_DIR" <<'PY'
import json, sys, collections, os
from pathlib import Path

out_path = Path(sys.argv[1])
data_root = Path(sys.argv[2])

# Subcategory -> canonical key mapping (matches worldmemarena.py _SUBCAT_MAP values)
SUBCAT_MAP = {
    "agent/gui/excel":        "agent/arena/excel",
    "agent/gui/file_mgmt":    "agent/arena/file_mgmt",
    "agent/gui/image_edit":   "agent/arena/image_edit",
    "agent/gui/web":          "agent/arena/web",
    "agent/gui/word_docs":    "agent/arena/word_docs",
    "agent/gui/css":          "agent/vab/css",
    "agent/gui/mobile":       "agent/vab/mobile",
    "agent/gui/webarena_lite":"agent/vab/webarena-lite",
    "agent/embodied/eb_alfred_base":               "agent/eb_alfred/base",
    "agent/embodied/eb_alfred_common_sense":       "agent/eb_alfred/common_sense",
    "agent/embodied/eb_alfred_complex_instruction":"agent/eb_alfred/complex_instruction",
    "agent/embodied/eb_alfred_long_horizon":       "agent/eb_alfred/long_horizon",
    "agent/embodied/eb_alfred_visual_appearance":  "agent/eb_alfred/visual_appearance",
    "agent/embodied/eb_nav_base":                  "agent/eb_nav/base",
    "agent/embodied/eb_nav_common_sense":          "agent/eb_nav/common_sense",
    "agent/embodied/eb_nav_complex_instruction":   "agent/eb_nav/complex_instruction",
    "agent/embodied/eb_nav_long_horizon":          "agent/eb_nav/long_horizon",
    "agent/embodied/eb_nav_visual_appearance":     "agent/eb_nav/visual_appearance",
    "agent/embodied/minecraft":                    "agent/vab/minecraft",
    "agent/embodied/omnigibson":                   "agent/vab/omnigibson",
    "lifelong/project/academic":  "lifelong/domain_a_v2/academic",
    "lifelong/project/education": "lifelong/domain_a_v2/education",
    "lifelong/project/finance":   "lifelong/domain_a_v2/finance",
    "lifelong/project/health":    "lifelong/domain_a_v2/health",
    "lifelong/project/software":  "lifelong/domain_a_v2/software",
    "lifelong/project/startup":   "lifelong/domain_a_v2/startup",
    "lifelong/personal":          "lifelong/domain_b_v2",
}

QUOTA = {
    "lifelong/domain_a_v2/academic": 2,
    "lifelong/domain_a_v2/education": 2,
    "lifelong/domain_a_v2/finance":   2,
    "lifelong/domain_a_v2/health":    2,
    "lifelong/domain_a_v2/software":  2,
    "lifelong/domain_a_v2/startup":   2,
    "lifelong/domain_b_v2": 10,
    "agent/arena/excel": 21,
    "agent/arena/file_mgmt": 17,
    "agent/arena/image_edit": 28,
    "agent/arena/web": 24,
    "agent/arena/word_docs": 22,
    "agent/eb_alfred/base": 13,
    "agent/eb_alfred/common_sense": 12,
    "agent/eb_alfred/complex_instruction": 12,
    "agent/eb_alfred/long_horizon": 12,
    "agent/eb_alfred/visual_appearance": 10,
    "agent/eb_nav/base": 14,
    "agent/eb_nav/common_sense": 13,
    "agent/eb_nav/complex_instruction": 15,
    "agent/eb_nav/long_horizon": 16,
    "agent/eb_nav/visual_appearance": 11,
    "agent/vab/css": 18,
    "agent/vab/minecraft": 30,
    "agent/vab/mobile": 17,
    "agent/vab/omnigibson": 37,
    "agent/vab/webarena-lite": 36,
}
target = sum(QUOTA.values())
assert target == 400, target

# Enumerate all sample json files sorted (same order as loader)
excluded = {"id_mapping.json"}
all_files = sorted(
    p for p in data_root.rglob("*.json")
    if p.name not in excluded and p.parent != data_root
)

# Map each file to canonical subcategory
def subcat_from_path(p):
    rel = str(p.parent.relative_to(data_root)).replace(os.sep, "/")
    return SUBCAT_MAP.get(rel, rel)

taken = collections.Counter()
picked = []
for i, p in enumerate(all_files, start=1):
    sub = subcat_from_path(p)
    if taken[sub] < QUOTA.get(sub, 0):
        picked.append(i)
        taken[sub] += 1

missing = {k: QUOTA[k] - taken[k] for k in QUOTA if taken[k] != QUOTA[k]}
if missing:
    raise SystemExit(f"quota not satisfied: {missing}")
assert len(picked) == target, (len(picked), target)

out_path.write_text("\n".join(map(str, picked)) + "\n")
print(f"[run_worldmemarena] selected {len(picked)} samples -> {out_path}")
PY

mapfile -t INDICES < "$SELECTED_FILE"
echo "[run_worldmemarena] ${#INDICES[@]} sample indices selected"

# ---- Build driver command ----
DEFAULT_BASELINES=(
  A-Mem MGMemory SimpleMem Omni-SimpleMem M2A
  ViLoMem MIRIX AUGUSTUSMemory Qwen3-VL-Embedding-8B UniversalRAGMemory
  MMFU_Single
)

cmd=(
  python -m eval_framework.cli
  --run-data150-gpt
  --dataset-type worldmemarena
  --dataset "${DATASET_DIR}"
  --output-dir "$OUTPUT_DIR_REL"
  --progress-filename "$PROGRESS_FILENAME"
  --data150-count 461
  --data150-sample-indices "${INDICES[@]}"
  --per-baseline-workers "${PER_BASELINE_WORKERS:-3}"
)
if [ -n "${MAX_BASELINE_WORKERS:-}" ]; then
  cmd+=(--max-baseline-workers "$MAX_BASELINE_WORKERS")
fi
if [ -n "${BASELINES:-}" ]; then
  # shellcheck disable=SC2206
  baseline_arr=(${BASELINES})
  cmd+=(--baselines "${baseline_arr[@]}")
else
  cmd+=(--baselines "${DEFAULT_BASELINES[@]}")
fi

# ---- Retry helper ----
has_unfinished_selected() {
  local csv_file="$OUTPUT_DIR_ABS/$PROGRESS_FILENAME"
  if [ ! -f "$csv_file" ]; then return 0; fi
  python3 - "$csv_file" "$SELECTED_FILE" <<'PY'
import csv, sys
csv_path, sel_path = sys.argv[1], sys.argv[2]
selected = set(open(sel_path).read().split())
with open(csv_path, newline="") as fh:
    rows = list(csv.DictReader(fh))
for row in rows:
    if row.get("id") not in selected:
        continue
    for key, value in row.items():
        if key == "id":
            continue
        if value != "done":
            print(f"unfinished id={row['id']} {key}={value}")
            sys.exit(0)
sys.exit(1)
PY
}

run_loop() {
  for i in $(seq 1 "$MAX_RETRIES"); do
    echo "[run_worldmemarena] $(date -Iseconds) attempt $i/$MAX_RETRIES"
    "${cmd[@]}" || true
    if ! has_unfinished_selected >/dev/null; then
      echo "[run_worldmemarena] all selected cells done after $i round(s)"
      return 0
    fi
    echo "[run_worldmemarena] some cells still unfinished, will retry"
  done
  echo "[run_worldmemarena] reached MAX_RETRIES=$MAX_RETRIES; inspect $OUTPUT_DIR_ABS/$PROGRESS_FILENAME"
  return 1
}

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${cmd[@]}"; echo
  exit 0
fi

echo $$ > "$PID_FILE"
run_loop 2>&1 | tee -a "$RUN_LOG"
