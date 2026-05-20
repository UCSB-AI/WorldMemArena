# MMFU_Single

**Long-context single-model baseline, AMA-Bench-aligned.**

`MMFU_Single` mirrors `AMA-Bench/src/method/longcontext.py`'s
`LongContextMethod`: no embedding, no retrieval, no top-k split. The
answering model is handed the **entire conversation trajectory**
(plus all accessible images) and is expected to answer from that
alone. This is the single-model long-context *upper bound* that
retrieval-based baselines must beat to justify their added complexity.

The backend is `MMFUMemory` (full LinearStorage of every
user+assistant observation). The adapter (`MemGalleryNativeAdapter`)
special-cases this baseline in `retrieve()` to produce a single
context item rather than `top_k=10` lexical-reranked fragments.

## Files

| path | purpose |
|------|---------|
| `longcontext.py` | Mixed-FIFO truncation algorithm (per-turn text + per-image). Loaded by the adapter via `sys.path.insert` → `import longcontext`. |
| `README.md` | This file — algorithm spec, env vars, per-model presets. |

Upstream reference: `baselines/AMA-Bench/src/method/longcontext.py`.
Our `longcontext.py` is a re-implementation because the AMA-Bench
upstream uses a single 70-% head / 30-% tail cut that does not
distinguish image-token from text-token budget — which our multimodal
eval needs.

## How the prompt is built at QA time

```
total_prompt_budget = ANSWER_MODEL_CTX                    # e.g. 128000
                    − ANSWER_MODEL_BUFFER                 # e.g. 8000  (reserve for answer)
                    − LC_SAFETY_BUFFER                    # e.g. 300   (prompt template overhead)
                    − len(tokenize(question))             # dynamic per-QA
```

If the full conversation fits inside `total_prompt_budget`:
→ the entire trajectory goes to the VLM.

If it overflows (e.g. very long multi-session samples vs. a 32k model):
→ **keep the first 70 % of the budget from the HEAD, and the last 30 %
   from the TAIL.** This preserves both kickoff context (project
   setup, stakeholders, goals) and the most recent state (current
   blockers, decisions, results). The middle is dropped.

Images whose `image_id` survives the truncated region are emitted as
additional image-only `RetrievalItem`s (empty text, one `image_path`
each) so `cli.py`'s image-url loop picks them up and forwards them to
the VLM (bounded by `MM_MAX_IMAGES_PER_QA`).

## Environment variables

| env var              | default | meaning                                                |
|----------------------|---------|--------------------------------------------------------|
| `ANSWER_MODEL_CTX`   | 128000  | Answering model's max context in tokens                |
| `ANSWER_MODEL_BUFFER`| 8000    | Reserved for system prompt + answer                    |
| `LC_SAFETY_BUFFER`   | 300     | Extra headroom for prompt template / formatting        |
| `LC_TOKENIZER_PATH`  | gpt2    | HF model id for budget-counting tokenizer              |
| `MM_MODE`            | text    | Set to `image` to forward real image bytes to the VLM  |
| `MM_MAX_IMAGES_PER_QA`| 5      | Cap on number of images attached per VLM call          |

## Per-model presets

```bash
# GPT-4o (128k)
ANSWER_MODEL_CTX=128000 ANSWER_MODEL_BUFFER=8000 \
  python -m eval_framework.cli --baseline MMFU_Single ...

# Claude 3.5 Sonnet (200k)
ANSWER_MODEL_CTX=200000 ANSWER_MODEL_BUFFER=8000 \
  python -m eval_framework.cli --baseline MMFU_Single ...

# Gemini 1.5 Pro (1M)
ANSWER_MODEL_CTX=1000000 ANSWER_MODEL_BUFFER=32000 \
  python -m eval_framework.cli --baseline MMFU_Single ...

# Smaller local model (32k)
ANSWER_MODEL_CTX=32000 ANSWER_MODEL_BUFFER=4000 \
  python -m eval_framework.cli --baseline MMFU_Single ...
```

## Comparison with related baselines

| aspect                      | MMFUMemory                    | MMFU_Single                         |
|-----------------------------|-------------------------------|-------------------------------------|
| Backend memory class        | `MMFUMemory`                  | `MMFUMemory` (identical)            |
| Backend truncation          | `LMTruncation` 50k words      | effectively no-op (adapter handles) |
| Retrieval shape             | top-10 lexical-reranked pieces| 1 full-context item + image-only    |
| Token-budget source         | baseline code constant        | `ANSWER_MODEL_CTX` env var          |
| Overflow policy             | cut tail                      | keep 70% head + 30% tail            |
| Question-overhead aware     | no                            | yes (deducts `len(tokenize(q))`)    |
| Per-model reconfig          | code change                   | env var flip                        |
| Matches AMA-Bench LongContext| no                           | yes (1-to-1 port)                   |

## Entry point

```bash
python -m eval_framework.cli \
    --baseline MMFU_Single \
    --dataset <dataset-dir> \
    --dataset-type domain_a_v2 \
    --output-dir <out> \
    --smoke \
    --max-sessions 10
```

## Inspecting a run

`qa_records.jsonl` stores `retrieval.raw_trace` with per-QA
diagnostics:

```json
{
  "baseline": "MMFU_Single",
  "mode": "longcontext",
  "budget_tokens": 119690,
  "buffer_tokens": 54321,
  "kept_tokens": 54321,
  "truncated": false,
  "ctx_window": 128000,
  "buffer_reserve": 8000,
  "question_overhead": 10,
  "images_available": 47
}
```

- `buffer_tokens`: full conversation's GPT-2-tokenized length
- `kept_tokens`: what actually went to the VLM
- `truncated`: whether head-70 / tail-30 slicing was triggered
- `images_available`: unique images that survived into the kept region
