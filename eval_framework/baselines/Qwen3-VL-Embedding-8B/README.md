# Qwen3-VL-Embedding-8B

**Text + image embedding RAG baseline (4096-dim).**

Retrieval-only baseline using Alibaba's Qwen3-VL-Embedding-8B as a
shared text + image encoder. Stores one vector per memory turn
(text + any attached image); retrieval is cosine similarity against
the query embedding, then top-k text (and images, if `MM_MODE=image`)
go to the answer VLM.

## Upstream

| component | source |
|-----------|--------|
| **weights** | HuggingFace `Qwen/Qwen3-VL-Embedding-8B` (auto-downloaded on first load) |
| **loader code** | `sentence_transformers.SentenceTransformer` (pip, `trust_remote_code=True`) |

Weight-only baseline — see the text-variant README
(`baselines/Qwen3-Embedding-8B/README.md`) for the rationale behind
the weight-only / no-clone structure.

This directory holds:

- `qwen_vl_loader.py`: `load_encoder(model_id)` wrapping SentenceTransformer.
  Imported from `memory_adapters/qwen_embed_adapter.py` via sys.path.
  Uniquely named (not just `loader`) so its `sys.modules` entry does
  not collide with the text variant's loader.
- `README.md`: this file.

## Config (eval_framework/config.yaml)

```yaml
baselines:
  Qwen3-VL-Embedding-8B:
    model_id: "Qwen/Qwen3-VL-Embedding-8B"
    batch_size: 8
    dtype: "bfloat16"
    multi_gpu_min_items: 4
```

## Environment overrides

| env | purpose |
|-----|---------|
| `QWEN_VL_EMBED_MODEL` | Override HF model id at runtime |
| `QWEN_EMBED_DEVICE` | Pin to a single GPU (shared with the text variant) |
| `QWEN_EMBED_MULTI_GPU` | Set to `0` to disable the multi-GPU encode pool |
