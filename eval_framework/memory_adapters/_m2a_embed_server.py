"""Local OpenAI-compatible embedding server for M2A.

M2A's ``SemanticStore`` expects two embedding services:
- A text embedding service that returns 384-dim vectors (Milvus ``text_dense``
  schema is hard-wired to ``dim=384``), exposed as POST /v1/embeddings with
  the standard OpenAI ``{input: [...]}`` body.
- A SigLIP2-style multimodal service that the OpenAI SDK reaches via
  POST /embeddings with a ``{messages: [...]}`` body (vLLM's Chat Embeddings
  extension).  The returned vectors are 768-dim and in practice aren't used
  by any live code path (request_3 is built but never wired into the
  hybrid search), so a zero stub is sufficient.

This module can be launched directly as a FastAPI app:

    uvicorn eval_framework.memory_adapters._m2a_embed_server:app \
        --host 127.0.0.1 --port <port>
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from fastapi import FastAPI
from openai import OpenAI

from eval_framework.config import resolve_embedding_base_url, resolve_embedding_model

app = FastAPI()
_TEXT_MODEL_NAME = resolve_embedding_model()
# Upstream M²A's Milvus schema hard-codes text_dense dim=384 (matches
# all-MiniLM-L6-v2).  We pin the OpenAI ``dimensions`` parameter to the
# same value so existing schemas keep working.
_TEXT_DIM = int(os.getenv("LOCAL_EMBEDDING_DIMS") or "384")
_MM_DIM = 768

_openai_client: OpenAI | None = None


def _ensure_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=resolve_embedding_base_url(),
        )
    return _openai_client


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "text_dim": _TEXT_DIM,
        "mm_dim": _MM_DIM,
        "model": _TEXT_MODEL_NAME,
    }


@app.post("/v1/embeddings")
@app.post("/embeddings")
async def embeddings(body: dict[str, Any]) -> dict[str, Any]:
    if "input" in body:
        inputs = body["input"]
        if isinstance(inputs, str):
            inputs = [inputs]
        client = _ensure_openai_client()
        kwargs: dict[str, Any] = {"model": _TEXT_MODEL_NAME, "input": inputs}
        if _TEXT_MODEL_NAME.startswith("text-embedding-3"):
            kwargs["dimensions"] = _TEXT_DIM
        resp = client.embeddings.create(**kwargs)
        data = [
            {
                "embedding": list(item.embedding),
                "index": item.index,
                "object": "embedding",
            }
            for item in resp.data
        ]
        return {
            "object": "list",
            "data": data,
            "model": _TEXT_MODEL_NAME,
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }
    if "messages" in body:
        # Stub multimodal path — return zero vector matching M2A's image_embs dim.
        vec = np.zeros(_MM_DIM, dtype="float32")
        return {
            "object": "list",
            "data": [{"embedding": vec.tolist(), "index": 0, "object": "embedding"}],
            "model": body.get("model", "siglip2-stub"),
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }
    return {
        "object": "list",
        "data": [],
        "model": body.get("model", "unknown"),
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


if __name__ == "__main__":
    import sys
    import uvicorn

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8510
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
