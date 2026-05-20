"""AMA-Bench MMLMTruncation-style long-context retrieval for MMFU_Single.

This module implements the **mixed-granularity FIFO** variant used by the
``MMFU_Single`` baseline in this repo. Its shape (one full-context text
item + image-only items, bounded by token/image caps) mirrors
``AMA-Bench/src/method/longcontext.py``'s ``LongContextMethod``, but the
eviction policy is specialised for multi-modal per-turn ingestion:

  * Image cap exceeded → evict the *single* oldest image (image-level
    FIFO; text of that turn stays).
  * Token cap exceeded → evict the *whole* oldest turn (its text +
    any images still attached to it both go away).

Budget:

    L_max  = retrieval.answer_model_ctx - retrieval.answer_model_buffer
             - safety - question_overhead
    T_img  = baselines.MMFU_Single.tokens_per_image    (256 Qwen / 576 GPT,Gemini)
    N_max  = baselines.MMFU_Single.max_images          (25 default; matches AMA-Bench)

Upstream reference (unmodified AMA-Bench): see ``baselines/AMA-Bench/src/method/longcontext.py``.
This file re-implements the algorithm because AMA-Bench's upstream uses a
single 70%-head / 30%-tail cut that does not support the per-turn
image-vs-token split our benchmark needs.
"""

from __future__ import annotations

import re
from typing import Any

from eval_framework.config import (
    resolve_baseline_param,
    resolve_retrieval_answer_model_buffer,
    resolve_retrieval_answer_model_ctx,
)
from eval_framework.datasets.schemas import RetrievalItem, RetrievalRecord

_LC_TOKENIZER: Any = None


class _ApproxTokenizer:
    """Small offline fallback for long-context budget accounting only."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        if not text:
            return []
        pieces = re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\s]", text)
        n_tokens = 0
        for piece in pieces:
            if piece.isascii() and any(ch.isalnum() for ch in piece):
                n_tokens += max(1, (len(piece) + 3) // 4)
            else:
                n_tokens += 1
        return [0] * n_tokens


def lc_tokenizer() -> Any:
    """GPT-2 tokenizer, lazy-loaded + module-cached.

    GPT-2 tokens approximate most modern tokenizers within a few percent —
    good enough for budget accounting. For per-model precision, override
    via ``baselines.MMFU_Single.tokenizer_path`` in config.yaml.
    """
    global _LC_TOKENIZER
    if _LC_TOKENIZER is None:
        from transformers import AutoTokenizer
        path = str(resolve_baseline_param("MMFU_Single", "tokenizer_path", "gpt2"))
        try:
            _LC_TOKENIZER = AutoTokenizer.from_pretrained(path, local_files_only=True)
        except Exception:
            _LC_TOKENIZER = _ApproxTokenizer()
    return _LC_TOKENIZER


def longcontext_retrieve(
    *,
    query: str,
    top_k: int,
    memory_list: list[dict[str, Any]],
    dialogue_id_to_image_paths: dict[str, list[str]],
    trace: dict[str, Any],
) -> RetrievalRecord:
    """Return ``RetrievalRecord`` for the MMFU_Single long-context baseline.

    Algorithm (chronological sliding-window):

        turns_kept = []   # newest at end; each = {idx,text,n_text,images}

        for each turn (oldest → newest):
            append turn → turns_kept
            update total_text_tokens, total_images

            # Image cap: drop ONE oldest image at a time
            while total_images > N_max:
                find earliest turn that still has images
                pop one image from it
                total_images -= 1

            # Token cap: drop oldest WHOLE turn
            while total_text_tokens + total_images * T_img > L_max:
                oldest = turns_kept.pop(0)
                total_text_tokens -= oldest.n_text
                total_images     -= len(oldest.images_remaining)

    The newest content always wins. Token eviction is coarse (whole
    turn) so a token-overflow disposal also removes that turn's images;
    image-only eviction is granular (one image at a time) so individual
    images can rotate out without losing the surrounding text.
    """
    tok = lc_tokenizer()

    ctx_window = resolve_retrieval_answer_model_ctx()
    buffer_len = resolve_retrieval_answer_model_buffer()
    safety = int(resolve_baseline_param("MMFU_Single", "lc_safety_buffer", 300))
    tokens_per_image = int(resolve_baseline_param("MMFU_Single", "tokens_per_image", 256))
    max_images = int(resolve_baseline_param("MMFU_Single", "max_images", 25))

    q_ids = tok.encode(query or "", add_special_tokens=False)
    question_overhead = len(q_ids)
    L_max = max(100, ctx_window - buffer_len - safety - question_overhead)

    turns_kept: list[dict[str, Any]] = []
    total_text_tokens = 0
    total_images = 0
    raw_buffer_tokens_total = 0
    raw_image_count_total = 0
    evicted_turns = 0
    evicted_images = 0

    for idx in range(len(memory_list)):
        mem = memory_list[idx]
        text = str(mem.get("text", "") or "")
        n_text = len(tok.encode(text, add_special_tokens=False))
        raw_buffer_tokens_total += n_text

        did = mem.get("dialogue_id") or ""
        img_paths = [p for p in dialogue_id_to_image_paths.get(did, []) if p]
        raw_image_count_total += len(img_paths)

        turns_kept.append({
            "idx": idx,
            "text": text,
            "n_text": n_text,
            "images": list(img_paths),
        })
        total_text_tokens += n_text
        total_images += len(img_paths)

        while total_images > max_images:
            for t in turns_kept:
                if t["images"]:
                    t["images"].pop(0)
                    total_images -= 1
                    evicted_images += 1
                    break

        while (
            total_text_tokens + total_images * tokens_per_image > L_max
            and turns_kept
        ):
            oldest = turns_kept.pop(0)
            total_text_tokens -= oldest["n_text"]
            total_images -= len(oldest["images"])
            evicted_turns += 1

    kept_text = "\n\n".join(t["text"] for t in turns_kept)
    ordered_paths: list[str] = []
    seen_paths: set[str] = set()
    for t in turns_kept:
        for p in t["images"]:
            if p in seen_paths:
                continue
            seen_paths.add(p)
            ordered_paths.append(p)
    total_images_unique = len(ordered_paths)
    total_image_tokens = total_images * tokens_per_image
    total_tokens = total_text_tokens + total_image_tokens

    items: list[RetrievalItem] = [
        RetrievalItem(
            rank=0,
            memory_id="memgallery:longcontext",
            text=kept_text,
            score=1.0,
            raw_backend_id=None,
            image_path=(ordered_paths[0] if ordered_paths else None),
        )
    ]
    img_slots = max(0, min(top_k - 1, max_images - 1))
    for i, p in enumerate(ordered_paths[1 : 1 + img_slots], start=1):
        items.append(
            RetrievalItem(
                rank=i,
                memory_id=f"memgallery:longcontext:img{i}",
                text="",
                score=1.0,
                raw_backend_id=None,
                image_path=p,
            )
        )

    trace = dict(trace)
    trace.update({
        "mode": "longcontext_mmlm_mixed_fifo",
        "L_max": L_max,
        "T_img": tokens_per_image,
        "max_images": max_images,
        "ctx_window": ctx_window,
        "buffer_reserve": buffer_len,
        "question_overhead": question_overhead,
        "raw_text_tokens": raw_buffer_tokens_total,
        "raw_image_count": raw_image_count_total,
        "kept_text_tokens": total_text_tokens,
        "kept_image_tokens": total_image_tokens,
        "kept_total_tokens": total_tokens,
        "kept_turns": len(turns_kept),
        "kept_images": total_images_unique,
        "total_turns": len(memory_list),
        "evicted_turns": evicted_turns,
        "evicted_images": evicted_images,
        "truncated": evicted_turns > 0 or evicted_images > 0,
    })
    return RetrievalRecord(
        query=query, top_k=top_k, items=items[: max(1, top_k)], raw_trace=trace
    )
