"""Shared checkpoint QA: retrieval via adapter + answer from an injected callable."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from eval_framework.datasets.wma_bundle import NormalizedCheckpoint, NormalizedCheckpointQuestion
from eval_framework.datasets.schemas import RetrievalItem, RetrievalRecord
from eval_framework.memory_adapters.base import MemoryAdapter
from eval_framework.openai_compat import merge_token_usage, token_usage_scope
from eval_framework.pipeline.records import PipelineCheckpointQARecord

AnswerFn = Callable[[NormalizedCheckpointQuestion, RetrievalRecord], Any]


def _normalize_answer_output(raw: Any) -> tuple[str, dict[str, int]]:
    """Accept legacy str answers plus richer dict/tuple outputs from custom answer_fns."""
    if isinstance(raw, dict):
        answer = raw.get("answer", raw.get("generated_answer", ""))
        return str(answer), dict(raw.get("token_usage") or {})
    if isinstance(raw, tuple) and raw:
        answer = raw[0]
        usage = raw[1] if len(raw) > 1 and isinstance(raw[1], dict) else {}
        return str(answer), dict(usage)
    return str(raw), {}


def _source_sessions_for_item(
    item: RetrievalItem,
    *,
    memory_session_map: dict[str, str],
    image_session_map: dict[str, str],
) -> list[str]:
    sessions: list[str] = []
    for key in (item.memory_id, item.raw_backend_id):
        if key and key in memory_session_map and memory_session_map[key] not in sessions:
            sessions.append(memory_session_map[key])
    if item.image_path and item.image_path in image_session_map:
        sid = image_session_map[item.image_path]
        if sid not in sessions:
            sessions.append(sid)
    text = item.text or ""
    for image_key, sid in image_session_map.items():
        if image_key and image_key in text and sid not in sessions:
            sessions.append(sid)
    return sessions


def _with_answer_context(
    retrieval: RetrievalRecord,
    *,
    answer_evidence_mode: str,
    session_dialogues: dict[str, str],
    memory_session_map: dict[str, str],
    image_session_map: dict[str, str],
) -> RetrievalRecord:
    mode = answer_evidence_mode.strip().lower()
    if mode not in {"memory", "session"}:
        mode = "memory"

    raw_trace = dict(retrieval.raw_trace or {})
    raw_trace["answer_evidence_mode"] = mode

    source_by_rank: dict[str, list[str]] = {}
    context_items: list[dict[str, object]] = []
    seen_sessions: set[str] = set()
    fallback_count = 0

    if mode == "session":
        for item in retrieval.items[: retrieval.top_k]:
            source_sessions = _source_sessions_for_item(
                item,
                memory_session_map=memory_session_map,
                image_session_map=image_session_map,
            )
            source_by_rank[str(item.rank)] = source_sessions
            had_source = bool(source_sessions)
            added = False
            for sid in source_sessions:
                if sid in seen_sessions:
                    continue
                text = session_dialogues.get(sid, "")
                if not text.strip():
                    continue
                seen_sessions.add(sid)
                added = True
                context_items.append({
                    "rank": item.rank,
                    "source_type": "session",
                    "source_session_id": sid,
                    "text": text,
                })
            if not had_source and not added and item.text.strip():
                fallback_count += 1
                context_items.append({
                    "rank": item.rank,
                    "source_type": "memory_fallback",
                    "source_session_id": None,
                    "text": item.text,
                })
    else:
        for item in retrieval.items[: retrieval.top_k]:
            source_sessions = _source_sessions_for_item(
                item,
                memory_session_map=memory_session_map,
                image_session_map=image_session_map,
            )
            source_by_rank[str(item.rank)] = source_sessions

    raw_trace["retrieval_source_sessions_by_rank"] = source_by_rank
    raw_trace["answer_context_items"] = context_items
    raw_trace["answer_context_fallback_count"] = fallback_count
    return RetrievalRecord(
        query=retrieval.query,
        top_k=retrieval.top_k,
        items=retrieval.items,
        raw_trace=raw_trace,
    )


def _is_harness_native(adapter: MemoryAdapter) -> bool:
    caps = adapter.get_capabilities()
    return caps.get("backend") == "HarnessNativeMemory"


def _run_checkpoint_qa_batched_harness(
    adapter: MemoryAdapter,
    *,
    sample_id: str,
    sample_uuid: str,
    checkpoint: NormalizedCheckpoint,
    top_k: int,
    answer_evidence_mode: str,
    session_dialogues: dict[str, str] | None,
    memory_session_map: dict[str, str] | None,
    image_session_map: dict[str, str] | None,
) -> tuple[PipelineCheckpointQARecord, ...]:
    """Batch all checkpoint questions into a single harness call (black-box)."""
    questions = checkpoint.questions
    if not questions:
        return ()

    retrieval_start = time.perf_counter()
    with token_usage_scope() as retrieval_usage:
        retrieval = adapter.retrieve(
            questions[0].question, top_k, category=questions[0].question_type_abbrev
        )
    retrieval = _with_answer_context(
        retrieval,
        answer_evidence_mode=answer_evidence_mode,
        session_dialogues=session_dialogues or {},
        memory_session_map=memory_session_map or {},
        image_session_map=image_session_map or {},
    )
    retrieval_seconds = time.perf_counter() - retrieval_start

    answer_start = time.perf_counter()
    with token_usage_scope() as answer_usage_scope:
        answers = adapter.answer_batch([q.question for q in questions])
    answer_seconds = time.perf_counter() - answer_start
    answer_usage = dict(answer_usage_scope)
    per_q_answer_seconds = answer_seconds / max(len(questions), 1)

    out: list[PipelineCheckpointQARecord] = []
    for i, q in enumerate(questions):
        out.append(
            PipelineCheckpointQARecord(
                sample_id=sample_id,
                sample_uuid=sample_uuid,
                checkpoint_id=checkpoint.checkpoint_id,
                question=q.question,
                gold_answer=q.gold_answer,
                gold_evidence_memory_ids=q.gold_evidence_memory_ids,
                gold_evidence_contents=q.gold_evidence_contents,
                question_type=q.question_type,
                question_type_abbrev=q.question_type_abbrev,
                difficulty=q.difficulty,
                retrieval=retrieval,
                generated_answer=answers[i] if i < len(answers) else "",
                retrieval_seconds=retrieval_seconds if i == 0 else 0.0,
                answer_seconds=per_q_answer_seconds,
                retrieval_token_usage=dict(retrieval_usage) if i == 0 else {},
                answer_token_usage=answer_usage if i == 0 else {},
            )
        )
    return tuple(out)


def run_checkpoint_qa_records(
    adapter: MemoryAdapter,
    *,
    sample_id: str,
    sample_uuid: str,
    checkpoint: NormalizedCheckpoint,
    top_k: int,
    answer_fn: AnswerFn,
    answer_evidence_mode: str = "memory",
    session_dialogues: dict[str, str] | None = None,
    memory_session_map: dict[str, str] | None = None,
    image_session_map: dict[str, str] | None = None,
) -> tuple[PipelineCheckpointQARecord, ...]:
    """For each question: retrieve top-K then call ``answer_fn`` for the final answer.

    For harness native memory adapters, batches all questions into a single
    harness call to avoid per-question agent startup overhead.
    """
    if _is_harness_native(adapter):
        return _run_checkpoint_qa_batched_harness(
            adapter,
            sample_id=sample_id,
            sample_uuid=sample_uuid,
            checkpoint=checkpoint,
            top_k=top_k,
            answer_evidence_mode=answer_evidence_mode,
            session_dialogues=session_dialogues,
            memory_session_map=memory_session_map,
            image_session_map=image_session_map,
        )

    out: list[PipelineCheckpointQARecord] = []
    for q in checkpoint.questions:
        retrieval_start = time.perf_counter()
        with token_usage_scope() as retrieval_usage:
            retrieval = adapter.retrieve(
                q.question, top_k, category=q.question_type_abbrev
            )
        retrieval = _with_answer_context(
            retrieval,
            answer_evidence_mode=answer_evidence_mode,
            session_dialogues=session_dialogues or {},
            memory_session_map=memory_session_map or {},
            image_session_map=image_session_map or {},
        )
        retrieval_seconds = time.perf_counter() - retrieval_start

        answer_start = time.perf_counter()
        with token_usage_scope() as scoped_answer_usage:
            generated_raw = answer_fn(q, retrieval)
        answer_seconds = time.perf_counter() - answer_start
        generated, returned_answer_usage = _normalize_answer_output(generated_raw)
        answer_usage = merge_token_usage(scoped_answer_usage, returned_answer_usage)

        out.append(
            PipelineCheckpointQARecord(
                sample_id=sample_id,
                sample_uuid=sample_uuid,
                checkpoint_id=checkpoint.checkpoint_id,
                question=q.question,
                gold_answer=q.gold_answer,
                gold_evidence_memory_ids=q.gold_evidence_memory_ids,
                gold_evidence_contents=q.gold_evidence_contents,
                question_type=q.question_type,
                question_type_abbrev=q.question_type_abbrev,
                difficulty=q.difficulty,
                retrieval=retrieval,
                generated_answer=generated,
                retrieval_seconds=retrieval_seconds,
                answer_seconds=answer_seconds,
                retrieval_token_usage=dict(retrieval_usage),
                answer_token_usage=answer_usage,
            )
        )
    return tuple(out)
