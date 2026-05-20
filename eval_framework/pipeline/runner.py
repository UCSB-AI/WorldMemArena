"""Session-by-session ingest, memory export, and checkpoint QA orchestration."""

from __future__ import annotations

import time
from collections.abc import Callable

from eval_framework.datasets.wma_bundle import (
    EvalSample,
    NormalizedCheckpointQuestion,
)
from eval_framework.memory_adapters.base import MemoryAdapter
from eval_framework.pipeline.dialogue_format import format_session_dialogue
from eval_framework.pipeline.qa_runner import run_checkpoint_qa_records
from eval_framework.pipeline.records import PipelineCheckpointQARecord, PipelineSessionRecord
from eval_framework.datasets.schemas import RetrievalRecord
from eval_framework.openai_compat import token_usage_scope


def ensure_adapter_available(adapter: MemoryAdapter) -> None:
    caps = adapter.get_capabilities()
    if caps.get("available") is False:
        backend = caps.get("backend", type(adapter).__name__)
        detail = caps.get("integration_error") or caps.get(
            "integration_status", "available=False"
        )
        raise RuntimeError(
            f"Memory adapter {backend!r} is not available for pipeline runs: {detail}"
        )


def run_eval_sample(
    adapter: MemoryAdapter,
    sample: EvalSample,
    *,
    top_k: int | None = None,
    answer_fn: Callable | None = None,
    max_sessions: int | None = None,
    answer_evidence_mode: str = "memory",
) -> tuple[tuple[PipelineSessionRecord, ...], tuple[PipelineCheckpointQARecord, ...]]:
    if top_k is None:
        from eval_framework.config import resolve_retrieval_top_k
        top_k = resolve_retrieval_top_k()
    """Run all sessions in order, emit one session record per session, then checkpoint QA when due.

    ``max_sessions``, if set, truncates to the first N sessions (and uses only
    checkpoints whose ``covered_sessions`` are all inside that window).
    """
    ensure_adapter_available(adapter)
    if sample.normalized_checkpoints and answer_fn is None:
        raise ValueError(
            "answer_fn is required when the sample defines normalized checkpoints"
        )

    adapter.reset()
    session_out: list[PipelineSessionRecord] = []
    qa_out: list[PipelineCheckpointQARecord] = []
    completed_sessions: set[str] = set()

    sessions = sample.sessions
    gold_states = sample.session_gold_states
    if max_sessions is not None and max_sessions > 0:
        sessions = sessions[:max_sessions]
        gold_states = gold_states[:max_sessions]

    session_order = {
        session.session_id: index for index, session in enumerate(sessions)
    }
    allowed_ids = set(session_order.keys())
    image_session_map: dict[str, str] = {}
    for session in sessions:
        for turn in session.turns:
            for att in turn.attachments:
                if att.image_id:
                    image_session_map[att.image_id] = session.session_id
                if att.file_path:
                    image_session_map[att.file_path] = session.session_id

    if len(sessions) != len(gold_states):
        raise ValueError(
            "sample.sessions and sample.session_gold_states length mismatch"
        )

    total_sessions = len(sessions)
    for sess_idx, (sess, gold) in enumerate(zip(sessions, gold_states)):
        if sess.session_id != gold.session_id:
            raise ValueError(
                f"session / gold_state id mismatch: {sess.session_id!r} vs {gold.session_id!r}"
            )
        num_turns = len(sess.turns)
        storage_start = time.perf_counter()
        with token_usage_scope() as storage_usage:
            for turn_idx, turn in enumerate(sess.turns):
                adapter.ingest_turn(turn)
                if (turn_idx + 1) % 10 == 0 or turn_idx == num_turns - 1:
                    print(
                        f"    [{sess.session_id}] ingest {turn_idx + 1}/{num_turns} turns",
                        flush=True,
                    )
            adapter.end_session(sess.session_id)
        storage_seconds = time.perf_counter() - storage_start
        print(
            f"    Session {sess_idx + 1}/{total_sessions} ({sess.session_id}) done",
            flush=True,
        )

        snapshot = tuple(adapter.snapshot_memories())
        delta = tuple(adapter.export_memory_delta(sess.session_id))
        session_out.append(
            PipelineSessionRecord(
                sample_id=sample.sample_id,
                sample_uuid=sample.uuid,
                session_id=sess.session_id,
                memory_snapshot=snapshot,
                memory_delta=delta,
                gold_state=gold,
                dialogue_str=format_session_dialogue(sess.turns),
                storage_seconds=storage_seconds,
                storage_token_usage=dict(storage_usage),
            )
        )
        completed_sessions.add(sess.session_id)

        for cp in sample.normalized_checkpoints:
            covered = cp.covered_sessions
            if not covered:
                continue
            # When max_sessions truncates the run, skip checkpoints whose
            # covered sessions fall outside the window.
            if not set(covered).issubset(allowed_ids):
                continue
            if not set(covered).issubset(completed_sessions):
                continue
            trigger_session_id = max(covered, key=session_order.__getitem__)
            if sess.session_id != trigger_session_id:
                continue
            session_dialogues = {
                rec.session_id: rec.dialogue_str for rec in session_out
            }
            memory_session_map: dict[str, str] = {}
            for mem in snapshot:
                for key in (mem.memory_id, mem.raw_backend_id):
                    if key:
                        memory_session_map[key] = mem.session_id
            qa_out.extend(
                run_checkpoint_qa_records(
                    adapter,
                    sample_id=sample.sample_id,
                    sample_uuid=sample.uuid,
                    checkpoint=cp,
                    top_k=top_k,
                    answer_fn=answer_fn,
                    answer_evidence_mode=answer_evidence_mode,
                    session_dialogues=session_dialogues,
                    memory_session_map=memory_session_map,
                    image_session_map=image_session_map,
                )
            )

    return tuple(session_out), tuple(qa_out)
