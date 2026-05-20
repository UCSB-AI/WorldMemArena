"""Unified session evaluation: recall + correctness (includes update & interference).

Per session:
  Recall: how many of this session's gold points are covered by the session's
          memory delta (add/update ops)?
  Correctness: is each delta memory correct, hallucinated, or erroneous?
               (reference = this session's [normal] + [update] gold points only;
                interference is handled by the dedicated rejection metric)
  Update handling: per-update gold point, check the FULL memory snapshot
                   after this session — did the old fact get replaced?
  Interference rejection: per-interference gold point, check the FULL memory
                          snapshot after this session — was it not stored?

Aggregate: per-session recall/correctness averaged across sessions.
"""

from __future__ import annotations

from eval_framework.judges import (
    evaluate_correctness_batch,
    evaluate_interference_single,
    evaluate_recall_batch,
    evaluate_update_single,
)
from eval_framework.openai_compat import merge_token_usage
from eval_framework.pipeline.records import PipelineSessionRecord


def _delta_to_text(session: PipelineSessionRecord) -> str:
    """Only the memories added or updated in THIS session (not the full snapshot)."""
    lines: list[str] = []
    idx = 0
    for d in session.memory_delta:
        if d.op in ("add", "update"):
            idx += 1
            lines.append(f"[{idx}] {d.text}")
    return "\n".join(lines)


def _delta_texts(session: PipelineSessionRecord) -> list[str]:
    """Text list of memories added or updated in THIS session."""
    return [d.text for d in session.memory_delta if d.op in ("add", "update")]


def _snapshot_to_text(session: PipelineSessionRecord) -> str:
    """Full system memory state after THIS session (for interference judging)."""
    lines: list[str] = []
    for idx, s in enumerate(session.memory_snapshot, start=1):
        lines.append(f"[{idx}] {s.text}")
    return "\n".join(lines)


def _build_recall_gold_points(session: PipelineSessionRecord) -> list[str]:
    """Current session's new + update gold points only (NOT cumulative)."""
    out: list[str] = []
    for g in session.gold_state.session_new_memories:
        out.append(f"[normal] {g.memory_content}")
    for g in session.gold_state.session_update_memories:
        out.append(f"[update] {g.memory_content}")
    return out


def _build_recall_gold_entries(session: PipelineSessionRecord) -> list[object]:
    return [
        *list(session.gold_state.session_new_memories),
        *list(session.gold_state.session_update_memories),
    ]


def _build_correctness_gold_points(session: PipelineSessionRecord) -> list[str]:
    """Current session's new + update gold points as reference.

    Interference is intentionally excluded — it's judged separately by the
    dedicated interference rejection metric.
    """
    out: list[str] = []
    for g in session.gold_state.session_new_memories:
        out.append(f"[normal] {g.memory_content}")
    for g in session.gold_state.session_update_memories:
        out.append(f"[update] {g.memory_content}")
    return out


def evaluate_extraction(
    session: PipelineSessionRecord,
    **_kwargs: object,
) -> dict[str, object]:
    """Unified session evaluation: recall + correctness in 2 LLM calls.

    Uses only THIS session's new gold points for recall and correctness,
    not the cumulative history. Aggregate averages per-session scores.
    """

    delta_str = _delta_to_text(session)
    delta_texts = _delta_texts(session)

    # --- Call 1: Recall (this session's gold points vs this session's delta) ---
    recall_gold_entries = _build_recall_gold_entries(session)
    recall_gold = _build_recall_gold_points(session)

    if not recall_gold:
        recall = None
        recall_result: dict[str, object] = {
            "covered_count": 0, "total": 0,
            "covered_indices": [],
            "reasoning": "No new gold points in this session.",
        }
    elif not delta_str.strip():
        recall = 0.0
        recall_result = {
            "covered_count": 0, "total": len(recall_gold),
            "covered_indices": [],
            "reasoning": "No add/update memories in this session's delta.",
        }
    else:
        recall_result = evaluate_recall_batch(delta_str, recall_gold)

    covered = recall_result.get("covered_count")
    total_gold = recall_result.get("total", len(recall_gold))
    covered_indices_raw = recall_result.get("covered_indices") or []
    covered_indices: list[int] = []
    if isinstance(covered_indices_raw, list):
        for item in covered_indices_raw:
            try:
                idx = int(item)
            except (TypeError, ValueError):
                continue
            if 1 <= idx <= len(recall_gold_entries) and idx not in covered_indices:
                covered_indices.append(idx)

    if recall_gold:
        recall = float(covered) / float(total_gold) if covered is not None and total_gold else None
    weights = [
        float(getattr(g, "importance", 0.0) or 0.0)
        for g in recall_gold_entries
    ]
    if recall_gold_entries and not any(w > 0 for w in weights):
        weights = [1.0 for _ in recall_gold_entries]
    total_importance = sum(weights)
    weighted_covered_importance = sum(
        weights[idx - 1] for idx in covered_indices if 1 <= idx <= len(weights)
    )
    weighted_recall = (
        weighted_covered_importance / total_importance
        if total_importance > 0
        else None
    )
    if (
        recall is not None
        and total_importance > 0
        and not covered_indices
        and isinstance(covered, int)
        and covered > 0
    ):
        # Backward-compatible fallback for judge outputs that only return a
        # count. Exact weighting needs covered_indices, so use unweighted recall.
        weighted_recall = recall
        weighted_covered_importance = recall * total_importance

    # --- Call 2: Correctness (delta memories vs. gold + dialogue) ---
    correctness_gold = _build_correctness_gold_points(session)
    dialogue_str = getattr(session, "dialogue_str", "") or ""
    correctness_result = evaluate_correctness_batch(
        delta_texts, correctness_gold, dialogue_str=dialogue_str,
    )
    correctness_records = correctness_result.get("results", [])

    num_correct = sum(1 for r in correctness_records if r.get("label") == "correct")
    num_hallucination = sum(1 for r in correctness_records if r.get("label") == "hallucination")
    num_error = sum(1 for r in correctness_records if r.get("label") == "error")
    num_memories = len(delta_texts)
    correctness_rate = float(num_correct) / float(num_memories) if num_memories else 0.0
    judge_token_usage = merge_token_usage(
        recall_result.get("token_usage"),
        correctness_result.get("token_usage"),
    )

    # --- Call 3+: Update handling (one LLM call per update gold point) ---
    # Judged against the FULL snapshot (what the system currently stores),
    # not just this session's delta — same as interference rejection.
    # "Did the old get replaced by the new?" is a state question about the
    # final memory, not an event question about this turn's delta.
    snapshot_str = _snapshot_to_text(session)
    update_records: list[dict[str, object]] = []
    for g in session.gold_state.session_update_memories:
        res = evaluate_update_single(
            snapshot_str,
            new_content=g.memory_content,
            old_contents=list(g.original_memories),
        )
        judge_token_usage = merge_token_usage(judge_token_usage, res.get("token_usage"))
        update_records.append({
            "memory_id": g.memory_id,
            "label": res["label"],
            "reasoning": res["reasoning"],
        })

    num_updated = sum(1 for r in update_records if r["label"] == "updated")
    num_both = sum(1 for r in update_records if r["label"] == "both")
    num_outdated = sum(1 for r in update_records if r["label"] == "outdated")
    update_total_items = len(update_records)
    # Score: updated=1.0, both=0.5, outdated=0.0
    update_score = (
        (num_updated * 1.0 + num_both * 0.5) / update_total_items
        if update_total_items else None
    )

    # --- Call 4+: Interference rejection (one LLM call per interference gold point) ---
    # Also judged against the full snapshot (reuse snapshot_str from above).
    interference_records: list[dict[str, object]] = []
    for g in session.gold_state.session_interference_memories:
        res = evaluate_interference_single(
            snapshot_str,
            interference_content=g.memory_content,
        )
        judge_token_usage = merge_token_usage(judge_token_usage, res.get("token_usage"))
        interference_records.append({
            "memory_id": g.memory_id,
            "label": res["label"],
            "reasoning": res["reasoning"],
        })

    num_rejected = sum(1 for r in interference_records if r["label"] == "rejected")
    num_memorized = sum(1 for r in interference_records if r["label"] == "memorized")
    interference_total_items = len(interference_records)
    # Score: rejected=1.0, memorized=0.0
    interference_score = (
        float(num_rejected) / interference_total_items
        if interference_total_items else None
    )

    return {
        "session_id": session.session_id,
        "recall": recall,
        "weighted_recall": weighted_recall,
        "covered_count": covered,
        "num_gold": total_gold,
        "recall_covered_indices": covered_indices,
        "weighted_covered_importance": weighted_covered_importance,
        "total_importance": total_importance,
        "recall_reasoning": recall_result.get("reasoning", ""),
        "correctness_rate": correctness_rate,
        "num_memories": num_memories,
        "num_correct": num_correct,
        "num_hallucination": num_hallucination,
        "num_error": num_error,
        "correctness_reasoning": correctness_result.get("reasoning", ""),
        "correctness_records": correctness_records,
        # Update handling
        "update_score": update_score,
        "update_num_updated": num_updated,
        "update_num_both": num_both,
        "update_num_outdated": num_outdated,
        "update_total_items": update_total_items,
        "update_records": update_records,
        # Interference rejection
        "interference_score": interference_score,
        "interference_num_rejected": num_rejected,
        "interference_num_memorized": num_memorized,
        "interference_total_items": interference_total_items,
        "interference_records": interference_records,
        # Runtime / token accounting
        "storage_seconds": float(getattr(session, "storage_seconds", 0.0) or 0.0),
        "storage_token_usage": dict(getattr(session, "storage_token_usage", {}) or {}),
        "judge_token_usage": judge_token_usage,
    }
