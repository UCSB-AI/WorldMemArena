"""Optional itemwise memory label evaluation (does not replace batch judges).

One LLM call per candidate delta memory → correct / hallucination / error
(using dialogue + gold as ground truth).

Recall is NOT computed here — use the existing batch recall instead.
"""

from __future__ import annotations

from eval_framework.judges import evaluate_candidate_label_itemwise
from eval_framework.pipeline.records import PipelineSessionRecord


def _gold_lines(session: PipelineSessionRecord) -> list[str]:
    lines: list[str] = []
    for g in session.gold_state.session_new_memories:
        lines.append(f"[normal] {g.memory_content}")
    for g in session.gold_state.session_update_memories:
        lines.append(f"[update] {g.memory_content}")
    return lines


def _golden_memories_str(golds: list[str]) -> str:
    if not golds:
        return "(none)"
    return "\n".join(f"[{i+1}] {g}" for i, g in enumerate(golds))


def _delta_candidates(session: PipelineSessionRecord) -> list[str]:
    return [d.text for d in session.memory_delta if d.op in ("add", "update")]


def evaluate_memory_accuracy_itemwise(session: PipelineSessionRecord) -> dict[str, object]:
    """Label pass only: per candidate → correct / hallucination / error."""
    dialogue_str = (session.dialogue_str or "").strip()
    golds = _gold_lines(session)
    num_gold = len(golds)
    candidates = _delta_candidates(session)
    num_cand = len(candidates)

    if not dialogue_str:
        return {
            "skipped": True,
            "reason": "Empty dialogue_str — re-run pipeline (not eval-only) after upgrading.",
            "num_candidates": num_cand,
            "num_gold": num_gold,
            "per_candidate": [],
        }

    gold_str = _golden_memories_str(golds)

    n_correct = 0
    n_hallu = 0
    n_error = 0
    label_rows: list[dict[str, object]] = []

    for i, cand in enumerate(candidates):
        res = evaluate_candidate_label_itemwise(dialogue_str, gold_str, cand)
        label = str(res.get("label", "error")).lower().strip()
        if label not in ("correct", "hallucination", "error"):
            label = "error"
        if label == "correct":
            n_correct += 1
        elif label == "hallucination":
            n_hallu += 1
        else:
            n_error += 1
        label_rows.append({
            "index": i,
            "candidate_memory": cand,
            "label": label,
            "reasoning": res.get("reasoning", ""),
        })

    return {
        "skipped": False,
        "num_candidates": num_cand,
        "num_gold": num_gold,
        "itemwise_correctness": n_correct / num_cand if num_cand else 0.0,
        "itemwise_hallucination": n_hallu / num_cand if num_cand else 0.0,
        "itemwise_error": n_error / num_cand if num_cand else 0.0,
        "num_correct": n_correct,
        "num_hallucination": n_hallu,
        "num_error": n_error,
        "per_candidate": label_rows,
    }
