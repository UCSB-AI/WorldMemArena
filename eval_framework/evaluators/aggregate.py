"""Roll up per-session and per-QA evaluations into baseline-level summaries.

- Recall & correctness: per-session average (not pooled cumulative).
- Update / interference: pooled across sessions.
- QA metrics: pooled overall **and** broken down by ``question_type_abbrev``.
- Retrieval vs. citation are kept as **two separate coverage metrics** so
  retrieval capability is not coupled with the answer model's conservatism.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from statistics import mean

from eval_framework.config import is_answer_only_eval_baseline
from eval_framework.openai_compat import empty_token_usage, merge_token_usage

_RANKING_KS = ("1", "5", "10")
_ANSWER_ONLY_SKIP_REASON = "answer_only_baseline"
_SKIPPED_METRIC_BLOCK: dict[str, object] = {
    "skipped": True,
    "reason": _ANSWER_ONLY_SKIP_REASON,
}


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _mean(xs: Sequence[float]) -> float:
    return mean(xs) if xs else 0.0


def _usage_from_mapping(obj: object, key: str) -> dict[str, int]:
    if not isinstance(obj, Mapping):
        return empty_token_usage()
    usage = obj.get(key)
    return merge_token_usage(usage)


def _qa_breakdown(
    qa_evaluations: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    """Compute QA metrics overall and per ``question_type_abbrev``.

    Returns ``(overall_qa_stats, per_type_qa_stats)`` where each stats dict
    carries:
      - num_total / num_valid
      - correct_ratio / hallucination_ratio / omission_ratio
      - retrieval_coverage: hit_rate + covered/total counts
      - notmention_when_retrieved_ratio: diagnostic for answer-prompt conservatism
    """
    # Per-type buckets
    buckets: dict[str | None, dict[str, list]] = defaultdict(
        lambda: {
            "labels": [],
            "retrieval_hits": [],
            "ranking_recall_at": {k: [] for k in _RANKING_KS},
            "ranking_ndcg_at": {k: [] for k in _RANKING_KS},
            "answer_f1": [],
            "answer_bleu1": [],
            "notmention_when_retrieved": [],  # 1 if answer=Omission AND retrieval_hit>0
            "retrieval_covered": 0,
            "num_evidence": 0,
        }
    )
    overall_bucket = {
        "labels": [],
        "retrieval_hits": [],
        "ranking_recall_at": {k: [] for k in _RANKING_KS},
        "ranking_ndcg_at": {k: [] for k in _RANKING_KS},
        "answer_f1": [],
        "answer_bleu1": [],
        "notmention_when_retrieved": [],
        "retrieval_covered": 0,
        "num_evidence": 0,
    }

    for q in qa_evaluations:
        qtype = q.get("question_type_abbrev") or "UNKNOWN"
        label = q.get("answer_label")
        ret_hit = float(q.get("retrieval_hit_rate", 0.0) or 0.0)
        recall_at = q.get("retrieval_recall_at")
        ndcg_at = q.get("retrieval_ndcg_at")
        answer_f1 = float(q.get("answer_f1", 0.0) or 0.0)
        answer_bleu1 = float(q.get("answer_bleu1", 0.0) or 0.0)
        notmention_miss = 1 if (label == "Omission" and ret_hit > 0) else 0
        ret_covered = int(q.get("retrieval_covered_count") or 0)
        num_ev = int(q.get("num_evidence") or 0)

        for bucket in (buckets[qtype], overall_bucket):
            bucket["labels"].append(label)
            bucket["retrieval_hits"].append(ret_hit)
            if isinstance(recall_at, Mapping):
                for k in _RANKING_KS:
                    bucket["ranking_recall_at"][k].append(float(recall_at.get(k, 0.0) or 0.0))
            if isinstance(ndcg_at, Mapping):
                for k in _RANKING_KS:
                    bucket["ranking_ndcg_at"][k].append(float(ndcg_at.get(k, 0.0) or 0.0))
            bucket["answer_f1"].append(answer_f1)
            bucket["answer_bleu1"].append(answer_bleu1)
            bucket["notmention_when_retrieved"].append(notmention_miss)
            bucket["retrieval_covered"] += ret_covered
            bucket["num_evidence"] += num_ev

    def summarize(bucket: dict[str, list | int]) -> dict[str, object]:
        labels = bucket["labels"]
        total = len(labels)
        valid_labels = [l for l in labels if l in ("Correct", "Hallucination", "Omission")]
        n_valid = len(valid_labels)
        n_correct = sum(1 for l in valid_labels if l == "Correct")
        n_hallu = sum(1 for l in valid_labels if l == "Hallucination")
        n_omit = sum(1 for l in valid_labels if l == "Omission")
        return {
            "num_total": total,
            "num_valid": n_valid,
            "correct_ratio": _safe_div(n_correct, n_valid),
            "hallucination_ratio": _safe_div(n_hallu, n_valid),
            "omission_ratio": _safe_div(n_omit, n_valid),
            "retrieval_coverage": {
                "hit_rate": _mean(bucket["retrieval_hits"]),
                "num_covered": int(bucket["retrieval_covered"]),
                "num_total": int(bucket["num_evidence"]),
            },
            "retrieval_ranking": {
                "recall_at": {
                    k: _mean(bucket["ranking_recall_at"][k])
                    for k in _RANKING_KS
                },
                "ndcg_at": {
                    k: _mean(bucket["ranking_ndcg_at"][k])
                    for k in _RANKING_KS
                },
            },
            "answer_matching": {
                "avg_f1": _mean(bucket["answer_f1"]),
                "avg_bleu1": _mean(bucket["answer_bleu1"]),
            },
            # Diagnostic: fraction of Omissions that happened even though
            # retrieval surfaced at least one gold evidence point.  High ⇒
            # answer prompt is too conservative / model refuses to synthesize.
            "notmention_when_retrieved_ratio": _mean(bucket["notmention_when_retrieved"]),
        }

    per_type = {
        qt: summarize(buckets[qt])
        for qt in sorted(buckets.keys(), key=lambda x: (x is None, x or ""))
    }
    return summarize(overall_bucket), per_type


def _qa_breakdown_answer_only(
    qa_evaluations: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    """QA metrics for answer-only baselines (no retrieval coverage / ranking)."""
    buckets: dict[str | None, dict[str, list]] = defaultdict(
        lambda: {"labels": [], "answer_f1": [], "answer_bleu1": []}
    )
    overall_bucket: dict[str, list] = {
        "labels": [],
        "answer_f1": [],
        "answer_bleu1": [],
    }

    for q in qa_evaluations:
        qtype = q.get("question_type_abbrev") or "UNKNOWN"
        label = q.get("answer_label")
        answer_f1 = float(q.get("answer_f1", 0.0) or 0.0)
        answer_bleu1 = float(q.get("answer_bleu1", 0.0) or 0.0)
        for bucket in (buckets[qtype], overall_bucket):
            bucket["labels"].append(label)
            bucket["answer_f1"].append(answer_f1)
            bucket["answer_bleu1"].append(answer_bleu1)

    def summarize(bucket: dict[str, list]) -> dict[str, object]:
        labels = bucket["labels"]
        total = len(labels)
        valid_labels = [l for l in labels if l in ("Correct", "Hallucination", "Omission")]
        n_valid = len(valid_labels)
        n_correct = sum(1 for l in valid_labels if l == "Correct")
        n_hallu = sum(1 for l in valid_labels if l == "Hallucination")
        n_omit = sum(1 for l in valid_labels if l == "Omission")
        return {
            "num_total": total,
            "num_valid": n_valid,
            "correct_ratio": _safe_div(n_correct, n_valid),
            "hallucination_ratio": _safe_div(n_hallu, n_valid),
            "omission_ratio": _safe_div(n_omit, n_valid),
            "retrieval_coverage": _SKIPPED_METRIC_BLOCK,
            "retrieval_ranking": _SKIPPED_METRIC_BLOCK,
            "answer_matching": {
                "avg_f1": _mean(bucket["answer_f1"]),
                "avg_bleu1": _mean(bucket["answer_bleu1"]),
            },
            "notmention_when_retrieved_ratio": _SKIPPED_METRIC_BLOCK,
        }

    per_type = {
        qt: summarize(buckets[qt])
        for qt in sorted(buckets.keys(), key=lambda x: (x is None, x or ""))
    }
    return summarize(overall_bucket), per_type


def aggregate_memory_accuracy_itemwise(
    session_evaluations: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    """Pool itemwise correct/hallucination/error across sessions (label pass only)."""
    correctness_scores: list[float] = []
    hallucination_scores: list[float] = []
    error_scores: list[float] = []
    total_candidates = 0
    total_correct = 0
    total_hallu = 0
    total_error = 0
    n_sessions = 0

    for s in session_evaluations:
        block = s.get("memory_accuracy_itemwise")
        if not isinstance(block, dict):
            continue
        if block.get("skipped"):
            continue
        n_cand = int(block.get("num_candidates", 0))
        if n_cand == 0:
            continue
        n_sessions += 1
        total_candidates += n_cand

        nc = int(block.get("num_correct", 0))
        nh = int(block.get("num_hallucination", 0))
        ne = int(block.get("num_error", 0))
        total_correct += nc
        total_hallu += nh
        total_error += ne

        correctness_scores.append(nc / n_cand)
        hallucination_scores.append(nh / n_cand)
        error_scores.append(ne / n_cand)

    if not correctness_scores:
        return None
    return {
        "num_sessions_with_itemwise": n_sessions,
        "total_candidates": total_candidates,
        "avg_correctness": _safe_div(sum(correctness_scores), len(correctness_scores)),
        "avg_hallucination": _safe_div(sum(hallucination_scores), len(hallucination_scores)),
        "avg_error": _safe_div(sum(error_scores), len(error_scores)),
        "total_correct": total_correct,
        "total_hallucination": total_hallu,
        "total_error": total_error,
    }


def aggregate_metrics(
    baseline_id: str,
    *,
    session_evaluations: Sequence[Mapping[str, object]] = (),
    qa_evaluations: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    """Aggregate all per-session and per-QA evaluations."""
    if is_answer_only_eval_baseline(baseline_id):
        return _aggregate_metrics_answer_only(
            baseline_id,
            session_evaluations=session_evaluations,
            qa_evaluations=qa_evaluations,
        )

    # --- Per-session recall / correctness ---
    recall_scores: list[float] = []
    weighted_recall_scores: list[float] = []
    correctness_scores: list[float] = []
    hallucination_scores: list[float] = []
    error_scores: list[float] = []

    # --- Update handling (pooled) ---
    upd_num_updated = 0
    upd_num_both = 0
    upd_num_outdated = 0
    upd_total_items = 0

    # --- Interference rejection (pooled) ---
    interf_num_rejected = 0
    interf_num_memorized = 0
    interf_total_items = 0

    # --- Per-session detail counters ---
    total_gold_points = 0
    total_covered = 0
    total_weighted_covered = 0.0
    total_importance = 0.0
    total_memories = 0
    total_correct = 0
    total_hallucination = 0
    total_error = 0

    for s in session_evaluations:
        r = s.get("recall")
        if r is not None:
            recall_scores.append(float(r))
        wr = s.get("weighted_recall")
        if wr is not None:
            weighted_recall_scores.append(float(wr))

        cr = s.get("correctness_rate")
        if cr is not None:
            correctness_scores.append(float(cr))

        nm = int(s.get("num_memories", 0))
        if nm > 0:
            hallucination_scores.append(float(s.get("num_hallucination", 0)) / nm)
            error_scores.append(float(s.get("num_error", 0)) / nm)

        c = s.get("covered_count")
        if c is not None:
            total_covered += int(c)
        total_gold_points += int(s.get("num_gold", 0))
        total_weighted_covered += float(s.get("weighted_covered_importance", 0.0) or 0.0)
        total_importance += float(s.get("total_importance", 0.0) or 0.0)
        total_memories += nm
        total_correct += int(s.get("num_correct", 0))
        total_hallucination += int(s.get("num_hallucination", 0))
        total_error += int(s.get("num_error", 0))

        upd_num_updated += int(s.get("update_num_updated", 0))
        upd_num_both += int(s.get("update_num_both", 0))
        upd_num_outdated += int(s.get("update_num_outdated", 0))
        upd_total_items += int(s.get("update_total_items", 0))

        interf_num_rejected += int(s.get("interference_num_rejected", 0))
        interf_num_memorized += int(s.get("interference_num_memorized", 0))
        interf_total_items += int(s.get("interference_total_items", 0))

    n_recall = len(recall_scores)
    n_weighted_recall = len(weighted_recall_scores)
    n_correct = len(correctness_scores)
    n_hallu = len(hallucination_scores)
    n_error = len(error_scores)

    # --- QA aggregation: overall + per question_type_abbrev ---
    qa_overall, qa_by_type = _qa_breakdown(qa_evaluations)

    itemwise_block = aggregate_memory_accuracy_itemwise(session_evaluations)

    total_storage_seconds = sum(float(s.get("storage_seconds", 0.0) or 0.0) for s in session_evaluations)
    total_retrieval_seconds = sum(float(q.get("retrieval_seconds", 0.0) or 0.0) for q in qa_evaluations)
    total_answer_seconds = sum(float(q.get("answer_seconds", 0.0) or 0.0) for q in qa_evaluations)

    storage_usage = empty_token_usage()
    session_judge_usage = empty_token_usage()
    for s in session_evaluations:
        storage_usage = merge_token_usage(storage_usage, _usage_from_mapping(s, "storage_token_usage"))
        session_judge_usage = merge_token_usage(session_judge_usage, _usage_from_mapping(s, "judge_token_usage"))

    retrieval_usage = empty_token_usage()
    answer_usage = empty_token_usage()
    qa_judge_usage = empty_token_usage()
    for q in qa_evaluations:
        retrieval_usage = merge_token_usage(retrieval_usage, _usage_from_mapping(q, "retrieval_token_usage"))
        answer_usage = merge_token_usage(answer_usage, _usage_from_mapping(q, "answer_token_usage"))
        qa_judge_usage = merge_token_usage(qa_judge_usage, _usage_from_mapping(q, "judge_token_usage"))
    judge_usage = merge_token_usage(session_judge_usage, qa_judge_usage)
    total_usage = merge_token_usage(storage_usage, retrieval_usage, answer_usage, judge_usage)

    out: dict[str, object] = {
        "baseline_id": baseline_id,
        "memory_recall": {
            "avg_recall": _safe_div(sum(recall_scores), n_recall),
            "avg_weighted_recall": _safe_div(sum(weighted_recall_scores), n_weighted_recall),
            "num_sessions_with_recall": n_recall,
            "num_sessions_with_weighted_recall": n_weighted_recall,
            "total_covered": total_covered,
            "total_gold": total_gold_points,
            "total_weighted_covered": total_weighted_covered,
            "total_importance": total_importance,
            "pooled_weighted_recall": _safe_div(total_weighted_covered, total_importance),
        },
        "memory_correctness": {
            "avg_correctness": _safe_div(sum(correctness_scores), n_correct),
            "avg_hallucination": _safe_div(sum(hallucination_scores), n_hallu),
            "avg_error": _safe_div(sum(error_scores), n_error),
            # V11-compatible aliases; current staged code names this bucket "error".
            "avg_irrelevant": _safe_div(sum(error_scores), n_error),
            "num_sessions": n_correct,
            "total_memories": total_memories,
            "total_correct": total_correct,
            "total_hallucination": total_hallucination,
            "total_error": total_error,
            "total_irrelevant": total_error,
        },
        "update_handling": {
            "score": _safe_div(upd_num_updated * 1.0 + upd_num_both * 0.5, upd_total_items),
            "num_updated": upd_num_updated,
            "num_both": upd_num_both,
            "num_outdated": upd_num_outdated,
            "num_total": upd_total_items,
        },
        "interference_rejection": {
            "score": _safe_div(interf_num_rejected, interf_total_items),
            "num_rejected": interf_num_rejected,
            "num_memorized": interf_num_memorized,
            "num_total": interf_total_items,
        },
        "question_answering": qa_overall,
        "question_answering_by_type": qa_by_type,
        "runtime": {
            "storage_seconds": total_storage_seconds,
            "retrieval_seconds": total_retrieval_seconds,
            "answer_seconds": total_answer_seconds,
            "avg_storage_seconds_per_session": _safe_div(total_storage_seconds, len(session_evaluations)),
            "avg_retrieval_seconds_per_qa": _safe_div(total_retrieval_seconds, len(qa_evaluations)),
            "avg_answer_seconds_per_qa": _safe_div(total_answer_seconds, len(qa_evaluations)),
        },
        "token_usage": {
            "storage": storage_usage,
            "retrieval": retrieval_usage,
            "answer": answer_usage,
            "judge": judge_usage,
            "total": total_usage,
        },
    }
    if itemwise_block is not None:
        out["memory_accuracy_itemwise"] = itemwise_block
    return out


def _aggregate_metrics_answer_only(
    baseline_id: str,
    *,
    session_evaluations: Sequence[Mapping[str, object]] = (),
    qa_evaluations: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    """Aggregate checkpoint answer metrics only (BaseModel / Harness)."""
    qa_overall, qa_by_type = _qa_breakdown_answer_only(qa_evaluations)

    total_storage_seconds = sum(
        float(s.get("storage_seconds", 0.0) or 0.0) for s in session_evaluations
    )
    total_retrieval_seconds = sum(
        float(q.get("retrieval_seconds", 0.0) or 0.0) for q in qa_evaluations
    )
    total_answer_seconds = sum(
        float(q.get("answer_seconds", 0.0) or 0.0) for q in qa_evaluations
    )

    storage_usage = empty_token_usage()
    for s in session_evaluations:
        storage_usage = merge_token_usage(
            storage_usage, _usage_from_mapping(s, "storage_token_usage")
        )

    retrieval_usage = empty_token_usage()
    answer_usage = empty_token_usage()
    qa_judge_usage = empty_token_usage()
    for q in qa_evaluations:
        retrieval_usage = merge_token_usage(
            retrieval_usage, _usage_from_mapping(q, "retrieval_token_usage")
        )
        answer_usage = merge_token_usage(
            answer_usage, _usage_from_mapping(q, "answer_token_usage")
        )
        qa_judge_usage = merge_token_usage(
            qa_judge_usage, _usage_from_mapping(q, "judge_token_usage")
        )
    total_usage = merge_token_usage(storage_usage, retrieval_usage, answer_usage, qa_judge_usage)

    return {
        "baseline_id": baseline_id,
        "eval_mode": "answer_only",
        "memory_recall": dict(_SKIPPED_METRIC_BLOCK),
        "memory_correctness": dict(_SKIPPED_METRIC_BLOCK),
        "update_handling": dict(_SKIPPED_METRIC_BLOCK),
        "interference_rejection": dict(_SKIPPED_METRIC_BLOCK),
        "memory_accuracy_itemwise": dict(_SKIPPED_METRIC_BLOCK),
        "question_answering": qa_overall,
        "question_answering_by_type": qa_by_type,
        "runtime": {
            "storage_seconds": total_storage_seconds,
            "retrieval_seconds": total_retrieval_seconds,
            "answer_seconds": total_answer_seconds,
            "avg_storage_seconds_per_session": _safe_div(
                total_storage_seconds, len(session_evaluations)
            ),
            "avg_retrieval_seconds_per_qa": _safe_div(
                total_retrieval_seconds, len(qa_evaluations)
            ),
            "avg_answer_seconds_per_qa": _safe_div(
                total_answer_seconds, len(qa_evaluations)
            ),
        },
        "token_usage": {
            "storage": storage_usage,
            "retrieval": retrieval_usage,
            "answer": answer_usage,
            "judge": qa_judge_usage,
            "total": total_usage,
        },
    }
