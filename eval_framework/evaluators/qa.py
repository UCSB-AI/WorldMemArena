"""Checkpoint QA evaluation: answer quality + retrieval coverage.

Two independent dimensions:

1. **Answer quality** (Correct / Hallucination / Omission) — 1 LLM call.
   Input: ``generated_answer`` vs ``gold_answer`` + ``gold_evidence``.

2. **Retrieval coverage** — 1 LLM call.
   Input: the full ``retrieval.items[:top_k]`` vs ``gold_evidence_contents``.
   Measures whether the backend's retriever *surfaced* the evidence,
   independent of how the answering model used it.

No citation metric — the model's self-reported citations are unreliable
and coupled with the answer prompt's conservatism.  We only judge what
the retriever actually returned.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from collections.abc import Mapping, Sequence

import regex
from nltk.stem import PorterStemmer
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from eval_framework.judges import evaluate_evidence_batch, evaluate_qa_llm
from eval_framework.openai_compat import merge_token_usage
from eval_framework.pipeline.records import PipelineCheckpointQARecord


_TOKEN_RE = re.compile(r"\w+")
_NDCG_KS = (1, 5, 10)
_ANSWER_STEMMER = PorterStemmer()
_BLEU1_WEIGHTS = (1.0, 0.0, 0.0, 0.0)
_BLEU_SMOOTHER = SmoothingFunction().method1


def _normalize_answer_universal(text: str) -> str:
    """Mem-Gallery answer normalization for lexical answer metrics."""

    normalized = str(text or "").lower()

    dot_placeholder = "DOTPLACEHOLDER"
    underscore_placeholder = "UNDERSCOREPLACEHOLDER"
    normalized = regex.sub(r"(?<=\d)\.(?=\d)", dot_placeholder, normalized)
    normalized = normalized.replace("_", underscore_placeholder)
    normalized = regex.sub(r"\b(a|an|the|and)\b", " ", normalized)

    punctuation = set(string.punctuation)
    normalized = "".join(
        ch if ch not in punctuation else " "
        for ch in normalized
    )
    normalized = normalized.replace(dot_placeholder, ".")
    normalized = normalized.replace(underscore_placeholder, "_")
    return " ".join(normalized.split())


def _answer_tokens(text: str, *, stem: bool = False) -> list[str]:
    tokens = _normalize_answer_universal(text).split()
    if stem:
        return [_ANSWER_STEMMER.stem(token) for token in tokens]
    return tokens


def _answer_f1(prediction: str, reference: str) -> float:
    pred = _answer_tokens(prediction, stem=True)
    ref = _answer_tokens(reference, stem=True)
    common = Counter(pred) & Counter(ref)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def _answer_bleu1(prediction: str, reference: str) -> float:
    pred = _answer_tokens(prediction)
    ref = _answer_tokens(reference)
    if not pred or not ref:
        return 0.0
    return float(
        sentence_bleu(
            [ref],
            pred,
            weights=_BLEU1_WEIGHTS,
            smoothing_function=_BLEU_SMOOTHER,
        )
    )


def _build_retrieval_text(record: PipelineCheckpointQARecord) -> str:
    lines = [f"[{item.rank}] {item.text}" for item in record.retrieval.items]
    return "\n".join(lines)


def _norm(text: str) -> str:
    return " ".join(_TOKEN_RE.findall((text or "").lower()))


def _gold_session_id(gold_id: str) -> str | None:
    m = re.search(r"\bmp_(S\d+)_", gold_id or "")
    return m.group(1) if m else None


def _source_sessions_for_rank(
    raw_trace: Mapping[str, object],
    rank: int,
) -> list[str]:
    mapping = raw_trace.get("retrieval_source_sessions_by_rank")
    if not isinstance(mapping, Mapping):
        return []
    raw = mapping.get(str(rank))
    if not isinstance(raw, list):
        return []
    return [str(x) for x in raw if x is not None]


def _item_matches_gold(
    *,
    item: object,
    raw_trace: Mapping[str, object],
    gold_id: str,
    gold_content: str,
) -> bool:
    memory_id = str(getattr(item, "memory_id", "") or "")
    raw_backend_id = str(getattr(item, "raw_backend_id", "") or "")
    image_path = str(getattr(item, "image_path", "") or "")
    text = str(getattr(item, "text", "") or "")
    rank = int(getattr(item, "rank", 0) or 0)

    ids = {memory_id, raw_backend_id}
    if gold_id and gold_id in ids:
        return True

    haystack = "\n".join([memory_id, raw_backend_id, image_path, text]).lower()
    if gold_id and gold_id.lower() in haystack:
        return True

    sid = _gold_session_id(gold_id)
    if sid and sid in _source_sessions_for_rank(raw_trace, rank):
        return True

    gold_norm = _norm(gold_content)
    item_norm = _norm(text)
    if gold_norm and item_norm:
        if gold_norm in item_norm:
            return True
        gold_tokens = set(gold_norm.split())
        if gold_tokens:
            overlap = len(gold_tokens & set(item_norm.split())) / len(gold_tokens)
            if overlap >= 0.75:
                return True
    return False


def _dcg(relevances: Sequence[int], k: int) -> float:
    total = 0.0
    for idx, rel in enumerate(relevances[:k]):
        if idx == 0:
            total += float(rel)
        else:
            total += float(rel) / math.log2(idx + 1)
    return total


def _ranking_metrics(record: PipelineCheckpointQARecord) -> tuple[dict[str, float], dict[str, float]]:
    gold_ids = list(record.gold_evidence_memory_ids)
    gold_contents = list(record.gold_evidence_contents)
    if not gold_ids and gold_contents:
        gold_ids = ["" for _ in gold_contents]
    while len(gold_contents) < len(gold_ids):
        gold_contents.append("")
    gold_pairs = [
        (gid, gold_contents[idx] if idx < len(gold_contents) else "")
        for idx, gid in enumerate(gold_ids)
    ]
    items = list(record.retrieval.items)
    raw_trace = record.retrieval.raw_trace or {}

    recall_at: dict[str, float] = {}
    ndcg_at: dict[str, float] = {}
    for k in _NDCG_KS:
        top_items = items[:k]
        if not gold_pairs:
            recall_at[str(k)] = 0.0
            ndcg_at[str(k)] = 0.0
            continue

        covered_gold = 0
        for gid, content in gold_pairs:
            if any(
                _item_matches_gold(
                    item=item,
                    raw_trace=raw_trace,
                    gold_id=gid,
                    gold_content=content,
                )
                for item in top_items
            ):
                covered_gold += 1
        recall_at[str(k)] = covered_gold / len(gold_pairs)

        seen_gold: set[int] = set()
        relevances: list[int] = []
        for item in items:
            matched_indices = [
                idx for idx, (gid, content) in enumerate(gold_pairs)
                if idx not in seen_gold and
                _item_matches_gold(
                    item=item,
                    raw_trace=raw_trace,
                    gold_id=gid,
                    gold_content=content,
                )
            ]
            if matched_indices:
                relevances.append(1)
                seen_gold.update(matched_indices)
            else:
                relevances.append(0)
        ideal = [1] * min(len(gold_pairs), k)
        idcg = _dcg(ideal, k)
        ndcg_at[str(k)] = _dcg(relevances, k) / idcg if idcg else 0.0
    return recall_at, ndcg_at


def _evaluate_checkpoint_qa_answer_core(
    record: PipelineCheckpointQARecord,
) -> tuple[dict[str, object], float, float, dict[str, int]]:
    """Shared answer LLM judge + lexical metrics."""
    gold_contents = list(record.gold_evidence_contents)
    gold_evidence_str = (
        "\n".join(gold_contents) if gold_contents else "No evidence available."
    )
    answer_result = evaluate_qa_llm(
        question=record.question,
        reference_answer=record.gold_answer,
        key_memory_points=gold_evidence_str,
        system_response=record.generated_answer,
    )
    answer_label = answer_result.get("evaluation_result")
    answer_f1 = _answer_f1(record.generated_answer, record.gold_answer)
    answer_bleu1 = _answer_bleu1(record.generated_answer, record.gold_answer)
    judge_token_usage = dict(answer_result.get("token_usage") or {})
    return answer_result, answer_f1, answer_bleu1, judge_token_usage


def evaluate_checkpoint_qa_answer_only(
    record: PipelineCheckpointQARecord,
    **_kwargs: object,
) -> dict[str, object]:
    """Checkpoint QA: answer correctness only (BaseModel / Harness)."""
    answer_result, answer_f1, answer_bleu1, judge_token_usage = (
        _evaluate_checkpoint_qa_answer_core(record)
    )
    answer_label = answer_result.get("evaluation_result")
    retrieval_token_usage = dict(getattr(record, "retrieval_token_usage", {}) or {})
    answer_token_usage = dict(getattr(record, "answer_token_usage", {}) or {})
    total_token_usage = merge_token_usage(
        retrieval_token_usage,
        answer_token_usage,
        judge_token_usage,
    )
    return {
        "eval_mode": "answer_only",
        "answer_label": answer_label,
        "answer_reasoning": answer_result.get("reasoning", ""),
        "answer_is_valid": answer_label in ("Correct", "Hallucination", "Omission"),
        "answer_f1": answer_f1,
        "answer_bleu1": answer_bleu1,
        "retrieval_eval_skipped": True,
        "retrieval_hit_rate": None,
        "retrieval_covered_count": None,
        "retrieval_reasoning": "",
        "retrieval_recall_at": None,
        "retrieval_ndcg_at": None,
        "retrieval_ranking_num_gold": len(record.gold_evidence_memory_ids),
        "num_retrieved": len(record.retrieval.items),
        "num_evidence": len(record.gold_evidence_contents),
        "question_type": record.question_type,
        "question_type_abbrev": record.question_type_abbrev,
        "difficulty": record.difficulty,
        "retrieval_seconds": float(getattr(record, "retrieval_seconds", 0.0) or 0.0),
        "answer_seconds": float(getattr(record, "answer_seconds", 0.0) or 0.0),
        "retrieval_token_usage": retrieval_token_usage,
        "answer_token_usage": answer_token_usage,
        "judge_token_usage": judge_token_usage,
        "total_token_usage": total_token_usage,
    }


def evaluate_checkpoint_qa(
    record: PipelineCheckpointQARecord,
    **_kwargs: object,
) -> dict[str, object]:
    """Two independent LLM judgements: answer correctness + retrieval coverage."""
    if bool(_kwargs.get("answer_only", False)):
        return evaluate_checkpoint_qa_answer_only(record)

    answer_result, answer_f1, answer_bleu1, judge_token_usage = (
        _evaluate_checkpoint_qa_answer_core(record)
    )
    answer_label = answer_result.get("evaluation_result")
    ranking_recall_at, ranking_ndcg_at = _ranking_metrics(record)

    gold_contents = list(record.gold_evidence_contents)

    # --- 2. Retrieval coverage ---
    retrieval_str = _build_retrieval_text(record)
    evidence_token_usage: dict[str, int] = {}
    if not gold_contents or not retrieval_str.strip():
        ret_hit = 0.0
        ret_covered: int | None = 0
        ret_total = len(gold_contents)
        ret_reason = ""
    else:
        result = evaluate_evidence_batch(retrieval_str, gold_contents)
        ret_covered = result.get("covered_count")
        ret_total = result.get("total", len(gold_contents))
        ret_hit = (
            float(ret_covered) / float(ret_total)
            if ret_covered is not None and ret_total
            else 0.0
        )
        ret_reason = result.get("reasoning", "")
        evidence_token_usage = dict(result.get("token_usage") or {})
    judge_token_usage = merge_token_usage(judge_token_usage, evidence_token_usage)
    retrieval_token_usage = dict(getattr(record, "retrieval_token_usage", {}) or {})
    answer_token_usage = dict(getattr(record, "answer_token_usage", {}) or {})
    total_token_usage = merge_token_usage(
        retrieval_token_usage,
        answer_token_usage,
        judge_token_usage,
    )

    return {
        "answer_label": answer_label,
        "answer_reasoning": answer_result.get("reasoning", ""),
        "answer_is_valid": answer_label in ("Correct", "Hallucination", "Omission"),
        "answer_f1": answer_f1,
        "answer_bleu1": answer_bleu1,
        # Retrieval side — independent of the answer model's behaviour
        "retrieval_hit_rate": ret_hit,
        "retrieval_covered_count": ret_covered,
        "retrieval_reasoning": ret_reason,
        "retrieval_recall_at": ranking_recall_at,
        "retrieval_ndcg_at": ranking_ndcg_at,
        "retrieval_ranking_num_gold": len(record.gold_evidence_memory_ids),
        "num_retrieved": len(record.retrieval.items),
        "num_evidence": ret_total,
        # Metadata for per-type aggregation
        "question_type": record.question_type,
        "question_type_abbrev": record.question_type_abbrev,
        "difficulty": record.difficulty,
        # Runtime / token accounting
        "retrieval_seconds": float(getattr(record, "retrieval_seconds", 0.0) or 0.0),
        "answer_seconds": float(getattr(record, "answer_seconds", 0.0) or 0.0),
        "retrieval_token_usage": retrieval_token_usage,
        "answer_token_usage": answer_token_usage,
        "judge_token_usage": judge_token_usage,
        "total_token_usage": total_token_usage,
    }
