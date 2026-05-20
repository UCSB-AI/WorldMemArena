"""Judge stack: batch LLM evaluation.

Session: 2 calls (recall + correctness) + per-item calls for update/interference.
QA: 2 calls (answer + evidence).
"""

from __future__ import annotations

from eval_framework.judges.llm_client import llm_request_for_json
from eval_framework.judges.prompts import (
    CORRECTNESS_BATCH_PROMPT,
    EVIDENCE_BATCH_PROMPT,
    INTERFERENCE_EVAL_PROMPT,
    MEMORY_ACCURACY_ITEMWISE_LABEL_PROMPT,
    MEMORY_ACCURACY_ITEMWISE_RECALL_PROMPT,
    QA_EVALUATION_PROMPT,
    RECALL_BATCH_PROMPT,
    UPDATE_EVAL_PROMPT,
)
from eval_framework.openai_compat import empty_token_usage

__all__ = [
    "evaluate_recall_batch",
    "evaluate_correctness_batch",
    "evaluate_candidate_label_itemwise",
    "evaluate_gold_recall_itemwise",
    "evaluate_update_single",
    "evaluate_interference_single",
    "evaluate_evidence_batch",
    "evaluate_qa_llm",
    "llm_request_for_json",
]


def _usage_from_result(result: dict[str, object]) -> dict[str, int]:
    usage = result.get("_llm_usage")
    if not isinstance(usage, dict):
        return empty_token_usage()
    out = empty_token_usage()
    for key in out:
        try:
            out[key] = int(usage.get(key) or 0)
        except (TypeError, ValueError):
            out[key] = 0
    return out


def _covered_indices_from_result(
    result: dict[str, object],
    *,
    total: int,
) -> list[int]:
    raw = result.get("covered_indices")
    if raw is None:
        raw = result.get("covered_ids")
    indices: list[int] = []
    if isinstance(raw, list):
        for item in raw:
            try:
                idx = int(item)
            except (TypeError, ValueError):
                continue
            if 1 <= idx <= total and idx not in indices:
                indices.append(idx)
    return indices


def evaluate_recall_batch(
    extracted_memories_str: str,
    gold_points_tagged: list[str],
) -> dict[str, object]:
    """One LLM call: how many gold points are covered?

    gold_points_tagged: list of "[normal] content" or "[update] content" strings.
    Returns {covered_count, total, reasoning}.
    """
    if not extracted_memories_str.strip():
        return {
            "covered_count": 0,
            "total": len(gold_points_tagged),
            "covered_indices": [],
            "reasoning": "No extracted memories.",
            "token_usage": empty_token_usage(),
        }
    if not gold_points_tagged:
        return {
            "covered_count": 0,
            "total": 0,
            "covered_indices": [],
            "reasoning": "No gold points.",
            "token_usage": empty_token_usage(),
        }

    numbered = "\n".join(f"[{i+1}] {p}" for i, p in enumerate(gold_points_tagged))

    prompt = RECALL_BATCH_PROMPT.format(memories=extracted_memories_str, gold_points=numbered)
    try:
        result = llm_request_for_json(prompt)
        covered_indices = _covered_indices_from_result(
            result, total=len(gold_points_tagged)
        )
        raw_covered = result.get("covered_count")
        covered = len(covered_indices) if raw_covered is None else int(raw_covered)
        return {
            "covered_count": min(covered, len(gold_points_tagged)),
            "total": len(gold_points_tagged),
            "covered_indices": covered_indices,
            "reasoning": result.get("reasoning", ""),
            "token_usage": _usage_from_result(result),
        }
    except Exception as e:
        return {
            "covered_count": None,
            "total": len(gold_points_tagged),
            "covered_indices": [],
            "reasoning": f"LLM error: {e}",
            "token_usage": empty_token_usage(),
        }


def evaluate_correctness_batch(
    snapshot_memories: list[str],
    gold_points_tagged: list[str],
    dialogue_str: str = "",
) -> dict[str, object]:
    """One LLM call: is each memory correct / hallucination / error?

    Uses dialogue + gold as ground truth.
    Returns {results: [{id, label}], reasoning}.
    """
    if not snapshot_memories:
        return {
            "results": [],
            "reasoning": "No snapshot memories.",
            "token_usage": empty_token_usage(),
        }

    num_memories = len(snapshot_memories)
    numbered_memories = "\n".join(f"[{i+1}] {m}" for i, m in enumerate(snapshot_memories))
    numbered_golds = "\n".join(f"- {p}" for p in gold_points_tagged) if gold_points_tagged else "(no ground-truth)"

    prompt = CORRECTNESS_BATCH_PROMPT.format(
        memories=numbered_memories,
        gold_points=numbered_golds,
        num_memories=num_memories,
        dialogue=_escape_format_braces(dialogue_str) if dialogue_str else "(no dialogue available)",
    )
    valid_labels = {"correct", "hallucination", "error"}
    try:
        result = llm_request_for_json(prompt)
        raw_results = result.get("results", [])
        cleaned = []
        for r in raw_results:
            label = str(r.get("label", "error")).lower().strip()
            if label not in valid_labels:
                label = "error"
            cleaned.append({"id": r.get("id"), "label": label})

        if len(cleaned) < num_memories:
            labeled_ids = {r["id"] for r in cleaned if r["id"] is not None}
            for i in range(1, num_memories + 1):
                if i not in labeled_ids:
                    cleaned.append({"id": i, "label": "error"})
            cleaned.sort(key=lambda r: (r["id"] if isinstance(r["id"], int) else 9999))
            cleaned = cleaned[:num_memories]

        return {
            "results": cleaned,
            "reasoning": result.get("reasoning", ""),
            "token_usage": _usage_from_result(result),
        }
    except Exception as e:
        fallback = [{"id": i + 1, "label": "error"} for i in range(num_memories)]
        return {
            "results": fallback,
            "reasoning": f"LLM error: {e}",
            "token_usage": empty_token_usage(),
        }


def _escape_format_braces(s: str) -> str:
    """Allow arbitrary text in str.format without interpreting {{ }} from user."""
    return s.replace("{", "{{").replace("}", "}}")


def evaluate_candidate_label_itemwise(
    dialogue_str: str,
    golden_memories_str: str,
    candidate_memory: str,
) -> dict[str, object]:
    """One LLM call per candidate: correct / hallucination / error."""
    if not candidate_memory.strip():
        return {
            "label": "error",
            "reasoning": "Empty candidate memory.",
            "token_usage": empty_token_usage(),
        }
    prompt = MEMORY_ACCURACY_ITEMWISE_LABEL_PROMPT.format(
        dialogue_str=_escape_format_braces(dialogue_str or "(empty)"),
        golden_memories_str=_escape_format_braces(
            golden_memories_str or "(no golden memories for this session)"
        ),
        candidate_memory=_escape_format_braces(candidate_memory.strip()),
    )
    valid_labels = {"correct", "hallucination", "error"}
    try:
        result = llm_request_for_json(prompt)
        label = str(result.get("label", "error")).lower().strip()
        if label not in valid_labels:
            label = "error"
        return {
            "label": label,
            "reasoning": str(result.get("reasoning", "")),
            "token_usage": _usage_from_result(result),
        }
    except Exception as e:
        return {
            "label": "error",
            "reasoning": f"LLM error: {e}",
            "token_usage": empty_token_usage(),
        }


def evaluate_gold_recall_itemwise(
    gold_memory: str,
    candidate_memories_str: str,
) -> dict[str, object]:
    """One LLM call per gold line: is this gold covered by any candidate?"""
    if not gold_memory.strip():
        return {
            "covered": False,
            "matched_candidate_indices": [],
            "reasoning": "Empty gold.",
            "token_usage": empty_token_usage(),
        }
    if not candidate_memories_str.strip():
        return {
            "covered": False,
            "matched_candidate_indices": [],
            "reasoning": "No candidates.",
            "token_usage": empty_token_usage(),
        }
    prompt = MEMORY_ACCURACY_ITEMWISE_RECALL_PROMPT.format(
        gold_memory=_escape_format_braces(gold_memory.strip()),
        candidate_memories_str=_escape_format_braces(candidate_memories_str),
    )
    try:
        result = llm_request_for_json(prompt)
        cov = result.get("covered")
        if isinstance(cov, str):
            cov = cov.lower() in ("true", "1", "yes")
        covered = bool(cov)
        raw_ids = result.get("matched_candidate_indices")
        indices: list[int] = []
        if isinstance(raw_ids, list):
            for x in raw_ids:
                try:
                    indices.append(int(x))
                except (TypeError, ValueError):
                    continue
        return {
            "covered": covered,
            "matched_candidate_indices": indices,
            "reasoning": str(result.get("reasoning", "")),
            "token_usage": _usage_from_result(result),
        }
    except Exception as e:
        return {
            "covered": False,
            "matched_candidate_indices": [],
            "reasoning": f"LLM error: {e}",
            "token_usage": empty_token_usage(),
        }


def evaluate_update_single(
    delta_memories_str: str,
    new_content: str,
    old_contents: list[str],
) -> dict[str, object]:
    """One LLM call: how did the system handle a single memory update?

    Returns {label: "updated"|"both"|"outdated", reasoning}.
    """
    old_str = "\n".join(f"- {o}" for o in old_contents) if old_contents else "(none)"
    prompt = UPDATE_EVAL_PROMPT.format(
        memories=delta_memories_str,
        new_content=new_content,
        old_contents=old_str,
    )
    try:
        result = llm_request_for_json(prompt)
        label = str(result.get("label", "outdated")).lower().strip()
        if label not in ("updated", "both", "outdated"):
            label = "outdated"
        return {
            "label": label,
            "reasoning": result.get("reasoning", ""),
            "token_usage": _usage_from_result(result),
        }
    except Exception as e:
        return {
            "label": None,
            "reasoning": f"LLM error: {e}",
            "token_usage": empty_token_usage(),
        }


def evaluate_interference_single(
    delta_memories_str: str,
    interference_content: str,
) -> dict[str, object]:
    """One LLM call: did the system incorrectly memorize an interference point?

    Returns {label: "rejected"|"memorized", reasoning}.
    """
    prompt = INTERFERENCE_EVAL_PROMPT.format(
        memories=delta_memories_str,
        interference_content=interference_content,
    )
    try:
        result = llm_request_for_json(prompt)
        label = str(result.get("label", "memorized")).lower().strip()
        if label not in ("rejected", "memorized"):
            label = "memorized"
        return {
            "label": label,
            "reasoning": result.get("reasoning", ""),
            "token_usage": _usage_from_result(result),
        }
    except Exception as e:
        return {
            "label": None,
            "reasoning": f"LLM error: {e}",
            "token_usage": empty_token_usage(),
        }


def evaluate_evidence_batch(
    retrieved_memories_str: str,
    evidence_points: list[str],
) -> dict[str, object]:
    """One LLM call: how many gold evidence points are covered by retrieval?"""
    if not retrieved_memories_str.strip():
        return {
            "covered_count": 0,
            "total": len(evidence_points),
            "reasoning": "No retrieved memories.",
            "token_usage": empty_token_usage(),
        }
    if not evidence_points:
        return {
            "covered_count": 0,
            "total": 0,
            "reasoning": "No evidence points.",
            "token_usage": empty_token_usage(),
        }

    numbered = "\n".join(f"[{i+1}] {p}" for i, p in enumerate(evidence_points))
    prompt = EVIDENCE_BATCH_PROMPT.format(retrieved_memories=retrieved_memories_str, gold_evidence_points=numbered)
    try:
        result = llm_request_for_json(prompt)
        covered = int(result.get("covered_count", 0))
        return {
            "covered_count": min(covered, len(evidence_points)),
            "total": len(evidence_points),
            "reasoning": result.get("reasoning", ""),
            "token_usage": _usage_from_result(result),
        }
    except Exception as e:
        return {
            "covered_count": None,
            "total": len(evidence_points),
            "reasoning": f"LLM error: {e}",
            "token_usage": empty_token_usage(),
        }


def evaluate_qa_llm(
    question: str,
    reference_answer: str,
    key_memory_points: str,
    system_response: str,
) -> dict[str, object]:
    """LLM judge: classify the QA response as Correct/Hallucination/Omission."""
    if not system_response.strip():
        return {
            "evaluation_result": "Omission",
            "reasoning": "Empty system response.",
            "token_usage": empty_token_usage(),
        }

    prompt = QA_EVALUATION_PROMPT.format(
        question=question, reference_answer=reference_answer,
        key_memory_points=key_memory_points, response=system_response,
    )
    try:
        result = llm_request_for_json(prompt)
        label = result.get("evaluation_result", "Omission")
        if label not in ("Correct", "Hallucination", "Omission"):
            label = "Omission"
        return {
            "evaluation_result": label,
            "reasoning": result.get("reasoning", ""),
            "token_usage": _usage_from_result(result),
        }
    except Exception as e:
        return {
            "evaluation_result": None,
            "reasoning": f"LLM judge error: {e}",
            "token_usage": empty_token_usage(),
        }
