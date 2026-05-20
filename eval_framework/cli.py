"""CLI: dataset path, baseline, output dir, dry-run, smoke eval.

Evaluation uses batch LLM judge: 2 calls/session + 2 calls/QA.
Session and QA evaluations run in parallel via ThreadPoolExecutor.
Pipeline results are checkpointed before eval so --eval-only can resume.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
import threading

from eval_framework.baselines._clients.base_model_api import (
    call_chat_target,
    is_anthropic_target,
    target_for_baseline,
)
from eval_framework.config import EvalConfig, is_answer_only_eval_baseline
from eval_framework.datasets.wma_bundle import (
    EvalBundle,
    NormalizedCheckpointQuestion,
    load_legacy_bundle,
)
from functools import partial

from eval_framework.datasets.worldmemarena import load_worldmemarena

DATASET_LOADERS: dict[str, Callable[[Path], EvalBundle]] = {
    "legacy": load_legacy_bundle,
    "worldmemarena": load_worldmemarena,
}


def _resolve_loader(ns: argparse.Namespace) -> Callable[[Path], EvalBundle]:
    """Return a dataset loader, binding --split for worldmemarena."""
    base_loader = DATASET_LOADERS[ns.dataset_type]
    split = getattr(ns, "split", "all")
    if split != "all" and ns.dataset_type == "worldmemarena":
        return partial(load_worldmemarena, split=split)
    return base_loader


from eval_framework.datasets.schemas import (
    MemoryDeltaRecord,
    MemorySnapshotRecord,
    RetrievalItem,
    RetrievalRecord,
)
from eval_framework.evaluators.aggregate import aggregate_metrics
from eval_framework.evaluators.extraction import evaluate_extraction
from eval_framework.evaluators.memory_accuracy_itemwise import (
    evaluate_memory_accuracy_itemwise,
)
from eval_framework.evaluators.qa import evaluate_checkpoint_qa
from eval_framework.json_repair_utils import loads_repaired_or_none
from eval_framework.memory_adapters.base import MemoryAdapter
from eval_framework.openai_compat import patch_openai_chat_completions
from eval_framework.pipeline.gold_state import GoldMemoryPoint, SessionGoldState
from eval_framework.pipeline.records import PipelineCheckpointQARecord, PipelineSessionRecord
from eval_framework.pipeline.runner import run_eval_sample


_CHECKPOINT_SESSIONS = "pipeline_sessions.jsonl"
_CHECKPOINT_QA = "pipeline_qa.jsonl"
_SAMPLE_RESULTS_DIR = "sample_results"
_PARTIAL_SESSION_RECORDS = "partial_session_records.jsonl"
_PARTIAL_QA_RECORDS = "partial_qa_records.jsonl"
_PARTIAL_AGGREGATE = "partial_aggregate_metrics.json"

_SESSION_EVAL_SKIPPED_ANSWER_ONLY: dict[str, object] = {
    "eval_mode": "answer_only",
    "skipped": True,
    "reason": "answer_only_baseline",
}


# ---------------------------------------------------------------------------
# Checkpoint deserialization: dict -> frozen dataclass
# ---------------------------------------------------------------------------

def _gold_point_from_dict(d: dict[str, Any]) -> GoldMemoryPoint:
    return GoldMemoryPoint(
        memory_id=d["memory_id"],
        memory_content=d["memory_content"],
        memory_type=d["memory_type"],
        memory_source=d["memory_source"],
        is_update=bool(d["is_update"]),
        original_memories=tuple(d.get("original_memories") or ()),
        importance=float(d.get("importance", 0.0)),
        timestamp=d.get("timestamp"),
        update_type=d.get("update_type", ""),
    )


def _gold_state_from_dict(d: dict[str, Any]) -> SessionGoldState:
    return SessionGoldState(
        session_id=d["session_id"],
        cumulative_gold_memories=tuple(_gold_point_from_dict(g) for g in d["cumulative_gold_memories"]),
        session_new_memories=tuple(_gold_point_from_dict(g) for g in d["session_new_memories"]),
        session_update_memories=tuple(_gold_point_from_dict(g) for g in d["session_update_memories"]),
        session_interference_memories=tuple(_gold_point_from_dict(g) for g in d["session_interference_memories"]),
    )


def _snapshot_record_from_dict(d: dict[str, Any]) -> MemorySnapshotRecord:
    return MemorySnapshotRecord(
        memory_id=d["memory_id"],
        text=d["text"],
        session_id=d["session_id"],
        status=d["status"],
        source=d.get("source"),
        raw_backend_id=d.get("raw_backend_id"),
        raw_backend_type=d.get("raw_backend_type"),
        metadata=d.get("metadata") or {},
    )


def _delta_record_from_dict(d: dict[str, Any]) -> MemoryDeltaRecord:
    return MemoryDeltaRecord(
        session_id=d["session_id"],
        op=d["op"],
        text=d["text"],
        linked_previous=tuple(d.get("linked_previous") or ()),
        raw_backend_id=d.get("raw_backend_id"),
        metadata=d.get("metadata") or {},
    )


def _retrieval_item_from_dict(d: dict[str, Any]) -> RetrievalItem:
    return RetrievalItem(
        rank=int(d["rank"]),
        memory_id=d["memory_id"],
        text=d["text"],
        score=float(d["score"]),
        raw_backend_id=d.get("raw_backend_id"),
        image_path=d.get("image_path"),
    )


def _retrieval_record_from_dict(d: dict[str, Any]) -> RetrievalRecord:
    return RetrievalRecord(
        query=d["query"],
        top_k=int(d["top_k"]),
        items=[_retrieval_item_from_dict(i) for i in d["items"]],
        raw_trace=d.get("raw_trace") or {},
    )


def _token_usage_from_dict(d: Any) -> dict[str, int]:
    if not isinstance(d, dict):
        return {}
    out: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        try:
            out[key] = int(d.get(key) or 0)
        except (TypeError, ValueError):
            out[key] = 0
    return out


def _session_record_from_dict(d: dict[str, Any]) -> PipelineSessionRecord:
    return PipelineSessionRecord(
        sample_id=d["sample_id"],
        sample_uuid=d["sample_uuid"],
        session_id=d["session_id"],
        memory_snapshot=tuple(_snapshot_record_from_dict(s) for s in d["memory_snapshot"]),
        memory_delta=tuple(_delta_record_from_dict(dl) for dl in d["memory_delta"]),
        gold_state=_gold_state_from_dict(d["gold_state"]),
        dialogue_str=str(d.get("dialogue_str") or ""),
        storage_seconds=float(d.get("storage_seconds") or 0.0),
        storage_token_usage=_token_usage_from_dict(d.get("storage_token_usage")),
    )


def _qa_record_from_dict(d: dict[str, Any]) -> PipelineCheckpointQARecord:
    return PipelineCheckpointQARecord(
        sample_id=d["sample_id"],
        sample_uuid=d["sample_uuid"],
        checkpoint_id=d["checkpoint_id"],
        question=d["question"],
        gold_answer=d["gold_answer"],
        gold_evidence_memory_ids=tuple(d.get("gold_evidence_memory_ids") or ()),
        gold_evidence_contents=tuple(d.get("gold_evidence_contents") or ()),
        question_type=d["question_type"],
        question_type_abbrev=d["question_type_abbrev"],
        difficulty=d["difficulty"],
        retrieval=_retrieval_record_from_dict(d["retrieval"]),
        generated_answer=d["generated_answer"],
        cited_memories=tuple(d.get("cited_memories") or ()),
        retrieval_seconds=float(d.get("retrieval_seconds") or 0.0),
        answer_seconds=float(d.get("answer_seconds") or 0.0),
        retrieval_token_usage=_token_usage_from_dict(d.get("retrieval_token_usage")),
        answer_token_usage=_token_usage_from_dict(d.get("answer_token_usage")),
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_pipeline_checkpoint(
    output_dir: Path,
) -> tuple[list[PipelineSessionRecord], list[PipelineCheckpointQARecord]]:
    """Restore pipeline records from checkpoint JSONL files."""
    sess_path = output_dir / _CHECKPOINT_SESSIONS
    qa_path = output_dir / _CHECKPOINT_QA
    if not sess_path.exists() or not qa_path.exists():
        raise SystemExit(
            f"Checkpoint files not found in {output_dir}. "
            f"Run without --eval-only first to generate them."
        )
    session_records = [_session_record_from_dict(d) for d in _read_jsonl(sess_path)]
    qa_records = [_qa_record_from_dict(d) for d in _read_jsonl(qa_path)]
    return session_records, qa_records


def _sample_status_payload(sample_dir: Path) -> dict[str, Any]:
    status_path = sample_dir / "sample_status.json"
    if not status_path.exists():
        return {}
    try:
        raw = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _sample_status(sample_dir: Path) -> str:
    raw = _sample_status_payload(sample_dir)
    return str(raw.get("status") or "")


def _default_create_adapter(baseline_name: str) -> MemoryAdapter:
    from eval_framework.memory_adapters import registry as reg

    if baseline_name in reg.MEMGALLERY_NATIVE_REGISTRY:
        return reg.MEMGALLERY_NATIVE_REGISTRY[baseline_name]()
    try:
        return reg.create_external_adapter(baseline_name)
    except KeyError:
        pass
    known = sorted(
        reg.MEMGALLERY_NATIVE_BASELINES | reg.EXTERNAL_ADAPTER_KEYS
    )
    raise SystemExit(
        f"Unknown baseline {baseline_name!r}. "
        f"Expected one of: {', '.join(known)}"
    )


def _gold_echo_answer(
    q: NormalizedCheckpointQuestion, _retrieval: RetrievalRecord
) -> str:
    return q.gold_answer


def _parse_answer_json(raw: str) -> str:
    """Extract the ``answer`` field from the model's JSON response."""
    data = loads_repaired_or_none(raw)
    if isinstance(data, dict):
        return str(data.get("answer", ""))
    if isinstance(data, str):
        return data.strip()
    import re
    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        data = loads_repaired_or_none(m.group())
        if isinstance(data, dict):
            return str(data.get("answer", ""))
        if isinstance(data, str):
            return data.strip()
    return raw.strip()


def _parse_batch_answer_json(raw: str, num_questions: int) -> list[str]:
    """Parse batch QA JSON response into a list of answers."""
    import re
    # Try direct parse as list
    data = loads_repaired_or_none(raw, expected_type=list)
    if isinstance(data, list):
        answers = [""] * num_questions
        for item in data:
            if isinstance(item, dict):
                try:
                    idx = int(item.get("q_index", -1))
                except (TypeError, ValueError):
                    continue
                if 0 <= idx < num_questions:
                    answers[idx] = str(item.get("answer", ""))
        if any(answers):
            return answers
    # Try finding a JSON array in the output
    m = re.search(r"\[[\s\S]*\]", raw)
    if m:
        data = loads_repaired_or_none(m.group(), expected_type=list)
        if isinstance(data, list):
            answers = [""] * num_questions
            for item in data:
                if isinstance(item, dict):
                    try:
                        idx = int(item.get("q_index", -1))
                    except (TypeError, ValueError):
                        continue
                    if 0 <= idx < num_questions:
                        answers[idx] = str(item.get("answer", ""))
            if any(answers):
                return answers
    # Fallback: try to find individual JSON objects
    answers = []
    for obj_match in re.finditer(r"\{[^{}]*\}", raw):
        item = loads_repaired_or_none(obj_match.group(), expected_type=dict)
        if isinstance(item, dict) and "answer" in item:
            answers.append(str(item["answer"]))
    if len(answers) >= num_questions:
        return answers[:num_questions]
    # Pad with empty strings if partial parse
    while len(answers) < num_questions:
        answers.append("")
    return answers


_ANSWER_SYSTEM_PROMPT = """You are an intelligent memory assistant. Answer the user's question based on the retrieved memories provided.

# Instructions:
1. Carefully analyze all retrieved memories. Synthesize across multiple entries if a single memory is insufficient.
2. Pay attention to timestamps. If memories conflict, prioritize the most recent.
3. Convert relative time references ("last year", "two months ago") into absolute dates based on the memory timestamps. Example: a memory dated 2025-05-04 saying "went to India last year" means the trip was in 2024.
4. Ground every claim in the retrieved memories. You may use world knowledge only to interpret content already present (e.g. identify a landmark from its description). Do NOT invent missing facts.
5. If the question asks about an image, return the relevant image_id(s) found in the memory captions (e.g. "S04_img_4"). When multiple apply, list them comma-separated.
6. Keep answers concise — a single short phrase or under 15 words. No introductory filler like "The answer is".
7. Only if the retrieved memories truly contain NO information relevant to the question, reply exactly: "Not mentioned in memory."

# Approach (think step by step before answering):
1. Identify which memories are relevant to the question.
2. Examine their timestamps and content.
3. Synthesize across multiple memories if needed.
4. Perform any required calculation (e.g. relative → absolute time).
5. Formulate a precise, concise answer grounded in the evidence.

Always respond in the requested JSON format."""


_ANSWER_USER_TEMPLATE = """Retrieved memories:
{context}

Question: {question}

Respond in JSON:
{{
  "answer": "your concise answer (or exactly 'Not mentioned in memory.' only if truly absent)"
}}"""


_HARNESS_NATIVE_QA_PROMPT_TEMPLATE = """You have previously been given conversation sessions to remember.
Answer the following question based on what you remember.

Keep the answer concise, preferably a single short phrase or under 15 words.
If you truly cannot answer from memory, say exactly: "Not mentioned in memory."

Question: {question}

Respond in JSON:
{{
  "answer": "your concise answer"
}}"""


def build_default_answer_fn(
    evidence_mode: str = "memory",
) -> Callable[
    [NormalizedCheckpointQuestion, RetrievalRecord], str
]:
    from eval_framework.config import (
        is_base_model_baseline,
        is_harness_baseline,
        resolve_baseline_param,
        resolve_base_model_max_image_bytes,
        resolve_base_model_max_images,
        resolve_base_model_mode,
    )

    mm_enabled = resolve_base_model_mode() == "image"
    max_images = resolve_base_model_max_images()
    max_image_bytes = resolve_base_model_max_image_bytes()
    evidence_mode = evidence_mode.strip().lower()
    if evidence_mode not in {"memory", "session"}:
        evidence_mode = "memory"

    def _answer(
        q: NormalizedCheckpointQuestion, retrieval: RetrievalRecord
    ) -> Any:
        trace = retrieval.raw_trace or {}

        def _normalize_baseline_name(raw_name: object) -> str:
            name = str(raw_name or "").strip()
            if name == "SimpleMem-text":
                return "SimpleMem"
            if name == "SimpleMem-omni":
                return "Omni-SimpleMem"
            return name

        baseline_name = _normalize_baseline_name(trace.get("baseline"))
        harness_target = None
        target = None
        if is_harness_baseline(baseline_name):
            from eval_framework.baselines._clients.harness_api import build_harness_target
            harness_target = build_harness_target(baseline_name)
            if harness_target is None:
                raise RuntimeError(f"Missing harness config for {baseline_name!r}")
            runtime_workdir = trace.get("harness_workdir")
            if runtime_workdir:
                harness_target = replace(
                    harness_target,
                    workdir=Path(str(runtime_workdir)).expanduser().resolve(),
                )
        else:
            target = target_for_baseline(baseline_name)
            if not target.api_key:
                return q.gold_answer

        baseline_allows_images = True
        if harness_target is not None:
            baseline_allows_images = harness_target.mm_mode == "image"
        elif is_base_model_baseline(baseline_name):
            assert target is not None
            baseline_allows_images = target.mm_mode == "image"
        elif baseline_name:
            mm_mode_cfg = resolve_baseline_param(baseline_name, "mm_mode", None)
            if mm_mode_cfg is not None:
                mode = str(mm_mode_cfg).strip().lower()
                baseline_allows_images = mode in {"image", "mm", "multimodal"}
        allow_answer_images = mm_enabled and baseline_allows_images

        # BaseModel targets may declare per-provider image caps (e.g. claude
        # haiku has a 32 MB request body limit). Honor those overrides; fall
        # back to the global config.yaml cap otherwise.
        eff_max_images = max_images
        eff_max_image_bytes = max_image_bytes
        if target is not None:
            if getattr(target, "max_images", -1) >= 0:
                eff_max_images = target.max_images
            if getattr(target, "max_image_bytes", -1) >= 0:
                eff_max_image_bytes = target.max_image_bytes

        if harness_target is not None and trace.get("harness_native_memory"):
            from eval_framework.baselines._clients.harness_api import call_harness_prompt

            raw, _usage = call_harness_prompt(
                harness_target,
                prompt=_HARNESS_NATIVE_QA_PROMPT_TEMPLATE.format(question=q.question),
                image_paths=[],
                max_images=0,
                max_image_bytes=0,
                use_workdir=True,
            )
            answer = _parse_answer_json(raw)
            return answer

        context_items = trace.get("answer_context_items")
        context_lines: list[str] = []
        if evidence_mode == "session" and isinstance(context_items, list):
            for idx, item in enumerate(context_items):
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text") or "")
                if text.strip():
                    sid = item.get("source_session_id") or item.get("source_type") or idx
                    context_lines.append(f"[{sid}] {text}")
        if not context_lines:
            context_lines = [
                f"[{item.rank}] {item.text}" for item in retrieval.items[: retrieval.top_k]
            ]
        context = "\n".join(context_lines) if context_lines else "No retrieved memories."
        user_prompt = _ANSWER_USER_TEMPLATE.format(
            context=context, question=q.question
        )

        # Images are attached only when the global base_model.mode enables image
        # answering and this baseline/provider is image-like.
        image_paths: list[str] = []
        if allow_answer_images:
            seen_paths: set[str] = set()
            for item in retrieval.items[: retrieval.top_k]:
                ip = getattr(item, "image_path", None)
                if not ip or ip in seen_paths:
                    continue
                seen_paths.add(ip)
                image_paths.append(ip)
                if len(image_paths) >= eff_max_images:
                    break

        try:
            if harness_target is not None:
                from eval_framework.baselines._clients.harness_api import call_harness_target
                raw, usage = call_harness_target(
                    harness_target,
                    system_prompt=_ANSWER_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    image_paths=image_paths,
                    max_images=eff_max_images,
                    max_image_bytes=eff_max_image_bytes,
                )
            else:
                assert target is not None
                raw, usage = call_chat_target(
                    target,
                    system_prompt=_ANSWER_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    image_paths=image_paths,
                    max_images=eff_max_images,
                    max_image_bytes=eff_max_image_bytes,
                )
        except Exception as exc:
            print(
                f"    [_answer] {baseline_name} call failed after retries: {exc.__class__.__name__}: {exc}",
                flush=True,
            )
            raw = ""
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        answer = _parse_answer_json(raw)
        if target is not None and is_anthropic_target(target):
            return {"answer": answer, "token_usage": usage}
        return answer

    return _answer


def config_from_namespace(ns: argparse.Namespace) -> EvalConfig:
    max_sessions = getattr(ns, "max_sessions", None)
    if max_sessions is not None and max_sessions <= 0:
        max_sessions = None
    return EvalConfig(
        dataset_path=Path(ns.dataset).expanduser().resolve(),
        output_dir=Path(ns.output_dir).expanduser().resolve(),
        baseline=str(ns.baseline),
        smoke=bool(ns.smoke),
        dry_run=bool(ns.dry_run),
        max_sessions=max_sessions,
        memory_accuracy_itemwise=bool(getattr(ns, "memory_accuracy_itemwise", False)),
        answer_evidence_mode=str(getattr(ns, "answer_evidence_mode", "memory")),
        sample_index=getattr(ns, "sample_index", None),
    )


def _record_to_json_obj(obj: Any) -> dict[str, Any]:
    return asdict(obj)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _safe_path_component(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._=-]+", "__", value.strip())
    return safe.strip("._") or "sample"


def _sample_result_dir(output_dir: Path, sample_id: str) -> Path:
    return output_dir / _SAMPLE_RESULTS_DIR / _safe_path_component(sample_id)


def _records_with_eval(
    records: list[Any],
    evaluations: list[dict[str, object] | None],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec, ev in zip(records, evaluations):
        row = _record_to_json_obj(rec)
        row["eval"] = ev
        rows.append(row)
    return rows


def _write_pipeline_checkpoint(
    output_dir: Path,
    session_records: list[PipelineSessionRecord],
    qa_records: list[PipelineCheckpointQARecord],
) -> None:
    _write_jsonl(
        output_dir / _CHECKPOINT_SESSIONS,
        [_record_to_json_obj(r) for r in session_records],
    )
    _write_jsonl(
        output_dir / _CHECKPOINT_QA,
        [_record_to_json_obj(r) for r in qa_records],
    )


def _write_sample_pipeline_result(
    output_dir: Path,
    sample_id: str,
    session_records: list[PipelineSessionRecord],
    qa_records: list[PipelineCheckpointQARecord],
    *,
    pipeline_seconds: float,
) -> None:
    sample_dir = _sample_result_dir(output_dir, sample_id)
    _write_pipeline_checkpoint(sample_dir, session_records, qa_records)
    _write_json(
        sample_dir / "sample_status.json",
        {
            "sample_id": sample_id,
            "status": "PIPELINE_DONE",
            "sessions": len(session_records),
            "qa": len(qa_records),
            "pipeline_seconds": round(pipeline_seconds, 2),
        },
    )


def _aggregate_with_timing(
    baseline: str,
    session_evals: list[dict[str, object] | None],
    qa_evals: list[dict[str, object] | None],
    *,
    pipeline_seconds: float,
    eval_seconds: float,
    total_seconds: float,
) -> dict[str, Any]:
    agg = aggregate_metrics(
        baseline,
        session_evaluations=[e for e in session_evals if e is not None],
        qa_evaluations=[e for e in qa_evals if e is not None],
    )
    agg["timing"] = {
        "pipeline_seconds": round(pipeline_seconds, 2),
        "eval_seconds": round(eval_seconds, 2),
        "total_seconds": round(total_seconds, 2),
    }
    return agg


def _write_evaluated_outputs(
    output_dir: Path,
    *,
    baseline: str,
    session_records: list[PipelineSessionRecord],
    qa_records: list[PipelineCheckpointQARecord],
    session_evals: list[dict[str, object] | None],
    qa_evals: list[dict[str, object] | None],
    pipeline_seconds: float,
    eval_seconds: float,
    total_seconds: float,
    aggregate_name: str = "aggregate_metrics.json",
    session_name: str = "session_records.jsonl",
    qa_name: str = "qa_records.jsonl",
) -> dict[str, Any]:
    agg = _aggregate_with_timing(
        baseline,
        session_evals,
        qa_evals,
        pipeline_seconds=pipeline_seconds,
        eval_seconds=eval_seconds,
        total_seconds=total_seconds,
    )
    _write_jsonl(output_dir / session_name, _records_with_eval(session_records, session_evals))
    _write_jsonl(output_dir / qa_name, _records_with_eval(qa_records, qa_evals))
    _write_json(output_dir / aggregate_name, agg)
    return agg


def _write_partial_outputs(
    output_dir: Path,
    *,
    baseline: str,
    session_records: list[PipelineSessionRecord],
    qa_records: list[PipelineCheckpointQARecord],
    session_evals: list[dict[str, object] | None],
    qa_evals: list[dict[str, object] | None],
    pipeline_seconds: float,
    eval_seconds: float,
    total_seconds: float,
    completed_samples: list[str],
) -> None:
    _write_evaluated_outputs(
        output_dir,
        baseline=baseline,
        session_records=session_records,
        qa_records=qa_records,
        session_evals=session_evals,
        qa_evals=qa_evals,
        pipeline_seconds=pipeline_seconds,
        eval_seconds=eval_seconds,
        total_seconds=total_seconds,
        aggregate_name=_PARTIAL_AGGREGATE,
        session_name=_PARTIAL_SESSION_RECORDS,
        qa_name=_PARTIAL_QA_RECORDS,
    )
    _write_json(
        output_dir / "partial_status.json",
        {
            "status": "PARTIAL",
            "completed_samples": completed_samples,
            "num_completed_samples": len(completed_samples),
            "sessions": len(session_records),
            "qa": len(qa_records),
        },
    )


def _auto_dataset_loader(dataset_path: Path):
    """Pick a dataset loader based on directory contents.

    - WorldMemArena layout (has ``agent/`` and ``lifelong/`` subdirs) →
      ``load_worldmemarena``
    - Everything else → ``load_legacy_bundle``
    """
    if dataset_path.is_dir():
        if (dataset_path / "agent").is_dir() and (dataset_path / "lifelong").is_dir():
            from eval_framework.datasets.worldmemarena import load_worldmemarena
            return load_worldmemarena
    return load_legacy_bundle


def _evaluate_session_bundle(
    srec: PipelineSessionRecord,
    *,
    memory_accuracy_itemwise: bool,
) -> dict[str, object]:
    """Batch extraction judge + optional per-candidate memory_accuracy judge."""
    ev = evaluate_extraction(srec)
    if memory_accuracy_itemwise:
        ev["memory_accuracy_itemwise"] = evaluate_memory_accuracy_itemwise(srec)
    return ev


def _evaluate_records(
    session_records: list[PipelineSessionRecord],
    qa_records: list[PipelineCheckpointQARecord],
    *,
    baseline: str,
    max_eval_workers: int,
    memory_accuracy_itemwise: bool,
    label: str = "",
) -> tuple[list[dict[str, object] | None], list[dict[str, object] | None], float]:
    """Evaluate one batch and return aligned session/QA eval lists."""
    if not session_records and not qa_records:
        return [], [], 0.0

    workers = max(1, int(max_eval_workers))
    prefix = f"  [{label}] " if label else "  "
    answer_only = is_answer_only_eval_baseline(baseline)
    if answer_only:
        print(
            f"{prefix}Evaluating {len(qa_records)} QA (answer-only, workers={workers})...",
            flush=True,
        )
    else:
        print(
            f"{prefix}Evaluating {len(session_records)} sessions + "
            f"{len(qa_records)} QA with LLM judge (workers={workers})...",
            flush=True,
        )

    t_start = time.perf_counter()
    if answer_only:
        session_evals: list[dict[str, object] | None] = [
            dict(_SESSION_EVAL_SKIPPED_ANSWER_ONLY) for _ in session_records
        ]
    else:
        session_evals = [None] * len(session_records)
    qa_evals: list[dict[str, object] | None] = [None] * len(qa_records)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures: dict[Any, tuple[str, int]] = {}
        if not answer_only:
            for idx, srec in enumerate(session_records):
                fut = pool.submit(
                    _evaluate_session_bundle,
                    srec,
                    memory_accuracy_itemwise=memory_accuracy_itemwise,
                )
                futures[fut] = ("session", idx)
        qa_fn = partial(evaluate_checkpoint_qa, answer_only=True) if answer_only else evaluate_checkpoint_qa
        for idx, qrec in enumerate(qa_records):
            fut = pool.submit(qa_fn, qrec)
            futures[fut] = ("qa", idx)

        done_sessions = 0 if answer_only else 0
        done_qa = 0
        for fut in as_completed(futures):
            kind, idx = futures[fut]
            try:
                result = fut.result()
            except Exception as e:
                result = {"error": str(e)}
            if kind == "session":
                session_evals[idx] = result
                done_sessions += 1
                if done_sessions % 10 == 0 or done_sessions == len(session_records):
                    print(
                        f"{prefix}Sessions: {done_sessions}/{len(session_records)} done",
                        flush=True,
                    )
            else:
                qa_evals[idx] = result
                done_qa += 1
                if done_qa % 20 == 0 or done_qa == len(qa_records):
                    print(
                        f"{prefix}QA: {done_qa}/{len(qa_records)} done",
                        flush=True,
                    )

    return session_evals, qa_evals, time.perf_counter() - t_start


def _sample_ids_in_order(
    session_records: list[PipelineSessionRecord],
    qa_records: list[PipelineCheckpointQARecord],
) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for rec in [*session_records, *qa_records]:
        if rec.sample_id not in seen:
            seen.add(rec.sample_id)
            ordered.append(rec.sample_id)
    return ordered


def _records_for_sample(
    sample_id: str,
    session_records: list[PipelineSessionRecord],
    qa_records: list[PipelineCheckpointQARecord],
) -> tuple[list[PipelineSessionRecord], list[PipelineCheckpointQARecord]]:
    return (
        [rec for rec in session_records if rec.sample_id == sample_id],
        [rec for rec in qa_records if rec.sample_id == sample_id],
    )


def run_eval(
    config: EvalConfig,
    *,
    load_domain_bundle: Callable[[Path], EvalBundle] | None = None,
    create_adapter: Callable[[str], MemoryAdapter] | None = None,
    answer_fn: Callable | None = None,
    max_eval_workers: int = 5,
    eval_only: bool = False,
) -> None:
    """Load data and persist completed sample results as they finish."""
    patch_openai_chat_completions()
    if config.dry_run:
        return

    t_total_start = time.perf_counter()
    t_pipeline_elapsed = 0.0
    t_eval_elapsed = 0.0

    out = config.output_dir
    out.mkdir(parents=True, exist_ok=True)

    itemwise = bool(config.memory_accuracy_itemwise)
    answer_only_eval = is_answer_only_eval_baseline(config.baseline)
    if answer_only_eval:
        print(
            f"[Eval] Answer-only mode for {config.baseline!r}: "
            "checkpoint QA labels + F1/BLEU only (no session memory or retrieval judge)."
        )
        itemwise = False
    elif itemwise:
        print(
            f"[Eval] Session stage: batch recall/correctness + "
            f"itemwise memory_accuracy (extra LLM calls per delta memory)."
        )

    session_records: list[PipelineSessionRecord] = []
    qa_records: list[PipelineCheckpointQARecord] = []
    session_evals: list[dict[str, object] | None] = []
    qa_evals: list[dict[str, object] | None] = []
    completed_samples: list[str] = []

    if eval_only:
        # --- Resume from checkpoint ---
        print(f"[Eval-only] Loading pipeline checkpoint from {out}")
        loaded_sessions, loaded_qa = _load_pipeline_checkpoint(out)
        print(f"[Eval-only] Loaded {len(loaded_sessions)} sessions + {len(loaded_qa)} QA records")

        sample_ids = _sample_ids_in_order(loaded_sessions, loaded_qa)
        print(f"[Eval-only] Evaluating {len(sample_ids)} sample(s)")
        for idx, sample_id in enumerate(sample_ids):
            sample_sessions, sample_qa = _records_for_sample(sample_id, loaded_sessions, loaded_qa)
            print(f"  Sample {idx + 1}/{len(sample_ids)}: {sample_id}")
            sample_session_evals, sample_qa_evals, sample_eval_seconds = _evaluate_records(
                sample_sessions,
                sample_qa,
                baseline=config.baseline,
                max_eval_workers=max_eval_workers,
                memory_accuracy_itemwise=itemwise,
                label=sample_id,
            )
            t_eval_elapsed += sample_eval_seconds

            session_records.extend(sample_sessions)
            qa_records.extend(sample_qa)
            session_evals.extend(sample_session_evals)
            qa_evals.extend(sample_qa_evals)
            completed_samples.append(sample_id)

            sample_dir = _sample_result_dir(out, sample_id)
            _write_evaluated_outputs(
                sample_dir,
                baseline=config.baseline,
                session_records=sample_sessions,
                qa_records=sample_qa,
                session_evals=sample_session_evals,
                qa_evals=sample_qa_evals,
                pipeline_seconds=0.0,
                eval_seconds=sample_eval_seconds,
                total_seconds=sample_eval_seconds,
            )
            _write_json(
                sample_dir / "sample_status.json",
                {
                    "sample_id": sample_id,
                    "status": "DONE",
                    "sessions": len(sample_sessions),
                    "qa": len(sample_qa),
                    "pipeline_seconds": 0.0,
                    "eval_seconds": round(sample_eval_seconds, 2),
                    "total_seconds": round(sample_eval_seconds, 2),
                },
            )
            _write_partial_outputs(
                out,
                baseline=config.baseline,
                session_records=session_records,
                qa_records=qa_records,
                session_evals=session_evals,
                qa_evals=qa_evals,
                pipeline_seconds=t_pipeline_elapsed,
                eval_seconds=t_eval_elapsed,
                total_seconds=time.perf_counter() - t_total_start,
                completed_samples=completed_samples,
            )
            print(f"  [Sample saved] {sample_id} -> {sample_dir}", flush=True)
    else:
        # --- Pipeline + eval per sample. The adapter is stateful within one sample. ---
        adapter_factory = create_adapter or _default_create_adapter
        loader = load_domain_bundle or _auto_dataset_loader(config.dataset_path)
        bundle = loader(config.dataset_path)
        samples = bundle.samples[:1] if config.smoke else bundle.samples
        if config.sample_index is not None:
            if config.sample_index < 1 or config.sample_index > len(bundle.samples):
                raise ValueError(
                    f"sample_index must be in 1..{len(bundle.samples)}, got {config.sample_index}"
                )
            samples = (bundle.samples[config.sample_index - 1],)
        _answer = answer_fn if answer_fn is not None else build_default_answer_fn(
            config.answer_evidence_mode
        )

        print(f"[Pipeline] Running {len(samples)} sample(s) with baseline={config.baseline}")
        baseline_storage_root = out / "_baseline_storage"
        for i, sample in enumerate(samples):
            print(f"  Sample {i + 1}/{len(samples)}: {sample.sample_id}")
            sample_dir = _sample_result_dir(out, sample.sample_id)
            status_payload = _sample_status_payload(sample_dir)
            status = str(status_payload.get("status") or "")

            if status in {"PIPELINE_DONE", "DONE"}:
                try:
                    sample_sessions, sample_qa = _load_pipeline_checkpoint(sample_dir)
                except Exception as exc:
                    print(f"  [Resume] Could not load sample pipeline ({exc}); rerunning.", flush=True)
                else:
                    if status == "DONE":
                        print(f"  [Resume] Re-judging completed sample: {sample.sample_id}", flush=True)
                    sample_pipeline_seconds = float(status_payload.get("pipeline_seconds") or 0.0)
                    print(
                        f"  [Resume] Loaded sample pipeline: "
                        f"{len(sample_sessions)} sessions + {len(sample_qa)} QA",
                        flush=True,
                    )
                    t_pipeline_elapsed += sample_pipeline_seconds

                    sample_session_evals, sample_qa_evals, sample_eval_seconds = _evaluate_records(
                        sample_sessions,
                        sample_qa,
                        baseline=config.baseline,
                        max_eval_workers=max_eval_workers,
                        memory_accuracy_itemwise=itemwise,
                        label=sample.sample_id,
                    )
                    t_eval_elapsed += sample_eval_seconds

                    session_records.extend(sample_sessions)
                    qa_records.extend(sample_qa)
                    session_evals.extend(sample_session_evals)
                    qa_evals.extend(sample_qa_evals)
                    completed_samples.append(sample.sample_id)

                    sample_total_seconds = sample_pipeline_seconds + sample_eval_seconds
                    _write_evaluated_outputs(
                        sample_dir,
                        baseline=config.baseline,
                        session_records=sample_sessions,
                        qa_records=sample_qa,
                        session_evals=sample_session_evals,
                        qa_evals=sample_qa_evals,
                        pipeline_seconds=sample_pipeline_seconds,
                        eval_seconds=sample_eval_seconds,
                        total_seconds=sample_total_seconds,
                    )
                    _write_json(
                        sample_dir / "sample_status.json",
                        {
                            "sample_id": sample.sample_id,
                            "sample_uuid": sample.uuid,
                            "status": "DONE",
                            "sessions": len(sample_sessions),
                            "qa": len(sample_qa),
                            "pipeline_seconds": round(sample_pipeline_seconds, 2),
                            "eval_seconds": round(sample_eval_seconds, 2),
                            "total_seconds": round(sample_total_seconds, 2),
                        },
                    )
                    _write_partial_outputs(
                        out,
                        baseline=config.baseline,
                        session_records=session_records,
                        qa_records=qa_records,
                        session_evals=session_evals,
                        qa_evals=qa_evals,
                        pipeline_seconds=t_pipeline_elapsed,
                        eval_seconds=t_eval_elapsed,
                        total_seconds=time.perf_counter() - t_total_start,
                        completed_samples=completed_samples,
                    )
                    print(f"  [Sample saved] {sample.sample_id} -> {sample_dir}", flush=True)
                    continue

            # Isolate per-sample baseline storage under this run's output dir so
            # LanceDB / omni_memory_data don't pollute cwd or bleed across runs.
            sample_storage = baseline_storage_root / sample.sample_id
            sample_storage.mkdir(parents=True, exist_ok=True)
            os.environ["SIMPLEMEM_LANCEDB_PATH"] = str(sample_storage / "lancedb")
            os.environ["OMNI_MEMORY_DATA_DIR"] = str(sample_storage / "omni_memory")
            os.environ["EVAL_FRAMEWORK_BASELINE_STORAGE_DIR"] = str(sample_storage)
            adapter = adapter_factory(config.baseline)
            t_sample_pipeline_start = time.perf_counter()
            sess, qa = run_eval_sample(
                adapter,
                sample,
                answer_fn=_answer,
                max_sessions=config.max_sessions,
                answer_evidence_mode=config.answer_evidence_mode,
            )
            sample_pipeline_seconds = time.perf_counter() - t_sample_pipeline_start
            t_pipeline_elapsed += sample_pipeline_seconds

            sample_sessions = list(sess)
            sample_qa = list(qa)
            session_records.extend(sample_sessions)
            qa_records.extend(sample_qa)

            _write_pipeline_checkpoint(out, session_records, qa_records)
            _write_sample_pipeline_result(
                out,
                sample.sample_id,
                sample_sessions,
                sample_qa,
                pipeline_seconds=sample_pipeline_seconds,
            )
            print(
                f"  [Checkpoint] Saved sample pipeline: "
                f"{len(sample_sessions)} sessions + {len(sample_qa)} QA",
                flush=True,
            )

            sample_session_evals, sample_qa_evals, sample_eval_seconds = _evaluate_records(
                sample_sessions,
                sample_qa,
                baseline=config.baseline,
                max_eval_workers=max_eval_workers,
                memory_accuracy_itemwise=itemwise,
                label=sample.sample_id,
            )
            t_eval_elapsed += sample_eval_seconds

            session_evals.extend(sample_session_evals)
            qa_evals.extend(sample_qa_evals)
            completed_samples.append(sample.sample_id)

            sample_total_seconds = sample_pipeline_seconds + sample_eval_seconds
            sample_dir = _sample_result_dir(out, sample.sample_id)
            _write_evaluated_outputs(
                sample_dir,
                baseline=config.baseline,
                session_records=sample_sessions,
                qa_records=sample_qa,
                session_evals=sample_session_evals,
                qa_evals=sample_qa_evals,
                pipeline_seconds=sample_pipeline_seconds,
                eval_seconds=sample_eval_seconds,
                total_seconds=sample_total_seconds,
            )
            _write_json(
                sample_dir / "sample_status.json",
                {
                    "sample_id": sample.sample_id,
                    "sample_uuid": sample.uuid,
                    "status": "DONE",
                    "sessions": len(sample_sessions),
                    "qa": len(sample_qa),
                    "pipeline_seconds": round(sample_pipeline_seconds, 2),
                    "eval_seconds": round(sample_eval_seconds, 2),
                    "total_seconds": round(sample_total_seconds, 2),
                },
            )
            _write_partial_outputs(
                out,
                baseline=config.baseline,
                session_records=session_records,
                qa_records=qa_records,
                session_evals=session_evals,
                qa_evals=qa_evals,
                pipeline_seconds=t_pipeline_elapsed,
                eval_seconds=t_eval_elapsed,
                total_seconds=time.perf_counter() - t_total_start,
                completed_samples=completed_samples,
            )
            print(f"  [Sample saved] {sample.sample_id} -> {sample_dir}", flush=True)

    # --- Stage 3: Aggregate + write ---
    t_total_elapsed = time.perf_counter() - t_total_start
    agg = _write_evaluated_outputs(
        out,
        baseline=config.baseline,
        session_records=session_records,
        qa_records=qa_records,
        session_evals=session_evals,
        qa_evals=qa_evals,
        pipeline_seconds=t_pipeline_elapsed,
        eval_seconds=t_eval_elapsed,
        total_seconds=t_total_elapsed,
    )

    print(f"\n[Done] Results written to {out}")
    print(f"  Timing: pipeline={t_pipeline_elapsed:.1f}s  eval={t_eval_elapsed:.1f}s  total={t_total_elapsed:.1f}s")
    print(f"  Aggregate: {json.dumps(agg, indent=2)}")


def _default_data150_baselines() -> list[str]:
    return [
        "Qwen3-VL-Embedding-8B",
        "UniversalRAGMemory",
        "A-Mem",
        "MGMemory",
        "SimpleMem",
        "Omni-SimpleMem",
        "M2A",
        "ViLoMem",
        "MIRIX",
        "AUGUSTUSMemory",
    ]


def _domain_slug_for_sample(sample: Any) -> str:
    # WorldMemArena samples carry a _subcategory attribute set by the loader.
    subcat = getattr(sample, "_subcategory", None)
    if subcat:
        return _safe_path_component(subcat.replace("/", "__"))

    raw = str(getattr(sample, "uuid", "") or getattr(sample, "sample_id", ""))
    known_prefixes = (
        # Legacy converted/all sample_id prefixes
        ("arena_excel_", "agent__arena__excel"),
        ("arena_file_mgmt_", "agent__arena__file_mgmt"),
        ("arena_image_edit_", "agent__arena__image_edit"),
        ("arena_web_", "agent__arena__web"),
        ("arena_word_docs_", "agent__arena__word_docs"),
        ("eb_alfred_base_", "agent__eb_alfred__base"),
        ("eb_alfred_common_sense_", "agent__eb_alfred__common_sense"),
        ("eb_alfred_complex_instruction_", "agent__eb_alfred__complex_instruction"),
        ("eb_alfred_long_horizon_", "agent__eb_alfred__long_horizon"),
        ("eb_alfred_visual_appearance_", "agent__eb_alfred__visual_appearance"),
        ("eb_nav_base_", "agent__eb_nav__base"),
        ("eb_nav_common_sense_", "agent__eb_nav__common_sense"),
        ("eb_nav_complex_instruction_", "agent__eb_nav__complex_instruction"),
        ("eb_nav_long_horizon_", "agent__eb_nav__long_horizon"),
        ("eb_nav_visual_appearance_", "agent__eb_nav__visual_appearance"),
        ("vab_css_", "agent__vab__css"),
        ("vab_minecraft_", "agent__vab__minecraft"),
        ("vab_mobile_", "agent__vab__mobile"),
        ("vab_omnigibson_", "agent__vab__omnigibson"),
        ("vab_webarena_lite_", "agent__vab__webarena-lite"),
        ("domain_a_v2_academic_", "lifelong__domain_a_v2__academic"),
        ("domain_a_v2_education_", "lifelong__domain_a_v2__education"),
        ("domain_a_v2_finance_", "lifelong__domain_a_v2__finance"),
        ("domain_a_v2_health_", "lifelong__domain_a_v2__health"),
        ("domain_a_v2_software_", "lifelong__domain_a_v2__software"),
        ("domain_a_v2_startup_", "lifelong__domain_a_v2__startup"),
        ("halumem_v2_", "lifelong__domain_b_v2"),
        # WorldMemArena sample_id prefixes (new naming scheme)
        ("excel_", "agent__arena__excel"),
        ("file_mgmt_", "agent__arena__file_mgmt"),
        ("image_edit_", "agent__arena__image_edit"),
        ("web_", "agent__arena__web"),
        ("word_docs_", "agent__arena__word_docs"),
        ("css_", "agent__vab__css"),
        ("mobile_", "agent__vab__mobile"),
        ("webarena_lite_", "agent__vab__webarena-lite"),
        ("eb_alfred_base_", "agent__eb_alfred__base"),
        ("eb_alfred_common_sense_", "agent__eb_alfred__common_sense"),
        ("eb_alfred_complex_instruction_", "agent__eb_alfred__complex_instruction"),
        ("eb_alfred_long_horizon_", "agent__eb_alfred__long_horizon"),
        ("eb_alfred_visual_appearance_", "agent__eb_alfred__visual_appearance"),
        ("eb_nav_base_", "agent__eb_nav__base"),
        ("eb_nav_common_sense_", "agent__eb_nav__common_sense"),
        ("eb_nav_complex_instruction_", "agent__eb_nav__complex_instruction"),
        ("eb_nav_long_horizon_", "agent__eb_nav__long_horizon"),
        ("eb_nav_visual_appearance_", "agent__eb_nav__visual_appearance"),
        ("minecraft_", "agent__vab__minecraft"),
        ("omnigibson_", "agent__vab__omnigibson"),
        ("academic_", "lifelong__domain_a_v2__academic"),
        ("education_", "lifelong__domain_a_v2__education"),
        ("finance_", "lifelong__domain_a_v2__finance"),
        ("health_", "lifelong__domain_a_v2__health"),
        ("software_", "lifelong__domain_a_v2__software"),
        ("startup_", "lifelong__domain_a_v2__startup"),
        ("personal_", "lifelong__domain_b_v2"),
    )
    for prefix, slug in known_prefixes:
        if raw.startswith(prefix):
            return slug
    return _safe_path_component(raw)


def _read_progress_log(path: Path, baselines: list[str], count: int) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {str(i): {baseline: "pending" for baseline in baselines} for i in range(1, count + 1)}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        existing_baselines = [field for field in (reader.fieldnames or []) if field != "id"]
        rows = list(reader)
    log_baselines = list(dict.fromkeys([*existing_baselines, *baselines]))
    progress: dict[str, dict[str, str]] = {}
    for row in rows:
        idx = str(row.get("id") or "").strip()
        if not idx:
            continue
        progress[idx] = {
            baseline: str(row.get(baseline) or "pending") for baseline in log_baselines
        }
    for i in range(1, count + 1):
        progress.setdefault(str(i), {})
        for baseline in log_baselines:
            progress[str(i)].setdefault(baseline, "pending")
    return progress


def _read_progress_log_for_ids(
    path: Path,
    baselines: list[str],
    sample_ids: list[int],
) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {str(i): {baseline: "pending" for baseline in baselines} for i in sample_ids}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        existing_baselines = [field for field in (reader.fieldnames or []) if field != "id"]
        rows = list(reader)
    log_baselines = list(dict.fromkeys([*existing_baselines, *baselines]))
    existing: dict[str, dict[str, str]] = {}
    for row in rows:
        idx = str(row.get("id") or "").strip()
        if not idx:
            continue
        existing[idx] = {
            baseline: str(row.get(baseline) or "pending") for baseline in log_baselines
        }
    progress: dict[str, dict[str, str]] = {}
    for sample_id in sample_ids:
        idx = str(sample_id)
        progress[idx] = existing.get(idx, {})
        for baseline in log_baselines:
            progress[idx].setdefault(baseline, "pending")
    return progress


def _write_progress_log(path: Path, baselines: list[str], progress: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", *baselines])
        writer.writeheader()
        for idx in sorted(progress, key=lambda x: int(x)):
            row = {"id": idx}
            row.update({baseline: progress[idx].get(baseline, "pending") for baseline in baselines})
            writer.writerow(row)
    os.replace(tmp, path)


def _data150_command(
    *,
    dataset: Path,
    dataset_type: str,
    baseline: str,
    output_dir: Path,
    sample_index: int,
    max_eval_workers: int,
    answer_evidence_mode: str,
    max_sessions: int | None,
    memory_accuracy_itemwise: bool,
    eval_only: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "eval_framework.cli",
        "--dataset",
        str(dataset),
        "--dataset-type",
        dataset_type,
        "--baseline",
        baseline,
        "--output-dir",
        str(output_dir),
        "--sample-index",
        str(sample_index),
        "--max-eval-workers",
        str(max_eval_workers),
        "--answer-evidence-mode",
        answer_evidence_mode,
    ]
    if max_sessions:
        cmd.extend(["--max-sessions", str(max_sessions)])
    if memory_accuracy_itemwise:
        cmd.append("--memory-accuracy-itemwise")
    if eval_only:
        cmd.append("--eval-only")
    return cmd


def _has_pipeline_checkpoint(output_dir: Path) -> bool:
    return (output_dir / _CHECKPOINT_SESSIONS).exists() and (output_dir / _CHECKPOINT_QA).exists()


def _reconcile_orphan_running(
    progress: dict[str, dict[str, str]],
    output_root: Path,
    samples: list,
    baselines: list[str],
) -> int:
    """Reset 'running' cells that lack a finished aggregate_metrics back to 'pending'.

    This cleans up stale state left behind when a driver is killed mid-run.
    Cells that *do* have a valid aggregate_metrics.json on disk are set to 'done'.
    Returns the number of cells that were reset.
    """
    reset_count = 0
    for idx_str, bl_map in progress.items():
        for bl_name in baselines:
            if bl_map.get(bl_name) != "running":
                continue
            idx = int(idx_str)
            if idx < 1 or idx > len(samples):
                continue
            sample = samples[idx - 1]
            domain = _domain_slug_for_sample(sample)
            baseline_dir = output_root / f"{idx}_{domain}" / _safe_path_component(bl_name)
            disk_status = _data150_output_status(baseline_dir)
            if disk_status == "done":
                bl_map[bl_name] = "done"
            else:
                bl_map[bl_name] = "pending"
                reset_count += 1
    return reset_count


def _data150_output_status(output_dir: Path) -> str | None:
    metrics_path = output_dir / "aggregate_metrics.json"
    if not metrics_path.exists():
        return None
    try:
        json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return "failed"
    return "done"


def _run_logged_command(cmd: list[str], output_dir: Path, env: dict[str, str] | None = None) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"
    with log_path.open("a", encoding="utf-8") as log_fh:
        log_fh.write(f"\n$ {' '.join(cmd)}\n")
        if env and "CUDA_VISIBLE_DEVICES" in env:
            log_fh.write(f"# CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']!r}\n")
        proc = subprocess.run(
            cmd,
            cwd=str(Path(__file__).resolve().parents[1]),
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return proc.returncode


def _sample_match_key(sample: Any) -> str:
    return str(getattr(sample, "sample_id", "") or getattr(sample, "uuid", "")).strip()


def _subset_source_sample_pairs(source_samples: list[Any], subset_samples: list[Any]) -> list[tuple[int, Any]]:
    source_by_key: dict[str, tuple[int, Any]] = {}
    for source_index, sample in enumerate(source_samples, start=1):
        key = _sample_match_key(sample)
        if key:
            source_by_key.setdefault(key, (source_index, sample))
    pairs: list[tuple[int, Any]] = []
    missing: list[str] = []
    for subset in subset_samples:
        key = _sample_match_key(subset)
        if key not in source_by_key:
            missing.append(key or "<empty-key>")
            continue
        pairs.append(source_by_key[key])
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"{len(missing)} data50 samples were not found in data150, first missing: {preview}")
    return pairs


def run_data150_gpt_batch(ns: argparse.Namespace) -> None:
    dataset = Path(ns.dataset or "eval_framework/converted/data150").expanduser().resolve()
    output_dir_arg = ns.output_dir
    if not output_dir_arg or output_dir_arg == "eval_framework/results":
        output_dir_arg = "eval_framework/exp_log/data150_gpt"
    output_root = Path(output_dir_arg).expanduser().resolve()
    loader = _resolve_loader(ns)
    bundle = loader(dataset)
    total = min(int(ns.data150_count or 150), len(bundle.samples))
    baselines = list(ns.baselines or _default_data150_baselines())
    max_workers = int(ns.max_baseline_workers or len(baselines) or 1)
    per_baseline_workers = max(1, int(ns.per_baseline_workers or 1))
    baseline_cuda_visible_devices = os.getenv("DATA150_BASELINE_CUDA_VISIBLE_DEVICES", "")
    if ns.dry_run:
        print(
            json.dumps(
                {
                    "dataset": str(dataset),
                    "output_dir": str(output_root),
                    "samples": total,
                    "baselines": baselines,
                    "max_baseline_workers": max_workers,
                    "per_baseline_workers": per_baseline_workers,
                    "baseline_cuda_visible_devices": baseline_cuda_visible_devices,
                },
                indent=2,
            )
        )
        return
    progress_path = output_root / getattr(ns, "progress_filename", "log.csv")
    progress = _read_progress_log(progress_path, baselines, total)
    log_baselines = list(next(iter(progress.values())).keys()) if progress else baselines

    samples_for_reconcile = list(bundle.samples[:total])
    n_reset = _reconcile_orphan_running(progress, output_root, samples_for_reconcile, log_baselines)
    if n_reset:
        print(f"[data150_gpt] reconciled {n_reset} orphan 'running' cell(s) -> 'pending'", flush=True)

    progress_lock = threading.Lock()
    _write_progress_log(progress_path, log_baselines, progress)

    print(
        f"[data150_gpt] dataset={dataset} output={output_root} "
        f"samples={total} baselines={len(baselines)} workers={max_workers}"
    )

    samples = list(bundle.samples[:total])
    only_indices_raw = list(getattr(ns, "data150_sample_indices", None) or [])
    only_indices: set[int] | None = (
        {int(i) for i in only_indices_raw} if only_indices_raw else None
    )
    if only_indices_raw:
        seen: set[int] = set()
        ordered_indices: list[int] = []
        for v in only_indices_raw:
            iv = int(v)
            if 1 <= iv <= total and iv not in seen:
                seen.add(iv)
                ordered_indices.append(iv)
    else:
        ordered_indices = list(range(1, total + 1))

    def run_baseline_sample(baseline: str, sample_index: int, sample: Any) -> None:
        domain = _domain_slug_for_sample(sample)
        domain_dir = output_root / f"{sample_index}_{domain}"
        baseline_dir = domain_dir / _safe_path_component(baseline)
        with progress_lock:
            existing_status = _data150_output_status(baseline_dir)
            if existing_status == "done":
                progress[str(sample_index)][baseline] = existing_status
                _write_progress_log(progress_path, log_baselines, progress)
                print(f"  [skip:{existing_status}] id={sample_index} {baseline}", flush=True)
                return
            progress[str(sample_index)][baseline] = "running"
            _write_progress_log(progress_path, log_baselines, progress)
        cmd = _data150_command(
            dataset=dataset,
            dataset_type=ns.dataset_type,
            baseline=baseline,
            output_dir=baseline_dir,
            sample_index=sample_index,
            max_eval_workers=ns.max_eval_workers,
            answer_evidence_mode=ns.answer_evidence_mode,
            max_sessions=ns.max_sessions,
            memory_accuracy_itemwise=ns.memory_accuracy_itemwise,
            eval_only=ns.eval_only or _has_pipeline_checkpoint(baseline_dir),
        )
        print(f"[data150_gpt] id={sample_index}/{total} domain={domain} baseline={baseline}", flush=True)
        child_env = os.environ.copy()
        child_env["CUDA_VISIBLE_DEVICES"] = baseline_cuda_visible_devices
        returncode = _run_logged_command(cmd, baseline_dir, env=child_env)
        with progress_lock:
            output_status = _data150_output_status(baseline_dir)
            progress[str(sample_index)][baseline] = output_status or ("done" if returncode == 0 else "failed")
            _write_progress_log(progress_path, log_baselines, progress)
        print(f"  [{progress[str(sample_index)][baseline]}] id={sample_index} {baseline}", flush=True)

    def run_baseline_stream(baseline: str) -> None:
        with ThreadPoolExecutor(max_workers=per_baseline_workers) as sample_pool:
            futures = [
                sample_pool.submit(run_baseline_sample, baseline, idx, samples[idx - 1])
                for idx in ordered_indices
            ]
            for fut in as_completed(futures):
                fut.result()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(run_baseline_stream, baseline) for baseline in baselines]
        for fut in as_completed(futures):
            fut.result()


def run_data50_gpt_batch(ns: argparse.Namespace) -> None:
    subset_dataset = Path(ns.dataset or "eval_framework/converted/data50").expanduser().resolve()
    source_dataset = Path(ns.data50_source_dataset or "eval_framework/converted/data150").expanduser().resolve()
    output_dir_arg = ns.output_dir
    if not output_dir_arg or output_dir_arg == "eval_framework/results":
        output_dir_arg = "eval_framework/exp_log/data50_gpt"
    output_root = Path(output_dir_arg).expanduser().resolve()
    loader = _resolve_loader(ns)
    subset_bundle = loader(subset_dataset)
    source_bundle = loader(source_dataset)
    selected = _subset_source_sample_pairs(list(source_bundle.samples), list(subset_bundle.samples))
    total = min(int(ns.data50_count or 50), len(selected))
    selected = selected[:total]
    source_ids = [source_index for source_index, _sample in selected]
    baselines = list(ns.baselines or _default_data150_baselines())
    max_workers = int(ns.max_baseline_workers or len(baselines) or 1)
    per_baseline_workers = max(1, int(ns.per_baseline_workers or 1))
    baseline_cuda_visible_devices = os.getenv("DATA50_BASELINE_CUDA_VISIBLE_DEVICES", "")
    if ns.dry_run:
        print(
            json.dumps(
                {
                    "subset_dataset": str(subset_dataset),
                    "source_dataset": str(source_dataset),
                    "output_dir": str(output_root),
                    "samples": total,
                    "source_ids": source_ids,
                    "baselines": baselines,
                    "max_baseline_workers": max_workers,
                    "per_baseline_workers": per_baseline_workers,
                    "baseline_cuda_visible_devices": baseline_cuda_visible_devices,
                },
                indent=2,
            )
        )
        return
    progress_path = output_root / getattr(ns, "progress_filename", "log.csv")
    progress = _read_progress_log_for_ids(progress_path, baselines, source_ids)
    log_baselines = list(next(iter(progress.values())).keys()) if progress else baselines
    progress_lock = threading.Lock()
    _write_progress_log(progress_path, log_baselines, progress)

    print(
        f"[data50_gpt] subset={subset_dataset} source={source_dataset} output={output_root} "
        f"samples={total} baselines={len(baselines)} workers={max_workers}"
    )
    print(f"[data50_gpt] source_ids={source_ids}", flush=True)

    def run_baseline_sample(
        baseline: str,
        ordinal: int,
        source_index: int,
        source_sample: Any,
    ) -> None:
        domain = _domain_slug_for_sample(source_sample)
        domain_dir = output_root / f"{source_index}_{domain}"
        baseline_dir = domain_dir / _safe_path_component(baseline)
        with progress_lock:
            existing_status = _data150_output_status(baseline_dir)
            if existing_status == "done":
                progress[str(source_index)][baseline] = existing_status
                _write_progress_log(progress_path, log_baselines, progress)
                print(f"  [skip:{existing_status}] data50={ordinal}/{total} id={source_index} {baseline}", flush=True)
                return
            progress[str(source_index)][baseline] = "pending"
            _write_progress_log(progress_path, log_baselines, progress)
        cmd = _data150_command(
            dataset=source_dataset,
            dataset_type=ns.dataset_type,
            baseline=baseline,
            output_dir=baseline_dir,
            sample_index=source_index,
            max_eval_workers=ns.max_eval_workers,
            answer_evidence_mode=ns.answer_evidence_mode,
            max_sessions=ns.max_sessions,
            memory_accuracy_itemwise=ns.memory_accuracy_itemwise,
            eval_only=ns.eval_only or _has_pipeline_checkpoint(baseline_dir),
        )
        print(
            f"[data50_gpt] data50={ordinal}/{total} id={source_index} "
            f"domain={domain} baseline={baseline}",
            flush=True,
        )
        child_env = os.environ.copy()
        child_env["CUDA_VISIBLE_DEVICES"] = baseline_cuda_visible_devices
        returncode = _run_logged_command(cmd, baseline_dir, env=child_env)
        with progress_lock:
            output_status = _data150_output_status(baseline_dir)
            progress[str(source_index)][baseline] = output_status or ("done" if returncode == 0 else "failed")
            _write_progress_log(progress_path, log_baselines, progress)
        print(f"  [{progress[str(source_index)][baseline]}] data50={ordinal}/{total} id={source_index} {baseline}", flush=True)

    def run_baseline_stream(baseline: str) -> None:
        with ThreadPoolExecutor(max_workers=per_baseline_workers) as sample_pool:
            futures = [
                sample_pool.submit(run_baseline_sample, baseline, ordinal, source_index, sample)
                for ordinal, (source_index, sample) in enumerate(selected, start=1)
            ]
            for fut in as_completed(futures):
                fut.result()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(run_baseline_stream, baseline) for baseline in baselines]
        for fut in as_completed(futures):
            fut.result()


def run_data150_openclaw_harness_batch(ns: argparse.Namespace) -> None:
    dataset = Path(ns.dataset or "eval_framework/converted/data150").expanduser().resolve()
    output_dir_arg = ns.output_dir
    if not output_dir_arg or output_dir_arg == "eval_framework/results":
        output_dir_arg = "eval_framework/exp_log/data150_gpt"
    output_root = Path(output_dir_arg).expanduser().resolve()
    loader = _resolve_loader(ns)
    bundle = loader(dataset)
    total = min(int(ns.data150_count or 150), len(bundle.samples))
    from eval_framework.config import _harness_entries  # noqa: PLC0415

    cfg_keys = [k for k, _ in _harness_entries()]
    requested_keys = list(getattr(ns, "harness_keys", None) or []) or cfg_keys
    unknown = [k for k in requested_keys if k not in cfg_keys]
    if unknown:
        raise SystemExit(
            f"--harness-keys contains unknown harness keys: {unknown}. "
            f"Available in harness_config.yaml: {cfg_keys}"
        )
    harness_keys = requested_keys
    baselines = [f"Harness-{key.replace('_', '-')}" for key in harness_keys]
    progress_path = output_root / "log_harness.csv"
    progress = _read_progress_log(progress_path, baselines, total)
    log_baselines = list(next(iter(progress.values())).keys()) if progress else baselines
    progress_lock = threading.Lock()
    _write_progress_log(progress_path, log_baselines, progress)
    samples = list(bundle.samples[:total])
    only_indices_raw = list(getattr(ns, "data150_sample_indices", None) or [])
    only_indices: set[int] | None = (
        {int(i) for i in only_indices_raw} if only_indices_raw else None
    )
    if only_indices_raw:
        seen: set[int] = set()
        ordered_indices: list[int] = []
        for v in only_indices_raw:
            iv = int(v)
            if 1 <= iv <= total and iv not in seen:
                seen.add(iv)
                ordered_indices.append(iv)
    else:
        ordered_indices = list(range(1, total + 1))
    per_baseline_workers = max(1, int(ns.per_baseline_workers or 1))
    max_workers = int(ns.max_baseline_workers or len(baselines) or 1)
    if ns.dry_run:
        print(
            json.dumps(
                {
                    "dataset": str(dataset),
                    "output_dir": str(output_root),
                    "samples": total,
                    "harness_keys": harness_keys,
                    "baselines": baselines,
                    "max_baseline_workers": max_workers,
                    "per_baseline_workers": per_baseline_workers,
                    "only_indices": sorted(only_indices) if only_indices else None,
                },
                indent=2,
            )
        )
        return

    print(
        f"[data150_harness] dataset={dataset} output={output_root} "
        f"samples={total} baselines={len(baselines)} workers={max_workers}"
    )
    print(f"[data150_harness] harness_keys={harness_keys}", flush=True)

    def run_harness_sample(key: str, baseline: str, sample_index: int, sample: Any) -> None:
        domain = _domain_slug_for_sample(sample)
        run_dir = output_root / f"{sample_index}_{domain}_harness_{key}"
        with progress_lock:
            existing_status = _data150_output_status(run_dir)
            if existing_status == "done":
                progress[str(sample_index)][baseline] = existing_status
                _write_progress_log(progress_path, log_baselines, progress)
                print(f"  [skip:{existing_status}] id={sample_index} {baseline}", flush=True)
                return
            progress[str(sample_index)][baseline] = "running"
            _write_progress_log(progress_path, log_baselines, progress)
        cmd = _data150_command(
            dataset=dataset,
            dataset_type=ns.dataset_type,
            baseline=baseline,
            output_dir=run_dir,
            sample_index=sample_index,
            max_eval_workers=ns.max_eval_workers,
            answer_evidence_mode=ns.answer_evidence_mode,
            max_sessions=ns.max_sessions,
            memory_accuracy_itemwise=ns.memory_accuracy_itemwise,
            eval_only=ns.eval_only or _has_pipeline_checkpoint(run_dir),
        )
        print(f"[data150_harness] id={sample_index}/{total} domain={domain} baseline={baseline}", flush=True)
        returncode = _run_logged_command(cmd, run_dir)
        with progress_lock:
            output_status = _data150_output_status(run_dir)
            progress[str(sample_index)][baseline] = output_status or ("done" if returncode == 0 else "failed")
            _write_progress_log(progress_path, log_baselines, progress)
        print(f"  [{progress[str(sample_index)][baseline]}] id={sample_index} {baseline}", flush=True)

    def run_harness_stream(key: str, baseline: str) -> None:
        with ThreadPoolExecutor(max_workers=per_baseline_workers) as sample_pool:
            futures = [
                sample_pool.submit(run_harness_sample, key, baseline, idx, samples[idx - 1])
                for idx in ordered_indices
            ]
            for fut in as_completed(futures):
                fut.result()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(run_harness_stream, key, baseline)
            for key, baseline in zip(harness_keys, baselines)
        ]
        for fut in as_completed(futures):
            fut.result()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="eval_framework")
    p.add_argument("--dataset", default=None)
    p.add_argument("--dataset-type", default="worldmemarena",
                    choices=sorted(DATASET_LOADERS.keys()),
                    help="Which loader to dispatch (default worldmemarena).")
    p.add_argument("--split", default="all",
                    choices=["all", "small"],
                    help="Dataset split: all (full 461 samples) or small (150-sample subset).")
    p.add_argument("--baseline", default=None)
    p.add_argument("--baselines", nargs="+", default=None)
    p.add_argument("--output-dir", default="eval_framework/results")
    p.add_argument(
        "--progress-filename",
        default="log.csv",
        help=(
            "Filename (within output-dir) for the progress CSV. Default 'log.csv'. "
            "Set a custom value when running two driver processes that share an "
            "output-dir but track disjoint baseline sets, so they don't collide on "
            "the same progress file."
        ),
    )
    p.add_argument("--run-data150-gpt", action="store_true")
    p.add_argument("--run-data50-gpt", action="store_true")
    p.add_argument("--run-data150-openclaw-harness", action="store_true")
    p.add_argument(
        "--harness-keys",
        nargs="+",
        default=None,
        help=(
            "When set, only run these harness keys (top-level keys in "
            "harness_config.yaml). Default: all keys in the yaml."
        ),
    )
    p.add_argument("--data150-count", type=int, default=150)
    p.add_argument(
        "--data150-sample-indices",
        nargs="+",
        type=int,
        default=None,
        help="When set, only run these 1-based sample indices (subset of "
        "--data150-count). Useful for smoke testing one sample.",
    )
    p.add_argument("--data50-count", type=int, default=50)
    p.add_argument("--data50-source-dataset", default="eval_framework/converted/data150",
                    help="Source dataset used to preserve original data150 sample IDs for --run-data50-gpt.")
    p.add_argument("--max-baseline-workers", type=int, default=None)
    p.add_argument("--per-baseline-workers", type=int, default=3,
                    help="Concurrent samples per baseline in data150 batch mode (default 3).")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--eval-only", action="store_true",
                    help="Skip pipeline, load from checkpoint in output-dir.")
    p.add_argument("--max-eval-workers", type=int, default=20,
                    help="Parallel threads for eval stage (default 20).")
    p.add_argument("--max-sessions", type=int, default=None,
                    help="Truncate each sample to the first N sessions (checkpoints referencing later sessions are skipped).")
    p.add_argument("--sample-index", type=int, default=None,
                    help="Run only the 1-based sample index from the loaded dataset.")
    p.add_argument(
        "--answer-evidence-mode",
        choices=("memory", "session"),
        default="memory",
        help=(
            "Evidence text fed to the answer model: retrieved memory text "
            "(memory, default) or source session dialogue text for retrieved items (session)."
        ),
    )
    p.add_argument(
        "--memory-accuracy-itemwise",
        action="store_true",
        help=(
            "After batch extraction judges, run one LLM call per delta memory: "
            "match this line against all session gold MPs (matches_golden_memories + indices)."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.run_data150_gpt:
        run_data150_gpt_batch(args)
        return
    if args.run_data50_gpt:
        run_data50_gpt_batch(args)
        return
    if args.run_data150_openclaw_harness:
        run_data150_openclaw_harness_batch(args)
        return
    if not args.dataset:
        raise SystemExit("--dataset is required unless --run-data150-gpt or --run-data50-gpt is set")
    if not args.baseline:
        raise SystemExit("--baseline is required unless --run-data150-gpt or --run-data50-gpt is set")
    cfg = config_from_namespace(args)

    if cfg.dry_run:
        print(json.dumps(cfg.to_display_dict(), indent=2))
        return

    eval_only = bool(args.eval_only)

    loader = _resolve_loader(args)
    if not eval_only and not cfg.dataset_path.exists():
        raise SystemExit(f"Dataset path does not exist: {cfg.dataset_path}")

    run_eval(
        cfg,
        load_domain_bundle=loader,
        max_eval_workers=args.max_eval_workers,
        eval_only=eval_only,
    )


if __name__ == "__main__":
    main()
