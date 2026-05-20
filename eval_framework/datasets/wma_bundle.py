"""WorldMemArena eval bundle: core data model for samples, sessions, QA checkpoints."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from eval_framework.datasets.schemas import NormalizedTurn, normalize_turn
from eval_framework.pipeline.gold_state import (
    SessionGoldState,
    build_session_gold_states,
)


@dataclass(frozen=True)
class Stage4Record:
    uuid: str
    sample_id: str
    memory_sessions: tuple[tuple[str, tuple[Mapping[str, Any], ...]], ...]


@dataclass(frozen=True)
class QARecord:
    uuid: str
    sample_id: str
    raw_checkpoints: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class NormalizedCheckpointQuestion:
    question: str
    gold_answer: str
    gold_evidence_memory_ids: tuple[str, ...]
    gold_evidence_contents: tuple[str, ...]
    question_type: str
    question_type_abbrev: str
    difficulty: str


@dataclass(frozen=True)
class NormalizedCheckpoint:
    checkpoint_id: str
    covered_sessions: tuple[str, ...]
    questions: tuple[NormalizedCheckpointQuestion, ...]


@dataclass(frozen=True)
class EvalSession:
    session_id: str
    turns: tuple[NormalizedTurn, ...]


@dataclass(frozen=True)
class EvalSample:
    uuid: str
    sample_id: str
    sessions: tuple[EvalSession, ...]
    stage4: Stage4Record
    qa_record: QARecord
    normalized_checkpoints: tuple[NormalizedCheckpoint, ...]
    session_gold_states: tuple[SessionGoldState, ...]


@dataclass(frozen=True)
class EvalBundle:
    samples: tuple[EvalSample, ...]


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _stage4_from_obj(obj: Mapping[str, Any]) -> Stage4Record:
    blocks: list[tuple[str, tuple[Mapping[str, Any], ...]]] = []
    for ms in obj.get("memory_sessions") or []:
        sid = str(ms.get("session_id", ""))
        pts = ms.get("memory_points") or []
        if not isinstance(pts, list):
            pts = []
        blocks.append((sid, tuple(pts)))
    return Stage4Record(
        uuid=str(obj["uuid"]),
        sample_id=str(obj["sample_id"]),
        memory_sessions=tuple(blocks),
    )


def _qa_from_obj(obj: Mapping[str, Any]) -> QARecord:
    cps = obj.get("checkpoints") or []
    if not isinstance(cps, list):
        cps = []
    return QARecord(
        uuid=str(obj["uuid"]),
        sample_id=str(obj["sample_id"]),
        raw_checkpoints=tuple(cps),
    )


def _normalize_checkpoint_question(
    raw: Mapping[str, Any],
    memory_content_map: Mapping[str, str],
) -> NormalizedCheckpointQuestion:
    evidence = raw.get("evidence") or []
    mem_ids: list[str] = []
    mem_contents: list[str] = []
    if isinstance(evidence, list):
        for item in evidence:
            if not isinstance(item, dict):
                continue
            # Text-anchored evidence: {"memory_id": "mem_..."}
            if "memory_id" in item and item["memory_id"]:
                mid = str(item["memory_id"])
                mem_ids.append(mid)
                content = memory_content_map.get(mid, "")
                if content:
                    mem_contents.append(content)
                continue
            # Visual evidence: {"image_id": "S04_img_4"} — used by VFR/VS/VU.
            # The loader inlines ``image:\nimage_id: SXX_img_Y\nimage_caption: ...``
            # into every turn's text, so the retrieval judge can match the
            # image_id as a substring of retrieval items. Store the id both
            # as the canonical mem_id and as a formatted content line.
            if "image_id" in item and item["image_id"]:
                iid = str(item["image_id"])
                mem_ids.append(iid)
                mem_contents.append(f"image_id: {iid}")
    return NormalizedCheckpointQuestion(
        question=str(raw.get("question", "")),
        gold_answer=str(raw.get("answer", "")),
        gold_evidence_memory_ids=tuple(mem_ids),
        gold_evidence_contents=tuple(mem_contents),
        question_type=str(raw.get("question_type", "")),
        question_type_abbrev=str(raw.get("question_type_abbrev", "")),
        difficulty=str(raw.get("difficulty", "")),
    )


def _normalize_checkpoints(
    raw_checkpoints: tuple[Mapping[str, Any], ...],
    memory_content_map: Mapping[str, str],
) -> tuple[NormalizedCheckpoint, ...]:
    out: list[NormalizedCheckpoint] = []
    for cp in raw_checkpoints:
        qs = cp.get("questions") or []
        if not isinstance(qs, list):
            qs = []
        covered = cp.get("covered_sessions") or []
        if not isinstance(covered, list):
            covered = []
        out.append(
            NormalizedCheckpoint(
                checkpoint_id=str(cp.get("checkpoint_id", "")),
                covered_sessions=tuple(str(x) for x in covered),
                questions=tuple(
                    _normalize_checkpoint_question(q, memory_content_map)
                    for q in qs
                    if isinstance(q, Mapping)
                ),
            )
        )
    return tuple(out)


def _dialogue_turns(
    sample_id: str,
    session_id: str,
    dialogue: list[Any],
    *,
    images_root: Path | None = None,
) -> tuple[NormalizedTurn, ...]:
    """Build NormalizedTurn objects from one session's dialogue.

    Captions are still inlined into ``text`` (so text-only adapters keep
    working unchanged), but ``attachments`` is now populated with
    ``image_id`` + resolved ``file_path`` so that multimodal adapters can
    opt in to feeding the real image file.
    """
    # Ablation toggle: when ``STRIP_IMAGE_PATHS=1`` we populate attachments
    # with captions only (no resolved ``file_path``).  Image-capable
    # baselines fall back to caption text and we can compare against
    # their image-mode V7 numbers.
    strip_image_paths = os.getenv("STRIP_IMAGE_PATHS", "").strip().lower() in {"1", "true", "yes"}

    turns: list[NormalizedTurn] = []
    for turn_index, entry in enumerate(dialogue):
        if not isinstance(entry, dict):
            continue
        text = str(entry.get("content", ""))
        attachments_raw = entry.get("attachments") or []
        image_text_parts: list[str] = []
        attachments_out: list[dict[str, Any]] = []
        if isinstance(attachments_raw, list):
            for att in attachments_raw:
                if not isinstance(att, dict):
                    continue
                cap = att.get("caption", "")
                cap_str = cap if isinstance(cap, str) else str(cap)
                iid = att.get("image_id")
                iid_str = str(iid) if iid else None
                typ = att.get("type", "image_caption")
                typ_str = typ if isinstance(typ, str) else str(typ)
                # Accept either new `file_path` or legacy `image_path` from
                # the raw attachment dict; also try to resolve from images_root.
                raw_fp = att.get("file_path") or att.get("image_path")
                file_path: str | None = None
                if strip_image_paths:
                    file_path = None
                elif isinstance(raw_fp, str) and raw_fp:
                    file_path = raw_fp
                elif images_root is not None and iid_str:
                    # Images stored at <data_dir>/images/<sample_id>/<image_id>.png
                    candidate = images_root / sample_id / f"{iid_str}.png"
                    if candidate.exists():
                        file_path = str(candidate)
                attachments_out.append(
                    {
                        "caption": cap_str,
                        "type": typ_str,
                        "image_id": iid_str,
                        "file_path": file_path,
                    }
                )
                # Fold attachment into text so every adapter (text-only or
                # multimodal) sees image_id *and* caption. Mirrors
                # Mem-Gallery run_bench.py; lets the answer model return
                # the correct image_id for VS/VFR/VU questions.
                if iid_str or cap_str:
                    block_lines = ["image:"]
                    if iid_str:
                        block_lines.append(f"image_id: {iid_str}")
                    if cap_str:
                        block_lines.append(f"image_caption: {cap_str}")
                    image_text_parts.append("\n".join(block_lines))
        if image_text_parts:
            text = text + "\n\n" + "\n".join(image_text_parts)
        ts = entry.get("timestamp")
        timestamp = ts if isinstance(ts, str) else (str(ts) if ts is not None else None)
        raw_turn = {
            "sample_id": sample_id,
            "session_id": session_id,
            "turn_index": turn_index,
            "role": str(entry.get("role", "user")),
            "text": text,
            "attachments": attachments_out,
            "timestamp": timestamp,
        }
        turns.append(normalize_turn(raw_turn))
    return tuple(turns)


def load_legacy_bundle(data_dir: Path) -> EvalBundle:
    """Load a legacy domain_a_v2-format dataset directory into an EvalBundle."""
    data_dir = data_dir.resolve()
    main_path = data_dir / "domain_a_v2.json"
    stage4_path = data_dir / "stage4_memory_points.jsonl"
    qa_path = data_dir / "stage4b_qa_checkpoints.jsonl"
    images_root: Path | None = data_dir / "images"
    if not images_root.is_dir():
        images_root = None  # graceful: text-only datasets won't have this dir

    raw_samples = json.loads(main_path.read_text(encoding="utf-8"))
    if not isinstance(raw_samples, list):
        raise ValueError("domain_a_v2.json must be a list")

    stage4_by_id: dict[str, Stage4Record] = {}
    for obj in _read_jsonl(stage4_path):
        rec = _stage4_from_obj(obj)
        stage4_by_id[rec.sample_id] = rec

    qa_by_id: dict[str, QARecord] = {}
    for obj in _read_jsonl(qa_path):
        rec = _qa_from_obj(obj)
        qa_by_id[rec.sample_id] = rec

    built: list[EvalSample] = []
    for item in raw_samples:
        if not isinstance(item, dict):
            continue
        sample_id = str(item["sample_id"])
        uuid = str(item["uuid"])
        stage4 = stage4_by_id.get(sample_id)
        qa = qa_by_id.get(sample_id)
        if stage4 is None or qa is None:
            raise KeyError(f"missing stage4 or QA row for sample_id={sample_id}")

        stage4_map = {sid: pts for sid, pts in stage4.memory_sessions}

        sessions_raw = item.get("sessions") or []
        if not isinstance(sessions_raw, list):
            sessions_raw = []

        session_blocks: list[EvalSession] = []
        ordered_ids: list[str] = []
        s00_points: tuple[Mapping[str, Any], ...] = ()

        for sess in sessions_raw:
            if not isinstance(sess, dict):
                continue
            sid = str(sess.get("_v2_session_id", ""))
            if not sid:
                continue
            ordered_ids.append(sid)
            dialogue = sess.get("dialogue") or []
            if not isinstance(dialogue, list):
                dialogue = []
            session_blocks.append(
                EvalSession(
                    session_id=sid,
                    turns=_dialogue_turns(
                        sample_id, sid, dialogue, images_root=images_root,
                    ),
                )
            )
            if sid == "S00":
                mps = sess.get("memory_points") or []
                if isinstance(mps, list):
                    s00_points = tuple(mps)

        gold_states = build_session_gold_states(
            ordered_ids,
            s00_memory_points=s00_points,
            stage4_by_session_id=stage4_map,
        )

        # Build memory_id -> memory_content map from all sources
        memory_content_map: dict[str, str] = {}
        for mp_raw in s00_points:
            if isinstance(mp_raw, Mapping):
                mid = mp_raw.get("memory_id")
                mc = mp_raw.get("memory_content")
                if mid is not None and mc is not None:
                    memory_content_map[str(mid)] = str(mc)
        for _sid, pts in stage4.memory_sessions:
            for mp_raw in pts:
                if isinstance(mp_raw, Mapping):
                    mid = mp_raw.get("memory_id")
                    mc = mp_raw.get("memory_content")
                    if mid is not None and mc is not None:
                        memory_content_map[str(mid)] = str(mc)

        built.append(
            EvalSample(
                uuid=uuid,
                sample_id=sample_id,
                sessions=tuple(session_blocks),
                stage4=stage4,
                qa_record=qa,
                normalized_checkpoints=_normalize_checkpoints(
                    qa.raw_checkpoints, memory_content_map
                ),
                session_gold_states=gold_states,
            )
        )

    return EvalBundle(samples=tuple(built))
