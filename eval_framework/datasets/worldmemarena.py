"""WorldMemArena dataset loader.

Reads the WorldMemArena directory structure directly:

    WorldMemArena/
    ├── agent/
    │   ├── gui/{excel,file_mgmt,image_edit,web,word_docs,css,mobile,webarena_lite}/
    │   │   ├── <sample_id>.json
    │   │   └── images/<sample_id>/        (relative to JSON file)
    │   └── embodied/{eb_alfred_base,...,minecraft,omnigibson}/
    │       ├── <sample_id>.json
    │       └── images/<sample_id>/
    └── lifelong/
        ├── project/{academic,education,finance,health,software,startup}/
        └── personal/

Each JSON file contains:
  - sample_id: str
  - sessions: [{_v2_session_id, dialogue: [{role, content, timestamp, attachments}]}]
  - memory_points: [{session_id, memory_points: [...]}]   (top-level, all sessions)
  - qa_checkpoints: [{checkpoint_id, covered_sessions, questions: [...]}]

Image file_path values inside attachments are relative to the JSON file's
parent directory and are resolved to absolute paths here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from eval_framework.datasets.wma_bundle import (
    EvalBundle,
    EvalSample,
    EvalSession,
    NormalizedCheckpoint,
    QARecord,
    Stage4Record,
    _dialogue_turns,
    _normalize_checkpoints,
)
from eval_framework.pipeline.gold_state import build_session_gold_states

# ---------------------------------------------------------------------------
# Subcategory mapping: directory path (relative to WorldMemArena root)
# -> canonical _subcategory string used throughout the pipeline.
# ---------------------------------------------------------------------------
_SUBCAT_MAP: dict[str, str] = {
    # GUI agent
    "agent/gui/excel":        "agent/arena/excel",
    "agent/gui/file_mgmt":    "agent/arena/file_mgmt",
    "agent/gui/image_edit":   "agent/arena/image_edit",
    "agent/gui/web":          "agent/arena/web",
    "agent/gui/word_docs":    "agent/arena/word_docs",
    "agent/gui/css":          "agent/vab/css",
    "agent/gui/mobile":       "agent/vab/mobile",
    "agent/gui/webarena_lite": "agent/vab/webarena-lite",
    # Embodied agent
    "agent/embodied/eb_alfred_base":               "agent/eb_alfred/base",
    "agent/embodied/eb_alfred_common_sense":       "agent/eb_alfred/common_sense",
    "agent/embodied/eb_alfred_complex_instruction":"agent/eb_alfred/complex_instruction",
    "agent/embodied/eb_alfred_long_horizon":       "agent/eb_alfred/long_horizon",
    "agent/embodied/eb_alfred_visual_appearance":  "agent/eb_alfred/visual_appearance",
    "agent/embodied/eb_nav_base":                  "agent/eb_nav/base",
    "agent/embodied/eb_nav_common_sense":          "agent/eb_nav/common_sense",
    "agent/embodied/eb_nav_complex_instruction":   "agent/eb_nav/complex_instruction",
    "agent/embodied/eb_nav_long_horizon":          "agent/eb_nav/long_horizon",
    "agent/embodied/eb_nav_visual_appearance":     "agent/eb_nav/visual_appearance",
    "agent/embodied/minecraft":                    "agent/vab/minecraft",
    "agent/embodied/omnigibson":                   "agent/vab/omnigibson",
    # Lifelong
    "lifelong/project/academic":  "lifelong/domain_a_v2/academic",
    "lifelong/project/education": "lifelong/domain_a_v2/education",
    "lifelong/project/finance":   "lifelong/domain_a_v2/finance",
    "lifelong/project/health":    "lifelong/domain_a_v2/health",
    "lifelong/project/software":  "lifelong/domain_a_v2/software",
    "lifelong/project/startup":   "lifelong/domain_a_v2/startup",
    "lifelong/personal":          "lifelong/domain_b_v2",
}


def _subcategory_from_path(json_path: Path, data_root: Path) -> str:
    """Derive the canonical _subcategory from the JSON file's directory."""
    rel = json_path.parent.relative_to(data_root)
    rel_str = str(rel).replace(os.sep, "/")
    return _SUBCAT_MAP.get(rel_str, rel_str)


def _stage4_from_sample(sample_id: str, raw_memory_points: list[Any]) -> Stage4Record:
    """Build a Stage4Record from the top-level memory_points list in a sample JSON."""
    blocks: list[tuple[str, tuple[Mapping[str, Any], ...]]] = []
    for ms in raw_memory_points:
        if not isinstance(ms, dict):
            continue
        sid = str(ms.get("session_id", ""))
        pts = ms.get("memory_points") or []
        if not isinstance(pts, list):
            pts = []
        # Skip the S00 bootstrapping session (handled separately)
        if sid == "S00":
            continue
        blocks.append((sid, tuple(pts)))
    return Stage4Record(
        uuid=sample_id,
        sample_id=sample_id,
        memory_sessions=tuple(blocks),
    )


def _qa_record_from_sample(sample_id: str, raw_qa_checkpoints: list[Any]) -> QARecord:
    """Build a QARecord from the qa_checkpoints list in a sample JSON."""
    return QARecord(
        uuid=sample_id,
        sample_id=sample_id,
        raw_checkpoints=tuple(raw_qa_checkpoints),
    )


def _load_one_sample(
    json_path: Path,
    data_root: Path,
) -> EvalSample:
    """Load a single WorldMemArena sample JSON into a EvalSample."""
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    sample_id = str(raw["sample_id"])

    # Image paths in this format are relative to the JSON file's directory.
    # We resolve them to absolute paths so downstream adapters can open the file.
    json_dir = json_path.parent
    strip_image_paths = os.getenv("STRIP_IMAGE_PATHS", "").strip().lower() in {"1", "true", "yes"}

    # Build Stage4Record (memory_points)
    raw_mp: list[Any] = raw.get("memory_points") or []
    stage4 = _stage4_from_sample(sample_id, raw_mp)

    # Build QARecord (qa_checkpoints)
    raw_qa: list[Any] = raw.get("qa_checkpoints") or []
    qa_record = _qa_record_from_sample(sample_id, raw_qa)

    # Build session blocks
    sessions_raw: list[Any] = raw.get("sessions") or []
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

        # Resolve relative image paths to absolute before passing to _dialogue_turns.
        # _dialogue_turns accepts raw_fp as a string; we pre-resolve here so it
        # just picks up the already-absolute path without needing images_root.
        if not strip_image_paths:
            resolved_dialogue = _resolve_image_paths(dialogue, json_dir)
        else:
            resolved_dialogue = dialogue

        session_blocks.append(
            EvalSession(
                session_id=sid,
                turns=_dialogue_turns(
                    sample_id, sid, resolved_dialogue, images_root=None,
                ),
            )
        )

        if sid == "S00":
            # S00 may carry memory_points directly (used for gold-state bootstrap)
            mps = sess.get("memory_points") or []
            if isinstance(mps, list):
                s00_points = tuple(mps)

    # Stage4 map keyed by session_id (excludes S00 — included in s00_points)
    stage4_map = {sid: pts for sid, pts in stage4.memory_sessions}

    gold_states = build_session_gold_states(
        ordered_ids,
        s00_memory_points=s00_points,
        stage4_by_session_id=stage4_map,
    )

    # Build memory_id -> memory_content lookup for checkpoint normalization
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

    return EvalSample(
        uuid=sample_id,
        sample_id=sample_id,
        sessions=tuple(session_blocks),
        stage4=stage4,
        qa_record=qa_record,
        normalized_checkpoints=_normalize_checkpoints(
            qa_record.raw_checkpoints, memory_content_map
        ),
        session_gold_states=gold_states,
    )


def _resolve_image_paths(dialogue: list[Any], json_dir: Path) -> list[Any]:
    """Return a copy of the dialogue with relative file_path values resolved to
    absolute paths.  Does not modify the original list."""
    resolved: list[Any] = []
    for entry in dialogue:
        if not isinstance(entry, dict):
            resolved.append(entry)
            continue
        atts_raw = entry.get("attachments")
        if not atts_raw:
            resolved.append(entry)
            continue
        new_atts: list[Any] = []
        changed = False
        for att in atts_raw:
            if not isinstance(att, dict):
                new_atts.append(att)
                continue
            fp = att.get("file_path")
            if isinstance(fp, str) and fp and not Path(fp).is_absolute():
                abs_fp = str((json_dir / fp).resolve())
                att = {**att, "file_path": abs_fp}
                changed = True
            new_atts.append(att)
        if changed:
            entry = {**entry, "attachments": new_atts}
        resolved.append(entry)
    return resolved


def _iter_sample_json_files(data_root: Path):
    """Yield all sample JSON files under data_root, sorted for reproducibility."""
    excluded = {"small_ids.json"}
    paths = sorted(
        p for p in data_root.rglob("*.json")
        if p.name not in excluded and p.parent != data_root
    )
    return paths


def _load_split_ids(data_dir: Path) -> set[str] | None:
    """Load small_ids.json if present; returns a set of sample IDs or None."""
    ids_file = data_dir / "small_ids.json"
    if not ids_file.exists():
        return None
    return set(json.loads(ids_file.read_text(encoding="utf-8")))


def load_worldmemarena(data_dir: Path, split: str = "all") -> EvalBundle:
    """Load the WorldMemArena dataset from its native directory layout.

    ``data_dir`` should point to the top-level ``WorldMemArena/`` folder that
    contains ``agent/`` and ``lifelong/`` subdirectories.

    ``split`` controls which subset to load:
      - ``"all"`` — all samples (default)
      - ``"small"`` — only the 150 samples listed in ``small_ids.json``

    Returns a :class:`EvalBundle` with the same schema as the
    legacy ``converted/all`` loader, so all downstream pipeline code and
    baselines work without modification.
    """
    data_dir = data_dir.resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"WorldMemArena directory not found: {data_dir}")

    json_files = _iter_sample_json_files(data_dir)
    if not json_files:
        raise FileNotFoundError(
            f"No sample JSON files found under {data_dir}. "
            "Expected agent/ and lifelong/ subdirectories."
        )

    small_ids: set[str] | None = None
    if split == "small":
        small_ids = _load_split_ids(data_dir)
        if small_ids is None:
            raise FileNotFoundError(
                f"--split=small requires small_ids.json in {data_dir}"
            )

    samples: list[EvalSample] = []
    for json_path in json_files:
        sample = _load_one_sample(json_path, data_dir)

        if small_ids is not None and sample.sample_id not in small_ids:
            continue

        object.__setattr__(
            sample, "_subcategory",
            _subcategory_from_path(json_path, data_dir),
        )
        samples.append(sample)

    return EvalBundle(samples=tuple(samples))
