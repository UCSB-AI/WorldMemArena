"""Pipeline-facing aliases and runtime record types emitted by the eval runner."""

from __future__ import annotations

from dataclasses import dataclass, field

from eval_framework.datasets.schemas import (
    Attachment,
    MemoryDeltaRecord,
    MemorySnapshotRecord,
    NormalizedTurn,
    RetrievalItem,
    RetrievalRecord,
    normalize_turn,
)
from eval_framework.pipeline.gold_state import SessionGoldState

__all__ = [
    "Attachment",
    "MemoryDeltaRecord",
    "MemorySnapshotRecord",
    "NormalizedTurn",
    "PipelineCheckpointQARecord",
    "PipelineSessionRecord",
    "RetrievalItem",
    "RetrievalRecord",
    "SessionGoldState",
    "normalize_turn",
]


@dataclass(frozen=True)
class PipelineSessionRecord:
    """Normalized outputs after one dialogue session (one row per session)."""

    sample_id: str
    sample_uuid: str
    session_id: str
    memory_snapshot: tuple[MemorySnapshotRecord, ...]
    memory_delta: tuple[MemoryDeltaRecord, ...]
    gold_state: SessionGoldState
    dialogue_str: str = ""
    storage_seconds: float = 0.0
    storage_token_usage: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineCheckpointQARecord:
    """One checkpoint question: retrieval trace plus model-generated answer."""

    sample_id: str
    sample_uuid: str
    checkpoint_id: str
    question: str
    gold_answer: str
    gold_evidence_memory_ids: tuple[str, ...]
    gold_evidence_contents: tuple[str, ...]
    question_type: str
    question_type_abbrev: str
    difficulty: str
    retrieval: RetrievalRecord
    generated_answer: str
    cited_memories: tuple[str, ...] = ()
    retrieval_seconds: float = 0.0
    answer_seconds: float = 0.0
    retrieval_token_usage: dict[str, int] = field(default_factory=dict)
    answer_token_usage: dict[str, int] = field(default_factory=dict)
