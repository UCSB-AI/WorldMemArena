"""Dummy memory adapter for end-to-end pipeline testing.

Stores all ingested turns as raw text and retrieves by simple substring match.
No external dependencies required.
"""

from __future__ import annotations

from typing import Any

from eval_framework.datasets.schemas import (
    MemoryDeltaRecord,
    MemorySnapshotRecord,
    NormalizedTurn,
    RetrievalItem,
    RetrievalRecord,
)
from eval_framework.memory_adapters.base import MemoryAdapter


class DummyAdapter(MemoryAdapter):
    """Minimal adapter that stores turns verbatim — for pipeline testing."""

    def __init__(self) -> None:
        self._memories: list[dict[str, str]] = []
        self._session_id = ""
        self._prev_ids: set[str] = set()

    def reset(self) -> None:
        self._memories = []
        self._session_id = ""
        self._prev_ids = set()

    def ingest_turn(self, turn: NormalizedTurn) -> None:
        self._session_id = turn.session_id
        text = f"{turn.role}: {turn.text}"
        for att in turn.attachments:
            text += f"\n[{att.type}] {att.caption}"
        mid = str(len(self._memories))
        self._memories.append({
            "id": mid,
            "text": text,
            "session_id": turn.session_id,
        })

    def end_session(self, session_id: str) -> None:
        self._session_id = session_id

    def snapshot_memories(self) -> list[MemorySnapshotRecord]:
        return [
            MemorySnapshotRecord(
                memory_id=m["id"],
                text=m["text"],
                session_id=m["session_id"],
                status="active",
                source="Dummy",
                raw_backend_id=m["id"],
                raw_backend_type="dummy",
                metadata={},
            )
            for m in self._memories
        ]

    def export_memory_delta(self, session_id: str) -> list[MemoryDeltaRecord]:
        current_ids = {m["id"] for m in self._memories}
        new_ids = current_ids - self._prev_ids
        deltas = [
            MemoryDeltaRecord(
                session_id=session_id,
                op="add",
                text=m["text"],
                linked_previous=(),
                raw_backend_id=m["id"],
                metadata={"baseline": "Dummy"},
            )
            for m in self._memories
            if m["id"] in new_ids
        ]
        self._prev_ids = current_ids
        return deltas

    def retrieve(self, query: str, top_k: int, **_: Any) -> RetrievalRecord:
        query_lower = query.lower()
        scored = []
        for m in self._memories:
            text_lower = m["text"].lower()
            # Simple word overlap score
            query_words = set(query_lower.split())
            text_words = set(text_lower.split())
            overlap = len(query_words & text_words)
            scored.append((overlap, m))
        scored.sort(key=lambda x: x[0], reverse=True)

        items = [
            RetrievalItem(
                rank=i,
                memory_id=m["id"],
                text=m["text"],
                score=float(overlap) / max(len(query.split()), 1),
                raw_backend_id=m["id"],
            )
            for i, (overlap, m) in enumerate(scored[:top_k])
        ]
        return RetrievalRecord(
            query=query,
            top_k=top_k,
            items=items,
            raw_trace={"baseline": "Dummy"},
        )

    def get_capabilities(self) -> dict[str, Any]:
        return {
            "backend": "Dummy",
            "baseline": "Dummy",
            "available": True,
            "delta_granularity": "per_turn",
            "snapshot_mode": "full",
        }
