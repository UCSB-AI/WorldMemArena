"""Adapter for A-Mem (new API: agentic_memory.AgenticMemorySystem)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# chromadb requires sqlite3 >= 3.35.0; use pysqlite3 if the system version is too old.
try:
    import pysqlite3  # noqa: F401
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

from eval_framework.datasets.schemas import (
    MemoryDeltaRecord,
    MemorySnapshotRecord,
    NormalizedTurn,
    RetrievalItem,
    RetrievalRecord,
)
from eval_framework.memory_adapters.base import MemoryAdapter

_DEFAULT_SOURCE = Path(__file__).resolve().parents[1] / "baselines" / "A-Mem"


class AMemV2Adapter(MemoryAdapter):
    """Adapter for A-Mem (new agentic_memory API)."""

    def __init__(
        self,
        *,
        source_root: str | os.PathLike[str] | None = None,
        **kwargs: Any,
    ) -> None:
        root = Path(source_root or _DEFAULT_SOURCE).resolve()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        from agentic_memory.memory_system import AgenticMemorySystem

        self._cls = AgenticMemorySystem
        self._backend: Any = None
        self._session_id = ""
        self._prev_snapshot_ids: set[str] = set()
        self._init_backend()

    def _init_backend(self) -> None:
        from eval_framework.config import resolve_embedding_model, resolve_openai_model
        self._backend = self._cls(
            model_name=resolve_embedding_model(),
            llm_backend="openai",
            llm_model=resolve_openai_model(),
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    def reset(self) -> None:
        self._init_backend()
        self._prev_snapshot_ids = set()
        self._pending_user_turn: NormalizedTurn | None = None

    def ingest_turn(self, turn: NormalizedTurn) -> None:
        """Buffer user turns; store merged user+assistant pair on assistant turn.

        Matches the Mem-Gallery benchmark pattern: each dialogue round
        (user + assistant) is merged into one observation before storing.
        """
        self._session_id = turn.session_id
        if turn.role == "user":
            if self._pending_user_turn is not None:
                self._store_merged(self._pending_user_turn, assistant_turn=None)
            self._pending_user_turn = turn
        else:
            self._store_merged(self._pending_user_turn, assistant_turn=turn)
            self._pending_user_turn = None

    def _store_merged(
        self, user_turn: NormalizedTurn | None, assistant_turn: NormalizedTurn | None
    ) -> None:
        parts: list[str] = []
        timestamp = None
        if user_turn is not None:
            parts.append(f"user: {user_turn.text}")
            for att in user_turn.attachments:
                parts.append(f"[{att.type}] {att.caption}")
            timestamp = user_turn.timestamp
        if assistant_turn is not None:
            parts.append(f"assistant: {assistant_turn.text}")
            for att in assistant_turn.attachments:
                parts.append(f"[{att.type}] {att.caption}")
            if timestamp is None:
                timestamp = assistant_turn.timestamp
        text = "\n".join(parts)
        # Call analyze_content() to extract keywords/context/tags via LLM.
        # A-Mem defines this method but never wires it into add_note() —
        # without it every note keeps the defaults (keywords=[], context="General")
        # and the snapshot judge has nothing meaningful to evaluate.
        try:
            analysis = self._backend.analyze_content(text)
        except Exception:
            analysis = {}
        self._backend.add_note(
            text,
            time=timestamp,
            keywords=analysis.get("keywords") or [],
            context=analysis.get("context") or "General",
            tags=analysis.get("tags") or [],
        )

    def end_session(self, session_id: str) -> None:
        if self._pending_user_turn is not None:
            self._store_merged(self._pending_user_turn, assistant_turn=None)
            self._pending_user_turn = None
        self._session_id = session_id

    def snapshot_memories(self) -> list[MemorySnapshotRecord]:
        """Expose A-Mem's LLM-extracted metadata (context / keywords / tags /
        category) for eval.  Raw ``note.content`` is intentionally excluded —
        the eval grades what A-Mem distilled, not the verbatim input.
        """
        rows: list[MemorySnapshotRecord] = []
        for mid, note in self._backend.memories.items():
            context = str(getattr(note, "context", "") or "")
            keywords = list(getattr(note, "keywords", []) or [])
            tags = list(getattr(note, "tags", []) or [])
            category = str(getattr(note, "category", "") or "")

            parts: list[str] = []
            if context and context.lower() != "general":
                parts.append(f"context: {context}")
            if keywords:
                parts.append(f"keywords: {', '.join(keywords)}")
            if tags:
                parts.append(f"tags: {', '.join(tags)}")
            if category and category.lower() != "uncategorized":
                parts.append(f"category: {category}")
            text = "\n".join(parts)

            rows.append(MemorySnapshotRecord(
                memory_id=str(mid),
                text=text,
                session_id=self._session_id,
                status="active",
                source="A-Mem",
                raw_backend_id=str(mid),
                raw_backend_type="a_mem_note_metadata",
                metadata={
                    "context": context,
                    "keywords": keywords,
                    "tags": tags,
                    "category": category,
                },
            ))
        return rows

    def export_memory_delta(self, session_id: str) -> list[MemoryDeltaRecord]:
        current = self.snapshot_memories()
        current_ids = {s.memory_id for s in current}
        deltas = [
            MemoryDeltaRecord(
                session_id=session_id, op="add", text=s.text,
                linked_previous=(), raw_backend_id=s.raw_backend_id,
                metadata={"baseline": "A-Mem"},
            )
            for s in current if s.memory_id not in self._prev_snapshot_ids
        ]
        self._prev_snapshot_ids = current_ids
        return deltas

    def retrieve(self, query: str, top_k: int, **_: Any) -> RetrievalRecord:
        items: list[RetrievalItem] = []
        try:
            results = self._backend.search(query, k=top_k)
            for i, r in enumerate(results[:top_k]):
                text = r.get("content", str(r)) if isinstance(r, dict) else str(r)
                mid = r.get("id", str(i)) if isinstance(r, dict) else str(i)
                score = float(r.get("score", 1.0 / (i + 1))) if isinstance(r, dict) else 1.0 / (i + 1)
                items.append(RetrievalItem(
                    rank=i, memory_id=str(mid), text=text,
                    score=score, raw_backend_id=str(mid),
                ))
        except Exception:
            # Fallback to raw search
            try:
                raw = self._backend.find_related_memories_raw(query, k=top_k)
                if raw:
                    items.append(RetrievalItem(
                        rank=0, memory_id="bundle", text=str(raw),
                        score=1.0, raw_backend_id=None,
                    ))
            except Exception:
                pass

        return RetrievalRecord(
            query=query, top_k=top_k, items=items[:top_k],
            raw_trace={"baseline": "A-Mem"},
        )

    def get_capabilities(self) -> dict[str, Any]:
        return {
            "backend": "A-Mem",
            "baseline": "A-Mem",
            "available": self._backend is not None,
            "delta_granularity": "snapshot_diff",
            "snapshot_mode": "full",
        }
