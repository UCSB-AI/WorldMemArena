"""Adapter for ViLoMem — dual-stream (logic + visual) memory with embedding retrieval."""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
import uuid as _uuid
from pathlib import Path
from typing import Any

import numpy as np
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

logger = logging.getLogger(__name__)

_DEFAULT_SOURCE = Path(__file__).resolve().parents[1] / "baselines" / "ViLoMem"


class ViLoMemAdapter(MemoryAdapter):
    """Adapter for ViLoMem dual-stream memory (logic + visual).

    Uses ViLoMem's file-based memory storage and embedding retrieval.
    Each turn is stored as a logic memory entry; image captions are
    stored as visual memory entries.
    """

    def __init__(
        self,
        *,
        source_root: str | os.PathLike[str] | None = None,
        **kwargs: Any,
    ) -> None:
        root = Path(source_root or _DEFAULT_SOURCE).resolve()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        src_path = str(root / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        self._root = root
        self._work_dir = Path(tempfile.mkdtemp(prefix="vilomem_eval_"))
        self._logic_file = self._work_dir / "logic_memories.json"
        self._visual_file = self._work_dir / "visual_memories.json"
        self._session_id = ""
        self._prev_snapshot_ids: set[str] = set()
        self._pending_user_turn: NormalizedTurn | None = None

        # Import ViLoMem memory functions
        from vl_agent.memory import (
            load_memories,
            save_memory,
        )
        self._load_memories = load_memories
        self._save_memory = save_memory

        # Try to import embedding functions for retrieval
        from eval_framework.config import resolve_embedding_base_url, resolve_embedding_model
        self._embedding_available = False
        self._embed_model = resolve_embedding_model()
        try:
            from openai import OpenAI
            self._embed_client = OpenAI(
                api_key=os.getenv("OPENAI_EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY"),
                base_url=resolve_embedding_base_url(),
            )
            self._embedding_available = True
        except Exception:
            pass

    def _get_embedding(self, text: str) -> list[float] | None:
        if not self._embedding_available:
            return None
        try:
            from eval_framework.config import resolve_baseline_param
            char_limit = int(resolve_baseline_param("ViLoMem", "embed_char_limit", 8000))
            resp = self._embed_client.embeddings.create(
                model=self._embed_model,
                input=text[:char_limit],
            )
            return resp.data[0].embedding
        except Exception:
            return None

    def _describe_image(self, image_path: str, caption: str = "") -> str:
        """Use shared vision utility; fall back to caption on failure."""
        from eval_framework.memory_adapters._vision import caption_with_vision
        return caption_with_vision(caption, image_path)

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        a_np, b_np = np.array(a), np.array(b)
        norm = np.linalg.norm(a_np) * np.linalg.norm(b_np)
        return float(np.dot(a_np, b_np) / norm) if norm > 0 else 0.0

    def reset(self) -> None:
        if self._work_dir.exists():
            shutil.rmtree(self._work_dir, ignore_errors=True)
        self._work_dir = Path(tempfile.mkdtemp(prefix="vilomem_eval_"))
        self._logic_file = self._work_dir / "logic_memories.json"
        self._visual_file = self._work_dir / "visual_memories.json"
        self._prev_snapshot_ids = set()
        self._pending_user_turn = None

    def ingest_turn(self, turn: NormalizedTurn) -> None:
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
        # Visual items: list of (file_path, caption) pairs
        visual_items: list[tuple[str, str]] = []
        timestamp = None

        for t in (user_turn, assistant_turn):
            if t is None:
                continue
            parts.append(f"{t.role}: {t.text}")
            if timestamp is None:
                timestamp = t.timestamp
            for att in t.attachments:
                visual_items.append((att.file_path or "", att.caption or ""))

        text = "\n".join(parts)
        mid = _uuid.uuid4().hex[:12]

        # Store as logic memory
        logic_entry = {
            "memory_id": f"logic_{mid}",
            "memory_type": "logic",
            "content": text,
            "problem_type": "",
            "error_description": "",
            "corrected_approach": text,
            "created_at": timestamp or "",
            "usage_count": 0,
            "session_id": self._session_id,
        }
        emb = self._get_embedding(text)
        if emb:
            logic_entry["embedding"] = emb
        self._save_memory(self._logic_file, logic_entry)

        # Store visual memories: use vision model on actual image when
        # image_path is available, else fall back to caption text.
        for image_path, cap in visual_items:
            if not image_path and not cap.strip():
                continue
            if image_path and os.path.isfile(image_path):
                content = self._describe_image(image_path, caption=cap)
                source = "vision_model"
            else:
                content = cap
                source = "caption_only"
            visual_entry = {
                "memory_id": f"visual_{_uuid.uuid4().hex[:12]}",
                "memory_type": "visual",
                "content": content,
                "image_path": image_path,
                "source": source,
                "created_at": timestamp or "",
                "usage_count": 0,
                "session_id": self._session_id,
            }
            v_emb = self._get_embedding(content)
            if v_emb:
                visual_entry["embedding"] = v_emb
            self._save_memory(self._visual_file, visual_entry)

    def end_session(self, session_id: str) -> None:
        if self._pending_user_turn is not None:
            self._store_merged(self._pending_user_turn, assistant_turn=None)
            self._pending_user_turn = None
        self._session_id = session_id

    def snapshot_memories(self) -> list[MemorySnapshotRecord]:
        rows: list[MemorySnapshotRecord] = []
        for mem_file, source_label in [
            (self._logic_file, "ViLoMem/logic"),
            (self._visual_file, "ViLoMem/visual"),
        ]:
            for mem in self._load_memories(mem_file):
                mid = mem.get("memory_id", "")
                rows.append(MemorySnapshotRecord(
                    memory_id=mid,
                    text=mem.get("content") or mem.get("corrected_approach", ""),
                    session_id=mem.get("session_id", self._session_id),
                    status="active",
                    source=source_label,
                    raw_backend_id=mid,
                    raw_backend_type=mem.get("memory_type", "logic"),
                    metadata={},
                ))
        return rows

    def export_memory_delta(self, session_id: str) -> list[MemoryDeltaRecord]:
        current = self.snapshot_memories()
        current_ids = {s.memory_id for s in current}
        deltas = [
            MemoryDeltaRecord(
                session_id=session_id, op="add", text=s.text,
                linked_previous=(), raw_backend_id=s.raw_backend_id,
                metadata={"baseline": "ViLoMem", "source": s.source or ""},
            )
            for s in current if s.memory_id not in self._prev_snapshot_ids
        ]
        self._prev_snapshot_ids = current_ids
        return deltas

    def retrieve(self, query: str, top_k: int, **_: Any) -> RetrievalRecord:
        items: list[RetrievalItem] = []
        query_emb = self._get_embedding(query)

        all_memories = (
            self._load_memories(self._logic_file)
            + self._load_memories(self._visual_file)
        )

        if query_emb and all_memories:
            scored = []
            for mem in all_memories:
                mem_emb = mem.get("embedding")
                if mem_emb:
                    sim = self._cosine_sim(query_emb, mem_emb)
                    scored.append((sim, mem))
            scored.sort(key=lambda x: x[0], reverse=True)

            for i, (sim, mem) in enumerate(scored[:top_k]):
                items.append(RetrievalItem(
                    rank=i,
                    memory_id=mem.get("memory_id", str(i)),
                    text=mem.get("content") or mem.get("corrected_approach", ""),
                    score=float(sim),
                    raw_backend_id=mem.get("memory_id", ""),
                ))
        elif all_memories:
            # Fallback: word overlap
            q_words = set(query.lower().split())
            scored = []
            for mem in all_memories:
                text = mem.get("content") or mem.get("corrected_approach", "")
                overlap = len(q_words & set(text.lower().split()))
                scored.append((overlap, mem))
            scored.sort(key=lambda x: x[0], reverse=True)
            for i, (sc, mem) in enumerate(scored[:top_k]):
                items.append(RetrievalItem(
                    rank=i,
                    memory_id=mem.get("memory_id", str(i)),
                    text=mem.get("content") or mem.get("corrected_approach", ""),
                    score=float(sc) / max(len(q_words), 1),
                    raw_backend_id=mem.get("memory_id", ""),
                ))

        return RetrievalRecord(
            query=query, top_k=top_k, items=items[:top_k],
            raw_trace={"baseline": "ViLoMem"},
        )

    def get_capabilities(self) -> dict[str, Any]:
        return {
            "backend": "ViLoMem",
            "baseline": "ViLoMem",
            "available": True,
            "embedding_available": self._embedding_available,
            "delta_granularity": "per_session",
            "snapshot_mode": "dual_stream",
        }
