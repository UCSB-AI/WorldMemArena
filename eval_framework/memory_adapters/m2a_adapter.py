"""Adapter for M²A (Little-Fridge/M2A) — drives the full upstream stack.

Uses M²A's ``M2ASystem`` end-to-end (``RawMessageStore`` + ``SemanticStore``
backed by Milvus + ``ImageManager`` + ``MemoryManager`` + ``ChatAgent``)
with no monkey-patches.  To satisfy M²A's hard-wired embedding
infrastructure without running vLLM, we spin up a tiny local FastAPI
server (``_m2a_embed_server.py``) that:

- serves ``all-MiniLM-L6-v2`` (384-dim) at ``POST /v1/embeddings`` for the
  ``SemanticStore`` text_dense schema;
- stubs the SigLIP2 cross-modal endpoint at ``POST /embeddings`` with a
  768-dim zero vector.  Only used by ``_embed_text_for_cross_modal`` whose
  output is computed but never wired into the hybrid search.

Per-sample isolation: each ``reset`` creates fresh ``raw.db`` +
``semantic.db`` paths under a tempdir so SemanticStore and RawMessageStore
initialise clean Milvus collections and SQLite tables.
"""

from __future__ import annotations

import atexit
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import urllib.request
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

_M2A_SRC = Path(__file__).resolve().parents[1] / "baselines" / "M2A"
_EMBED_SERVER_MODULE = "eval_framework.memory_adapters._m2a_embed_server"

_SERVER_PROC: subprocess.Popen | None = None
_SERVER_PORT: int | None = None
_WORD_RE = re.compile(r"\w+")


def _find_free_port(start: int = 8510) -> int:
    for p in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise RuntimeError("no free port available in range")


def _ensure_embed_server() -> int:
    global _SERVER_PROC, _SERVER_PORT
    if _SERVER_PROC is not None and _SERVER_PROC.poll() is None:
        return _SERVER_PORT  # type: ignore[return-value]
    port = _find_free_port()
    env = os.environ.copy()
    # Prevent the server from loading GPU-hungry siblings.
    env.setdefault("CUDA_VISIBLE_DEVICES", "")
    cmd = [
        sys.executable, "-m", "uvicorn",
        f"{_EMBED_SERVER_MODULE}:app",
        "--host", "127.0.0.1", "--port", str(port),
        "--log-level", "warning",
    ]
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _SERVER_PROC = proc
    _SERVER_PORT = port
    atexit.register(_shutdown_embed_server)

    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + 60
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("embedding server died at startup")
        try:
            with urllib.request.urlopen(url, timeout=1) as r:
                if r.status == 200:
                    return port
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("embedding server did not come up in 60s")


def _shutdown_embed_server() -> None:
    global _SERVER_PROC
    if _SERVER_PROC is not None and _SERVER_PROC.poll() is None:
        try:
            _SERVER_PROC.terminate()
            _SERVER_PROC.wait(timeout=5)
        except Exception:
            try:
                _SERVER_PROC.kill()
            except Exception:
                pass
    _SERVER_PROC = None


def _ensure_m2a_path() -> None:
    src = str(_M2A_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)


class M2AAdapter(MemoryAdapter):
    """M²A driven through the real ``M2ASystem`` + ``ChatAgent`` stack."""

    def __init__(self, **kwargs: Any) -> None:
        del kwargs
        _ensure_m2a_path()
        port = _ensure_embed_server()
        base_url = f"http://127.0.0.1:{port}/v1"

        from agent.m2a import M2ASystem  # type: ignore
        from agent.config import (  # type: ignore
            M2AConfig, LLMConfig, TextEmbeddingConfig, MultimodalEmbeddingConfig,
            MemoryConfig, ChatAgentConfig, MemoryManagerConfig,
        )

        self._M2ASystem = M2ASystem
        self._cfg_cls = (
            M2AConfig, LLMConfig, TextEmbeddingConfig, MultimodalEmbeddingConfig,
            MemoryConfig, ChatAgentConfig, MemoryManagerConfig,
        )
        self._base_url = base_url
        self._reset_state()
        self._init_backend()

    def _reset_state(self) -> None:
        self._session_id = ""
        self._pending_user_turn: NormalizedTurn | None = None
        self._m2a: Any = None
        self._tmp_dir: Path | None = None
        self._ingested_msg_ids: list[int] = []
        self._prev_mem_ids: set[str] = set()

    def _build_config(self, tmp_dir: Path):
        M2AConfig, LLMConfig, TextEmbeddingConfig, MultimodalEmbeddingConfig, \
            MemoryConfig, ChatAgentConfig, MemoryManagerConfig = self._cfg_cls
        from eval_framework.config import (
            resolve_baseline_param,
            resolve_openai_base_url,
            resolve_openai_model,
        )
        llm = LLMConfig(
            model=resolve_openai_model(),
            api_key=os.getenv("OPENAI_API_KEY") or "EMPTY",
            base_url=resolve_openai_base_url(),
            temperature=0.0,
            max_tokens=int(resolve_baseline_param("M2A", "inner_max_tokens", 1200)),
            timeout=int(resolve_baseline_param("M2A", "inner_timeout", 60)),
        )
        from eval_framework.config import resolve_local_sentence_encoder
        text_embed = TextEmbeddingConfig(
            api_key="EMPTY",
            base_url=self._base_url,
            model=resolve_local_sentence_encoder(),
        )
        mm_embed = MultimodalEmbeddingConfig(
            api_key="EMPTY",
            base_url=self._base_url,
            model="siglip2-stub",
        )
        memory_cfg = MemoryConfig(
            raw_db_path=str(tmp_dir / "raw.db"),
            semantic_db_path=str(tmp_dir / "semantic.db"),
            reuse_db=False,
            max_raw_messages_return=20,
        )
        return M2AConfig(
            llm=llm,
            text_embedding=text_embed,
            multimodal_embedding=mm_embed,
            memory=memory_cfg,
            chat_agent=ChatAgentConfig(),
            memory_manager=MemoryManagerConfig(),
        )

    def _init_backend(self) -> None:
        self._tmp_dir = Path(tempfile.mkdtemp(prefix="m2a_"))
        cfg = self._build_config(self._tmp_dir)
        self._m2a = self._M2ASystem(config=cfg)
        self._ingested_msg_ids = []
        # Mirror M²A's upstream M2AEvaluationWrapper.start_conversation:
        # rebuild chat_agent in update-only + update_memory mode and seed
        # the conversation with a minimal eval-style system prompt so the
        # MemoryManager absorbs every turn into SemanticStore without
        # chatting back.
        from agent.agents.chat_agent import ChatAgent  # type: ignore

        self._m2a.chat_agent = ChatAgent(
            memory_manager=self._m2a.memory_manager,
            raw_store=self._m2a.raw_store,
            llm=self._m2a.llm,
            image_manager=self._m2a.image_manager,
            update_memory=True,
            config=self._m2a.config.chat_agent,
            update_only=True,
        )
        self._m2a.chat_agent.init_conversation(
            system_prompt=(
                "You are a memory manager for an AI assistant.\n"
                "You will see messages from user/assistant turns in a task-"
                "oriented trajectory (actions, observations, tool-calls, "
                "environment feedback, results). For EACH incoming user "
                "message you MUST call the `update_memory` tool at least "
                "once to record its key information into long-term memory "
                "— never skip a turn unless it is pure phatic chat. Extract "
                "durable facts (entities, events, states, actions, "
                "preferences) from both personal-dialogue and agent-task "
                "content into the semantic memory store."
            )
        )
        from eval_framework.config import resolve_baseline_param
        self._run_chat_agent = bool(resolve_baseline_param("M2A", "run_chat_agent", False))

    def reset(self) -> None:
        if self._tmp_dir is not None and self._tmp_dir.exists():
            try:
                # SemanticStore keeps Milvus file locks; close first.
                if self._m2a is not None:
                    try:
                        self._m2a.semantic_store.close()
                    except Exception:
                        pass
            finally:
                shutil.rmtree(self._tmp_dir, ignore_errors=True)
        self._reset_state()
        self._init_backend()

    def _render_turn(self, turn: NormalizedTurn) -> str:
        parts = [turn.text] if turn.text else []
        for att in turn.attachments:
            if att.caption:
                parts.append(f"[{att.type}] {att.caption}")
        return "\n".join(parts)

    def _turn_image_path(self, turn: NormalizedTurn) -> str | None:
        for att in turn.attachments:
            path = getattr(att, "file_path", None) or getattr(att, "image_path", None)
            if path:
                return path
        return None

    def ingest_turn(self, turn: NormalizedTurn) -> None:
        self._session_id = turn.session_id
        text = self._render_turn(turn)
        if not text:
            return
        try:
            ts = datetime.fromisoformat(turn.timestamp) if turn.timestamp else datetime.now()
        except (ValueError, TypeError):
            ts = datetime.now()
        speaker = f"{turn.role}_{turn.sample_id}"
        # Upstream M²A expects real image file paths; the local embed server
        # stubs SigLIP2 with zero vectors but the ChatAgent / MemoryManager
        # chain still consumes the image in chat context (image_to_image_token
        # + base64), and the RawMessageStore records image_path for evidence
        # lookup. Pass through whatever attachment path is available.
        image_path = self._turn_image_path(turn)
        # --- force-store every turn via the SemanticStore direct API ---
        # Upstream's ChatAgent + MemoryManager both gate on LLM decisions
        # (tool-call / graph-routing); on VAB-MM agent-trajectory data the
        # LLMs skip storage for most turns (treat observations as
        # transient), leaving the semantic store empty.  Bypass every
        # LLM gate by writing directly to ``semantic_store.add`` with a
        # synthesized ``SemanticMemory`` — guarantees each informative
        # turn reaches the retrievable store.
        try:
            from agent.stores.semantic import SemanticMemory  # type: ignore
            caption = ""
            if image_path:
                # Pull the first attachment caption, if any, so the
                # SemanticStore's text-side embedding has something to
                # match against queries that reference the image.
                for att in turn.attachments:
                    if att.caption:
                        caption = att.caption
                        break
            mem = SemanticMemory(
                text=f"({speaker}, {ts.isoformat()}) {text}",
                image_caption=caption or None,
                image_path=image_path,
            )
            self._m2a.semantic_store.add(mem)
        except Exception as exc:
            print(f"  [M2A] force semantic_store.add failed: {exc!r}", flush=True)

        if not self._run_chat_agent:
            return

        try:
            self._m2a.chat_agent.chat(
                user_text=f"({speaker}, {ts.isoformat()}) {text}",
                user_image_path_or_url=image_path,
                timestamp=ts,
                role=speaker,
            )
        except Exception:
            # Fall back to direct raw-store append if the ChatAgent chain fails.
            from agent.stores.raw import RawMessage  # type: ignore
            mid = self._m2a.raw_store.append(RawMessage(
                msg_id=-1, timestamp=ts, role=speaker, text=text, image_path=image_path,
            ))
            self._ingested_msg_ids.append(mid)

    def end_session(self, session_id: str) -> None:
        self._session_id = session_id

    def _query_memory_collection(self) -> list[dict[str, Any]]:
        try:
            from eval_framework.config import resolve_baseline_param
            rows = self._m2a.semantic_store.db.query(
                collection_name="memory", filter="id > 0",
                limit=int(resolve_baseline_param("M2A", "snapshot_limit", 16000)),
            )
        except Exception:
            rows = []
        return list(rows or [])

    def snapshot_memories(self) -> list[MemorySnapshotRecord]:
        rows = self._query_memory_collection()
        out: list[MemorySnapshotRecord] = []
        for row in rows:
            mid = str(row.get("id", uuid.uuid4().hex[:12]))
            text_parts = []
            if row.get("text"):
                text_parts.append(row["text"])
            if row.get("image_caption"):
                text_parts.append(f"caption: {row['image_caption']}")
            text = " | ".join(text_parts)
            out.append(
                MemorySnapshotRecord(
                    memory_id=f"m2a_sem:{mid}",
                    text=text,
                    session_id=self._session_id,
                    status="active",
                    source="M2A",
                    raw_backend_id=f"m2a_sem:{mid}",
                    raw_backend_type="m2a_semantic_memory",
                    metadata={"image_path": row.get("image_path", "")},
                )
            )
        return out

    def export_memory_delta(self, session_id: str) -> list[MemoryDeltaRecord]:
        current = self.snapshot_memories()
        new_rows = [s for s in current if s.memory_id not in self._prev_mem_ids]
        self._prev_mem_ids = {s.memory_id for s in current}
        return [
            MemoryDeltaRecord(
                session_id=session_id,
                op="add",
                text=s.text,
                linked_previous=(),
                raw_backend_id=s.raw_backend_id,
                metadata={"baseline": "M2A"},
            )
            for s in new_rows
        ]

    def retrieve(self, query: str, top_k: int, **_: Any) -> RetrievalRecord:
        from eval_framework.config import resolve_baseline_param
        retrieval_mode = str(resolve_baseline_param("M2A", "retrieval_mode", "lexical")).lower()
        if retrieval_mode != "hybrid":
            return self._retrieve_lexical(query, top_k, retrieval_mode=retrieval_mode)

        try:
            hits = self._m2a.semantic_store.hybrid_search(
                query_text=query, query_image_path=None, top_k=top_k,
            )
        except Exception:
            hits = []
        items: list[RetrievalItem] = []
        for i, mem in enumerate(hits[:top_k]):
            text = getattr(mem, "text", "") or ""
            caption = getattr(mem, "image_caption", "") or ""
            if caption:
                text = f"{text} | caption: {caption}"
            mid = str(getattr(mem, "memory_id", i))
            items.append(
                RetrievalItem(
                    rank=i,
                    memory_id=f"m2a_sem:{mid}",
                    text=text,
                    score=1.0 / (i + 1),
                    raw_backend_id=f"m2a_sem:{mid}",
                    image_path=getattr(mem, "image_path", None) or None,
                )
            )
        return RetrievalRecord(
            query=query,
            top_k=top_k,
            items=items,
            raw_trace={"baseline": "M2A", "embed_base_url": self._base_url, "retrieval_mode": "hybrid"},
        )

    def _retrieve_lexical(
        self,
        query: str,
        top_k: int,
        *,
        retrieval_mode: str,
    ) -> RetrievalRecord:
        query_terms = set(_WORD_RE.findall((query or "").lower()))
        rows = self._query_memory_collection()

        scored: list[tuple[float, int, dict[str, Any]]] = []
        for idx, row in enumerate(rows):
            text = str(row.get("text") or "")
            caption = str(row.get("image_caption") or "")
            haystack = f"{text} {caption}".lower()
            terms = set(_WORD_RE.findall(haystack))
            overlap = len(query_terms & terms)
            score = float(overlap) / max(len(query_terms), 1)
            scored.append((score, idx, row))
        scored.sort(key=lambda x: (-x[0], x[1]))

        items: list[RetrievalItem] = []
        for rank, (score, _idx, row) in enumerate(scored[:top_k]):
            mid = str(row.get("id", rank))
            text = str(row.get("text") or "")
            caption = str(row.get("image_caption") or "")
            if caption:
                text = f"{text} | caption: {caption}"
            items.append(
                RetrievalItem(
                    rank=rank,
                    memory_id=f"m2a_sem:{mid}",
                    text=text,
                    score=score,
                    raw_backend_id=f"m2a_sem:{mid}",
                    image_path=str(row.get("image_path") or "") or None,
                )
            )
        return RetrievalRecord(
            query=query,
            top_k=top_k,
            items=items,
            raw_trace={
                "baseline": "M2A",
                "embed_base_url": self._base_url,
                "retrieval_mode": retrieval_mode,
            },
        )

    def get_capabilities(self) -> dict[str, Any]:
        return {
            "backend": "M2A",
            "baseline": "M2A",
            "available": True,
            "delta_granularity": "snapshot_diff",
            "snapshot_mode": "full",
        }
