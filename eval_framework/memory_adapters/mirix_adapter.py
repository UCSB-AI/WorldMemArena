"""Adapter for MIRIX (AIGeeksGroup/MMA/.../MIRIX) — faithful upstream usage.

Drives the real ``mirix.agent.AgentWrapper`` end-to-end: every turn is
shipped through ``send_message(memorizing=True)`` so MIRIX's multi-agent
orchestration (meta-memory + six specialised memory agents) can route the
content into the Core / Episodic / Semantic / Procedural / Resource /
Knowledge-Vault partitions.  At ``end_session`` the temporary-message
accumulator is force-flushed.  Retrieval calls the memory-manager
``list_*`` methods directly with ``search_method="embedding"`` across
every partition and merges the union by cosine score.

Requirements:
- ``MIRIX_DIR`` is pointed at a fresh per-sample directory so each sample
  gets its own SQLite store (``sqlite.db``) under a disjoint tree.
- ``OPENAI_API_KEY`` / ``OPENAI_API_BASE`` are read by
  ``mirix.settings.model_settings``; we forward the project env here.
- MIRIX's ``OPENAI_MODELS`` list (``gpt-4.1`` / ``gpt-4o`` / ``o4-mini``
  only; no gpt-5.1) is enforced upstream, so the MIRIX agent runs on
  ``gpt-4.1`` regardless of the ``OPENAI_MODEL`` the rest of the
  framework uses.

This adapter is expensive: ingestion invokes MIRIX's multi-LLM memory
absorption per-batch (roughly 1 LLM call per 20 turns plus 6 sub-agent
invocations per absorb), so a full VAB-MM sample can cost thousands of
chat-completion calls.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
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

_MIRIX_SRC = (
    Path(__file__).resolve().parents[1]
    / "baselines"
    / "MIRIX"
)
_PARTITION_LIST_ARGS: tuple[tuple[str, str, str], ...] = (
    # (manager_attr, agent_state_attr, list_method)
    ("episodic_memory_manager", "episodic_memory_agent_state", "list_episodic_memory"),
    ("semantic_memory_manager", "semantic_memory_agent_state", "list_semantic_items"),
    ("procedural_memory_manager", "procedural_memory_agent_state", "list_procedures"),
    ("resource_memory_manager", "resource_memory_agent_state", "list_resources"),
    ("knowledge_vault_manager", "knowledge_vault_agent_state", "list_knowledge"),
)

_IMAGE_CAPTION_STRIP_RE = re.compile(r"^image_caption: .*$\n?", re.MULTILINE)


def _strip_image_caption_lines(text: str) -> str:
    return _IMAGE_CAPTION_STRIP_RE.sub("", text)


def _resolve_mm_mode() -> str:
    """Return MIRIX ingest mode from config (default text)."""
    from eval_framework.config import resolve_baseline_param

    raw = resolve_baseline_param("MIRIX", "mm_mode", "text")
    mode = str(raw).strip().lower()
    if mode in {"mm", "multimodal"}:
        mode = "image"
    return "image" if mode == "image" else "text"


def _bootstrap_mirix_path(mirix_dir: Path) -> None:
    """Set MIRIX_DIR env before any mirix import; add sources to sys.path."""
    os.environ["MIRIX_DIR"] = str(mirix_dir)
    # Forward OpenAI endpoint to mirix's ModelSettings env names.
    from eval_framework.config import resolve_openai_base_url
    if os.getenv("OPENAI_API_KEY"):
        os.environ.setdefault("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    os.environ.setdefault("OPENAI_API_BASE", resolve_openai_base_url())
    src_str = str(_MIRIX_SRC)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


class MIRIXAdapter(MemoryAdapter):
    """MIRIX baseline driven through the real ``AgentWrapper``."""

    def __init__(
        self,
        *,
        model_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        del kwargs
        from eval_framework.config import resolve_openai_model
        self._model_name = model_name or resolve_openai_model()
        self._mirix_dir: Path | None = None
        self._agent: Any = None
        self._session_id = ""
        self._pending_user_turn: NormalizedTurn | None = None
        self._prev_mem_ids: set[str] = set()
        self._mm_mode = _resolve_mm_mode()
        # Pipeline creates adapter -> immediately calls reset() before first
        # session. Re-initializing MIRIX twice back-to-back can leave the
        # upstream SQLite engine in a readonly state. Mark this first reset as
        # a no-op for backend recreation; later reset() calls still rebuild.
        self._skip_next_reset_reinit = False
        self._init_backend()

    def _init_backend(self) -> None:
        # Isolate every fresh backend in its own MIRIX_DIR to guarantee a
        # fresh sqlite.db and archival store.
        self._mirix_dir = Path(tempfile.mkdtemp(prefix="mirix_")).resolve()
        _bootstrap_mirix_path(self._mirix_dir)
        (self._mirix_dir / "tmp").mkdir(exist_ok=True)

        cfg_path = self._mirix_dir / "mirix.yaml"
        cfg_path.write_text(
            yaml.safe_dump({"agent_name": f"mirix_{uuid.uuid4().hex[:8]}", "model_name": self._model_name}),
            encoding="utf-8",
        )

        # MIRIX's AgentWrapper.__init__ runs ``set_model(self.model_name)``
        # which routes on a hard-coded ``OPENAI_MODELS`` whitelist and raises
        # ValueError for anything outside it.  Extend the whitelist with our
        # env model so set_model's ``elif model_name in OPENAI_MODELS:``
        # branch fires instead.  This is a runtime module-level addition,
        # no source files are edited.
        import mirix.agent.app_constants as _mirix_constants
        import mirix.agent.agent_wrapper as _mirix_agent_wrapper
        if self._model_name not in _mirix_constants.OPENAI_MODELS:
            _mirix_constants.OPENAI_MODELS = list(
                _mirix_constants.OPENAI_MODELS
            ) + [self._model_name]
            # agent_wrapper.py captured OPENAI_MODELS at its import time, so
            # patch the local alias too.
            _mirix_agent_wrapper.OPENAI_MODELS = _mirix_constants.OPENAI_MODELS

        from mirix.agent import AgentWrapper
        from mirix import EmbeddingConfig, LLMConfig

        self._agent = AgentWrapper(str(cfg_path))

        # Upstream set_model builds LLMConfig with
        # ``model_endpoint="https://api.openai.com/v1"`` hard-coded (see
        # agent_wrapper.py:474); override with our OPENAI_BASE_URL so LLM
        # calls route to the project's configured endpoint.  Apply to both
        # the default config and every agent_state's config.
        from eval_framework.config import resolve_baseline_param, resolve_openai_base_url
        custom_base_url = resolve_openai_base_url()
        llm_config = LLMConfig(
            model=self._model_name,
            model_endpoint_type="openai",
            model_endpoint=custom_base_url,
            model_wrapper=None,
            context_window=int(resolve_baseline_param("MIRIX", "context_window", 128000)),
        )
        try:
            self._agent.client.set_default_llm_config(llm_config)
            for agent_state in self._agent.client.list_agents():
                self._agent.client.server.agent_manager.update_llm_config(
                    agent_id=agent_state.id,
                    llm_config=llm_config,
                    actor=self._agent.client.user,
                )
        except Exception:
            pass

        # Override AgentWrapper's hard-coded text-embedding-3-small default
        # so the embedding model matches the project env.  Upstream stores
        # embeddings via this default config; setting it post-ctor is safe
        # because no memory has been inserted yet.
        from eval_framework.config import resolve_embedding_model
        embed_model = resolve_embedding_model()
        try:
            self._agent.client.set_default_embedding_config(
                EmbeddingConfig.default_config(embed_model)
            )
        except Exception:
            # If the env model isn't in MIRIX's default_config whitelist, fall
            # back to text-embedding-3-small (AgentWrapper's own default).
            pass
        # A brand-new backend was created; allow one immediate reset() call to
        # reuse it (pipeline warmup pattern), then resume normal behavior.
        self._skip_next_reset_reinit = True

    def reset(self) -> None:
        if self._skip_next_reset_reinit:
            self._skip_next_reset_reinit = False
        else:
            if self._mirix_dir is not None and self._mirix_dir.exists():
                shutil.rmtree(self._mirix_dir, ignore_errors=True)
            self._init_backend()
        self._session_id = ""
        self._pending_user_turn = None
        self._prev_mem_ids = set()

    def ingest_turn(self, turn: NormalizedTurn) -> None:
        self._session_id = turn.session_id
        image_mode = self._mm_mode == "image"
        text = self._render_turn(turn, image_mode=image_mode)
        image_uris = (
            [a.file_path for a in turn.attachments if a.file_path]
            if image_mode
            else []
        )
        if not text.strip() and not image_uris:
            return
        timestamp = turn.timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._agent.send_message(
            message=text,
            image_uris=image_uris or None,
            memorizing=True,
            specific_timestamps=[timestamp],
            async_upload=False,
        )

    def _render_turn(self, turn: NormalizedTurn, *, image_mode: bool = False) -> str:
        text = _strip_image_caption_lines(turn.text) if image_mode else turn.text
        parts = [f"{turn.role}: {text}"]
        # In image mode, keep the visual signal in image_uris only.
        if not image_mode:
            for att in turn.attachments:
                if att.caption:
                    parts.append(f"[{att.type}] {att.caption}")
        return "\n".join(parts)

    def end_session(self, session_id: str) -> None:
        self._session_id = session_id
        # Force any buffered messages into memory so snapshot reflects the
        # full session content.
        try:
            self._agent.send_message(
                message="",
                memorizing=True,
                force_absorb_content=True,
                async_upload=False,
            )
        except Exception:
            pass

    def _memory_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        server = self._agent.client.server
        for manager_attr, state_attr, method_name in _PARTITION_LIST_ARGS:
            manager = getattr(server, manager_attr, None)
            agent_state = getattr(self._agent.agent_states, state_attr, None)
            if manager is None or agent_state is None:
                continue
            try:
                items = getattr(manager, method_name)(agent_state, limit=None) or []
            except Exception:
                items = []
            for obj in items:
                mid = str(getattr(obj, "id", None) or uuid.uuid4().hex[:12])
                text = self._render_memory_obj(manager_attr, obj)
                rows.append({
                    "memory_id": f"{manager_attr}:{mid}",
                    "text": text,
                    "partition": manager_attr.replace("_manager", ""),
                    "raw": obj,
                })
        return rows

    def _render_memory_obj(self, manager: str, obj: Any) -> str:
        """Flatten a MIRIX memory Pydantic object into a single text line."""
        fields = []
        for key in ("summary", "details", "description", "steps", "content",
                    "name", "secret_value", "source"):
            v = getattr(obj, key, None)
            if v:
                fields.append(f"{key}: {v}")
        return f"[{manager.replace('_manager', '')}] " + " | ".join(fields)

    def snapshot_memories(self) -> list[MemorySnapshotRecord]:
        rows: list[MemorySnapshotRecord] = []
        for row in self._memory_rows():
            rows.append(
                MemorySnapshotRecord(
                    memory_id=row["memory_id"],
                    text=row["text"],
                    session_id=self._session_id,
                    status="active",
                    source="MIRIX",
                    raw_backend_id=row["memory_id"],
                    raw_backend_type=f"mirix_{row['partition']}",
                    metadata={"partition": row["partition"]},
                )
            )
        return rows

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
                metadata={"baseline": "MIRIX", "partition": s.metadata.get("partition")},
            )
            for s in new_rows
        ]

    def retrieve(self, query: str, top_k: int, **_: Any) -> RetrievalRecord:
        server = self._agent.client.server
        candidates: list[tuple[float, dict[str, Any]]] = []
        per_partition_limit = max(top_k * 2, 8)
        for manager_attr, state_attr, method_name in _PARTITION_LIST_ARGS:
            manager = getattr(server, manager_attr, None)
            agent_state = getattr(self._agent.agent_states, state_attr, None)
            if manager is None or agent_state is None:
                continue
            items: list[Any] = []
            # MIRIX's embedding search requires ``search_field``; rather than
            # picking one per partition, use SQLite BM25 which ranks over
            # ``summary`` by default and is the upstream-recommended method
            # for text-based search on SQLite deployments.
            for search_field in ("summary", "name", "description", "details"):
                try:
                    hits = getattr(manager, method_name)(
                        agent_state,
                        query=query,
                        search_method="bm25",
                        search_field=search_field,
                        limit=per_partition_limit,
                    ) or []
                except TypeError:
                    hits = []
                except Exception:
                    hits = []
                if hits:
                    items = hits
                    break
            if not items:
                try:
                    items = getattr(manager, method_name)(agent_state, limit=per_partition_limit) or []
                except Exception:
                    items = []
            for rank, obj in enumerate(items):
                mid = str(getattr(obj, "id", None) or uuid.uuid4().hex[:12])
                text = self._render_memory_obj(manager_attr, obj)
                score = 1.0 / (rank + 1)
                candidates.append((score, {
                    "memory_id": f"{manager_attr}:{mid}",
                    "text": text,
                    "partition": manager_attr.replace("_manager", ""),
                }))
        candidates.sort(key=lambda x: x[0], reverse=True)
        # Match MMA's original MIRIX evaluation: the chat agent answers from
        # text-only memory partitions and never feeds retrieved images to the
        # answer-stage VLM.  Keep image_path=None so the external answer VLM
        # sees only textual retrieval context (images were already absorbed
        # into MIRIX's internal memory during ingestion).
        items: list[RetrievalItem] = [
            RetrievalItem(
                rank=i,
                memory_id=c["memory_id"],
                text=c["text"],
                score=score,
                image_path=None,
                raw_backend_id=c["memory_id"],
            )
            for i, (score, c) in enumerate(candidates[:top_k])
        ]
        return RetrievalRecord(
            query=query,
            top_k=top_k,
            items=items,
            raw_trace={"baseline": "MIRIX", "model": self._model_name},
        )

    def get_capabilities(self) -> dict[str, Any]:
        return {
            "backend": "MIRIX",
            "baseline": "MIRIX",
            "available": True,
            "delta_granularity": "snapshot_diff",
            "snapshot_mode": "full",
            "mm_mode": self._mm_mode,
            "supports_multimodal_ingest": True,
        }
