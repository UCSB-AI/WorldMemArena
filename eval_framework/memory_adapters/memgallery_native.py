"""Mem-Gallery native baseline wrappers with conservative schema normalization."""

from __future__ import annotations

import copy
import os
import re
from typing import Any, Callable

# Strips lines of the form ``image_caption: ...`` (single line) while keeping
# the ``image:`` header and ``image_id: SXX_img_Y``. Used in MM image mode so
# the encoder cannot rely on caption text — forcing a real test of the
# baseline's visual storage/retrieval capability.
_IMAGE_CAPTION_STRIP_RE = re.compile(r"^image_caption: .*$\n?", re.MULTILINE)


def _strip_image_caption_lines(text: str) -> str:
    return _IMAGE_CAPTION_STRIP_RE.sub("", text)


# Mem-Gallery baselines whose backends (MMMemoryStore, MMFUMemoryStore,
# NGMemoryStore, AUGUSTUSMemoryStore, UniversalRAGMemoryStore) route
# observations through ``MultiModalRetrieval.add({'text':..., 'image':...})``.
# Only these baselines get real image paths fed in when config ``mm_mode=image``.
_MM_CAPABLE_BASELINES: frozenset[str] = frozenset(
    {"MMMemory", "MMFUMemory", "MMFU_Single", "NGMemory", "AUGUSTUSMemory", "UniversalRAGMemory"}
)


def _get_mm_mode(baseline_name: str) -> str:
    """Return 'text' (default) or 'image' for this baseline.

    Resolution order:
    1) ``base_model_config.yaml`` for ``BaseModel-*`` provider baselines
    2) ``config.yaml``: ``baselines.<baseline_name>.mm_mode``
    3) default ``text``
    """
    from eval_framework.config import (
        is_base_model_baseline,
        is_harness_baseline,
        resolve_baseline_param,
    )

    if is_base_model_baseline(baseline_name):
        from eval_framework.baselines._clients.base_model_api import build_base_model_target

        target = build_base_model_target(baseline_name)
        return target.mm_mode if target is not None else "text"
    if is_harness_baseline(baseline_name):
        from eval_framework.baselines._clients.harness_api import build_harness_target

        target = build_harness_target(baseline_name)
        return target.mm_mode if target is not None else "text"

    raw = resolve_baseline_param(baseline_name, "mm_mode", "text")
    val = str(raw).strip().lower()
    return "image" if val in {"image", "mm", "multimodal"} else "text"


def _is_base_model_baseline(baseline_name: str) -> bool:
    from eval_framework.config import is_base_model_baseline

    return is_base_model_baseline(baseline_name)


def _is_harness_baseline(baseline_name: str) -> bool:
    from eval_framework.config import is_harness_baseline

    return is_harness_baseline(baseline_name)


def _is_longcontext_baseline(baseline_name: str) -> bool:
    """Baselines that should use the MMFU_Single long-context retrieval path."""
    return (
        baseline_name == "MMFU_Single"
        or _is_base_model_baseline(baseline_name)
        or _is_harness_baseline(baseline_name)
    )


def _allows_real_images_in_store(baseline_name: str) -> bool:
    """Whether this backend should strip captions and store real image paths."""
    return (
        baseline_name in _MM_CAPABLE_BASELINES
        or _is_base_model_baseline(baseline_name)
        or _is_harness_baseline(baseline_name)
    )


from eval_framework.datasets.schemas import (
    MemoryDeltaRecord,
    MemorySnapshotRecord,
    NormalizedTurn,
    RetrievalItem,
    RetrievalRecord,
)
from eval_framework.memory_adapters.base import MemoryAdapter
from eval_framework.memory_adapters.export_utils import (
    linear_element_to_snapshot,
    memory_element_text,
    normalize_recall_to_retrieval,
    turn_to_observation_dict,
)


def _deep_merge_dict(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, val in overrides.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(val, dict)
        ):
            out[key] = _deep_merge_dict(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _default_config_for_baseline(name: str) -> dict[str, Any]:
    import default_config.DefaultMemoryConfig as dmc  # type: ignore[import-not-found]

    key = {
        "FUMemory": "DEFAULT_FUMEMORY",
        "STMemory": "DEFAULT_STMEMORY",
        "LTMemory": "DEFAULT_LTMEMORY",
        "GAMemory": "DEFAULT_GAMEMORY",
        "MGMemory": "DEFAULT_MGMEMORY",
        "RFMemory": "DEFAULT_RFMEMORY",
        "MMMemory": "DEFAULT_MMMEMORY",
        "MMFUMemory": "DEFAULT_MMFUMEMORY",
        "MMFU_Single": "DEFAULT_MMFUMEMORY",
        "NGMemory": "DEFAULT_NGMEMORY",
        "AUGUSTUSMemory": "DEFAULT_AUGUSTUSMEMORY",
        "UniversalRAGMemory": "DEFAULT_UNIVERSALRAGMEMORY",
    }[name]
    cfg = getattr(dmc, key)
    return copy.deepcopy(cfg)


def _import_memory_class(name: str) -> Callable[..., Any]:
    modmap = {
        "FUMemory": ("memengine.memory.FUMemory", "FUMemory"),
        "STMemory": ("memengine.memory.STMemory", "STMemory"),
        "LTMemory": ("memengine.memory.LTMemory", "LTMemory"),
        "GAMemory": ("memengine.memory.GAMemory", "GAMemory"),
        "MGMemory": ("memengine.memory.MGMemory", "MGMemory"),
        "RFMemory": ("memengine.memory.RFMemory", "RFMemory"),
        "MMMemory": ("memengine.memory.MMMemory", "MMMemory"),
        "MMFUMemory": ("memengine.memory.MMFUMemory", "MMFUMemory"),
        "MMFU_Single": ("memengine.memory.MMFUMemory", "MMFUMemory"),
        "NGMemory": ("memengine.memory.NGMemory", "NGMemory"),
        "AUGUSTUSMemory": ("memengine.memory.AUGUSTUSMemory", "AUGUSTUSMemory"),
        "UniversalRAGMemory": ("memengine.memory.UniversalRAGMemory", "UniversalRAGMemory"),
    }
    module_path, cls_name = modmap[name]
    import importlib

    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)


def instantiate_memgallery_memory(
    baseline_name: str,
    config: dict[str, Any] | None = None,
) -> Any:
    """Construct a Mem-Gallery memory object with optional config overrides."""
    base_cfg = _default_config_for_baseline(baseline_name)
    merged = _deep_merge_dict(base_cfg, config or {})
    from memengine.config.Config import MemoryConfig  # type: ignore[import-not-found]

    cls = _import_memory_class(baseline_name)
    return cls(MemoryConfig(merged))


def _graph_nodes_to_snapshots(
    storage: Any,
    *,
    session_id: str,
    source: str,
    include_concepts: bool = False,
) -> list[MemorySnapshotRecord]:
    out: list[MemorySnapshotRecord] = []
    order = getattr(storage, "memory_order_map", []) or []
    node_concepts = getattr(storage, "node_concepts", {})
    for mid_idx, node_id in enumerate(order):
        node = storage.node[node_id]
        cid = node.get("counter_id", mid_idx)
        memory_id = f"n{node_id}"
        text = memory_element_text(node)
        # For AUGUSTUS: append concept tags extracted by the system
        if include_concepts:
            concepts = node_concepts.get(node_id, set())
            if concepts:
                text = f"{text}\n[concepts] {', '.join(sorted(concepts))}"
        out.append(
            MemorySnapshotRecord(
                memory_id=memory_id,
                text=text,
                session_id=session_id,
                status="active",
                source=source,
                raw_backend_id=str(cid),
                raw_backend_type="graph_node",
                metadata={"node_id": node_id},
            )
        )
    return out


def _linear_storage_snapshots(
    storage: Any,
    *,
    session_id: str,
    source: str,
) -> list[MemorySnapshotRecord]:
    rows: list[MemorySnapshotRecord] = []
    for i, m in enumerate(storage.memory_list):
        cid = m.get("counter_id", i)
        rows.append(
            linear_element_to_snapshot(
                m,
                memory_id=str(cid),
                session_id=session_id,
                source=source,
            )
        )
    return rows


def collect_memgallery_snapshots(
    memory: Any,
    baseline_name: str,
    session_id: str,
) -> list[MemorySnapshotRecord]:
    """Best-effort snapshot of backend-visible memories."""
    source = baseline_name
    if baseline_name == "MGMemory":
        out: list[MemorySnapshotRecord] = []
        # store_op/recall_op have their own main_context references;
        # prefer store_op's view as it holds the actual stored data.
        mc = getattr(memory.store_op, "main_context", None) or memory.main_context
        recall_storage = getattr(memory.recall_op, "recall_storage",
                                 getattr(memory, "recall_storage", None))
        archival_storage = getattr(memory.recall_op, "archival_storage",
                                   getattr(memory, "archival_storage", None))
        storages = [("wm", mc["working_context"]), ("fifo", mc["FIFO_queue"])]
        if recall_storage is not None:
            storages.append(("recall", recall_storage))
        if archival_storage is not None:
            storages.append(("archival", archival_storage))
        for prefix, st in storages:
            for i, m in enumerate(st.memory_list):
                cid = m.get("counter_id", i)
                mid = f"{prefix}-{cid}"
                rows = linear_element_to_snapshot(
                    m,
                    memory_id=mid,
                    session_id=session_id,
                    source=source,
                )
                out.append(rows)
        gsum = mc.get("recursive_summary", {}).get("global")
        if gsum and str(gsum) != "None":
            out.append(
                MemorySnapshotRecord(
                    memory_id="recursive_summary",
                    text=str(gsum),
                    session_id=session_id,
                    status="active",
                    source=source,
                    raw_backend_id=None,
                    raw_backend_type="mg_summary",
                    metadata={},
                )
            )
        return out

    if baseline_name == "RFMemory":
        rows = _linear_storage_snapshots(
            memory.storage, session_id=session_id, source=source
        )
        insight = getattr(memory, "insight", {}).get("global_insight", "")
        if insight:
            rows.append(
                MemorySnapshotRecord(
                    memory_id="rf_insight",
                    text=str(insight),
                    session_id=session_id,
                    status="active",
                    source=source,
                    raw_backend_id=None,
                    raw_backend_type="rf_insight",
                    metadata={},
                )
            )
        return rows

    if baseline_name == "NGMemory":
        return _graph_nodes_to_snapshots(
            memory.storage, session_id=session_id, source=source
        )

    if baseline_name == "AUGUSTUSMemory":
        raw_snapshots = _graph_nodes_to_snapshots(
            memory.contextual_memory, session_id=session_id, source=source,
            include_concepts=False,
        )
        # Enrich with concept tags from the memory's concept index
        node_concepts = getattr(memory, "node_concepts", {})
        enriched: list[MemorySnapshotRecord] = []
        for snap in raw_snapshots:
            node_id = snap.metadata.get("node_id")
            concepts = node_concepts.get(node_id, [])
            text = snap.text
            if concepts:
                if isinstance(concepts, list):
                    tags = ', '.join(c['name'] if isinstance(c, dict) else str(c) for c in concepts)
                else:
                    tags = ', '.join(sorted(concepts))
                text = f"{text}\n[concepts: {tags}]"
            enriched.append(MemorySnapshotRecord(
                memory_id=snap.memory_id, text=text,
                session_id=snap.session_id, status=snap.status,
                source=snap.source, raw_backend_id=snap.raw_backend_id,
                raw_backend_type=snap.raw_backend_type, metadata=snap.metadata,
            ))
        return enriched

    if baseline_name == "UniversalRAGMemory":
        return _linear_storage_snapshots(
            memory.storage, session_id=session_id, source=source
        )

    if hasattr(memory, "storage") and hasattr(memory.storage, "memory_list"):
        return _linear_storage_snapshots(
            memory.storage, session_id=session_id, source=source
        )

    return []


# --- Long-context (AMA-Bench-style) retrieval helper -----------------------
#
# Mirrors ``AMA-Bench/src/method/longcontext.py::LongContextMethod`` with a
# per-turn mixed-FIFO eviction policy. The implementation lives under
# ``baselines/MMFU_Single/longcontext.py`` — loaded via sys.path so the
# source can be audited/patched alongside the upstream AMA-Bench checkout.

import sys as _sys
from pathlib import Path as _Path

_MMFU_SINGLE_ROOT = _Path(__file__).resolve().parents[1] / "baselines" / "MMFU_Single"
if str(_MMFU_SINGLE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_MMFU_SINGLE_ROOT))

from longcontext import longcontext_retrieve as _longcontext_retrieve  # type: ignore  # noqa: E402


class MemGalleryNativeAdapter(MemoryAdapter):
    """Thin wrapper that forwards to Mem-Gallery memories and normalizes I/O."""

    def __init__(self, memory: Any, *, baseline_name: str) -> None:
        self._memory = memory
        self._baseline_name = baseline_name
        self._session_id = ""
        self._prev_snapshot_ids: set[str] = set()
        self._pending_user_turn: NormalizedTurn | None = None
        self._session_turns: list[str] = []  # collect turn texts for RF optimize
        # Mem-Gallery's string bundle only surfaces ``image_id`` inline; keep
        # a mapping so retrieve() can restore the real file path for the VLM.
        self._img_id_to_path: dict[str, str] = {}
        # When the upstream MM baseline (NG/AUGUSTUS/UniversalRAG) returns a
        # text bundle that has stripped the ``image_id`` markers, we still
        # need a way to surface the images for the VLM.  Each store also
        # records ``dialogue_id -> [image_paths]`` so ``retrieve()`` can
        # back-fill image attachments using ``recall_op.last_retrieved_ids``
        # (a list of dialogue ids the upstream actually ranked).
        self._dialogue_id_to_image_paths: dict[str, list[str]] = {}

    @classmethod
    def from_baseline(
        cls,
        baseline_name: str,
        *,
        config: dict[str, Any] | None = None,
    ) -> MemGalleryNativeAdapter:
        mem = instantiate_memgallery_memory(baseline_name, config)
        return cls(mem, baseline_name=baseline_name)

    def ingest_turn(self, turn: NormalizedTurn) -> None:
        """Buffer user turns; store merged user+assistant pair on assistant turn.

        This matches the original Mem-Gallery benchmark behavior where each
        dialogue round (user + assistant) is merged into a single observation
        before calling store().
        """
        self._session_id = turn.session_id
        if turn.role == "user":
            # Flush any prior unpaired user turn, then buffer this one
            if self._pending_user_turn is not None:
                self._store_observation(self._pending_user_turn, assistant_turn=None)
            self._pending_user_turn = turn
        else:
            # Assistant turn: merge with buffered user turn and store
            self._store_observation(self._pending_user_turn, assistant_turn=turn)
            self._pending_user_turn = None

    def _store_observation(
        self,
        user_turn: NormalizedTurn | None,
        assistant_turn: NormalizedTurn | None,
    ) -> None:
        """Build a merged observation dict (matching original benchmark format) and store.

        When config ``mm_mode=image`` *and* the baseline is multimodal-capable, the
        observation dict gets an ``image`` key pointing to a real image file
        path. Text-only baselines always ignore attachments' file_path, and
        text mode still keeps the caption inlined in ``obs['text']`` so the
        text encoder can use it.
        """
        mm_mode = _get_mm_mode(self._baseline_name)
        is_mm_image = (
            mm_mode == "image" and _allows_real_images_in_store(self._baseline_name)
        )

        parts: list[str] = []
        timestamp = None
        dialogue_id = ""
        image_records: list[dict[str, Any]] = []
        for turn, tag in ((user_turn, "user"), (assistant_turn, "assistant")):
            if turn is None:
                continue
            # In image mode for MM-capable baselines, strip the inlined
            # ``image_caption: ...`` lines so the encoder does NOT see the
            # caption text in obs['text']. The encoder is forced to rely on
            # the real image pixels. Keep ``image:`` header and
            # ``image_id: SXX_img_Y`` so the retrieval judge can still match
            # image_id substrings.
            turn_text = (
                _strip_image_caption_lines(turn.text) if is_mm_image else turn.text
            )
            parts.append(f"{tag}: {turn_text}")
            for att in turn.attachments:
                if att.image_id and att.file_path:
                    # Remember every image_id → path we see so retrieve()
                    # can restore image_path on items that survive the
                    # string-bundle round trip.
                    self._img_id_to_path[att.image_id] = att.file_path
                if att.file_path:
                    image_records.append(
                        {
                            "path": att.file_path,
                            "img_id": att.image_id or "",
                            # In image mode also DROP caption from the image
                            # dict so downstream text-pipe consumers (e.g.
                            # export_utils.memory_element_text, VLM prompt
                            # builders) don't smuggle caption back into text.
                            "caption": "" if is_mm_image else (att.caption or ""),
                        }
                    )
            if timestamp is None:
                timestamp = turn.timestamp
            if not dialogue_id:
                dialogue_id = f"{turn.session_id}:{turn.turn_index}"

        obs: dict[str, Any] = {"text": "\n".join(parts)}
        if timestamp:
            obs["timestamp"] = timestamp
        obs["dialogue_id"] = dialogue_id

        # Track every file_path attached to this dialogue id so that
        # retrieve() can later look images up by dialogue id even when the
        # upstream recall format strips ``image_id`` markers (NG / AUGUSTUS
        # / UniversalRAG under the GME-aligned MultiModalRetrieval path).
        if dialogue_id and image_records:
            paths = [r["path"] for r in image_records if r.get("path")]
            if paths:
                self._dialogue_id_to_image_paths.setdefault(
                    dialogue_id, []
                ).extend(paths)

        if image_records and is_mm_image:
            # Encoder reads image['path']; img_id and caption are present for
            # downstream consumers. In image mode caption is already blank.
            obs["image"] = image_records[0]

        self._memory.store(obs)
        self._session_turns.append(obs["text"])

    def end_session(self, session_id: str) -> None:
        # Flush any remaining unpaired user turn
        if self._pending_user_turn is not None:
            self._store_observation(self._pending_user_turn, assistant_turn=None)
            self._pending_user_turn = None

        # --- Trigger backend-specific post-session processing ---
        # GAMemory: self-reflection generates insights and stores them
        if self._baseline_name == "GAMemory":
            try:
                self._memory.manage("reflect")
            except Exception:
                pass  # reflection may fail if accumulated importance < threshold

        # RFMemory: optimize generates a global insight from the session trial
        if self._baseline_name == "RFMemory" and self._session_turns:
            try:
                trial = "\n".join(self._session_turns)
                self._memory.optimize(new_trial=trial)
            except Exception:
                pass

        self._session_turns = []

    def snapshot_memories(self) -> list[MemorySnapshotRecord]:
        sid = self._session_id or ""
        return collect_memgallery_snapshots(
            self._memory, self._baseline_name, sid
        )

    def export_memory_delta(self, session_id: str) -> list[MemoryDeltaRecord]:
        """Export delta by diffing current backend snapshot against previous snapshot.

        This reflects what the backend ACTUALLY stores, not what was fed in.
        For FU/ST/LT/GA/RF (LinearStorage), this is the raw observations added.
        For MGMemory, this includes FIFO items, summaries, and archival entries.
        """
        current_snapshot = self.snapshot_memories()
        prev_ids = self._prev_snapshot_ids
        deltas: list[MemoryDeltaRecord] = []
        current_ids: set[str] = set()

        for snap in current_snapshot:
            current_ids.add(snap.memory_id)
            if snap.memory_id not in prev_ids:
                deltas.append(
                    MemoryDeltaRecord(
                        session_id=session_id,
                        op="add",
                        text=snap.text,
                        linked_previous=(),
                        raw_backend_id=snap.raw_backend_id,
                        metadata={
                            "baseline": self._baseline_name,
                            "source": snap.source,
                            "backend_type": snap.raw_backend_type,
                        },
                    )
                )

        self._prev_snapshot_ids = current_ids
        return deltas

    def reset(self) -> None:
        self._memory.reset()
        self._prev_snapshot_ids = set()
        self._pending_user_turn = None
        self._session_turns = []
        self._img_id_to_path = {}
        self._dialogue_id_to_image_paths = {}

    def retrieve(self, query: str, top_k: int, **_: Any) -> RetrievalRecord:
        trace: dict[str, Any] = {"baseline": self._baseline_name}

        # Long-context single-model baseline — bypass backend recall + the
        # lexical-re-rank top-k split entirely.  We read straight from the
        # backend's LinearStorage and apply the AMA-Bench MMLMTruncation
        # algorithm at the per-turn granularity (mixing text-token +
        # image-token budget).  Skipping ``memory.recall(query)`` also
        # avoids the no-op ``LMTruncation(number=10M)`` that crashes the
        # gpt2 fast tokenizer on multi-session conversation buffers.
        if _is_longcontext_baseline(self._baseline_name):
            backend_storage = getattr(self._memory, "storage", None)
            mem_list = getattr(backend_storage, "memory_list", []) if backend_storage else []
            return _longcontext_retrieve(
                query=query,
                top_k=top_k,
                memory_list=mem_list,
                dialogue_id_to_image_paths=self._dialogue_id_to_image_paths,
                trace=trace,
            )

        raw = self._memory.recall(query)
        ro = getattr(self._memory, "recall_op", None)
        if ro is not None and hasattr(ro, "last_retrieved_ids"):
            trace["last_retrieved_ids"] = list(ro.last_retrieved_ids)

        record = normalize_recall_to_retrieval(
            query,
            top_k,
            raw,
            raw_trace=trace,
            image_id_to_path=self._img_id_to_path,
        )

        # Image back-fill via dialogue-id lookup.  Upstream MM baselines
        # (NG / AUGUSTUS / UniversalRAG with MultiModalRetrieval) format
        # their recall as ``[Memory N] timestamp:...\nuser: ...`` which
        # *strips* the ``image_id:`` markers — so the in-text image_id
        # parser cannot recover the path.  Instead we use
        # ``recall_op.last_retrieved_ids`` (a list of dialogue ids the
        # backend actually ranked) to look up each turn's stored images
        # and append them as image-only RetrievalItems so the VLM still
        # sees the visual content.
        retrieved_ids = trace.get("last_retrieved_ids") or []
        if retrieved_ids and self._dialogue_id_to_image_paths:
            existing_paths: set[str] = {
                it.image_path for it in record.items if it.image_path
            }
            extra_imgs: list[str] = []
            for did in retrieved_ids:
                for p in self._dialogue_id_to_image_paths.get(did, ()):  # ordered
                    if p and p not in existing_paths:
                        existing_paths.add(p)
                        extra_imgs.append(p)
            # Cap by remaining slots in top_k.  cli.py iterates items in
            # rank order and pulls image_path from each, so adding extras
            # past top_k is wasted work.
            slots_left = max(0, top_k - len(record.items))
            for i, p in enumerate(extra_imgs[:slots_left]):
                record.items.append(
                    RetrievalItem(
                        rank=len(record.items),
                        memory_id=f"memgallery:dialog_img:{i}",
                        text="",
                        score=1.0,
                        raw_backend_id=None,
                        image_path=p,
                    )
                )
            if extra_imgs:
                record.raw_trace.setdefault("dialog_image_backfill", {}).update(
                    {
                        "available": len(extra_imgs),
                        "attached": min(len(extra_imgs), slots_left),
                    }
                )
        return record

    def get_capabilities(self) -> dict[str, Any]:
        return {
            "backend": "MemGallery",
            "baseline": self._baseline_name,
            "delta_granularity": "ingest_turn_only",
            "snapshot_mode": "conservative",
            "notes": (
                "Deltas record adapter ingest only; backend-internal rewrite, reflection, "
                "or graph reshaping is not diffed. Snapshots read observable storage where supported."
            ),
        }
