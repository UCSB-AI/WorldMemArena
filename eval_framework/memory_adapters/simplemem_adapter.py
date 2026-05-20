"""Adapter for SimpleMem and Omni-SimpleMem baselines.

The omni-mode path mirrors the official Mem-Gallery adapter shipped in
``baselines/SimpleMem/OmniSimpleMem/benchmarks/memgallery/adapter.py`` so that
retrieval behaviour matches the published baseline byte-for-byte:

* ``store()`` always calls ``add_text()``, attaching ``session_id:/dialogue_id:/
  timestamp:/image_id:`` tags. When ``baselines.Omni-SimpleMem.mm_mode`` is
  ``image``, it also calls ``add_image()`` with the resolved file path.
* ``recall()`` supports category markers ``[VS]/[VR]/[FR]/[TR]/[KR]/[CD]/[MR]/
  [AR]/[TTL]`` which drive dynamic ``top_k``, BM25 augmentation, time sort and a
  separate image catalog for visual queries.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from eval_framework.datasets.schemas import (
    MemoryDeltaRecord,
    MemorySnapshotRecord,
    NormalizedTurn,
    RetrievalItem,
    RetrievalRecord,
)
from eval_framework.memory_adapters.base import MemoryAdapter

_DEFAULT_SOURCE = Path(__file__).resolve().parents[1] / "baselines" / "SimpleMem"

_PUNCT_RE = re.compile(r"[^\w\s]")

# Strips ``image_caption: ...`` lines while keeping ``image:`` header and
# ``image_id: SXX_img_Y``. Used when a baseline's config ``mm_mode=image`` so
# the text stored for
# retrieval does NOT leak the human-written caption — forcing the backend's
# visual encoder (OpenVision-CLIP + VLM) to do the actual image understanding.
_IMAGE_CAPTION_STRIP_RE = re.compile(r"^image_caption: .*$\n?", re.MULTILINE)


def _strip_image_caption_lines(text: str) -> str:
    return _IMAGE_CAPTION_STRIP_RE.sub("", text)


def _get_mm_mode(baseline_name: str) -> str:
    """Return 'text' (default) or 'image' from config.yaml.

    Controls whether the omni backend
    ingests real image pixels via ``add_image()`` or only the caption/image_id
    as text."""
    from eval_framework.config import resolve_baseline_param

    val = str(resolve_baseline_param(baseline_name, "mm_mode", "text")).strip().lower()
    return "image" if val in {"image", "mm", "multimodal"} else "text"


def _get_bool_param(baseline_name: str, key: str, default: bool) -> bool:
    from eval_framework.config import resolve_baseline_param

    raw = resolve_baseline_param(baseline_name, key, default)
    if isinstance(raw, str):
        return raw.strip().lower() not in {"0", "false", "no", "off"}
    return bool(raw)

_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "was", "were", "are", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "out", "off", "over",
    "under", "again", "further", "then", "once", "and", "but", "or", "nor",
    "not", "so", "very", "just", "about", "up", "down", "that", "this",
    "these", "those", "it", "its", "he", "she", "they", "them", "his",
    "her", "their", "my", "your", "our", "what", "which", "who", "whom",
    "how", "when", "where", "why", "if", "than", "too", "also", "both",
    "each", "all", "any", "some", "such", "no", "more", "most", "other",
    "only", "same", "own", "here", "there", "please", "based", "conversation",
    "user", "assistant", "question", "answer", "mention", "mentioned",
    "discuss", "discussed", "talk", "talked", "say", "said",
})


# Eval question_type_abbrev → official OmniMem category marker.
# The official marker set is {AR, CD, VS, VR, TR, FR, KR, MR, TTL}.
_CATEGORY_FROM_ABBREV: dict[str, str] = {
    "FR": "FR",       # Fact Recall
    "MB": "FR",       # Memory Boundary — same broad recall path as FR
    "TR": "TR",       # Temporal Reasoning
    "KR": "KR",       # Knowledge Reasoning
    "DU": "CD",       # Dynamic Update → Counterfactual / Contrastive Details
    "CMR": "MR",      # Cross-modal Reasoning → Multi-step Reasoning
    "VS": "VS",       # Visual Search
    "VFR": "VR",      # Visual Fact Recall → Visual Reasoning
    "VU": "VS",       # Visual Update → image-catalog path
}


def _tokenize(text: str) -> list[str]:
    return _PUNCT_RE.sub(" ", text.lower()).split()


def _memory_entry_to_metadata(entry: Any) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    if getattr(entry, "keywords", None):
        meta["keywords"] = list(entry.keywords)
    if getattr(entry, "persons", None):
        meta["persons"] = list(entry.persons)
    if getattr(entry, "entities", None):
        meta["entities"] = list(entry.entities)
    if getattr(entry, "timestamp", None):
        meta["timestamp"] = entry.timestamp
    if getattr(entry, "location", None):
        meta["location"] = entry.location
    if getattr(entry, "topic", None):
        meta["topic"] = entry.topic
    return meta


def _get_backend(mem: Any) -> Any:
    backend = getattr(mem, "_backend", None)
    return backend if backend is not None else mem


class SimpleMemAdapter(MemoryAdapter):
    """Adapter for SimpleMem (text mode) or Omni-SimpleMem (omni mode)."""

    def __init__(
        self,
        *,
        mode: str = "text",
        source_root: str | os.PathLike[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self._mode = mode  # "text" or "omni"
        self._baseline_name = "Omni-SimpleMem" if mode == "omni" else "SimpleMem"
        root = Path(source_root or _DEFAULT_SOURCE).resolve()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        omni_path = str(root / "OmniSimpleMem")
        if omni_path not in sys.path:
            sys.path.insert(0, omni_path)

        import simplemem_router as simplemem
        self._simplemem = simplemem
        self._mem: Any = None
        self._session_id = ""
        self._pending_user_turn: NormalizedTurn | None = None
        self._prev_snapshot_ids: set[str] = set()

        # Indexes mirrored from the official OmniMemAdapter (lazy-built).
        self._bm25: Any = None
        self._bm25_mau_ids: list[str] = []
        self._bm25_texts: list[list[str]] = []
        self._image_catalog: dict[str, dict[str, Any]] | None = None
        self._image_bm25: Any = None
        self._image_bm25_mau_ids: list[str] = []
        self._image_bm25_texts: list[list[str]] = []
        # image_id → resolved absolute file_path, populated during ingest.
        # Used to restore image_path on retrieval items whose MAU modality
        # isn't explicitly "visual" but whose text tags carry an image_id.
        self._img_id_to_path: dict[str, str] = {}
        # Text mode only: user_text fragment → image_path, since
        # ``add_dialogue`` drops attachment metadata and hybrid_retriever
        # returns an LLM-paraphrased ``lossless_restatement``. Substring-match
        # the retrieved text against cached fragments to restore image_path.
        self._img_by_user_text: dict[str, str] = {}

        self._init_mem()

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def _init_mem(self) -> None:
        if self._mode == "text":
            from eval_framework.config import resolve_openai_base_url, resolve_openai_model
            self._mem = self._simplemem.create(
                mode=self._mode,
                clear_db=True,
                api_key=os.getenv("OPENAI_API_KEY") or "",
                model=resolve_openai_model(),
                base_url=resolve_openai_base_url(),
                db_path=os.getenv("SIMPLEMEM_LANCEDB_PATH"),
            )
            return
        # omni mode
        import tempfile
        self._omni_data_dir = tempfile.mkdtemp(prefix="omnisimplemem_eval_")
        from omni_memory import OmniMemoryConfig
        config = OmniMemoryConfig()
        config.storage.base_dir = self._omni_data_dir
        config.storage.cold_storage_dir = os.path.join(self._omni_data_dir, "cold_storage")
        config.storage.index_dir = os.path.join(self._omni_data_dir, "index")
        from eval_framework.config import (
            resolve_embedding_model,
            resolve_openai_base_url,
            resolve_openai_model,
        )
        model = resolve_openai_model()
        config.llm.summary_model = model
        config.llm.query_model = model
        config.llm.caption_model = model
        config.llm.api_key = os.getenv("OPENAI_API_KEY") or ""
        config.llm.api_base_url = resolve_openai_base_url()
        # Route Omni's text embedding through the OpenAI API. Pin
        # ``embedding_dim`` so FAISS / index files stay 1024-d (consistent
        # with the SimpleMem text mode below).
        config.embedding.model_name = resolve_embedding_model()
        config.embedding.embedding_dim = 1024
        self._mem = self._simplemem.create(
            mode=self._mode, config=config, data_dir=self._omni_data_dir
        )

        # Skip CLIP image embedding: the 768-dim vector goes into
        # DualVectorStore._visual_store which is NEVER queried by any
        # production path (only text FAISS is reached by text queries;
        # search_visual/search_hybrid have zero callers in the upstream
        # codebase). We keep VLM captioning (mau.summary — used by BM25)
        # and raw-image storage (raw_pointer — used by the answer VLM).
        # Monkeypatch at the instance level so upstream code is untouched.
        if _get_bool_param("Omni-SimpleMem", "skip_clip", True):
            try:
                image_proc = self._mem.image_processor
                image_proc.generate_embedding = lambda *_a, **_kw: []
                print(
                    "  [Omni-SimpleMem] CLIP image embedding disabled "
                    "(visual_store is never queried; saves forward + FAISS write)",
                    flush=True,
                )
            except AttributeError:
                pass

    def reset(self) -> None:
        if self._mem is not None:
            try:
                self._mem.close()
            except Exception:
                pass
        if hasattr(self, "_omni_data_dir") and os.path.isdir(self._omni_data_dir):
            import shutil
            shutil.rmtree(self._omni_data_dir, ignore_errors=True)
        self._init_mem()
        self._prev_snapshot_ids = set()
        self._bm25 = None
        self._bm25_mau_ids = []
        self._bm25_texts = []
        self._image_catalog = None
        self._image_bm25 = None
        self._image_bm25_mau_ids = []
        self._image_bm25_texts = []

    # ------------------------------------------------------------------
    # ingest
    # ------------------------------------------------------------------

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
        self,
        user_turn: NormalizedTurn | None,
        assistant_turn: NormalizedTurn | None,
    ) -> None:
        mm_mode = _get_mm_mode(self._baseline_name)
        is_omni_image = self._mode == "omni" and mm_mode == "image"

        parts: list[str] = []
        timestamp: str | None = None
        image_ids: list[str] = []
        image_paths: list[tuple[str, str]] = []  # (path, img_id)
        session_id = ""
        turn_index: int | None = None
        for turn, tag in ((user_turn, "user"), (assistant_turn, "assistant")):
            if turn is None:
                continue
            # In config ``mm_mode=image``, strip the inlined ``image_caption: ...`` lines
            # so the stored text does NOT leak the human-written caption. The
            # backend's visual encoder (OpenVision-CLIP + VLM) becomes the only
            # source of image-derived semantic signal. image_id line is kept so
            # the retrieval judge can match image_id as a substring.
            turn_text = (
                _strip_image_caption_lines(turn.text) if is_omni_image else turn.text
            )
            parts.append(f"{tag}: {turn_text}")
            for att in turn.attachments:
                if att.image_id:
                    image_ids.append(att.image_id)
                if att.file_path and att.image_id:
                    image_paths.append((att.file_path, att.image_id))
                    self._img_id_to_path[att.image_id] = att.file_path
            if timestamp is None:
                timestamp = turn.timestamp
            session_id = turn.session_id
            if turn_index is None:
                turn_index = turn.turn_index
        text = "\n".join(parts)
        if not text:
            return

        if self._mode == "text":
            # Cache text mode user_text → image_path so retrieve() can
            # restore image_path after the LLM paraphrase.
            if user_turn and user_turn.text and image_paths:
                self._img_by_user_text[user_turn.text.strip()[:80]] = image_paths[0][0]
            self._mem.add_dialogue("User", text, timestamp or "")
            return

        # omni: tags for both the text MAU and the visual MAUs
        tags: list[str] = []
        if session_id:
            tags.append(f"session_id:{session_id}")
        if turn_index is not None:
            tags.append(f"dialogue_id:{session_id}:{turn_index}")
        if timestamp:
            tags.append(f"timestamp:{timestamp}")
        for iid in image_ids:
            tags.append(f"image_id:{iid}")

        # Always store the dialogue text (with caption stripped in image mode).
        try:
            self._mem.add_text(text, tags=tags or None)
        except Exception as exc:
            print(f"  [Omni-SimpleMem] add_text failed: {exc!r}", flush=True)

        # Image mode only: additionally push each real image through the
        # orchestrator's visual encoder. This produces a VISUAL MAU whose
        # ``summary`` is a VLM-generated caption and whose embedding is a
        # CLIP 768-dim vector (stored in the dedicated visual FAISS). The
        # VLM caption is indexed by the orchestrator's internal BM25 and by
        # our image_catalog BM25, making the visual MAU reachable even though
        # text queries (384-dim) cannot directly hit the 768-dim visual FAISS.
        if is_omni_image and image_paths:
            for path, iid in image_paths:
                img_tags = list(tags)
                # Replace the ambient image_id tags with just this image's id.
                img_tags = [t for t in img_tags if not t.startswith("image_id:")]
                img_tags.append(f"image_id:{iid}")
                try:
                    self._mem.add_image(
                        path,
                        session_id=session_id or None,
                        tags=img_tags,
                    )
                except Exception as exc:
                    print(
                        f"  [Omni-SimpleMem] add_image({path}) failed: {exc!r}",
                        flush=True,
                    )

    def end_session(self, session_id: str) -> None:
        if self._pending_user_turn is not None:
            self._store_merged(self._pending_user_turn, assistant_turn=None)
            self._pending_user_turn = None
        self._session_id = session_id
        if self._mode == "text":
            try:
                self._mem.finalize()
            except Exception:
                pass
        # Invalidate indexes so they rebuild on next recall.
        self._bm25 = None
        self._bm25_mau_ids = []
        self._bm25_texts = []
        self._image_catalog = None
        self._image_bm25 = None
        self._image_bm25_mau_ids = []
        self._image_bm25_texts = []

    # ------------------------------------------------------------------
    # snapshot / delta
    # ------------------------------------------------------------------

    def snapshot_memories(self) -> list[MemorySnapshotRecord]:
        if self._mode == "text":
            return self._snapshot_from_backend()
        return self._snapshot_omni()

    def _snapshot_from_backend(self) -> list[MemorySnapshotRecord]:
        entries = None
        try:
            entries = self._mem.get_all_memories()
        except Exception:
            backend = _get_backend(self._mem)
            entries = backend.vector_store.get_all_entries()
        if entries is None:
            raise RuntimeError(
                "SimpleMem get_all_memories() returned None — backend may not "
                "have processed any turns yet."
            )
        records: list[MemorySnapshotRecord] = []
        for entry in entries:
            eid = getattr(entry, "entry_id", None) or ""
            restatement = getattr(entry, "lossless_restatement", "") or ""
            meta = _memory_entry_to_metadata(entry)
            records.append(
                MemorySnapshotRecord(
                    memory_id=eid,
                    text=restatement,
                    session_id=self._session_id,
                    status="active",
                    source=f"SimpleMem-{self._mode}",
                    raw_backend_id=eid,
                    raw_backend_type="simplemem_memory_entry",
                    metadata=meta,
                )
            )
        return records

    def _snapshot_omni(self) -> list[MemorySnapshotRecord]:
        mau_store = getattr(self._mem, "mau_store", None)
        if mau_store is None:
            raise RuntimeError(
                "OmniMemory backend has no mau_store attribute — cannot "
                "snapshot real stored memories."
            )
        records: list[MemorySnapshotRecord] = []
        for mau in mau_store.iter_all():
            meta_obj = getattr(mau, "metadata", None)
            tags = list(getattr(meta_obj, "tags", []) or [])
            session_id = getattr(meta_obj, "session_id", None) or self._session_id
            if not session_id:
                for tag in tags:
                    if isinstance(tag, str) and tag.startswith("session_id:"):
                        session_id = tag.split(":", 1)[1]
                        break
            meta: dict[str, Any] = {
                "tags": tags,
                "modality": getattr(mau.modality_type, "value", str(mau.modality_type)),
                "timestamp": getattr(mau, "timestamp", None),
            }
            for field in ("persons", "entities", "keywords", "location", "topic"):
                val = getattr(meta_obj, field, None)
                if val:
                    meta[field] = val
            records.append(
                MemorySnapshotRecord(
                    memory_id=mau.id,
                    text=getattr(mau, "summary", "") or "",
                    session_id=session_id or "",
                    status=str(getattr(mau, "status", "active")).lower(),
                    source=f"SimpleMem-{self._mode}",
                    raw_backend_id=mau.id,
                    raw_backend_type="simplemem_omni_mau",
                    metadata=meta,
                )
            )
        return records

    def export_memory_delta(self, session_id: str) -> list[MemoryDeltaRecord]:
        current = self.snapshot_memories()
        current_ids = {s.memory_id for s in current}
        deltas = [
            MemoryDeltaRecord(
                session_id=session_id, op="add", text=s.text,
                linked_previous=(), raw_backend_id=s.raw_backend_id,
                metadata={"baseline": f"SimpleMem-{self._mode}", **s.metadata},
            )
            for s in current if s.memory_id not in self._prev_snapshot_ids
        ]
        self._prev_snapshot_ids = current_ids
        return deltas

    # ------------------------------------------------------------------
    # retrieve (official OmniMemAdapter.recall mirror)
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int,
        **kwargs: Any,
    ) -> RetrievalRecord:
        if self._mode == "text":
            items = self._retrieve_text(query, top_k)
            return RetrievalRecord(
                query=query, top_k=top_k, items=items[:top_k],
                raw_trace={
                    "baseline": f"SimpleMem-{self._mode}",
                    "retrieval_method": "hybrid_retriever",
                },
            )

        category = _CATEGORY_FROM_ABBREV.get(str(kwargs.get("category") or ""))
        items = self._recall_omni(query, category)
        return RetrievalRecord(
            query=query, top_k=top_k, items=items[:top_k],
            raw_trace={
                "baseline": f"SimpleMem-{self._mode}",
                "category_marker": category or "",
            },
        )

    def _retrieve_text(self, query: str, top_k: int) -> list[RetrievalItem]:
        backend = _get_backend(self._mem)
        retriever = getattr(backend, "hybrid_retriever", None)
        if retriever is None:
            raise RuntimeError(
                "SimpleMem hybrid_retriever is not available on the backend. "
                "Ensure turns have been ingested and finalized."
            )
        entries = retriever.retrieve(query)
        items: list[RetrievalItem] = []
        for i, entry in enumerate(entries[:top_k]):
            eid = getattr(entry, "entry_id", "") or str(i)
            text = getattr(entry, "lossless_restatement", str(entry))
            img = next(
                (p for frag, p in self._img_by_user_text.items()
                 if frag and frag in text),
                None,
            )
            items.append(RetrievalItem(
                rank=i, memory_id=eid, text=text,
                score=1.0 / (i + 1), image_path=img, raw_backend_id=eid,
            ))
        return items

    # ----- official-style omni retrieval (category aware) -------------

    def _recall_omni(self, query: str, category: str | None) -> list[RetrievalItem]:
        # VS/VR route: image-specific retrieval.
        if category in ("VS", "VR"):
            return self._recall_visual(query, category)

        effective_top_k = self._get_dynamic_top_k(query, category)

        # FR list questions broaden retrieval.
        is_list_question = False
        if category == "FR":
            q_lower = query.lower()
            if any(kw in q_lower for kw in (
                "list all", "please list", "what are all", "name all",
                "what were all", "how many",
            )):
                is_list_question = True
                effective_top_k = max(effective_top_k, 40)

        try:
            result = self._mem.query(query, top_k=effective_top_k, auto_expand=False)
        except Exception:
            return []
        items = list(getattr(result, "items", []) or [])

        # BM25 augmentation (set-union).
        bm25_extra = 15 if (category == "FR" and is_list_question) or category == "KR" else 10
        try:
            existing_ids = {item.get("id") for item in items}
            bm25_ids = self._bm25_search(query, top_k=effective_top_k)
            new_bm25_ids = [mid for mid in bm25_ids if mid not in existing_ids]
            if new_bm25_ids:
                expanded = self._mem.expand(new_bm25_ids[:bm25_extra])
                for bm25_item in expanded.items:
                    items.append(bm25_item)
        except Exception:
            pass

        if not items:
            return []

        # TR/CD chronological sort by (session, turn).
        if category in ("TR", "CD"):
            items.sort(key=self._sort_by_session_turn)

        # Expand summaries to full_text-bearing items (like official _format_items).
        try:
            mau_ids_need_details = [
                item["id"] for item in items
                if item.get("id") and not item.get("details")
            ]
            if mau_ids_need_details:
                expanded = self._mem.expand(mau_ids_need_details)
                expanded_map = {it["id"]: it for it in expanded.items}
                for i, item in enumerate(items):
                    if item.get("id") in expanded_map:
                        items[i] = {**item, **expanded_map[item["id"]]}
        except Exception:
            pass

        is_image_mode = _get_mm_mode(self._baseline_name) == "image"
        return [
            self._item_dict_to_retrieval_item(item, rank, is_image_mode=is_image_mode)
            for rank, item in enumerate(items)
        ]

    def _recall_visual(self, query: str, category: str) -> list[RetrievalItem]:
        out: list[RetrievalItem] = []
        rank = 0
        # In image mode, the VLM-generated caption (stored as mau.summary on
        # visual MAUs) is used ONLY to power BM25 retrieval — it must NOT
        # leak into the text context passed to the answer VLM, otherwise the
        # model would "cheat" by reading the caption instead of looking at
        # the real image pixels.
        is_image_mode = _get_mm_mode(self._baseline_name) == "image"

        # Standard FAISS + BM25 retrieval first.
        try:
            result = self._mem.query(query, top_k=15, auto_expand=False)
            items = list(getattr(result, "items", []) or [])
        except Exception:
            items = []
        try:
            existing_ids = {item.get("id") for item in items}
            bm25_ids = self._bm25_search(query, top_k=15)
            new_bm25_ids = [mid for mid in bm25_ids if mid not in existing_ids]
            if new_bm25_ids:
                expanded = self._mem.expand(new_bm25_ids[:10])
                for bm25_item in expanded.items:
                    items.append(bm25_item)
        except Exception:
            pass
        try:
            mau_ids_need_details = [
                item["id"] for item in items
                if item.get("id") and not item.get("details")
            ]
            if mau_ids_need_details:
                expanded = self._mem.expand(mau_ids_need_details)
                expanded_map = {it["id"]: it for it in expanded.items}
                for i, item in enumerate(items):
                    if item.get("id") in expanded_map:
                        items[i] = {**item, **expanded_map[item["id"]]}
        except Exception:
            pass
        for item in items:
            out.append(self._item_dict_to_retrieval_item(item, rank, is_image_mode=is_image_mode))
            rank += 1

        # Image catalog search.
        image_results = self._image_search(
            query, return_all=(category == "VR"), top_k=30,
        )
        seen_img_ids: set[str] = set()
        for idx, img in enumerate(image_results, start=1):
            img_id = img.get("image_id", "") or ""
            if img_id and img_id in seen_img_ids:
                continue
            if img_id:
                seen_img_ids.add(img_id)
            caption = img.get("caption", "") or ""
            session = img.get("session_id", "") or ""
            full_text = img.get("full_text", "") or ""
            modality = img.get("modality", "") or ""
            raw_pointer = img.get("raw_pointer") or None
            parts = [f"[IMG-{idx}] image_id: {img_id}"]
            if session:
                parts.append(f"SESSION:{session}")
            # In image mode, deliberately omit VLM caption and surrounding
            # text context from the retrieved item — the answer VLM must
            # rely on the attached JPEG. In text mode we keep the human
            # caption + full_text as the primary textual signal.
            if not is_image_mode:
                if caption:
                    parts.append(f"image_caption: {caption}")
                if full_text and full_text != caption:
                    parts.append(f"context: {full_text[:300]}")
            text = "\n".join(parts)
            score = float(img.get("score") or 0.0)
            # Visual MAU's raw_pointer points at cold_storage JPEG — expose as
            # image_path so the answer_fn's VLM pipeline can load real pixels.
            image_path = raw_pointer if modality in ("visual", "video") else None
            # Fallback: if this text MAU carries an image_id tag but the
            # modality isn't "visual", look the id up in the adapter's
            # image_id → path mapping so the VLM still sees the right pixels.
            if image_path is None and img_id and img_id in self._img_id_to_path:
                image_path = self._img_id_to_path[img_id]
            out.append(RetrievalItem(
                rank=rank,
                memory_id=img.get("mau_id") or img_id or f"img-{rank}",
                text=text,
                score=score if score > 0 else 1.0 / (rank + 1),
                raw_backend_id=img.get("mau_id"),
                image_path=image_path,
            ))
            rank += 1

        return out

    # ----- indexing helpers (read MAU jsonl like the official adapter) -

    def _mau_store_dir(self) -> Path | None:
        data_dir = getattr(self, "_omni_data_dir", None)
        if not data_dir:
            return None
        p = Path(data_dir) / "index" / "mau_store"
        return p if p.exists() else None

    def _iter_mau_jsonl(self) -> list[dict[str, Any]]:
        store_dir = self._mau_store_dir()
        if store_dir is None:
            return []
        records: list[dict[str, Any]] = []
        for jsonl_file in sorted(store_dir.glob("*.jsonl")):
            try:
                with open(jsonl_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            records.append(json.loads(line))
                        except Exception:
                            continue
            except Exception:
                continue
        return records

    def _build_bm25_index(self) -> None:
        if self._bm25 is not None:
            return
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            return
        for mau_data in self._iter_mau_jsonl():
            mau_id = mau_data.get("id", "")
            if not mau_id:
                continue
            details = mau_data.get("details") or {}
            text = details.get("full_text", "") if isinstance(details, dict) else ""
            if not text:
                text = mau_data.get("summary", "") or ""
            if not text:
                continue
            self._bm25_mau_ids.append(mau_id)
            self._bm25_texts.append(_tokenize(text))
        if self._bm25_texts:
            self._bm25 = BM25Okapi(self._bm25_texts)

    def _bm25_search(self, query: str, top_k: int = 10) -> list[str]:
        if self._bm25 is None:
            self._build_bm25_index()
        if self._bm25 is None or not self._bm25_mau_ids:
            return []
        tokenized = _tokenize(query)
        scores = self._bm25.get_scores(tokenized)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self._bm25_mau_ids[i] for i in top_indices if scores[i] > 0]

    def _build_image_catalog(self) -> None:
        """Collect every MAU carrying an ``image_id:`` tag into a BM25 index.

        Covers both ingest modes:
        - ``mm_mode=text``: text MAUs whose ``details.full_text`` contains
          ``image_caption: <human caption>`` (tagged with image_id).
        - ``mm_mode=image``: visual MAUs whose ``summary`` is a VLM-generated
          caption (``full_text`` may be empty). We fall back to ``summary``
          so visual MAUs are findable too.
        """
        if self._image_catalog is not None:
            return
        self._image_catalog = {}
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            return
        for mau_data in self._iter_mau_jsonl():
            mau_id = mau_data.get("id", "")
            if not mau_id:
                continue
            tags = (mau_data.get("metadata") or {}).get("tags", []) or []
            image_ids = [t.split(":", 1)[1] for t in tags if t.startswith("image_id:")]
            if not image_ids:
                continue
            details = mau_data.get("details") or {}
            full_text = details.get("full_text", "") if isinstance(details, dict) else ""
            summary = mau_data.get("summary", "") or ""
            modality = mau_data.get("modality_type", "") or ""
            raw_pointer = mau_data.get("raw_pointer", "") or ""
            # Caption source in priority order:
            # 1) explicit inline 'image_caption: ...' line (text MAU, mm_mode=text)
            # 2) VLM-generated summary (visual MAU, mm_mode=image)
            if "image_caption:" in full_text:
                caption = full_text.split("image_caption:", 1)[1].strip()
            else:
                caption = summary
            session_id = ""
            for tag in tags:
                if tag.startswith("session_id:"):
                    session_id = tag.split(":", 1)[1]
                    break
            for img_id in image_ids:
                self._image_catalog[mau_id] = {
                    "image_id": img_id,
                    "caption": caption,
                    "session_id": session_id,
                    "full_text": full_text,
                    "summary": summary,
                    "modality": modality,
                    "raw_pointer": raw_pointer,
                    "tags": tags,
                }
            # BM25 search text: prefer full_text (richer context), else VLM summary.
            search_text = full_text or summary or caption
            if search_text:
                self._image_bm25_mau_ids.append(mau_id)
                self._image_bm25_texts.append(_tokenize(search_text))
        if self._image_bm25_texts:
            self._image_bm25 = BM25Okapi(self._image_bm25_texts)

    def _image_search(
        self, query: str, *, return_all: bool = False, top_k: int = 30,
    ) -> list[dict[str, Any]]:
        self._build_image_catalog()
        if not self._image_bm25 or not self._image_bm25_mau_ids:
            return []
        tokenized = _tokenize(query)
        scores = self._image_bm25.get_scores(tokenized)
        if return_all:
            indices = [i for i in range(len(scores)) if scores[i] > 0]
            indices.sort(key=lambda i: scores[i], reverse=True)
        else:
            indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            indices = [i for i in indices if scores[i] > 0]
        results: list[dict[str, Any]] = []
        assert self._image_catalog is not None
        for i in indices:
            mau_id = self._image_bm25_mau_ids[i]
            entry = self._image_catalog.get(mau_id)
            if entry:
                results.append({**entry, "mau_id": mau_id, "score": float(scores[i])})
        return results

    # ----- small helpers ---------------------------------------------

    def _get_dynamic_top_k(self, query: str, category: str | None) -> int:
        """Official per-category default top_k, with list/how-many overrides."""
        cat_top_k = {
            "AR": 10, "CD": 15, "VS": 15, "VR": 15, "TR": 20,
            "FR": 30, "KR": 20, "MR": 25, "TTL": 15,
        }
        base_k = cat_top_k.get(category or "", 20)
        q_lower = query.lower()
        if any(kw in q_lower for kw in (
            "list all", "please list", "what are all", "name all", "what were all",
        )):
            base_k = max(base_k, 35)
        if "how many" in q_lower:
            base_k = max(base_k, 30)
        proper_nouns = [
            w for w in query.split()
            if w[:1].isupper() and w.lower() not in _STOP_WORDS
        ]
        if len(proper_nouns) >= 3:
            base_k = max(base_k, 25)
        return base_k

    @staticmethod
    def _sort_by_session_turn(item: dict[str, Any]) -> tuple[int, int]:
        tags = item.get("tags") or []
        if not tags:
            meta = item.get("metadata")
            if isinstance(meta, dict):
                tags = meta.get("tags") or []
        session = ""
        turn = ""
        for tag in tags:
            if tag.startswith("session_id:"):
                session = tag[len("session_id:"):]
            elif tag.startswith("dialogue_id:"):
                turn = tag[len("dialogue_id:"):]
        s_num = 0
        try:
            s_num = int(session.lstrip("S").lstrip("D")) if session else 0
        except (ValueError, AttributeError):
            pass
        t_num = 0
        # dialogue_id may look like "S03:7" — split out the turn index.
        if ":" in turn:
            turn = turn.rsplit(":", 1)[1]
        try:
            t_num = int(turn) if turn else 0
        except (ValueError, AttributeError):
            pass
        return (s_num, t_num)

    def _item_dict_to_retrieval_item(
        self, item: dict[str, Any], rank: int, *, is_image_mode: bool = False,
    ) -> RetrievalItem:
        modality = item.get("modality_type") or ""

        # In image mode, visual-MAU body would be the VLM-generated caption
        # (mau.summary). That caption is indexed by BM25 so retrieval can
        # find the right image, but we must NOT leak it into the text
        # context for the answer VLM — otherwise the model never needs to
        # actually look at the JPEG. Emit only the header for visual items
        # in image mode; the attached raw JPEG is the sole visual signal.
        if is_image_mode and modality in ("visual", "video"):
            body = ""
        else:
            details = item.get("details")
            if isinstance(details, dict):
                body = details.get("full_text", "") or ""
            elif isinstance(details, str):
                body = details
            else:
                body = ""
            if not body:
                body = item.get("summary") or item.get("text") or ""

        tags = item.get("tags") or []
        if not tags:
            meta = item.get("metadata")
            if isinstance(meta, dict):
                tags = meta.get("tags") or []
        meta_parts: list[str] = []
        for tag in tags:
            if tag.startswith("timestamp:"):
                meta_parts.append(f"DATE:{tag[len('timestamp:'):]}")
            elif tag.startswith("session_id:"):
                meta_parts.append(f"SESSION:{tag[len('session_id:'):]}")
            elif tag.startswith("dialogue_id:"):
                meta_parts.append(f"TURN:{tag[len('dialogue_id:'):]}")
            elif tag.startswith("image_id:"):
                meta_parts.append(f"IMG:{tag[len('image_id:'):]}")

        header = f"[{rank + 1}]"
        if meta_parts:
            header += " " + " | ".join(meta_parts)
        text = f"{header}\n{body}" if body else header

        score = float(item.get("score") or 0.0)
        if score <= 0:
            score = 1.0 / (rank + 1)

        # Surface raw image path for visual MAUs so cli.py can attach the
        # real JPEG as an image_url part (independent of the caption strip).
        raw_pointer = item.get("raw_pointer")
        image_path = raw_pointer if modality in ("visual", "video") and raw_pointer else None
        if image_path is None:
            # Fallback: parse the first image_id: tag out and resolve via
            # the adapter-side mapping (VAB-MM ingest populates it during
            # _store_merged).
            for tag in tags:
                if isinstance(tag, str) and tag.startswith("image_id:"):
                    iid = tag[len("image_id:"):]
                    if iid in self._img_id_to_path:
                        image_path = self._img_id_to_path[iid]
                        break

        return RetrievalItem(
            rank=rank,
            memory_id=str(item.get("id") or f"item-{rank}"),
            text=text,
            score=score,
            raw_backend_id=str(item.get("id") or ""),
            image_path=image_path,
        )

    # ------------------------------------------------------------------
    # capabilities
    # ------------------------------------------------------------------

    def get_capabilities(self) -> dict[str, Any]:
        return {
            "backend": self._baseline_name, "baseline": self._baseline_name,
            "available": self._mem is not None,
            "delta_granularity": "per_session",
            "snapshot_mode": "full",
        }
