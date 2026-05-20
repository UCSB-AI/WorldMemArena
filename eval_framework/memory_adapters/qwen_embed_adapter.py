"""Adapters for Qwen3 embedding baselines (RAG / embedding-only retrieval).

Two flavours, both upstream-native via ``sentence_transformers``:

- ``QwenEmbedAdapter`` — ``Qwen/Qwen3-Embedding-8B`` (text-only, 4096-dim).
  Matches the template's "Qwen3-Embedding-8B" RAG baseline.
- ``QwenVLEmbedAdapter`` — ``Qwen/Qwen3-VL-Embedding-8B`` (text + image,
  4096-dim).  Matches "Qwen3-VL-Embedding-8B".

Both are **embedding-only**:
- ``ingest_turn`` encodes each conversational round (user + assistant text
  plus attachment caption / image path) and stores the embedding alongside
  the raw text.
- ``retrieve(query, top_k)`` encodes the query with the model's recommended
  "query" prompt template and returns the top-k most similar stored rounds
  by cosine similarity.
- No memory abstraction / summarization / LLM extraction — this is
  standard dense retrieval, the way a RAG baseline operates.

Environment variables:
- ``QWEN_EMBED_MODEL``     — text model id (default Qwen/Qwen3-Embedding-8B)
- ``QWEN_VL_EMBED_MODEL``  — VL model id (default Qwen/Qwen3-VL-Embedding-8B)
- ``QWEN_EMBED_DEVICE``    — ``cuda`` / ``cpu`` (default auto)
- ``QWEN_EMBED_BATCH``     — encode batch size (default 8)
"""

from __future__ import annotations

import os
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


import atexit
import base64
import mimetypes

_SHARED_MODELS: dict[str, Any] = {}
_SHARED_POOLS: dict[str, Any] = {}


def _shutdown_pools() -> None:
    """Stop any live multi-GPU encoding pools so children don't linger."""
    for model_id, pool in list(_SHARED_POOLS.items()):
        try:
            model = _SHARED_MODELS.get(model_id)
            if model is not None:
                model.stop_multi_process_pool(pool)
        except Exception:
            pass
        _SHARED_POOLS.pop(model_id, None)


atexit.register(_shutdown_pools)


# Put each Qwen3 baseline's checked-in loader module on sys.path. Unique
# module names (``qwen_text_loader`` / ``qwen_vl_loader``) keep the two
# checkouts from colliding in ``sys.modules``.
import sys as _sys
from pathlib import Path as _Path
for _baseline in ("Qwen3-Embedding-8B", "Qwen3-VL-Embedding-8B"):
    _p = str(_Path(__file__).resolve().parents[1] / "baselines" / _baseline)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)


def _load_sentence_transformer(model_id: str):
    """Load a SentenceTransformer lazily + cache per-model.

    The loader itself lives in the corresponding ``baselines/<Name>/``
    directory (``qwen_text_loader.py`` or ``qwen_vl_loader.py``) so the
    baseline is self-contained per convention. The caller-side cache
    below dedupes model instances across both baselines when they
    happen to ask for the same model_id.
    """
    if model_id in _SHARED_MODELS:
        return _SHARED_MODELS[model_id]
    if "VL" in model_id or "vl" in model_id:
        from qwen_vl_loader import load_encoder  # type: ignore  # noqa: E402
    else:
        from qwen_text_loader import load_encoder  # type: ignore  # noqa: E402
    model = load_encoder(model_id)
    _SHARED_MODELS[model_id] = model
    return model


def _get_encode_pool(model_id: str):
    """Lazily start (and cache) a multi-GPU encoding pool.

    Each child process holds its own model copy in bfloat16 (≈16 GB on
    cuda:X); for four A6000s the combined footprint is 64 GB, spread
    evenly across the cards, which lets batch encodes fan out 4-way
    instead of saturating ``cuda:0`` alone.  Returns ``None`` when fewer
    than two GPUs are visible, in which case the caller should fall back
    to single-device ``model.encode``.
    """
    import torch

    # Multi-GPU pool ON by default: spawns one bf16 worker per CUDA device
    # (≈16 GB × N GPUs) so bulk encodes at ``end_session`` fan out across
    # every available card.  Set ``QWEN_EMBED_MULTI_GPU=0`` to disable when
    # the host has only 1 GPU, when concurrent baselines already saturate
    # cuda:0, or when debugging.  Explicit ``QWEN_EMBED_DEVICE=cuda:X``
    # overrides and pins to a single GPU as before.
    if os.getenv("QWEN_EMBED_MULTI_GPU", "1") != "1":
        return None
    if os.getenv("QWEN_EMBED_DEVICE"):
        return None
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return None
    if model_id in _SHARED_POOLS:
        return _SHARED_POOLS[model_id]

    model = _load_sentence_transformer(model_id)
    num_gpus = int(os.getenv("QWEN_EMBED_NUM_GPUS", "4"))
    devices = [f"cuda:{i}" for i in range(min(torch.cuda.device_count(), num_gpus))]
    if len(devices) < 2:
        return None
    pool = model.start_multi_process_pool(target_devices=devices)
    _SHARED_POOLS[model_id] = pool
    return pool


class _BaseQwenEmbedAdapter(MemoryAdapter):
    """Shared ingest + retrieve plumbing; subclasses pick the model id."""

    _baseline_name: str = "QwenEmbed"
    _default_model_id: str = "Qwen/Qwen3-Embedding-8B"
    _remote_default_model_id: str | None = None
    _env_model_var: str = "QWEN_EMBED_MODEL"
    _encode_images: bool = False
    _env_base_url_var: str = ""

    def __init__(self, **kwargs: Any) -> None:
        del kwargs
        from eval_framework.config import resolve_baseline_param
        self._model_id = (
            os.getenv(self._env_model_var)
            or resolve_baseline_param(self._baseline_name, "model_id", self._default_model_id)
        )
        self._remote_base_url = os.getenv(self._env_base_url_var, "").strip() if self._env_base_url_var else ""
        if self._remote_base_url and not os.getenv(self._env_model_var) and self._remote_default_model_id:
            self._model_id = self._remote_default_model_id
        self._remote_api_key = os.getenv("QWEN_VL_EMBED_API_KEY", "EMPTY")
        if self._env_base_url_var and not self._remote_base_url:
            raise RuntimeError(
                f"{self._baseline_name} requires {self._env_base_url_var}; "
                "local SentenceTransformer fallback is disabled."
            )
        self._model = None if self._remote_base_url else _load_sentence_transformer(self._model_id)
        self._batch = int(resolve_baseline_param(self._baseline_name, "batch_size", 8))
        self._multi_gpu_min_items = int(
            resolve_baseline_param(self._baseline_name, "multi_gpu_min_items", 256)
        )
        self._reset_state()

    def _reset_state(self) -> None:
        self._session_id = ""
        self._pending_user_turn: NormalizedTurn | None = None
        # Per-memory store: parallel arrays of rounded rows + normalized vectors.
        self._rounds: list[dict[str, Any]] = []  # {memory_id, text, session_id, image_path, image_id}
        self._embeddings: list[np.ndarray] = []
        self._prev_mem_ids: set[str] = set()
        # Rows buffered during ingest but not yet encoded.  Flushed as one
        # batch at ``end_session`` so the multi-GPU pool can fan out across
        # every CUDA device instead of being called per round with 1–2
        # items (which would pay dispatch overhead without parallelism).
        self._pending_rows: list[dict[str, Any]] = []

    def reset(self) -> None:
        self._reset_state()

    def ingest_turn(self, turn: NormalizedTurn) -> None:
        self._session_id = turn.session_id
        if turn.role == "user":
            if self._pending_user_turn is not None:
                self._store_round(self._pending_user_turn, None)
            self._pending_user_turn = turn
        else:
            self._store_round(self._pending_user_turn, turn)
            self._pending_user_turn = None

    def _store_round(
        self,
        user_turn: NormalizedTurn | None,
        assistant_turn: NormalizedTurn | None,
    ) -> None:
        parts: list[str] = []
        image_paths: list[str] = []
        image_ids: list[str] = []
        session_id = self._session_id
        for turn in (user_turn, assistant_turn):
            if turn is None:
                continue
            session_id = turn.session_id
            parts.append(f"{turn.role}: {turn.text}")
            for att in turn.attachments:
                caption = att.caption or ""
                if caption:
                    parts.append(f"[{att.type}] {caption}")
                path = getattr(att, "file_path", None) or getattr(att, "image_path", None)
                if path:
                    image_paths.append(path)
                iid = getattr(att, "image_id", None)
                if iid:
                    image_ids.append(iid)
        text = "\n".join(parts).strip()
        if not text and not image_paths:
            return

        # Text-only baseline: encode just the concatenated text.
        # VL baseline: encode each image and the text block, store one row per
        # modality so both text and image queries can hit.
        rows_to_store: list[tuple[str, str | None, str | None]] = []  # (text_for_embed, image_path_hint, image_id)
        rows_to_store.append((text, image_paths[0] if image_paths else None, image_ids[0] if image_ids else None))
        if self._encode_images:
            for path, iid in zip(image_paths, image_ids + [None] * (len(image_paths) - len(image_ids))):
                rows_to_store.append((path, path, iid))

        # Buffer rows; actual encoding happens in bulk at end_session().
        for emb_input, image_path, image_id in rows_to_store:
            self._pending_rows.append({
                "emb_input": emb_input,
                "text": text if not (self._encode_images and emb_input == image_path) else f"[image] {image_id or image_path}",
                "session_id": session_id,
                "image_path": image_path,
                "image_id": image_id,
            })

    def _encode_docs(self, inputs: list[str]) -> np.ndarray:
        """Encode inputs as document-side vectors; normalize to unit length.

        Multi-GPU threshold is configurable per baseline. Qwen3-VL uses
        ``multi_gpu_min_items: 4`` so small session flushes fan out over four
        visible CUDA cards; text-only runs can keep a larger threshold.
        """
        if self._remote_base_url:
            return self._encode_remote(inputs)
        pool = (
            _get_encode_pool(self._model_id)
            if len(inputs) >= self._multi_gpu_min_items
            else None
        )
        if pool is not None:
            vecs = self._model.encode_multi_process(
                inputs,
                pool=pool,
                batch_size=self._batch,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        else:
            vecs = self._model.encode(
                inputs,
                batch_size=self._batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return np.asarray(vecs, dtype="float32")

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode a query with the model's recommended query prompt if supported."""
        if self._remote_base_url:
            return self._encode_remote([query], is_query=True)[0]
        try:
            vec = self._model.encode(
                [query],
                batch_size=1,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                prompt_name="query",
            )
        except Exception:
            # Not every SBERT model bundles a ``query`` prompt template.
            vec = self._model.encode(
                [query],
                batch_size=1,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return vec.astype("float32")[0]

    def _input_to_message(self, value: str, *, is_query: bool = False) -> list[dict[str, Any]]:
        instruction = "Represent the user's query for retrieving relevant memories." if is_query else "Represent the user's input."
        content: list[dict[str, Any]] = []
        send_remote_images = os.getenv("QWEN_VL_EMBED_REMOTE_IMAGES", "0") == "1"
        if self._encode_images and send_remote_images and _looks_like_existing_image(value):
            content.append({"type": "image_url", "image_url": {"url": _image_data_url(value)}})
            content.append({"type": "text", "text": ""})
        else:
            content.append({"type": "text", "text": value})
        return [
            {"role": "system", "content": [{"type": "text", "text": instruction}]},
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": ""}]},
        ]

    def _encode_remote(self, inputs: list[str], *, is_query: bool = False) -> np.ndarray:
        import httpx

        vectors: list[list[float]] = []
        headers = {"Authorization": f"Bearer {self._remote_api_key}"}
        url = self._remote_base_url.rstrip("/") + "/embeddings"
        with httpx.Client(timeout=120.0) as client:
            for start in range(0, len(inputs), self._batch):
                batch = inputs[start : start + self._batch]
                for item in batch:
                    response = client.post(
                        url,
                        headers=headers,
                        json={
                            "messages": self._input_to_message(item, is_query=is_query),
                            "model": self._model_id,
                            "encoding_format": "float",
                            "continue_final_message": True,
                            "add_special_tokens": True,
                        },
                    )
                    response.raise_for_status()
                    data = response.json()["data"]
                    emb = data[0]["embedding"]
                    vectors.append([float(x) for x in emb])
        arr = np.asarray(vectors, dtype="float32")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norms, 1e-12)

    def end_session(self, session_id: str) -> None:
        if self._pending_user_turn is not None:
            self._store_round(self._pending_user_turn, None)
            self._pending_user_turn = None
        self._flush_pending_rows()
        self._session_id = session_id

    def _flush_pending_rows(self) -> None:
        """Encode every buffered row in one batch (fans out across GPUs)."""
        if not self._pending_rows:
            return
        inputs = [r["emb_input"] for r in self._pending_rows]
        try:
            vecs = self._encode_docs(inputs)
        except Exception:
            if self._remote_base_url:
                raise
            vecs_list: list[np.ndarray | None] = []
            for row in self._pending_rows:
                try:
                    vecs_list.append(self._encode_docs([row["emb_input"]])[0])
                except Exception:
                    if row["text"] and row["text"] != row["emb_input"]:
                        try:
                            vecs_list.append(self._encode_docs([row["text"]])[0])
                            continue
                        except Exception:
                            pass
                    vecs_list.append(None)
            kept_rows = []
            kept_vecs = []
            for row, vec in zip(self._pending_rows, vecs_list):
                if vec is not None:
                    kept_rows.append(row)
                    kept_vecs.append(vec)
            self._pending_rows = kept_rows
            vecs = np.asarray(kept_vecs, dtype="float32")
            if len(self._pending_rows) == 0:
                return
        for row, vec in zip(self._pending_rows, vecs):
            mem_id = f"r{len(self._rounds):05d}"
            self._rounds.append({
                "memory_id": mem_id,
                "text": row["text"],
                "session_id": row["session_id"],
                "image_path": row["image_path"],
                "image_id": row["image_id"],
            })
            self._embeddings.append(vec)
        self._pending_rows.clear()

    def snapshot_memories(self) -> list[MemorySnapshotRecord]:
        rows: list[MemorySnapshotRecord] = []
        for item in self._rounds:
            rows.append(
                MemorySnapshotRecord(
                    memory_id=item["memory_id"],
                    text=item["text"],
                    session_id=item["session_id"],
                    status="active",
                    source=self._baseline_name,
                    raw_backend_id=item["memory_id"],
                    raw_backend_type="qwen_embed_round",
                    metadata={
                        "image_path": item["image_path"] or "",
                        "image_id": item["image_id"] or "",
                    },
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
                metadata={"baseline": self._baseline_name},
            )
            for s in new_rows
        ]

    def retrieve(self, query: str, top_k: int, **_: Any) -> RetrievalRecord:
        # Flush any rows buffered since the last end_session — retrieval
        # must see the full memory store even if the caller skipped
        # ``end_session``.
        if self._pending_rows:
            self._flush_pending_rows()
        if not self._rounds:
            return RetrievalRecord(
                query=query,
                top_k=top_k,
                items=[],
                raw_trace={"baseline": self._baseline_name, "reason": "no_memory"},
            )
        q_vec = self._encode_query(query)
        mat = np.stack(self._embeddings, axis=0)
        scores = mat @ q_vec  # cosine (both normalized)
        order = np.argsort(-scores)[: max(0, top_k)]
        items: list[RetrievalItem] = []
        for rank, idx in enumerate(order):
            item = self._rounds[int(idx)]
            items.append(
                RetrievalItem(
                    rank=rank,
                    memory_id=item["memory_id"],
                    text=item["text"],
                    score=float(scores[int(idx)]),
                    raw_backend_id=item["memory_id"],
                    image_path=item["image_path"] or None,
                )
            )
        return RetrievalRecord(
            query=query,
            top_k=top_k,
            items=items,
            raw_trace={
                "baseline": self._baseline_name,
                "model_id": self._model_id,
                "base_url": self._remote_base_url,
                "num_memories": len(self._rounds),
            },
        )

    def get_capabilities(self) -> dict[str, Any]:
        return {
            "backend": self._baseline_name,
            "baseline": self._baseline_name,
            "available": True,
            "delta_granularity": "snapshot_diff",
            "snapshot_mode": "full",
        }


class QwenEmbedAdapter(_BaseQwenEmbedAdapter):
    """Qwen3-Embedding-8B — text-only RAG baseline."""
    _baseline_name = "Qwen3-Embedding-8B"
    _default_model_id = "Qwen/Qwen3-Embedding-8B"
    _env_model_var = "QWEN_EMBED_MODEL"
    _encode_images = False


class QwenVLEmbedAdapter(_BaseQwenEmbedAdapter):
    """Qwen3-VL-Embedding-8B — joint text + image RAG baseline.

    For every dialogue round, the adapter stores:
      1. A text vector from the concatenated round transcript.
      2. One image vector per attached image file path (the VL model
         natively handles URL / local path strings via SentenceTransformer).

    Queries are text-only and encoded with the model's ``query`` prompt;
    cosine similarity is computed against every stored vector.
    """
    _baseline_name = "Qwen3-VL-Embedding-8B"
    _default_model_id = "Qwen/Qwen3-VL-Embedding-8B"
    _remote_default_model_id = "Qwen3-VL-Embedding-8B"
    _env_model_var = "QWEN_VL_EMBED_MODEL"
    _env_base_url_var = "QWEN_VL_EMBED_BASE_URL"
    _encode_images = True


def _looks_like_existing_image(value: str) -> bool:
    path = Path(value)
    if not path.exists() or not path.is_file():
        return False
    mime, _ = mimetypes.guess_type(str(path))
    return bool(mime and mime.startswith("image/"))


def _image_data_url(value: str) -> str:
    path = Path(value)
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"
