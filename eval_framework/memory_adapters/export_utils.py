"""Helpers to map turns, backend memory dicts, and recall outputs into shared schemas."""

from __future__ import annotations

from typing import Any, Mapping

from eval_framework.datasets.schemas import (
    MemorySnapshotRecord,
    NormalizedTurn,
    RetrievalItem,
    RetrievalRecord,
)


def turn_to_observation_dict(turn: NormalizedTurn) -> dict[str, Any]:
    """Build a Mem-Gallery store observation from a normalized turn."""
    parts: list[str] = [turn.text]
    for att in turn.attachments:
        parts.append(f"[{att.type}] {att.caption}")
    text = "\n".join(parts)
    obs: dict[str, Any] = {"text": text}
    if turn.timestamp:
        obs["timestamp"] = turn.timestamp
    obs["dialogue_id"] = f"{turn.session_id}:{turn.turn_index}"
    # Surface the first attachment's image file_path so MM-capable backends
    # can pick it up. Multi-image turns still only get the first path here;
    # adapters that need every image should build their own obs.
    for att in turn.attachments:
        if att.file_path:
            obs["image"] = {
                "path": att.file_path,
                "img_id": att.image_id or "",
                "caption": att.caption or "",
            }
            break
    return obs


def memory_element_text(element: Mapping[str, Any]) -> str:
    """Best-effort text extraction from a Mem-Gallery memory dict."""
    raw = element.get("text", "")
    if isinstance(raw, list):
        return " ".join(str(x) for x in raw)
    if raw is None:
        base = ""
    else:
        base = str(raw)
    image = element.get("image")
    if isinstance(image, dict):
        iid = image.get("img_id") or image.get("image_id")
        cap = image.get("caption")
        if iid or cap:
            line = "[image]"
            if iid:
                line += f" image_id: {iid}"
            if cap:
                line += f" caption: {cap}"
            base = f"{base}\n{line}".strip()
    return base


def linear_element_to_snapshot(
    element: Mapping[str, Any],
    *,
    memory_id: str,
    session_id: str,
    source: str,
    status: str = "active",
) -> MemorySnapshotRecord:
    """Map a linear-storage memory dict into MemorySnapshotRecord."""
    cid = element.get("counter_id")
    raw_id = str(cid) if cid is not None else memory_id
    return MemorySnapshotRecord(
        memory_id=memory_id,
        text=memory_element_text(element),
        session_id=session_id,
        status=status,
        source=source,
        raw_backend_id=raw_id,
        raw_backend_type="linear",
        metadata={},
    )


def _split_memgallery_bundle(raw: str, top_k: int) -> list[str]:
    """Split a Mem-Gallery recall-string bundle into per-memory pieces.

    Upstream's string bundles follow one of two shapes we can parse:

    * **Numbered blocks** (MGMemory): ``[0] user: ...``, ``[1] user: ...``
      sitting inside a section like ``(FIFO Memory)``.  We split on
      ``\\n[<N>] `` while keeping the ``[N]`` marker inside the piece so
      downstream substring matches (judge retrieval coverage) still work.
    * **user/A pairs** (FU / LT / NG / AUGUSTUS / UniversalRAG): consecutive
      ``user: ...\\n\\nA: ...`` rounds separated by ``\\n\\nuser:``.

    If neither pattern produces multiple pieces we return ``[raw]`` so the
    caller can surface the original bundle untouched.
    """
    import re

    if not raw:
        return [raw]

    # Strip any structural header the baseline may prepend (e.g. MGMemory's
    # "[Memory Start]" / "(FIFO Memory)" block labels).  We keep the header
    # outside the splits by searching inside the full raw string.
    numbered = re.split(r"\n(?=\[\d+\]\s)", raw)
    if len(numbered) > 1:
        return [p.strip() for p in numbered if p.strip()]

    # Round boundary: a "user:" line following a prior round.  We accept
    # either a single or double-newline preceding it so we catch both the
    # "user:\n\nA:\n\nuser:" and "user:\nA:\nuser:" layouts upstream emits.
    pair = re.split(r"\n(?=user:\s)", raw)
    if len(pair) > 1:
        return [p.strip() for p in pair if p.strip()]

    # Nothing to split on — let the caller use the single bundle form.
    return [raw]


# Token set used by ``_lexical_score`` — length>2 filters tiny stop-like
# tokens ("is", "a", "to") while keeping content words ("advisor", "shipment").
# The scorer is intentionally lightweight so it has no external deps and
# behaves identically across the 11 MemGallery native baselines. For
# baselines whose upstream ``recall()`` already ranks pieces by embedding
# similarity, the lexical pass only re-breaks ties — upstream's ordering
# shows through via the stable sort.
_TOKEN_RE = __import__("re").compile(r"\w+")


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 2}


def _lexical_score(query: str, piece: str) -> float:
    """Query-recall token-overlap score in [0, 1].

    Fraction of the query's content tokens that appear in the piece. This
    mirrors what a retrieval judge would look for (did the candidate text
    mention the query's key nouns?) without depending on an embedding model
    or corpus statistics.
    """
    q = _tokens(query)
    if not q:
        return 0.0
    p = _tokens(piece)
    if not p:
        return 0.0
    return len(q & p) / len(q)


def _lookup_image_path_for_piece(
    piece_text: str,
    image_id_to_path: Mapping[str, str] | None,
) -> str | None:
    """Parse ``image_id: <path>`` out of a text piece + resolve to a real
    absolute path using the adapter-supplied mapping.

    Mem-Gallery MM baselines return string bundles with ``image_id: X`` lines
    inline; the structured ``image_path`` gets dropped on the string round
    trip.  To restore it for the downstream VLM we map the image_id (which is
    itself an absolute-ish path in this codebase) back to the real file path
    the dataset loader resolved.  If no match is found in the mapping, fall
    back to the inline id itself (already absolute in our VAB-MM data).
    """
    import re

    if image_id_to_path is None:
        image_id_to_path = {}
    m = re.search(r"image_id:\s*([^\n]+?)(?:\n|$)", piece_text)
    if not m:
        return None
    raw_id = m.group(1).strip()
    if raw_id in image_id_to_path:
        return image_id_to_path[raw_id]
    # Fall back to the raw id string if it points at an existing file
    # (VAB-MM image_ids are absolute-looking paths).
    import os
    if raw_id.startswith("/") and os.path.exists(raw_id):
        return raw_id
    return None


def normalize_recall_to_retrieval(
    query: str,
    top_k: int,
    raw: Any,
    *,
    raw_trace: dict[str, Any] | None = None,
    image_id_to_path: Mapping[str, str] | None = None,
) -> RetrievalRecord:
    """Normalize Mem-Gallery recall outputs into RetrievalRecord.

    Mem-Gallery's upstream ``recall()`` returns one of two shapes:

    - ``str``  — an already-assembled context bundle (FU/ST/LT/GA/MG/RF/
      MM/MMFU/NG/AUGUSTUS/UniversalRAG).  We surface this as a **single
      RetrievalItem with rank=0** because the bundle is the upstream's
      atomic unit of context — internally it may concatenate many
      memories joined by newlines, but splitting + truncating to top_k
      would drop information the bundle intentionally packs together
      (e.g. FUMemory's 50k-word window, NGMemory's entity neighbourhood,
      AUGUSTUS's concept-tagged observations).  The judge's retrieval
      coverage metric and answer_fn both read ``item.text`` which still
      contains the full bundle.
    - ``list[dict]`` — per-memory rows (Mem-Gallery's LT-class embedding
      retrieval path) which we expand into one RetrievalItem per row,
      respecting top_k.
    """
    trace = dict(raw_trace or {})
    items: list[RetrievalItem] = []

    if isinstance(raw, str):
        # Mem-Gallery upstream returns a pre-assembled context bundle. Split
        # along known round / block delimiters, then re-rank pieces by their
        # lexical overlap with the query. This fixes the upstream bias in
        # insertion-order baselines (NGMemory / FUMemory / MMFUMemory /
        # STMemory) where recall() concatenates in time order, which used to
        # turn adapter-level head-truncation into a silent "always return
        # session 0" bug. For baselines whose recall() is already relevance-
        # sorted (MMMemory / AUGUSTUSMemory / UniversalRAGMemory / LT / GA /
        # MG / RF), the lexical pass mostly preserves upstream order via the
        # stable sort on (−score, original_index).
        pieces = _split_memgallery_bundle(raw, top_k)
        if len(pieces) > 1:
            scored = [
                (_lexical_score(query, piece), i, piece)
                for i, piece in enumerate(pieces)
            ]
            scored.sort(key=lambda x: (-x[0], x[1]))
            for rank, (score, _src_idx, piece) in enumerate(
                scored[: max(0, top_k)]
            ):
                img_path = _lookup_image_path_for_piece(piece, image_id_to_path)
                items.append(
                    RetrievalItem(
                        rank=rank,
                        memory_id=f"memgallery:bundle:{rank}",
                        text=piece,
                        score=float(score),
                        raw_backend_id=None,
                        image_path=img_path,
                    )
                )
        else:
            img_path = _lookup_image_path_for_piece(raw, image_id_to_path)
            items.append(
                RetrievalItem(
                    rank=0,
                    memory_id="memgallery:string_bundle",
                    text=raw,
                    score=float(_lexical_score(query, raw)) if raw else 0.0,
                    raw_backend_id=None,
                    image_path=img_path,
                )
            )
    elif isinstance(raw, list):
        for i, row in enumerate(raw[: max(0, top_k)]):
            if isinstance(row, dict):
                mid = row.get("counter_id")
                # Pull out the local image path if the backend stored one
                # (Mem-Gallery MM baselines keep {'image': {'path': ...}}).
                img_path: str | None = None
                img = row.get("image")
                if isinstance(img, dict):
                    p = img.get("path")
                    if isinstance(p, str) and p:
                        img_path = p
                elif isinstance(img, str) and img:
                    img_path = img
                items.append(
                    RetrievalItem(
                        rank=i,
                        memory_id=str(mid if mid is not None else i),
                        text=memory_element_text(row),
                        score=float(row.get("score", 1.0)),
                        raw_backend_id=str(mid) if mid is not None else None,
                        image_path=img_path,
                    )
                )
            else:
                items.append(
                    RetrievalItem(
                        rank=i,
                        memory_id=str(i),
                        text=str(row),
                        score=1.0,
                        raw_backend_id=None,
                    )
                )
    else:
        items.append(
            RetrievalItem(
                rank=0,
                memory_id="memgallery:object_bundle",
                text=str(raw),
                score=1.0,
                raw_backend_id=None,
            )
        )

    return RetrievalRecord(query=query, top_k=top_k, items=items[:top_k], raw_trace=trace)
