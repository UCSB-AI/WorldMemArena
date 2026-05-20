"""Normalized runtime schemas shared across adapters, pipeline, and evaluators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

MemoryDeltaOp = str

_VALID_DELTA_OPS: frozenset[str] = frozenset(
    {"add", "update", "keep", "suppress", "archive"}
)


@dataclass(frozen=True)
class Attachment:
    """Caption-first attachment; image_id / file_path are optional.

    ``file_path`` is an absolute local path to the image file (when available).
    It is populated by the dataset loader so that multimodal-capable adapters
    can feed the real image to their encoder. Text-only adapters ignore it.
    """

    caption: str
    type: str = "image_caption"
    image_id: str | None = None
    file_path: str | None = None


@dataclass(frozen=True)
class NormalizedTurn:
    sample_id: str
    session_id: str
    turn_index: int
    role: str
    text: str
    attachments: tuple[Attachment, ...] = ()
    timestamp: str | None = None


def normalize_turn(raw: Mapping[str, Any]) -> NormalizedTurn:
    """Build a turn record, keeping attachments that only carry captions."""
    attachments: list[Attachment] = []
    for item in raw.get("attachments") or []:
        if not isinstance(item, dict):
            continue
        cap = item.get("caption", "")
        caption = cap if isinstance(cap, str) else str(cap)
        iid = item.get("image_id")
        if iid is None or iid == "":
            image_id: str | None = None
        else:
            image_id = str(iid)
        typ = item.get("type", "image_caption")
        type_str = typ if isinstance(typ, str) else str(typ)
        fp = item.get("file_path")
        if (fp is None or fp == "") and "image_path" in item:
            fp = item.get("image_path")  # legacy key used by older converters
        file_path = fp if isinstance(fp, str) and fp else None
        attachments.append(
            Attachment(
                caption=caption,
                type=type_str,
                image_id=image_id,
                file_path=file_path,
            )
        )
    ts = raw.get("timestamp")
    timestamp = ts if isinstance(ts, str) else (str(ts) if ts is not None else None)
    return NormalizedTurn(
        sample_id=str(raw["sample_id"]),
        session_id=str(raw["session_id"]),
        turn_index=int(raw["turn_index"]),
        role=str(raw["role"]),
        text=str(raw["text"]),
        attachments=tuple(attachments),
        timestamp=timestamp,
    )


@dataclass(frozen=True)
class MemorySnapshotRecord:
    memory_id: str
    text: str
    session_id: str
    status: str
    source: str | None = None
    raw_backend_id: str | None = None
    raw_backend_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryDeltaRecord:
    session_id: str
    op: MemoryDeltaOp
    text: str
    linked_previous: tuple[str, ...] = ()
    raw_backend_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.op not in _VALID_DELTA_OPS:
            raise ValueError(f"invalid memory delta op: {self.op!r}")


@dataclass(frozen=True)
class RetrievalItem:
    rank: int
    memory_id: str
    text: str
    score: float
    raw_backend_id: str | None = None
    image_path: str | None = None  # absolute local path; set by MM-capable adapters


@dataclass(frozen=True)
class RetrievalRecord:
    query: str
    top_k: int
    items: list[RetrievalItem]
    raw_trace: dict[str, Any] = field(default_factory=dict)
