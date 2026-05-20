"""Cumulative gold memory state from staged memory-point annotations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class GoldMemoryPoint:
    memory_id: str
    memory_content: str
    memory_type: str
    memory_source: str
    is_update: bool
    original_memories: tuple[str, ...]
    importance: float
    timestamp: str | None = None
    update_type: str = ""


@dataclass(frozen=True)
class SessionGoldState:
    session_id: str
    cumulative_gold_memories: tuple[GoldMemoryPoint, ...]
    session_new_memories: tuple[GoldMemoryPoint, ...]
    session_update_memories: tuple[GoldMemoryPoint, ...]
    session_interference_memories: tuple[GoldMemoryPoint, ...]


def _as_str_tuple(val: Any) -> tuple[str, ...]:
    if val is None:
        return ()
    if isinstance(val, str):
        return (val,)
    if isinstance(val, Sequence) and not isinstance(val, (str, bytes)):
        out: list[str] = []
        for x in val:
            out.append(x if isinstance(x, str) else str(x))
        return tuple(out)
    return (str(val),)


def _parse_is_update(raw: Mapping[str, Any]) -> bool:
    v = raw.get("is_update")
    if v is True:
        return True
    if v is False or v is None:
        return False
    if isinstance(v, str):
        return v.strip().lower() == "true"
    return bool(v)


def _parse_importance(raw: Mapping[str, Any]) -> float:
    value = raw.get("importance", 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def gold_point_from_raw(raw: Mapping[str, Any]) -> GoldMemoryPoint:
    mid = raw.get("memory_id")
    content = raw.get("memory_content")
    return GoldMemoryPoint(
        memory_id=str(mid) if mid is not None else "",
        memory_content=str(content) if content is not None else "",
        memory_type=str(raw.get("memory_type", "")),
        memory_source=str(raw.get("memory_source", "")),
        is_update=_parse_is_update(raw),
        original_memories=_as_str_tuple(raw.get("original_memories")),
        importance=_parse_importance(raw),
        timestamp=(
            str(raw["timestamp"])
            if raw.get("timestamp") is not None
            else None
        ),
        update_type=str(raw.get("update_type", "") or ""),
    )


def build_session_gold_states(
    ordered_session_ids: Sequence[str],
    *,
    s00_memory_points: Sequence[Mapping[str, Any]],
    stage4_by_session_id: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[SessionGoldState, ...]:
    """Accumulate non-interference gold memories through sessions in order.

    S00 is taken from the domain JSON session; later sessions prefer ``stage4``
    rows when present, since those drive staged evaluation labels.
    """
    cumulative: list[GoldMemoryPoint] = []
    states: list[SessionGoldState] = []

    for sid in ordered_session_ids:
        if sid == "S00":
            raw_points: Sequence[Mapping[str, Any]] = s00_memory_points
        else:
            raw_points = stage4_by_session_id.get(sid)
            if raw_points is None:
                raw_points = ()

        news: list[GoldMemoryPoint] = []
        updates: list[GoldMemoryPoint] = []
        interference: list[GoldMemoryPoint] = []

        for raw in raw_points:
            gp = gold_point_from_raw(raw)
            if gp.memory_source == "interference":
                interference.append(gp)
                continue
            cumulative.append(gp)
            if gp.is_update:
                updates.append(gp)
            else:
                news.append(gp)

        states.append(
            SessionGoldState(
                session_id=sid,
                cumulative_gold_memories=tuple(cumulative),
                session_new_memories=tuple(news),
                session_update_memories=tuple(updates),
                session_interference_memories=tuple(interference),
            )
        )

    return tuple(states)
