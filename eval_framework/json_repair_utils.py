"""JSON parsing helpers with json_repair fallback."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def loads_repaired(raw: str) -> Any:
    """Parse JSON using json_repair's strict-first parser when available."""
    try:
        import json_repair
    except Exception:
        return json.loads(raw)
    return json_repair.loads(raw)


def loads_repaired_or_none(
    raw: str,
    *,
    expected_type: type | tuple[type, ...] | None = None,
) -> Any | None:
    try:
        parsed = loads_repaired(raw)
    except Exception as exc:
        logger.debug("json_repair parse failed: %s", exc)
        return None
    if expected_type is not None and not isinstance(parsed, expected_type):
        return None
    return parsed
