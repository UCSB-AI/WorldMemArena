"""Serialize NormalizedTurn dialogue for optional evaluators (e.g. itemwise judge)."""

from __future__ import annotations

from eval_framework.datasets.schemas import NormalizedTurn


def format_session_dialogue(turns: tuple[NormalizedTurn, ...]) -> str:
    """One block of text: role + body + short attachment captions."""
    lines: list[str] = []
    for t in turns:
        extra = ""
        if t.attachments:
            caps = []
            for a in t.attachments:
                if a.caption:
                    caps.append(a.caption.strip()[:400])
            if caps:
                extra = "\n    [attachments] " + " | ".join(caps)
        lines.append(f"[{t.role.upper()}] {t.text}{extra}")
    return "\n".join(lines)
