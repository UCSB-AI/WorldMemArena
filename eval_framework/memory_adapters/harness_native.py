"""Native-memory adapter for terminal harness baselines.

Pure black-box interactive mode: maintains a persistent agent session.
Dialogue is fed sequentially, and questions are asked at checkpoints.
The agent uses its own context/memory to answer — no framework-imposed
storage or retrieval.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from eval_framework.datasets.schemas import (
    MemoryDeltaRecord,
    MemorySnapshotRecord,
    NormalizedTurn,
    RetrievalRecord,
)
from eval_framework.memory_adapters.base import MemoryAdapter


_BATCH_QA_PROMPT = """Now answer ALL of the following questions based on the conversation sessions I gave you above.

Rules:
- Be concise: single short phrase or under 15 words per answer.
- Only use information from the sessions provided earlier.
- If you truly cannot answer, say exactly: "Not mentioned in memory."

Questions:
{questions_block}

Respond in JSON — a list with one object per question, in order:
[
  {{"q_index": 0, "answer": "..."}},
  {{"q_index": 1, "answer": "..."}},
  ...
]
"""


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", value.strip()).strip("-").lower()
    return slug or "harness"


def _format_turn(turn: NormalizedTurn) -> str:
    header = (
        f"Turn {turn.turn_index} | role={turn.role}"
        + (f" | timestamp={turn.timestamp}" if turn.timestamp else "")
    )
    lines = [header, turn.text]
    if turn.attachments:
        lines.append("Attachments:")
        for att in turn.attachments:
            parts = [f"type={att.type}"]
            if att.image_id:
                parts.append(f"image_id={att.image_id}")
            if att.file_path:
                parts.append(f"file_path={att.file_path}")
            lines.append("- " + ", ".join(parts))
            if att.caption:
                lines.append(f"  caption: {att.caption}")
    return "\n".join(lines)


def _format_session(session_id: str, turns: list[NormalizedTurn]) -> str:
    if not turns:
        return ""
    lines = [_format_turn(turn) for turn in turns]
    return "\n\n".join(lines)


class HarnessMemoryAdapter(MemoryAdapter):
    """MemoryAdapter using a persistent interactive agent session.

    Feeds dialogue and asks questions sequentially in the same session,
    relying on the agent's own context window / memory for recall.
    """

    def __init__(self, *, baseline_name: str) -> None:
        from eval_framework.baselines._clients.harness_api import build_harness_target

        target = build_harness_target(baseline_name)
        if target is None:
            raise RuntimeError(f"Missing harness config for {baseline_name!r}")
        self._baseline_name = baseline_name
        self._target = target
        self._current_turns: list[NormalizedTurn] = []
        self._current_session_id = ""
        self._pending_sessions: list[tuple[str, str]] = []
        self._ingested_session_ids: list[str] = []

        self._openclaw_session: Any = None
        self._openclaw_runtime: Any = None
        self._session_initialized = False

    def _ensure_session(self) -> None:
        """Lazily initialize the persistent OpenClaw session."""
        if self._session_initialized:
            return

        if self._target.kind == "openclaw_general":
            self._init_openclaw_session()
        self._session_initialized = True

    def _init_openclaw_session(self) -> None:
        sys.path.insert(0, str(Path(self._target.runner).resolve().parent))
        from run_openclaw_general import OpenClawLocalRuntime

        storage_root = os.environ.get("EVAL_FRAMEWORK_BASELINE_STORAGE_DIR", "").strip()
        if storage_root:
            workspace = (
                Path(storage_root).expanduser().resolve()
                / "harness_native"
                / _slug(self._baseline_name)
            )
        else:
            workspace = None

        run_id = f"harness-eval-{uuid.uuid4().hex[:8]}"
        self._openclaw_runtime = OpenClawLocalRuntime()
        from eval_framework.baselines._clients.harness_api import harness_env_overrides

        overrides = harness_env_overrides(self._target)
        previous = {key: os.environ.get(key) for key in overrides}
        try:
            os.environ.update(overrides)
            self._openclaw_session = self._openclaw_runtime.prepare_session(
                run_id=run_id,
                model=self._target.model,
                workspace=workspace,
                timeout_seconds=self._target.timeout,
            )
            openclaw_bin = shutil.which("openclaw")
            if openclaw_bin and Path(openclaw_bin).exists():
                self._openclaw_session = replace(
                    self._openclaw_session,
                    command=(openclaw_bin, *self._openclaw_session.command[1:]),
                )
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def _send_message(self, message: str) -> str:
        """Send a message to the persistent session and return the response."""
        self._ensure_session()

        if self._target.kind == "openclaw_general":
            return self._send_openclaw_message(message)
        elif self._target.kind == "codex_exec":
            return self._send_codex_message(message)
        raise RuntimeError(f"Unsupported harness kind: {self._target.kind}")

    def _send_openclaw_message(self, message: str) -> str:
        session = self._openclaw_session
        started_at = time.time()
        completed = subprocess.run(
            [*session.command, "--message", message],
            cwd=str(session.workspace),
            env=session.env,
            capture_output=True,
            text=True,
            timeout=self._target.timeout + 30,
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(
                f"{self._baseline_name}: openclaw failed exit {completed.returncode}: {detail[:500]}"
            )

        from run_openclaw_general import _read_final_output
        final_output, _ = _read_final_output(session.session_path, created_after=started_at)
        if final_output:
            return final_output
        return completed.stdout.strip()

    def _send_codex_message(self, message: str) -> str:
        import tempfile
        from eval_framework.baselines._clients.harness_api import _call_codex_exec_prompt
        raw, _ = _call_codex_exec_prompt(
            self._target,
            prompt=message,
            image_paths=None,
            max_images=0,
            max_image_bytes=0,
        )
        return raw

    def reset(self) -> None:
        if self._openclaw_runtime is not None:
            self._openclaw_runtime.cleanup()
        self._openclaw_session = None
        self._openclaw_runtime = None
        self._session_initialized = False
        self._current_turns = []
        self._current_session_id = ""
        self._pending_sessions = []
        self._ingested_session_ids = []

    def ingest_turn(self, turn: NormalizedTurn) -> None:
        if self._current_session_id and turn.session_id != self._current_session_id:
            self.end_session(self._current_session_id)
        self._current_session_id = turn.session_id
        self._current_turns.append(turn)

    def end_session(self, session_id: str) -> None:
        if not self._current_turns:
            return
        dialogue = _format_session(session_id, self._current_turns)
        self._pending_sessions.append((session_id, dialogue))
        self._current_turns = []
        self._current_session_id = ""

    def flush_ingest(self) -> None:
        """Send pending session dialogues to the agent."""
        if self._current_turns:
            self.end_session(self._current_session_id or self._current_turns[-1].session_id)
        if not self._pending_sessions:
            return

        for session_id, dialogue in self._pending_sessions:
            msg = (
                f"Here is a new conversation session (Session {session_id}). "
                f"Remember all the details — you will be asked questions about it later.\n\n"
                f"--- SESSION {session_id} ---\n"
                f"{dialogue}\n"
                f"--- END SESSION ---"
            )
            self._send_message(msg)
            self._ingested_session_ids.append(session_id)
            print(f"    [{session_id}] fed to agent", flush=True)

        self._pending_sessions = []

    def answer_batch(self, questions: list[str]) -> list[str]:
        """Ask all questions and parse answers from the agent."""
        from eval_framework.cli import _parse_batch_answer_json

        self.flush_ingest()

        questions_block = "\n".join(
            f"Q{i}: {q}" for i, q in enumerate(questions)
        )
        prompt = _BATCH_QA_PROMPT.format(questions_block=questions_block)

        raw = self._send_message(prompt)
        return _parse_batch_answer_json(raw, len(questions))

    def snapshot_memories(self) -> list[MemorySnapshotRecord]:
        return []

    def export_memory_delta(self, session_id: str) -> list[MemoryDeltaRecord]:
        del session_id
        return []

    def retrieve(self, query: str, top_k: int, **_: Any) -> RetrievalRecord:
        self.flush_ingest()
        trace: dict[str, Any] = {
            "baseline": self._baseline_name,
            "mode": "harness_native_memory",
            "harness_native_memory": True,
            "harness_ingested_sessions": list(self._ingested_session_ids),
        }
        return RetrievalRecord(query=query, top_k=top_k, items=[], raw_trace=trace)

    def get_capabilities(self) -> dict[str, Any]:
        return {
            "available": True,
            "backend": "HarnessNativeMemory",
            "baseline": self._baseline_name,
            "storage": "terminal_harness_owned",
            "retrieval": "terminal_harness_owned",
            "framework_snapshots": False,
            "framework_retrieval_items": False,
        }
