"""Unified memory evaluation framework (package scaffold)."""

from __future__ import annotations

import os as _os
from pathlib import Path as _Path

from dotenv import load_dotenv as _load_dotenv

# Load .env from the project root (WorldMemArena/.env) exactly once at package
# import time so every adapter / judge / cli import sees the same environment.
# Falls back to eval_framework/.env for backwards compatibility.
#
# override=False: env vars already set by the parent process (e.g. wrapper
# shells that export OPENAI_MODEL / OPENAI_BASE_URL to redirect the
# system-under-test to a local vLLM endpoint) take precedence over the .env
# defaults.  Keys absent from os.environ are filled in from .env as a fallback.
_PKG_DIR = _Path(__file__).resolve().parent
for _candidate in (_PKG_DIR.parent / ".env", _PKG_DIR / ".env"):
    if _candidate.is_file():
        _load_dotenv(_candidate, override=False)
        break

# This benchmark should be reproducible/offline for HuggingFace assets.  Model
# paths in config.yaml point at local snapshots; these flags prevent upstream
# libraries from issuing metadata HEAD requests to huggingface.co.
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
_os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def _normalize_openai_env() -> None:
    """Cross-wire OPENAI_BASE_URL <-> OPENAI_API_BASE for upstream baselines.

    Different upstream packages read different env names for the OpenAI
    base URL:
    - MemVerse (``build_memory.py``, LightRAG ``openai.py``), SimpleMem
      (``omni_memory.app``), ViLoMem and mem0/embedchain read
      ``OPENAI_API_BASE``.
    - Our adapters + cli + most other upstream SDKs read
      ``OPENAI_BASE_URL`` (the modern OpenAI SDK name).

    To keep ``.env`` a single source of truth we mirror whichever the
    user sets into the other name.  Same for API key and embedding
    endpoints.
    """
    pairs = (
        ("OPENAI_BASE_URL", "OPENAI_API_BASE"),
        ("OPENAI_API_KEY", "OPENAI_KEY"),
        ("OPENAI_EMBEDDING_BASE_URL", "OPENAI_EMBEDDING_API_BASE"),
    )
    for a, b in pairs:
        va = _os.environ.get(a)
        vb = _os.environ.get(b)
        if va and not vb:
            _os.environ[b] = va
        elif vb and not va:
            _os.environ[a] = vb


_normalize_openai_env()

__all__: list[str] = []
