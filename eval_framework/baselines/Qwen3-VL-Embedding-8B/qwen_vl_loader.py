"""SentenceTransformer loader for Qwen3-VL-Embedding-8B (text + image, 4096-dim).

Loads weights from the **in-repo checkout** at ``./weights/`` (symlink to
the HF cache snapshot). No HF-hub hit at runtime. If ``./weights/`` is
absent, we fall back to the HF model_id string.

Called from ``memory_adapters/qwen_embed_adapter.py`` via sys.path.
Named ``qwen_vl_loader`` (not just ``loader``) so Python's module
cache does not collide with ``qwen_text_loader`` in the sibling
Qwen3-Embedding-8B directory.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_BASELINE_DIR = Path(__file__).resolve().parent
_LOCAL_WEIGHTS = _BASELINE_DIR / "weights"


def load_encoder(model_id: str) -> Any:
    """Instantiate SentenceTransformer in bf16 on cuda:0 (or CPU fallback).

    Prefers the local ``./weights/`` directory over the HF hub model_id.
    Identical body to the text-variant loader — kept as a separate file
    per the project's self-contained-baseline convention.
    """
    from sentence_transformers import SentenceTransformer
    import torch

    if _LOCAL_WEIGHTS.exists():
        model_path: str = str(_LOCAL_WEIGHTS.resolve())
        logger.info("Qwen3-VL-Embedding-8B: loading from local weights %s", model_path)
    else:
        model_path = model_id
        logger.info("Qwen3-VL-Embedding-8B: local weights/ missing, falling back to HF hub %s", model_id)

    explicit_device = os.getenv("QWEN_EMBED_DEVICE")
    if torch.cuda.is_available():
        return SentenceTransformer(
            model_path,
            device=explicit_device or "cuda:0",
            trust_remote_code=True,
            model_kwargs={"dtype": torch.bfloat16},
        )
    return SentenceTransformer(model_path, device="cpu", trust_remote_code=True)
