"""Shared OpenAI vision description utility for multimodal adapters.

Multimodal baselines that cannot ingest raw image tensors can still benefit
from actual image inputs by asking a vision-capable LLM to produce a rich
description, which is then fed as text. This module centralizes that logic
and caches results per image path.
"""

from __future__ import annotations

import base64
import logging
import os
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

_VISION_PROMPT = (
    "Describe this image in a single concise paragraph (<=120 words). "
    "Focus on concrete visible elements: objects, entities, spatial layout, "
    "actions, text/UI elements, screen content. Do not invent details not visible. "
    "Output only the description, no preamble."
)


def _load_client():
    """Return a cached OpenAI client or None if unavailable."""
    try:
        from openai import OpenAI
    except ImportError:
        return None
    try:
        from eval_framework.config import resolve_openai_base_url
        return OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=resolve_openai_base_url(),
        )
    except Exception:
        return None


@lru_cache(maxsize=1)
def _client():
    return _load_client()


def _vision_model() -> str:
    # Vision path uses the same chat model as the rest of the framework.
    from eval_framework.config import resolve_openai_model
    return resolve_openai_model()


@lru_cache(maxsize=4096)
def describe_image(image_path: str) -> Optional[str]:
    """Ask a vision-capable LLM to describe the image at ``image_path``.

    Cached per path. Returns None on any failure.
    """
    if not image_path or not os.path.isfile(image_path):
        return None
    client = _client()
    if client is None:
        return None
    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("ascii")
    except Exception as exc:
        logger.warning("describe_image: cannot read %s: %s", image_path, exc)
        return None

    ext = os.path.splitext(image_path)[1].lstrip(".").lower() or "png"
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}.get(
        ext, "png"
    )
    try:
        resp = client.chat.completions.create(
            model=_vision_model(),
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": _VISION_PROMPT},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/{mime};base64,{img_b64}"
                    }},
                ],
            }],
            max_tokens=256,
        )
        text = (resp.choices[0].message.content or "").strip()
        return text or None
    except Exception as exc:
        logger.warning("describe_image: API call failed for %s: %s", image_path, exc)
        return None


def caption_with_vision(caption: str, image_path: str | None) -> str:
    """Return the best available description for an attachment.

    Priority: vision_model(image_path) > caption > "".
    """
    if image_path:
        desc = describe_image(image_path)
        if desc:
            return desc
    return caption or ""
