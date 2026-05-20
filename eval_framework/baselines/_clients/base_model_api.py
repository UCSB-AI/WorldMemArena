"""Runtime helpers for BaseModel answer targets.

BaseModel variants are configured in ``base_model_config.yaml`` and exposed as
``BaseModel-<provider>`` baselines. Most providers are OpenAI-compatible; direct
Anthropic endpoints use the Anthropic SDK.
"""

from __future__ import annotations

import base64
import mimetypes
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval_framework.openai_compat import (
    empty_token_usage,
    extract_token_usage,
    rewrite_chat_completion_kwargs,
)


@dataclass(frozen=True)
class ChatTarget:
    provider: str
    baseline_name: str
    model: str
    base_url: str
    api_key: str
    mm_mode: str
    temperature: float
    max_tokens: int
    timeout: int
    # Per-target image caps for the answer stage. -1 means "fall back to the
    # global config.yaml base_model.max_images_per_qa / max_image_bytes_mb".
    max_images: int = -1
    max_image_bytes: int = -1


def _normalize_mm_mode(raw: Any) -> str:
    mode = str(raw or "text").strip().lower()
    if mode in {"image", "mm", "multimodal"}:
        return "image"
    return "text"


def _resolve_api_key(raw: Any, env_name: Any = None) -> str:
    if raw:
        return str(raw)
    if env_name:
        return os.getenv(str(env_name), "")
    return ""


def build_default_target() -> ChatTarget:
    """Target for normal memory baselines: config.yaml + OPENAI_API_KEY."""
    from eval_framework.config import (
        resolve_openai_base_url,
        resolve_openai_max_tokens,
        resolve_openai_model,
        resolve_openai_temperature,
        resolve_openai_timeout,
    )

    return ChatTarget(
        provider="default",
        baseline_name="",
        model=resolve_openai_model(),
        base_url=resolve_openai_base_url(),
        api_key=os.getenv("OPENAI_API_KEY") or "",
        mm_mode="image",
        temperature=resolve_openai_temperature(),
        max_tokens=resolve_openai_max_tokens(),
        timeout=resolve_openai_timeout(),
    )


def build_base_model_target(baseline_name: str) -> ChatTarget | None:
    """Target for ``BaseModel-<provider>`` baselines."""
    from eval_framework.config import (
        base_model_baseline_name,
        resolve_base_model_config_for_baseline,
        resolve_openai_max_tokens,
        resolve_openai_temperature,
        resolve_openai_timeout,
    )

    cfg = resolve_base_model_config_for_baseline(baseline_name)
    if cfg is None:
        return None
    provider = str(cfg.get("key") or baseline_name)
    canonical_baseline = str(cfg.get("baseline") or base_model_baseline_name(provider))
    raw_max_images = cfg.get("max_images_per_qa")
    raw_max_image_mb = cfg.get("max_image_bytes_mb")
    max_images = int(raw_max_images) if raw_max_images is not None else -1
    max_image_bytes = (
        int(raw_max_image_mb) * 1024 * 1024 if raw_max_image_mb is not None else -1
    )
    return ChatTarget(
        provider=provider,
        baseline_name=canonical_baseline,
        model=str(cfg.get("model") or ""),
        base_url=str(cfg.get("base_url") or "https://api.openai.com/v1"),
        api_key=_resolve_api_key(cfg.get("api_key"), cfg.get("api_key_env")),
        mm_mode=_normalize_mm_mode(cfg.get("mm_mode")),
        temperature=float(cfg.get("temperature", resolve_openai_temperature())),
        max_tokens=int(cfg.get("max_tokens", resolve_openai_max_tokens())),
        timeout=int(cfg.get("timeout", resolve_openai_timeout())),
        max_images=max_images,
        max_image_bytes=max_image_bytes,
    )


def target_for_baseline(baseline_name: str) -> ChatTarget:
    from eval_framework.config import resolve_base_model_baseline_names

    name = str(baseline_name or "").strip()
    if name == "BaseModel" or name.startswith("BaseModel-"):
        target = build_base_model_target(baseline_name)
        if target is not None:
            return target
        configured = ", ".join(resolve_base_model_baseline_names()) or "none"
        raise RuntimeError(
            f"Unknown BaseModel baseline {name!r}. Configured BaseModel baselines: {configured}"
        )
    return build_default_target()


def is_anthropic_target(target: ChatTarget) -> bool:
    base = target.base_url.lower()
    model = target.model.lower()
    provider = target.provider.lower()
    return "anthropic" in base or model.startswith("claude") or provider == "claude"


def _encode_image(path: str) -> dict[str, Any] | None:
    try:
        raw = Path(path).read_bytes()
    except OSError:
        return None
    mt, _ = mimetypes.guess_type(path)
    mt = mt or "image/png"
    b64 = base64.b64encode(raw).decode("ascii")
    return {
        "path": path,
        "media_type": mt,
        "b64": b64,
        "data_url": f"data:{mt};base64,{b64}",
        "num_bytes": len(raw),
    }


def collect_image_payloads(
    image_paths: list[str],
    *,
    max_images: int,
    max_image_bytes: int,
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    seen: set[str] = set()
    total = 0
    for path in image_paths:
        if not path or path in seen:
            continue
        seen.add(path)
        encoded = _encode_image(path)
        if encoded is None:
            continue
        raw_bytes = int(encoded["num_bytes"])
        if total + raw_bytes > max_image_bytes:
            continue
        payloads.append(encoded)
        total += raw_bytes
        if len(payloads) >= max_images:
            break
    return payloads


def _anthropic_text_and_usage(response: Any) -> tuple[str, dict[str, int]]:
    parts: list[str] = []
    for block in getattr(response, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(str(text))
    usage_obj = getattr(response, "usage", None)
    if usage_obj is None:
        return "\n".join(parts), empty_token_usage()
    input_tokens = int(getattr(usage_obj, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage_obj, "output_tokens", 0) or 0)
    return "\n".join(parts), {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def call_chat_target(
    target: ChatTarget,
    *,
    system_prompt: str,
    user_prompt: str,
    image_paths: list[str] | None = None,
    max_images: int = 0,
    max_image_bytes: int = 0,
) -> tuple[str, dict[str, int]]:
    """Call a configured chat target and return ``(raw_text, token_usage)``."""
    if not target.api_key:
        raise RuntimeError(f"{target.provider}: missing API key")
    if not target.model:
        raise RuntimeError(f"{target.provider}: missing model")

    def should_retry(exc: Exception) -> bool:
        text = f"{exc.__class__.__name__}: {exc}".lower()
        return any(x in text for x in ("timeout", "connection", "rate limit", "temporarily"))

    def with_retries(fn):
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                return fn()
            except Exception as exc:
                last_exc = exc
                if attempt == 2 or not should_retry(exc):
                    raise
                time.sleep(2 * (attempt + 1))
        assert last_exc is not None
        raise last_exc

    payloads = collect_image_payloads(
        image_paths or [],
        max_images=max_images,
        max_image_bytes=max_image_bytes,
    )

    if is_anthropic_target(target):
        from anthropic import Anthropic

        client = Anthropic(
            api_key=target.api_key,
            base_url=target.base_url,
            timeout=target.timeout,
            max_retries=0,
        )
        content: Any
        if payloads:
            content = [{"type": "text", "text": user_prompt}]
            content.extend(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": p["media_type"],
                        "data": p["b64"],
                    },
                }
                for p in payloads
            )
        else:
            content = user_prompt
        response = with_retries(
            lambda: client.messages.create(
                model=target.model,
                system=system_prompt,
                messages=[{"role": "user", "content": content}],
                temperature=target.temperature,
                max_tokens=target.max_tokens,
            )
        )
        return _anthropic_text_and_usage(response)

    from openai import OpenAI

    client = OpenAI(
        api_key=target.api_key,
        base_url=target.base_url,
        timeout=target.timeout,
        max_retries=0,
    )
    user_content: Any = user_prompt
    if payloads:
        user_content = [{"type": "text", "text": user_prompt}]
        user_content.extend(
            {"type": "image_url", "image_url": {"url": p["data_url"]}}
            for p in payloads
        )
    request_kwargs = rewrite_chat_completion_kwargs(
        {
            "model": target.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": target.temperature,
            "max_tokens": target.max_tokens,
        }
    )
    response = with_retries(lambda: client.chat.completions.create(**request_kwargs))
    if isinstance(response, str):
        raw = response
        usage = empty_token_usage()
    elif isinstance(response, dict):
        choices = response.get("choices") or []
        message = choices[0].get("message", {}) if choices else {}
        raw = str(message.get("content") or "")
        usage = empty_token_usage()
    elif isinstance(response, list):
        # json_repair fallback (openai_compat.patch_httpx_response_json_repair)
        # can recover a truncated body as a list. Treat as empty answer instead
        # of crashing the sample.
        raw = ""
        usage = empty_token_usage()
    else:
        raw = response.choices[0].message.content or ""
        usage = extract_token_usage(response)
    return raw, usage
