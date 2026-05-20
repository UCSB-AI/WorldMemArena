"""Compatibility helpers for OpenAI chat completions across model families."""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from eval_framework.json_repair_utils import loads_repaired


TokenUsage = dict[str, int]

_CURRENT_TOKEN_USAGE: ContextVar[TokenUsage | None] = ContextVar(
    "eval_framework_token_usage", default=None
)
_API_LOG_LOCK = threading.Lock()


def empty_token_usage() -> TokenUsage:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def extract_token_usage(response: Any) -> TokenUsage:
    usage = getattr(response, "usage", None)
    if usage is None:
        return empty_token_usage()
    return {
        "prompt_tokens": _as_int(getattr(usage, "prompt_tokens", None)),
        "completion_tokens": _as_int(getattr(usage, "completion_tokens", None)),
        "total_tokens": _as_int(getattr(usage, "total_tokens", None)),
    }


def merge_token_usage(*usages: Any) -> TokenUsage:
    out = empty_token_usage()
    for usage in usages:
        if not isinstance(usage, dict):
            continue
        out["prompt_tokens"] += _as_int(usage.get("prompt_tokens"))
        out["completion_tokens"] += _as_int(usage.get("completion_tokens"))
        out["total_tokens"] += _as_int(usage.get("total_tokens"))
    return out


def _record_response_usage(response: Any) -> None:
    bucket = _CURRENT_TOKEN_USAGE.get()
    if bucket is None:
        return
    merged = merge_token_usage(bucket, extract_token_usage(response))
    bucket.update(merged)


@contextmanager
def token_usage_scope() -> Any:
    """Collect sync OpenAI chat-completion token usage inside this context."""
    bucket = empty_token_usage()
    token = _CURRENT_TOKEN_USAGE.set(bucket)
    try:
        yield bucket
    finally:
        _CURRENT_TOKEN_USAGE.reset(token)


def rewrite_chat_completion_kwargs(payload: dict[str, Any]) -> dict[str, Any]:
    """Translate deprecated chat completion parameters for reasoning models."""
    rewritten = dict(payload)
    model = str(rewritten.get("model") or "")
    if (
        model.startswith("gpt-5")
        and "max_tokens" in rewritten
        and "max_completion_tokens" not in rewritten
    ):
        rewritten["max_completion_tokens"] = rewritten.pop("max_tokens")
    return rewritten


def _safe_len(value: Any) -> int:
    try:
        return len(value)
    except TypeError:
        return 0


def _client_base_url(resource: Any) -> str:
    client = getattr(resource, "_client", None)
    base_url = getattr(client, "base_url", None)
    return str(base_url or "")


def _host_from_url(raw: str) -> str:
    if not raw:
        return ""
    try:
        parsed = urlparse(raw)
        return parsed.netloc or parsed.path
    except Exception:
        return ""


def _count_images(value: Any) -> int:
    if isinstance(value, list):
        total = 0
        for item in value:
            total += _count_images(item)
        return total
    if isinstance(value, dict):
        typ = str(value.get("type") or "")
        total = 1 if typ in {"image_url", "image"} else 0
        for child in value.values():
            if isinstance(child, (dict, list)):
                total += _count_images(child)
        return total
    return 0


def _input_count(value: Any) -> int:
    if isinstance(value, str):
        return 1
    if isinstance(value, list):
        return len(value)
    return 0


def _write_api_call_log(event: dict[str, Any]) -> None:
    path = os.getenv("EVAL_API_CALL_LOG", "").strip()
    if not path:
        return
    event.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    event.setdefault("pid", os.getpid())
    event.setdefault("baseline", os.getenv("EVAL_API_LOG_BASELINE", ""))
    event.setdefault("domain", os.getenv("EVAL_API_LOG_DOMAIN", ""))
    event.setdefault("output_dir", os.getenv("EVAL_API_LOG_OUTPUT_DIR", ""))
    try:
        log_path = Path(path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
        with _API_LOG_LOCK:
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(line)
    except Exception:
        # Logging must never change eval behavior.
        return


def _log_openai_call(
    *,
    resource: Any,
    endpoint: str,
    kwargs: dict[str, Any],
    start: float,
    status: str,
    response: Any = None,
    error: Exception | None = None,
) -> None:
    base_url = _client_base_url(resource)
    event: dict[str, Any] = {
        "endpoint": endpoint,
        "model": str(kwargs.get("model") or ""),
        "base_url_host": _host_from_url(base_url),
        "duration_seconds": round(time.perf_counter() - start, 6),
        "status": status,
    }
    if endpoint == "chat.completions.create":
        messages = kwargs.get("messages")
        event["message_count"] = _safe_len(messages) if isinstance(messages, list) else 0
        event["image_count"] = _count_images(messages)
    elif endpoint == "embeddings.create":
        event["input_count"] = _input_count(kwargs.get("input"))
    if response is not None:
        event["usage"] = extract_token_usage(response)
    if error is not None:
        event["error_type"] = error.__class__.__name__
        event["error_message"] = str(error)[:500]
    _write_api_call_log(event)


def patch_openai_chat_completions() -> bool:
    """Monkeypatch OpenAI SDK calls for compatibility and metadata logging."""
    patch_httpx_response_json_repair()
    try:
        from openai.resources.chat.completions.completions import Completions
        from openai.resources.embeddings import Embeddings
    except Exception:
        return False

    chat_current = Completions.create
    embeddings_current = Embeddings.create
    if (
        getattr(chat_current, "_eval_framework_patched", False)
        and getattr(embeddings_current, "_eval_framework_patched", False)
    ):
        return True

    original_chat_create = chat_current
    original_embeddings_create = embeddings_current

    def _patched_create(self: Any, *args: Any, **kwargs: Any) -> Any:
        rewritten = rewrite_chat_completion_kwargs(kwargs)
        start = time.perf_counter()
        try:
            response = original_chat_create(self, *args, **rewritten)
            _record_response_usage(response)
            _log_openai_call(
                resource=self,
                endpoint="chat.completions.create",
                kwargs=rewritten,
                start=start,
                status="ok",
                response=response,
            )
            return response
        except Exception as exc:
            if (
                "Unsupported parameter: 'max_tokens'" in str(exc)
                and "max_tokens" in kwargs
            ):
                retried = rewrite_chat_completion_kwargs(kwargs)
                try:
                    response = original_chat_create(self, *args, **retried)
                    _record_response_usage(response)
                    _log_openai_call(
                        resource=self,
                        endpoint="chat.completions.create",
                        kwargs=retried,
                        start=start,
                        status="ok",
                        response=response,
                    )
                    return response
                except Exception as retry_exc:
                    _log_openai_call(
                        resource=self,
                        endpoint="chat.completions.create",
                        kwargs=retried,
                        start=start,
                        status="error",
                        error=retry_exc,
                    )
                    raise
            _log_openai_call(
                resource=self,
                endpoint="chat.completions.create",
                kwargs=rewritten,
                start=start,
                status="error",
                error=exc,
            )
            raise

    def _patched_embeddings_create(self: Any, *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            response = original_embeddings_create(self, *args, **kwargs)
            _log_openai_call(
                resource=self,
                endpoint="embeddings.create",
                kwargs=kwargs,
                start=start,
                status="ok",
                response=response,
            )
            return response
        except Exception as exc:
            _log_openai_call(
                resource=self,
                endpoint="embeddings.create",
                kwargs=kwargs,
                start=start,
                status="error",
                error=exc,
            )
            raise

    _patched_create._eval_framework_patched = True  # type: ignore[attr-defined]
    _patched_embeddings_create._eval_framework_patched = True  # type: ignore[attr-defined]
    Completions.create = _patched_create  # type: ignore[assignment]
    Embeddings.create = _patched_embeddings_create  # type: ignore[assignment]
    return True


def patch_httpx_response_json_repair() -> bool:
    """Fallback to json_repair for malformed JSON response bodies."""
    try:
        import httpx
    except Exception:
        return False

    current = httpx.Response.json
    if getattr(current, "_eval_framework_json_repair_patched", False):
        return True

    original_json = current

    def _patched_json(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return original_json(self, *args, **kwargs)
        except json.JSONDecodeError as exc:
            try:
                repaired = loads_repaired(self.text)
            except Exception:
                raise exc
            if not isinstance(repaired, (dict, list)):
                raise exc
            _write_api_call_log(
                {
                    "endpoint": "httpx.Response.json",
                    "status": "json_repaired",
                    "status_code": getattr(self, "status_code", None),
                    "body_chars": len(getattr(self, "text", "") or ""),
                }
            )
            return repaired

    _patched_json._eval_framework_json_repair_patched = True  # type: ignore[attr-defined]
    httpx.Response.json = _patched_json  # type: ignore[assignment]
    return True
