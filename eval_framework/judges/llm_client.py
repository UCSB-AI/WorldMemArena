"""OpenAI LLM client for judge calls with retry logic and concurrency control."""

from __future__ import annotations

import os
import re
import logging
import threading
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_random_exponential, before_sleep_log

from eval_framework.json_repair_utils import loads_repaired_or_none

# Load judge/runtime API config from eval_framework/.env.
_EVAL_FRAMEWORK_DIR = Path(__file__).resolve().parents[1]
load_dotenv(_EVAL_FRAMEWORK_DIR / ".env", override=False)

logger = logging.getLogger(__name__)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
_JSON_FENCE_OPEN_RE = re.compile(r"```(?:json)?\s*\n?(.*)", re.DOTALL)

_client: Any = None
_client_lock = threading.Lock()
_semaphore: threading.Semaphore | None = None


def _build_client() -> Any:
    from openai import OpenAI
    from eval_framework.config import resolve_judge_api_key, resolve_judge_base_url
    return OpenAI(
        api_key=resolve_judge_api_key(),
        base_url=resolve_judge_base_url(),
    )


def _get_client() -> Any:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = _build_client()
    return _client


def _get_semaphore() -> threading.Semaphore:
    global _semaphore
    if _semaphore is None:
        from eval_framework.config import resolve_llm_max_concurrent
        _semaphore = threading.Semaphore(resolve_llm_max_concurrent())
    return _semaphore


def _common_params() -> dict[str, Any]:
    from eval_framework.config import (
        resolve_judge_model,
        resolve_judge_temperature,
        resolve_openai_max_tokens,
        resolve_openai_timeout,
    )
    params: dict[str, Any] = {}
    model = resolve_judge_model()
    max_tok = resolve_openai_max_tokens()
    if model.startswith("gpt-5") or model.startswith("o"):
        params["max_completion_tokens"] = max_tok
    else:
        params["max_tokens"] = max_tok
    params["temperature"] = resolve_judge_temperature()
    params["timeout"] = resolve_openai_timeout()
    # Disable DeepSeek thinking/reasoning mode for judge calls (saves tokens, faster)
    if model.startswith("deepseek"):
        params["extra_body"] = {"thinking": {"type": "disabled"}}
    return params


@retry(
    wait=wait_random_exponential(min=2, max=60),
    stop=stop_after_attempt(8),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def llm_request_for_json(prompt: str) -> dict[str, Any]:
    """Send a prompt to the LLM and parse the JSON block from the response.

    Respects global concurrency limit (LLM_MAX_CONCURRENT env var, default 5).
    """
    from eval_framework.config import resolve_judge_model
    sem = _get_semaphore()
    sem.acquire()
    from eval_framework.openai_compat import token_usage_scope
    try:
        client = _get_client()
        model = resolve_judge_model()

        with token_usage_scope() as token_usage:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **_common_params(),
            )
        content = response.choices[0].message.content or ""
    finally:
        sem.release()

    parsed = _extract_json(content)
    if parsed is not None:
        parsed["_llm_usage"] = dict(token_usage)
        return parsed

    raise ValueError(f"No JSON block found in model output: {content[:500]}")


def _extract_json(content: str) -> dict[str, Any] | None:
    """Try to extract a JSON object from model output, with truncation repair."""
    # 1. Closed fence: ```json ... ```
    for match in _JSON_FENCE_RE.finditer(content):
        candidate = match.group(1).strip()
        if candidate.startswith("{"):
            parsed = loads_repaired_or_none(candidate, expected_type=dict)
            if isinstance(parsed, dict):
                return parsed

    # 2. Open fence (output truncated before closing ```): ```json ...EOF
    match = _JSON_FENCE_OPEN_RE.search(content)
    if match:
        candidate = match.group(1).strip()
        if candidate.startswith("{"):
            repaired = _repair_truncated_json(candidate)
            if repaired is not None:
                return repaired

    # 3. Raw JSON without fences
    stripped = content.strip()
    if stripped.startswith("{"):
        parsed = loads_repaired_or_none(stripped, expected_type=dict)
        if isinstance(parsed, dict):
            return parsed
        repaired = _repair_truncated_json(stripped)
        if repaired is not None:
            return repaired

    return None


def _repair_truncated_json(text: str) -> dict[str, Any] | None:
    """Best-effort repair of truncated JSON by closing open brackets/braces."""
    # Remove trailing partial tokens (incomplete key/value after last comma)
    text = re.sub(r',\s*"[^"]*$', "", text)         # trailing partial key
    text = re.sub(r',\s*\{[^}]*$', "", text)         # trailing partial object
    text = re.sub(r',\s*$', "", text)                 # trailing comma

    # Count unclosed brackets and braces, then append closers
    open_braces = text.count("{") - text.count("}")
    open_brackets = text.count("[") - text.count("]")
    suffix = "]" * max(open_brackets, 0) + "}" * max(open_braces, 0)
    candidate = text + suffix

    result = loads_repaired_or_none(candidate, expected_type=dict)
    if isinstance(result, dict):
        logger.warning("Repaired truncated JSON (appended %r)", suffix)
        return result
    return None
