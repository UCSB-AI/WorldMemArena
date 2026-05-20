"""Configuration types for eval runs.

``config.yaml`` provides defaults for non-secret params. Runtime environment
variables loaded from ``eval_framework/.env`` may override model, endpoint,
timeout, token, concurrency, and API-key settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os
import re

import yaml


_CONFIG_YAML = Path(__file__).resolve().parent / "config.yaml"
_BASE_MODEL_CONFIG_YAML = Path(__file__).resolve().parent / "baselines" / "_clients" / "base_model_config.yaml"
_HARNESS_CONFIG_YAML = Path(__file__).resolve().parent / "baselines" / "_clients" / "harness_config.yaml"
_cache: dict[str, Any] | None = None
_base_model_cache: dict[str, Any] | None = None
_harness_cache: dict[str, Any] | None = None


def _env(name: str) -> str | None:
    value = os.getenv(name)
    return value if value else None


def _env_int(name: str, fallback: Any) -> int:
    return int(_env(name) or fallback)


def _env_float(name: str, fallback: Any) -> float:
    return float(_env(name) or fallback)


def load_baseline_config(path: Path | None = None) -> dict[str, Any]:
    """Load config.yaml (cached)."""
    global _cache
    if _cache is not None and path is None:
        return _cache
    p = path or _CONFIG_YAML
    if not p.exists():
        cfg: dict[str, Any] = {}
    else:
        with p.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    if path is None:
        _cache = cfg
    return cfg


def load_base_model_config(path: Path | None = None) -> dict[str, Any]:
    """Load base_model_config.yaml (cached).

    This file enumerates the long-context BaseModel variants. Each top-level key
    becomes a runnable baseline named ``BaseModel-<key>``.
    """
    global _base_model_cache
    if _base_model_cache is not None and path is None:
        return _base_model_cache
    p = path or _BASE_MODEL_CONFIG_YAML
    if not p.exists():
        cfg: dict[str, Any] = {}
    else:
        with p.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    if path is None:
        _base_model_cache = cfg
    return cfg


def load_harness_config(path: Path | None = None) -> dict[str, Any]:
    """Load harness_config.yaml (cached).

    Each top-level key becomes a runnable baseline named ``Harness-<key>``.
    Harness baselines use a native-memory terminal adapter: checkpoint-prefix
    dialogue is sent to the harness, and both storage and QA recall are owned
    by that harness.
    """
    global _harness_cache
    if _harness_cache is not None and path is None:
        return _harness_cache
    p = path or _HARNESS_CONFIG_YAML
    if not p.exists():
        cfg: dict[str, Any] = {}
    else:
        with p.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    if path is None:
        _harness_cache = cfg
    return cfg


def _get(cfg: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def _base_model_entries() -> list[tuple[str, dict[str, Any]]]:
    cfg = load_base_model_config()
    return [
        (str(key), value)
        for key, value in cfg.items()
        if isinstance(value, dict)
    ]


def _base_model_slug(key: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", str(key).strip()).strip("-").lower()
    return slug or "default"


def _harness_entries() -> list[tuple[str, dict[str, Any]]]:
    cfg = load_harness_config()
    return [
        (str(key), value)
        for key, value in cfg.items()
        if isinstance(value, dict)
    ]


def _harness_slug(key: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", str(key).strip()).strip("-").lower()
    return slug or "default"


def base_model_baseline_name(key: str) -> str:
    return f"BaseModel-{_base_model_slug(key)}"


def harness_baseline_name(key: str) -> str:
    return f"Harness-{_harness_slug(key)}"


def iter_base_model_baselines() -> list[tuple[str, str, dict[str, Any]]]:
    """Return ``(baseline_name, key, config)`` rows from base_model_config.yaml."""
    return [
        (base_model_baseline_name(key), key, dict(value))
        for key, value in _base_model_entries()
    ]


def resolve_base_model_baseline_names() -> list[str]:
    return [name for name, _, _ in iter_base_model_baselines()]


def resolve_base_model_key(baseline_name: str) -> str | None:
    """Map ``BaseModel-foo`` or legacy ``BaseModel`` to a config key."""
    name = str(baseline_name or "").strip()
    entries = _base_model_entries()
    if not entries:
        return None
    if name == "BaseModel":
        keys = [key for key, _ in entries]
        return "openai" if "openai" in keys else keys[0]
    prefix = "BaseModel-"
    if not name.startswith(prefix):
        return None
    requested = name[len(prefix):].strip().lower()
    for key, _ in entries:
        if _base_model_slug(key) == requested:
            return key
    return None


def is_base_model_baseline(baseline_name: str) -> bool:
    return resolve_base_model_key(baseline_name) is not None


def resolve_base_model_config_for_baseline(baseline_name: str) -> dict[str, Any] | None:
    key = resolve_base_model_key(baseline_name)
    if key is None:
        return None
    cfg = load_base_model_config().get(key)
    if not isinstance(cfg, dict):
        return None
    out = dict(cfg)
    out.setdefault("key", key)
    out.setdefault("baseline", base_model_baseline_name(key))
    default_labels = {
        "openai": "OpenAI",
        "deepseek": "DeepSeek",
        "qwen": "Qwen",
        "gemini": "Gemini",
        "claude": "Claude",
    }
    out.setdefault(
        "label",
        str(out.get("name") or default_labels.get(key.lower()) or key)
        .replace("_", " ")
        .title()
        if key.lower() not in default_labels
        else default_labels[key.lower()],
    )
    return out


def resolve_base_model_display_label(baseline_name: str) -> str:
    cfg = resolve_base_model_config_for_baseline(baseline_name)
    if cfg is None:
        return str(baseline_name)
    return str(cfg.get("label") or cfg.get("key") or baseline_name)


def iter_harness_baselines() -> list[tuple[str, str, dict[str, Any]]]:
    """Return ``(baseline_name, key, config)`` rows from harness_config.yaml."""
    return [
        (str(value.get("baseline") or harness_baseline_name(key)), key, dict(value))
        for key, value in _harness_entries()
    ]


def resolve_harness_baseline_names() -> list[str]:
    return [name for name, _, _ in iter_harness_baselines()]


def resolve_harness_key(baseline_name: str) -> str | None:
    """Map ``Harness-foo`` or legacy ``Harness`` to a config key."""
    name = str(baseline_name or "").strip()
    entries = _harness_entries()
    if not entries:
        return None
    if name == "Harness":
        keys = [key for key, _ in entries]
        return "openclaw" if "openclaw" in keys else keys[0]
    prefix = "Harness-"
    if not name.startswith(prefix):
        return None
    for key, cfg in entries:
        explicit = str(cfg.get("baseline") or "").strip()
        if explicit and explicit == name:
            return key
    requested = name[len(prefix):].strip().lower()
    for key, _ in entries:
        if _harness_slug(key) == requested:
            return key
    return None


def is_harness_baseline(baseline_name: str) -> bool:
    return resolve_harness_key(baseline_name) is not None


def is_answer_only_eval_baseline(baseline_name: str) -> bool:
    """Baselines evaluated on checkpoint answers only (no session memory / QA retrieval)."""
    return is_base_model_baseline(baseline_name) or is_harness_baseline(baseline_name)


def resolve_harness_config_for_baseline(baseline_name: str) -> dict[str, Any] | None:
    key = resolve_harness_key(baseline_name)
    if key is None:
        return None
    cfg = load_harness_config().get(key)
    if not isinstance(cfg, dict):
        return None
    out = dict(cfg)
    out.setdefault("key", key)
    out.setdefault("baseline", harness_baseline_name(key))
    out.setdefault(
        "label",
        str(out.get("name") or key).replace("_", " ").title(),
    )
    return out


# ─── Resolver helpers (.env overrides config.yaml defaults) ────────────


def resolve_openai_model() -> str:
    model = _env("OPENAI_MODEL")
    if not model:
        raise RuntimeError(
            "OPENAI_MODEL is required. Set it in eval_framework/.env or the process environment."
        )
    return model


def resolve_openai_base_url() -> str:
    """Chat base_url: prefer ``OPENAI_BASE_URL`` env var, else ``llm.base_url``
    in ``config.yaml``. Allows .env to override the proxy endpoint without
    editing the committed config."""
    return (
        _env("OPENAI_BASE_URL")
        or _get(load_baseline_config(), "llm", "base_url", default="https://api.openai.com/v1")
    )


def resolve_openai_temperature() -> float:
    return _env_float(
        "OPENAI_TEMPERATURE",
        _get(load_baseline_config(), "llm", "temperature", default=0.0),
    )


def resolve_openai_max_tokens() -> int:
    return _env_int(
        "OPENAI_MAX_TOKENS",
        _get(load_baseline_config(), "llm", "max_tokens", default=1024),
    )


def resolve_openai_timeout() -> int:
    return _env_int(
        "OPENAI_TIMEOUT",
        _get(load_baseline_config(), "llm", "timeout", default=120),
    )


def resolve_llm_max_concurrent() -> int:
    return _env_int(
        "LLM_MAX_CONCURRENT",
        _get(load_baseline_config(), "llm", "max_concurrent", default=5),
    )


def resolve_embedding_model() -> str:
    return _env("OPENAI_EMBEDDING_MODEL") or _get(
        load_baseline_config(),
        "embedding",
        "model",
        default="text-embedding-3-small",
    )


def resolve_embedding_dims() -> int:
    return _env_int(
        "OPENAI_EMBEDDING_DIMS",
        _get(load_baseline_config(), "embedding", "dimensions", default=1536),
    )


def resolve_embedding_base_url() -> str:
    """Embedding base_url: prefer ``OPENAI_EMBEDDING_BASE_URL`` env var, then
    ``embedding.base_url`` from config.yaml, else fall back to the chat
    base_url (``OPENAI_BASE_URL`` env / ``llm.base_url``)."""
    return (
        _env("OPENAI_EMBEDDING_BASE_URL")
        or _get(load_baseline_config(), "embedding", "base_url")
        or resolve_openai_base_url()
    )


def resolve_judge_model() -> str:
    """Judge model: judge.model if set, else OPENAI_MODEL."""
    return (
        _env("OPENAI_MODEL_JUDGE")
        or _env("JUDGE_MODEL")
        or _get(load_baseline_config(), "judge", "model")
        or resolve_openai_model()
    )


def resolve_judge_base_url() -> str:
    return _env("OPENAI_BASE_URL_JUDGE") or resolve_openai_base_url()


def resolve_judge_api_key() -> str:
    return _env("OPENAI_API_KEY_JUDGE") or _env("OPENAI_API_KEY") or ""


def resolve_judge_temperature() -> float:
    temperature = _env_float(
        "JUDGE_TEMPERATURE",
        _get(load_baseline_config(), "judge", "temperature", default=0.0),
    )
    judge_model = resolve_judge_model().lower()
    if judge_model.startswith("gpt-5-nano") and temperature == 0.0:
        return 1.0
    return temperature


def resolve_local_sentence_encoder() -> str:
    return _get(load_baseline_config(), "local", "sentence_encoder", default="all-MiniLM-L6-v2")


def resolve_local_model(key: str, env_var: str = "", default: str = "") -> str:
    """Generic resolver for local HF models — reads ``local.<key>`` from config.yaml.

    The ``env_var`` / ``default`` arguments are kept for backwards-compat with
    callers; they are no longer consulted (config.yaml is authoritative).
    """
    del env_var
    return _get(load_baseline_config(), "local", key, default=default)


def resolve_base_model_mode() -> str:
    """Base Model answer mode: 'text' (no images) or 'image' (attach images)."""
    mode = str(_get(load_baseline_config(), "base_model", "mode", default="image")).strip().lower()
    if mode in {"mm", "multimodal"}:
        mode = "image"
    if mode not in {"text", "image"}:
        raise ValueError(f"base_model.mode must be 'text' or 'image', got {mode!r}")
    return mode


def resolve_base_model_max_images() -> int:
    return int(_get(load_baseline_config(), "base_model", "max_images_per_qa", default=5))


def resolve_base_model_max_image_bytes() -> int:
    """Return limit in bytes (YAML stores MB). OpenAI caps combined payload at 50 MB."""
    return int(_get(load_baseline_config(), "base_model", "max_image_bytes_mb", default=45)) * 1024 * 1024


# ─── Framework-wide retrieval knobs ────────────────────────────────────


def resolve_retrieval_top_k() -> int:
    """Default top_k for every adapter.retrieve() call across the pipeline."""
    return int(_get(load_baseline_config(), "retrieval", "top_k", default=10))


def resolve_retrieval_multimodal_topk() -> int:
    return int(_get(load_baseline_config(), "retrieval", "multimodal_topk", default=10))


def resolve_retrieval_word_truncation() -> int:
    return int(_get(load_baseline_config(), "retrieval", "word_truncation_words", default=50000))


def resolve_retrieval_answer_model_ctx() -> int:
    return int(_get(load_baseline_config(), "retrieval", "answer_model_ctx", default=128000))


def resolve_retrieval_answer_model_buffer() -> int:
    return int(_get(load_baseline_config(), "retrieval", "answer_model_buffer", default=8000))


def resolve_retrieval_truncation_tokenizer() -> str:
    return str(_get(load_baseline_config(), "retrieval", "truncation_tokenizer", default="gpt2"))


def resolve_judge_max_workers() -> int:
    return int(_get(load_baseline_config(), "judge", "max_workers", default=5))


# ─── Per-baseline method params ────────────────────────────────────────


def resolve_baseline_param(baseline_name: str, key: str, default: Any = None) -> Any:
    """Look up `baselines.<name>.<key>` in config.yaml."""
    return _get(load_baseline_config(), "baselines", baseline_name, key, default=default)


@dataclass
class EvalConfig:
    """Resolved paths and flags for one eval invocation."""

    dataset_path: Path
    output_dir: Path
    baseline: str
    smoke: bool = False
    dry_run: bool = False
    max_sessions: int | None = None
    memory_accuracy_itemwise: bool = False
    answer_evidence_mode: str = "memory"
    sample_index: int | None = None

    def to_display_dict(self) -> dict[str, Any]:
        """JSON-friendly snapshot for dry-run and logging."""
        return {
            "dataset_path": str(self.dataset_path),
            "output_dir": str(self.output_dir),
            "baseline": self.baseline,
            "smoke": self.smoke,
            "dry_run": self.dry_run,
            "max_sessions": self.max_sessions,
            "memory_accuracy_itemwise": self.memory_accuracy_itemwise,
            "answer_evidence_mode": self.answer_evidence_mode,
            "sample_index": self.sample_index,
            "dataset_profile": "domain_a_v2_academic",
            "judge": "llm (OpenAI API)",
        }
