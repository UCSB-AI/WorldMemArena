from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Type, TypeVar, Any, get_type_hints

T = TypeVar("T")
TIME_FMT = r"%Y-%m-%d %H:%M"


def _from_dict(cls: Type[T], data: dict[str, Any]) -> T:
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, val in data.items():
        if key not in known:
            continue
        hint = hints[key]
        if isinstance(val, dict) and hasattr(hint, "__dataclass_fields__"):
            val = _from_dict(hint, val)
        kwargs[key] = val
    return cls(**kwargs)


def _override_from_env(obj: T, prefix: str) -> T:
    hints = get_type_hints(type(obj))
    updates: dict[str, Any] = {}
    for f in fields(obj):
        raw = os.environ.get(f"{prefix}_{f.name.upper()}")
        if raw is None:
            continue
        hint = hints[f.name]
        if hasattr(hint, "__dataclass_fields__"):
            continue
        try:
            if hint is bool:
                updates[f.name] = raw.lower() in ("1", "true", "yes")
            elif hint is int:
                updates[f.name] = int(raw)
            elif hint is float:
                updates[f.name] = float(raw)
            else:
                updates[f.name] = raw
        except (ValueError, TypeError):
            pass
    if not updates:
        return obj
    return _from_dict(type(obj), {**asdict(obj), **updates})


@dataclass
class LLMConfig:
    model: str = "gpt-4o-mini"
    api_key: str = "EMPTY"
    base_url: str = "https://api.openai.com/v1"
    temperature: float = 0
    max_tokens: int = 1200
    timeout: int = 20


@dataclass
class TextEmbeddingConfig:
    api_key: str = "EMPTY"
    base_url: str = "http://localhost:8100/v1"
    model: str = "text-embedding-3-small"


@dataclass
class MultimodalEmbeddingConfig:
    api_key: str = "EMPTY"
    base_url: str = "http://localhost:8050/v1"
    model: str = "siglip2-base-patch16-384"


@dataclass
class MemoryManagerConfig:
    context_window: int = 5
    max_iteration: int = 5


@dataclass
class MemoryConfig:
    raw_db_path: str = "raw_messages.db"
    semantic_db_path: str = "semantic_store.db"
    reuse_db: bool = False
    max_raw_messages_return: int = 20


@dataclass
class ChatAgentConfig:
    max_chat_context: int = 8000
    max_query_iteration: int = 5
    max_update_iteration: int = 5


@dataclass
class M2AConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    text_embedding: TextEmbeddingConfig = field(default_factory=TextEmbeddingConfig)
    multimodal_embedding: MultimodalEmbeddingConfig = field(default_factory=MultimodalEmbeddingConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    chat_agent: ChatAgentConfig = field(default_factory=ChatAgentConfig)
    memory_manager: MemoryManagerConfig = field(default_factory=MemoryManagerConfig)

    def _apply_env_overrides(self) -> None:
        hints = get_type_hints(type(self))
        for f in fields(self):
            hint = hints[f.name]
            if hasattr(hint, "__dataclass_fields__"):
                prefix = f"M2A_{f.name.upper()}"
                setattr(self, f.name, _override_from_env(getattr(self, f.name), prefix))

    @classmethod
    def from_env(cls) -> M2AConfig:
        cfg = cls()
        cfg._apply_env_overrides()
        return cfg

    @classmethod
    def from_file(cls, config_path: str) -> M2AConfig:
        import tomllib
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return cls._from_dict_nested(data)

    @classmethod
    def from_file_and_env(cls, config_path: str) -> M2AConfig:
        cfg = cls.from_file(config_path)
        cfg._apply_env_overrides()
        return cfg

    @classmethod
    def _from_dict_nested(cls, data: dict) -> M2AConfig:
        cfg = cls()
        hints = get_type_hints(cls)
        for f in fields(cls):
            hint = hints[f.name]
            if hasattr(hint, "__dataclass_fields__") and f.name in data:
                current = asdict(getattr(cfg, f.name))
                current.update(data[f.name])
                setattr(cfg, f.name, _from_dict(hint, current))
        return cfg

    def save_to_file(self, config_path: str) -> None:
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    def to_dict(self) -> dict:
        return asdict(self)


def get_default_config() -> M2AConfig:
    return M2AConfig()


def get_config_from_env() -> M2AConfig:
    return M2AConfig.from_env()