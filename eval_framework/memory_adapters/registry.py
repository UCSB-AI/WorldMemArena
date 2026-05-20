"""Registry and factory for native Mem-Gallery and external placeholder adapters."""

from __future__ import annotations

import os
import sys
import types
from contextlib import nullcontext
from functools import partial
import importlib
from pathlib import Path
from typing import Any, Callable

from eval_framework.memory_adapters.base import MemoryAdapter
from eval_framework.memory_adapters.dummy import DummyAdapter
from eval_framework.memory_adapters.memgallery_native import MemGalleryNativeAdapter
from eval_framework.config import (
    is_base_model_baseline,
    is_harness_baseline,
    resolve_base_model_baseline_names,
    resolve_harness_baseline_names,
)

MEMGALLERY_NATIVE_BASELINES: frozenset[str] = frozenset(
    {
        "FUMemory",
        "STMemory",
        "LTMemory",
        "GAMemory",
        "MGMemory",
        "RFMemory",
        "MMMemory",
        "MMFUMemory",
        "MMFU_Single",
        "NGMemory",
        "AUGUSTUSMemory",
        "UniversalRAGMemory",
    }
)


def _word_mode_truncation(number: int | None = None) -> dict[str, Any]:
    from eval_framework.config import resolve_retrieval_word_truncation
    if number is None:
        number = resolve_retrieval_word_truncation()
    return {
        "method": "LMTruncation",
        "mode": "word",
        "number": number,
        "path": "",
    }


def _token_mode_truncation(
    number: int | None = None,
    path: str | None = None,
    method: str = "LMTruncation",
) -> dict[str, Any]:
    """Token-budget truncation. Reads ``retrieval.*`` from config.yaml."""
    from eval_framework.config import (
        resolve_retrieval_answer_model_buffer,
        resolve_retrieval_answer_model_ctx,
        resolve_retrieval_truncation_tokenizer,
    )
    if number is None:
        ctx_window = resolve_retrieval_answer_model_ctx()
        buffer_len = resolve_retrieval_answer_model_buffer()
        number = max(4096, ctx_window - buffer_len)
    if path is None:
        path = resolve_retrieval_truncation_tokenizer()
    return {
        "method": method,
        "mode": "token",
        "number": int(number),
        "path": path,
    }


def _text_encoder_override() -> dict[str, Any]:
    """Use OpenAI's ``text-embedding-3-small`` (pinned to 384 dims to match
    the historical ``all-MiniLM-L6-v2`` collections).
    """
    from eval_framework.config import (
        resolve_embedding_base_url,
        resolve_embedding_model,
    )
    return {
        "method": "OpenAIEncoder",
        "path": resolve_embedding_model(),
        "dimensions": 384,
        "base_url": resolve_embedding_base_url(),
    }


def _gme_encoder_override() -> dict[str, Any]:
    """GME-Qwen2-VL-2B-Instruct multimodal encoder served via local vLLM
    ``/v1/embeddings``. ``path`` carries the served-model name, ``base_url``
    points at the vLLM endpoint. Falls back to the Qwen3-VL embedding server
    config if a dedicated GME endpoint isn't configured."""
    base_url = (
        os.getenv("GME_BASE_URL")
        or os.getenv("QWEN_VL_EMBED_BASE_URL")
        or "http://127.0.0.1:8014/v1"
    )
    api_key = (
        os.getenv("GME_API_KEY")
        or os.getenv("QWEN_VL_EMBED_API_KEY")
        or "EMPTY"
    )
    model_name = os.getenv("GME_MODEL", "gme-Qwen2-VL-2B-Instruct")
    return {
        "method": "GMEEncoder",
        "name": "gme-qwen2-vl-2b",
        "dimension": 1536,
        "path": model_name,
        "base_url": base_url,
        "api_key": api_key,
    }


def _multimodal_retrieval_override() -> dict[str, Any]:
    from eval_framework.config import resolve_retrieval_multimodal_topk
    return {
        "method": "MultiModalRetrieval",
        "encoder": _gme_encoder_override(),
        "mode": "cosine",
        "topk": resolve_retrieval_multimodal_topk(),
    }


def _openai_llm_override() -> dict[str, Any]:
    from eval_framework.config import (
        resolve_openai_base_url,
        resolve_openai_model,
        resolve_openai_temperature,
    )
    return {
        "method": "APILLM",
        "name": resolve_openai_model(),
        "api_key": os.getenv("OPENAI_API_KEY") or "",
        "base_url": resolve_openai_base_url(),
        "temperature": resolve_openai_temperature(),
    }


def _default_memgallery_runtime_overrides(baseline_name: str) -> dict[str, Any]:
    overrides: dict[str, Any] = {}

    # --- text-only baselines ---
    if baseline_name in {"FUMemory", "STMemory", "LTMemory", "RFMemory"}:
        overrides["recall"] = {"truncation": _word_mode_truncation()}
    if baseline_name == "LTMemory":
        overrides.setdefault("recall", {})
        overrides["recall"]["text_retrieval"] = {"encoder": _text_encoder_override()}
    if baseline_name == "GAMemory":
        overrides = {
            "recall": {
                "truncation": _word_mode_truncation(),
                "text_retrieval": {"encoder": _text_encoder_override()},
                "importance_judge": {"LLM_config": _openai_llm_override()},
            },
            "reflect": {
                "reflector": {"LLM_config": _openai_llm_override()},
            },
        }
    if baseline_name == "MGMemory":
        overrides = {
            "recall": {
                "truncation": _word_mode_truncation(),
                "recall_retrieval": {"encoder": _text_encoder_override()},
                "archival_retrieval": {"encoder": _text_encoder_override()},
                "trigger": {"LLM_config": _openai_llm_override()},
            },
            "store": {
                "flush_checker": _word_mode_truncation(),
                "summarizer": {"LLM_config": _openai_llm_override()},
            },
        }
    if baseline_name == "RFMemory":
        overrides.setdefault("optimize", {})
        overrides["optimize"] = {
            "reflector": {"LLM_config": _openai_llm_override()},
        }

    # --- multimodal baselines ---
    if baseline_name == "MMMemory":
        overrides = {
            "recall": {
                "truncation": _word_mode_truncation(),
                "text_retrieval": {"encoder": _text_encoder_override()},
            },
        }
    if baseline_name == "MMFUMemory":
        overrides = {
            "recall": {
                "truncation": _word_mode_truncation(),
            },
        }
    if baseline_name == "MMFU_Single":
        # AMA-Bench-aligned long-context baseline. The backend truncation
        # is set to a near-infinite budget (10M tokens) so MMFUMemory
        # returns the *full* buffer unchanged. All real budget accounting
        # happens in the adapter's ``_longcontext_retrieve`` path, where
        # we match ``AMA-Bench/src/method/longcontext.py``:
        #
        #   budget = max_model_len − max_response_tokens − safety_buffer
        #            − question_overhead
        #
        # and apply a **70% head + 30% tail** truncation so both the
        # kickoff context (project goals, stakeholders) and the recent
        # state (current decisions, blockers) survive when the
        # conversation overflows the window.
        overrides = {
            "recall": {
                "truncation": _token_mode_truncation(number=10_000_000),
            },
        }
    if baseline_name == "NGMemory":
        # Upstream-aligned: NGMemory uses MultiModalRetrieval with the GME
        # multimodal encoder for both store + recall paths.
        overrides = {
            "recall": {
                "truncation": _word_mode_truncation(),
                "multimodal_retrieval": _multimodal_retrieval_override(),
            },
            "multimodal_retrieval": _multimodal_retrieval_override(),
            "entity_extractor": {
                "LLM_config": _openai_llm_override(),
            },
        }
    if baseline_name == "AUGUSTUSMemory":
        # Upstream-aligned: AUGUSTUSMemory uses MultiModalRetrieval +
        # ConceptBasedRetrieval. Both share the same GME encoder.
        overrides = {
            "recall": {
                "truncation": _word_mode_truncation(),
                "multimodal_retrieval": _multimodal_retrieval_override(),
            },
            "multimodal_retrieval": _multimodal_retrieval_override(),
            "concept_extractor": {
                "llm": _openai_llm_override(),
            },
        }
    if baseline_name == "UniversalRAGMemory":
        # Upstream-aligned: UniversalRAGRetrieval embeds modality-specific
        # storages (text / image / table) all through the GME encoder.
        overrides = {
            "recall": {
                "truncation": _word_mode_truncation(),
            },
            "storage": {
                "encoder": _gme_encoder_override(),
            },
            "routing": {"llm": _openai_llm_override()},
        }
    return overrides


def _resolve_baselines_root() -> Path:
    """Return the ``baselines/`` directory (sibling of eval_framework/).

    Layout::

        nips26/
        ├── eval_framework/
        └── baselines/
            ├── memengine/
            └── default_config/
    """
    # registry.py -> memory_adapters/ -> eval_framework/ -> nips26/
    return Path(__file__).resolve().parents[1] / "baselines"


def _ensure_memgallery_benchmark_on_path() -> Path:
    """Add ``baselines/`` to sys.path so that ``memengine`` and
    ``default_config`` packages are importable."""
    baselines_root = _resolve_baselines_root()
    if not (baselines_root / "memengine").is_dir():
        raise FileNotFoundError(
            f"memengine/ not found under {baselines_root}. "
            f"Clone MemEngine into baselines/memengine."
        )
    s = str(baselines_root)
    if s not in sys.path:
        sys.path.insert(0, s)
    _bootstrap_memengine_namespace(baselines_root)
    return baselines_root


def _bootstrap_memengine_namespace(root: Path) -> None:
    """
    Pre-seed lightweight namespace packages for the co-located memengine package.

    memengine's package-level ``__init__.py`` eagerly imports all memories and function
    modules, which pulls in heavyweight optional dependencies like ``torch`` even for
    simple baselines such as ``FUMemory``. By registering package shells in ``sys.modules``
    first, we can import only the specific submodules we need.

    *root* is the ``our/`` directory that contains ``memengine/``.
    """
    package_paths = {
        "memengine": root / "memengine",
        "memengine.config": root / "memengine" / "config",
        "memengine.memory": root / "memengine" / "memory",
        "memengine.function": root / "memengine" / "function",
        "memengine.operation": root / "memengine" / "operation",
        "memengine.utils": root / "memengine" / "utils",
    }
    for package_name, package_path in package_paths.items():
        existing = sys.modules.get(package_name)
        if existing is not None:
            continue
        module = types.ModuleType(package_name)
        module.__path__ = [str(package_path)]  # type: ignore[attr-defined]
        module.__package__ = package_name
        sys.modules[package_name] = module

    for package_name in package_paths:
        if "." not in package_name:
            continue
        parent_name, child_name = package_name.rsplit(".", 1)
        parent = sys.modules.get(parent_name)
        child = sys.modules.get(package_name)
        if parent is not None and child is not None and not hasattr(parent, child_name):
            setattr(parent, child_name, child)

    _bootstrap_optional_dependency_stubs()
    _populate_safe_memengine_function_exports()


def _bootstrap_optional_dependency_stubs() -> None:
    """Provide narrow stubs for optional imports needed only on unused code paths."""
    if "torch" not in sys.modules:
        try:
            sys.modules["torch"] = importlib.import_module("torch")
        except Exception:
            pass
    if "torch" not in sys.modules:
        torch_module = types.ModuleType("torch")

        def _torch_unavailable(*args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            raise RuntimeError(
                "PyTorch is required for encoder-backed or tensor-based Mem-Gallery "
                "baselines, but `torch` is not installed in this environment."
            )

        torch_module.cuda = types.SimpleNamespace(is_available=lambda: False)  # type: ignore[attr-defined]
        torch_module.device = lambda spec: spec  # type: ignore[attr-defined]
        torch_module.no_grad = lambda: nullcontext()  # type: ignore[attr-defined]
        torch_module.from_numpy = _torch_unavailable  # type: ignore[attr-defined]
        torch_module.stack = _torch_unavailable  # type: ignore[attr-defined]
        torch_module.sort = _torch_unavailable  # type: ignore[attr-defined]
        torch_module.matmul = _torch_unavailable  # type: ignore[attr-defined]
        torch_module.ones = _torch_unavailable  # type: ignore[attr-defined]
        torch_module.nn = types.SimpleNamespace(  # type: ignore[attr-defined]
            functional=types.SimpleNamespace(normalize=_torch_unavailable)
        )
        sys.modules["torch"] = torch_module

    if "transformers" not in sys.modules:
        try:
            sys.modules["transformers"] = importlib.import_module("transformers")
        except Exception:
            pass
    if "transformers" not in sys.modules:
        transformers_module = types.ModuleType("transformers")

        class _UnavailableAutoTokenizer:
            @classmethod
            def from_pretrained(cls, *args: Any, **kwargs: Any) -> Any:
                del args, kwargs
                raise RuntimeError(
                    "transformers.AutoTokenizer is required for token-mode truncation "
                    "or encoder-backed baselines, but `transformers` is not installed."
                )

        transformers_module.AutoTokenizer = _UnavailableAutoTokenizer  # type: ignore[attr-defined]
        sys.modules["transformers"] = transformers_module


def _populate_safe_memengine_function_exports() -> None:
    """Expose all function symbols for complete baseline deployment without running package __init__."""
    function_pkg = sys.modules.get("memengine.function")
    if function_pkg is None:
        return

    # Complete list — covers every module referenced by any of the 11 baselines:
    #   FU/ST/LT/GA/MG/RF (text-only) + MM/MMFU/NG/AUGUSTUS/UniversalRAG (multimodal)
    for module_name in (
        # --- text-only baselines ---
        "memengine.function.Encoder",
        "memengine.function.Retrieval",
        "memengine.function.LLM",
        "memengine.function.Judge",
        "memengine.function.Reflector",
        "memengine.function.Summarizer",
        "memengine.function.Truncation",
        "memengine.function.Trigger",
        "memengine.function.Utilization",
        "memengine.function.Forget",
        # --- multimodal / graph / concept baselines ---
        "memengine.function.MultiModalEncoder",
        "memengine.function.MultiModalRetrieval",
        "memengine.function.ConceptExtractor",
        "memengine.function.ConceptBasedRetrieval",
        "memengine.function.EntityExtractor",
        "memengine.function.FactExtractor",
        "memengine.function.UniversalRAGRouting",
        "memengine.function.UniversalRAGRetrieval",
    ):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            # Some modules may depend on optional heavy deps (torch, transformers).
            # Skip gracefully — they will fail loudly if the baseline actually needs them.
            continue
        for attr_name, value in vars(module).items():
            if attr_name.startswith("_"):
                continue
            if not hasattr(function_pkg, attr_name):
                setattr(function_pkg, attr_name, value)


def create_memgallery_adapter(
    baseline_name: str,
    *,
    config_overrides: dict[str, Any] | None = None,
) -> MemGalleryNativeAdapter:
    """
    Instantiate a native Mem-Gallery adapter for a known baseline name.

    Loads default_config + memengine from the Mem-Gallery benchmark tree.
    """
    if baseline_name not in MEMGALLERY_NATIVE_BASELINES:
        raise KeyError(f"unknown Mem-Gallery baseline: {baseline_name!r}")
    _ensure_memgallery_benchmark_on_path()
    runtime_overrides = _default_memgallery_runtime_overrides(baseline_name)
    if config_overrides:
        runtime_overrides = {
            **runtime_overrides,
            **config_overrides,
        }
    return MemGalleryNativeAdapter.from_baseline(
        baseline_name, config=runtime_overrides or None
    )


MEMGALLERY_NATIVE_REGISTRY: dict[str, Callable[..., MemGalleryNativeAdapter]] = {
    name: partial(create_memgallery_adapter, name) for name in MEMGALLERY_NATIVE_BASELINES
}

EXTERNAL_ADAPTER_KEYS: frozenset[str] = frozenset({
    "A-Mem", "Dummy",
    "SimpleMem", "Omni-SimpleMem",
    "ViLoMem",
    "M2A", "MIRIX",
    "Qwen3-VL-Embedding-8B",
    "BaseModel",
}) | frozenset(resolve_base_model_baseline_names()) | frozenset(resolve_harness_baseline_names())


def create_amem_adapter(**kwargs: Any) -> MemoryAdapter:
    from eval_framework.memory_adapters.amem_v2 import AMemV2Adapter
    return AMemV2Adapter(**kwargs)


def create_dummy_adapter(**kwargs: Any) -> DummyAdapter:
    return DummyAdapter()


def create_simplemem_adapter(**kwargs: Any) -> MemoryAdapter:
    from eval_framework.memory_adapters.simplemem_adapter import SimpleMemAdapter
    return SimpleMemAdapter(mode="text", **kwargs)


def create_omni_simplemem_adapter(**kwargs: Any) -> MemoryAdapter:
    from eval_framework.memory_adapters.simplemem_adapter import SimpleMemAdapter
    return SimpleMemAdapter(mode="omni", **kwargs)


def create_vilomem_adapter(**kwargs: Any) -> MemoryAdapter:
    from eval_framework.memory_adapters.vilomem_adapter import ViLoMemAdapter
    return ViLoMemAdapter(**kwargs)


def create_m2a_adapter(**kwargs: Any) -> MemoryAdapter:
    from eval_framework.memory_adapters.m2a_adapter import M2AAdapter
    return M2AAdapter(**kwargs)


def create_mirix_adapter(**kwargs: Any) -> MemoryAdapter:
    from eval_framework.memory_adapters.mirix_adapter import MIRIXAdapter
    return MIRIXAdapter(**kwargs)


def create_qwen_vl_embed_adapter(**kwargs: Any) -> MemoryAdapter:
    from eval_framework.memory_adapters.qwen_embed_adapter import QwenVLEmbedAdapter
    return QwenVLEmbedAdapter(**kwargs)


def create_base_model_adapter(**kwargs: Any) -> MemoryAdapter:
    from eval_framework.memory_adapters.memgallery_native import (
        MemGalleryNativeAdapter,
        instantiate_memgallery_memory,
    )

    baseline_name = str(kwargs.pop("baseline_name", "BaseModel"))
    _ensure_memgallery_benchmark_on_path()
    memory = instantiate_memgallery_memory(
        "MMFU_Single",
        _default_memgallery_runtime_overrides("MMFU_Single"),
    )
    return MemGalleryNativeAdapter(memory, baseline_name=baseline_name)


def create_harness_adapter(**kwargs: Any) -> MemoryAdapter:
    from eval_framework.memory_adapters.harness_native import HarnessMemoryAdapter

    baseline_name = str(kwargs.pop("baseline_name", "Harness"))
    return HarnessMemoryAdapter(baseline_name=baseline_name)


EXTERNAL_ADAPTER_REGISTRY: dict[str, Callable[..., MemoryAdapter]] = {
    "A-Mem": create_amem_adapter,
    "Dummy": create_dummy_adapter,
    "SimpleMem": create_simplemem_adapter,
    "Omni-SimpleMem": create_omni_simplemem_adapter,
    "ViLoMem": create_vilomem_adapter,
    "M2A": create_m2a_adapter,
    "MIRIX": create_mirix_adapter,
    "Qwen3-VL-Embedding-8B": create_qwen_vl_embed_adapter,
    "BaseModel": create_base_model_adapter,
}


def create_external_adapter(
    name: str,
    *,
    config_overrides: dict[str, Any] | None = None,
) -> MemoryAdapter:
    """Instantiate an external adapter for a known baseline name."""
    if is_base_model_baseline(name):
        return create_base_model_adapter(baseline_name=name)
    if is_harness_baseline(name):
        return create_harness_adapter(baseline_name=name)
    if name not in EXTERNAL_ADAPTER_KEYS:
        raise KeyError(f"unknown external adapter: {name!r}")
    return EXTERNAL_ADAPTER_REGISTRY[name](**(config_overrides or {}))
