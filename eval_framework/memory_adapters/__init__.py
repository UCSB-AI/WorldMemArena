"""Memory system adapters for the eval framework."""

from eval_framework.memory_adapters.base import MemoryAdapter
from eval_framework.memory_adapters.memgallery_native import (
    MemGalleryNativeAdapter,
    instantiate_memgallery_memory,
)
from eval_framework.memory_adapters.registry import (
    EXTERNAL_ADAPTER_KEYS,
    EXTERNAL_ADAPTER_REGISTRY,
    MEMGALLERY_NATIVE_BASELINES,
    MEMGALLERY_NATIVE_REGISTRY,
    create_external_adapter,
    create_memgallery_adapter,
)

__all__ = [
    "EXTERNAL_ADAPTER_KEYS",
    "EXTERNAL_ADAPTER_REGISTRY",
    "MEMGALLERY_NATIVE_BASELINES",
    "MEMGALLERY_NATIVE_REGISTRY",
    "MemoryAdapter",
    "MemGalleryNativeAdapter",
    "create_external_adapter",
    "create_memgallery_adapter",
    "instantiate_memgallery_memory",
]
