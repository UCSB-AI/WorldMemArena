"""Abstract memory adapter API for eval baselines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from eval_framework.datasets.schemas import (
    MemoryDeltaRecord,
    MemorySnapshotRecord,
    NormalizedTurn,
    RetrievalRecord,
)


class MemoryAdapter(ABC):
    """Baseline-agnostic adapter surface used by the eval pipeline."""

    @abstractmethod
    def reset(self) -> None:
        """Clear backend state and any adapter-side bookkeeping."""

    @abstractmethod
    def ingest_turn(self, turn: NormalizedTurn) -> None:
        """Feed one conversation turn into the memory system."""

    @abstractmethod
    def end_session(self, session_id: str) -> None:
        """Notify the adapter that a session boundary was reached (optional for many backends)."""

    @abstractmethod
    def snapshot_memories(self) -> list[MemorySnapshotRecord]:
        """Return a normalized view of memories observable in the backend."""

    @abstractmethod
    def export_memory_delta(self, session_id: str) -> list[MemoryDeltaRecord]:
        """Export memory changes for the given session since the last call."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int, **kwargs: Any) -> RetrievalRecord:
        """Run retrieval and normalize results.

        Extra kwargs are adapter-specific hints. The pipeline currently forwards
        ``category`` (the question_type_abbrev, e.g. ``"VS"``, ``"FR"``) to
        support baselines whose official retrieval is category-aware
        (Omni-SimpleMem). Adapters that do not need them may ignore kwargs.
        """

    def answer_batch(self, questions: list[str]) -> list[str]:
        """Answer multiple questions in a single call (for harness baselines).

        Default implementation raises NotImplementedError; only harness adapters
        that own their own memory/retrieval need to implement this.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support answer_batch"
        )

    @abstractmethod
    def get_capabilities(self) -> dict[str, Any]:
        """Describe adapter behavior limits (deltas, snapshots, backend id)."""
