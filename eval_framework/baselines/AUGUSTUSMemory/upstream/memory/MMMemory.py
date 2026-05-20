"""MMMemory: Multimodal Memory — separate text and caption storage with dual-encoder retrieval."""

from memengine.memory.BaseMemory import ExplicitMemory
from memengine.utils.Storage import LinearStorage
from memengine.operation.Recall import LTMemoryRecall
from memengine.operation.Store import LTMemoryStore
from memengine.utils.Display import *


class MMMemory(ExplicitMemory):
    """Multimodal Memory: stores text and multimodal captions in separate
    linear storages, with embedding-based retrieval that searches both.

    Unlike FUMemory which simply concatenates everything, MMMemory
    maintains separate text and caption indices and merges retrieval
    results from both at recall time.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        # Primary text storage (dialogue content)
        self.storage = LinearStorage(self.config.args.storage)
        # Separate storage for multimodal captions
        self.caption_storage = LinearStorage(self.config.args.storage)

        self.recall_op = LTMemoryRecall(self.config.args.recall, storage=self.storage)
        self.store_op = LTMemoryStore(
            self.config.args.store,
            storage=self.storage,
            text_retrieval=self.recall_op.text_retrieval,
        )

        # Caption retrieval uses same encoder but separate index
        self.caption_recall_op = LTMemoryRecall(
            self.config.args.recall, storage=self.caption_storage
        )
        self.caption_store_op = LTMemoryStore(
            self.config.args.store,
            storage=self.caption_storage,
            text_retrieval=self.caption_recall_op.text_retrieval,
        )

        self.auto_display = eval(self.config.args.display.method)(
            self.config.args.display,
            register_dict={'Memory Storage': self.storage},
        )

    def reset(self) -> None:
        self.__reset_objects__([
            self.storage, self.store_op, self.recall_op,
            self.caption_storage, self.caption_store_op, self.caption_recall_op,
        ])

    def store(self, observation) -> None:
        if isinstance(observation, str):
            observation = {'text': observation}
        text = observation.get('text', '')

        # Split: lines starting with [ are treated as captions
        text_parts = []
        caption_parts = []
        for line in text.split('\n'):
            stripped = line.strip()
            if stripped.startswith('[') and ']' in stripped:
                caption_parts.append(stripped)
            else:
                text_parts.append(line)

        text_content = '\n'.join(text_parts).strip()
        caption_content = '\n'.join(caption_parts).strip()

        if text_content:
            self.store_op({'text': text_content})
        if caption_content:
            self.caption_store_op({'text': caption_content})
        if not text_content and not caption_content:
            self.store_op(observation)

    def recall(self, query) -> object:
        # Retrieve from both storages and merge
        text_result = self.recall_op(query) if not self.storage.is_empty() else ''
        caption_result = (
            self.caption_recall_op(query)
            if not self.caption_storage.is_empty()
            else ''
        )

        parts = []
        if text_result:
            parts.append(str(text_result))
        if caption_result:
            parts.append("[Multimodal Context]\n" + str(caption_result))
        return '\n'.join(parts)

    def display(self) -> None:
        self.auto_display(self.storage.counter)

    def manage(self, operation, **kwargs) -> None:
        pass

    def optimize(self, **kwargs) -> None:
        pass
