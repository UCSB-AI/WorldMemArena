"""MMFUMemory: Multimodal Full Memory — full-context with separate caption tracking."""

from memengine.memory.BaseMemory import ExplicitMemory
from memengine.utils.Storage import LinearStorage
from memengine.operation.Recall import FUMemoryRecall
from memengine.operation.Store import FUMemoryStore
from memengine.utils.Display import *


class MMFUMemory(ExplicitMemory):
    """Multimodal Full Memory: full-context memory that preserves the structure
    of multimodal observations by separating text and captions.

    Stores all content in a single linear storage like FUMemory, but also
    maintains a separate caption index. At recall time, returns full context
    with captions explicitly marked and grouped.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self.storage = LinearStorage(self.config.args.storage)
        self.caption_storage = LinearStorage(self.config.args.storage)
        self.store_op = FUMemoryStore(self.config.args.store, storage=self.storage)
        self.recall_op = FUMemoryRecall(self.config.args.recall, storage=self.storage)
        self.auto_display = eval(self.config.args.display.method)(
            self.config.args.display,
            register_dict={'Memory Storage': self.storage},
        )

    def reset(self) -> None:
        self.__reset_objects__([
            self.storage, self.store_op, self.recall_op,
            self.caption_storage,
        ])

    def store(self, observation) -> None:
        if isinstance(observation, str):
            observation = {'text': observation}
        text = observation.get('text', '')

        # Store full observation in primary
        self.store_op(observation)

        # Also index captions separately for structured recall
        caption_parts = []
        for line in text.split('\n'):
            stripped = line.strip()
            if stripped.startswith('[') and ']' in stripped:
                caption_parts.append(stripped)
        if caption_parts:
            self.caption_storage.add({'text': '\n'.join(caption_parts)})

    def recall(self, query) -> object:
        # Full context recall
        text_result = self.recall_op(query)
        if not text_result:
            return ''

        # Append caption summary if available
        if not self.caption_storage.is_empty():
            all_captions = self.caption_storage.get_all_memory_in_order()
            caption_texts = [c.get('text', '') for c in all_captions if c.get('text')]
            if caption_texts:
                return (
                    str(text_result)
                    + "\n\n[Multimodal Summary]\n"
                    + '\n'.join(caption_texts[-5:])
                )

        return text_result

    def display(self) -> None:
        self.auto_display(self.storage.counter)

    def manage(self, operation, **kwargs) -> None:
        pass

    def optimize(self, **kwargs) -> None:
        pass
