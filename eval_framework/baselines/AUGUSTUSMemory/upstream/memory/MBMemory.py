from memengine.memory.BaseMemory import ExplicitMemory
from memengine.utils.Storage import LinearStorage
from memengine.operation.Recall import MBMemoryRecall
from memengine.operation.Store import MBMemoryStore
from memengine.utils.Display import *

class MBMemory(ExplicitMemory):
    """
    MBMemory (MemoryBank [1]): A multi-layered memory model with dynamic summarization and forgetting mechanism.
    We consider one step as one day in the original implementation.

    [1] Zhong, Wanjun, et al. "Memorybank: Enhancing large language models with long-term memory." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 17. 2024.
    """
    def __init__(self, config) -> None:
        super().__init__(config)

        self.storage = LinearStorage(self.config.args.storage)
        self.recall_op = MBMemoryRecall(self.config.args.recall, storage = self.storage)
        self.store_op = MBMemoryStore(
            self.config.args.store,
            storage = self.storage,
            text_retrieval = self.recall_op.text_retrieval,
            summary = self.recall_op.summary
        )

        self.auto_display = eval(self.config.args.display.method)(self.config.args.display, register_dict = {
            'Memory Storage': self.storage,
            'Memory Summary': self.recall_op.summary
        })

    def reset(self) -> None:
        self.__reset_objects__([self.storage, self.store_op, self.recall_op])

    def store(self, observation) -> None:
        self.store_op(observation)

    def recall(self, query) -> str:
        return self.recall_op(query)
    
    def display(self) -> None:
        self.auto_display(self.storage.counter)

    def manage(self, operation, **kwargs) -> None:
        pass
    
    def optimize(self, **kwargs) -> None:
        pass
