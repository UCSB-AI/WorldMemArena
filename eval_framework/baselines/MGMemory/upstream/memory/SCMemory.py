from memengine.memory.BaseMemory import ExplicitMemory
from memengine.utils.Storage import LinearStorage
from memengine.operation.Recall import SCMemoryRecall
from memengine.operation.Store import SCMemoryStore
from memengine.utils.Display import *

class SCMemory(ExplicitMemory):
    """
    SCMemory (SCM [1]): A self-controlled memory model that can recall minimum but necessary information for inference.
    
    [1] Wang, Bing, et al. "Enhancing large language model with self-controlled memory framework." arXiv preprint arXiv:2304.13343 (2023).
    """
    def __init__(self, config):
        super().__init__(config)

        self.storage = LinearStorage(self.config.args.storage)
        self.recall_op = SCMemoryRecall(
            self.config.args.recall,
            storage = self.storage
        )
        self.store_op = SCMemoryStore(
            self.config.args.store,
            storage = self.storage,
            summarizer = self.recall_op.summarizer,
            text_retrieval = self.recall_op.text_retrieval,
            time_retrieval = self.recall_op.time_retrieval
        )

        self.auto_display = eval(self.config.args.display.method)(self.config.args.display, register_dict = {
            'Memory Storage': self.storage
        })

    def reset(self) -> None:
        self.__reset_objects__([self.storage, self.store_op, self.recall_op])

    def store(self, observation) -> None:
        self.store_op(observation)

    def recall(self, observation) -> object:
        return self.recall_op(observation)
    
    def display(self) -> None:
        self.auto_display(self.storage.counter)
            
    def manage(self, operation, **kwargs) -> None:
        pass
    
    def optimize(self, **kwargs) -> None:
        pass
