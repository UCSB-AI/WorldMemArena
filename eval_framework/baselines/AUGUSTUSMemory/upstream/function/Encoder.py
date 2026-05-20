import os
from abc import ABC, abstractmethod

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

class BaseEncoder(ABC):
    """
    Transfer textual messages into embeddings to represent in latent space by pre-trained models.
    """
    def __init__(self, config) -> None:
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def reset(self):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

class LMEncoder(BaseEncoder):
    """
    Embedding vias LM transformers.
    """
    def __init__(self, config):
        super().__init__(config)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.path)
        self.model = AutoModel.from_pretrained(self.config.path).to(self.device)
    
    def __call__(self, text, return_type = 'numpy'):
        res = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            embeddings = self.model(**res).last_hidden_state[:, -1, :]
        if return_type == 'numpy':
            return embeddings.numpy()
        elif return_type == 'tensor':
            return embeddings.to(self.device)
        else:
            return 'Unrecognized Return Type.'

class STEncoder(BaseEncoder):
    """
    Embedding via Sentence Transformer (lazy import: small embeddings normally
    flow through the OpenAI-compatible API, so ``sentence_transformers`` is
    only required if a config explicitly selects this class).
    """
    def __init__(self, config):
        super().__init__(config)

        from sentence_transformers import SentenceTransformer  # local import
        self.model = SentenceTransformer(self.config.path).to(self.device)

    def __call__(self, text, return_type = 'numpy'):
        embeddings = self.model.encode([text], normalize_embeddings=True)
        if return_type == 'numpy':
            return embeddings.cpu().numpy()
        elif return_type == 'tensor':
            return torch.from_numpy(embeddings).to(self.device)
        else:
            return 'Unrecognized Return Type.'


class OpenAIEncoder(BaseEncoder):
    """OpenAI-compatible embedding encoder.

    ``config.path`` is reused to carry the model name (e.g.
    ``text-embedding-3-small``).  ``config.dimensions`` (optional) pins the
    output width via the OpenAI ``dimensions`` parameter so existing stores
    that hard-code a vector dim (Milvus / Chroma collections built around
    384-d ``all-MiniLM-L6-v2`` vectors) keep working.
    """

    def __init__(self, config):
        super().__init__(config)
        from openai import OpenAI

        api_base = (
            os.getenv("OPENAI_EMBEDDING_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or getattr(self.config, "base_url", None)
        )
        api_key = os.getenv("OPENAI_API_KEY") or getattr(self.config, "api_key", None)
        self.model_name = getattr(self.config, "path", None) or "text-embedding-3-small"
        self.dimensions = getattr(self.config, "dimensions", None)
        self.client = OpenAI(api_key=api_key, base_url=api_base)

    def __call__(self, text, return_type="numpy"):
        kwargs = {"model": self.model_name, "input": [text]}
        if self.dimensions is not None and self.model_name.startswith("text-embedding-3"):
            kwargs["dimensions"] = int(self.dimensions)
        resp = self.client.embeddings.create(**kwargs)
        embeddings = np.asarray([item.embedding for item in resp.data], dtype="float32")
        if return_type == "numpy":
            return embeddings
        elif return_type == "tensor":
            return torch.from_numpy(embeddings).to(self.device)
        else:
            return "Unrecognized Return Type."