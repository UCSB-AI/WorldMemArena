from typing import TypedDict
from openai import OpenAI
from langchain.embeddings.base import Embeddings


class TextEmbedding(Embeddings):
    """OpenAI text embedding service"""

    def __init__(self, api_key: str, base_url, model: str = "text-embedding-3-small"):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        responses = self.client.embeddings.create(
            input=texts,
            model=self.model,
        )
        return [data.embedding for data in responses.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
