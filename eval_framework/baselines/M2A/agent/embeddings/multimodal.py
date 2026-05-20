from typing import TypedDict
from openai import OpenAI
from openai.types.create_embedding_response import CreateEmbeddingResponse

from ..utils.message import encode_image_to_base64

class MultimodalEmbedder:
    """Local SigLIP2 image embedding service"""

    def __init__(self, api_key: str, base_url: str, model: str = None):
        base_url = base_url.rstrip("/")
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        if model is None:
            models = client.models.list()
            model = models.data[0].id
        self.model = model

    def _create_chat_embeddings(self, messages: list[dict]) -> list[float]:
        """
        Convenience function for accessing vLLM's Chat Embeddings API,
        which is an extension of OpenAI's existing Embeddings API.
        """
        return self.client.post(
            "/embeddings",
            cast_to=CreateEmbeddingResponse,
            body={
                "messages": messages,
                "model": self.model,
                "encoding_format": 'float',
            },
        ).data[0].embedding

    def _create_embeddings(self, content: list[dict]) -> list[float]:
        return self._create_chat_embeddings(
            [{"role": "user", "content": content}]
        )

    def embed_text(self, text: str) -> list[float]:
        return self._create_embeddings([
            {"type": "text", "text": text}
        ])

    def encode_image(self, image: str) -> list[float]:
        """
        Get image embeddings. 

        Arguments:
            image: should be either path to image or image url
        """
        if not image.startswith("http"):
            # local image
            image = encode_image_to_base64(image)
        return self._create_embeddings([
            {"type":"image_url", "image_url": {"url": image}}
        ])
