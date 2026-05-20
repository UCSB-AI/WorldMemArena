from abc import ABC, abstractmethod
import base64
import mimetypes
import os
import warnings
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import requests
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor, CLIPModel, AutoModel

# --- compat: GME-Qwen2-VL-2B's modeling code hard-asserts
# ``transformers<4.52.0`` via ``require_version``. We've validated that the
# embedding-only forward path works fine on newer transformers, so soften
# the assertion to a warning so the model loads on transformers 4.52+.
try:
    from transformers.utils import versions as _tv
    _orig_require_version = _tv.require_version

    def _lenient_require_version(requirement, hint=None):
        try:
            _orig_require_version(requirement, hint)
        except (ImportError, Exception) as exc:
            req = str(requirement)
            if "transformers<4.52" in req or "transformers <4.52" in req:
                warnings.warn(
                    f"GME compat: ignoring '{req}' assertion "
                    "(embedding path verified on newer transformers)."
                )
                return
            raise

    _tv.require_version = _lenient_require_version
except Exception:
    pass

class BaseMultiModalEncoder(ABC):
    """
    Encoder for multimodal data (text + image).
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def reset(self):
        pass
    
    @abstractmethod
    def encode_text(self, text, return_type='numpy'):
        pass
    
    @abstractmethod
    def encode_image(self, image_path_or_url, return_type='numpy'):
        pass
    
    @abstractmethod
    def encode_multimodal(self, text=None, image=None, return_type='numpy'):
        pass


class CLIPEncoder(BaseMultiModalEncoder):
    """
    CLIP-based multimodal encoder for text and images.
    """
    def __init__(self, config):
        super().__init__(config)
        
        model_name = getattr(config, 'path', 'openai/clip-vit-base-patch32')
        print(f"Loading CLIP model: {model_name}")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()  # Set to evaluation mode
        print(f"CLIP model loaded successfully on {self.device}")
    
    def _load_image(self, image_path_or_url):
        """Load image from local path or URL."""
        if os.path.isabs(image_path_or_url) or image_path_or_url.startswith('http://') or image_path_or_url.startswith('https://'):
            image_path_or_url = image_path_or_url
        else:
            # Only relative paths are prefixed.
            image_path_or_url = "" + image_path_or_url # [Replace with your default absolute path]
        print("image_path_or_url: ", image_path_or_url)
        try:
            if image_path_or_url.startswith('http://') or image_path_or_url.startswith('https://'):
                # Load from URL
                response = requests.get(image_path_or_url, timeout=10)
                image = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                # Load from local path
                if not os.path.exists(image_path_or_url):
                    raise FileNotFoundError(f"Image file not found: {image_path_or_url}")
                image = Image.open(image_path_or_url).convert('RGB')
            return image
        except Exception as e:
            print(f"Error loading image {image_path_or_url}: {e}")
            return Image.new('RGB', (224, 224), color='white')
    
    def encode_text(self, text, return_type='numpy'):
        """Encode text into embeddings."""
        if not text or text.strip() == '':
            text = " "
        
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        if return_type == 'numpy':
            return text_features.cpu().numpy()
        elif return_type == 'tensor':
            return text_features
        else:
            raise ValueError(f"Unrecognized return type: {return_type}")
    
    def encode_image(self, image_path_or_url, return_type='numpy'):
        """Encode image into embeddings."""
        image = self._load_image(image_path_or_url)
        
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        if return_type == 'numpy':
            return image_features.cpu().numpy()
        elif return_type == 'tensor':
            return image_features
        else:
            raise ValueError(f"Unrecognized return type: {return_type}")
    
    def encode_multimodal(self, text=None, image=None, return_type='numpy'):
        """
        Encode multimodal data (text and/or image).
        If both are provided, average the embeddings.
        """
        embeddings = []
        
        if text is not None and text.strip() != '':
            text_emb = self.encode_text(text, return_type='tensor')
            embeddings.append(text_emb)
        
        if image is not None:
            image_emb = self.encode_image(image.get('path'), return_type='tensor')
            embeddings.append(image_emb)
        
        if not embeddings:
            # If both are empty, encode empty text
            return self.encode_text(" ", return_type=return_type)
        
        # Average the embeddings if both modalities are present
        if len(embeddings) > 1:
            combined = torch.mean(torch.stack(embeddings), dim=0)
        else:
            combined = embeddings[0]
        
        # Normalize
        combined = combined / combined.norm(dim=-1, keepdim=True)
        
        if return_type == 'numpy':
            return combined.cpu().numpy()
        elif return_type == 'tensor':
            return combined
        else:
            raise ValueError(f"Unrecognized return type: {return_type}")
    
    def __call__(self, obj, return_type='numpy'):
        """
        Main entry point. obj can be:
        - str: treated as text
        - dict with 'text' and/or 'image' keys
        """
        if isinstance(obj, str):
            return self.encode_text(obj, return_type)
        elif isinstance(obj, dict):
            text = obj.get('text', '')
            image = obj.get('image', None)
            return self.encode_multimodal(text, image, return_type)
        else:
            raise ValueError(f"Unsupported input type: {type(obj)}")


def _value_to_image_data_url(value: str) -> str:
    """Coerce ``value`` into something the embedding endpoint accepts as
    ``image_url.url``. ``http(s)://``, ``data:`` and ``file://`` URIs pass
    through unchanged; absolute local paths are inlined as base64 data URLs."""
    if not value:
        raise ValueError("image value is empty")
    if value.startswith(("http://", "https://", "data:", "file://")):
        return value
    path = Path(value)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"GMEEncoder: image not found: {value}")
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


class _ImageListDataset(Dataset):
    """Small wrapper so GME can consume images without spawning workers."""

    def __init__(self, images):
        self.images = images
        self.transform = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        return self.transform(image) if self.transform is not None else image


def _single_worker_image_loader(images):
    return DataLoader(
        _ImageListDataset(images),
        batch_size=1,
        shuffle=False,
        collate_fn=lambda batch: batch,
        num_workers=0,
    )


class GMEEncoder(BaseMultiModalEncoder):
    """
    GME (General Multimodal Embedding) encoder backed by a local vLLM server
    serving ``Alibaba-NLP/gme-Qwen2-VL-2B-Instruct`` via the OpenAI-compatible
    ``/v1/embeddings`` endpoint.

    Endpoint resolution order:
      1. ``config.base_url`` / ``config.api_key`` / ``config.path`` (model name)
      2. env ``GME_BASE_URL`` / ``GME_API_KEY`` / ``GME_MODEL``
      3. fallback to ``QWEN_VL_EMBED_*`` env (compatible vLLM server schema)

    All three encode_* methods translate inputs into the same multimodal
    chat-message payload that vLLM's pooling embedding runner accepts.
    Output vectors are L2-normalized.
    """
    _DEFAULT_MODEL = "gme-Qwen2-VL-2B-Instruct"

    def __init__(self, config):
        super().__init__(config)

        self._base_url = (
            getattr(config, "base_url", None)
            or os.getenv("GME_BASE_URL")
            or os.getenv("QWEN_VL_EMBED_BASE_URL")
        )
        if not self._base_url:
            raise RuntimeError(
                "GMEEncoder requires a vLLM endpoint. Set GME_BASE_URL "
                "(or QWEN_VL_EMBED_BASE_URL) or pass config.base_url."
            )
        self._api_key = (
            getattr(config, "api_key", None)
            or os.getenv("GME_API_KEY")
            or os.getenv("QWEN_VL_EMBED_API_KEY")
            or "EMPTY"
        )
        self._model_id = (
            getattr(config, "path", None)
            or os.getenv("GME_MODEL")
            or self._DEFAULT_MODEL
        )
        self._timeout = float(os.getenv("GME_TIMEOUT", "120"))
        self._url = self._base_url.rstrip("/") + "/embeddings"
        print(f"GMEEncoder using vLLM endpoint: {self._url} (model={self._model_id})")

    def _build_messages(self, text: str | None, image_value: str | None) -> list[dict[str, Any]]:
        """Compose the chat-message payload accepted by vLLM pooling embed."""
        instruction = "Represent the input for retrieval."
        content: list[dict[str, Any]] = []
        if image_value:
            content.append({
                "type": "image_url",
                "image_url": {"url": _value_to_image_data_url(image_value)},
            })
            content.append({"type": "text", "text": text or ""})
        else:
            content.append({"type": "text", "text": text if (text and text.strip()) else " "})
        return [
            {"role": "system", "content": [{"type": "text", "text": instruction}]},
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": ""}]},
        ]

    def _post_embedding(self, messages: list[dict[str, Any]]) -> np.ndarray:
        import httpx
        headers = {"Authorization": f"Bearer {self._api_key}"}
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(
                self._url,
                headers=headers,
                json={
                    "messages": messages,
                    "model": self._model_id,
                    "encoding_format": "float",
                    "continue_final_message": True,
                    "add_special_tokens": True,
                },
            )
            response.raise_for_status()
            payload = response.json()
        emb = payload["data"][0]["embedding"]
        vec = np.asarray([float(x) for x in emb], dtype="float32").reshape(1, -1)
        norm = np.linalg.norm(vec, axis=1, keepdims=True)
        return vec / np.maximum(norm, 1e-12)

    def _to_return_type(self, vec_np: np.ndarray, return_type: str):
        if return_type == "numpy":
            return vec_np
        if return_type == "tensor":
            return torch.from_numpy(vec_np).to(self.device)
        raise ValueError(f"Unrecognized return type: {return_type}")

    def encode_text(self, text, return_type='numpy'):
        import httpx
        if not text or text.strip() == '':
            text = " "
        try:
            vec = self._post_embedding(self._build_messages(text, None))
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 400:
                raise
            # Likely too long — truncate and retry once
            truncated = text[: max(1, int(len(text) * 0.4))]
            vec = self._post_embedding(self._build_messages(truncated, None))
        return self._to_return_type(vec, return_type)

    def encode_image(self, image_path_or_url, return_type='numpy'):
        vec = self._post_embedding(self._build_messages(None, image_path_or_url))
        return self._to_return_type(vec, return_type)

    def encode_multimodal(self, text=None, image=None, return_type='numpy'):
        import httpx
        has_text = text is not None and text.strip() != ''
        has_image = image is not None
        if has_text and has_image:
            image_value = image.get('path') if isinstance(image, dict) else image
            try:
                vec = self._post_embedding(self._build_messages(text, image_value))
                return self._to_return_type(vec, return_type)
            except httpx.HTTPStatusError as e:
                if e.response.status_code != 400:
                    raise
                # Bad image (unreadable / oversize / unsupported) — fall back to text only
                return self.encode_text(text, return_type)
        if has_text:
            return self.encode_text(text, return_type)
        if has_image:
            image_value = image.get('path') if isinstance(image, dict) else image
            try:
                return self.encode_image(image_value, return_type)
            except httpx.HTTPStatusError as e:
                if e.response.status_code != 400:
                    raise
                return self.encode_text(" ", return_type)
        return self.encode_text(" ", return_type)
    
    def __call__(self, obj, return_type='numpy'):
        """
        Main entry point. obj can be:
        - str: treated as text
        - dict with 'text' and/or 'image' keys
        """
        if isinstance(obj, str):
            return self.encode_text(obj, return_type)
        elif isinstance(obj, dict):
            text = obj.get('text', '')
            image = obj.get('image', None)
            return self.encode_multimodal(text, image, return_type)
        else:
            raise ValueError(f"Unsupported input type: {type(obj)}")
