from typing import List
from globule.core.interfaces import IEmbeddingProvider
from globule.core.errors import EmbeddingError
from .ollama_provider import OllamaEmbeddingProvider
import numpy as np


class OllamaEmbeddingAdapter(IEmbeddingProvider):
    def __init__(self, provider: OllamaEmbeddingProvider):
        self._provider = provider
    
    async def embed(self, text: str) -> List[float]:
        try:
            embedding_result = await self._provider.embed(text)
            # Handle both numpy arrays and lists
            if hasattr(embedding_result, 'tolist'):
                return embedding_result.tolist()
            else:
                return embedding_result
        except Exception as e:
            raise EmbeddingError(f"Ollama embedding provider failed: {e}") from e