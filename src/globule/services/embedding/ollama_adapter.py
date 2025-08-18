import time
from typing import List
from globule.core.interfaces import BaseEmbeddingAdapter
from globule.core.models import EmbeddingResult
from globule.core.errors import EmbeddingError
from .ollama_provider import OllamaEmbeddingProvider
import numpy as np


class OllamaEmbeddingAdapter(BaseEmbeddingAdapter):
    """Ollama embedding adapter implementing BaseEmbeddingAdapter interface."""
    
    def __init__(self, provider: OllamaEmbeddingProvider):
        self._provider = provider
    
    async def embed_single(self, text: str) -> EmbeddingResult:
        """Generate embedding for a single text using Ollama provider."""
        start_time = time.time()
        try:
            embedding_vector = await self._provider.embed(text)
            # Handle both numpy arrays and lists
            if hasattr(embedding_vector, 'tolist'):
                embedding_vector = embedding_vector.tolist()
            
            processing_time = (time.time() - start_time) * 1000
            
            return EmbeddingResult(
                embedding=embedding_vector,
                dimensions=len(embedding_vector),
                model_name=self.get_model_name(),
                processing_time_ms=processing_time,
                metadata={
                    "provider": "ollama",
                    "base_url": self._provider.base_url,
                    "timeout": self._provider.timeout
                }
            )
        except Exception as e:
            raise EmbeddingError(f"Ollama embedding provider failed: {e}") from e
    
    async def batch_embed(self, texts: List[str]) -> List[EmbeddingResult]:
        """Generate embeddings for multiple texts efficiently."""
        # For now, process sequentially. Could be optimized later if Ollama supports batch operations
        results = []
        for text in texts:
            result = await self.embed_single(text)
            results.append(result)
        return results
    
    def get_dimensions(self) -> int:
        """Return the embedding dimensions for this model."""
        # This would ideally be cached or retrieved from model info
        # For now, we'll use a reasonable default based on common Ollama models
        return 1024  # Common dimension for many embedding models
    
    def get_model_name(self) -> str:
        """Return the name of the embedding model being used."""
        return self._provider.model