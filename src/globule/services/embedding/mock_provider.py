"""
Mock embedding provider for testing and fallback scenarios.

Used when Ollama is not available or for testing purposes.
Provides consistent interface with OllamaEmbeddingProvider.
"""

import numpy as np
from typing import List
import asyncio


class MockEmbeddingProvider:
    """Mock embedding provider that generates random embeddings."""
    
    def __init__(self, dimension: int = 1024):
        """
        Initialize mock provider.
        
        Args:
            dimension: Dimension of generated embeddings
        """
        self.dimension = dimension
        self._closed = False
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension
    
    async def embed(self, text: str) -> np.ndarray:
        """
        Generate a mock embedding for the given text.
        
        Args:
            text: Input text to embed
            
        Returns:
            Random normalized embedding vector
        """
        if self._closed:
            raise RuntimeError("MockEmbeddingProvider has been closed")
        
        # Generate consistent embeddings based on text hash for reproducibility
        seed = hash(text) % 2**32
        rng = np.random.RandomState(seed)
        embedding = rng.randn(self.dimension).astype(np.float32)
        
        # Normalize to unit vector
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
    
    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate mock embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        return [await self.embed(text) for text in texts]
    
    async def close(self) -> None:
        """Close the provider (cleanup)."""
        self._closed = True
    
    async def health_check(self) -> bool:
        """Always returns True for mock provider."""
        return not self._closed