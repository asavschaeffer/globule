"""Embedding engine for Globule - converts text to embeddings for semantic search."""

import asyncio
from typing import List, Optional, Protocol
from functools import lru_cache

import numpy as np
import httpx
from sentence_transformers import SentenceTransformer


class Embedder(Protocol):
    """Abstract interface for embedding engines."""
    
    async def embed_text(self, text: str) -> np.ndarray:
        """Convert text to embedding vector."""
        ...
    
    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Convert multiple texts to embedding vectors."""
        ...


class OllamaEmbedder:
    """Ollama-based embedding engine using mxbai-embed-large model."""
    
    def __init__(self, model_name: str = "mxbai-embed-large:latest", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def embed_text(self, text: str) -> np.ndarray:
        """Convert text to embedding vector using Ollama."""
        try:
            response = await self.client.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model_name,
                    "prompt": text
                }
            )
            response.raise_for_status()
            
            result = response.json()
            embedding = np.array(result["embedding"], dtype=np.float32)
            return embedding
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding: {e}")
    
    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Convert multiple texts to embedding vectors."""
        tasks = [self.embed_text(text) for text in texts]
        return await asyncio.gather(*tasks)
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()


class SentenceTransformerEmbedder:
    """Fallback embedding engine using sentence-transformers (local)."""
    
    def __init__(self, model_name: str = "BAAI/bge-small-en"):
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the model."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    async def embed_text(self, text: str) -> np.ndarray:
        """Convert text to embedding vector."""
        loop = asyncio.get_event_loop()
        
        # Run in thread pool to avoid blocking
        embedding = await loop.run_in_executor(
            None, 
            lambda: self.model.encode(text, convert_to_numpy=True)
        )
        
        return embedding.astype(np.float32)
    
    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Convert multiple texts to embedding vectors."""
        loop = asyncio.get_event_loop()
        
        embeddings = await loop.run_in_executor(
            None,
            lambda: self.model.encode(texts, convert_to_numpy=True)
        )
        
        return [emb.astype(np.float32) for emb in embeddings]


class CachedEmbedder:
    """Wrapper that adds caching to any embedder."""
    
    def __init__(self, embedder: Embedder, cache_size: int = 1000):
        self.embedder = embedder
        self.cache_size = cache_size
    
    @lru_cache(maxsize=1000)
    def _cached_embed(self, text: str) -> np.ndarray:
        """Cached embedding function (sync)."""
        # This will be replaced with proper async caching
        return None
    
    async def embed_text(self, text: str) -> np.ndarray:
        """Convert text to embedding vector with caching."""
        # Simple cache key based on text hash
        cache_key = hash(text)
        
        # For now, just pass through to the underlying embedder
        # TODO: Implement proper async LRU cache
        return await self.embedder.embed_text(text)
    
    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Convert multiple texts to embedding vectors with caching."""
        return await self.embedder.embed_batch(texts)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def find_most_similar(query_embedding: np.ndarray, embeddings: List[np.ndarray], top_k: int = 5) -> List[tuple]:
    """Find the most similar embeddings to the query."""
    similarities = []
    
    for i, embedding in enumerate(embeddings):
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((i, similarity))
    
    # Sort by similarity (descending) and return top k
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


async def create_embedder(use_ollama: bool = True) -> Embedder:
    """Factory function to create the appropriate embedder."""
    if use_ollama:
        try:
            embedder = OllamaEmbedder()
            # Test if Ollama is available
            await embedder.embed_text("test")
            return embedder
        except Exception:
            # Fall back to sentence-transformers
            print("Ollama not available, falling back to sentence-transformers")
    
    return SentenceTransformerEmbedder()