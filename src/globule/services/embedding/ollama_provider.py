"""
Ollama-based embedding provider for Globule.

Implements the EmbeddingProvider interface using Ollama's local API.
"""

import asyncio
import aiohttp
import numpy as np
from typing import List, Optional
import logging

from globule.core.interfaces import IEmbeddingProvider
from globule.config.settings import get_config

logger = logging.getLogger(__name__)


class OllamaEmbeddingProvider(IEmbeddingProvider):
    """Ollama implementation of EmbeddingProvider"""
    
    def __init__(self, 
                 base_url: Optional[str] = None,
                 model: Optional[str] = None,
                 timeout: Optional[int] = None):
        self.config = get_config()
        self.base_url = base_url or self.config.ollama_base_url
        self.model = model or self.config.default_embedding_model
        self.timeout = timeout or self.config.ollama_timeout
        self._session: Optional[aiohttp.ClientSession] = None
        self._dimension: Optional[int] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def embed(self, text: str) -> np.ndarray:
        """Generate embedding for single text"""
        session = await self._get_session()
        
        payload = {
            "model": self.model,
            "input": text.strip(),
            "truncate": True
        }
        
        try:
            async with session.post(
                f"{self.base_url}/api/embed",
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"Ollama API error: {response.status} - {error_text}")
                
                data = await response.json()
                
                # Handle Ollama's response format
                if "embeddings" in data and len(data["embeddings"]) > 0:
                    embedding = np.array(data["embeddings"][0], dtype=np.float32)
                    
                    # Cache dimension on first call
                    if self._dimension is None:
                        self._dimension = len(embedding)
                    
                    return embedding
                elif "embedding" in data:
                    # Alternative response format
                    embedding = np.array(data["embedding"], dtype=np.float32)
                    
                    if self._dimension is None:
                        self._dimension = len(embedding)
                    
                    return embedding
                else:
                    raise RuntimeError(f"Invalid response format from Ollama: {data}")
                    
        except aiohttp.ClientError as e:
            raise RuntimeError(f"Failed to connect to Ollama: {e}")
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts"""
        # For Phase 1, implement as sequential calls
        # Phase 2 can optimize with true batch processing
        embeddings = []
        for text in texts:
            embedding = await self.embed(text)
            embeddings.append(embedding)
        return embeddings
    
    def get_dimension(self) -> int:
        """Return embedding dimensionality"""
        if self._dimension is None:
            # Default dimension for mxbai-embed-large
            return 1024
        return self._dimension
    
    async def close(self) -> None:
        """Close HTTP session"""
        if self._session:
            await self._session.close()
            self._session = None
    
    async def health_check(self) -> bool:
        """Check if Ollama is running and model is available"""
        try:
            session = await self._get_session()
            
            # Check if Ollama is running
            async with session.get(f"{self.base_url}/api/tags") as response:
                if response.status != 200:
                    return False
                
                data = await response.json()
                available_models = [model["name"] for model in data.get("models", [])]
                
                # Check if our model is available, accounting for tags like ":latest"
                return any(m.startswith(self.model) for m in available_models)
                
        except Exception as e:
            logger.debug(f"Health check failed: {e}")
            return False