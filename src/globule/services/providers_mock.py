"""
Mock providers implementing the core interfaces for testing and Phase 1.

These providers implement the IParserProvider, IEmbeddingProvider, and IStorageManager
interfaces with dummy implementations for use during Phase 1 testing.
"""

import asyncio
import time
from typing import Dict, Any, List
from uuid import UUID

from globule.core.interfaces import IParserProvider, IEmbeddingProvider, IStorageManager
from globule.core.models import ProcessedGlobuleV1
from globule.core.errors import ParserError, EmbeddingError, StorageError


class MockParserProvider(IParserProvider):
    """Mock parser provider for Phase 1 testing."""
    
    async def parse(self, text: str) -> dict:
        """Return mock parsed data."""
        # Simulate processing time
        await asyncio.sleep(0.01)
        
        return {
            "title": text[:50] + "..." if len(text) > 50 else text,
            "category": "note", 
            "domain": "general",
            "keywords": [],
            "entities": [],
            "metadata": {
                "mock": True,
                "parser_version": "mock-1.0.0"
            }
        }


class MockEmbeddingProvider(IEmbeddingProvider):
    """Mock embedding provider for Phase 1 testing."""
    
    def __init__(self, dimension: int = 1024):
        self.dimension = dimension
    
    async def embed(self, text: str) -> List[float]:
        """Return mock embedding vector."""
        # Simulate processing time
        await asyncio.sleep(0.01)
        
        # Generate consistent embeddings based on text hash for reproducibility
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        seed = int(hash_obj.hexdigest()[:8], 16)
        
        # Simple deterministic generation
        embedding = []
        for i in range(self.dimension):
            val = ((seed + i) % 10000) / 10000.0 - 0.5
            embedding.append(val)
        
        # Normalize to unit vector
        norm = sum(x*x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x/norm for x in embedding]
        
        return embedding


class MockStorageManager(IStorageManager):
    """Mock storage manager for Phase 1 testing."""
    
    def __init__(self):
        self._storage = {}  # In-memory storage for testing
    
    def save(self, globule: ProcessedGlobuleV1) -> None:
        """Save globule to in-memory storage."""
        self._storage[globule.globule_id] = globule
    
    def get(self, globule_id: UUID) -> ProcessedGlobuleV1:
        """Retrieve globule from in-memory storage."""
        if globule_id not in self._storage:
            raise StorageError(f"Globule {globule_id} not found")
        return self._storage[globule_id]
    
    def list_all(self) -> List[ProcessedGlobuleV1]:
        """Return all stored globules."""
        return list(self._storage.values())
    
    def clear(self) -> None:
        """Clear all stored globules (for testing)."""
        self._storage.clear()
    
    async def search(self, query: str, limit: int = 10) -> List[ProcessedGlobuleV1]:
        """Mock search implementation."""
        # Simple mock search - return globules containing query text
        results = []
        for globule in self._storage.values():
            if query.lower() in globule.original_globule.raw_text.lower():
                results.append(globule)
        
        return results[:limit]
    
    async def execute_sql(self, query: str, query_name: str = "Query") -> Dict[str, Any]:
        """Mock SQL execution."""
        return {
            "type": "sql_results", 
            "query": query,
            "query_name": query_name,
            "results": [{"mock": "result", "count": len(self._storage)}],
            "headers": ["mock", "count"],
            "count": 1
        }