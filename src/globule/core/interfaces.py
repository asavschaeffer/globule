"""
Abstract interfaces for Globule components.

These interfaces define the contracts that all implementations must follow,
enabling pluggability and testing.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import numpy as np

from .models import ProcessedGlobule, EnrichedInput, EmbeddingResult, ParsingResult


class EmbeddingProvider(ABC):
    """Abstract base for embedding providers"""
    
    @abstractmethod
    async def embed(self, text: str) -> np.ndarray:
        """Generate embedding for single text"""
        pass
    
    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts"""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Return embedding dimensionality"""
        pass


class ParsingProvider(ABC):
    """Abstract base for parsing providers"""
    
    @abstractmethod
    async def parse(self, text: str, schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Parse text to extract structured data"""
        pass


class StorageManager(ABC):
    """Abstract interface for storage operations"""
    
    @abstractmethod
    async def store_globule(self, globule: ProcessedGlobule) -> str:
        """Store a processed globule and return its ID"""
        pass
    
    @abstractmethod
    async def get_globule(self, globule_id: str) -> Optional[ProcessedGlobule]:
        """Retrieve a globule by ID"""
        pass
    
    @abstractmethod
    async def get_recent_globules(self, limit: int = 100) -> List[ProcessedGlobule]:
        """Get recent globules ordered by creation time"""
        pass
    
    @abstractmethod
    async def search_by_embedding(
        self, 
        query_vector: np.ndarray, 
        limit: int = 50,
        similarity_threshold: float = 0.5
    ) -> List[tuple[ProcessedGlobule, float]]:
        """Find semantically similar globules"""
        pass


class OrchestrationEngine(ABC):
    """Abstract interface for orchestration engines"""
    
    @abstractmethod
    async def process_globule(self, enriched_input: EnrichedInput) -> ProcessedGlobule:
        """Process an enriched input into a processed globule"""
        pass