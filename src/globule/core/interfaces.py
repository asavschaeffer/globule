"""
Abstract service contracts (interfaces) for the Globule core architecture.

These Abstract Base Classes (ABCs) define the "verbs" of the system,
establishing the boundaries between the orchestration engine and its providers.
"""
from abc import ABC, abstractmethod
from typing import List
from uuid import UUID

from globule.core.models import GlobuleV1, ProcessedGlobuleV1
from globule.core.errors import ParserError, EmbeddingError, StorageError

class IParserProvider(ABC):
    """Interface for a service that parses raw text into structured data."""
    
    @abstractmethod
    def parse(self, text: str) -> dict:
        """
        Parses the raw text from a Globule.

        Args:
            text: The raw input text.

        Returns:
            A dictionary containing the extracted structured data.
            
        Raises:
            ParserError: If parsing fails.
        """
        pass

class IEmbeddingProvider(ABC):
    """Interface for a service that generates vector embeddings."""
    
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """
        Generates a vector embedding for the given text.

        Args:
            text: The input text.

        Returns:
            A list of floats representing the vector embedding.
            
        Raises:
            EmbeddingError: If embedding generation fails.
        """
        pass

class IStorageManager(ABC):
    """Interface for a service that handles storage and retrieval of Globules."""
    
    @abstractmethod
    def save(self, globule: ProcessedGlobuleV1) -> None:
        """
        Saves a ProcessedGlobule to the storage backend.

        Args:
            globule: The ProcessedGlobule to save.
            
        Raises:
            StorageError: If saving fails.
        """
        pass

    @abstractmethod
    def get(self, globule_id: UUID) -> ProcessedGlobuleV1:
        """
        Retrieves a ProcessedGlobule from the storage backend.

        Args:
            globule_id: The UUID of the Globule to retrieve.

        Returns:
            The retrieved ProcessedGlobule.
            
        Raises:
            StorageError: If the Globule is not found or retrieval fails.
        """
        pass

class IOrchestrationEngine(ABC):
    """Interface for the core orchestration engine."""
    
    @abstractmethod
    def process(self, globule: GlobuleV1) -> ProcessedGlobuleV1:
        """
        Processes a raw Globule into a ProcessedGlobule.

        This method orchestrates calls to the parser, embedder, and other
        services to enrich the raw Globule.

        Args:
            globule: The raw GlobuleV1 to process.

        Returns:
            The resulting ProcessedGlobuleV1.
        """
        pass
