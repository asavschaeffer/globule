"""
Abstract service contracts (interfaces) for the Globule core architecture.

These Abstract Base Classes (ABCs) define the "verbs" of the system,
establishing the boundaries between the orchestration engine and its providers.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from uuid import UUID

from globule.core.models import GlobuleV1, ProcessedGlobuleV1
from globule.core.errors import ParserError, EmbeddingError, StorageError

class IParserProvider(ABC):
    """Interface for a service that parses raw text into structured data."""
    
    @abstractmethod
    async def parse(self, text: str) -> dict:
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
    async def embed(self, text: str) -> List[float]:
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

    @abstractmethod
    async def search(self, query: str, limit: int = 10) -> List[ProcessedGlobuleV1]:
        """
        Search for globules using a natural language query.

        Args:
            query: The search query string.
            limit: Maximum number of results to return.

        Returns:
            List of ProcessedGlobules matching the query.
            
        Raises:
            StorageError: If search fails.
        """
        pass

    @abstractmethod
    async def execute_sql(self, query: str, query_name: str = "Query") -> Dict[str, Any]:
        """
        Execute a SQL query against the storage backend.

        Args:
            query: The SQL query to execute.
            query_name: Optional name for the query (for logging/metadata).

        Returns:
            Dictionary containing query results and metadata.
            
        Raises:
            StorageError: If query execution fails or query is invalid.
        """
        pass

class ISchemaManager(ABC):
    """Interface for a service that manages loading and accessing schemas."""

    @abstractmethod
    def get_schema(self, schema_name: str) -> Dict[str, Any]:
        """Retrieves a schema by its name."""
        pass

    @abstractmethod
    def detect_schema_for_text(self, text: str) -> str | None:
        """Detects the most appropriate schema for a given text."""
        pass

    @abstractmethod
    def get_available_schemas(self) -> List[str]:
        """Returns a list of all available schema names."""
        pass

class IOrchestrationEngine(ABC):
    """Interface for the core orchestration engine."""
    
    @abstractmethod
    async def process(self, globule: GlobuleV1) -> ProcessedGlobuleV1:
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
