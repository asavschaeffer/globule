"""
GlobuleAPI: A clean, UI-agnostic interface to the core application logic.

This class provides a stable, high-level API for any frontend (TUI, Web, etc.)
to interact with Globule's features without needing to know about the
underlying orchestration, storage, or service providers.
"""

from typing import List, Dict, Any, Optional
from uuid import UUID

from .interfaces import IStorageManager
from .models import ProcessedGlobuleV1, StructuredQuery
from ..orchestration.engine import OrchestrationEngine
from ..storage.file_manager import FileManager
from ..storage.sqlite_manager import SQLiteStorageManager

class GlobuleAPI:
    """
    The single point of entry for all frontend interactions.
    """

    def __init__(self, storage: IStorageManager, orchestrator: OrchestrationEngine):
        self.storage = storage
        self.orchestrator = orchestrator
        self.file_manager = FileManager()

    async def add_thought(self, text: str, source: str = "api") -> ProcessedGlobuleV1:
        """
        Adds and processes a new thought.

        Args:
            text: The raw text of the thought.
            source: The source of the input (e.g., 'tui', 'web').

        Returns:
            The processed globule object.
        """
        from .models import GlobuleV1
        from ..core.models import EnrichedInput
        from datetime import datetime
        
        # Create enriched input for orchestrator
        enriched_input = EnrichedInput(
            original_text=text,
            enriched_text=text,
            detected_schema_id=None,
            schema_config=None,
            additional_context={},
            source=source,
            timestamp=datetime.now(),
            verbosity="concise"
        )
        
        # Process through orchestrator
        processed_globule = await self.orchestrator.process_globule(enriched_input)
        
        # Store the result
        globule_id = await self.storage.store_globule(processed_globule)
        
        return processed_globule

    async def search_semantic(self, query: str, limit: int = 10) -> List[ProcessedGlobuleV1]:
        """
        Performs a semantic vector search for globules.

        Args:
            query: The natural language query.
            limit: The maximum number of results to return.

        Returns:
            A list of processed globules that are semantically similar to the query.
        """
        # Generate embedding for the query
        embedding_result = await self.orchestrator.embedding_provider.embed_single(query)
        
        # Search storage by embedding
        results = await self.storage.search_by_embedding(embedding_result.embedding, limit)
        
        return [result[0] for result in results]  # Extract globules from (globule, similarity) tuples

    async def search_structured(self, query: StructuredQuery) -> List[ProcessedGlobuleV1]:
        """
        Performs a structured search based on metadata filters.

        Args:
            query: A StructuredQuery object with filters.

        Returns:
            A list of globules matching the structured query.
        """
        return await self.storage.query_structured(query)

    async def get_globule_by_id(self, globule_id: UUID) -> Optional[ProcessedGlobuleV1]:
        """
        Retrieves a single globule by its unique ID.

        Args:
            globule_id: The UUID of the globule.

        Returns:
            The processed globule object, or None if not found.
        """
        return await self.storage.get(globule_id)

    async def get_all_globules(self, limit: int = 100) -> List[ProcessedGlobuleV1]:
        """
        Retrieves all globules from storage, up to a limit.

        Args:
            limit: The maximum number of globules to retrieve.

        Returns:
            A list of all processed globules.
        """
        return await self.storage.get_all(limit)

    async def get_summary_for_text(self, text: str) -> str:
        """
        Generates a summary for a given piece of text.

        Args:
            text: The text to summarize.

        Returns:
            The generated summary.
        """
        try:
            # Use parser to generate summary
            schema_param = {"name": "default"}  # Use default schema for summary
            parsed_data = await self.orchestrator.parser_provider.parse(text, schema_param)
            
            # Try to extract summary from various possible fields
            if isinstance(parsed_data, dict):
                return (parsed_data.get("summary") or 
                       parsed_data.get("title") or 
                       text[:200] + "..." if len(text) > 200 else text)
            
            return text[:200] + "..." if len(text) > 200 else text
            
        except Exception:
            # Fallback to simple truncation
            return text[:200] + "..." if len(text) > 200 else text

    async def reconcile_files(self) -> Dict[str, Any]:
        """
        Reconciles the file system with the database.

        Returns:
            Statistics about the reconciliation process.
        """
        if isinstance(self.storage, SQLiteStorageManager):
            return await self.file_manager.reconcile_files_with_database(self.storage)
        return {"error": "Reconciliation only supported for SQLiteStorageManager"}

    async def export_draft(self, draft_content: str, file_path: str) -> bool:
        """
        Exports draft content to a file.

        Args:
            draft_content: The content of the draft to export.
            file_path: The path to save the file to.

        Returns:
            True if the export was successful, False otherwise.
        """
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(draft_content)
            return True
        except IOError:
            return False

    async def get_clusters(self) -> Any:
        """
        Analyzes all globules and groups them into semantic clusters.

        Returns:
            A ClusteringAnalysis object containing the discovered clusters.
        """
        from ..services.clustering.semantic_clustering import SemanticClusteringEngine

        all_globules = await self.get_all_globules(limit=1000)

        # Filter for globules that are suitable for clustering
        clusterable_globules = []
        for globule in all_globules:
            # This logic is moved from the engine to the API layer
            if (hasattr(globule, 'embedding') and globule.embedding and
                hasattr(globule, 'embedding_confidence') and globule.embedding_confidence > 0.5 and
                len(globule.original_globule.raw_text.strip()) > 10):
                clusterable_globules.append(globule)

        clustering_engine = SemanticClusteringEngine()
        analysis = await clustering_engine.analyze_semantic_clusters(
            globules=clusterable_globules,
            min_globules=5
        )
        return analysis
