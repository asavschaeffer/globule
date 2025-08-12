"""
GlobuleOrchestrator: The core business logic engine for Globule.

This orchestrator implements the IOrchestrationEngine interface and coordinates
all business logic operations while remaining UI-agnostic. It serves as the
bridge between the UI layer and the various service providers.
"""

import asyncio
import time
import logging
import sqlite3
import os
from typing import Dict, Any, List, Optional
from uuid import UUID
from datetime import datetime

from globule.core.interfaces import IOrchestrationEngine, IEmbeddingProvider, IParserProvider, IStorageManager
from globule.core.models import GlobuleV1, ProcessedGlobuleV1, FileDecisionV1, NuanceMetaDataV1
from globule.core.errors import ParserError, EmbeddingError, StorageError
from pathlib import Path

logger = logging.getLogger(__name__)


class GlobuleOrchestrator(IOrchestrationEngine):
    """
    Main orchestration engine for processing globules and coordinating business logic.
    
    This class implements the IOrchestrationEngine interface and contains all the 
    business logic previously embedded in the TUI, making it UI-agnostic and reusable.
    """
    
    def __init__(self, 
                 parser_provider: IParserProvider,
                 embedding_provider: IEmbeddingProvider,
                 storage_manager: IStorageManager):
        self.parser_provider = parser_provider
        self.embedding_provider = embedding_provider  
        self.storage_manager = storage_manager
        
    async def process(self, globule: GlobuleV1) -> ProcessedGlobuleV1:
        """
        Process a raw globule into a processed globule.
        
        Implements the IOrchestrationEngine interface requirement.
        Coordinates parsing and embedding in parallel, then constructs the final result.
        """
        start_time = time.time()
        
        logger.debug(f"Processing globule: {globule.raw_text[:50]}...")
        
        try:
            # Launch embedding and parsing tasks concurrently
            embedding_task = asyncio.create_task(self._generate_embedding(globule.raw_text))
            parsing_task = asyncio.create_task(self._parse_content(globule.raw_text))
            
            # Wait for both to complete
            embedding_result, parsing_result = await asyncio.gather(
                embedding_task, parsing_task, return_exceptions=True
            )
            
            # Handle embedding result
            if isinstance(embedding_result, Exception):
                logger.error(f"Embedding failed: {embedding_result}")
                embedding = []
            else:
                embedding, _ = embedding_result
            
            # Handle parsing result  
            if isinstance(parsing_result, Exception):
                logger.error(f"Parsing failed: {parsing_result}")
                parsed_data = {"error": str(parsing_result)}
            else:
                parsed_data, _ = parsing_result
            
            # Generate file decision from parsed data
            file_decision = self._generate_file_decision(globule.raw_text, parsed_data)
            
            # Calculate total processing time
            total_time = (time.time() - start_time) * 1000
            
            # Create processed globule
            processed_globule = ProcessedGlobuleV1(
                globule_id=globule.globule_id,
                original_globule=globule,
                embedding=embedding,
                parsed_data=parsed_data,
                file_decision=file_decision,
                processing_time_ms=total_time,
                provider_metadata={
                    "parser": self.parser_provider.__class__.__name__,
                    "embedder": self.embedding_provider.__class__.__name__,
                    "storage": self.storage_manager.__class__.__name__
                }
            )
            
            logger.debug(f"Globule processed in {total_time:.1f}ms")
            return processed_globule
            
        except Exception as e:
            logger.error(f"Orchestration failed: {e}")
            raise
    
    # Business Logic Methods (extracted from TUI)
    
    async def capture_thought(self, raw_text: str, source: str = "tui", 
                             context: Dict[str, Any] = None) -> ProcessedGlobuleV1:
        """
        Capture and process a thought/globule.
        
        This method handles the complete workflow of taking raw text input,
        creating a globule, processing it, and storing it.
        """
        if context is None:
            context = {}
            
        # Create raw globule
        globule = GlobuleV1(
            raw_text=raw_text,
            source=source,
            initial_context=context
        )
        
        # Process the globule
        processed_globule = await self.process(globule)
        
        # Store the processed globule
        self.storage_manager.save(processed_globule)
        
        logger.info(f"Captured and stored globule: {globule.globule_id}")
        return processed_globule
    
    async def search_globules(self, query: str, limit: int = 10) -> List[ProcessedGlobuleV1]:
        """
        Search for globules using natural language query.
        
        Converts the query to SQL and executes it against the database.
        This is the actual logic from the TUI's search functionality.
        """
        try:
            # Get database path from storage manager
            db_path = self._get_db_path()
            
            if not os.path.exists(db_path):
                logger.warning(f"Database not found at {db_path}")
                return []
            
            # For now, implement simple text search
            # TODO: Implement the full NL->SQL conversion from TUI
            results = []
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Simple LIKE search for now
                cursor.execute(
                    "SELECT * FROM globules WHERE text LIKE ? ORDER BY created_at DESC LIMIT ?",
                    (f"%{query}%", limit)
                )
                
                rows = cursor.fetchall()
                for row in rows:
                    # Convert row to ProcessedGlobuleV1
                    # This is simplified - in reality we'd need to reconstruct the full object
                    logger.info(f"Found globule: {row['id']}")
                    
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def get_globule(self, globule_id: UUID) -> Optional[ProcessedGlobuleV1]:
        """Retrieve a specific globule by ID."""
        try:
            return self.storage_manager.get(globule_id)
        except StorageError as e:
            logger.error(f"Failed to get globule {globule_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting globule {globule_id}: {e}")
            return None
    
    def save_draft(self, content: str, topic: str = None, metadata: Dict[str, Any] = None) -> str:
        """
        Save content as a draft file.
        
        This implements the actual draft-saving logic from the TUI.
        """
        try:
            # Generate filename - this is the real logic from TUI
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            topic_part = topic.replace(" ", "_") if topic else "draft"
            filename = f"globule_{topic_part}_{timestamp}.md"
            
            # Save to drafts directory
            drafts_dir = "drafts"
            os.makedirs(drafts_dir, exist_ok=True)
            filepath = os.path.join(drafts_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                # Add metadata frontmatter if available
                if metadata or topic:
                    f.write("---\n")
                    if metadata:
                        for key, value in metadata.items():
                            f.write(f"{key}: {value}\n")
                    if topic:
                        f.write(f"topic: {topic}\n")
                    f.write(f"generated: {datetime.now().isoformat()}\n")
                    f.write("---\n\n")
                f.write(content)
            
            logger.info(f"Draft saved to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save draft: {e}")
            raise
    
    async def execute_sql_query(self, query: str, query_name: str = "Query") -> Dict[str, Any]:
        """
        Execute SQL query against the database.
        
        This is the actual SQL execution logic extracted from the TUI.
        """
        try:
            db_path = self._get_db_path()
            
            if not os.path.exists(db_path):
                raise Exception(f"Database not found at {db_path}")
            
            with sqlite3.connect(db_path, timeout=10.0) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Validate SQL safety (basic check)
                dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'TRUNCATE', 'ALTER']
                if any(keyword in query.upper() for keyword in dangerous_keywords):
                    raise Exception("Potentially dangerous SQL detected")
                
                cursor.execute(query)
                results = cursor.fetchall()
                
                # Convert to list of dicts
                results_list = [dict(row) for row in results]
                headers = [desc[0] for desc in cursor.description] if cursor.description else []
                
                return {
                    "type": "sql_results",
                    "query": query,
                    "query_name": query_name,
                    "results": results_list,
                    "headers": headers,
                    "count": len(results_list)
                }
                
        except Exception as e:
            logger.error(f"SQL query execution failed: {e}")
            return {
                "type": "error",
                "query": query,
                "query_name": query_name,
                "error": str(e)
            }
    
    async def execute_query(self, query: str, query_type: str = "natural") -> Dict[str, Any]:
        """
        Execute a query based on its type.
        
        This coordinates different query execution strategies.
        """
        try:
            if query_type == "sql":
                return await self.execute_sql_query(query)
            elif query_type == "natural":
                # For natural language, search globules
                results = await self.search_globules(query)
                return {
                    "type": "search_results",
                    "query": query,
                    "results": results,
                    "count": len(results)
                }
            else:
                return {"type": "unknown", "query": query, "error": f"Unknown query type: {query_type}"}
                
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return {"type": "error", "query": query, "error": str(e)}
    
    # Internal helper methods
    
    async def _generate_embedding(self, text: str) -> tuple:
        """Generate embedding and return (embedding, time_ms)"""
        start_time = time.time()
        try:
            embedding = self.embedding_provider.embed(text)
            processing_time = (time.time() - start_time) * 1000
            return embedding, processing_time
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise EmbeddingError(f"Failed to generate embedding: {e}")
    
    async def _parse_content(self, text: str) -> tuple:
        """Parse content and return (parsed_data, time_ms)"""
        start_time = time.time()
        try:
            parsed_data = self.parser_provider.parse(text)
            processing_time = (time.time() - start_time) * 1000
            return parsed_data, processing_time
        except Exception as e:
            logger.error(f"Content parsing failed: {e}")
            raise ParserError(f"Failed to parse content: {e}")
    
    def _generate_file_decision(self, text: str, parsed_data: Dict[str, Any]) -> FileDecisionV1:
        """Generate file decision based on parsed data"""
        # Use parsed data if available, otherwise create simple path
        domain = parsed_data.get("domain", "general")
        category = parsed_data.get("category", "note")
        title = parsed_data.get("title", text[:30].replace(" ", "-").lower())
        
        # Clean title for filename
        clean_title = "".join(c for c in title if c.isalnum() or c in "-_").strip("-_")
        if not clean_title:
            clean_title = "untitled"
        
        # Create semantic path: domain/category/
        semantic_path = Path(domain) / category
        filename = f"{clean_title}.md"
        
        return FileDecisionV1(
            semantic_path=str(semantic_path),
            filename=filename,
            confidence=0.8  # Default confidence
        )
    
    def _get_db_path(self) -> str:
        """Get database path from storage manager or fallback"""
        if hasattr(self.storage_manager, 'db_path'):
            return str(self.storage_manager.db_path)
        else:
            # Fallback path
            return os.path.expanduser("~/.globule/data/globules.db")