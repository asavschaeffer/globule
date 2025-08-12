"""
Core API: Unified interface for all frontend implementations.

This module provides a clean, stable API that can be used by any frontend
(CLI, TUI, Web, etc.). It abstracts away implementation details and provides
consistent function signatures and return values.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from globule.core.interfaces import StorageManager
from globule.orchestration.engine import search_globules_nlp
from globule.core.draft_manager import DraftManager

logger = logging.getLogger(__name__)


class GlobuleAPI:
    """
    Core API for all Globule operations.
    
    This class provides a unified interface that can be used by any frontend
    implementation. All methods return consistent data structures and handle
    errors gracefully.
    """
    
    def __init__(self, storage_manager: StorageManager):
        """
        Initialize the API with required services.
        
        Args:
            storage_manager: Storage manager for database operations
        """
        self.storage_manager = storage_manager
        self.draft_manager = DraftManager()
    
    async def search(self, query: str, output_format: str = "markdown") -> Dict[str, Any]:
        """
        Search globules using natural language query.
        
        Args:
            query: Natural language search query
            output_format: Output format ("markdown", "json", "raw")
            
        Returns:
            Dict with keys: success, data, message, metadata
        """
        try:
            logger.info(f"Starting search for: {query}")
            start_time = datetime.now()
            
            # Use the existing search function
            result = await search_globules_nlp(query, self.storage_manager)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Parse result for different formats
            if output_format == "json":
                # TODO: Parse markdown result into structured JSON
                data = {"markdown": result, "query": query}
            else:
                data = result
            
            return {
                "success": True,
                "data": data,
                "message": f"Search completed successfully",
                "metadata": {
                    "query": query,
                    "duration_seconds": duration,
                    "timestamp": end_time.isoformat(),
                    "format": output_format
                }
            }
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {
                "success": False,
                "data": None,
                "message": f"Search failed: {str(e)}",
                "metadata": {"query": query, "error": str(e)}
            }
    
    async def add_to_draft(self, content: str, globule_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Add content to the current draft.
        
        Args:
            content: Content to add (can be globule content or custom text)
            globule_id: Optional globule ID for metadata
            
        Returns:
            Dict with keys: success, data, message, metadata
        """
        try:
            logger.info(f"Adding content to draft (ID: {globule_id or 'N/A'})")
            
            # Use DraftManager to add content
            success = self.draft_manager.add_to_draft(content, globule_id)
            
            if success:
                stats = self.draft_manager.get_draft_stats()
                return {
                    "success": True,
                    "data": {"draft_path": self.draft_manager.draft_path},
                    "message": "Content added to draft successfully",
                    "metadata": {
                        "globule_id": globule_id,
                        "draft_stats": stats,
                        "timestamp": datetime.now().isoformat()
                    }
                }
            else:
                return {
                    "success": False,
                    "data": None,
                    "message": "Failed to add content to draft",
                    "metadata": {"globule_id": globule_id}
                }
                
        except Exception as e:
            logger.error(f"Add to draft failed: {e}")
            return {
                "success": False,
                "data": None,
                "message": f"Add to draft failed: {str(e)}",
                "metadata": {"globule_id": globule_id, "error": str(e)}
            }
    
    async def export_draft(self, output_path: str, include_metadata: bool = True) -> Dict[str, Any]:
        """
        Export current draft to a file.
        
        Args:
            output_path: Path where to save the exported draft
            include_metadata: Whether to include metadata in export
            
        Returns:
            Dict with keys: success, data, message, metadata
        """
        try:
            logger.info(f"Exporting draft to: {output_path}")
            
            success = self.draft_manager.export_draft(output_path, include_metadata)
            
            if success:
                stats = self.draft_manager.get_draft_stats()
                return {
                    "success": True,
                    "data": {"export_path": output_path},
                    "message": f"Draft exported successfully to {output_path}",
                    "metadata": {
                        "export_path": output_path,
                        "include_metadata": include_metadata,
                        "draft_stats": stats,
                        "timestamp": datetime.now().isoformat()
                    }
                }
            else:
                return {
                    "success": False,
                    "data": None,
                    "message": f"Failed to export draft to {output_path}",
                    "metadata": {"export_path": output_path}
                }
                
        except Exception as e:
            logger.error(f"Export draft failed: {e}")
            return {
                "success": False,
                "data": None,
                "message": f"Export draft failed: {str(e)}",
                "metadata": {"export_path": output_path, "error": str(e)}
            }
    
    async def get_draft_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current draft.
        
        Returns:
            Dict with keys: success, data, message, metadata
        """
        try:
            stats = self.draft_manager.get_draft_stats()
            
            return {
                "success": True,
                "data": stats,
                "message": "Draft stats retrieved successfully",
                "metadata": {"timestamp": datetime.now().isoformat()}
            }
            
        except Exception as e:
            logger.error(f"Get draft stats failed: {e}")
            return {
                "success": False,
                "data": None,
                "message": f"Get draft stats failed: {str(e)}",
                "metadata": {"error": str(e)}
            }
    
    async def fetch_globule_by_id(self, globule_id: str) -> Dict[str, Any]:
        """
        Fetch a specific globule by ID.
        
        Args:
            globule_id: ID of the globule to fetch
            
        Returns:
            Dict with keys: success, data, message, metadata
        """
        try:
            logger.info(f"Fetching globule: {globule_id}")
            
            # Import here to avoid circular imports
            from globule.orchestration.engine import fetch_globule_content
            content = await fetch_globule_content(globule_id, self.storage_manager)
            
            return {
                "success": True,
                "data": {"content": content, "id": globule_id},
                "message": f"Globule {globule_id} fetched successfully",
                "metadata": {
                    "globule_id": globule_id,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Fetch globule failed: {e}")
            return {
                "success": False,
                "data": None,
                "message": f"Fetch globule failed: {str(e)}",
                "metadata": {"globule_id": globule_id, "error": str(e)}
            }


# Convenience functions for simple use cases
async def search_globules(query: str, storage_manager: StorageManager) -> str:
    """Convenience function for simple search (returns raw markdown)."""
    api = GlobuleAPI(storage_manager)
    result = await api.search(query)
    return result["data"] if result["success"] else f"Error: {result['message']}"


async def add_content_to_draft(content: str, globule_id: Optional[str] = None) -> bool:
    """Convenience function for simple draft addition."""
    # Create a dummy storage manager for draft-only operations
    api = GlobuleAPI(None)  # DraftManager doesn't need storage_manager
    result = await api.add_to_draft(content, globule_id)
    return result["success"]