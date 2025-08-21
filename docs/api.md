# Globule API Reference

This document provides reference documentation for the `GlobuleAPI`, the primary interface for interacting with the Globule application.

## Overview

The `GlobuleAPI` class, defined in `src/globule/core/api.py`, provides a stable, high-level interface for all core application features. All frontends (CLI, TUI, etc.) use this API to ensure consistent behavior.

## API Methods

### `async def add_thought(self, text: str, source: str = "api") -> ProcessedGlobuleV1`

Adds and processes a new thought.

-   **Parameters:**
    -   `text` (str): The raw text of the thought.
    -   `source` (str): Optional. The source of the input (e.g., 'tui', 'cli'). Defaults to 'api'.
-   **Returns:**
    -   A `ProcessedGlobuleV1` object containing the enriched and saved thought.

### `async def search_semantic(self, query: str, limit: int = 10) -> List[ProcessedGlobuleV1]`

Performs a semantic vector search for globules.

-   **Parameters:**
    -   `query` (str): The natural language query.
    -   `limit` (int): Optional. The maximum number of results to return. Defaults to 10.
-   **Returns:**
    -   A list of `ProcessedGlobuleV1` objects that are semantically similar to the query.

### `async def search_structured(self, query: StructuredQuery) -> List[ProcessedGlobuleV1]`

Performs a structured search based on metadata filters.

-   **Parameters:**
    -   `query` (StructuredQuery): A `StructuredQuery` Pydantic model containing filters for domain, category, etc.
-   **Returns:**
    -   A list of `ProcessedGlobuleV1` objects matching the structured query.

### `async def get_globule_by_id(self, globule_id: UUID) -> Optional[ProcessedGlobuleV1]`

Retrieves a single globule by its unique ID.

-   **Parameters:**
    -   `globule_id` (UUID): The UUID of the globule.
-   **Returns:**
    -   The `ProcessedGlobuleV1` object, or `None` if not found.

### `async def get_all_globules(self, limit: int = 100) -> List[ProcessedGlobuleV1]`

Retrieves all globules from storage, up to a limit.

-   **Parameters:**
    -   `limit` (int): Optional. The maximum number of globules to retrieve. Defaults to 100.
-   **Returns:**
    -   A list of all `ProcessedGlobuleV1` objects.

### `async def get_clusters(self) -> ClusteringAnalysis`

Analyzes all globules and groups them into semantic clusters.

-   **Parameters:** None.
-   **Returns:**
    -   A `ClusteringAnalysis` object containing the discovered clusters and analysis metadata.

### `async def reconcile_files(self) -> Dict[str, Any]`

Reconciles the file system with the database.

-   **Parameters:** None.
-   **Returns:**
    -   A dictionary containing statistics about the reconciliation process.

### `async def export_draft(self, draft_content: str, file_path: str) -> bool`

Exports draft content to a file.

-   **Parameters:**
    -   `draft_content` (str): The content of the draft to export.
    -   `file_path` (str): The path to save the file to.
-   **Returns:**
    -   `True` if the export was successful, `False` otherwise.
