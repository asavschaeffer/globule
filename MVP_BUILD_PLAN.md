# Globule MVP (Ollie) - Technical Build Plan

## 1. Objective

This document outlines the concrete technical plan for building the Globule MVP, codenamed "Ollie". The goal is a functional, performant, and extensible local-first application that proves the core concepts of the "dual-intelligence" architecture.

This is a build plan, not a vision document. We will implement the following core user story:
> A user can capture fragmented thoughts via a CLI, have them automatically organized and enriched by the system, and then use an interactive TUI to synthesize those thoughts into a structured document.

## 2. Core Architectural Principles

- **Local-First**: All data and processing reside on the user's machine. No cloud dependencies for core functionality.
- **Separation of Concerns**: Each component has a single, well-defined responsibility.
- **Interface-Driven**: Components communicate through abstract interfaces, not concrete implementations.
- **Storage Agnostic**: The core logic is decoupled from the database implementation (SQLite for the MVP).
- **Asynchronous by Default**: The system will be built on `asyncio` to ensure a non-blocking UI and efficient I/O.

## 3. Technology Stack

- **Language**: Python 3.11+
- **Core AI**: Ollama (local inference)
  - **Embedding**: `mxbai-embed-large`
  - **Parsing**: `llama3.2:3b`
- **Database**: SQLite (via `aiosqlite`)
- **TUI Framework**: `textual`
- **Configuration**: `PyYAML`
- **Data Validation**: `Pydantic`

## 4. Core Data Models

These Pydantic models will be the primary data contracts between components, defined in `src/globule/core/models.py`.

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from uuid import UUID, uuid4
from datetime import datetime

class Globule(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    embedding: Optional[List[float]] = None
    parsed_data: Optional[Dict[str, Any]] = None
    file_path: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ProcessedGlobule(Globule):
    confidence_scores: Dict[str, float]
    processing_time: float
    schema_used: Optional[str]

class EnrichedInput(BaseModel):
    original_text: str
    enriched_text: str
    detected_schema: Optional[str]
    additional_context: Dict
```

## 5. Implementation Phases

Development will proceed in four distinct, sequential phases.

### Phase 1: The Foundation (Weeks 1-2)

Goal: Establish the non-AI backbone of the application.

1.  **Configuration System (`src/globule/config/manager.py`)**
    - Implement the three-tier configuration cascade (system, user, context).
    - Load from YAML files.
    - Provide a simple, global access point for configuration values.

2.  **Schema Engine (`src/globule/schemas/manager.py`)**
    - Implement loading and validation of user-defined schemas from YAML.
    - Define the structure for triggers, actions, and templates.
    - Create default schemas (e.g., `default.json`, `creative.json`).

3.  **Storage Manager (`src/globule/storage/sqlite_manager.py`)**
    - Define the abstract `StorageManager` interface in `src/globule/core/interfaces.py`.
    - Implement the `SQLiteStorageManager` with `aiosqlite`.
    - Create the `globules` table as specified in the architecture.
    - Implement core CRUD operations and temporal search.

### Phase 2: The Intelligence (Weeks 3-4)

Goal: Implement the core AI processing pipeline.

1.  **Embedding Service (`src/globule/services/embedding/ollama_provider.py`)**
    - Create an interface for embedding providers.
    - Implement an `OllamaEmbeddingProvider` to connect to a local Ollama instance and generate vectors with `mxbai-embed-large`.
    - Include batching capabilities.

2.  **Parsing Service (`src/globule/services/parsing/ollama_parser.py`)**
    - Create an interface for parsing providers.
    - Implement an `OllamaParser` to extract structured JSON from text using `llama3.2:3b`.
    - Implement context-aware prompt construction.

3.  **Orchestration Engine (`src/globule/orchestration/engine.py`)**
    - Implement the `OrchestrationEngine` to coordinate the embedding and parsing services.
    - Manage the dual-track processing flow.
    - Implement logic for handling disagreements and generating semantic file paths.

### Phase 3: The User Experience (Weeks 5-6)

Goal: Build the user-facing interfaces.

1.  **CLI / Input Module (`src/globule/interfaces/cli/main.py`)**
    - Build the main CLI application using a library like `typer` or `click`.
    - Implement the `globule add` command.
    - Integrate the "Adaptive Input" logic for schema detection and conversational confirmation.

2.  **TUI / Synthesis Engine (`src/globule/tui/app.py`)**
    - Build the two-pane TUI application using `textual`.
    - Implement the `PalettePane` for displaying and navigating clustered globules.
    - Implement the `CanvasPane` for drafting.
    - Integrate the "Build Mode" (Enter) and "Explore Mode" (Tab) logic.
    - Hook up basic AI co-pilot actions (summarize, expand) to the Orchestration Engine.

### Phase 4: Integration & Polish

Goal: Solidify the MVP into a usable tool.

1.  **End-to-End Testing**: Write integration tests that cover the full user journey.
2.  **Performance Tuning**: Benchmark all critical paths and optimize bottlenecks.
3.  **Dependency Management**: Finalize `pyproject.toml` and lock dependencies.
4.  **Documentation**: Write a `README.md` with clear installation and usage instructions.

## 6. MVP Exit Criteria

The MVP is complete when:
- A user can successfully run `globule add "..."` and the thought is processed and stored correctly.
- A user can run `globule draft` and see their recent thoughts in the TUI.
- A user can select at least 10 thoughts from the Palette and synthesize them into a coherent draft on the Canvas.
- The "Explore Mode" successfully retrieves semantically related thoughts from the database.
- The entire process is completed locally without reliance on external cloud services.
- Core operations (add, draft, search) complete within specified performance targets (<500ms).
