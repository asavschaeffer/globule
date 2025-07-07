# Build Globule MVP - A Semantic Thought Processor

## Project Overview
Build a CLI-based thought capture and retrieval system that uses AI to understand and organize information without manual filing. Users input text naturally (e.g., "Meeting with Sarah about budget cuts"), the system processes it with embeddings and LLM parsing in parallel, stores it intelligently in SQLite, and retrieves it semantically (e.g., "Show me budget-related thoughts").

## Core Architecture Requirements

### 1. Project Structure
Create a Python project with:
- **CLI**: Use `Click` for simplicity (Typer is fine too, but Click is more beginner-friendly)
- **Async Processing**: Use `asyncio` for non-blocking operations
- **Storage**: SQLite with JSON support for flexibility
- **Modularity**: Separate files for each component (e.g., `input_handler.py`, `storage.py`)
- **Type Hints**: Use Python 3.10+ with type annotations everywhere
- **Dependency Management**: Use Poetry for a clean setup

### 2. Database Schema
```sql
CREATE TABLE globules (
    id TEXT PRIMARY KEY,           -- UUID or similar unique string
    content TEXT NOT NULL,         -- Raw user input
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    embedding BLOB,                -- Serialized numpy array for semantic search
    parsed_data JSON,              -- Structured JSON from LLM parsing
    entities JSON,                 -- Extracted entities (e.g., {"Sarah": "person"})
    domain TEXT,                   -- Category like "work" or "personal"
    metadata JSON                  -- Extra info (e.g., source, tags)
);

CREATE INDEX idx_created_at ON globules(created_at);  -- For temporal queries
CREATE INDEX idx_domain ON globules(domain);          -- For domain filtering
```

### 3. Core Components to Implement

#### Input Handler (`input_handler.py`)
- **Purpose**: Capture user input from the CLI
- **Features**:
  - Command: `globule add "text here"`
  - Cache input immediately (e.g., in memory or SQLite) for responsiveness
  - Print "✓ Captured! Processing..." and return control to the user
  - Trigger async processing in the background

#### Embedding Engine (`embedding_engine.py`)
- **Purpose**: Convert text to embeddings for semantic search
- **Features**:
  - Use `sentence-transformers` with `'BAAI/bge-small-en'` (lightweight and effective)
  - Cache embeddings in memory to avoid recomputing (e.g., LRU cache)
  - Store as numpy arrays serialized with `msgpack` in SQLite BLOBs
  - Abstract interface (e.g., `Embedder` class) for future model swaps

#### Parser Engine (`parser_engine.py`)
- **Purpose**: Extract meaning from text using an LLM
- **Features**:
  - Use `llama.cpp` with `llama3.2:3b` (local) or Anthropic/OpenAI API (cloud)
  - Parse text to extract:
    - Entities (e.g., "Sarah" → person)
    - Category (e.g., "meeting")
    - Topics (e.g., "budget")
    - Temporal references (e.g., "today")
  - Return JSON like: `{"entities": [{"name": "Sarah", "type": "person"}], "category": "meeting", "topics": ["budget"], "temporal": "today"}`
  - Abstract interface for LLM provider swaps

#### Storage Manager (`storage.py`)
- **Purpose**: Handle data persistence
- **Features**:
  - Abstract interface (e.g., `Storage` class) with SQLite implementation
  - Support JSON fields and BLOBs
  - Methods: `store_globule()`, `retrieve_by_id()`, `search_semantic()`, `search_temporal()`
  - Use connection pooling with `aiosqlite` for async access
  - Ensure atomic writes to prevent data loss

#### Query Engine (`query_engine.py`)
- **Purpose**: Retrieve thoughts intelligently
- **Features**:
  - Parse natural language queries (e.g., "budget complaints") using LLM or simple rules
  - Semantic search with cosine similarity on embeddings
  - Temporal search (e.g., "today", "this week")
  - Combined search (e.g., semantic + domain)
  - Return ranked results (e.g., top 5 matches)

#### Synthesis Engine (`synthesis.py`)
- **Purpose**: Summarize thoughts into reports
- **Features**:
  - Generate daily summaries from globules
  - Group by category (e.g., "meetings", "ideas")
  - Use LLM to create narrative text (e.g., "Today, you met Sarah about budget cuts...")
  - Output in markdown or plain text
  - Support customizable report templates

### 4. Processing Pipeline
```python
import asyncio

async def process_input(text: str) -> None:
    # Step 1: Store raw input immediately
    globule_id = generate_id()  # e.g., UUID
    await cache_input(globule_id, text)  # Quick save
    print("✓ Captured! Processing in background...")

    # Step 2: Parallel processing
    embedding_task = asyncio.create_task(embed_text(text))
    parsing_task = asyncio.create_task(parse_text(text))

    # Step 3: Gather results
    embedding, parsed_data = await asyncio.gather(embedding_task, parsing_task)

    # Step 4: Store everything
    entities = parsed_data.get("entities", [])
    domain = detect_domain(parsed_data)  # Simple heuristic or LLM-based

    await store_globule(
        id=globule_id,
        content=text,
        embedding=embedding,
        parsed_data=parsed_data,
        entities=entities,
        domain=domain
    )
```

### 5. CLI Commands
```bash
# Add a thought
globule add "Mr Jones complained about scratch on bumper"

# Search thoughts
globule search "damage complaints"  # Semantic search
globule today                       # Today’s entries
globule report                      # Daily summary

# Utilities
globule stats                       # DB stats (e.g., entry count)
globule config                      # Edit settings
```

### 6. Configuration System
- **File**: Use `config.yaml` (YAML is beginner-friendly)
- **Settings**:
  - `llm_provider`: "local" or "cloud"
  - `embedding_model`: e.g., "BAAI/bge-small-en"
  - `db_path`: Path to SQLite file
  - `report_template`: e.g., "daily.md"
  - `api_keys`: For cloud LLMs (stored securely)
- **Example**:
```yaml
llm_provider: local
embedding_model: BAAI/bge-small-en
db_path: ./globule.db
report_template: daily.md
```

### 7. Key Design Patterns
- **Async Everything**: Keep the CLI snappy with `asyncio`
- **Abstract Interfaces**: Make storage, LLM, and embeddings swappable
- **Plugin Ready**: Structure code for future extensions
- **Fail Gracefully**: Log errors (use `logging`) and never lose input
- **Feedback**: Show users progress (e.g., "Processing...")

### 8. Dependencies
```toml
[tool.poetry.dependencies]
python = "^3.10"
click = "^8.0"              # CLI framework
sentence-transformers = "^2.0"  # Embeddings
numpy = "^1.24"             # Array handling
aiosqlite = "^0.19"         # Async SQLite
pydantic = "^2.0"           # Data validation
rich = "^13.0"              # Pretty CLI output
python-dotenv = "^1.0"      # Env vars for API keys
msgpack = "^1.0"            # Embedding serialization
pyyaml = "^6.0"             # Config parsing

[tool.poetry.dependencies.optional]
llama-cpp-python = "^0.2"   # Local LLM
anthropic = "^0.3"          # Claude API
openai = "^1.0"             # OpenAI API
```

### 9. Testing Requirements
- **Unit Tests**: Test each component (e.g., embedding generation)
- **Integration Tests**: Test the full pipeline
- **Mocks**: Fake LLM and embedding responses
- **Test Data**: Generate sample inputs (e.g., "Meeting with Bob")
- **Benchmarks**: Check performance targets

### 10. Example Usage Flow
```bash
$ globule add "Meeting with Sarah about Q3 budget concerns. Need to cut cloud costs by 20%"
✓ Captured! Processing in background...

$ globule add "Sarah suggested reserved instances for savings"
✓ Captured! Processing in background...

$ globule search "budget sarah"
Found 2 entries:
1. [2024-01-15 10:30] Meeting with Sarah about Q3 budget concerns...
2. [2024-01-15 10:45] Sarah suggested reserved instances...

$ globule report
# Daily Summary - 2024-01-15
## Meetings
- Budget talk with Sarah: Cut cloud costs by 20%
## Ideas
- Use reserved instances (Sarah’s suggestion)
```

### 11. Performance Targets
- Input capture: <50ms
- Embedding: <200ms
- LLM parsing: <1s (local), <3s (cloud)
- Semantic search: <100ms (10k entries)
- Report generation: <5s

### 12. Future-Proofing
- **Storage**: Support future graph relationships (e.g., linking globules)
- **Embeddings**: Use standard formats (msgpack) for migration
- **AI Abstraction**: Easy model upgrades
- **Plugins**: Leave hooks for extensions
- **Extras**: Plan for batch inputs, data export, and caching

## Start Here
1. Run `poetry init` and set up the project
2. Build the CLI with `globule add`
3. Add SQLite storage with `storage.py`
4. Implement embeddings in `embedding_engine.py`
5. Add LLM parsing in `parser_engine.py`
6. Build search in `query_engine.py`
7. Create summaries in `synthesis.py`
8. Write basic tests

Focus on getting each part working solo, then wire them together. The MVP should capture thoughts, understand them, and retrieve them smartly—proving AI can grok meaning, not just keywords.