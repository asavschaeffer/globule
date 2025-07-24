# Globule MVP - Walking Skeleton

This is the Phase 1 implementation of the Globule MVP. It provides the basic end-to-end functionality to validate our architecture.

## What Works (Phase 1)

✅ **Core Architecture**
- Data models (ProcessedGlobule, EnrichedInput)
- Abstract interfaces (EmbeddingProvider, ParsingProvider, StorageManager)  
- SQLite storage with basic schema
- Parallel orchestration engine

✅ **Basic CLI Commands**
- `globule add "your thought"` - Capture thoughts
- `globule draft` - Launch TUI to view thoughts

✅ **Simple TUI**
- Display recent globules in a scrollable list
- Basic Textual interface

✅ **Mock AI Integration**
- Mock parser for testing without LLM
- Ollama embedding provider (requires Ollama)

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

2. **Start Ollama (optional for Phase 1):**
   ```bash
   docker-compose up -d
   # Wait for startup, then pull models:
   docker exec globule-ollama ollama pull mxbai-embed-large
   ```

3. **Try it out:**
   ```bash
   # Add some thoughts
   globule add "I should write more consistently"
   globule add "The concept of flow state is fascinating"
   globule add "Need to research semantic embeddings"
   
   # View in TUI
   globule draft
   ```

## Phase 1 Validation

The walking skeleton proves these architectural decisions work:

1. **Dual-Intelligence Abstraction** - Embedding and parsing services work in parallel
2. **Storage Manager Abstraction** - Clean interface for data persistence
3. **Core Data Model** - ProcessedGlobule handles all data flow
4. **Async TUI Foundation** - Textual app loads and displays data

## What's Next (Phase 2)

- Real LLM parsing with Ollama
- Vector search with sqlite-vec
- Semantic clustering of globules
- Improved TUI with two-pane layout

## Testing

Run tests to verify the walking skeleton:

```bash
pytest tests/test_walking_skeleton.py -v
```

## Configuration

Config is stored in `~/.globule/config.yaml`. Default settings:
- Storage path: `~/.globule/data/`
- Ollama URL: `http://localhost:11434`
- Models: `mxbai-embed-large`, `llama3.2:3b`

## Architecture Notes

This implementation follows the MVP kickoff memo principles:
- Single ParallelStrategy (no complex orchestration)
- Hard-coded Pydantic models (no schema engine)
- Simple config.yaml (no advanced configuration)
- Focus on core user experience

The foundation is solid and ready for Phase 2 enhancements!