# Phase 2: Core Intelligence Guide

## Overview

Phase 2 transforms Globule from a "walking skeleton" into a genuinely intelligent system that understands your thoughts and helps organize them meaningfully. This phase introduces **real AI integration** and **semantic understanding**.

## New Capabilities

### 🧠 Intelligent Content Analysis

The new `OllamaParser` replaces the mock parser with genuine AI-powered text understanding:

```bash
# Your thoughts are now intelligently analyzed
globule add "The concept of 'progressive overload' in fitness could apply to creative stamina."

# Results in:
# ✓ Domain: creative
# ✓ Category: idea  
# ✓ Keywords: progressive, overload, creative, stamina
# ✓ Sentiment: positive
# ✓ Confidence: 0.85
```

### 📊 Smart Classification

Content is automatically classified across multiple dimensions:

- **Domain**: creative, technical, personal, academic, business, philosophy
- **Category**: note, idea, question, task, reference, draft, quote, observation
- **Content Type**: prose, list, code, data, dialogue, poetry, instructions
- **Sentiment**: positive, negative, neutral, mixed

### 🔄 Graceful Fallback

When Ollama is unavailable, the system uses enhanced heuristic analysis:

- Keyword extraction using frequency analysis
- Pattern-based entity recognition
- Rule-based sentiment analysis
- Structural content type detection

## Glass Engine Integration

The Glass Engine now showcases Phase 2 intelligence:

```bash
# Run the enhanced demo
globule tutorial --mode demo

# See intelligent analysis in action:
# - Real-time content classification
# - Keyword and entity extraction
# - Sentiment analysis
# - Confidence scoring
```

## Architecture

### OllamaParser Structure

```python
class OllamaParser(ParsingProvider):
    """Production Ollama parser with intelligent analysis."""
    
    async def parse(self, text: str) -> Dict[str, Any]:
        """
        Parse text using:
        1. LLM-based analysis (when available)
        2. Enhanced fallback parsing (when offline)
        3. Empty input handling
        """
```

### Integration Points

- **CLI Commands**: `globule add` now uses intelligent parsing
- **Glass Engine**: All three modes showcase intelligence
- **Orchestration**: Parallel processing with smart analysis
- **Storage**: Rich metadata for future search capabilities

## Configuration

Add parsing preferences to your `~/.globule/config.yaml`:

```yaml
# AI model settings
default_parsing_model: "llama3.2:3b"
ollama_base_url: "http://localhost:11434"
ollama_timeout: 30

# Performance settings
max_concurrent_requests: 5
```

## Development

### Testing

Comprehensive test suite covers:

- LLM integration scenarios
- Fallback parsing accuracy  
- Concurrent processing
- Error handling
- Content type classification

```bash
# Run Phase 2 tests
pytest tests/test_ollama_parser.py -v

# Test specific scenarios
pytest tests/test_ollama_parser.py::TestOllamaParser::test_enhanced_fallback_creative -v
```

### Extending Analysis

To add new content analysis features:

1. **Enhance LLM Prompt**: Update `parsing_prompt` template
2. **Improve Fallback**: Add heuristics to `_enhanced_fallback_parse`
3. **Add Tests**: Cover new scenarios in test suite
4. **Update Glass Engine**: Showcase new capabilities

## Next Steps

Phase 2 establishes the intelligence foundation for:

- **Vector Search**: Semantic similarity matching
- **Smart Clustering**: Automatic theme detection  
- **Enhanced TUI**: Intelligent content organization
- **AI-Assisted Writing**: Context-aware suggestions

## Troubleshooting

### Ollama Connection Issues

```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# Verify model availability
ollama list | grep llama3.2
```

### Parsing Quality

- **Low Confidence**: Content may be ambiguous or novel
- **Wrong Classification**: Consider updating training examples
- **Fallback Mode**: Ollama service unavailable, using heuristics

### Performance

- **Slow Parsing**: Check Ollama model size and system resources
- **High Memory**: Consider lighter models for resource-constrained environments

## User Stories Satisfied

✅ **"My thoughts are intelligently categorized"**
- Automatic domain and category classification
- Keyword and entity extraction
- Sentiment analysis

✅ **"The system understands context"**
- Content type detection (prose, code, lists, etc.)
- Cross-domain reasoning recognition
- Personal vs. professional distinction

✅ **"It works even when offline"**
- Enhanced fallback parsing
- Local heuristic analysis
- Graceful degradation

✅ **"I can see how it thinks"**
- Glass Engine transparency
- Confidence scoring
- Parser type identification

Phase 2 transforms scattered thoughts into intelligently organized knowledge, ready for the semantic search and clustering capabilities of Phase 3.