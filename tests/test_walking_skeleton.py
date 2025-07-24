"""
Test the walking skeleton implementation.

This verifies that the basic end-to-end flow works:
1. Add a globule 
2. Display it in the TUI
"""

import pytest
import asyncio
import tempfile
from pathlib import Path

from globule.core.models import EnrichedInput
from globule.storage.sqlite_manager import SQLiteStorageManager
from globule.embedding.ollama_provider import OllamaEmbeddingProvider
from globule.parsing.mock_parser import MockOllamaParser
from globule.orchestration.parallel_strategy import ParallelOrchestrationEngine


@pytest.mark.asyncio
async def test_storage_initialization():
    """Test that storage manager can initialize database"""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"
        storage = SQLiteStorageManager(db_path)
        await storage.initialize()
        
        # Verify database file was created
        assert db_path.exists()
        
        await storage.close()


@pytest.mark.asyncio
async def test_mock_parser():
    """Test that mock parser returns expected structure"""
    parser = MockOllamaParser()
    result = await parser.parse("This is a test thought")
    
    assert isinstance(result, dict)
    assert "title" in result
    assert "category" in result
    assert "domain" in result
    assert result["metadata"]["mock"] is True


@pytest.mark.asyncio
async def test_orchestration_without_ollama():
    """Test orchestration with mock embedding (no Ollama required)"""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"
        storage = SQLiteStorageManager(db_path)
        await storage.initialize()
        
        # Create mock embedding provider for testing
        class MockEmbeddingProvider:
            def get_dimension(self):
                return 1024
            
            async def embed(self, text):
                # Return mock embedding
                import numpy as np
                return np.random.randn(1024).astype(np.float32)
            
            async def embed_batch(self, texts):
                return [await self.embed(text) for text in texts]
            
            async def close(self):
                pass
        
        embedding_provider = MockEmbeddingProvider()
        parsing_provider = MockOllamaParser()
        orchestrator = ParallelOrchestrationEngine(
            embedding_provider, parsing_provider, storage
        )
        
        # Create test input
        enriched_input = EnrichedInput(
            original_text="This is a test thought",
            enriched_text="This is a test thought",
            detected_schema_id=None,
            schema_config=None,
            additional_context={},
            source="test"
        )
        
        # Process globule
        processed_globule = await orchestrator.process_globule(enriched_input)
        
        # Verify result
        assert processed_globule.text == "This is a test thought"
        assert processed_globule.embedding is not None
        assert len(processed_globule.embedding) == 1024
        assert processed_globule.parsed_data["metadata"]["mock"] is True
        
        # Store and retrieve
        globule_id = await storage.store_globule(processed_globule)
        retrieved = await storage.get_globule(globule_id)
        
        assert retrieved is not None
        assert retrieved.text == "This is a test thought"
        assert retrieved.id == globule_id
        
        await storage.close()


@pytest.mark.asyncio
async def test_recent_globules_retrieval():
    """Test retrieving recent globules"""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"
        storage = SQLiteStorageManager(db_path)
        await storage.initialize()
        
        # Add some test globules
        from globule.core.models import ProcessedGlobule
        import numpy as np
        
        for i in range(3):
            globule = ProcessedGlobule(
                text=f"Test thought {i}",
                embedding=np.random.randn(1024).astype(np.float32),
                embedding_confidence=0.9,
                parsed_data={"title": f"Thought {i}", "mock": True},
                parsing_confidence=0.9
            )
            await storage.store_globule(globule)
        
        # Retrieve recent globules
        recent = await storage.get_recent_globules(limit=5)
        
        assert len(recent) == 3
        assert all(g.text.startswith("Test thought") for g in recent)
        # Verify parsed data structure
        assert all(g.parsed_data.get("metadata", {}).get("mock") is True for g in recent)
        
        await storage.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])