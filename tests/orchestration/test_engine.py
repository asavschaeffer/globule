"""
Unit tests for GlobuleOrchestrator.

These tests validate that the orchestrator correctly implements the IOrchestrationEngine
interface and properly coordinates business logic using the injected providers.
"""

import pytest
import asyncio
import os
import tempfile
from uuid import uuid4
from datetime import datetime

from globule.orchestration.engine import GlobuleOrchestrator
from globule.services.providers_mock import MockParserProvider, MockEmbeddingProvider, MockStorageManager
from globule.core.models import GlobuleV1, ProcessedGlobuleV1
from globule.core.interfaces import IOrchestrationEngine


class TestGlobuleOrchestrator:
    """Test suite for GlobuleOrchestrator business logic."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator with mock providers."""
        parser = MockParserProvider()
        embedder = MockEmbeddingProvider()
        storage = MockStorageManager()
        
        return GlobuleOrchestrator(
            parser_provider=parser,
            embedding_provider=embedder,
            storage_manager=storage
        )
    
    @pytest.fixture
    def sample_globule(self):
        """Create a sample globule for testing."""
        return GlobuleV1(
            raw_text="This is a test thought about machine learning",
            source="test",
            initial_context={"test": True}
        )
    
    def test_implements_interface(self, orchestrator):
        """Test that orchestrator properly implements IOrchestrationEngine."""
        assert isinstance(orchestrator, IOrchestrationEngine)
        
        # Test that required methods exist
        assert hasattr(orchestrator, 'process')
        assert callable(orchestrator.process)
    
    def test_process_interface_compliance(self, orchestrator, sample_globule):
        """Test that process method complies with interface."""
        result = orchestrator.process(sample_globule)
        
        # Should return ProcessedGlobuleV1
        assert isinstance(result, ProcessedGlobuleV1)
        assert result.globule_id == sample_globule.globule_id
        assert result.original_globule == sample_globule
    
    @pytest.mark.asyncio
    async def test_capture_thought_workflow(self, orchestrator):
        """Test complete thought capture workflow."""
        raw_text = "I need to remember to implement async processing"
        
        result = await orchestrator.capture_thought(
            raw_text=raw_text,
            source="test",
            context={"priority": "high"}
        )
        
        # Verify result structure
        assert isinstance(result, ProcessedGlobuleV1)
        assert result.original_globule.raw_text == raw_text
        assert result.original_globule.source == "test"
        assert result.original_globule.initial_context["priority"] == "high"
        
        # Verify processing occurred
        assert result.processing_time_ms > 0
        assert result.parsed_data is not None
        assert len(result.embedding) > 0  # Mock provider should return non-empty embedding
        
        # Verify provider metadata
        assert "MockParserProvider" in result.provider_metadata["parser"]
        assert "MockEmbeddingProvider" in result.provider_metadata["embedder"]
        assert "MockStorageManager" in result.provider_metadata["storage"]
    
    @pytest.mark.asyncio
    async def test_search_globules(self, orchestrator):
        """Test globule search functionality."""
        # Test basic search
        results = await orchestrator.search_globules("machine learning", limit=5)
        
        # Should return list (empty for mock implementation)
        assert isinstance(results, list)
        assert len(results) <= 5
    
    @pytest.mark.asyncio 
    async def test_get_globule_by_id(self, orchestrator):
        """Test globule retrieval by ID."""
        # First capture a thought to have something to retrieve
        captured = await orchestrator.capture_thought("Test thought for retrieval")
        globule_id = captured.globule_id
        
        # Retrieve it
        retrieved = await orchestrator.get_globule(globule_id)
        
        assert retrieved is not None
        assert retrieved.globule_id == globule_id
        assert retrieved.original_globule.raw_text == "Test thought for retrieval"
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_globule(self, orchestrator):
        """Test retrieval of non-existent globule returns None."""
        fake_id = uuid4()
        result = await orchestrator.get_globule(fake_id)
        assert result is None
    
    def test_save_draft_functionality(self, orchestrator):
        """Test draft saving with metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Change to temp directory for test
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            
            try:
                content = "# Test Draft\n\nThis is test content for draft saving."
                topic = "test_topic"
                metadata = {"schema": "test_schema", "priority": "high"}
                
                filepath = orchestrator.save_draft(
                    content=content,
                    topic=topic,
                    metadata=metadata
                )
                
                # Verify file was created
                assert os.path.exists(filepath)
                
                # Verify content
                with open(filepath, 'r', encoding='utf-8') as f:
                    saved_content = f.read()
                
                assert "schema: test_schema" in saved_content
                assert "priority: high" in saved_content
                assert "topic: test_topic" in saved_content
                assert content in saved_content
                
            finally:
                os.chdir(original_cwd)
    
    def test_save_draft_without_metadata(self, orchestrator):
        """Test draft saving without metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            
            try:
                content = "Simple draft without metadata"
                
                filepath = orchestrator.save_draft(content=content)
                
                assert os.path.exists(filepath)
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    saved_content = f.read()
                
                assert content in saved_content
                assert "---" not in saved_content  # No metadata frontmatter
                
            finally:
                os.chdir(original_cwd)
    
    @pytest.mark.asyncio
    async def test_execute_query_natural_language(self, orchestrator):
        """Test query execution with natural language."""
        query = "find all thoughts about testing"
        
        result = await orchestrator.execute_query(query, "natural")
        
        assert result["type"] == "search_results"
        assert result["query"] == query
        assert "results" in result
        assert "count" in result
        assert isinstance(result["results"], list)
    
    @pytest.mark.asyncio
    async def test_execute_query_unknown_type(self, orchestrator):
        """Test query execution with unknown query type."""
        result = await orchestrator.execute_query("test query", "unknown_type")
        
        assert result["type"] == "unknown"
        assert result["query"] == "test query"
    
    def test_provider_injection(self, orchestrator):
        """Test that providers are properly injected and accessible."""
        assert orchestrator.parser_provider is not None
        assert orchestrator.embedding_provider is not None
        assert orchestrator.storage_manager is not None
        
        # Test provider types
        assert isinstance(orchestrator.parser_provider, MockParserProvider)
        assert isinstance(orchestrator.embedding_provider, MockEmbeddingProvider)
        assert isinstance(orchestrator.storage_manager, MockStorageManager)
    
    @pytest.mark.asyncio
    async def test_processing_times_recorded(self, orchestrator, sample_globule):
        """Test that processing times are properly recorded."""
        result = await orchestrator._process_async(sample_globule)
        
        assert result.processing_time_ms > 0
        assert result.processing_time_ms < 10000  # Should be reasonable for mock providers
    
    @pytest.mark.asyncio
    async def test_error_handling_in_processing(self, orchestrator):
        """Test error handling during globule processing."""
        # Create a globule that might cause issues
        problematic_globule = GlobuleV1(
            raw_text="",  # Empty text might cause parsing issues
            source="test"
        )
        
        # Should not raise exception, but handle gracefully
        result = await orchestrator._process_async(problematic_globule)
        
        assert isinstance(result, ProcessedGlobuleV1)
        assert result.original_globule == problematic_globule


class TestOrchestrator_MockProviders:
    """Test the mock providers themselves to ensure they work correctly."""
    
    def test_mock_parser_provider(self):
        """Test MockParserProvider functionality."""
        parser = MockParserProvider()
        
        result = parser.parse("Test text for parsing")
        
        assert isinstance(result, dict)
        assert "title" in result
        assert "category" in result
        assert "domain" in result
        assert result["metadata"]["mock"] is True
    
    def test_mock_embedding_provider(self):
        """Test MockEmbeddingProvider functionality."""
        embedder = MockEmbeddingProvider(dimension=512)
        
        embedding = embedder.embed("Test text for embedding")
        
        assert isinstance(embedding, list)
        assert len(embedding) == 512
        assert all(isinstance(x, float) for x in embedding)
        
        # Test reproducibility - same text should give same embedding
        embedding2 = embedder.embed("Test text for embedding")
        assert embedding == embedding2
    
    def test_mock_storage_manager(self):
        """Test MockStorageManager functionality."""
        storage = MockStorageManager()
        
        # Create a test globule
        from globule.core.models import GlobuleV1
        globule = GlobuleV1(raw_text="Test", source="test")
        processed = ProcessedGlobuleV1(
            globule_id=globule.globule_id,
            original_globule=globule,
            embedding=[0.1, 0.2, 0.3],
            processing_time_ms=100.0
        )
        
        # Test save and retrieve
        storage.save(processed)
        retrieved = storage.get(processed.globule_id)
        
        assert retrieved.globule_id == processed.globule_id
        assert retrieved.original_globule.raw_text == "Test"