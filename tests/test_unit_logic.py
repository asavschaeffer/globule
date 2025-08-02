"""
Unit tests for core logic that doesn't require external dependencies.

These tests verify the core functionality without requiring:
- vec0 SQLite extension
- Ollama service
- External AI services
"""

import pytest
import numpy as np
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, AsyncMock

from globule.core.models import ProcessedGlobule, EnrichedInput, FileDecision
from globule.orchestration.engine import OrchestrationEngine


class TestCoreModels:
    """Test core data models."""
    
    def test_processed_globule_creation(self):
        """Test ProcessedGlobule can be created with minimal data."""
        globule = ProcessedGlobule(
            text="Test thought",
            parsed_data={"title": "Test"},
            created_at=datetime.now(),
            modified_at=datetime.now()
        )
        
        assert globule.text == "Test thought"
        assert globule.parsed_data["title"] == "Test"
        assert globule.id is None  # Should be auto-generated when needed
        assert globule.embedding is None
        assert globule.confidence_scores == {}
    
    def test_enriched_input_creation(self):
        """Test EnrichedInput creation."""
        enriched = EnrichedInput(
            original_text="Original text",
            enriched_text="Enhanced text",
            detected_schema_id=None,
            schema_config=None,
            additional_context={},
            source="test"
        )
        
        assert enriched.original_text == "Original text"
        assert enriched.enriched_text == "Enhanced text"
        assert enriched.source == "test"
        assert enriched.additional_context == {}
    
    def test_file_decision_creation(self):
        """Test FileDecision creation."""
        decision = FileDecision(
            semantic_path=Path("notes/personal"),
            filename="test.md",
            metadata={},
            confidence=0.9,
            alternative_paths=[]
        )
        
        assert decision.semantic_path == Path("notes/personal")
        assert decision.filename == "test.md"
        assert decision.confidence == 0.9
        assert decision.metadata == {}
        assert decision.alternative_paths == []


class TestOrchestrationEngine:
    """Test orchestration engine logic without external dependencies."""
    
    @pytest.fixture
    def mock_embedding_provider(self):
        """Create a mock embedding provider."""
        provider = Mock()
        provider.embed = AsyncMock(return_value=np.random.randn(1024).astype(np.float32))
        provider.get_dimension = Mock(return_value=1024)
        return provider
    
    @pytest.fixture
    def mock_parsing_provider(self):
        """Create a mock parsing provider."""
        provider = Mock()
        provider.parse = AsyncMock(return_value={
            "title": "Test Title",
            "category": "test",
            "domain": "testing",
            "keywords": ["test", "mock"]
        })
        return provider
    
    @pytest.fixture
    def mock_storage_manager(self):
        """Create a mock storage manager."""
        storage = Mock()
        storage.store_globule = AsyncMock(return_value="test-id-123")
        return storage
    
    @pytest.fixture
    def orchestrator(self, mock_embedding_provider, mock_parsing_provider, mock_storage_manager):
        """Create orchestrator with mocked dependencies."""
        return OrchestrationEngine(
            mock_embedding_provider,
            mock_parsing_provider, 
            mock_storage_manager
        )
    
    @pytest.mark.asyncio
    async def test_process_globule_success(self, orchestrator, mock_embedding_provider, mock_parsing_provider):
        """Test successful globule processing."""
        # Create test input
        enriched_input = EnrichedInput(
            original_text="Test thought for processing",
            enriched_text="Test thought for processing",
            detected_schema_id=None,
            schema_config=None,
            additional_context={},
            source="test"
        )
        
        # Process globule
        result = await orchestrator.process_globule(enriched_input)
        
        # Verify result structure
        assert isinstance(result, ProcessedGlobule)
        assert result.text == "Test thought for processing"
        assert result.embedding is not None
        assert len(result.embedding) == 1024
        assert result.parsed_data["title"] == "Test Title"
        assert result.orchestration_strategy == "parallel"
        
        # Verify providers were called
        mock_embedding_provider.embed.assert_called_once_with("Test thought for processing")
        mock_parsing_provider.parse.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_globule_embedding_failure(self, orchestrator, mock_embedding_provider, mock_parsing_provider):
        """Test globule processing when embedding fails."""
        # Make embedding fail
        mock_embedding_provider.embed.side_effect = Exception("Embedding service unavailable")
        
        enriched_input = EnrichedInput(
            original_text="Test thought",
            enriched_text="Test thought",
            detected_schema_id=None,
            schema_config=None,
            additional_context={},
            source="test"
        )
        
        # Process should still succeed with graceful fallback
        result = await orchestrator.process_globule(enriched_input)
        
        assert isinstance(result, ProcessedGlobule)
        assert result.text == "Test thought"
        assert result.embedding is None  # Should be None due to failure
        assert result.embedding_confidence == 0.0  # Should indicate failure
        assert result.parsed_data["title"] == "Test Title"  # Parsing should still work
    
    @pytest.mark.asyncio
    async def test_process_globule_parsing_failure(self, orchestrator, mock_embedding_provider, mock_parsing_provider):
        """Test globule processing when parsing fails."""
        # Make parsing fail
        mock_parsing_provider.parse.side_effect = Exception("Parser unavailable")
        
        enriched_input = EnrichedInput(
            original_text="Test thought",
            enriched_text="Test thought",
            detected_schema_id=None,
            schema_config=None,
            additional_context={},
            source="test"
        )
        
        # Process should still succeed with graceful fallback
        result = await orchestrator.process_globule(enriched_input)
        
        assert isinstance(result, ProcessedGlobule)
        assert result.text == "Test thought"
        assert result.embedding is not None  # Embedding should still work
        assert result.parsing_confidence == 0.0  # Should indicate parsing failure
        assert "error" in result.parsed_data  # Should contain error info


class TestFilePath:
    """Test file path generation logic."""
    
    def test_file_decision_path_construction(self):
        """Test that file paths are constructed correctly."""
        decision = FileDecision(
            semantic_path=Path("projects/ai"),
            filename="neural-networks.md",
            metadata={},
            confidence=0.85,
            alternative_paths=[]
        )
        
        full_path = decision.semantic_path / decision.filename
        # Use forward slashes for cross-platform compatibility
        assert str(full_path).replace("\\", "/") == "projects/ai/neural-networks.md"
        assert full_path.suffix == ".md"
        assert full_path.stem == "neural-networks"


class TestVectorOperations:
    """Test vector operations without requiring database."""
    
    def test_vector_normalization(self):
        """Test vector normalization logic."""
        # Create a test vector
        vector = np.array([3.0, 4.0, 0.0])
        
        # Normalize it
        norm = np.linalg.norm(vector)
        normalized = vector / norm if norm > 0 else vector
        
        # Check properties
        assert np.allclose(np.linalg.norm(normalized), 1.0)
        assert normalized.shape == vector.shape
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        # Create test vectors
        vec_a = np.array([1.0, 0.0, 0.0])
        vec_b = np.array([0.0, 1.0, 0.0])
        vec_c = np.array([1.0, 0.0, 0.0])
        
        # Calculate similarities
        def cosine_similarity(a, b):
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot_product / (norm_a * norm_b)
        
        # Test cases
        assert np.isclose(cosine_similarity(vec_a, vec_c), 1.0)  # Same direction
        assert np.isclose(cosine_similarity(vec_a, vec_b), 0.0)  # Orthogonal
        assert np.isclose(cosine_similarity(vec_a, -vec_a), -1.0)  # Opposite


if __name__ == "__main__":
    pytest.main([__file__, "-v"])