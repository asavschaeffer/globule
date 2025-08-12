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

from globule.core.models import ProcessedGlobuleV1, FileDecisionV1
from globule.orchestration.engine import OrchestrationEngine


class TestCoreModels:
    """Test core data models."""
    
    def test_processed_globule_creation(self):
        """Test ProcessedGlobuleV1 can be created with minimal data."""
        globule = ProcessedGlobuleV1(
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
    
    def test_file_decision_creation(self):
        """Test FileDecision creation."""
        decision = FileDecisionV1(
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


class TestFilePath:
    """Test file path generation logic."""
    
    def test_file_decision_path_construction(self):
        """Test that file paths are constructed correctly."""
        decision = FileDecisionV1(
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