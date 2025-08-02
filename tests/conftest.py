"""
Test configuration and fixtures for Globule tests.

Provides common fixtures and utilities for testing without requiring
external dependencies like the vec0 SQLite extension.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from globule.core.models import ProcessedGlobule, FileDecision


@pytest.fixture
def mock_storage_manager():
    """Create a mock storage manager that doesn't require vec0 extension."""
    storage = AsyncMock()
    storage.initialize = AsyncMock()
    storage.close = AsyncMock()
    storage.store_globule = AsyncMock(return_value="test-id-123")
    storage.get_globule = AsyncMock()
    storage.get_recent_globules = AsyncMock(return_value=[])
    storage.search_by_embedding = AsyncMock(return_value=[])
    storage.update_globule = AsyncMock(return_value=True)
    storage.delete_globule = AsyncMock(return_value=True)
    return storage


@pytest.fixture
def sample_globule():
    """Create a sample ProcessedGlobule for testing."""
    return ProcessedGlobule(
        id="test-globule-123",
        text="This is a test thought for unit testing",
        embedding=np.random.randn(1024).astype(np.float32),
        embedding_confidence=0.95,
        parsed_data={
            "title": "Test Thought",
            "category": "testing",
            "domain": "development",
            "keywords": ["test", "unit", "development"],
            "metadata": {"source": "test"}
        },
        parsing_confidence=0.90,
        file_decision=FileDecision(
            semantic_path=Path("development/testing"),
            filename="test-thought.md",
            metadata={"test": True},
            confidence=0.8,
            alternative_paths=[]
        ),
        orchestration_strategy="parallel",
        confidence_scores={"embedding": 0.95, "parsing": 0.90},
        processing_time_ms={"total": 150, "embedding": 80, "parsing": 70},
        semantic_neighbors=["neighbor-1", "neighbor-2"],
        processing_notes=["Test note"],
        created_at=datetime.now(),
        modified_at=datetime.now()
    )


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings with known relationships for testing."""
    # Create a base vector that we can create variations of
    base_vector = np.random.rand(1024).astype(np.float32)
    
    return {
        "similar_1": base_vector + np.random.normal(0, 0.1, 1024).astype(np.float32),
        "similar_2": base_vector + np.random.normal(0, 0.1, 1024).astype(np.float32),
        "different_1": np.random.rand(1024).astype(np.float32),
        "different_2": np.random.rand(1024).astype(np.float32),
        "query": base_vector + np.random.normal(0, 0.05, 1024).astype(np.float32)
    }


@pytest.fixture
def temp_db_path():
    """Create a temporary database path for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir) / "test.db"


@pytest.fixture
def mock_embedding_provider():
    """Create a mock embedding provider."""
    provider = MagicMock()
    provider.embed = AsyncMock(return_value=np.random.randn(1024).astype(np.float32))
    provider.embed_batch = AsyncMock(side_effect=lambda texts: [
        np.random.randn(1024).astype(np.float32) for _ in texts
    ])
    provider.get_dimension = MagicMock(return_value=1024)
    provider.health_check = AsyncMock(return_value=True)
    provider.close = AsyncMock()
    return provider


@pytest.fixture
def mock_parsing_provider():
    """Create a mock parsing provider."""
    provider = MagicMock()
    provider.parse = AsyncMock(return_value={
        "title": "Test Title",
        "category": "test",
        "domain": "testing",
        "keywords": ["test", "mock"],
        "metadata": {"mock": True}
    })
    provider.health_check = AsyncMock(return_value=True)
    provider.close = AsyncMock()
    return provider


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "requires_vec0: mark test as requiring vec0 SQLite extension"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip vec0-dependent tests if extension unavailable."""
    import sqlite3
    
    # Check if vec0 extension is available
    vec0_available = False
    try:
        import sqlite_vec
        conn = sqlite3.connect(":memory:")
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        vec0_available = True
        conn.close()
    except Exception:
        # Try fallback to old vec0 name
        try:
            conn = sqlite3.connect(":memory:")
            conn.enable_load_extension(True)
            conn.load_extension("vec0")
            vec0_available = True
            conn.close()
        except Exception:
            vec0_available = False
    
    if not vec0_available:
        skip_vec0 = pytest.mark.skip(reason="vec0 SQLite extension not available")
        for item in items:
            if "requires_vec0" in item.keywords:
                item.add_marker(skip_vec0)
            # Also skip tests that use storage initialization
            if "storage" in item.name.lower() and "mock" not in item.name.lower():
                item.add_marker(skip_vec0)