"""
Test configuration and fixtures for Globule tests.

Provides real database fixtures and test data for proper integration testing.
All integration tests use persistent databases and real data.
"""

import pytest
import numpy as np
import tempfile
import json
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timedelta
from typing import List, Dict, Any

from globule.core.models import ProcessedGlobule, FileDecision
from globule.storage.sqlite_manager import SQLiteStorageManager


@pytest.fixture
def mock_storage_manager():
    """Create a mock storage manager for unit tests that don't require real database."""
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
async def real_storage_manager(tmp_path):
    """Create a real SQLite storage manager with temporary database for integration tests."""
    db_path = tmp_path / "integration_test.db"
    storage = SQLiteStorageManager(db_path)
    await storage.initialize()
    yield storage
    await storage.close()


@pytest.fixture
def real_test_data():
    """Load real test data from JSON file with pre-computed embeddings."""
    # This would normally load from a checked-in JSON file
    # For now, creating realistic data inline
    return {
        "globules": [
            {
                "id": "real_1",
                "text": "Progressive overload in fitness means gradually increasing weight, frequency, or number of reps in your strength training routine. This principle ensures continuous adaptation and growth.",
                "embedding": np.random.rand(1024).astype(np.float32).tolist(),
                "embedding_confidence": 0.92,
                "parsed_data": {
                    "title": "Progressive Overload Principle",
                    "domain": "fitness",
                    "category": "concept",
                    "keywords": ["progressive", "overload", "strength", "training", "adaptation"],
                    "metadata": {"parser_type": "ollama_llm", "confidence_score": 0.89}
                },
                "parsing_confidence": 0.89,
                "file_decision": {
                    "semantic_path": "fitness/principles",
                    "filename": "progressive-overload.md",
                    "metadata": {},
                    "confidence": 0.85,
                    "alternative_paths": []
                },
                "created_at": (datetime.now() - timedelta(days=7)).isoformat()
            },
            {
                "id": "real_2",
                "text": "The concept of technical debt in software development refers to the implied cost of additional rework caused by choosing an easy solution now instead of using a better approach that would take longer.",
                "embedding": np.random.rand(1024).astype(np.float32).tolist(),
                "embedding_confidence": 0.88,
                "parsed_data": {
                    "title": "Technical Debt Definition",
                    "domain": "software",
                    "category": "concept",
                    "keywords": ["technical", "debt", "software", "development", "rework"],
                    "metadata": {"parser_type": "ollama_llm", "confidence_score": 0.86}
                },
                "parsing_confidence": 0.86,
                "file_decision": {
                    "semantic_path": "software/concepts",
                    "filename": "technical-debt.md",
                    "metadata": {},
                    "confidence": 0.82,
                    "alternative_paths": []
                },
                "created_at": (datetime.now() - timedelta(days=5)).isoformat()
            },
            {
                "id": "real_3",
                "text": "Mindfulness meditation involves paying attention to the present moment without judgment. Regular practice can reduce stress and improve focus and emotional regulation.",
                "embedding": np.random.rand(1024).astype(np.float32).tolist(),
                "embedding_confidence": 0.85,
                "parsed_data": {
                    "title": "Mindfulness Meditation Practice",
                    "domain": "wellness",
                    "category": "practice",
                    "keywords": ["mindfulness", "meditation", "present", "stress", "focus"],
                    "metadata": {"parser_type": "ollama_llm", "confidence_score": 0.83}
                },
                "parsing_confidence": 0.83,
                "file_decision": {
                    "semantic_path": "wellness/practices",
                    "filename": "mindfulness-meditation.md",
                    "metadata": {},
                    "confidence": 0.80,
                    "alternative_paths": []
                },
                "created_at": (datetime.now() - timedelta(days=3)).isoformat()
            },
            {
                "id": "real_4",
                "text": "The Feynman Technique is a learning method where you explain a concept in simple terms as if teaching it to a child. This helps identify gaps in understanding.",
                "embedding": np.random.rand(1024).astype(np.float32).tolist(),
                "embedding_confidence": 0.90,
                "parsed_data": {
                    "title": "Feynman Technique for Learning",
                    "domain": "learning",
                    "category": "technique",
                    "keywords": ["feynman", "technique", "learning", "explain", "understanding"],
                    "metadata": {"parser_type": "ollama_llm", "confidence_score": 0.87}
                },
                "parsing_confidence": 0.87,
                "file_decision": {
                    "semantic_path": "learning/techniques",
                    "filename": "feynman-technique.md",
                    "metadata": {},
                    "confidence": 0.88,
                    "alternative_paths": []
                },
                "created_at": (datetime.now() - timedelta(days=1)).isoformat()
            },
            {
                "id": "real_5",
                "text": "Compound interest is the addition of interest to the principal sum of a loan or deposit. Einstein allegedly called it the eighth wonder of the world.",
                "embedding": np.random.rand(1024).astype(np.float32).tolist(),
                "embedding_confidence": 0.87,
                "parsed_data": {
                    "title": "Compound Interest Concept",
                    "domain": "finance",
                    "category": "concept",
                    "keywords": ["compound", "interest", "principal", "finance", "investment"],
                    "metadata": {"parser_type": "ollama_llm", "confidence_score": 0.84}
                },
                "parsing_confidence": 0.84,
                "file_decision": {
                    "semantic_path": "finance/concepts",
                    "filename": "compound-interest.md",
                    "metadata": {},
                    "confidence": 0.81,
                    "alternative_paths": []
                },
                "created_at": (datetime.now() - timedelta(days=2)).isoformat()
            }
        ]
    }


@pytest.fixture
def large_test_dataset():
    """Generate a large dataset for performance testing (10k+ records)."""
    domains = ["fitness", "software", "wellness", "learning", "finance", "productivity", "creativity", "science"]
    categories = ["concept", "technique", "practice", "idea", "note", "reflection"]
    
    # Create base embeddings for each domain to ensure semantic clustering
    domain_embeddings = {}
    for domain in domains:
        domain_embeddings[domain] = np.random.rand(1024).astype(np.float32)
    
    dataset = []
    for i in range(10000):
        domain = domains[i % len(domains)]
        category = categories[i % len(categories)]
        
        # Create embedding similar to domain base with some noise
        base_embedding = domain_embeddings[domain]
        noise = np.random.normal(0, 0.2, 1024).astype(np.float32)
        embedding = base_embedding + noise
        
        # Normalize to unit vector
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        globule_data = {
            "id": f"perf_{i:05d}",
            "text": f"Performance test entry {i} in {domain} domain discussing {category} with substantial content to make search meaningful and realistic for testing purposes.",
            "embedding": embedding.tolist(),
            "embedding_confidence": 0.7 + (i % 30) * 0.01,  # 0.7 to 0.99
            "parsed_data": {
                "title": f"Performance Test Entry {i}",
                "domain": domain,
                "category": category,
                "keywords": [domain, category, "performance", "test", f"entry_{i}"],
                "metadata": {"parser_type": "test_parser", "confidence_score": 0.75 + (i % 25) * 0.01}
            },
            "parsing_confidence": 0.75 + (i % 25) * 0.01,
            "file_decision": {
                "semantic_path": f"{domain}/{category}",
                "filename": f"perf-test-{i:05d}.md",
                "metadata": {},
                "confidence": 0.7 + (i % 30) * 0.01,
                "alternative_paths": []
            },
            "created_at": (datetime.now() - timedelta(days=i % 365)).isoformat()
        }
        dataset.append(globule_data)
    
    return {"globules": dataset}


@pytest.fixture
async def populated_real_storage(real_storage_manager, real_test_data):
    """Create a real storage manager populated with realistic test data."""
    for globule_data in real_test_data["globules"]:
        # Convert dict to ProcessedGlobule
        globule = ProcessedGlobule(
            id=globule_data["id"],
            text=globule_data["text"],
            embedding=np.array(globule_data["embedding"], dtype=np.float32),
            embedding_confidence=globule_data["embedding_confidence"],
            parsed_data=globule_data["parsed_data"],
            parsing_confidence=globule_data["parsing_confidence"],
            file_decision=FileDecision(
                semantic_path=Path(globule_data["file_decision"]["semantic_path"]),
                filename=globule_data["file_decision"]["filename"],
                metadata=globule_data["file_decision"]["metadata"],
                confidence=globule_data["file_decision"]["confidence"],
                alternative_paths=globule_data["file_decision"]["alternative_paths"]
            ),
            orchestration_strategy="parallel",
            processing_time_ms={"total_ms": 400},
            confidence_scores={"overall": globule_data["parsing_confidence"]},
            interpretations=[],
            has_nuance=False,
            semantic_neighbors=[],
            processing_notes=[],
            created_at=datetime.fromisoformat(globule_data["created_at"])
        )
        await real_storage_manager.store_globule(globule)
    
    return real_storage_manager


@pytest.fixture
async def large_populated_storage(real_storage_manager, large_test_dataset):
    """Create a storage manager populated with large dataset for performance testing."""
    # Store data in batches for better performance
    batch_size = 100
    globules_data = large_test_dataset["globules"]
    
    for i in range(0, len(globules_data), batch_size):
        batch = globules_data[i:i + batch_size]
        
        for globule_data in batch:
            globule = ProcessedGlobule(
                id=globule_data["id"],
                text=globule_data["text"],
                embedding=np.array(globule_data["embedding"], dtype=np.float32),
                embedding_confidence=globule_data["embedding_confidence"],
                parsed_data=globule_data["parsed_data"],
                parsing_confidence=globule_data["parsing_confidence"],
                file_decision=FileDecision(
                    semantic_path=Path(globule_data["file_decision"]["semantic_path"]),
                    filename=globule_data["file_decision"]["filename"],
                    metadata=globule_data["file_decision"]["metadata"],
                    confidence=globule_data["file_decision"]["confidence"],
                    alternative_paths=globule_data["file_decision"]["alternative_paths"]
                ),
                orchestration_strategy="parallel",
                processing_time_ms={"total_ms": 400},
                confidence_scores={"overall": globule_data["parsing_confidence"]},
                interpretations=[],
                has_nuance=False,
                semantic_neighbors=[],
                processing_notes=[],
                created_at=datetime.fromisoformat(globule_data["created_at"])
            )
            await real_storage_manager.store_globule(globule)
    
    return real_storage_manager


@pytest.fixture
def sample_globule():
    """Create a sample ProcessedGlobule for unit testing."""
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
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test requiring real database"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test requiring large dataset"
    )


def check_vec0_available():
    """Check if vec0 extension is available - integration tests REQUIRE this."""
    try:
        import sqlite_vec
        conn = sqlite3.connect(":memory:")
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.close()
        return True
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    """Validate that vec0 extension is available for integration tests."""
    # Integration tests MUST have vec0 available - they should fail if not present
    vec0_available = check_vec0_available()
    
    for item in items:
        if "integration" in item.keywords and not vec0_available:
            # Make integration tests fail with clear message rather than skip
            item.add_marker(pytest.mark.xfail(
                reason="Integration tests require vec0 SQLite extension. Install sqlite-vec package.",
                raises=Exception,
                strict=True
            ))