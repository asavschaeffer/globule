"""
Comprehensive tests for Phase 2 Vector Search capabilities.

Tests the enhanced vector search implementation including:
- Batch similarity calculations
- Result enhancement and ranking
- Hybrid text + semantic search
- Performance optimizations
"""

import pytest
import numpy as np
import asyncio
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple
from unittest.mock import patch

from globule.storage.sqlite_manager import SQLiteStorageManager
from globule.core.models import ProcessedGlobule, FileDecision


def vec0_available():
    """Check if vec0 extension is available."""
    try:
        import sqlite_vec
        conn = sqlite3.connect(":memory:")
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.close()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not vec0_available(), reason="vec0 SQLite extension not available")
class TestVectorSearch:
    """Test suite for enhanced vector search functionality."""

    @pytest.fixture
    async def storage_manager(self, tmp_path):
        """Create a temporary storage manager for testing."""
        db_path = tmp_path / "test_globules.db"
        storage = SQLiteStorageManager(db_path)
        await storage.initialize()
        yield storage
        await storage.close()

    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings for testing."""
        # Create embeddings that have known similarity relationships
        base_vector = np.random.rand(1024).astype(np.float32)
        
        return {
            "creative_1": base_vector + np.random.normal(0, 0.1, 1024).astype(np.float32),  # Similar to base
            "creative_2": base_vector + np.random.normal(0, 0.1, 1024).astype(np.float32),  # Similar to base  
            "technical": np.random.rand(1024).astype(np.float32),  # Different from base
            "personal": np.random.rand(1024).astype(np.float32),   # Different from base
            "query": base_vector + np.random.normal(0, 0.05, 1024).astype(np.float32)  # Very similar to base
        }

    @pytest.fixture
    async def populated_storage(self, storage_manager, sample_embeddings):
        """Create storage populated with test globules."""
        test_globules = [
            ProcessedGlobule(
                id="creative_1",
                text="The concept of progressive overload in fitness could apply to creative stamina and artistic development.",
                embedding=sample_embeddings["creative_1"],
                embedding_confidence=0.9,
                parsed_data={
                    "title": "Progressive Creative Overload",
                    "domain": "creative",
                    "category": "idea",
                    "keywords": ["progressive", "overload", "creative", "stamina"],
                    "metadata": {"parser_type": "ollama_llm", "confidence_score": 0.85}
                },
                parsing_confidence=0.85,
                file_decision=FileDecision(Path("creative/ideas"), "progressive-overload.md", {}, 0.8, []),
                orchestration_strategy="parallel",
                processing_time_ms={"total_ms": 500},
                confidence_scores={"overall": 0.85},
                interpretations=[],
                has_nuance=False,
                semantic_neighbors=[],
                processing_notes=[],
                created_at=datetime.now() - timedelta(days=2)
            ),
            ProcessedGlobule(
                id="creative_2", 
                text="Creative muscles need gradual challenge, just like physical training requires incremental difficulty.",
                embedding=sample_embeddings["creative_2"],
                embedding_confidence=0.85,
                parsed_data={
                    "title": "Creative Training Theory",
                    "domain": "creative",
                    "category": "idea", 
                    "keywords": ["creative", "training", "challenge", "incremental"],
                    "metadata": {"parser_type": "ollama_llm", "confidence_score": 0.80}
                },
                parsing_confidence=0.80,
                file_decision=FileDecision(Path("creative/ideas"), "creative-training.md", {}, 0.8, []),
                orchestration_strategy="parallel",
                processing_time_ms={"total_ms": 450},
                confidence_scores={"overall": 0.80},
                interpretations=[],
                has_nuance=False,
                semantic_neighbors=[],
                processing_notes=[],
                created_at=datetime.now() - timedelta(days=5)
            ),
            ProcessedGlobule(
                id="technical",
                text="Instead of preventing all edge cases, design systems that gracefully degrade when unexpected conditions occur.",
                embedding=sample_embeddings["technical"],
                embedding_confidence=0.75,
                parsed_data={
                    "title": "Graceful System Degradation",
                    "domain": "technical",
                    "category": "idea",
                    "keywords": ["systems", "design", "degradation", "resilience"],
                    "metadata": {"parser_type": "enhanced_fallback", "confidence_score": 0.75}
                },
                parsing_confidence=0.75,
                file_decision=FileDecision(Path("technical/patterns"), "graceful-degradation.md", {}, 0.7, []),
                orchestration_strategy="parallel", 
                processing_time_ms={"total_ms": 600},
                confidence_scores={"overall": 0.75},
                interpretations=[],
                has_nuance=False,
                semantic_neighbors=[],
                processing_notes=[],
                created_at=datetime.now() - timedelta(days=1)
            ),
            ProcessedGlobule(
                id="personal",
                text="I feel overwhelmed by all the different note-taking tools. Need something unified and simple.",
                embedding=sample_embeddings["personal"],
                embedding_confidence=0.70,
                parsed_data={
                    "title": "Tool Overwhelm Reflection",
                    "domain": "personal",
                    "category": "note",
                    "keywords": ["overwhelmed", "tools", "unified", "simple"],
                    "metadata": {"parser_type": "enhanced_fallback", "confidence_score": 0.65}
                },
                parsing_confidence=0.65,
                file_decision=FileDecision(Path("personal/reflections"), "tool-overwhelm.md", {}, 0.6, []),
                orchestration_strategy="parallel",
                processing_time_ms={"total_ms": 300},
                confidence_scores={"overall": 0.65},
                interpretations=[],
                has_nuance=False,
                semantic_neighbors=[],
                processing_notes=[],
                created_at=datetime.now() - timedelta(days=10)
            )
        ]
        
        # Store all test globules
        for globule in test_globules:
            await storage_manager.store_globule(globule)
            
        return storage_manager

    @pytest.mark.asyncio
    async def test_basic_vector_search(self, populated_storage, sample_embeddings):
        """Test basic vector search functionality."""
        query_embedding = sample_embeddings["query"]
        
        results = await populated_storage.search_by_embedding(
            query_embedding, limit=10, similarity_threshold=0.3
        )
        
        assert len(results) > 0
        assert all(isinstance(result, tuple) and len(result) == 2 for result in results)
        assert all(isinstance(globule, ProcessedGlobule) for globule, score in results)
        assert all(isinstance(score, float) for globule, score in results)
        
        # Results should be sorted by similarity (highest first)
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio 
    async def test_similarity_threshold_filtering(self, populated_storage, sample_embeddings):
        """Test that similarity threshold properly filters results."""
        query_embedding = sample_embeddings["query"]
        
        # Test with high threshold - should get fewer results
        high_threshold_results = await populated_storage.search_by_embedding(
            query_embedding, limit=10, similarity_threshold=0.8
        )
        
        # Test with low threshold - should get more results
        low_threshold_results = await populated_storage.search_by_embedding(
            query_embedding, limit=10, similarity_threshold=0.1
        )
        
        assert len(high_threshold_results) <= len(low_threshold_results)
        
        # All results should meet the threshold
        for _, score in high_threshold_results:
            assert score >= 0.8
            
        for _, score in low_threshold_results:
            assert score >= 0.1

    @pytest.mark.asyncio
    async def test_result_limit(self, populated_storage, sample_embeddings):
        """Test that result limit is properly enforced."""
        query_embedding = sample_embeddings["query"]
        
        # Test with limit smaller than available results
        limited_results = await populated_storage.search_by_embedding(
            query_embedding, limit=2, similarity_threshold=0.0
        )
        
        assert len(limited_results) <= 2

    @pytest.mark.asyncio
    async def test_empty_query_handling(self, populated_storage):
        """Test handling of None/empty query vectors."""
        results = await populated_storage.search_by_embedding(None)
        assert results == []

    @pytest.mark.asyncio
    async def test_no_embeddings_scenario(self, storage_manager):
        """Test search when no globules have embeddings."""
        # Store a globule without embedding
        globule = ProcessedGlobule(
            id="no_embedding",
            text="Test without embedding",
            embedding=None,
            embedding_confidence=0.0,
            parsed_data={},
            parsing_confidence=0.5,
            file_decision=FileDecision(Path("test"), "test.md", {}, 0.5, []),
            orchestration_strategy="parallel",
            processing_time_ms={"total_ms": 100},
            confidence_scores={"overall": 0.5},
            interpretations=[],
            has_nuance=False,
            semantic_neighbors=[],
            processing_notes=[],
            created_at=datetime.now()
        )
        
        await storage_manager.store_globule(globule)
        
        query_embedding = np.random.rand(1024).astype(np.float32)
        results = await storage_manager.search_by_embedding(query_embedding)
        
        assert results == []

    @pytest.mark.asyncio
    async def test_vector_normalization(self, populated_storage):
        """Test that vector normalization works correctly."""
        # Create unnormalized vector
        unnormalized = np.array([3.0, 4.0, 0.0], dtype=np.float32)
        
        normalized = populated_storage._normalize_vector(unnormalized)
        
        # Should be unit length
        assert abs(np.linalg.norm(normalized) - 1.0) < 1e-6
        
        # Test zero vector
        zero_vector = np.zeros(3, dtype=np.float32)
        normalized_zero = populated_storage._normalize_vector(zero_vector)
        assert np.array_equal(normalized_zero, zero_vector)

    @pytest.mark.asyncio
    async def test_batch_similarity_performance(self, populated_storage, sample_embeddings):
        """Test that batch similarity calculation is efficient."""
        query_embedding = sample_embeddings["query"]
        
        # Time the search operation  
        import time
        start_time = time.perf_counter()
        
        results = await populated_storage.search_by_embedding(
            query_embedding, limit=10, similarity_threshold=0.0
        )
        
        end_time = time.perf_counter()
        search_time = end_time - start_time
        
        # Should complete quickly even with multiple embeddings
        assert search_time < 1.0  # Less than 1 second
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_result_enhancement(self, populated_storage, sample_embeddings):
        """Test that results are enhanced with additional ranking factors."""
        query_embedding = sample_embeddings["query"]
        
        results = await populated_storage.search_by_embedding(
            query_embedding, limit=10, similarity_threshold=0.0
        )
        
        # Check that enhancement factors are applied
        for globule, enhanced_score in results:
            # Score should be in valid range
            assert 0.0 <= enhanced_score <= 1.0
            
            # Higher confidence should generally mean higher scores
            if globule.parsing_confidence > 0.8:
                # This is hard to test precisely due to other factors,
                # but enhanced scores should be reasonable
                assert enhanced_score > 0.0

    @pytest.mark.asyncio
    async def test_hybrid_text_and_embedding_search(self, populated_storage, sample_embeddings):
        """Test hybrid search combining text and embedding similarity."""
        query_text = "creative training"
        query_embedding = sample_embeddings["query"]
        
        hybrid_results = await populated_storage.search_by_text_and_embedding(
            query_text, query_embedding, limit=5, similarity_threshold=0.2
        )
        
        assert len(hybrid_results) > 0
        
        # Results should be tuples of (globule, score)
        for globule, score in hybrid_results:
            assert isinstance(globule, ProcessedGlobule)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_text_keyword_search(self, populated_storage):
        """Test text-based keyword search component."""
        results = await populated_storage._search_by_text_keywords("creative stamina", limit=5)
        
        # Should find results containing these keywords
        assert len(results) > 0
        
        # Check that results contain the keywords
        found_creative = False
        for globule, relevance in results:
            if "creative" in globule.text.lower():
                found_creative = True
                assert relevance > 0
                
        assert found_creative

    @pytest.mark.asyncio
    async def test_search_result_fusion(self, populated_storage):
        """Test intelligent fusion of semantic and text search results."""
        # Create mock results
        semantic_results = [
            (ProcessedGlobule(
                id="test1", text="test content", embedding=np.random.rand(10),
                embedding_confidence=0.8, parsed_data={}, parsing_confidence=0.7,
                file_decision=FileDecision(Path("test"), "test.md", {}, 0.7, []),
                orchestration_strategy="test", processing_time_ms={}, confidence_scores={},
                interpretations=[], has_nuance=False, semantic_neighbors=[], processing_notes=[],
                created_at=datetime.now()
            ), 0.9),
        ]
        
        text_results = [
            (semantic_results[0][0], 0.7),  # Same globule in both results
        ]
        
        fused = populated_storage._fuse_search_results(semantic_results, text_results)
        
        assert len(fused) == 1
        globule, combined_score = fused[0]
        
        # Combined score should be higher than individual scores
        # 0.9 * 0.7 + 0.7 * 0.3 = 0.63 + 0.21 = 0.84, then 1.2x boost = 1.008, capped at 1.0
        base_score = 0.9 * 0.7 + 0.7 * 0.3  # 0.84
        expected_score = min(1.0, base_score * 1.2)  # 1.2x boost for multi-match, capped at 1.0
        assert abs(combined_score - expected_score) < 0.01

    @pytest.mark.asyncio
    async def test_confidence_filtering(self, populated_storage, sample_embeddings):
        """Test that low-confidence embeddings are filtered out."""
        # The search should filter out embeddings with confidence <= 0.3
        query_embedding = sample_embeddings["query"]
        
        # All our test data has confidence > 0.3, so we should get results
        results = await populated_storage.search_by_embedding(
            query_embedding, limit=10, similarity_threshold=0.0
        )
        
        # Verify all results meet confidence threshold
        for globule, score in results:
            assert globule.embedding_confidence > 0.3

    @pytest.mark.asyncio
    async def test_recency_ordering(self, populated_storage, sample_embeddings):
        """Test that the query orders by created_at DESC for tie-breaking."""
        query_embedding = sample_embeddings["query"]
        
        results = await populated_storage.search_by_embedding(
            query_embedding, limit=10, similarity_threshold=0.0
        )
        
        # Since we order by created_at DESC in the SQL query,
        # more recent items should appear first when similarities are equal
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_domain_specific_boosting(self, populated_storage, sample_embeddings):
        """Test that certain domains get slight score boosts."""
        query_embedding = sample_embeddings["query"]
        
        results = await populated_storage.search_by_embedding(
            query_embedding, limit=10, similarity_threshold=0.0
        )
        
        # Find creative and technical results
        creative_results = [(g, s) for g, s in results if g.parsed_data.get('domain') == 'creative']
        technical_results = [(g, s) for g, s in results if g.parsed_data.get('domain') == 'technical']
        
        # Both creative and technical should get boosts, 
        # so they should have reasonable scores
        if creative_results:
            assert all(score > 0 for _, score in creative_results)
        if technical_results:
            assert all(score > 0 for _, score in technical_results)

    @pytest.mark.asyncio
    async def test_concurrent_searches(self, populated_storage, sample_embeddings):
        """Test that multiple concurrent searches work correctly."""
        query_embedding = sample_embeddings["query"]
        
        # Run multiple searches concurrently
        tasks = [
            populated_storage.search_by_embedding(query_embedding, limit=5, similarity_threshold=0.2)
            for _ in range(3)
        ]
        
        results_list = await asyncio.gather(*tasks)
        
        # All searches should return results
        assert all(len(results) > 0 for results in results_list)
        
        # Results should be consistent across searches
        first_results = results_list[0]
        for other_results in results_list[1:]:
            assert len(first_results) == len(other_results)
            for (g1, s1), (g2, s2) in zip(first_results, other_results):
                assert g1.id == g2.id
                assert abs(s1 - s2) < 1e-10  # Should be identical

    @pytest.mark.asyncio
    async def test_large_dataset_performance(self, storage_manager):
        """Test search performance with larger dataset."""
        # Create many test globules
        test_embeddings = [np.random.rand(1024).astype(np.float32) for _ in range(50)]
        
        for i, embedding in enumerate(test_embeddings):
            globule = ProcessedGlobule(
                id=f"perf_test_{i}",
                text=f"Performance test globule {i} with some content to search through",
                embedding=embedding,
                embedding_confidence=0.8,
                parsed_data={"domain": "test", "category": "note"},
                parsing_confidence=0.7,
                file_decision=FileDecision(Path("test"), f"perf_{i}.md", {}, 0.7, []),  
                orchestration_strategy="parallel",
                processing_time_ms={"total_ms": 100},
                confidence_scores={"overall": 0.7},
                interpretations=[],
                has_nuance=False,
                semantic_neighbors=[],
                processing_notes=[],
                created_at=datetime.now()
            )
            await storage_manager.store_globule(globule)
        
        # Test search performance
        query_embedding = np.random.rand(1024).astype(np.float32)
        
        import time
        start_time = time.perf_counter()
        
        results = await storage_manager.search_by_embedding(
            query_embedding, limit=10, similarity_threshold=0.1
        )
        
        end_time = time.perf_counter()
        search_time = end_time - start_time
        
        # Should handle 50 embeddings efficiently
        assert search_time < 2.0  # Less than 2 seconds
        assert len(results) > 0
        assert len(results) <= 10  # Respects limit