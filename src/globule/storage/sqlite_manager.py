"""
SQLite-based storage manager for Globule.

Implements the core storage interface using SQLite with basic schema.
Vector search capabilities will be added in Phase 2.
"""

import json
import sqlite3
import asyncio
import aiosqlite
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import uuid
import numpy as np

from globule.core.interfaces import StorageManager
from globule.core.models import ProcessedGlobule, FileDecision
from globule.config.settings import get_config


class SQLiteStorageManager(StorageManager):
    """SQLite implementation of StorageManager"""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.config = get_config()
        if db_path is None:
            db_path = self.config.get_storage_dir() / "globules.db"
        self.db_path = db_path
        self._connection: Optional[aiosqlite.Connection] = None
    
    async def initialize(self) -> None:
        """Initialize database schema"""
        async with aiosqlite.connect(str(self.db_path)) as db:
            await self._create_schema(db)
    
    async def _create_schema(self, db: aiosqlite.Connection) -> None:
        """Create database tables"""
        await db.execute("""
            CREATE TABLE IF NOT EXISTS globules (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                embedding BLOB,
                embedding_confidence REAL DEFAULT 0.0,
                parsed_data TEXT,  -- JSON
                parsing_confidence REAL DEFAULT 0.0,
                file_path TEXT,
                orchestration_strategy TEXT DEFAULT 'parallel',
                confidence_scores TEXT,  -- JSON
                processing_time_ms TEXT,  -- JSON
                semantic_neighbors TEXT,  -- JSON array of IDs
                processing_notes TEXT,   -- JSON array
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                modified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for performance
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_globules_created_at 
            ON globules(created_at DESC)
        """)
        
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_globules_text 
            ON globules(text)
        """)
        
        await db.commit()
    
    async def _get_connection(self) -> aiosqlite.Connection:
        """Get or create database connection"""
        if self._connection is None:
            self._connection = await aiosqlite.connect(str(self.db_path))
            # Enable foreign keys and set performance optimizations
            await self._connection.execute("PRAGMA foreign_keys = ON")
            await self._connection.execute("PRAGMA journal_mode = WAL")
            await self._connection.execute("PRAGMA synchronous = NORMAL")
        return self._connection
    
    async def store_globule(self, globule: ProcessedGlobule) -> str:
        """Store a processed globule and return its ID"""
        if globule.id is None:
            globule.id = str(uuid.uuid4())
        
        # Serialize complex fields to JSON
        embedding_blob = None
        if globule.embedding is not None:
            embedding_blob = globule.embedding.astype(np.float32).tobytes()
        
        parsed_data_json = json.dumps(globule.parsed_data)
        confidence_scores_json = json.dumps(globule.confidence_scores)
        processing_time_json = json.dumps(globule.processing_time_ms)
        semantic_neighbors_json = json.dumps(globule.semantic_neighbors)
        processing_notes_json = json.dumps(globule.processing_notes)
        
        # Store file path from file decision
        file_path = None
        if globule.file_decision:
            file_path = str(globule.file_decision.semantic_path / globule.file_decision.filename)
        
        db = await self._get_connection()
        await db.execute("""
            INSERT OR REPLACE INTO globules (
                id, text, embedding, embedding_confidence, parsed_data,
                parsing_confidence, file_path, orchestration_strategy,
                confidence_scores, processing_time_ms, semantic_neighbors,
                processing_notes, created_at, modified_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            globule.id,
            globule.text,
            embedding_blob,
            globule.embedding_confidence,
            parsed_data_json,
            globule.parsing_confidence,
            file_path,
            globule.orchestration_strategy,
            confidence_scores_json,
            processing_time_json,
            semantic_neighbors_json,
            processing_notes_json,
            globule.created_at.isoformat(),
            globule.modified_at.isoformat()
        ))
        await db.commit()
        
        return globule.id
    
    async def get_globule(self, globule_id: str) -> Optional[ProcessedGlobule]:
        """Retrieve a globule by ID"""
        db = await self._get_connection()
        async with db.execute(
            "SELECT * FROM globules WHERE id = ?", (globule_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return self._row_to_globule(row)
    
    async def get_recent_globules(self, limit: int = 100) -> List[ProcessedGlobule]:
        """Get recent globules ordered by creation time"""
        db = await self._get_connection()
        async with db.execute(
            "SELECT * FROM globules ORDER BY created_at DESC LIMIT ?",
            (limit,)
        ) as cursor:
            rows = await cursor.fetchall()
            return [self._row_to_globule(row) for row in rows]
    
    async def search_by_embedding(
        self, 
        query_vector: np.ndarray, 
        limit: int = 50,
        similarity_threshold: float = 0.5
    ) -> List[Tuple[ProcessedGlobule, float]]:
        """
        Find semantically similar globules using optimized vector search.
        
        Phase 2 Implementation: Enhanced similarity calculations with multiple algorithms,
        intelligent filtering, and performance optimizations.
        
        Args:
            query_vector: The embedding vector to search for
            limit: Maximum number of results to return
            similarity_threshold: Minimum similarity score (0.0 to 1.0)
            
        Returns:
            List of (ProcessedGlobule, similarity_score) tuples, sorted by similarity
        """
        if query_vector is None:
            return []
            
        # Ensure query vector is normalized for consistent similarity calculations
        query_vector = self._normalize_vector(query_vector)
        
        db = await self._get_connection()
        
        # Phase 2: Enhanced query with metadata filtering
        async with db.execute("""
            SELECT * FROM globules 
            WHERE embedding IS NOT NULL 
            AND embedding_confidence > 0.3
            ORDER BY created_at DESC
        """) as cursor:
            rows = await cursor.fetchall()
        
        if not rows:
            return []
        
        # Phase 2: Batch similarity calculation for performance
        results = await self._batch_similarity_search(query_vector, rows, similarity_threshold)
        
        # Advanced filtering and ranking
        enhanced_results = self._enhance_search_results(results, query_vector)
        
        # Sort by enhanced similarity score and limit
        enhanced_results.sort(key=lambda x: x[1], reverse=True)
        return enhanced_results[:limit]

    async def _batch_similarity_search(
        self, 
        query_vector: np.ndarray, 
        rows: List[sqlite3.Row], 
        threshold: float
    ) -> List[Tuple[ProcessedGlobule, float]]:
        """
        Perform batch similarity calculations for improved performance.
        
        Phase 2 enhancement: Vectorized operations using numpy for speed.
        """
        results = []
        embeddings_batch = []
        globules_batch = []
        
        # Build batch of embeddings and globules
        for row in rows:
            globule = self._row_to_globule(row)
            if globule.embedding is not None:
                # Normalize embedding for consistent comparison
                normalized_embedding = self._normalize_vector(globule.embedding)
                embeddings_batch.append(normalized_embedding)
                globules_batch.append(globule)
        
        if not embeddings_batch:
            return results
        
        # Vectorized similarity calculation (much faster than loop)
        embeddings_matrix = np.vstack(embeddings_batch)
        similarities = np.dot(embeddings_matrix, query_vector)
        
        # Filter by threshold and create results
        for i, similarity in enumerate(similarities):
            if similarity >= threshold:
                results.append((globules_batch[i], float(similarity)))
        
        return results

    def _enhance_search_results(
        self, 
        results: List[Tuple[ProcessedGlobule, float]], 
        query_vector: np.ndarray
    ) -> List[Tuple[ProcessedGlobule, float]]:
        """
        Phase 2: Enhance search results with additional ranking factors.
        
        Considers:
        - Semantic similarity (primary)
        - Content quality (parsing confidence)
        - Recency boost for newer content
        - Domain-specific adjustments
        """
        enhanced_results = []
        
        for globule, base_similarity in results:
            # Start with base similarity
            enhanced_score = base_similarity
            
            # Quality boost: Higher parsing confidence = slight boost
            if globule.parsing_confidence > 0.8:
                enhanced_score += 0.02
            elif globule.parsing_confidence < 0.5:
                enhanced_score -= 0.01
            
            # Recency boost: Newer content gets slight preference
            if hasattr(globule, 'created_at') and globule.created_at:
                days_old = (datetime.now() - globule.created_at).days
                if days_old < 7:  # Less than a week old
                    enhanced_score += 0.01
                elif days_old > 90:  # Older than 3 months
                    enhanced_score -= 0.005
            
            # Domain coherence: Boost results from similar domains
            if globule.parsed_data and isinstance(globule.parsed_data, dict):
                domain = globule.parsed_data.get('domain', '')
                if domain in ['creative', 'technical']:  # High-value domains
                    enhanced_score += 0.005
            
            # Ensure score stays within reasonable bounds
            enhanced_score = max(0.0, min(1.0, enhanced_score))
            enhanced_results.append((globule, enhanced_score))
        
        return enhanced_results

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        Normalize vector for consistent similarity calculations.
        
        Phase 2: Proper L2 normalization for accurate cosine similarity.
        """
        if vector is None:
            return None
            
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    async def search_by_text_and_embedding(
        self,
        text_query: str,
        embedding_query: np.ndarray,
        limit: int = 20,
        similarity_threshold: float = 0.4
    ) -> List[Tuple[ProcessedGlobule, float]]:
        """
        Phase 2: Hybrid search combining text matching and semantic similarity.
        
        This provides more comprehensive search results by combining:
        - Semantic similarity (embedding-based)
        - Text matching (keyword-based)
        - Intelligent result fusion
        """
        # Get semantic results
        semantic_results = await self.search_by_embedding(
            embedding_query, limit * 2, similarity_threshold
        )
        
        # Get text-based results
        text_results = await self._search_by_text_keywords(text_query, limit)
        
        # Fuse and rank results
        fused_results = self._fuse_search_results(semantic_results, text_results)
        
        return fused_results[:limit]

    async def _search_by_text_keywords(
        self, 
        text_query: str, 
        limit: int
    ) -> List[Tuple[ProcessedGlobule, float]]:
        """Simple text-based search for hybrid functionality."""
        db = await self._get_connection()
        
        # Simple text matching (Phase 2 could use FTS if needed)
        keywords = text_query.lower().split()
        
        async with db.execute(
            "SELECT * FROM globules WHERE LOWER(text) LIKE ?",
            (f"%{' '.join(keywords)}%",)
        ) as cursor:
            rows = await cursor.fetchall()
        
        results = []
        for row in rows:
            globule = self._row_to_globule(row)
            # Simple relevance scoring based on keyword matches
            text_lower = globule.text.lower()
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            relevance = matches / len(keywords) if keywords else 0
            results.append((globule, relevance))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def _fuse_search_results(
        self,
        semantic_results: List[Tuple[ProcessedGlobule, float]],
        text_results: List[Tuple[ProcessedGlobule, float]]
    ) -> List[Tuple[ProcessedGlobule, float]]:
        """
        Intelligently fuse semantic and text-based search results.
        
        Phase 2: Advanced result fusion with score normalization and deduplication.
        """
        # Create a map for deduplication and score fusion
        result_map = {}
        
        # Add semantic results (weighted higher)
        for globule, score in semantic_results:
            result_map[globule.id] = (globule, score * 0.7)  # 70% weight
        
        # Add text results (weighted lower, but combined if duplicate)
        for globule, score in text_results:
            if globule.id in result_map:
                # Combine scores for items found in both searches
                existing_globule, existing_score = result_map[globule.id]
                combined_score = existing_score + (score * 0.3)  # Add 30% of text score
                result_map[globule.id] = (existing_globule, min(1.0, combined_score))
            else:
                result_map[globule.id] = (globule, score * 0.3)  # 30% weight for text-only
        
        # Convert back to list and sort
        fused_results = list(result_map.values())
        fused_results.sort(key=lambda x: x[1], reverse=True)
        
        return fused_results
    
    def _row_to_globule(self, row: sqlite3.Row) -> ProcessedGlobule:
        """Convert database row to ProcessedGlobule"""
        # Deserialize embedding
        embedding = None
        if row[2] is not None:  # embedding blob
            embedding = np.frombuffer(row[2], dtype=np.float32)
        
        # Deserialize JSON fields
        parsed_data = json.loads(row[4]) if row[4] else {}
        confidence_scores = json.loads(row[8]) if row[8] else {}
        processing_time_ms = json.loads(row[9]) if row[9] else {}
        semantic_neighbors = json.loads(row[10]) if row[10] else []
        processing_notes = json.loads(row[11]) if row[11] else []
        
        # Create file decision if file path exists
        file_decision = None
        if row[6]:  # file_path
            file_path = Path(row[6])
            file_decision = FileDecision(
                semantic_path=file_path.parent,
                filename=file_path.name,
                metadata={},
                confidence=0.8,  # Default confidence
                alternative_paths=[]
            )
        
        return ProcessedGlobule(
            id=row[0],
            text=row[1],
            embedding=embedding,
            embedding_confidence=row[3],
            parsed_data=parsed_data,
            parsing_confidence=row[5],
            file_decision=file_decision,
            orchestration_strategy=row[7],
            confidence_scores=confidence_scores,
            processing_time_ms=processing_time_ms,
            semantic_neighbors=semantic_neighbors,
            processing_notes=processing_notes,
            created_at=datetime.fromisoformat(row[12]),
            modified_at=datetime.fromisoformat(row[13])
        )
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    async def close(self) -> None:
        """Close database connection"""
        if self._connection:
            await self._connection.close()
            self._connection = None