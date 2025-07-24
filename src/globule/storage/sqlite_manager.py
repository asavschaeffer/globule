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
        """Find semantically similar globules (basic implementation for Phase 1)"""
        # For Phase 1, we'll implement a simple linear search
        # Phase 2 will add proper vector search with sqlite-vec
        
        db = await self._get_connection()
        async with db.execute(
            "SELECT * FROM globules WHERE embedding IS NOT NULL"
        ) as cursor:
            rows = await cursor.fetchall()
        
        results = []
        for row in rows:
            globule = self._row_to_globule(row)
            if globule.embedding is not None:
                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_vector, globule.embedding)
                if similarity >= similarity_threshold:
                    results.append((globule, similarity))
        
        # Sort by similarity and limit results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
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