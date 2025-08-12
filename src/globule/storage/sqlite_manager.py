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

from globule.core.interfaces import IStorageManager
from globule.core.models import ProcessedGlobuleV1, FileDecisionV1
from globule.config.settings import get_config


class SQLiteStorageManager(IStorageManager):
    """SQLite implementation of StorageManager"""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.config = get_config()
        if db_path is None:
            db_path = self.config.get_storage_dir() / "globules.db"
        self.db_path = db_path
        self._connection: Optional[aiosqlite.Connection] = None
        
        # FileManager is a private, internal component
        from globule.storage.file_manager import FileManager
        self._file_manager = FileManager()
    
    async def initialize(self, auto_reconcile: bool = False) -> None:
        """
        Initialize database schema and optionally perform file reconciliation.
        
        Args:
            auto_reconcile: If True, automatically reconcile files with database on startup
        """
        db = await self._get_connection()
        await self._create_schema(db)
            
        # Optional automatic reconciliation on startup
        if auto_reconcile:
            await self._perform_startup_reconciliation()
    
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
        
        # Create vector search virtual table using sqlite-vec
        await db.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS vss_globules USING vec0(
                embedding FLOAT[1024]
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
    
    async def _perform_startup_reconciliation(self) -> None:
        """
        Perform automatic file reconciliation on startup.
        
        This ensures the database reflects the actual state of files on disk,
        handling cases where users have moved, renamed, or organized files.
        """
        try:
            from globule.storage.file_manager import FileManager
            
            file_manager = FileManager()
            print("STARTUP: Performing automatic file reconciliation...")
            
            stats = await file_manager.reconcile_files_with_database(self)
            
            if stats['database_records_updated'] > 0:
                print(f"RECONCILIATION: Updated {stats['database_records_updated']} database records to match file locations")
            
            if stats['files_orphaned'] > 0:
                print(f"RECONCILIATION: Found {stats['files_orphaned']} orphaned files without UUIDs")
                
            print("STARTUP: File reconciliation complete")
            
        except Exception as e:
            print(f"STARTUP WARNING: File reconciliation failed: {e}")
            # Don't fail initialization due to reconciliation errors
    
    async def _get_connection(self) -> aiosqlite.Connection:
        """Get or create database connection"""
        if self._connection is None:
            self._connection = await aiosqlite.connect(str(self.db_path))
            await self._connection.enable_load_extension(True)
            
            # Load sqlite-vec extension
            try:
                import sqlite_vec
                await self._connection.execute("SELECT load_extension(?)", (sqlite_vec.loadable_path(),))
            except ImportError:
                # Fallback to old vec0 name for compatibility
                await self._connection.load_extension("vec0")
            # Enable foreign keys and set performance optimizations
            await self._connection.execute("PRAGMA foreign_keys = ON")
            await self._connection.execute("PRAGMA journal_mode = WAL")
            await self._connection.execute("PRAGMA synchronous = NORMAL")
        return self._connection
    
    async def store_globule(self, globule: ProcessedGlobuleV1) -> str:
        """
        Store a processed globule using the transactional Outbox Pattern.
        
        This implementation ensures true atomicity:
        1. Determine final file path before any operations
        2. Create file in temporary location
        3. Execute database transaction with final file path
        4. Commit file to final location only after DB success
        5. Clean up temp file on any failure
        
        Args:
            globule: The processed globule to store
            
        Returns:
            The globule ID
            
        Raises:
            Exception: If any part of the atomic operation fails
        """
        if globule.id is None:
            globule.id = str(uuid.uuid4())
        
        # OUTBOX PATTERN STEP 1: Determine final file path before any operations
        final_file_path = self._file_manager.determine_path(globule)
        
        # Update globule's file_decision to reflect the determined path
        relative_path = final_file_path.relative_to(self._file_manager.base_path)
        globule.file_decision = FileDecisionV1(
            semantic_path=relative_path.parent,
            filename=relative_path.name,
            metadata={"outbox_pattern": True, "atomic_storage": True},
            confidence=1.0,  # High confidence as we determined the path
            alternative_paths=[]
        )
        
        # OUTBOX PATTERN STEP 2: Create file in temporary location
        temp_file_path = self._file_manager.save_to_temp(globule)
        
        try:
            # OUTBOX PATTERN STEP 3: Database transaction with final path
            # Serialize complex fields to JSON
            embedding_blob = None
            if globule.embedding is not None:
                # Normalize the embedding for consistent similarity calculations
                normalized_embedding = self._normalize_vector(globule.embedding.astype(np.float32))
                embedding_blob = normalized_embedding.tobytes()
            
            parsed_data_json = json.dumps(globule.parsed_data)
            confidence_scores_json = json.dumps(globule.confidence_scores)
            processing_time_json = json.dumps(globule.processing_time_ms)
            semantic_neighbors_json = json.dumps(globule.semantic_neighbors)
            processing_notes_json = json.dumps(globule.processing_notes)
            
            # Use the determined file path for database storage
            file_path = str(globule.file_decision.semantic_path / globule.file_decision.filename)
            
            db = await self._get_connection()
            
            # This transaction block guarantees all-or-nothing database operations
            try:
                await db.execute("BEGIN TRANSACTION")
                
                # Insert into the main table
                cursor = await db.execute("""
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
                
                globule_rowid = cursor.lastrowid
                
                # Insert into the vector search index
                if embedding_blob is not None:
                    await db.execute("""
                        INSERT OR REPLACE INTO vss_globules (rowid, embedding)
                        VALUES (?, ?)
                    """, (globule_rowid, embedding_blob))
                
                await db.commit()
            except Exception as db_error:
                await db.rollback()
                raise db_error
            
            # OUTBOX PATTERN STEP 4: Database transaction succeeded, commit file
            self._file_manager.commit_file(temp_file_path, final_file_path)
            
            return globule.id
            
        except Exception as e:
            # OUTBOX PATTERN STEP 5: Any failure - clean up temp file
            self._file_manager.cleanup_temp(temp_file_path)
            raise Exception(f"Atomic storage operation failed: {e}")
    
    async def update_globule(self, globule: ProcessedGlobuleV1) -> bool:
        """
        Update an existing globule atomically.
        
        Args:
            globule: ProcessedGlobule with existing ID to update
            
        Returns:
            True if update succeeded, False if globule doesn't exist
        """
        if globule.id is None:
            raise ValueError("Cannot update globule without an ID")
        
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
        
        # This transaction block guarantees all-or-nothing update.
        async with db.transaction():
            # Update the main table
            cursor = await db.execute("""
                UPDATE globules SET
                    text = ?, embedding = ?, embedding_confidence = ?, parsed_data = ?,
                    parsing_confidence = ?, file_path = ?, orchestration_strategy = ?,
                    confidence_scores = ?, processing_time_ms = ?, semantic_neighbors = ?,
                    processing_notes = ?, modified_at = ?
                WHERE id = ?
            """, (
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
                globule.modified_at.isoformat(),
                globule.id
            ))
            
            # Check if the update affected any rows
            if cursor.rowcount == 0:
                return False
            
            # Get the rowid for the vector table update
            async with db.execute("SELECT rowid FROM globules WHERE id = ?", (globule.id,)) as rowid_cursor:
                row = await rowid_cursor.fetchone()
                if not row:
                    return False
                globule_rowid = row[0]
            
            # Update the vector search index
            if embedding_blob is not None:
                await db.execute("""
                    INSERT OR REPLACE INTO vss_globules (rowid, embedding)
                    VALUES (?, ?)
                """, (globule_rowid, embedding_blob))
            else:
                # Remove from vector search if no embedding
                await db.execute("DELETE FROM vss_globules WHERE rowid = ?", (globule_rowid,))
        
        return True
    
    async def delete_globule(self, globule_id: str) -> bool:
        """
        Delete a globule and its vector embedding atomically.
        
        Args:
            globule_id: The ID of the globule to delete
            
        Returns:
            True if globule was deleted, False if it didn't exist
        """
        db = await self._get_connection()
        
        # This transaction block guarantees all-or-nothing deletion.
        async with db.transaction():
            # First get the rowid before deletion
            async with db.execute("SELECT rowid FROM globules WHERE id = ?", (globule_id,)) as cursor:
                row = await cursor.fetchone()
                if not row:
                    return False
                globule_rowid = row[0]
            
            # Delete from vector search table first
            await db.execute("DELETE FROM vss_globules WHERE rowid = ?", (globule_rowid,))
            
            # Delete from main table
            cursor = await db.execute("DELETE FROM globules WHERE id = ?", (globule_id,))
            
            # Check if the deletion affected any rows
            return cursor.rowcount > 0
    
    async def get_globule(self, globule_id: str) -> Optional[ProcessedGlobuleV1]:
        """Retrieve a globule by ID"""
        db = await self._get_connection()
        async with db.execute(
            "SELECT * FROM globules WHERE id = ?", (globule_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return self._row_to_globule(row)
    
    async def get_recent_globules(self, limit: int = 100) -> List[ProcessedGlobuleV1]:
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
        similarity_threshold: float = 0.1,
        min_embedding_confidence: Optional[float] = None
        ) -> List[Tuple[ProcessedGlobuleV1, float]]:
        """
        Finds semantically similar globules using a single, efficient query.
        This is the correct, non-looping implementation.
        
        Args:
            query_vector: The embedding vector to search for
            limit: Maximum number of results to return
            similarity_threshold: Minimum similarity score (0.0 to 1.0)
            min_embedding_confidence: Optional minimum embedding confidence filter
            
        Returns:
            List of (ProcessedGlobule, similarity_score) tuples, sorted by similarity
        """
        if query_vector is None:
            return []
        
        # Normalize the query vector to match stored embeddings
        normalized_query = self._normalize_vector(query_vector.astype(np.float32))
            
        db = await self._get_connection()
        
        # Step 1: Get the rowids of the nearest neighbors from the vector index.
        # This is a fast, native C operation.
        
        # First, check if we have any vectors in the table
        async with db.execute("SELECT COUNT(*) FROM vss_globules") as cursor:
            count_result = await cursor.fetchone()
            if count_result[0] == 0:
                return []  # No vectors in database
        
        async with db.execute("""
            SELECT rowid, distance
            FROM vss_globules
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?
        """, (normalized_query.tobytes(), limit)) as cursor:
            rows = await cursor.fetchall()
            if not rows:
                return []
        
        neighbor_ids = [row[0] for row in rows]
        distances = {row[0]: row[1] for row in rows}
        
        # Step 2: Fetch all the corresponding globules in a SINGLE query.
        # We build a query with the correct number of placeholders.
        placeholders = ','.join('?' for _ in neighbor_ids)
        
        if min_embedding_confidence is not None:
            sql = f"SELECT rowid, * FROM globules WHERE rowid IN ({placeholders}) AND embedding_confidence >= ?"
            params = neighbor_ids + [min_embedding_confidence]
        else:
            sql = f"SELECT rowid, * FROM globules WHERE rowid IN ({placeholders})"
            params = neighbor_ids
        
        async with db.execute(sql, params) as cursor:
            globule_rows = await cursor.fetchall()
        
        # The database does not guarantee the order of IN clauses,
        # so we re-order the results in Python to match the similarity ranking.
        # row[0] is rowid, row[1:] contains the globule data
        globule_map = {row[0]: self._row_to_globule(row[1:]) for row in globule_rows}
        
        results = []
        for neighbor_id in neighbor_ids:
            if neighbor_id in globule_map:
                globule = globule_map[neighbor_id]
                distance = distances[neighbor_id]
                # Convert distance to similarity score (0-1 range)
                # For cosine distance, similarity = 1 - distance/2 (since cosine distance is in range [0,2])
                # For euclidean distance, we use a different formula
                # Since we don't know the exact distance metric, use a robust conversion
                similarity = 1.0 / (1.0 + distance)  # This works for any positive distance
                
                if similarity >= similarity_threshold:
                    results.append((globule, similarity))
        
        return results


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

    
    def _row_to_globule(self, row: sqlite3.Row) -> ProcessedGlobuleV1:
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
            file_decision = FileDecisionV1(
                semantic_path=file_path.parent,
                filename=file_path.name,
                metadata={},
                confidence=0.8,  # Default confidence
                alternative_paths=[]
            )
        
        return ProcessedGlobuleV1(
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

    async def hybrid_search(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        limit: int = 20,
        similarity_threshold: float = 0.5
    ) -> List[Tuple[ProcessedGlobuleV1, float]]:
        """
        Hybrid search combining text and embedding similarity.
        
        Args:
            query_text: Text query for keyword matching
            query_embedding: Embedding vector for semantic search
            limit: Maximum results to return
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of (ProcessedGlobule, combined_score) tuples
        """
        # Get semantic results
        semantic_results = await self.search_by_embedding(
            query_embedding, limit=limit * 2, similarity_threshold=similarity_threshold
        )
        
        # Get text results
        text_results = await self._search_by_text_keywords(query_text, limit=limit * 2)
        
        # Fuse results
        fused_results = self._fuse_search_results(semantic_results, text_results)
        
        return fused_results[:limit]

    async def _search_by_text_keywords(
        self,
        query: str,
        limit: int = 20
    ) -> List[Tuple[ProcessedGlobuleV1, float]]:
        """
        Search for globules containing specific keywords.
        
        Args:
            query: Text query with keywords
            limit: Maximum results to return
            
        Returns:
            List of (ProcessedGlobule, relevance_score) tuples
        """
        db = await self._get_connection()
        
        # Simple keyword search using LIKE operator
        keywords = query.lower().split()
        where_clauses = []
        params = []
        
        for keyword in keywords:
            where_clauses.append("LOWER(text) LIKE ?")
            params.append(f"%{keyword}%")
        
        where_sql = " OR ".join(where_clauses)
        
        async with db.execute(f"""
            SELECT id, text, embedding, embedding_confidence, parsed_data,
                   parsing_confidence, file_path, orchestration_strategy,
                   confidence_scores, processing_time_ms, semantic_neighbors,
                   processing_notes, created_at, modified_at
            FROM globules
            WHERE {where_sql}
            ORDER BY embedding_confidence DESC
            LIMIT ?
        """, params + [limit]) as cursor:
            rows = await cursor.fetchall()
        
        results = []
        for row in rows:
            globule = self._row_to_globule(row)
            
            # Calculate simple relevance score based on keyword matches
            text_lower = globule.text.lower()
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            relevance = min(1.0, matches / len(keywords))
            
            results.append((globule, relevance))
        
        return results

    def _fuse_search_results(
        self,
        semantic_results: List[Tuple[ProcessedGlobuleV1, float]],
        text_results: List[Tuple[ProcessedGlobuleV1, float]]
    ) -> List[Tuple[ProcessedGlobuleV1, float]]:
        """
        Fuse semantic and text search results with intelligent scoring.
        
        Args:
            semantic_results: Results from embedding search
            text_results: Results from text keyword search
            
        Returns:
            Combined and deduplicated results with fused scores
        """
        # Create lookup for efficient merging
        semantic_scores = {globule.id: score for globule, score in semantic_results}
        text_scores = {globule.id: score for globule, score in text_results}
        
        # Collect all unique globules
        all_globules = {}
        for globule, _ in semantic_results:
            all_globules[globule.id] = globule
        for globule, _ in text_results:
            all_globules[globule.id] = globule
        
        # Calculate combined scores
        fused_results = []
        for globule_id, globule in all_globules.items():
            semantic_score = semantic_scores.get(globule_id, 0.0)
            text_score = text_scores.get(globule_id, 0.0)
            
            # Weighted combination: 70% semantic, 30% text
            combined_score = 0.7 * semantic_score + 0.3 * text_score
            
            # Boost if found in both searches
            if semantic_score > 0 and text_score > 0:
                combined_score *= 1.2  # 20% boost for multi-match
            
            combined_score = min(1.0, combined_score)  # Cap at 1.0
            fused_results.append((globule, combined_score))
        
        # Sort by combined score
        fused_results.sort(key=lambda x: x[1], reverse=True)
        return fused_results