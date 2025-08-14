"""
SQLite Storage Adapter for Globule.

This adapter implements the IStorageManager interface using SQLite as the backend,
following the adapter pattern to isolate external dependencies. It uses aiosqlite
for non-blocking I/O operations as recommended for async applications.
"""

import json
import aiosqlite
import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import uuid
from uuid import UUID
import numpy as np

from globule.core.interfaces import IStorageManager
from globule.core.models import ProcessedGlobuleV1, FileDecisionV1
from globule.core.errors import StorageError
from globule.config.settings import get_config


class SqliteStorageAdapter(IStorageManager):
    """
    SQLite adapter implementing the IStorageManager interface.
    
    This adapter isolates SQLite-specific logic behind the abstract interface,
    enabling easy swapping of storage backends in the future. Uses aiosqlite
    for async operations to avoid blocking the event loop.
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize the SQLite storage adapter.
        
        Args:
            db_path: Optional path to the SQLite database file.
                    If None, uses default path from config.
        """
        self.config = get_config()
        if db_path is None:
            db_path = self.config.get_storage_dir() / "globules.db"
        self.db_path = db_path
        self._connection: Optional[aiosqlite.Connection] = None
        self._db_lock = asyncio.Lock()
        
        # FileManager is a private, internal component
        # For Phase 2, we'll mock this dependency to focus on the adapter pattern
        self._file_manager = self._create_mock_file_manager()
    
    async def initialize(self, auto_reconcile: bool = False) -> None:
        """
        Initialize database schema and optionally perform file reconciliation.
        
        Args:
            auto_reconcile: If True, automatically reconcile files with database on startup
        """
        try:
            db = await self._get_connection()
            await self._create_schema(db)
                
            # Optional automatic reconciliation on startup
            if auto_reconcile:
                await self._perform_startup_reconciliation()
        except Exception as e:
            raise StorageError(f"Failed to initialize SQLite storage: {e}")
    
    async def _create_schema(self, db: aiosqlite.Connection) -> None:
        """Create database tables"""
        try:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS globules (
                    globule_id TEXT PRIMARY KEY,
                    contract_version TEXT DEFAULT '1.0',
                    processed_timestamp TIMESTAMP,
                    raw_text TEXT NOT NULL,
                    embedding BLOB,
                    parsed_data TEXT,  -- JSON
                    nuances TEXT,  -- JSON
                    file_path TEXT,
                    processing_time_ms REAL,
                    provider_metadata TEXT,  -- JSON
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Vector search will be added in a future phase
            # await db.execute("""
            #     CREATE VIRTUAL TABLE IF NOT EXISTS vss_globules USING vec0(
            #         embedding FLOAT[1024]
            #     )
            # """)
            
            # Create indexes for performance
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_globules_created_at 
                ON globules(created_at DESC)
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_globules_text 
                ON globules(raw_text)
            """)
            
            await db.commit()
        except Exception as e:
            raise StorageError(f"Failed to create database schema: {e}")
    
    async def _perform_startup_reconciliation(self) -> None:
        """
        Perform automatic file reconciliation on startup.
        
        This is a placeholder for Phase 2. In a real implementation,
        this would ensure the database reflects the actual state of files on disk.
        """
        try:
            print("STARTUP: File reconciliation skipped in Phase 2 adapter")
            # TODO: Implement proper file reconciliation in future phases
            
        except Exception as e:
            print(f"STARTUP WARNING: File reconciliation failed: {e}")
            # Don't fail initialization due to reconciliation errors
    
    async def _get_connection(self) -> aiosqlite.Connection:
        """Get or create database connection"""
        if self._connection is None:
            try:
                self._connection = await aiosqlite.connect(str(self.db_path))
                self._connection.row_factory = aiosqlite.Row  # Enable dict-like row access
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
            except Exception as e:
                raise StorageError(f"Failed to connect to SQLite database: {e}")
        return self._connection
    
    # Implement IStorageManager abstract methods
    
    def save(self, globule: ProcessedGlobuleV1) -> None:
        """
        Synchronous wrapper for store_globule.
        Required by the IStorageManager interface.
        """
        try:
            # Check if we're in an async context
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, can't use asyncio.run()
                raise StorageError("Cannot save globule synchronously from within an async context. Use store_globule() directly.")
            except RuntimeError:
                # No running loop, we can use asyncio.run()
                asyncio.run(self.store_globule(globule))
        except Exception as e:
            if isinstance(e, StorageError):
                raise
            raise StorageError(f"Failed to save globule: {e}")
    
    def get(self, globule_id: UUID) -> ProcessedGlobuleV1:
        """
        Synchronous wrapper for get_globule.
        Required by the IStorageManager interface.
        """
        try:
            # Check if we're in an async context
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, can't use asyncio.run()
                raise StorageError("Cannot get globule synchronously from within an async context. Use get_globule() directly.")
            except RuntimeError:
                # No running loop, we can use asyncio.run()
                result = asyncio.run(self.get_globule(str(globule_id)))
                
            if result is None:
                raise StorageError(f"Globule {globule_id} not found")
            return result
        except Exception as e:
            if isinstance(e, StorageError):
                raise
            raise StorageError(f"Failed to get globule: {e}")
    
    async def search(self, query: str, limit: int = 10) -> List[ProcessedGlobuleV1]:
        """
        Search for globules using natural language query.
        
        This method implements the search functionality using SQLite's
        text search capabilities. Future versions can add vector search.
        """
        try:
            db = await self._get_connection()
            
            # Simple LIKE search for now - can be enhanced with FTS or vector search
            async with db.execute("""
                SELECT globule_id, contract_version, processed_timestamp, raw_text,
                       embedding, parsed_data, nuances, file_path,
                       processing_time_ms, provider_metadata, created_at
                FROM globules 
                WHERE raw_text LIKE ? 
                ORDER BY created_at DESC 
                LIMIT ?
            """, (f"%{query}%", limit)) as cursor:
                rows = await cursor.fetchall()
                
                results = []
                for row in rows:
                    globule = self._row_to_globule(row)
                    results.append(globule)
                
                return results
                
        except Exception as e:
            raise StorageError(f"Search failed: {e}")

    async def execute_sql(self, query: str, query_name: str = "Query") -> Dict[str, Any]:
        """
        Execute SQL query against the database.
        
        This method implements SQL execution with proper safety checks
        and error handling, isolating the SQLite-specific logic.
        """
        try:
            db = await self._get_connection()
            
            # Validate SQL safety (basic check)
            dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'TRUNCATE', 'ALTER']
            if any(keyword in query.upper() for keyword in dangerous_keywords):
                raise StorageError("Potentially dangerous SQL detected")
            
            async with db.execute(query) as cursor:
                results = await cursor.fetchall()
                
                # Convert to list of dicts
                results_list = [dict(row) for row in results] if results else []
                headers = [desc[0] for desc in cursor.description] if cursor.description else []
                
                return {
                    "type": "sql_results",
                    "query": query,
                    "query_name": query_name,
                    "results": results_list,
                    "headers": headers,
                    "count": len(results_list)
                }
                
        except StorageError:
            # Re-raise storage errors as-is
            raise
        except Exception as e:
            raise StorageError(f"SQL query execution failed: {e}")
    
    # Additional async methods for better performance
    
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
            StorageError: If any part of the atomic operation fails
        """
        try:
            # Use globule_id as the identifier
            globule_id = str(globule.globule_id)
            
            # OUTBOX PATTERN STEP 1: Determine final file path before any operations
            final_file_path = self._file_manager.determine_path(globule)
            
            # Update globule's file_decision to reflect the determined path
            relative_path = final_file_path.relative_to(self._file_manager.base_path)
            globule.file_decision = FileDecisionV1(
                semantic_path=str(relative_path.parent),
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
                    # Store embedding as-is for Phase 2 (normalization can be added later)
                    embedding_array = np.array(globule.embedding, dtype=np.float32)
                    embedding_blob = embedding_array.tobytes()
                
                parsed_data_json = json.dumps(globule.parsed_data)
                provider_metadata_json = json.dumps(globule.provider_metadata)
                nuances_json = json.dumps(globule.nuances.model_dump())
                
                # Use the determined file path for database storage
                file_path = str(Path(globule.file_decision.semantic_path) / globule.file_decision.filename)
                
                db = await self._get_connection()
                
                # This transaction block guarantees all-or-nothing database operations
                async with self._db_lock:
                    try:
                        await db.execute("BEGIN TRANSACTION")
                        
                        # Insert into the main table
                        cursor = await db.execute("""
                            INSERT OR REPLACE INTO globules (
                                globule_id, contract_version, processed_timestamp, raw_text,
                                embedding, parsed_data, nuances, file_path,
                                processing_time_ms, provider_metadata, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            str(globule.globule_id),
                            globule.contract_version,
                            globule.processed_timestamp.isoformat(),
                            globule.original_globule.raw_text,
                            embedding_blob,
                            parsed_data_json,
                            nuances_json,
                            file_path,
                            globule.processing_time_ms,
                            provider_metadata_json,
                            datetime.now().isoformat()
                        ))
                        
                        globule_rowid = cursor.lastrowid
                        
                        # Vector search will be added in a future phase
                        # if embedding_blob is not None:
                        #     await db.execute("""
                        #         INSERT OR REPLACE INTO vss_globules (rowid, embedding)
                        #         VALUES (?, ?)
                        #     """, (globule_rowid, embedding_blob))
                        
                        await db.commit()
                    except Exception as db_error:
                        await db.rollback()
                        raise StorageError(f"Database transaction failed: {db_error}")
                
                # OUTBOX PATTERN STEP 4: Database transaction succeeded, commit file
                self._file_manager.commit_file(temp_file_path, final_file_path)
                
                return str(globule.globule_id)
                
            except Exception as e:
                # OUTBOX PATTERN STEP 5: Any failure - clean up temp file
                self._file_manager.cleanup_temp(temp_file_path)
                raise StorageError(f"Atomic storage operation failed: {e}")
                
        except Exception as e:
            if isinstance(e, StorageError):
                raise
            raise StorageError(f"Failed to store globule: {e}")
    
    async def get_globule(self, globule_id: str) -> Optional[ProcessedGlobuleV1]:
        """Retrieve a globule by ID"""
        try:
            db = await self._get_connection()
            async with db.execute("""
                SELECT globule_id, contract_version, processed_timestamp, raw_text,
                       embedding, parsed_data, nuances, file_path,
                       processing_time_ms, provider_metadata, created_at
                FROM globules WHERE globule_id = ?
            """, (globule_id,)) as cursor:
                row = await cursor.fetchone()
                if row is None:
                    return None
                return self._row_to_globule(row)
        except Exception as e:
            raise StorageError(f"Failed to get globule {globule_id}: {e}")
    
    async def close(self) -> None:
        """Close database connection"""
        if self._connection:
            await self._connection.close()
            self._connection = None
    
    # Helper methods
    
    def _create_mock_file_manager(self):
        """Create a mock file manager for Phase 2 adapter testing."""
        class MockFileManager:
            def __init__(self):
                self.base_path = Path("/tmp/globule_storage")
            
            def determine_path(self, globule: ProcessedGlobuleV1) -> Path:
                """Mock path determination."""
                return self.base_path / "determined_path" / f"{globule.globule_id}.md"
            
            def save_to_temp(self, globule: ProcessedGlobuleV1) -> Path:
                """Mock temp file creation."""
                return Path(f"/tmp/temp_{globule.globule_id}.md")
            
            def commit_file(self, temp_path: Path, final_path: Path) -> None:
                """Mock file commit operation."""
                pass
            
            def cleanup_temp(self, temp_path: Path) -> None:
                """Mock temp file cleanup."""
                pass
        
        return MockFileManager()
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        Normalize vector for consistent similarity calculations.
        
        Uses L2 normalization for accurate cosine similarity.
        """
        if vector is None:
            return None
            
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm
    
    def _row_to_globule(self, row) -> ProcessedGlobuleV1:
        """Convert database row to ProcessedGlobule using the correct V1 model"""
        try:
            # Row structure: globule_id, contract_version, processed_timestamp, raw_text,
            #                embedding, parsed_data, nuances, file_path,
            #                processing_time_ms, provider_metadata, created_at
            
            # Deserialize embedding from blob to list
            embedding = []
            if row[4] is not None:  # embedding blob
                embedding_array = np.frombuffer(row[4], dtype=np.float32)
                embedding = embedding_array.tolist()
            
            # Deserialize JSON fields
            parsed_data = json.loads(row[5]) if row[5] else {}
            nuances_dict = json.loads(row[6]) if row[6] else {}
            provider_metadata = json.loads(row[9]) if row[9] else {}
            
            # Create NuanceMetaDataV1 from deserialized dict
            from globule.core.models import NuanceMetaDataV1
            nuances = NuanceMetaDataV1(**nuances_dict) if nuances_dict else NuanceMetaDataV1()
            
            # Create file decision if file path exists
            file_decision = None
            if row[7]:  # file_path
                file_path = Path(row[7])
                file_decision = FileDecisionV1(
                    semantic_path=str(file_path.parent),
                    filename=file_path.name,
                    confidence=0.8
                )
            
            # Create original globule from stored raw text
            from globule.core.models import GlobuleV1
            original_globule = GlobuleV1(
                raw_text=row[3],
                source="stored"  # Default source for retrieved globules
            )
            
            return ProcessedGlobuleV1(
                globule_id=uuid.UUID(row[0]),
                original_globule=original_globule,
                embedding=embedding,
                parsed_data=parsed_data,
                nuances=nuances,
                file_decision=file_decision,
                processing_time_ms=row[8],  # This is a float
                provider_metadata=provider_metadata,
                contract_version=row[1],
                processed_timestamp=datetime.fromisoformat(row[2])
            )
        except Exception as e:
            raise StorageError(f"Failed to convert row to globule: {e}")