"""Storage manager for Globule - handles data persistence."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol
from uuid import uuid4

import aiosqlite
import msgpack
import numpy as np
from pydantic import BaseModel


class Globule(BaseModel):
    """Data model for a globule (thought entry)."""
    id: str
    content: str
    created_at: datetime
    embedding: Optional[np.ndarray] = None
    parsed_data: Optional[Dict[str, Any]] = None
    entities: Optional[List[Dict[str, Any]]] = None
    domain: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class Storage(Protocol):
    """Abstract interface for storage backends."""
    
    async def store_globule(self, globule: Globule) -> None:
        """Store a globule in the database."""
        ...
    
    async def retrieve_by_id(self, globule_id: str) -> Optional[Globule]:
        """Retrieve a globule by its ID."""
        ...
    
    async def search_semantic(self, query_embedding: np.ndarray, limit: int = 10) -> List[Globule]:
        """Search for globules using semantic similarity."""
        ...
    
    async def search_temporal(self, start_date: datetime, end_date: datetime) -> List[Globule]:
        """Search for globules within a date range."""
        ...


class SQLiteStorage:
    """SQLite implementation of the storage interface."""
    
    def __init__(self, db_path: str = "globule.db"):
        self.db_path = Path(db_path)
        self._initialized = False
    
    async def _init_db(self) -> None:
        """Initialize the database with required tables."""
        if self._initialized:
            return
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS globules (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    embedding BLOB,
                    parsed_data JSON,
                    entities JSON,
                    domain TEXT,
                    metadata JSON
                )
            """)
            
            await db.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON globules(created_at)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_domain ON globules(domain)")
            await db.commit()
        
        self._initialized = True
    
    async def store_globule(self, globule: Globule) -> None:
        """Store a globule in the SQLite database."""
        await self._init_db()
        
        # Serialize embedding if present
        embedding_blob = None
        if globule.embedding is not None:
            embedding_blob = msgpack.packb(globule.embedding.tolist())
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO globules 
                (id, content, created_at, embedding, parsed_data, entities, domain, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                globule.id,
                globule.content,
                globule.created_at.isoformat(),
                embedding_blob,
                json.dumps(globule.parsed_data) if globule.parsed_data else None,
                json.dumps(globule.entities) if globule.entities else None,
                globule.domain,
                json.dumps(globule.metadata) if globule.metadata else None
            ))
            await db.commit()
    
    async def retrieve_by_id(self, globule_id: str) -> Optional[Globule]:
        """Retrieve a globule by its ID."""
        await self._init_db()
        
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT * FROM globules WHERE id = ?", (globule_id,)
            )
            row = await cursor.fetchone()
            
            if not row:
                return None
            
            return self._row_to_globule(row)
    
    async def search_temporal(self, start_date: datetime, end_date: datetime) -> List[Globule]:
        """Search for globules within a date range."""
        await self._init_db()
        
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                SELECT * FROM globules 
                WHERE created_at BETWEEN ? AND ?
                ORDER BY created_at DESC
            """, (start_date.isoformat(), end_date.isoformat()))
            
            rows = await cursor.fetchall()
            return [self._row_to_globule(row) for row in rows]
    
    async def search_semantic(self, query_embedding: np.ndarray, limit: int = 10) -> List[Globule]:
        """Search for globules using semantic similarity."""
        await self._init_db()
        
        # For now, return all globules with embeddings
        # TODO: Implement proper cosine similarity search
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                SELECT * FROM globules 
                WHERE embedding IS NOT NULL
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))
            
            rows = await cursor.fetchall()
            return [self._row_to_globule(row) for row in rows]
    
    def _row_to_globule(self, row) -> Globule:
        """Convert a database row to a Globule object."""
        embedding = None
        if row[3]:  # embedding column
            embedding = np.array(msgpack.unpackb(row[3]))
        
        return Globule(
            id=row[0],
            content=row[1],
            created_at=datetime.fromisoformat(row[2]),
            embedding=embedding,
            parsed_data=json.loads(row[4]) if row[4] else None,
            entities=json.loads(row[5]) if row[5] else None,
            domain=row[6],
            metadata=json.loads(row[7]) if row[7] else None
        )


def generate_id() -> str:
    """Generate a unique ID for a globule."""
    return str(uuid4())