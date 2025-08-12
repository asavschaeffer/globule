"""
Orchestration Engine for Globule.

Coordinates embedding and parsing services concurrently to process globules
through the complete AI pipeline. Renamed from parallel_strategy.py for clarity.
"""

import asyncio
import time
import logging
from typing import Dict, Any

from globule.core.interfaces import IOrchestrationEngine as OrchestrationEngineInterface, IEmbeddingProvider, IParserProvider, IStorageManager
from globule.core.models import ProcessedGlobuleV1, FileDecisionV1
from pathlib import Path

logger = logging.getLogger(__name__)


class OrchestrationEngine(OrchestrationEngineInterface):
    """Main orchestration engine for processing globules through the AI pipeline"""
    
    def __init__(self, 
                 embedding_provider: IEmbeddingProvider,
                 parsing_provider: IParserProvider,
                 storage_manager: IStorageManager):
        self.embedding_provider = embedding_provider
        self.parsing_provider = parsing_provider
        self.storage_manager = storage_manager
        
    
    async def process_globule(self, globule: "GlobuleV1") -> "ProcessedGlobuleV1":
        """Process a raw globule into a processed globule"""
        start_time = time.time()
        processing_times = {}
        
        logger.debug(f"Processing globule: {globule.raw_text[:50]}...")
        
        # Launch embedding and parsing tasks concurrently
        embedding_task = asyncio.create_task(
            self._generate_embedding(globule.raw_text)
        )
        parsing_task = asyncio.create_task(
            self._parse_content(globule.raw_text)
        )
        
        # Wait for both to complete
        try:
            embedding_result, parsing_result = await asyncio.gather(
                embedding_task, parsing_task, return_exceptions=True
            )
            
            # Handle embedding result
            if isinstance(embedding_result, Exception):
                logger.error(f"Embedding failed: {embedding_result}")
                embedding = []
                processing_times["embedding_ms"] = 0
            else:
                embedding, embed_time = embedding_result
                processing_times["embedding_ms"] = embed_time
            
            # Handle parsing result  
            if isinstance(parsing_result, Exception):
                logger.error(f"Parsing failed: {parsing_result}")
                parsed_data = {"error": str(parsing_result)}
                processing_times["parsing_ms"] = 0
            else:
                parsed_data, parse_time = parsing_result
                processing_times["parsing_ms"] = parse_time
            
            # Generate file decision from parsed data
            file_decision = self._generate_file_decision(
                globule.raw_text, 
                parsed_data
            )
            
            # Calculate total processing time
            total_time = (time.time() - start_time) * 1000
            processing_times["total_ms"] = total_time
            processing_times["orchestration_ms"] = total_time - processing_times.get("embedding_ms", 0) - processing_times.get("parsing_ms", 0)
            
            # Create processed globule
            processed_globule = ProcessedGlobuleV1(
                globule_id=globule.globule_id,
                original_globule=globule,
                embedding=embedding,
                parsed_data=parsed_data,
                file_decision=file_decision,
                processing_time_ms=total_time,
                provider_metadata={
                    "parser": self.parsing_provider.__class__.__name__,
                    "embedder": self.embedding_provider.__class__.__name__,
                    "storage": self.storage_manager.__class__.__name__
                }
            )
            
            logger.debug(f"Globule processed in {total_time:.1f}ms")
            return processed_globule
            
        except Exception as e:
            logger.error(f"Orchestration failed: {e}")
            raise
    
    
    async def _generate_embedding(self, text: str) -> tuple:
        """Generate embedding and return (embedding, time_ms)"""
        logger.debug("TIMING: Starting embedding generation...")
        start_time = time.time()
        embedding = await self.embedding_provider.embed(text)
        processing_time = (time.time() - start_time) * 1000
        logger.info(f"TIMING: Embedding completed in {processing_time:.1f}ms")
        return embedding, processing_time
    
    async def _parse_content(self, text: str, schema_config: Dict[str, Any] = None) -> tuple:
        """Parse content and return (parsed_data, time_ms)"""
        logger.debug("TIMING: Starting content parsing...")
        start_time = time.time()
        parsed_data = await self.parsing_provider.parse(text, schema_config)
        processing_time = (time.time() - start_time) * 1000
        logger.info(f"TIMING: Parsing completed in {processing_time:.1f}ms")
        return parsed_data, processing_time
    
    def _generate_file_decision(self, text: str, parsed_data: Dict[str, Any]) -> FileDecisionV1:
        """Generate simple file decision for MVP"""
        # Use parsed data if available, otherwise create simple path
        domain = parsed_data.get("domain", "general")
        category = parsed_data.get("category", "note")
        title = parsed_data.get("title", text[:30].replace(" ", "-").lower())
        
        # Clean title for filename
        clean_title = "".join(c for c in title if c.isalnum() or c in "-_").strip("-_")
        if not clean_title:
            clean_title = "untitled"
        
        # Create semantic path: domain/category/
        semantic_path = Path(domain) / category
        filename = f"{clean_title}.md"
        
        return FileDecisionV1(
            semantic_path=str(semantic_path),
            filename=filename,
            confidence=0.8,  # Default confidence for MVP
        )


# Reusable API functions for CLI mirroring
async def search_globules_nlp(nl_query: str, storage_manager: IStorageManager) -> str:
    """
    Mirror TUI search: Convert natural language query to SQL and execute.
    Returns formatted Markdown module content.
    """
    from globule.services.parsing.ollama_parser import OllamaParser
    import sqlite3
    import os
    
    try:
        # Phase 1: Convert NL to SQL using AI
        sql = await _nl_to_sql_cli(nl_query)
        logger.info(f"Generated SQL for CLI search: {sql}")
        
        # Phase 2: Execute SQL
        results, headers = await _execute_sql_cli(sql, storage_manager)
        
        # Phase 3: Format as Markdown
        formatted_content = _format_module_cli(results, headers, nl_query)
        
        return formatted_content
        
    except Exception as e:
        logger.error(f"CLI search failed: {e}")
        return f"### {nl_query}\n\n**Error:** {str(e)}\n"


async def _nl_to_sql_cli(nl_query: str) -> str:
    """Convert natural language query to SQL (CLI version)"""
    from globule.services.parsing.ollama_parser import OllamaParser
    
    prompt = f"""
Translate this natural language query to SQL for the globules table.
Columns: id TEXT, text TEXT, created_at TIMESTAMP, embedding TEXT/BLOB, parsed_data JSON.
The parsed_data JSON contains keys like: valet_name, car_make, license_plate, parking_spot.

Examples:
- "maria" → WHERE json_extract(parsed_data, '$.valet_name') = 'maria'  
- "valet maria" → WHERE json_extract(parsed_data, '$.valet_name') = 'maria'
- "john" → WHERE json_extract(parsed_data, '$.valet_name') = 'john'
- "honda" → WHERE json_extract(parsed_data, '$.car_make') = 'honda'
- "valet:maria honda" → WHERE json_extract(parsed_data, '$.valet_name') = 'maria' AND json_extract(parsed_data, '$.car_make') = 'honda'

Query: {nl_query}
Return only the raw SQL SELECT statement, no explanations.
    """
    
    parser = OllamaParser()
    try:
        response = await parser.parse(prompt, {'action': 'sql_translate', 'output_format': 'text'})
        
        # Extract SQL from response
        if isinstance(response, dict):
            sql = response.get('sql', response.get('title', response.get('reasoning', '')))
        else:
            sql = str(response)
        
        # Clean up the SQL
        sql = sql.strip().replace('```sql', '').replace('```', '').strip()
        
        # Validate SQL format and security
        if sql and sql.upper().startswith('SELECT'):
            dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'TRUNCATE', 'ALTER']
            if any(keyword in sql.upper() for keyword in dangerous_keywords):
                raise ValueError("Potentially dangerous SQL detected")
            return sql
        else:
            raise ValueError("Invalid SQL format generated")
            
    except Exception as e:
        logger.warning(f"AI SQL generation failed: {e}, using fallback")
        # Fallback to simple text search
        return f"SELECT id, text, parsed_data FROM globules WHERE text LIKE '%{nl_query}%' LIMIT 10"
    finally:
        await parser.close()


async def _execute_sql_cli(sql: str, storage_manager: IStorageManager) -> tuple:
    """Execute SQL query and return results with headers (CLI version)"""
    import sqlite3
    import os
    
    # Get database path
    if hasattr(storage_manager, 'db_path'):
        db_path = str(storage_manager.db_path)
    else:
        db_path = os.path.expanduser("~/.globule/data/globules.db")
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}")
    
    try:
        conn = sqlite3.connect(db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(sql)
        results = cursor.fetchall()
        
        # Convert to tuples and get headers
        results = [tuple(row) for row in results]
        headers = [desc[0] for desc in cursor.description] if cursor.description else ['Result']
        
        # Limit results for CLI display
        if len(results) > 50:
            logger.info(f"Large result set ({len(results)} rows), limiting to 50")
            results = results[:50]
        
        return results, headers
        
    except Exception as e:
        raise Exception(f"SQL execution error: {str(e)}")
    finally:
        if 'conn' in locals():
            conn.close()


def _format_module_cli(results: list, headers: list, query: str) -> str:
    """Format query results as a Markdown module (CLI version)"""
    if not results:
        return f"### {query}\nNo data found."
    
    # Create markdown table
    md = f"### {query}\n\n"
    md += "| " + " | ".join(headers) + " |\n"
    md += "|" + " --- |" * len(headers) + "\n"
    
    # Add rows (limit for CLI display)
    for row in results[:20]:  # Limit to 20 rows
        formatted_row = []
        for value in row:
            if value is None:
                formatted_row.append("")
            elif isinstance(value, str) and len(value) > 50:
                # Truncate long text fields for CLI
                formatted_row.append(value[:50] + "...")
            else:
                formatted_row.append(str(value))
        md += "| " + " | ".join(formatted_row) + " |\n"
    
    if len(results) > 20:
        md += f"\n*... and {len(results) - 20} more rows*\n"
    
    return md


def fetch_globule_content(item_id: str, storage_manager: IStorageManager) -> str:
    """
    Fetch globule content by ID for CLI add-to-draft functionality.
    Returns formatted content ready for draft inclusion.
    """
    import sqlite3
    import os
    
    # Get database path
    if hasattr(storage_manager, 'db_path'):
        db_path = str(storage_manager.db_path)
    else:
        db_path = os.path.expanduser("~/.globule/data/globules.db")
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Query for globule by ID
        cursor.execute(
            "SELECT id, text, created_at, parsed_data FROM globules WHERE id = ?", 
            (item_id,)
        )
        result = cursor.fetchone()
        
        if not result:
            return ""
        
        globule_id, text, created_at, parsed_data_json = result
        
        # Format content for draft inclusion
        formatted_content = f"**Globule:** {globule_id}\n"
        formatted_content += f"**Added:** {created_at}\n\n"
        formatted_content += f"{text}\n"
        
        # Add parsed metadata if available
        if parsed_data_json:
            import json
            try:
                parsed_data = json.loads(parsed_data_json)
                if parsed_data:
                    formatted_content += f"\n**Metadata:** {json.dumps(parsed_data, indent=2)}\n"
            except:
                pass  # Skip if JSON parsing fails
        
        return formatted_content
        
    except Exception as e:
        logger.error(f"Failed to fetch globule {item_id}: {e}")
        return f"Error fetching globule {item_id}: {str(e)}"
    finally:
        if 'conn' in locals():
            conn.close()