"""
Orchestration Engine for Globule.

Coordinates embedding and parsing services concurrently to process globules
through the complete AI pipeline. Renamed from parallel_strategy.py for clarity.
"""

import asyncio
import time
import logging
from typing import Dict, Any

from globule.core.interfaces import OrchestrationEngine as OrchestrationEngineInterface, EmbeddingProvider, ParsingProvider, StorageManager
from globule.core.models import EnrichedInput, ProcessedGlobule, FileDecision
from pathlib import Path

logger = logging.getLogger(__name__)


class OrchestrationEngine(OrchestrationEngineInterface):
    """Main orchestration engine for processing globules through the AI pipeline"""
    
    def __init__(self, 
                 embedding_provider: EmbeddingProvider,
                 parsing_provider: ParsingProvider,
                 storage_manager: StorageManager):
        self.embedding_provider = embedding_provider
        self.parsing_provider = parsing_provider
        self.storage_manager = storage_manager
        
    
    async def process_globule(self, enriched_input: EnrichedInput) -> ProcessedGlobule:
        """Process an enriched input into a processed globule"""
        start_time = time.time()
        processing_times = {}
        
        logger.debug(f"Processing globule: {enriched_input.original_text[:50]}...")
        
        # Launch embedding and parsing tasks concurrently
        embedding_task = asyncio.create_task(
            self._generate_embedding(enriched_input.enriched_text)
        )
        parsing_task = asyncio.create_task(
            self._parse_content(enriched_input.enriched_text, enriched_input.schema_config)
        )
        
        # Wait for both to complete
        try:
            embedding_result, parsing_result = await asyncio.gather(
                embedding_task, parsing_task, return_exceptions=True
            )
            
            # Handle embedding result
            if isinstance(embedding_result, Exception):
                logger.error(f"Embedding failed: {embedding_result}")
                embedding = None
                embedding_confidence = 0.0
                processing_times["embedding_ms"] = 0
            else:
                embedding, embed_time = embedding_result
                embedding_confidence = 1.0  # Assume success = high confidence for MVP
                processing_times["embedding_ms"] = embed_time
            
            # Handle parsing result  
            if isinstance(parsing_result, Exception):
                logger.error(f"Parsing failed: {parsing_result}")
                parsed_data = {"error": str(parsing_result)}
                parsing_confidence = 0.0
                processing_times["parsing_ms"] = 0
            else:
                parsed_data, parse_time = parsing_result
                parsing_confidence = 1.0  # Assume success = high confidence for MVP
                processing_times["parsing_ms"] = parse_time
            
            # Generate file decision from parsed data
            file_decision = self._generate_file_decision(
                enriched_input.original_text, 
                parsed_data
            )
            
            # Calculate total processing time
            total_time = (time.time() - start_time) * 1000
            processing_times["total_ms"] = total_time
            processing_times["orchestration_ms"] = total_time - processing_times.get("embedding_ms", 0) - processing_times.get("parsing_ms", 0)
            
            # Create processed globule
            globule = ProcessedGlobule(
                text=enriched_input.original_text,
                embedding=embedding,
                embedding_confidence=embedding_confidence,
                parsed_data=parsed_data,
                parsing_confidence=parsing_confidence,
                file_decision=file_decision,
                orchestration_strategy="parallel",
                processing_time_ms=processing_times,
                confidence_scores={
                    "embedding": embedding_confidence,
                    "parsing": parsing_confidence,
                    "overall": (embedding_confidence + parsing_confidence) / 2
                },
                interpretations=[],  # MVP: no disagreement detection
                has_nuance=False,    # MVP: no nuance detection
                semantic_neighbors=[],  # Will be populated later
                processing_notes=[]
            )
            
            logger.debug(f"Globule processed in {total_time:.1f}ms")
            return globule
            
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
    
    def _generate_file_decision(self, text: str, parsed_data: Dict[str, Any]) -> FileDecision:
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
        
        return FileDecision(
            semantic_path=semantic_path,
            filename=filename,
            metadata={
                "auto_generated": True,
                "source": "parallel_orchestration"
            },
            confidence=0.8,  # Default confidence for MVP
            alternative_paths=[]
        )