"""
Main CLI commands for Globule.

Implements the core user experience:
- globule add "thought"
- globule draft "topic"
"""

import asyncio
import click
import logging
from datetime import datetime
from typing import Optional

from globule.core.models import EnrichedInput
from globule.storage.sqlite_manager import SQLiteStorageManager
from globule.embedding.ollama_provider import OllamaEmbeddingProvider
from globule.parsing.ollama_parser import OllamaParser
from globule.orchestration.parallel_strategy import ParallelOrchestrationEngine
from globule.config.settings import get_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """
    Globule: Turn your scattered thoughts into structured drafts. Effortlessly.
    
    Capture thoughts with 'globule add' and synthesize with 'globule draft'.
    """
    pass


@cli.command()
@click.argument('text', required=True)
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
async def add(text: str, verbose: bool) -> None:
    """Add a thought to your Globule collection."""
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize components
        config = get_config()
        storage = SQLiteStorageManager()
        await storage.initialize()
        
        # Try to use Ollama, fall back to mock if not available
        embedding_provider = OllamaEmbeddingProvider()
        health_ok = await embedding_provider.health_check()
        
        if not health_ok:
            click.echo("Warning: Ollama not accessible. Using mock embeddings.", err=True)
            # Use mock embedding provider for testing
            class MockEmbeddingProvider:
                def get_dimension(self):
                    return 1024
                
                async def embed(self, text):
                    import numpy as np
                    return np.random.randn(1024).astype(np.float32)
                
                async def embed_batch(self, texts):
                    return [await self.embed(text) for text in texts]
                
                async def close(self):
                    pass
                
                async def health_check(self):
                    return True
            
            embedding_provider = MockEmbeddingProvider()
        
        parsing_provider = OllamaParser()
        orchestrator = ParallelOrchestrationEngine(
            embedding_provider, parsing_provider, storage
        )
        
        # Create enriched input
        enriched_input = EnrichedInput(
            original_text=text,
            enriched_text=text,  # No preprocessing for MVP
            detected_schema_id=None,
            schema_config=None,
            additional_context={},
            source="cli",
            timestamp=datetime.now(),
            verbosity="verbose" if verbose else "concise"
        )
        
        # Process the globule
        click.echo("Processing your thought...")
        start_time = datetime.now()
        
        processed_globule = await orchestrator.process_globule(enriched_input)
        globule_id = await storage.store_globule(processed_globule)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Show results
        if verbose:
            click.echo(f"Thought captured as {globule_id}")
            click.echo(f"   Processing time: {processing_time:.1f}ms")
            click.echo(f"   Embedding confidence: {processed_globule.embedding_confidence:.2f}")
            click.echo(f"   Parsing confidence: {processed_globule.parsing_confidence:.2f}")
            if processed_globule.file_decision:
                file_path = processed_globule.file_decision.semantic_path / processed_globule.file_decision.filename
                click.echo(f"   Suggested path: {file_path}")
        else:
            click.echo(f"Thought captured in {processing_time:.0f}ms")
        
        # Cleanup
        await embedding_provider.close()
        await parsing_provider.close()
        await storage.close()
        
    except Exception as e:
        logger.error(f"Failed to add thought: {e}")
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.argument('topic', required=False)
@click.option('--limit', '-l', default=50, help='Maximum globules to consider')
async def draft(topic: Optional[str], limit: int) -> None:
    """Launch interactive drafting session."""
    
    try:
        # Import TUI here to avoid startup overhead for other commands
        from globule.tui.app import SynthesisApp
        
        # Initialize storage
        storage = SQLiteStorageManager()
        await storage.initialize()
        
        # Create and run TUI app
        app = SynthesisApp(storage_manager=storage, topic=topic, limit=limit)
        await app.run_async()
        
        # Cleanup
        await storage.close()
        
    except Exception as e:
        logger.error(f"Failed to launch drafting session: {e}")
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.argument('query', required=True)
@click.option('--limit', '-l', default=10, help='Maximum results to return')
@click.option('--threshold', '-t', default=0.4, help='Minimum similarity threshold (0.0-1.0)')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed search results')
async def search(query: str, limit: int, threshold: float, verbose: bool) -> None:
    """
    Search for similar thoughts using semantic vector search.
    
    This command demonstrates Phase 2 vector search capabilities by finding
    semantically related content based on meaning rather than exact keywords.
    
    Examples:
    \b
    globule search "creative writing process"
    globule search "system design patterns" --limit 5 --threshold 0.6
    """
    try:
        # Initialize components
        config = get_config()
        storage = SQLiteStorageManager()
        await storage.initialize()
        
        embedding_provider = OllamaEmbeddingProvider()
        
        click.echo(f"SEARCH: Searching for: '{query}'")
        click.echo(f"PARAMS: limit={limit}, threshold={threshold}")
        
        # Generate query embedding
        click.echo("EMBEDDING: Generating semantic embedding...")
        query_embedding = await embedding_provider.embed(query)
        
        # Perform vector search
        click.echo("SEARCH: Searching semantic database...")
        results = await storage.search_by_embedding(query_embedding, limit, threshold)
        
        if not results:
            click.echo("NO RESULTS: No similar thoughts found.")
            click.echo("TIP: Try lowering the --threshold or adding more content with 'globule add'")
            return
        
        # Display results
        click.echo(f"\nSUCCESS: Found {len(results)} similar thoughts:\n")
        
        for i, (globule, similarity) in enumerate(results, 1):
            # Format similarity score
            similarity_pct = similarity * 100
            similarity_bar = "=" * int(similarity * 20)  # Visual similarity bar
            
            click.echo(f"{i}. [{similarity_pct:.1f}% {similarity_bar}]")
            
            # Show content preview
            preview = globule.text[:100] + "..." if len(globule.text) > 100 else globule.text
            click.echo(f"   {preview}")
            
            if verbose:
                # Show detailed metadata
                click.echo(f"   CREATED: {globule.created_at}")
                click.echo(f"   ID: {globule.id}")
                
                if globule.parsed_data:
                    domain = globule.parsed_data.get('domain', 'unknown')
                    category = globule.parsed_data.get('category', 'unknown')
                    click.echo(f"   DOMAIN: {domain} | CATEGORY: {category}")
                    
                    if 'keywords' in globule.parsed_data:
                        keywords = globule.parsed_data['keywords'][:3]  # Top 3 keywords
                        click.echo(f"   KEYWORDS: {', '.join(keywords)}")
            
            click.echo()  # Blank line between results
        
        # Show search statistics
        if verbose:
            click.echo(f"STATS: Search completed in semantic space with {len(query_embedding)}-dimensional vectors")
        
        # Cleanup
        await embedding_provider.close()
        await storage.close()
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--mode', '-m', 
              type=click.Choice(['interactive', 'demo', 'debug']), 
              default='demo',
              help='Glass Engine mode: interactive (guided tutorial), demo (technical showcase), debug (raw system traces)')
async def tutorial(mode: str) -> None:
    """
    Run the Glass Engine tutorial to see how Globule works under the hood.
    
    The Glass Engine provides three modes for different audiences:
    
    \b
    • INTERACTIVE: Guided tutorial with hands-on learning (best for new users)
    • DEMO: Professional technical showcase with automated examples (best for stakeholders)  
    • DEBUG: Raw execution traces and system introspection (best for engineers/debugging)
    
    Each mode embodies the Glass Engine philosophy: tests become tutorials,
    tutorials become showcases, showcases become tests. Complete transparency.
    """
    
    try:
        # Import Glass Engine core
        from globule.tutorial.glass_engine_core import run_glass_engine, GlassEngineMode
        
        # Map string to enum
        mode_map = {
            'interactive': GlassEngineMode.INTERACTIVE,
            'demo': GlassEngineMode.DEMO,
            'debug': GlassEngineMode.DEBUG
        }
        
        # Run the selected Glass Engine mode
        glass_mode = mode_map[mode]
        metrics = await run_glass_engine(glass_mode)
        
        # Show brief completion summary
        click.echo(f"\nGlass Engine {mode} mode completed in {metrics.total_duration_ms:.1f}ms")
        click.echo(f"Status: {metrics.validation_status}")
        
        if metrics.error_log:
            click.echo(f"Warnings/Errors: {len(metrics.error_log)}", err=True)
        
    except Exception as e:
        logger.error(f"Failed to run Glass Engine tutorial: {e}")
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


def main():
    """Entry point for the CLI"""
    # Convert click commands to async
    def async_command(f):
        def wrapper(*args, **kwargs):
            return asyncio.run(f(*args, **kwargs))
        return wrapper
    
    # Make commands async-compatible
    cli.commands['add'].callback = async_command(cli.commands['add'].callback)
    cli.commands['draft'].callback = async_command(cli.commands['draft'].callback)
    cli.commands['search'].callback = async_command(cli.commands['search'].callback)
    cli.commands['tutorial'].callback = async_command(cli.commands['tutorial'].callback)
    
    cli()


if __name__ == '__main__':
    main()