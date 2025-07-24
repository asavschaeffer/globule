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
from globule.parsing.mock_parser import MockOllamaParser
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
        
        parsing_provider = MockOllamaParser()
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
    
    cli()


if __name__ == '__main__':
    main()