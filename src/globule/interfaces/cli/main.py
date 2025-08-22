"""
Main CLI commands for Globule.

Refactored to use the GlobuleAPI, providing a clean separation between the
command-line interface and the core application logic.
"""

import asyncio
import click
import logging
import sys
from typing import Optional, Any

import asyncclick as click

from globule.core.api import GlobuleAPI
from globule.core.models import EnrichedInput, StructuredQuery
from globule.storage.sqlite_manager import SQLiteStorageManager
from globule.services.embedding.ollama_provider import OllamaEmbeddingProvider
from globule.services.embedding.mock_adapter import MockEmbeddingAdapter as MockEmbeddingProvider
from globule.services.embedding.ollama_adapter import OllamaEmbeddingAdapter
from globule.services.parsing.ollama_parser import OllamaParser
from globule.services.parsing.ollama_adapter import OllamaParsingAdapter
from globule.orchestration.engine import OrchestrationEngine
from globule.core.frontend_manager import frontend_manager, FrontendType
from globule.config.settings import get_config
from globule.schemas.manager import SchemaManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GlobuleContext:
    """Shared context for CLI commands, centered around the GlobuleAPI."""

    def __init__(self):
        self.api: Optional[GlobuleAPI] = None
        self._storage = None
        self._embedding_provider = None
        self._parsing_provider = None
        self._initialized = False

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with guaranteed cleanup."""
        await self.cleanup()

    async def initialize(self, verbose: bool = False) -> None:
        """Initialize all services and create the GlobuleAPI instance."""
        if self._initialized:
            return

        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        config = get_config()

        # 1. Initialize Storage
        self._storage = SQLiteStorageManager()
        await self._storage.initialize()

        # 2. Initialize Embedding Provider
        embedding_provider = OllamaEmbeddingProvider()
        if not await embedding_provider.health_check():
            click.echo("Warning: Ollama not accessible. Using mock embeddings.", err=True)
            await embedding_provider.close()
            embedding_provider = MockEmbeddingProvider()
        self._embedding_provider = embedding_provider
        embedding_adapter = OllamaEmbeddingAdapter(self._embedding_provider)

        # 3. Initialize Parsing Provider
        self._parsing_provider = OllamaParser()
        parsing_adapter = OllamaParsingAdapter(self._parsing_provider)

        # 4. Initialize Orchestrator
        orchestrator = OrchestrationEngine(
            embedding_provider=embedding_adapter,
            parser_provider=parsing_adapter,
            storage_manager=self._storage
        )

        # 5. Create the API
        self.api = GlobuleAPI(storage=self._storage, orchestrator=orchestrator)

        self._initialized = True

    async def cleanup(self) -> None:
        """Clean up all services."""
        if not self._initialized:
            return

        if self._embedding_provider:
            await self._embedding_provider.close()
        if self._parsing_provider:
            await self._parsing_provider.close()
        if self._storage:
            await self._storage.close()
        self._initialized = False


@click.group()
@click.version_option(version="1.0.0")
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
async def cli(ctx: click.Context, verbose: bool):
    """
    Globule: Turn your scattered thoughts into structured drafts. Effortlessly.
    """
    ctx.ensure_object(dict)
    ctx.obj['context'] = GlobuleContext()
    ctx.obj['verbose'] = verbose


@click.command()
@click.argument('text', required=True)
@click.pass_context
async def add(ctx: click.Context, text: str) -> None:
    """Add a thought to your Globule collection."""
    verbose = ctx.obj.get('verbose', False)
    async with ctx.obj['context'] as context:
        try:
            await context.initialize(verbose)
            click.echo("Processing your thought...")
            processed_globule = await context.api.add_thought(text, source="cli")
            click.echo(f"Thought captured with ID: {processed_globule.globule_id}")

        except Exception as e:
            logger.error(f"Failed to add thought: {e}")
            click.echo(f"Error: {e}", err=True)
            raise click.Abort()


@click.command()
@click.argument('topic', required=True)
@click.option('--limit', '-l', default=100, help='Maximum globules to search.')
@click.option('--output', '-o', help='Output draft to file.')
@click.option('--frontend', '-f', type=click.Choice(['tui', 'web', 'cli'], case_sensitive=False),
              default='tui', help='Frontend to use.')
@click.option('--port', '-p', default=8000, help='Port for web frontend.')
@click.option('--host', default='127.0.0.1', help='Host for web frontend.')
@click.pass_context
async def draft(ctx: click.Context, topic: str, limit: int, output: Optional[str],
                frontend: str, port: int, host: str) -> None:
    """Interactive drafting from clustered thoughts."""
    verbose = ctx.obj.get('verbose', False)
    frontend_type = FrontendType(frontend.lower())
    
    async with ctx.obj['context'] as context:
        await context.initialize(verbose)

        if frontend_type == FrontendType.CLI:
            click.echo("[CLI] Searching for related thoughts...")
            results = await context.api.search_semantic(topic, limit=limit)
            if not results:
                click.echo("No related thoughts found.")
                return

            click.echo(f"Found {len(results)} related thoughts.")
            draft_content = "\n\n---\n\n".join([g.original_globule.raw_text for g in results])

            if output:
                if await context.api.export_draft(draft_content, output):
                    click.echo(f"Draft exported to: {output}")
                else:
                    click.echo(f"Error exporting draft to: {output}", err=True)
            else:
                click.echo("\n--- Draft Content ---\
")
                click.echo(draft_content)
            return

        launch_kwargs = {
            'topic': topic,
            'limit': limit,
            'output': output,
            'api': context.api
        }
        if frontend_type == FrontendType.WEB:
            launch_kwargs.update({'port': port, 'host': host})

        result = await frontend_manager.launch_frontend(frontend_type, **launch_kwargs)

        if not result['success']:
            click.echo(f"[ERROR] {result['message']}", err=True)
            raise click.Abort()

        click.echo(f"[SUCCESS] {result['message']}")


@click.command()
@click.argument('query', required=True)
@click.option('--limit', '-l', default=10, help='Maximum results to return.')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed search results.')
@click.pass_context
async def search(ctx: click.Context, query: str, limit: int, verbose: bool) -> None:
    """Search for similar thoughts using semantic vector search."""
    verbose = verbose or ctx.obj.get('verbose', False)
    async with ctx.obj['context'] as context:
        await context.initialize(verbose)
        click.echo(f"Searching for: '{query}'")
        results = await context.api.search_semantic(query, limit=limit)

        if not results:
            click.echo("No similar thoughts found.")
            return

        click.echo(f"Found {len(results)} similar thoughts:\n")
        for i, globule in enumerate(results, 1):
            click.echo(f"{i}. {globule.original_globule.raw_text[:100]}...")
            if verbose:
                click.echo(f"   ID: {globule.globule_id}")
                click.echo(f"   Created: {globule.processed_timestamp}")

@click.command()
@click.pass_context
async def reconcile(ctx: click.Context) -> None:
    """Reconcile files on disk with the database."""
    async with ctx.obj['context'] as context:
        await context.initialize(ctx.obj.get('verbose', False))
        click.echo("Starting file-database reconciliation...")
        stats = await context.api.reconcile_files()
        click.echo("Reconciliation Complete:")
        for key, value in stats.items():
            click.echo(f"  {key.replace('_', ' ').title()}: {value}")

@click.command()
@click.argument('query', required=True)
@click.pass_context
async def nlsearch(ctx: click.Context, query: str) -> None:
    """Ask a natural language question about your thoughts."""
    verbose = ctx.obj.get('verbose', False)
    async with ctx.obj['context'] as context:
        await context.initialize(verbose)
        click.echo(f"Answering question: '{query}'...")
        try:
            results = await context.api.natural_language_query(query)
            if not results:
                click.echo("Could not answer the question.")
                return

            from rich.table import Table
            from rich.console import Console
            table = Table(title=f"Result for: '{query}'")
            headers = results[0].keys() if results else []
            for header in headers:
                table.add_column(header, justify="left")
            for row in results:
                table.add_row(*[str(item) for item in row.values()])
            console = Console()
            console.print(table)

        except Exception as e:
            click.echo(f"Error during natural language query: {e}", err=True)

@click.command()
@click.option('--mode', '-m', type=click.Choice(['interactive', 'demo', 'debug']), default='demo', help='Glass Engine tutorial mode.')
@click.pass_context
async def tutorial(ctx: click.Context, mode: str) -> None:
    """Run the Glass Engine tutorial to see how Globule works."""
    from globule.tutorial.glass_engine_core import run_glass_engine, GlassEngineMode
    click.echo(f"Starting Glass Engine in {mode} mode...")
    try:
        mode_enum = GlassEngineMode[mode.upper()]
        await run_glass_engine(mode_enum)
        click.echo(f"Glass Engine finished.")
    except Exception as e:
        logger.error(f"Failed to run Glass Engine tutorial: {e}")
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()

@click.command()
@click.option('--verbose', '-v', is_flag=True, help='Show detailed cluster analysis')
@click.option('--export', '-e', help='Export results to JSON file')
@click.pass_context
async def cluster(ctx: click.Context, verbose: bool, export: Optional[str]) -> None:
    """Discover semantic clusters and themes in your thoughts."""
    verbose = verbose or ctx.obj.get('verbose', False)
    async with ctx.obj['context'] as context:
        await context.initialize(verbose)
        click.echo("Analyzing semantic clusters in your thoughts...")
        analysis = await context.api.get_clusters()
        if not analysis.clusters:
            click.echo("No clusters found. Add more thoughts to enable clustering.")
            return

        click.echo(f"\nDiscovered {len(analysis.clusters)} semantic clusters:\n")
        for i, cluster_obj in enumerate(analysis.clusters, 1):
            click.echo(f"{i}. {cluster_obj.label} ({cluster_obj.size} thoughts)")
            if verbose:
                click.echo(f"   Description: {cluster_obj.description}")
                click.echo(f"   Keywords: {', '.join(cluster_obj.keywords)}")
                click.echo(f"   Confidence: {cluster_obj.confidence_score:.2f}")
        
        if export:
            import json
            try:
                with open(export, 'w') as f:
                    json.dump(analysis.to_dict(), f, indent=2)
                click.echo(f"\nExported analysis to {export}")
            except Exception as e:
                click.echo(f"\nError exporting to file: {e}", err=True)

@click.group()
def skeleton():
    """Manage canvas skeleton templates."""
    pass

@skeleton.command(name="list")
@click.pass_context
async def skeleton_list(ctx: click.Context):
    """List all available skeleton templates."""
    async with ctx.obj['context'] as context:
        await context.initialize(ctx.obj.get('verbose', False))
        skeletons = context.api.list_skeletons()
        if not skeletons:
            click.echo("No skeletons found.")
            return
        click.echo("Available Skeletons:")
        for s in skeletons:
            click.echo(f"- {s['name']}: {s['description']}")

@skeleton.command(name="apply")
@click.argument('name', required=True)
@click.pass_context
async def skeleton_apply(ctx: click.Context, name: str):
    """Apply a skeleton template."""
    async with ctx.obj['context'] as context:
        await context.initialize(ctx.obj.get('verbose', False))
        click.echo(f"Applying skeleton: {name}...")
        modules = context.api.apply_skeleton(name)
        click.echo("Generated Modules:")
        for m in modules:
            click.echo(f"- {m['name']}")

@skeleton.command(name="stats")
@click.pass_context
async def skeleton_stats(ctx: click.Context):
    """Show statistics about skeleton templates."""
    async with ctx.obj['context'] as context:
        await context.initialize(ctx.obj.get('verbose', False))
        stats = context.api.get_skeleton_stats()
        click.echo("Skeleton Stats:")
        for key, value in stats.items():
            click.echo(f"- {key.replace('_', ' ').title()}: {value}")

@skeleton.command(name="create-defaults")
@click.pass_context
async def skeleton_create_defaults(ctx: click.Context):
    """Create default skeleton templates."""
    async with ctx.obj['context'] as context:
        await context.initialize(ctx.obj.get('verbose', False))
        created = context.api.create_default_skeletons()
        click.echo(f"Created {len(created)} default skeletons: {', '.join(created)}")

# Register all commands
cli.add_command(add)
cli.add_command(draft)
cli.add_command(search)
cli.add_command(reconcile)
cli.add_command(tutorial)
cli.add_command(cluster)
cli.add_command(nlsearch)
cli.add_command(skeleton)

def main():
    """Entry point for the CLI."""
    cli()

if __name__ == '__main__':
    main()