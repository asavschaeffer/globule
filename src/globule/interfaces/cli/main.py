"""
Main CLI commands for Globule.

Implements the core user experience:
- globule add "thought"
- globule draft "topic"

Refactored for performance and maintainability:
- Shared context with service initialization
- Thin command wrappers that delegate to services
- Clean separation of concerns
"""

import asyncio
import click
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

from globule.core.models import EnrichedInput
from globule.storage.sqlite_manager import SQLiteStorageManager
from globule.services.embedding.ollama_provider import OllamaEmbeddingProvider
from globule.services.embedding.mock_provider import MockEmbeddingProvider
from globule.services.parsing.ollama_parser import OllamaParser
from globule.orchestration.engine import OrchestrationEngine
from globule.config.settings import get_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GlobuleContext:
    """Shared context for CLI commands containing initialized services."""
    
    def __init__(self):
        self.config = None
        self.storage = None
        self.embedding_provider = None
        self.parsing_provider = None
        self.orchestrator = None
        self._initialized = False
    
    async def initialize(self, verbose: bool = False) -> None:
        """Initialize all services once."""
        if self._initialized:
            return
        
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Initialize configuration
        self.config = get_config()
        
        # Initialize storage
        self.storage = SQLiteStorageManager()
        await self.storage.initialize()
        
        # Initialize embedding provider with fallback to mock
        self.embedding_provider = OllamaEmbeddingProvider()
        health_ok = await self.embedding_provider.health_check()
        
        if not health_ok:
            click.echo("Warning: Ollama not accessible. Using mock embeddings.", err=True)
            await self.embedding_provider.close()
            self.embedding_provider = MockEmbeddingProvider()
        
        # Initialize parsing provider
        self.parsing_provider = OllamaParser()
        
        # Initialize orchestrator
        self.orchestrator = OrchestrationEngine(
            self.embedding_provider, 
            self.parsing_provider, 
            self.storage
        )
        
        self._initialized = True
    
    async def cleanup(self) -> None:
        """Clean up all services."""
        if self.embedding_provider:
            await self.embedding_provider.close()
        if self.parsing_provider:
            await self.parsing_provider.close()
        if self.storage:
            await self.storage.close()
        self._initialized = False


@click.group()
@click.version_option(version="1.0.0")
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx: click.Context, verbose: bool):
    """
    Globule: Turn your scattered thoughts into structured drafts. Effortlessly.
    
    Capture thoughts with 'globule add' and synthesize with 'globule draft'.
    """
    # Initialize shared context
    ctx.ensure_object(dict)
    ctx.obj['context'] = GlobuleContext()
    ctx.obj['verbose'] = verbose


@cli.command()
@click.argument('text', required=True)
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output for this command')
@click.pass_context
async def add(ctx: click.Context, text: str, verbose: bool) -> None:
    """Add a thought to your Globule collection."""
    
    # Use context verbose setting if not overridden
    verbose = verbose or ctx.obj.get('verbose', False)
    
    try:
        # Initialize context if needed
        context: GlobuleContext = ctx.obj['context']
        await context.initialize(verbose)
        
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
        
        # Process the globule using orchestrator
        click.echo("Processing your thought...")
        start_time = datetime.now()
        
        processed_globule = await context.orchestrator.process_globule(enriched_input)
        globule_id = await context.storage.store_globule(processed_globule)
        
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
        
    except Exception as e:
        logger.error(f"Failed to add thought: {e}")
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.argument('topic', required=True)
@click.option('--limit', '-l', default=100, help='Maximum globules to search (default: 100)')
@click.option('--output', '-o', help='Output draft to file')
@click.pass_context
async def draft(ctx: click.Context, topic: str, limit: int, output: Optional[str]) -> None:
    """
    Interactive drafting from clustered thoughts.
    
    This command provides a keyboard-driven interface for building drafts:
    1. Searches for globules related to the topic
    2. Clusters the results using semantic analysis  
    3. Launches an interactive TUI for navigation and selection
    4. Lets you add thoughts to your draft and export the result
    """
    
    try:
        from rich.console import Console
        from globule.interfaces.drafting.interactive_engine import InteractiveDraftingEngine
        from globule.services.clustering.semantic_clustering import SemanticClusteringEngine
        
        console = Console()
        
        # Initialize context
        context: GlobuleContext = ctx.obj['context']
        await context.initialize(ctx.obj.get('verbose', False))
        
        # Check if we have embeddings (can't draft with mock embeddings effectively)
        if isinstance(context.embedding_provider, MockEmbeddingProvider):
            console.print("[red]ERROR:[/red] Interactive drafting requires real embeddings.")
            console.print("Please ensure Ollama is running and try again.")
            return
        
        # Step 1: Vectorize topic and perform semantic search
        console.print(f"[blue]SEARCH:[/blue] Finding thoughts related to '{topic}'...")
        topic_embedding = await context.embedding_provider.embed(topic)
        search_results = await context.storage.search_by_embedding(topic_embedding, limit, 0.3)
        
        if not search_results:
            console.print(f"[red]NO RESULTS:[/red] No thoughts found related to '{topic}'")
            console.print("Try a different topic or add more content with 'globule add'")
            return
        
        # Extract globules from search results
        globules = [globule for globule, _ in search_results]
        console.print(f"[green]FOUND:[/green] {len(globules)} relevant thoughts")
        
        # Step 2: Cluster the search results
        console.print("[blue]CLUSTERING:[/blue] Analyzing semantic patterns...")
        clustering_engine = SemanticClusteringEngine(context.storage)
        
        analysis = await clustering_engine.analyze_semantic_clusters(min_globules=2)
        
        if not analysis.clusters:
            console.print("[yellow]NO CLUSTERS:[/yellow] Unable to find semantic clusters")
            console.print("Proceeding with chronological listing...")
        else:
            console.print(f"[green]CLUSTERS:[/green] Found {len(analysis.clusters)} semantic themes")
        
        # Step 3: Build globules-by-cluster mapping for the interactive engine
        globules_by_cluster = {}
        for cluster in analysis.clusters:
            cluster_globules = [g for g in globules if g.id in cluster.member_ids]
            globules_by_cluster[cluster.id] = cluster_globules
        
        # Step 4: Launch interactive drafting session
        console.print("[blue]INTERACTIVE:[/blue] Starting drafting session...")
        console.print("Use arrow keys to navigate, Enter to select, d/q to finish")
        
        drafting_engine = InteractiveDraftingEngine()
        draft_text = await drafting_engine.run_interactive_session(
            topic=topic,
            clusters=analysis.clusters,
            globules_by_cluster=globules_by_cluster,
            all_globules=globules
        )
        
        # Step 5: Output the final draft
        if output:
            with open(output, 'w', encoding='utf-8') as f:
                f.write(draft_text)
            console.print(f"[green]SUCCESS:[/green] Draft written to {output}")
        else:
            console.print("\n" + "="*60)
            console.print("[green]DRAFT COMPLETE[/green]")
            console.print("="*60)
            print(draft_text)  # Use print for clean output
        
    except KeyboardInterrupt:
        click.echo("\nDrafting session cancelled")
    except Exception as e:
        logger.error(f"Failed to run interactive draft: {e}")
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.argument('query', required=True)
@click.option('--limit', '-l', default=10, help='Maximum results to return')
@click.option('--threshold', '-t', default=0.4, help='Minimum similarity threshold (0.0-1.0)')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed search results')
@click.pass_context
async def search(ctx: click.Context, query: str, limit: int, threshold: float, verbose: bool) -> None:
    """
    Search for similar thoughts using semantic vector search.
    
    This command demonstrates Phase 2 vector search capabilities by finding
    semantically related content based on meaning rather than exact keywords.
    """
    # Use context verbose setting if not overridden
    verbose = verbose or ctx.obj.get('verbose', False)
    
    try:
        # Initialize context
        context: GlobuleContext = ctx.obj['context']
        await context.initialize(verbose)
        
        click.echo(f"SEARCH: Searching for: '{query}'")
        click.echo(f"PARAMS: limit={limit}, threshold={threshold}")
        
        # Generate query embedding
        click.echo("EMBEDDING: Generating semantic embedding...")
        query_embedding = await context.embedding_provider.embed(query)
        
        # Perform vector search
        click.echo("SEARCH: Searching semantic database...")
        results = await context.storage.search_by_embedding(query_embedding, limit, threshold)
        
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
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--min-globules', '-m', default=5, help='Minimum globules required for clustering')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed cluster analysis')
@click.option('--export', '-e', help='Export results to JSON file')
@click.pass_context
async def cluster(ctx: click.Context, min_globules: int, verbose: bool, export: Optional[str]) -> None:
    """
    Discover semantic clusters and themes in your thoughts.
    
    This command demonstrates Phase 2 clustering capabilities by automatically
    grouping related content and identifying common themes across your knowledge base.
    """
    # Use context verbose setting if not overridden
    verbose = verbose or ctx.obj.get('verbose', False)
    
    try:
        # Initialize context
        context: GlobuleContext = ctx.obj['context']
        await context.initialize(verbose)
        
        from globule.services.clustering.semantic_clustering import SemanticClusteringEngine
        clustering_engine = SemanticClusteringEngine(context.storage)
        
        click.echo(f"CLUSTERING: Analyzing semantic patterns in your thoughts...")
        click.echo(f"PARAMS: min_globules={min_globules}")
        
        # Perform clustering analysis
        analysis = await clustering_engine.analyze_semantic_clusters(min_globules)
        
        if not analysis.clusters:
            click.echo("NO CLUSTERS: Insufficient data for clustering analysis.")
            click.echo(f"TIP: You need at least {min_globules} thoughts with embeddings.")
            click.echo("     Add more content with 'globule add' and try again.")
            return
        
        # Display results
        click.echo(f"\nSUCCESS: Discovered {len(analysis.clusters)} semantic clusters:\n")
        
        for i, cluster in enumerate(analysis.clusters, 1):
            confidence_pct = cluster.confidence_score * 100
            confidence_bar = "=" * int(cluster.confidence_score * 15)
            
            click.echo(f"{i}. {cluster.label} [{confidence_pct:.1f}% {confidence_bar}]")
            click.echo(f"   {cluster.description}")
            click.echo(f"   SIZE: {cluster.size} thoughts | DOMAINS: {', '.join(cluster.domains)}")
            
            if cluster.keywords:
                click.echo(f"   KEYWORDS: {', '.join(cluster.keywords[:5])}")
            
            if verbose:
                click.echo(f"   ID: {cluster.id}")
                click.echo(f"   CREATED: {cluster.created_at}")
                
                if cluster.representative_samples:
                    click.echo(f"   SAMPLES:")
                    for sample in cluster.representative_samples[:2]:
                        click.echo(f"     - {sample}")
                
                if cluster.theme_analysis:
                    temporal = cluster.theme_analysis.get('temporal', {})
                    if temporal.get('span_days'):
                        click.echo(f"   TEMPORAL: {temporal['span_days']} day span")
            
            click.echo()  # Blank line between clusters
        
        # Show analysis summary
        click.echo(f"ANALYSIS SUMMARY:")
        click.echo(f"  Method: {analysis.clustering_method}")
        click.echo(f"  DBCV Score: {analysis.dbcv_score:.3f}")
        click.echo(f"  Processing Time: {analysis.processing_time_ms:.1f}ms")
        click.echo(f"  Total Thoughts Analyzed: {analysis.total_globules}")
        
        if verbose and analysis.quality_metrics:
            click.echo(f"\nDETAILED METRICS:")
            for metric, value in analysis.quality_metrics.items():
                click.echo(f"  {metric}: {value}")
        
        # Export if requested
        if export:
            import json
            with open(export, 'w') as f:
                json.dump(analysis.to_dict(), f, indent=2)
            click.echo(f"\nEXPORTED: Results saved to {export}")
        
    except Exception as e:
        logger.error(f"Clustering failed: {e}")
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--mode', '-m', 
              type=click.Choice(['interactive', 'demo', 'debug']), 
              default='demo',
              help='Glass Engine mode: interactive (guided tutorial), demo (technical showcase), debug (raw system traces)')
@click.pass_context
async def tutorial(ctx: click.Context, mode: str) -> None:
    """
    Run the Glass Engine tutorial to see how Globule works under the hood.
    
    The Glass Engine provides three modes for different audiences:
    
    \b
    • INTERACTIVE: Guided tutorial with hands-on learning (best for new users)
    • DEMO: Professional technical showcase with automated examples (best for stakeholders)  
    • DEBUG: Raw execution traces and system introspection (best for engineers/debugging)
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


@cli.command()
@click.option('--auto', is_flag=True, help='Run reconciliation automatically without prompts')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed reconciliation output')
@click.pass_context
async def reconcile(ctx: click.Context, auto: bool, verbose: bool) -> None:
    """
    Reconcile files on disk with database records using UUID canonical links.
    
    This is the core of Priority 4: ensuring database records reflect the actual
    state of files on disk, regardless of how users have moved or renamed them.
    """
    # Use context verbose setting if not overridden
    verbose = verbose or ctx.obj.get('verbose', False)
    
    try:
        from globule.storage.file_manager import FileManager
        
        # Initialize context
        context: GlobuleContext = ctx.obj['context']
        await context.initialize(verbose)
        
        file_manager = FileManager()
        
        if not auto:
            click.echo("RECONCILIATION: This will scan all files and update database records to match disk reality.")
            click.echo("FILES: Users may have moved, renamed, or organized files differently.")
            click.echo("SYSTEM: The UUID in YAML frontmatter provides the canonical link.")
            click.echo()
            
            if not click.confirm("Proceed with reconciliation?"):
                click.echo("Reconciliation cancelled.")
                return
        
        click.echo("RECONCILIATION: Starting file-database reconciliation...")
        stats = await file_manager.reconcile_files_with_database(context.storage)
        
        # Display results
        click.echo(f"\nRECONCILIATION COMPLETE:")
        click.echo(f"  Files scanned: {stats['files_scanned']}")
        click.echo(f"  Files reconciled: {stats['files_reconciled']}")
        click.echo(f"  Database records updated: {stats['database_records_updated']}")
        click.echo(f"  Orphaned files (no UUID): {stats['files_orphaned']}")
        
        if stats['errors']:
            click.echo(f"  Errors encountered: {len(stats['errors'])}")
            if verbose:
                for error in stats['errors']:
                    click.echo(f"    - {error}")
        
        if stats['files_orphaned'] > 0:
            click.echo(f"\nORPHANED FILES: Found {stats['files_orphaned']} files without UUIDs.")
            click.echo("These files exist on disk but aren't tracked in the database.")
            click.echo("Use 'globule import' to add them to the system (when implemented).")
        
        if stats['database_records_updated'] > 0:
            click.echo(f"\nUPDATE: {stats['database_records_updated']} database records updated to reflect actual file locations.")
            click.echo("PRINCIPLE: The filename is for the human; the UUID is for the machine.")
        
    except Exception as e:
        logger.error(f"Reconciliation failed: {e}")
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


def main():
    """Entry point for the CLI"""
    # Convert click commands to async
    def async_command(f):
        def wrapper(*args, **kwargs):
            # Get context from click if available
            ctx = None
            for arg in args:
                if isinstance(arg, click.Context):
                    ctx = arg
                    break
            
            # Run the async command
            try:
                result = asyncio.run(f(*args, **kwargs))
                return result
            finally:
                # Clean up context if it exists
                if ctx and ctx.obj and 'context' in ctx.obj:
                    context = ctx.obj['context']
                    if context._initialized:
                        asyncio.run(context.cleanup())
    
        return wrapper
    
    # Make commands async-compatible
    cli.commands['add'].callback = async_command(cli.commands['add'].callback)
    cli.commands['draft'].callback = async_command(cli.commands['draft'].callback)
    cli.commands['search'].callback = async_command(cli.commands['search'].callback)
    cli.commands['cluster'].callback = async_command(cli.commands['cluster'].callback)
    cli.commands['tutorial'].callback = async_command(cli.commands['tutorial'].callback)
    cli.commands['reconcile'].callback = async_command(cli.commands['reconcile'].callback)
    
    cli()


if __name__ == '__main__':
    main()