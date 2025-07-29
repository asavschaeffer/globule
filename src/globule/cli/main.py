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
from pathlib import Path

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
        # Store using the atomic Outbox Pattern - storage manager handles file creation internally
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
@click.option('--output', '-o', help='Output draft to file')
async def draft(topic: Optional[str], limit: int, output: Optional[str]) -> None:
    """Generate draft from your thoughts using semantic clustering."""
    
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich.text import Text
        
        console = Console()
        
        # Initialize components
        storage = SQLiteStorageManager()
        await storage.initialize()
        
        console.print("[blue]DRAFT:[/blue] Analyzing your thoughts...")
        
        # Get recent globules
        globules = await storage.get_recent_globules(limit)
        
        if not globules:
            console.print("[red]NO CONTENT:[/red] No thoughts found. Add some with 'globule add'")
            return
        
        # Filter by topic if provided
        if topic:
            embedding_provider = OllamaEmbeddingProvider()
            topic_embedding = await embedding_provider.embed(topic)
            results = await storage.search_by_embedding(topic_embedding, limit, 0.3)
            globules = [globule for globule, _ in results]
            console.print(f"[green]FILTERED:[/green] Found {len(globules)} thoughts related to '{topic}'")
            await embedding_provider.close()
        
        # Perform clustering
        from globule.clustering.semantic_clustering import SemanticClusteringEngine
        clustering_engine = SemanticClusteringEngine(storage)
        analysis = await clustering_engine.analyze_semantic_clusters(min_globules=2)
        
        # Generate draft content
        draft_content = []
        draft_content.append(f"# Draft: {topic or 'My Thoughts'}\n")
        draft_content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        draft_content.append(f"Total thoughts analyzed: {len(globules)}\n\n")
        
        if analysis.clusters:
            console.print(f"[green]CLUSTERS:[/green] Found {len(analysis.clusters)} semantic themes")
            
            for i, cluster in enumerate(analysis.clusters, 1):
                draft_content.append(f"## {cluster.label}\n")
                draft_content.append(f"{cluster.description}\n\n")
                
                # Add representative content
                cluster_globules = [g for g in globules if g.id in cluster.member_ids]
                for globule in cluster_globules[:3]:  # Top 3 from cluster
                    draft_content.append(f"- {globule.text}\n")
                
                draft_content.append("\n")
                
                # Show cluster summary
                table = Table(title=f"Cluster {i}: {cluster.label}")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="green")
                table.add_row("Size", str(cluster.size))
                table.add_row("Confidence", f"{cluster.confidence_score:.2f}")
                table.add_row("Keywords", ", ".join(cluster.keywords[:3]))
                table.add_row("Domains", ", ".join(cluster.domains))
                console.print(table)
        else:
            # No clusters, just list thoughts chronologically
            draft_content.append("## Your Thoughts\n\n")
            for globule in globules[:20]:  # Top 20
                draft_content.append(f"- {globule.text}\n")
            
            console.print("[yellow]NO CLUSTERS:[/yellow] Listed thoughts chronologically")
        
        # Output draft
        draft_text = "".join(draft_content)
        
        if output:
            with open(output, 'w', encoding='utf-8') as f:
                f.write(draft_text)
            console.print(f"[green]SUCCESS:[/green] Draft written to {output}")
        else:
            console.print("\n" + "="*60)
            console.print(Panel(draft_text, title="Generated Draft", border_style="blue"))
        
        # Show summary
        console.print(f"\n[blue]SUMMARY:[/blue] Draft generated from {len(globules)} thoughts")
        if analysis.clusters:
            console.print(f"[blue]THEMES:[/blue] Organized into {len(analysis.clusters)} semantic clusters")
        
        # Cleanup
        await storage.close()
        
    except Exception as e:
        logger.error(f"Failed to generate draft: {e}")
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
@click.option('--min-globules', '-m', default=5, help='Minimum globules required for clustering')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed cluster analysis')
@click.option('--export', '-e', help='Export results to JSON file')
async def cluster(min_globules: int, verbose: bool, export: Optional[str]) -> None:
    """
    Discover semantic clusters and themes in your thoughts.
    
    This command demonstrates Phase 2 clustering capabilities by automatically
    grouping related content and identifying common themes across your knowledge base.
    
    Examples:
    \b
    globule cluster                          # Basic clustering
    globule cluster --min-globules 3        # Lower threshold
    globule cluster --verbose               # Detailed analysis
    globule cluster --export clusters.json  # Save results
    """
    try:
        # Initialize components
        storage = SQLiteStorageManager()
        await storage.initialize()
        
        from globule.clustering.semantic_clustering import SemanticClusteringEngine
        clustering_engine = SemanticClusteringEngine(storage)
        
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
        
        # Cleanup
        await storage.close()
        
    except Exception as e:
        logger.error(f"Clustering failed: {e}")
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


# REMOVED: save command is now redundant - files are created automatically during 'add'

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


@cli.command()
@click.option('--auto', is_flag=True, help='Run reconciliation automatically without prompts')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed reconciliation output')
async def reconcile(auto: bool, verbose: bool) -> None:
    """
    Reconcile files on disk with database records using UUID canonical links.
    
    This is the core of Priority 4: ensuring database records reflect the actual
    state of files on disk, regardless of how users have moved or renamed them.
    
    The system uses UUIDs embedded in YAML frontmatter as the canonical link
    between files and database records. This allows users to organize, rename,
    and move files freely while maintaining system integrity.
    
    Examples:
    \\b
    globule reconcile                    # Interactive reconciliation with prompts
    globule reconcile --auto            # Automatic reconciliation
    globule reconcile --auto --verbose  # Detailed output
    """
    try:
        from globule.storage.file_manager import FileManager
        
        # Initialize components
        storage = SQLiteStorageManager()
        await storage.initialize()
        
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
        stats = await file_manager.reconcile_files_with_database(storage)
        
        # Display results
        click.echo(f"\\nRECONCILIATION COMPLETE:")
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
            click.echo(f"\\nORPHANED FILES: Found {stats['files_orphaned']} files without UUIDs.")
            click.echo("These files exist on disk but aren't tracked in the database.")
            click.echo("Use 'globule import' to add them to the system (when implemented).")
        
        if stats['database_records_updated'] > 0:
            click.echo(f"\\nUPDATE: {stats['database_records_updated']} database records updated to reflect actual file locations.")
            click.echo("PRINCIPLE: The filename is for the human; the UUID is for the machine.")
        
        # Cleanup
        await storage.close()
        
    except Exception as e:
        logger.error(f"Reconciliation failed: {e}")
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
    cli.commands['cluster'].callback = async_command(cli.commands['cluster'].callback)
    # REMOVED: save command - files are now created automatically during 'add'
    cli.commands['tutorial'].callback = async_command(cli.commands['tutorial'].callback)
    cli.commands['reconcile'].callback = async_command(cli.commands['reconcile'].callback)
    
    cli()


if __name__ == '__main__':
    main()