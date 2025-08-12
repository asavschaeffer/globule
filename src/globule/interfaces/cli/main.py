"""
Main CLI commands for Globule.

Implements the core user experience:
- globule add "thought"
- globule draft "topic"

Refactored for performance and maintainability:
- Proper async command handling with click-asyncio
- Async context manager for resource management
- Clean separation of concerns with thin command wrappers
"""

import asyncio
import click
import logging
import subprocess
import platform
import sys
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

import asyncclick as click

from globule.core.models import EnrichedInput
from globule.storage.sqlite_manager import SQLiteStorageManager
from globule.services.embedding.ollama_provider import OllamaEmbeddingProvider
from globule.services.embedding.mock_provider import MockEmbeddingProvider
from globule.services.parsing.ollama_parser import OllamaParser
from globule.orchestration.engine import OrchestrationEngine, search_globules_nlp, fetch_globule_content
from globule.core.draft_manager import DraftManager
from globule.core.frontend_manager import frontend_manager, FrontendType
from globule.config.settings import get_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GlobuleContext:
    """Shared context for CLI commands with async context manager support."""
    
    def __init__(self):
        self.config = None
        self.storage = None
        self.embedding_provider = None
        self.parsing_provider = None
        self.orchestrator = None
        self._initialized = False
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with guaranteed cleanup."""
        await self.cleanup()
    
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
        if not self._initialized:
            return
            
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
async def cli(ctx: click.Context, verbose: bool):
    """
    Globule: Turn your scattered thoughts into structured drafts. Effortlessly.
    
    Capture thoughts with 'globule add' and synthesize with 'globule draft'.
    """
    # Initialize shared context
    ctx.ensure_object(dict)
    ctx.obj['context'] = GlobuleContext()
    ctx.obj['verbose'] = verbose


@click.command()
@click.argument('text', required=True)
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output for this command')
@click.pass_context
async def add(ctx: click.Context, text: str, verbose: bool) -> None:
    """Add a thought to your Globule collection."""
    
    # Use context verbose setting if not overridden
    verbose = verbose or ctx.obj.get('verbose', False)
    
    async with ctx.obj['context'] as context:
        try:
            # Initialize context if needed
            await context.initialize(verbose)
            
            # Create enriched input
            from globule.schemas.manager import get_schema_manager
            schema_manager = get_schema_manager()
            
            detected_schema_id = schema_manager.detect_schema_for_text(text)
            schema_config = None
            additional_context = {}

            if detected_schema_id:
                if click.confirm(f"Detected schema '{detected_schema_id}'. Use it?"):
                    schema_config = schema_manager.get_schema(detected_schema_id)
                else:
                    detected_schema_id = None

            enriched_input = EnrichedInput(
                original_text=text,
                enriched_text=text,
                detected_schema_id=detected_schema_id,
                schema_config=schema_config,
                additional_context=additional_context,
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


@click.command()
@click.argument('topic', required=True)
@click.option('--limit', '-l', default=100, help='Maximum globules to search (default: 100)')
@click.option('--output', '-o', help='Output draft to file')
@click.option('--frontend', '-f', type=click.Choice(['tui', 'web', 'cli'], case_sensitive=False), 
              default='tui', help='Frontend to use (tui, web, cli)')
@click.option('--port', '-p', default=8000, help='Port for web frontend (default: 8000)')
@click.option('--host', default='127.0.0.1', help='Host for web frontend (default: 127.0.0.1)')
@click.pass_context
async def draft(ctx: click.Context, topic: str, limit: int, output: Optional[str], 
                frontend: str, port: int, host: str) -> None:
    """
    Interactive drafting from clustered thoughts.
    
    Supports multiple frontend options:
    - tui: Terminal UI in new window (default)
    - web: Web interface in browser
    - cli: Stay in current terminal (non-interactive)
    """
    
    # Set up logging in CLI (for parent logs)
    verbose = ctx.obj.get('verbose', False)
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Convert frontend string to enum
    frontend_type = FrontendType(frontend.lower())
    
    logger.info(f"Launching {frontend} frontend for topic '{topic}'")
    click.echo(f"[FRONTEND] Launching {frontend.upper()} for topic '{topic}'...")
    
    # Handle CLI frontend (non-interactive mode)
    if frontend_type == FrontendType.CLI:
        click.echo("[CLI] Non-interactive mode - performing basic search and draft setup")
        try:
            async with ctx.obj['context'] as context:
                await context.initialize(verbose)
                
                # Perform search for the topic
                from globule.core.api import GlobuleAPI
                api = GlobuleAPI(context.storage)
                
                search_result = await api.search(topic)
                if search_result['success']:
                    click.echo("Search Results:")
                    click.echo("=" * 50)
                    click.echo(search_result['data'])
                    
                    # Optionally add to draft
                    if output:
                        draft_result = await api.export_draft(output)
                        if draft_result['success']:
                            click.echo(f"\n[SUCCESS] Draft exported to: {output}")
                        else:
                            click.echo(f"[ERROR] Export failed: {draft_result['message']}")
                else:
                    click.echo(f"[ERROR] Search failed: {search_result['message']}")
                    
        except Exception as e:
            logger.error(f"CLI mode failed: {e}")
            click.echo(f"[ERROR] CLI mode failed: {e}", err=True)
            raise click.Abort()
        return
    
    # Use frontend manager for TUI and Web frontends
    try:
        # Prepare launch arguments
        launch_kwargs = {
            'topic': topic,
            'limit': limit,
            'output': output
        }
        
        # Add web-specific arguments
        if frontend_type == FrontendType.WEB:
            launch_kwargs.update({
                'port': port,
                'host': host
            })
        
        # Launch the selected frontend
        result = await frontend_manager.launch_frontend(frontend_type, **launch_kwargs)
        
        if result['success']:
            click.echo(f"[SUCCESS] {result['message']}")
            
            # Handle web frontend - provide browser access info
            if frontend_type == FrontendType.WEB and 'data' in result and 'url' in result['data']:
                url = result['data']['url']
                click.echo(f"[WEB] Access the interface at: {url}")
                
                # Try to open browser automatically
                try:
                    import webbrowser
                    webbrowser.open(url)
                    click.echo(f"[WEB] Opened {url} in your default browser")
                except:
                    click.echo(f"[WEB] Please open {url} manually in your browser")
                
                # Keep the CLI alive for web server
                if result['data'].get('status') in ['launched', 'placeholder_launched']:
                    click.echo("[WEB] Web server running... Press Ctrl+C to stop")
                    try:
                        # Keep alive until user interrupts
                        await asyncio.sleep(float('inf'))
                    except KeyboardInterrupt:
                        click.echo("\n[WEB] Shutting down web server...")
                        # Clean up server if available
                        if 'server' in result['data']:
                            result['data']['server'].shutdown()
            
            # Handle TUI frontend - monitor logs as before
            elif frontend_type == FrontendType.TUI:
                click.echo("[TUI] Launched in new window")
                click.echo("[INFO] TUI is running in a separate window")
                
        else:
            click.echo(f"[ERROR] {result['message']}", err=True)
            raise click.Abort()
            
    except Exception as e:
        logger.error(f"Failed to launch {frontend} frontend: {e}")
        click.echo(f"[ERROR] Failed to launch {frontend} frontend: {e}", err=True)
        
        # Fallback for TUI only
        if frontend_type == FrontendType.TUI:
            click.echo("[FALLBACK] Attempting inline TUI launch...")
            try:
                async with ctx.obj['context'] as context:
                    from globule.tui.app import DashboardApp
                    
                    # Initialize context
                    await context.initialize(verbose)
                    
                    # Create and run the dashboard app inline as fallback
                    app = DashboardApp(context.storage, topic)
                    await app.run_async()
                    
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                click.echo(f"[ERROR] Fallback failed: {fallback_error}", err=True)
                raise click.Abort()
        else:
            raise click.Abort()


@click.command()
@click.pass_context
async def frontends(ctx: click.Context) -> None:
    """
    List available frontend options and their capabilities.
    
    Shows which frontends are available for the draft command and what
    features each one supports.
    """
    click.echo("Available Frontends:")
    click.echo("=" * 50)
    
    frontends_info = frontend_manager.list_available_frontends()
    
    for frontend_type, info in frontends_info.items():
        status = "[AVAILABLE]" if info['available'] else "[UNAVAILABLE]"
        click.echo(f"\n{status} {frontend_type.upper()}")
        click.echo(f"  Name: {info['name']}")
        click.echo(f"  Description: {info['description']}")
        click.echo(f"  Capabilities: {', '.join(info['capabilities'])}")
        
        if 'note' in info:
            click.echo(f"  Note: {info['note']}")
    
    click.echo("\nUsage:")
    click.echo("  globule draft 'topic' --frontend=tui    # Terminal interface (default)")
    click.echo("  globule draft 'topic' --frontend=web    # Web browser interface")
    click.echo("  globule draft 'topic' --frontend=cli    # Non-interactive CLI mode")


@click.command()
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
    
    async with ctx.obj['context'] as context:
        try:
            # Initialize context
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


@click.command()
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
    
    async with ctx.obj['context'] as context:
        try:
            # Initialize context
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


@click.command()
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


@click.command()
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
    
    async with ctx.obj['context'] as context:
        try:
            from globule.storage.file_manager import FileManager
            
            # Initialize context
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


# CLI Mirroring Commands - Expose TUI functionality via CLI
@click.command()
@click.argument('query', required=True)
@click.option('--output', '-o', help='Save results to file instead of displaying')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output for this command')
@click.pass_context
async def nlsearch(ctx: click.Context, query: str, output: Optional[str], verbose: bool) -> None:
    """
    Mirror TUI search: Natural language query to AI SQL to Formatted output.
    
    Provides scriptable access to the same AI-powered search functionality
    available in the TUI interface. Uses natural language processing to
    convert queries into SQL and returns formatted Markdown results.
    """
    verbose = verbose or ctx.obj.get('verbose', False)
    
    async with ctx.obj['context'] as context:
        try:
            await context.initialize(verbose)
            
            if verbose:
                click.echo(f"[SEARCH] Processing natural language query: {query}")
            
            # Use the reusable search function from orchestration engine
            result = await search_globules_nlp(query, context.storage)
            
            if output:
                # Save to file
                with open(output, 'w', encoding='utf-8') as f:
                    f.write(result)
                click.echo(f"Results saved to: {output}")
                if verbose:
                    click.echo(f"Generated {len(result)} characters of content")
            else:
                # Display to console
                click.echo(result)
            
        except Exception as e:
            logger.error(f"Natural language search failed: {e}")
            click.echo(f"Error: {e}", err=True)
            raise click.Abort()


@click.command()
@click.argument('item_id', required=True)
@click.option('--draft', '-d', default='drafts/current_draft.md', help='Draft file to add to (default: drafts/current_draft.md)')
@click.option('--section', '-s', help='Optional section title for the content')
@click.pass_context
async def add_to_draft(ctx: click.Context, item_id: str, draft: str, section: Optional[str]) -> None:
    """
    Mirror TUI Enter: Add globule/module by ID to draft file.
    
    Fetches a globule by its ID and appends it to a draft file, similar
    to pressing Enter on an item in the TUI palette. Enables scriptable
    draft composition from the command line.
    """
    verbose = ctx.obj.get('verbose', False)
    
    async with ctx.obj['context'] as context:
        try:
            await context.initialize(verbose)
            
            if verbose:
                click.echo(f"[DRAFT] Adding globule {item_id} to draft: {draft}")
            
            # Fetch globule content
            content = fetch_globule_content(item_id, context.storage)
            
            if not content:
                click.echo(f"Error: Item {item_id} not found", err=True)
                return
            
            # Add to draft using DraftManager
            manager = DraftManager(draft)
            bytes_added = manager.add_to_draft(content, section)
            
            click.echo(f"[SUCCESS] Added item {item_id} to {draft}")
            if verbose:
                click.echo(f"Added {bytes_added} bytes to draft")
                stats = manager.get_draft_stats()
                click.echo(f"Draft now has {stats['words']} words, {stats['lines']} lines")
            
        except Exception as e:
            logger.error(f"Add to draft failed: {e}")
            click.echo(f"Error: {e}", err=True)
            raise click.Abort()


@click.command()
@click.option('--draft', '-d', default='drafts/current_draft.md', help='Draft file to export (default: drafts/current_draft.md)')
@click.option('--output', '-o', default='exports/export.md', help='Output file path (default: exports/export.md)')
@click.option('--format', '-f', default='md', help='Export format (default: md)')
@click.pass_context
async def export_draft(ctx: click.Context, draft: str, output: str, format: str) -> None:
    """
    Mirror TUI export: Save draft to file in specified format.
    
    Exports the current draft content to a file, similar to the export
    functionality in the TUI. Supports different formats and enables
    automation of the drafting → export workflow.
    """
    verbose = ctx.obj.get('verbose', False)
    
    try:
        if verbose:
            click.echo(f"[EXPORT] Exporting draft {draft} to {output} in {format} format")
        
        # Use DraftManager for export
        manager = DraftManager(draft)
        
        # Check if draft exists
        stats = manager.get_draft_stats()
        if not stats['exists']:
            click.echo(f"Error: Draft file {draft} does not exist", err=True)
            return
        
        # Perform export
        success = manager.export_draft(output, format)
        
        if success:
            click.echo(f"[SUCCESS] Exported to {output}")
            if verbose:
                click.echo(f"Exported {stats['words']} words, {stats['lines']} lines")
        else:
            click.echo(f"Error: Export to {output} failed", err=True)
            raise click.Abort()
        
    except Exception as e:
        logger.error(f"Export draft failed: {e}")
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


@click.command()
@click.option('--draft', '-d', default='drafts/current_draft.md', help='Draft file to show stats for (default: drafts/current_draft.md)')
@click.pass_context
async def draft_stats(ctx: click.Context, draft: str) -> None:
    """
    Show statistics and information about a draft file.
    
    Provides detailed information about the current state of a draft,
    including word count, line count, file size, and modification time.
    """
    try:
        manager = DraftManager(draft)
        stats = manager.get_draft_stats()
        
        if not stats['exists']:
            click.echo(f"Draft file {draft} does not exist")
            return
        
        click.echo(f"[STATS] Draft Statistics: {draft}")
        click.echo(f"=========================================")
        click.echo(f"Words:        {stats['words']}")
        click.echo(f"Lines:        {stats['lines']}")
        click.echo(f"Size:         {stats['size']} bytes")
        click.echo(f"Modified:     {stats['modified']}")
        click.echo(f"Path:         {stats['path']}")
        
    except Exception as e:
        logger.error(f"Draft stats failed: {e}")
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


# Register all commands with the CLI group
cli.add_command(add)
cli.add_command(draft)
cli.add_command(frontends)
cli.add_command(search)
cli.add_command(cluster)
cli.add_command(tutorial)
cli.add_command(reconcile)

# Register new CLI mirroring commands
cli.add_command(nlsearch)
cli.add_command(add_to_draft)
cli.add_command(export_draft)
cli.add_command(draft_stats)


def main():
    """Entry point for the CLI - single responsibility: start the CLI."""
    cli()


if __name__ == '__main__':
    main()