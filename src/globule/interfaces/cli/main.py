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
@click.pass_context
async def draft(ctx: click.Context, topic: str, limit: int, output: Optional[str]) -> None:
    """
    Interactive drafting from clustered thoughts.
    
    Launches the TUI in a new terminal window while keeping logs in the original terminal.
    This provides a clean separation between the interactive UI and debugging information.
    """
    
    # Set up logging in CLI (for parent logs)
    verbose = ctx.obj.get('verbose', False)
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Prepare TUI command
    tui_module = 'globule.tui.app'
    tui_cmd = [sys.executable, '-m', tui_module, '--topic', topic]
    
    # Platform-specific launch for new window/tab
    launch_cmd = []
    if platform.system() == 'Darwin':  # macOS - Use AppleScript for new Terminal tab
        applescript = f'tell app "Terminal" to do script "{" ".join(tui_cmd)}"'
        launch_cmd = ['osascript', '-e', applescript]
    elif platform.system() == 'Linux':  # Linux - Use gnome-terminal (or xterm if preferred)
        launch_cmd = ['gnome-terminal', '--', *tui_cmd]
        # Alternative: ['xterm', '-e', *tui_cmd]
    elif platform.system() == 'Windows':  # Windows - New cmd window
        launch_cmd = ['cmd', '/c', 'start', 'cmd', '/k', *tui_cmd]
    else:
        logger.error("Unsupported platform - Launch TUI manually")
        click.echo(f"Unsupported platform: {platform.system()}", err=True)
        click.echo(f"Please run manually: {' '.join(tui_cmd)}")
        return
    
    # Launch and log
    try:
        import tempfile
        import os
        
        # Create a temporary log file for communication between processes
        log_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.log', prefix='globule_tui_')
        log_file_path = log_file.name
        log_file.close()
        
        # Modify TUI command to include log file path
        tui_cmd_with_log = tui_cmd + ['--log-file', log_file_path]
        
        # Update launch command for different platforms
        if platform.system() == 'Darwin':  # macOS
            applescript = f'tell app "Terminal" to do script "{" ".join(tui_cmd_with_log)}"'
            launch_cmd = ['osascript', '-e', applescript]
        elif platform.system() == 'Linux':  # Linux
            launch_cmd = ['gnome-terminal', '--', *tui_cmd_with_log]
        elif platform.system() == 'Windows':  # Windows
            launch_cmd = ['cmd', '/c', 'start', 'cmd', '/k', *tui_cmd_with_log]
        
        # Launch the TUI in new window
        process = subprocess.Popen(launch_cmd)
        
        logger.info(f"TUI launched in new window for topic '{topic}'. Monitoring logs here...")
        click.echo(f"[TUI] Launched in new window for topic '{topic}'")
        click.echo("[LOGS] Monitoring TUI activity (logs will appear below):")
        click.echo("=" * 50)
        
        # Monitor the log file and display new entries
        last_position = 0
        no_activity_count = 0
        max_no_activity = 120  # Stop after 60 seconds of no log activity (120 * 0.5s)
        
        try:
            while True:
                # Read new log entries from the file
                found_new_content = False
                try:
                    if os.path.exists(log_file_path) and os.path.getsize(log_file_path) > last_position:
                        with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            f.seek(last_position)
                            new_content = f.read()
                            if new_content:
                                found_new_content = True
                                # Display new log entries
                                for line in new_content.strip().split('\n'):
                                    if line.strip():
                                        click.echo(f"[TUI-LOG] {line.strip()}")
                                last_position = f.tell()
                                no_activity_count = 0  # Reset counter when we find new content
                except (FileNotFoundError, PermissionError):
                    # Log file might not exist yet or be locked
                    pass
                
                if not found_new_content:
                    no_activity_count += 1
                    if no_activity_count >= max_no_activity:
                        click.echo("[LOGS] No TUI activity detected for 60 seconds, stopping monitoring")
                        break
                
                await asyncio.sleep(0.5)  # Check every 500ms
                
        except KeyboardInterrupt:
            click.echo("\n[LOGS] Stopping log monitoring...")
        
        # Final log read
        try:
            if os.path.exists(log_file_path):
                with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    f.seek(last_position)
                    final_content = f.read()
                    if final_content:
                        for line in final_content.strip().split('\n'):
                            if line.strip():
                                click.echo(f"[TUI-LOG] {line.strip()}")
        except:
            pass
        
        # Clean up log file
        try:
            os.unlink(log_file_path)
        except:
            pass
            
        click.echo("[LOGS] TUI session monitoring ended")
            
    except Exception as e:
        logger.error(f"Failed to launch TUI: {e}")
        click.echo(f"[ERROR] Failed to launch TUI: {e}", err=True)
        
        # Fallback: try to run inline
        click.echo("[FALLBACK] Attempting fallback inline launch...")
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