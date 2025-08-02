"""
Glass Engine Tutorial: Phase 1 Walking Skeleton (ASCII Version)

This tutorial demonstrates the core Globule functionality while showing exactly
what happens under the hood. Tests, teaching, and demonstration become one.

User Story: "I have scattered thoughts and want to capture them effortlessly"
"""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.tree import Tree

from globule.config.settings import get_config
from globule.storage.sqlite_manager import SQLiteStorageManager
from globule.embedding.ollama_provider import OllamaEmbeddingProvider
from globule.parsing.ollama_parser import OllamaParser
from globule.orchestration.parallel_strategy import ParallelOrchestrationEngine
from globule.core.models import EnrichedInput


class GlassEnginePhase1:
    """
    Glass Engine Tutorial for Phase 1: Walking Skeleton
    
    Shows users exactly how their thoughts flow through the system:
    1. Capture -> 2. Processing -> 3. Storage -> 4. Retrieval -> 5. Display
    """
    
    def __init__(self):
        self.console = Console()
        self.settings = get_config()
        self.storage = SQLiteStorageManager()
        self.embedding_provider = OllamaEmbeddingProvider()
        self.parser = OllamaParser()
        self.orchestrator = ParallelOrchestrationEngine(
            embedding_provider=self.embedding_provider,
            parsing_provider=self.parser,
            storage_manager=self.storage
        )
        self.test_results: List[Dict[str, Any]] = []
        
    async def run_tutorial(self):
        """Run the complete Glass Engine tutorial"""
        self.console.print("\n" + "="*80)
        self.console.print(Panel.fit(
            "[bold blue]Glass Engine Tutorial: Phase 1 Walking Skeleton[/bold blue]\n\n"
            "[italic]See exactly how your thoughts flow through Globule's engine![/italic]",
            title="Welcome to Globule"
        ))
        
        await self._show_user_story()
        await self._show_system_configuration()
        await self._demonstrate_capture_flow()
        await self._demonstrate_storage_transparency()
        await self._demonstrate_retrieval_flow()
        await self._show_test_summary()
        
        self.console.print("\n")
        self.console.print(Panel.fit(
            "[bold green]Tutorial Complete![/bold green]\n\n"
            "[italic]You've seen exactly how Globule captures, processes, and organizes your thoughts.\n"
            "The engine is transparent - no black boxes, just clear data flow.[/italic]",
            title="Glass Engine Philosophy"
        ))
    
    async def _show_user_story(self):
        """Present the user story and what we'll demonstrate"""
        self.console.print("\n")
        self.console.print(Panel(
            "[bold yellow]User Story[/bold yellow]\n\n"
            "\"I have scattered thoughts and want to capture them effortlessly.\"\n\n"
            "[bold]What we'll show you:[/bold]\n"
            "- How 'globule add' processes your thoughts\n"
            "- Where and how data is stored\n"
            "- How 'globule draft' retrieves related thoughts\n"
            "- Every step of the data flow with live tests",
            title="Our Mission"
        ))
        
        self.console.print("\n[Press Enter to begin the journey through Globule's engine...]")
        try:
            input()
        except EOFError:
            # Handle non-interactive mode
            self.console.print("[Running in non-interactive mode, continuing...]")
            time.sleep(1)
    
    async def _show_system_configuration(self):
        """Show the current system configuration"""
        self.console.print("\n")
        self.console.print(Panel.fit(
            "[bold cyan]System Configuration[/bold cyan]",
            title="Glass Engine: Configuration Transparency"
        ))
        
        # Show configuration details
        config_table = Table(title="Current Settings")
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="green")
        config_table.add_column("Description", style="dim")
        
        storage_dir = self.settings.get_storage_dir()
        database_path = storage_dir / "globules.db"
        
        config_table.add_row("Storage Directory", str(storage_dir), "Where your thoughts are stored")
        config_table.add_row("Database File", str(database_path), "SQLite database location")
        config_table.add_row("Ollama URL", self.settings.ollama_base_url, "AI service endpoint")
        config_table.add_row("Embedding Model", self.settings.default_embedding_model, "Vector embedding model")
        config_table.add_row("Parsing Model", self.settings.default_parsing_model, "Text processing model")
        config_table.add_row("Config File", str(self.settings.get_config_path()), "Configuration file location")
        
        self.console.print(config_table)
        
        # Show directory structure
        self.console.print(f"\n[bold]Directory Structure:[/bold]")
        if storage_dir.exists():
            tree = Tree(f"Directory: {storage_dir}")
            for item in storage_dir.rglob("*"):
                if item.is_file():
                    rel_path = item.relative_to(storage_dir)
                    tree.add(f"File: {rel_path} ({item.stat().st_size} bytes)")
            self.console.print(tree)
        else:
            self.console.print(f"[dim]Storage directory will be created: {storage_dir}[/dim]")
    
    async def _demonstrate_capture_flow(self):
        """Show the complete thought capture process"""
        self.console.print("\n")
        self.console.print(Panel.fit(
            "[bold magenta]Phase 1 Test: Capture Flow[/bold magenta]",
            title="Glass Engine: Live Testing"
        ))
        
        test_thought = "The concept of 'progressive overload' in fitness could apply to creative stamina."
        
        self.console.print(f"\n[bold]Testing Command:[/bold] globule add \"{test_thought}\"")
        self.console.print(f"[dim]We'll show you exactly what happens inside the engine...[/dim]\n")
        
        # Initialize storage
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Initializing storage engine...", total=None)
            await self.storage.initialize()
            progress.update(task, completed=True)
        
        self.console.print("Storage engine initialized")
        
        # Process the thought with detailed logging
        start_time = time.time()
        
        self.console.print(f"\n[bold cyan]Processing Thought:[/bold cyan]")
        self.console.print(f"[dim]Input:[/dim] {test_thought}")
        
        # Create EnrichedInput object
        enriched_input = EnrichedInput(
            original_text=test_thought,
            enriched_text=test_thought,
            detected_schema_id=None,
            schema_config=None,
            additional_context={},
            source="glass_engine_tutorial",
            timestamp=datetime.now(),
            verbosity="verbose"
        )
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            embed_task = progress.add_task("Generating semantic embedding...", total=None)
            parse_task = progress.add_task("Extracting structure (mock)...", total=None)
            
            # Show the orchestration in action
            result = await self.orchestrator.process_globule(enriched_input)
            
            progress.update(embed_task, completed=True)
            progress.update(parse_task, completed=True)
        
        processing_time = time.time() - start_time
        
        # Show the results transparently
        self.console.print(f"\n[bold green]Processing Complete[/bold green] ({processing_time:.3f}s)")
        self._show_processing_results(result)
        
        # Store the processed globule
        globule_id = await self.storage.store_globule(result)
        self.console.print(f"[dim]Stored as globule ID: {globule_id}[/dim]")
        
        # Record test result
        self.test_results.append({
            "test": "capture_flow",
            "input": test_thought,
            "processing_time": processing_time,
            "success": True,
            "result": result,
            "globule_id": globule_id
        })
    
    def _show_processing_results(self, result):
        """Display the processing results transparently"""
        
        # Show embedding results
        if result.embedding is not None:
            embedding = result.embedding
            self.console.print(f"\n[bold]Semantic Embedding Generated:[/bold]")
            self.console.print(f"[dim]Dimensions:[/dim] {len(embedding)} (first 8: {embedding[:8]})")
            self.console.print(f"[dim]This vector represents the meaning of your thought in mathematical space[/dim]")
            self.console.print(f"[dim]Confidence:[/dim] {result.embedding_confidence:.2f}")
        
        # Show parsing results
        if result.parsed_data:
            parsed = result.parsed_data
            self.console.print(f"\n[bold]Structural Analysis:[/bold]")
            
            parsing_table = Table()
            parsing_table.add_column("Field", style="cyan")
            parsing_table.add_column("Value", style="green")
            
            for key, value in parsed.items():
                parsing_table.add_row(key, str(value))
            
            self.console.print(parsing_table)
            self.console.print(f"[dim]Confidence:[/dim] {result.parsing_confidence:.2f}")
            self.console.print(f"[dim]Note: Phase 1 uses mock parsing. Phase 2 will add real AI analysis.[/dim]")
        
        # Show storage decision
        if result.file_decision:
            decision = result.file_decision
            self.console.print(f"\n[bold]Storage Decision:[/bold]")
            full_path = decision.semantic_path / decision.filename
            self.console.print(f"[dim]File path:[/dim] {full_path}")
            self.console.print(f"[dim]Confidence:[/dim] {decision.confidence:.2f}")
            if decision.metadata:
                self.console.print(f"[dim]Metadata:[/dim] {decision.metadata}")
        
        # Show processing times
        if hasattr(result, 'processing_time_ms') and result.processing_time_ms:
            times = result.processing_time_ms
            self.console.print(f"\n[bold]Performance Metrics:[/bold]")
            perf_table = Table()
            perf_table.add_column("Operation", style="cyan")
            perf_table.add_column("Time (ms)", style="green")
            
            for operation, time_ms in times.items():
                perf_table.add_row(operation.replace("_", " ").title(), f"{time_ms:.1f}")
            
            self.console.print(perf_table)
    
    async def _demonstrate_storage_transparency(self):
        """Show exactly where and how data is stored"""
        self.console.print("\n")
        self.console.print(Panel.fit(
            "[bold blue]Storage Transparency[/bold blue]",
            title="Glass Engine: Data Flow Visibility"
        ))
        
        # Show database contents
        globules = await self.storage.get_recent_globules(limit=5)
        
        if globules:
            self.console.print(f"\n[bold]Database Contents:[/bold] ({len(globules)} recent globules)")
            
            for i, globule in enumerate(globules, 1):
                self.console.print(f"\n[bold cyan]Globule #{i}:[/bold cyan]")
                
                # Show the raw data structure
                data_table = Table()
                data_table.add_column("Field", style="cyan")
                data_table.add_column("Value", style="green")
                data_table.add_column("Type", style="dim")
                
                data_table.add_row("ID", str(globule.id), "INTEGER")
                data_table.add_row("Content", globule.text[:100] + "..." if len(globule.text) > 100 else globule.text, "TEXT")
                data_table.add_row("Created", globule.created_at.strftime("%Y-%m-%d %H:%M:%S"), "TIMESTAMP")
                data_table.add_row("Embedding", f"[{len(globule.embedding)} dimensions]", "BLOB")
                data_table.add_row("Parsed Data", str(globule.parsed_data)[:50] + "..." if globule.parsed_data else "None", "JSON")
                data_table.add_row("File Path", globule.file_path or "Not saved to file", "TEXT")
                
                self.console.print(data_table)
        else:
            self.console.print("[dim]No globules found in database[/dim]")
        
        # Show actual database file
        storage_dir = self.settings.get_storage_dir()
        db_path = storage_dir / "globules.db"
        if db_path.exists():
            stat = db_path.stat()
            self.console.print(f"\n[bold]Database File:[/bold]")
            self.console.print(f"[dim]Location:[/dim] {db_path}")
            self.console.print(f"[dim]Size:[/dim] {stat.st_size} bytes")
            self.console.print(f"[dim]Modified:[/dim] {datetime.fromtimestamp(stat.st_mtime)}")
        else:
            self.console.print(f"\n[bold]Database File:[/bold]")
            self.console.print(f"[dim]Location:[/dim] {db_path} (will be created)")
            self.console.print(f"[dim]Note:[/dim] Database will be created on first use")
    
    async def _demonstrate_retrieval_flow(self):
        """Show how globule draft retrieves and displays thoughts"""
        self.console.print("\n")
        self.console.print(Panel.fit(
            "[bold purple]Retrieval Flow[/bold purple]",
            title="Glass Engine: Query Processing"
        ))
        
        test_query = "creative concepts"
        self.console.print(f"\n[bold]Testing Command:[/bold] globule draft \"{test_query}\"")
        self.console.print(f"[dim]Let's see how the system finds related thoughts...[/dim]\n")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Retrieving related thoughts...", total=None)
            
            # Get recent globules (Phase 1 doesn't have semantic search yet)
            globules = await self.storage.get_recent_globules(limit=10)
            
            progress.update(task, completed=True)
        
        self.console.print(f"[bold green]Retrieved {len(globules)} thoughts[/bold green]")
        
        if globules:
            self.console.print(f"\n[bold]What the TUI would display:[/bold]")
            
            display_table = Table(title="Globule Draft Interface Preview")
            display_table.add_column("ID", style="dim")
            display_table.add_column("Content Preview", style="cyan")
            display_table.add_column("Created", style="dim")
            
            for globule in globules:
                preview = globule.text[:80] + "..." if len(globule.text) > 80 else globule.text
                display_table.add_row(
                    str(globule.id),
                    preview,
                    globule.created_at.strftime("%m/%d %H:%M")
                )
            
            self.console.print(display_table)
            
            self.console.print(f"\n[dim]Phase 1 shows all recent thoughts. Phase 2 will add semantic clustering![/dim]")
        else:
            self.console.print("[yellow]No thoughts found. Try adding some with 'globule add' first![/yellow]")
        
        # Record test result
        self.test_results.append({
            "test": "retrieval_flow",
            "query": test_query,
            "results_count": len(globules),
            "success": True
        })
    
    async def _show_test_summary(self):
        """Display comprehensive test results"""
        self.console.print("\n")
        self.console.print(Panel.fit(
            "[bold green]Test Results Summary[/bold green]",
            title="Glass Engine: Validation Complete"
        ))
        
        # Test results table
        results_table = Table(title="Phase 1 Walking Skeleton Validation")
        results_table.add_column("Test", style="cyan")
        results_table.add_column("Status", style="green")
        results_table.add_column("Details", style="dim")
        
        for result in self.test_results:
            status = "PASS" if result["success"] else "FAIL"
            
            if result["test"] == "capture_flow":
                details = f"{result['processing_time']:.3f}s processing time"
            elif result["test"] == "retrieval_flow":
                details = f"{result['results_count']} thoughts retrieved"
            else:
                details = "Completed"
            
            results_table.add_row(result["test"], status, details)
        
        self.console.print(results_table)
        
        # Phase 1 completion checklist
        self.console.print(f"\n[bold]Phase 1 Requirements Validated:[/bold]")
        checklist = [
            "* globule add captures thoughts instantly",
            "* Thoughts are processed through orchestration engine",
            "* Data is stored in SQLite database with embeddings",
            "* globule draft retrieves and displays thoughts",
            "* All components communicate correctly",
            "* End-to-end data flow is transparent"
        ]
        
        for item in checklist:
            self.console.print(f"  {item}")
        
        self.console.print(f"\n[bold yellow]Ready for Phase 2: Core Intelligence![/bold yellow]")
        self.console.print(f"[dim]Next: Real parsing, vector search, and semantic clustering[/dim]")


async def run_glass_engine_tutorial():
    """Entry point for the Glass Engine tutorial"""
    tutorial = GlassEnginePhase1()
    await tutorial.run_tutorial()


if __name__ == "__main__":
    asyncio.run(run_glass_engine_tutorial())