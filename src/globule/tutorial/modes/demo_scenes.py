"""
Scenes for the Glass Engine's Demo Mode.

This module contains stateless functions responsible for rendering the different
stages of the demo tutorial using the rich library.
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn

from globule.core.api import GlobuleAPI
from globule.core.models import ProcessedGlobuleV1

def show_intro(console: Console):
    console.print("\n" + "="*80)
    console.print(Panel.fit(
        "[bold blue]Glass Engine Tutorial: Demo Mode[/bold blue]\n\n" +
        "[italic]See exactly how your thoughts flow through Globule's engine![/italic]",
        title="Welcome to Globule"
    ))
    console.print(Panel(
        "[bold yellow]👤 User Story[/bold yellow]\n\n" +
        "\"I have scattered thoughts and want to capture them effortlessly.\"\n\n" +
        "[bold]What we'll show you:[/bold]\n" +
        "• How 'globule add' processes your thoughts\n" +
        "• How the API provides a clean interface\n" +
        "• How data is stored and retrieved",
        title="Our Mission"
    ))
    input("\n[Press Enter to begin...]")

def show_system_configuration(console: Console, api: GlobuleAPI):
    console.print("\n" + Panel.fit(
        "[bold cyan]⚙️  System Configuration[/bold cyan]",
        title="Glass Engine: Configuration Transparency"
    ))
    # In the new architecture, the API abstracts away the direct config details.
    # We can show high-level info.
    console.print(f"[bold]API Endpoint:[/bold] {api.__class__.__name__}")
    console.print(f"[bold]Storage Backend:[/bold] {api.storage.__class__.__name__}")
    console.print(f"[bold]Orchestrator:[/bold] {api.orchestrator.__class__.__name__}")

async def demonstrate_capture_flow(console: Console, api: GlobuleAPI) -> ProcessedGlobuleV1:
    console.print("\n" + Panel.fit(
        "[bold magenta]🎯 Test: Capture Flow via API[/bold magenta]",
        title="Glass Engine: Live Testing"
    ))
    test_thought = "The concept of 'progressive overload' in fitness could apply to creative stamina."
    console.print(f"\n[bold]Action:[/bold] Calling `api.add_thought(\"{test_thought}\")`")

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        progress.add_task("Processing thought...", total=None)
        processed_globule = await api.add_thought(test_thought, source="glass_engine_demo")
    
    console.print("[bold green]✅ API call complete.[/bold green]")
    show_processing_results(console, processed_globule)
    return processed_globule

def show_processing_results(console: Console, result: ProcessedGlobuleV1):
    console.print(f"\n[bold]Returned Object:[/bold] `ProcessedGlobuleV1`")
    
    if result.embedding:
        console.print(f"[dim]  - Embedding:[/dim] {len(result.embedding)} dimensions")
    if result.parsed_data:
        console.print(f"[dim]  - Parsed Data:[/dim] {list(result.parsed_data.keys())}")
    if result.file_decision:
        path = f"{result.file_decision.semantic_path}/{result.file_decision.filename}"
        console.print(f"[dim]  - File Decision:[/dim] {path}")

async def demonstrate_retrieval_flow(console: Console, api: GlobuleAPI):
    console.print("\n" + Panel.fit(
        "[bold purple]🔍 Test: Retrieval Flow via API[/bold purple]",
        title="Glass Engine: Query Processing"
    ))
    test_query = "creative concepts"
    console.print(f"\n[bold]Action:[/bold] Calling `api.search_semantic(\"{test_query}\")`")

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        progress.add_task("Searching for related thoughts...", total=None)
        results = await api.search_semantic(test_query, limit=5)

    console.print(f"[bold green]✅ API call complete. Found {len(results)} results.[/bold green]")

    if results:
        table = Table(title="Semantic Search Results")
        table.add_column("ID", style="dim")
        table.add_column("Content Preview", style="cyan")
        for globule in results:
            table.add_row(str(globule.globule_id)[:8], globule.original_globule.raw_text[:80] + "...")
        console.print(table)

def show_final_summary(console: Console):
    console.print("\n" + Panel.fit(
        "[bold green]✅ Demo Complete![/bold green]\n\n" +
        "[italic]You've seen how the GlobuleAPI provides a clean, powerful interface\n" +
        "to the core features of the application.[/italic]",
        title="Glass Engine Philosophy"
    ))