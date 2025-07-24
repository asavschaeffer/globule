"""
Simple Demo Glass Engine Mode - Minimal Implementation

This is a simplified version of the demo mode that focuses on core functionality
without complex formatting to avoid syntax issues.
"""

import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from globule.tutorial.glass_engine_core import AbstractGlassEngine, GlassEngineMode
from globule.core.models import EnrichedInput


class SimpleDemoGlassEngine(AbstractGlassEngine):
    """
    Simplified Demo Glass Engine implementation for technical showcases.
    
    This class provides a basic demonstration of Globule's capabilities
    with minimal formatting complexity.
    """
    
    def __init__(self, console: Optional[Console] = None):
        """Initialize the Simple Demo Glass Engine."""
        super().__init__(console)
        self.demo_scenarios = [
            {
                "category": "Creative Writing",
                "input": "The concept of 'progressive overload' in fitness could apply to creative stamina. Just as muscles grow stronger when gradually challenged, perhaps our creative capacity expands when we consistently push slightly beyond our comfort zone.",
                "context": "Cross-domain thinking and metaphorical reasoning - showcasing creative domain detection"
            },
            {
                "category": "Technical Insight", 
                "input": "Instead of preventing all edge cases, design systems that gracefully degrade. When the unexpected happens, the system should fail in a predictable, controlled manner rather than catastrophically.",
                "context": "Systems thinking and resilience engineering - showcasing technical domain detection"
            },
            {
                "category": "Question Analysis",
                "input": "How can we measure the effectiveness of knowledge management systems in creative workflows?",
                "context": "Interrogative content analysis - showcasing question categorization"
            },
            {
                "category": "Personal Reflection",
                "input": "I feel like I'm constantly switching between different tools for note-taking, and it's becoming overwhelming. Need a unified system that actually works for my scattered thinking.",
                "context": "Personal domain classification and sentiment analysis"
            }
        ]
        
    def get_mode(self) -> GlassEngineMode:
        """Return the Demo Glass Engine mode."""
        return GlassEngineMode.DEMO
    
    async def execute_tutorial_flow(self) -> None:
        """Execute the simplified demo tutorial flow."""
        self.logger.info("Starting simplified demo showcase")
        
        # Phase 1: Welcome
        self.console.print(Panel.fit(
            "[bold blue]Globule Demo: Professional System Showcase[/bold blue]",
            title="Glass Engine Demo Mode"
        ))
        
        # Phase 2: Configuration
        await self._show_configuration()
        
        # Phase 3: Process demo scenarios
        await self._process_demo_scenarios()
        
        # Phase 4: Results
        self.console.print(Panel.fit(
            "[bold green]Demo Complete![/bold green]",
            title="Success"
        ))
        
        self.logger.info("Simplified demo showcase completed")
    
    async def _show_configuration(self) -> None:
        """Show basic system configuration."""
        self.console.print("\n[bold]System Configuration:[/bold]")
        
        config_table = Table(title="Current Settings")
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="green")
        
        storage_dir = self.config.get_storage_dir()
        config_table.add_row("Storage Directory", str(storage_dir))
        config_table.add_row("Ollama URL", self.config.ollama_base_url)
        config_table.add_row("Embedding Model", self.config.default_embedding_model)
        
        self.console.print(config_table)
    
    async def _process_demo_scenarios(self) -> None:
        """Process the demo scenarios."""
        self.console.print("\n[bold]Processing Demo Scenarios:[/bold]")
        
        for i, scenario in enumerate(self.demo_scenarios, 1):
            self.console.print(f"\nScenario {i}: {scenario['category']}")
            self.console.print(f"Input: {scenario['input']}")
            
            try:
                # Create enriched input
                enriched_input = self.create_test_input(
                    scenario["input"], 
                    f"demo_scenario_{i}"
                )
                
                # Process with timing
                start_time = datetime.now()
                result = await self.orchestrator.process_globule(enriched_input)
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                
                # Store the result
                globule_id = await self.storage.store_globule(result)
                
                # Show results
                self.console.print(f"[green]SUCCESS[/green] Processed in {processing_time:.1f}ms")
                self.console.print(f"[green]SUCCESS[/green] Stored as globule: {str(globule_id)[:8]}...")
                
                if result.embedding is not None:
                    self.console.print(f"[green]SUCCESS[/green] Generated {len(result.embedding)}-dimensional embedding")
                
                # Show intelligent parsing results (Phase 2 feature)
                if result.parsed_data:
                    parsed = result.parsed_data
                    self.console.print(f"[cyan]INTELLIGENCE[/cyan] Title: '{parsed.get('title', 'N/A')}'")
                    self.console.print(f"[cyan]INTELLIGENCE[/cyan] Domain: {parsed.get('domain', 'N/A')} | Category: {parsed.get('category', 'N/A')}")
                    
                    # Show keywords if available
                    keywords = parsed.get('keywords', [])
                    if keywords:
                        self.console.print(f"[cyan]INTELLIGENCE[/cyan] Keywords: {', '.join(keywords[:3])}")
                    
                    # Show metadata confidence and type
                    metadata = parsed.get('metadata', {})
                    if metadata:
                        parser_type = metadata.get('parser_type', 'unknown')
                        confidence = metadata.get('confidence_score', 0)
                        self.console.print(f"[cyan]INTELLIGENCE[/cyan] Parser: {parser_type} (confidence: {confidence:.2f})")
                        
                        if 'sentiment' in metadata:
                            self.console.print(f"[cyan]INTELLIGENCE[/cyan] Sentiment: {metadata['sentiment']}")
                
                # Record test result
                self.metrics.test_results.append({
                    "test": f"demo_scenario_{i}",
                    "input": scenario["input"],
                    "success": True,
                    "processing_time_ms": processing_time,
                    "globule_id": str(globule_id)
                })
                
            except Exception as e:
                self.console.print(f"[red]ERROR[/red]: {e}")
                self.metrics.add_error(e, f"demo_scenario_{i}")
                self.metrics.test_results.append({
                    "test": f"demo_scenario_{i}",
                    "input": scenario["input"],
                    "success": False,
                    "error": str(e)
                })
    
    def present_results(self) -> None:
        """Present demo results in a simple format."""
        self.console.print("\n" + "=" * 60)
        self.console.print(Panel.fit(
            "[bold blue]Demo Results Summary[/bold blue]",
            title="Glass Engine Results"
        ))
        
        # Test results
        results_table = Table(title="Test Results")
        results_table.add_column("Test", style="cyan")
        results_table.add_column("Status", style="green") 
        results_table.add_column("Time (ms)", style="yellow")
        
        for result in self.metrics.test_results:
            status = "PASS" if result.get("success", False) else "FAIL"
            time_ms = result.get("processing_time_ms", 0)
            results_table.add_row(
                result.get("test", "unknown"),
                status,
                f"{time_ms:.1f}" if time_ms else "N/A"
            )
        
        self.console.print(results_table)
        
        # Summary
        success_count = sum(1 for r in self.metrics.test_results if r.get("success", False))
        total_count = len(self.metrics.test_results)
        
        self.console.print(f"\nOverall Success Rate: {success_count}/{total_count}")
        self.console.print(f"Total Duration: {self.metrics.total_duration_ms:.1f}ms")