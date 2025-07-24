"""
Interactive Glass Engine Mode

This module implements the Interactive mode of the Glass Engine, designed as a
pedagogical tutorial that guides users through hands-on learning while simultaneously
testing system functionality.

The Interactive mode embodies the Glass Engine philosophy by:
- Teaching users how Globule works through direct experience
- Validating system functionality through user-driven tests
- Building confidence through transparent, step-by-step explanations
- Encouraging exploration and experimentation

Target Audience: New users learning the Globule system
Primary Purpose: Education and onboarding with integrated testing
User Experience: Guided discovery with pause points for comprehension

Author: Globule Team
Date: 2025-07-24
Version: 1.0.0
"""

import asyncio
import sys
from typing import Optional, Dict, Any, List
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text
from rich.markdown import Markdown

from globule.tutorial.glass_engine_core import AbstractGlassEngine, GlassEngineMode
from globule.core.models import EnrichedInput


class InteractiveGlassEngine(AbstractGlassEngine):
    """
    Interactive Glass Engine implementation for pedagogical tutorials.
    
    This class provides a guided, step-by-step tutorial experience that teaches
    users how Globule works while validating system functionality through
    user-driven interactions.
    
    The interactive flow follows educational best practices:
    1. Present learning objectives and context
    2. Guide hands-on exploration with explanations
    3. Validate understanding through practical exercises
    4. Summarize key concepts and next steps
    
    Attributes:
        user_inputs: List of user inputs collected during the tutorial
        learning_checkpoints: Educational milestones reached
        personalization_data: User preferences and learning patterns
    """
    
    def __init__(self, console: Optional[Console] = None):
        """
        Initialize the Interactive Glass Engine.
        
        Args:
            console: Rich console for interactive output. If None, creates new console.
        """
        super().__init__(console)
        self.user_inputs: List[str] = []
        self.learning_checkpoints: List[str] = []
        self.personalization_data: Dict[str, Any] = {}
        
    def get_mode(self) -> GlassEngineMode:
        """Return the Interactive Glass Engine mode."""
        return GlassEngineMode.INTERACTIVE
    
    async def execute_tutorial_flow(self) -> None:
        """
        Execute the interactive tutorial flow.
        
        This method implements the core educational experience, guiding users
        through hands-on exploration of Globule's capabilities while building
        understanding and confidence.
        """
        self.logger.info("Starting interactive tutorial flow")
        
        # Phase 1: Welcome and Learning Objectives
        await self._present_welcome_and_objectives()
        
        # Phase 2: System Overview and Configuration
        await self._explore_system_configuration()
        
        # Phase 3: Hands-on Thought Capture
        await self._guided_thought_capture_exercise()
        
        # Phase 4: Understanding Data Flow
        await self._explore_data_processing_pipeline()
        
        # Phase 5: Retrieval and Synthesis
        await self._guided_retrieval_exercise()
        
        # Phase 6: Summary and Next Steps
        await self._provide_learning_summary()
        
        self.logger.info("Interactive tutorial flow completed")
    
    async def _present_welcome_and_objectives(self) -> None:
        """
        Present welcome message and establish learning objectives.
        
        This method sets the educational context and helps users understand
        what they will learn and accomplish during the tutorial.
        """
        self.console.print("\n" + "=" * 80)
        self.console.print(Panel.fit(
            "[bold blue]Welcome to Globule: Interactive Learning Tutorial[/bold blue]\n\n"
            "[italic]Learn by doing - we'll guide you through hands-on exploration![/italic]",
            title="Glass Engine Interactive Mode"
        ))
        
        # Present learning objectives
        objectives_md = """
## What You'll Learn Today

By the end of this tutorial, you'll understand:

1. **How Globule captures your thoughts** - The complete process from input to storage
2. **Where your data lives** - Transparent file organization and database structure  
3. **How AI processes your ideas** - Embedding generation and semantic understanding
4. **How to find related thoughts** - Retrieval and clustering mechanisms
5. **Your privacy and control** - Local-first architecture and data ownership

## Learning Approach

This is a **hands-on tutorial** where you'll:
- Type your own thoughts and see them processed live
- Explore the system with guided exercises
- Ask questions and get immediate answers
- See exactly how everything works under the hood

Ready to begin your journey?
        """
        
        self.console.print(Panel(Markdown(objectives_md), title="Your Learning Journey"))
        
        # Get user consent and personalization
        if not Confirm.ask("\nShall we begin the interactive tutorial?"):
            self.console.print("[yellow]Tutorial cancelled. You can restart anytime with 'globule tutorial --interactive'[/yellow]")
            return
        
        # Optional personalization
        if Confirm.ask("Would you like to personalize your learning experience?"):
            self.personalization_data["name"] = Prompt.ask("What should we call you?", default="Explorer")
            self.personalization_data["experience"] = Prompt.ask(
                "How familiar are you with note-taking tools?",
                choices=["beginner", "intermediate", "advanced"],
                default="beginner"
            )
            self.personalization_data["interests"] = Prompt.ask(
                "What do you primarily want to capture? (optional)",
                default="ideas and thoughts"
            )
        
        self.log_user_interaction("personalization", self.personalization_data)
        self.learning_checkpoints.append("objectives_presented")
    
    async def _explore_system_configuration(self) -> None:
        """
        Guide users through understanding Globule's configuration and architecture.
        
        This method helps users understand where their data will be stored,
        how the system is configured, and what components are involved.
        """
        name = self.personalization_data.get("name", "Explorer")
        
        self.console.print(f"\n[bold cyan]Great to meet you, {name}! Let's explore how Globule is set up on your system.[/bold cyan]")
        
        # Show configuration in educational context
        config_panel = Panel(
            f"[bold]Understanding Your Globule Configuration[/bold]\n\n"
            f"Globule stores everything locally on your computer for privacy and control.\n"
            f"Let's see where your thoughts will live:",
            title="Local-First Philosophy"
        )
        self.console.print(config_panel)
        
        # Interactive configuration exploration
        config_table = Table(title="Your Globule Configuration")
        config_table.add_column("Component", style="cyan")
        config_table.add_column("Location/Setting", style="green")
        config_table.add_column("Purpose", style="dim")
        
        storage_dir = self.config.get_storage_dir()
        config_file = self.config.get_config_path()
        
        config_table.add_row(
            "💾 Data Directory", 
            str(storage_dir), 
            "Your thoughts and embeddings"
        )
        config_table.add_row(
            "🗄️ Database File", 
            str(storage_dir / "globules.db"), 
            "SQLite database for fast search"
        )
        config_table.add_row(
            "⚙️ Configuration", 
            str(config_file), 
            "System settings and preferences"
        )
        config_table.add_row(
            "🤖 AI Service", 
            self.config.ollama_base_url, 
            "Local AI for processing (Ollama)"
        )
        
        self.console.print(config_table)
        
        # Educational explanation with interaction
        explanation = """
**Why Local-First Matters:**

• **Privacy**: Your thoughts never leave your computer
• **Control**: You own your data completely  
• **Speed**: No internet required for most operations
• **Reliability**: Works offline, always available
• **Transparency**: You can inspect everything
        """
        
        self.console.print(Panel(Markdown(explanation), title="Key Benefits"))
        
        # Check if user wants to explore further
        if Confirm.ask("Would you like to see the actual files on your system?"):
            await self._show_file_system_exploration()
        
        self.learning_checkpoints.append("configuration_explored")
        self.log_user_interaction("configuration_exploration")
    
    async def _show_file_system_exploration(self) -> None:
        """Show users their actual file system and explain the structure."""
        from rich.tree import Tree
        
        storage_dir = self.config.get_storage_dir()
        
        self.console.print(f"\n[bold]Let's look at your actual Globule files:[/bold]")
        
        if storage_dir.exists():
            tree = Tree(f"📁 {storage_dir}")
            for item in storage_dir.rglob("*"):
                if item.is_file():
                    rel_path = item.relative_to(storage_dir)
                    size = item.stat().st_size
                    if size < 1024:
                        size_str = f"{size} bytes"
                    elif size < 1024 * 1024:
                        size_str = f"{size/1024:.1f} KB"
                    else:
                        size_str = f"{size/(1024*1024):.1f} MB"
                    tree.add(f"📄 {rel_path} ({size_str})")
            self.console.print(tree)
        else:
            self.console.print("[dim]Directory will be created when you add your first thought![/dim]")
        
        self.console.print("\n[green]You can explore these files anytime with your regular file explorer![/green]")
    
    async def _guided_thought_capture_exercise(self) -> None:
        """
        Guide users through capturing their own thoughts.
        
        This is the core hands-on exercise where users input their own thoughts
        and see the complete processing pipeline in action.
        """
        name = self.personalization_data.get("name", "Explorer")
        interests = self.personalization_data.get("interests", "ideas and thoughts")
        
        self.console.print(f"\n[bold magenta]Now for the exciting part, {name}! Let's capture some of your {interests}.[/bold magenta]")
        
        # Educational context setting
        exercise_intro = """
## The Thought Capture Process

When you add a thought to Globule, here's what happens:

1. **Input Processing** - Your text is prepared for analysis
2. **Semantic Embedding** - AI converts meaning to mathematical vectors  
3. **Structural Parsing** - Extract topics, categories, and metadata
4. **File Organization** - Decide where to store it semantically
5. **Database Storage** - Save for fast future retrieval

Let's see this in action with your own thoughts!
        """
        
        self.console.print(Panel(Markdown(exercise_intro), title="Understanding the Process"))
        
        # Get user's thought with guidance
        thought_guidance = f"""
Think of something related to {interests} that you'd like to capture.

**Examples:**
• A quote that inspired you
• An idea for a project  
• A connection between concepts
• Something you learned today
• A question you're pondering

What would you like to add to your Globule?
        """
        
        self.console.print(Panel(thought_guidance, title="Your Turn to Think"))
        
        user_thought = Prompt.ask("[bold cyan]Enter your thought[/bold cyan]")
        self.user_inputs.append(user_thought)
        self.log_user_interaction("thought_input", {"thought": user_thought})
        
        # Process the thought with educational narration
        self.console.print(f"\n[bold]Excellent! Let's process: \"{user_thought[:50]}{'...' if len(user_thought) > 50 else ''}\"[/bold]")
        
        await self._demonstrate_processing_with_narration(user_thought)
        
        # Encourage reflection
        if Confirm.ask("\nWould you like to add another thought to see how they connect?"):
            second_thought = Prompt.ask("[bold cyan]Enter a related (or different) thought[/bold cyan]")
            self.user_inputs.append(second_thought)
            self.log_user_interaction("thought_input", {"thought": second_thought})
            await self._demonstrate_processing_with_narration(second_thought)
        
        self.learning_checkpoints.append("thought_capture_completed")
    
    async def _demonstrate_processing_with_narration(self, thought: str) -> None:
        """
        Process a thought while providing educational narration.
        
        Args:
            thought: The user's thought to process
        """
        # Create enriched input
        enriched_input = self.create_test_input(thought, "interactive_tutorial")
        
        self.console.print("\n[bold cyan]Step 1: Preparing your thought for processing...[/bold cyan]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            
            # Step 1: Show input preparation
            prep_task = progress.add_task("Preparing input structure...", total=None)
            await asyncio.sleep(0.5)  # Simulate processing time for education
            progress.update(prep_task, completed=True)
            
            self.console.print("✓ Input prepared with metadata and context")
            
            # Step 2: Processing with orchestration
            self.console.print("\n[bold cyan]Step 2: Generating semantic understanding...[/bold cyan]")
            
            embed_task = progress.add_task("AI is analyzing the meaning...", total=None)
            parse_task = progress.add_task("Extracting structure and topics...", total=None)
            
            # Actually process the globule
            async with self.performance_timer("thought_processing"):
                result = await self.orchestrator.process_globule(enriched_input)
            
            progress.update(embed_task, completed=True)
            progress.update(parse_task, completed=True)
            
            # Step 3: Store the result
            store_task = progress.add_task("Saving to your personal database...", total=None)
            globule_id = await self.storage.store_globule(result)
            progress.update(store_task, completed=True)
        
        # Educational explanation of what happened
        self.console.print("\n[bold green]✓ Processing Complete! Here's what happened:[/bold green]")
        
        # Show the results educationally
        self._explain_processing_results(result, globule_id)
        
        # Record the test result
        self.metrics.test_results.append({
            "test": "interactive_thought_processing",
            "input": thought,
            "globule_id": str(globule_id),
            "success": True,
            "processing_time_ms": self.metrics.performance_data.get("thought_processing", 0)
        })
    
    def _explain_processing_results(self, result, globule_id: str) -> None:
        """
        Explain processing results to the user in educational terms.
        
        Args:
            result: The ProcessedGlobule result
            globule_id: The stored globule ID
        """
        # Embedding explanation
        if result.embedding is not None:
            embedding_panel = Panel(
                f"[bold]🧠 Semantic Understanding Generated[/bold]\n\n"
                f"Your thought was converted into a {len(result.embedding)}-dimensional vector.\n"
                f"This mathematical representation captures the *meaning* of your words,\n"
                f"allowing Globule to find related thoughts even if they use different words.\n\n"
                f"[dim]Vector preview: {result.embedding[:5]}...[/dim]\n"
                f"[dim]Confidence: {result.embedding_confidence:.1%}[/dim]",
                title="AI Processing Result"
            )
            self.console.print(embedding_panel)
        
        # Parsing explanation
        if result.parsed_data:
            parsing_table = Table(title="🏷️ Structure Analysis")
            parsing_table.add_column("Element", style="cyan")
            parsing_table.add_column("Value", style="green")
            parsing_table.add_column("Why This Matters", style="dim")
            
            title = result.parsed_data.get("title", "")[:50]
            parsing_table.add_row("Title", title, "For quick identification")
            
            category = result.parsed_data.get("category", "note")
            parsing_table.add_row("Category", category, "For organization")
            
            domain = result.parsed_data.get("domain", "general")
            parsing_table.add_row("Domain", domain, "For semantic grouping")
            
            self.console.print(parsing_table)
        
        # Storage explanation
        if result.file_decision:
            storage_panel = Panel(
                f"[bold]💾 Smart Storage Decision[/bold]\n\n"
                f"Your thought would be saved as:\n"
                f"[cyan]{result.file_decision.semantic_path / result.file_decision.filename}[/cyan]\n\n"
                f"This path was chosen based on the content and structure analysis.\n"
                f"You can always reorganize later if needed!\n\n"
                f"[dim]Storage confidence: {result.file_decision.confidence:.1%}[/dim]",
                title="File Organization"
            )
            self.console.print(storage_panel)
        
        # Database storage confirmation
        self.console.print(f"[green]✓ Saved to database with ID: {globule_id}[/green]")
        self.console.print("[dim]You can now find this thought when drafting related content![/dim]")
    
    async def _explore_data_processing_pipeline(self) -> None:
        """
        Deep dive into understanding how Globule processes data.
        
        This section helps users understand the technical aspects of how
        their thoughts are transformed and stored.
        """
        self.console.print("\n[bold purple]Let's explore what happens inside Globule's processing engine...[/bold purple]")
        
        pipeline_explanation = """
## The Globule Processing Pipeline

Your thought goes through several transformation stages:

### 1. Input Enrichment
- Raw text → Structured input object
- Metadata addition (timestamp, source, etc.)
- Context preservation

### 2. Parallel Processing
- **Embedding Generation**: AI converts text to meaning vectors
- **Structural Parsing**: Extract entities, topics, categories
- Both happen simultaneously for speed

### 3. Orchestration
- Combine results from both AI services
- Generate file organization suggestions
- Calculate confidence scores

### 4. Storage
- Save to SQLite database for fast search
- Optional file export to markdown
- Maintain semantic relationships

Would you like to see the raw data structures?
        """
        
        self.console.print(Panel(Markdown(pipeline_explanation), title="Under the Hood"))
        
        if Confirm.ask("Show me the technical details"):
            await self._show_technical_data_structures()
        
        self.learning_checkpoints.append("pipeline_explored")
    
    async def _show_technical_data_structures(self) -> None:
        """Show users the actual data structures used internally."""
        if self.user_inputs:
            latest_thought = self.user_inputs[-1]
            
            self.console.print(f"\n[bold]Technical view of processing \"{latest_thought[:30]}...\"[/bold]")
            
            # Show the EnrichedInput structure
            enriched_input = self.create_test_input(latest_thought, "technical_demo")
            
            input_data = {
                "original_text": enriched_input.original_text,
                "enriched_text": enriched_input.enriched_text,
                "source": enriched_input.source,
                "timestamp": enriched_input.timestamp.isoformat(),
                "verbosity": enriched_input.verbosity
            }
            
            from rich.json import JSON
            self.console.print(Panel(JSON.from_data(input_data), title="EnrichedInput Data Structure"))
            
            # Explain each field
            field_explanations = Table(title="Field Explanations")
            field_explanations.add_column("Field", style="cyan")
            field_explanations.add_column("Purpose", style="dim")
            
            field_explanations.add_row("original_text", "Your exact input, preserved unchanged")
            field_explanations.add_row("enriched_text", "Processed version (same for now, enhanced in Phase 2)")
            field_explanations.add_row("source", "Where this input came from (CLI, API, etc.)")
            field_explanations.add_row("timestamp", "When you created this thought")
            field_explanations.add_row("verbosity", "How much detail to show in responses")
            
            self.console.print(field_explanations)
    
    async def _guided_retrieval_exercise(self) -> None:
        """
        Guide users through finding and working with their stored thoughts.
        
        This exercise teaches users how to retrieve and use their captured thoughts
        for synthesis and drafting.
        """
        self.console.print("\n[bold green]Now let's see how to find and use your captured thoughts![/bold green]")
        
        # Show what's in their database
        globules = await self.storage.get_recent_globules(limit=10)
        
        if not globules:
            self.console.print("[yellow]You haven't captured any thoughts yet! Try the capture exercise first.[/yellow]")
            return
        
        retrieval_intro = """
## Finding Your Thoughts

Globule provides several ways to find related thoughts:

1. **Recent Thoughts** - Show what you've captured lately
2. **Semantic Search** - Find by meaning (Phase 2 feature)
3. **Topic Clustering** - Group related ideas together
4. **Full-Text Search** - Traditional keyword matching

Let's explore what you've captured so far!
        """
        
        self.console.print(Panel(Markdown(retrieval_intro), title="Retrieval Methods"))
        
        # Show their recent thoughts
        thoughts_table = Table(title=f"Your Recent Thoughts ({len(globules)} found)")
        thoughts_table.add_column("ID", style="dim", width=8)
        thoughts_table.add_column("Thought", style="cyan")
        thoughts_table.add_column("When", style="dim")
        
        for globule in globules:
            preview = globule.text[:60] + "..." if len(globule.text) > 60 else globule.text
            when = globule.created_at.strftime("%m/%d %H:%M")
            thoughts_table.add_row(str(globule.id)[:8], preview, when)
        
        self.console.print(thoughts_table)
        
        # Interactive exploration
        if Confirm.ask("Would you like to explore one of these thoughts in detail?"):
            await self._detailed_thought_exploration(globules)
        
        # Simulate draft mode
        if len(globules) > 1 and Confirm.ask("Want to see how these thoughts would appear in draft mode?"):
            await self._simulate_draft_interface(globules)
        
        self.learning_checkpoints.append("retrieval_explored")
    
    async def _detailed_thought_exploration(self, globules: List) -> None:
        """Allow user to explore a specific thought in detail."""
        if not globules:
            return
        
        # Let user pick a thought
        self.console.print("\n[bold]Choose a thought to explore:[/bold]")
        for i, globule in enumerate(globules[:5], 1):
            preview = globule.text[:50] + "..." if len(globule.text) > 50 else globule.text
            self.console.print(f"  {i}. {preview}")
        
        try:
            choice = int(Prompt.ask("Enter number (1-5)", default="1")) - 1
            if 0 <= choice < len(globules):
                selected = globules[choice]
                await self._show_thought_details(selected)
        except (ValueError, IndexError):
            self.console.print("[yellow]Invalid choice, showing first thought[/yellow]")
            await self._show_thought_details(globules[0])
    
    async def _show_thought_details(self, globule) -> None:
        """Show detailed information about a specific thought."""
        self.console.print(f"\n[bold cyan]Detailed View: Thought {str(globule.id)[:8]}[/bold cyan]")
        
        # Content
        content_panel = Panel(
            globule.text,
            title="Original Text",
            border_style="cyan"
        )
        self.console.print(content_panel)
        
        # Metadata table
        metadata_table = Table(title="Thought Metadata")
        metadata_table.add_column("Property", style="cyan")
        metadata_table.add_column("Value", style="green")
        
        metadata_table.add_row("ID", str(globule.id))
        metadata_table.add_row("Created", globule.created_at.strftime("%Y-%m-%d %H:%M:%S"))
        metadata_table.add_row("Text Length", f"{len(globule.text)} characters")
        
        if hasattr(globule, 'embedding') and globule.embedding is not None:
            metadata_table.add_row("Embedding Dimensions", str(len(globule.embedding)))
        
        if hasattr(globule, 'parsed_data') and globule.parsed_data:
            metadata_table.add_row("Category", globule.parsed_data.get("category", "unknown"))
            metadata_table.add_row("Domain", globule.parsed_data.get("domain", "unknown"))
        
        self.console.print(metadata_table)
    
    async def _simulate_draft_interface(self, globules: List) -> None:
        """Simulate what the draft interface would look like."""
        self.console.print("\n[bold purple]Draft Mode Simulation[/bold purple]")
        
        simulation_intro = """
This is what you'd see when running `globule draft "your topic"`:

The left side shows your related thoughts, the right side is where you'd write.
In this simulation, we'll show how your thoughts would be organized and presented.
        """
        
        self.console.print(Panel(simulation_intro, title="Draft Interface Preview"))
        
        # Simulate clustering (simplified for Phase 1)
        self.console.print("\n[bold]📚 Your Thought Library[/bold]")
        
        draft_table = Table(title="Available for Drafting")
        draft_table.add_column("Thought", style="cyan")
        draft_table.add_column("Relevance", style="green")
        draft_table.add_column("Actions", style="dim")
        
        for globule in globules[:5]:
            preview = globule.text[:50] + "..." if len(globule.text) > 50 else globule.text
            relevance = "High" if len(self.user_inputs) > 0 and any(word in globule.text.lower() for word in self.user_inputs[0].lower().split()[:3]) else "Medium"
            draft_table.add_row(preview, relevance, "Add to draft, View details")
        
        self.console.print(draft_table)
        
        self.console.print("\n[dim]In the full interface, you'd click thoughts to add them to your draft canvas![/dim]")
    
    async def _provide_learning_summary(self) -> None:
        """
        Provide a comprehensive learning summary and next steps.
        
        This method consolidates the learning experience and guides users
        toward continued exploration and mastery.
        """
        name = self.personalization_data.get("name", "Explorer")
        
        self.console.print(f"\n[bold green]Congratulations, {name}! You've completed the interactive tutorial![/bold green]")
        
        # Learning achievement summary
        achievements = []
        if "objectives_presented" in self.learning_checkpoints:
            achievements.append("✓ Understood Globule's purpose and benefits")
        if "configuration_explored" in self.learning_checkpoints:
            achievements.append("✓ Explored system configuration and local-first architecture")
        if "thought_capture_completed" in self.learning_checkpoints:
            achievements.append(f"✓ Successfully captured {len(self.user_inputs)} of your own thoughts")
        if "pipeline_explored" in self.learning_checkpoints:
            achievements.append("✓ Learned how the AI processing pipeline works")
        if "retrieval_explored" in self.learning_checkpoints:
            achievements.append("✓ Discovered how to find and organize your thoughts")
        
        achievements_panel = Panel(
            "\n".join(achievements),
            title="🎉 What You've Learned",
            border_style="green"
        )
        self.console.print(achievements_panel)
        
        # Personalized next steps
        next_steps = self._generate_personalized_next_steps()
        self.console.print(Panel(Markdown(next_steps), title="🚀 Your Next Steps"))
        
        # Usage statistics
        stats_table = Table(title="Your Tutorial Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")
        
        stats_table.add_row("Thoughts Captured", str(len(self.user_inputs)))
        stats_table.add_row("Learning Checkpoints", str(len(self.learning_checkpoints)))
        stats_table.add_row("User Interactions", str(self.metrics.user_interactions))
        stats_table.add_row("Time Spent Learning", f"{self.metrics.total_duration_ms/1000:.1f} seconds")
        
        self.console.print(stats_table)
        
        # Encourage continued exploration
        if Confirm.ask("Would you like to see other Glass Engine modes?"):
            self._suggest_other_modes()
        
        self.learning_checkpoints.append("tutorial_completed")
    
    def _generate_personalized_next_steps(self) -> str:
        """Generate personalized next steps based on user's learning journey."""
        experience = self.personalization_data.get("experience", "beginner")
        interests = self.personalization_data.get("interests", "ideas and thoughts")
        
        if experience == "beginner":
            return f"""
## Recommended Next Steps for Beginners

1. **Practice Daily Capture**
   - Try adding 3-5 {interests} each day
   - Experiment with different types of content
   - See how the AI understands different writing styles

2. **Explore the Demo Mode**
   - Run `globule tutorial --demo` to see advanced features
   - Watch the complete system showcase
   - Learn about upcoming Phase 2 capabilities

3. **Build Your Personal Knowledge Base**
   - Capture quotes, ideas, and reflections regularly
   - Use descriptive language to help the AI understand context
   - Try the draft mode when you have 10+ thoughts captured

4. **Join the Community**
   - Check the project documentation for tips
   - Share your experience with other users
   - Suggest improvements and new features
            """
        elif experience == "intermediate":
            return f"""
## Next Steps for Intermediate Users

1. **Advanced Usage Patterns**
   - Experiment with structured input formats
   - Try different verbosity levels (--verbose, --quiet)
   - Explore the configuration file for customization

2. **Integration Workflows**
   - Set up daily capture routines
   - Connect with your existing note-taking tools
   - Use Globule for project brainstorming and synthesis

3. **Deep System Understanding**
   - Run `globule tutorial --debug` for technical insights
   - Examine the database structure and file organization
   - Contribute to the open-source project

4. **Phase 2 Preparation**
   - Build a substantial knowledge base now
   - Prepare for advanced semantic search features
   - Consider custom schema development
            """
        else:  # advanced
            return f"""
## Advanced User Next Steps

1. **System Customization**
   - Modify configuration for your specific needs
   - Experiment with different AI models via Ollama
   - Create custom schemas for structured data

2. **Technical Exploration**
   - Run debug mode to understand execution traces
   - Examine the codebase and contribute improvements
   - Test performance with large datasets

3. **Community Leadership**
   - Help other users in forums and discussions
   - Create tutorials and documentation
   - Contribute code improvements and new features

4. **Phase 2 Beta Testing**
   - Prepare for advanced semantic clustering
   - Test vector search capabilities
   - Provide feedback on new features
            """
    
    def _suggest_other_modes(self) -> None:
        """Suggest other Glass Engine modes based on user's experience."""
        suggestions = """
## Other Glass Engine Modes

**Demo Mode** (`globule tutorial --demo`)
- Professional technical showcase
- See all features in automated demonstration
- Perfect for understanding capabilities quickly

**Debug Mode** (`globule tutorial --debug`)
- Deep technical introspection
- Raw execution traces and performance data
- Ideal for developers and system debugging

Each mode offers a different perspective on the same Glass Engine philosophy!
        """
        
        self.console.print(Panel(Markdown(suggestions), title="Explore Further"))
    
    def present_results(self) -> None:
        """
        Present the interactive tutorial results in an educational format.
        
        This method summarizes the learning experience and provides metrics
        in a way that reinforces educational objectives.
        """
        self.console.print("\n" + "=" * 80)
        self.console.print(Panel.fit(
            "[bold blue]Interactive Tutorial: Learning Summary[/bold blue]",
            title="Glass Engine Results"
        ))
        
        # Learning outcomes table
        outcomes_table = Table(title="Learning Outcomes Achieved")
        outcomes_table.add_column("Learning Objective", style="cyan")
        outcomes_table.add_column("Status", style="green")
        outcomes_table.add_column("Evidence", style="dim")
        
        # Map checkpoints to learning objectives
        objective_mapping = {
            "objectives_presented": ("Understanding Globule's Purpose", "Completed", "Engaged with tutorial objectives"),
            "configuration_explored": ("System Architecture Knowledge", "Completed", "Explored configuration and file structure"),
            "thought_capture_completed": ("Hands-on Experience", "Completed", f"Captured {len(self.user_inputs)} personal thoughts"),
            "pipeline_explored": ("Technical Understanding", "Completed", "Learned AI processing pipeline"),
            "retrieval_explored": ("Practical Skills", "Completed", "Used retrieval and draft features"),
            "tutorial_completed": ("Tutorial Mastery", "Completed", "Successfully completed all exercises")
        }
        
        for checkpoint in self.learning_checkpoints:
            if checkpoint in objective_mapping:
                objective, status, evidence = objective_mapping[checkpoint]
                outcomes_table.add_row(objective, status, evidence)
        
        self.console.print(outcomes_table)
        
        # Show validation results
        if self.metrics.test_results:
            self._present_validation_results()
        
        # Encourage continued learning
        self.console.print("\n[bold green]Keep exploring! Learning is a journey, not a destination.[/bold green]")
    
    def _present_validation_results(self) -> None:
        """Present validation results in educational context."""
        self.console.print("\n[bold]System Validation Results[/bold]")
        
        validation_table = Table(title="Tutorial Validation Tests")
        validation_table.add_column("Test", style="cyan")
        validation_table.add_column("Result", style="green")
        validation_table.add_column("Learning Value", style="dim")
        
        for result in self.metrics.test_results:
            test_name = result.get("test", "unknown")
            success = result.get("success", False)
            status = "✓ PASS" if success else "✗ FAIL"
            
            # Add educational context
            learning_value = {
                "interactive_thought_processing": "Validates core capture functionality",
                "storage": "Confirms data persistence works",
                "embedding": "Verifies AI processing pipeline",
                "parser": "Tests content analysis capabilities"
            }.get(test_name, "Confirms system reliability")
            
            validation_table.add_row(test_name, status, learning_value)
        
        self.console.print(validation_table)