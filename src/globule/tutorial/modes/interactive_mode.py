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
        
        # Phase 5: Semantic Clustering Experience
        await self._guided_clustering_exercise()
        
        # Phase 6: Interactive TUI Demonstration
        await self._guided_tui_experience()
        
        # Phase 7: AI Co-Pilot Hands-On Experience
        await self._guided_ai_copilot_exercise()
        
        # Phase 8: Retrieval and Advanced Features
        await self._guided_retrieval_exercise()
        
        # Phase 9: Summary and Next Steps
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

1. **How Globule captures your thoughts** - Real AI analysis with intelligent fallbacks
2. **Where your data lives** - Transparent file organization and vector storage
3. **How AI processes your ideas** - Advanced parsing, embedding, and classification
4. **How to find related thoughts** - Semantic clustering and vector search
5. **How to build drafts interactively** - The complete two-pane TUI experience
6. **Your privacy and control** - Local-first architecture with no cloud dependencies

## Phase 2 Capabilities You'll Experience

✨ **Real Intelligence**: Ollama-powered content analysis
🧠 **Semantic Understanding**: Vector embeddings and similarity search
📊 **Smart Clustering**: Automatic theme detection and grouping
🎨 **Interactive Drafting**: Live two-pane synthesis interface
🔍 **Glass Engine Philosophy**: Complete transparency in how everything works

## Learning Approach

This is a **hands-on tutorial** where you'll:
- Type your own thoughts and see them processed with real AI
- Experience semantic clustering of your ideas
- Use the interactive TUI for drafting
- Explore advanced Phase 2 features
- See exactly how the intelligence works under the hood

Ready to experience the magic?
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
        
        # Add a "learning moment" - this is Glass Engine philosophy in action
        self._provide_learning_moment(result)
        
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
    
    def _provide_learning_moment(self, result) -> None:
        """
        Provide a deeper learning insight - Glass Engine philosophy in action.
        
        This method helps users understand the 'why' behind what just happened,
        building trust through transparency about the system's decisions.
        """
        # Choose an insight based on what the system discovered
        insights = []
        
        if result.embedding_confidence > 0.8:
            insights.append("🧠 **High AI Confidence**: The system strongly understood your thought's meaning. This suggests your idea was clearly expressed and will be easy to find later.")
        
        if result.parsed_data:
            domain = result.parsed_data.get('domain', 'general')
            if domain != 'general':
                insights.append(f"🎯 **Domain Detection**: Your thought was classified as '{domain}' - this helps Globule organize related ideas together automatically.")
        
        if result.file_decision and result.file_decision.confidence > 0.7:
            insights.append("📁 **Smart Organization**: The system is confident about where this thought belongs in your knowledge structure.")
        
        # Always include a philosophical insight about what just happened
        philosophical_insights = [
            "🔍 **Transparency Principle**: You just saw every step of how your thought was processed - no black boxes, complete visibility.",
            "🎓 **Learning Through Doing**: By processing your own thoughts, you understand how the system works better than any manual could teach.",
            "🔗 **Semantic Understanding**: The AI didn't just store your words - it captured their meaning, creating connections you haven't even discovered yet."
        ]
        
        # Combine specific insights with philosophical understanding
        all_insights = insights + [philosophical_insights[len(insights) % len(philosophical_insights)]]
        
        if all_insights:
            learning_panel = Panel(
                "\n".join(all_insights),
                title="💡 Learning Moment: Why This Matters",
                border_style="dim blue"
            )
            self.console.print(learning_panel)
    
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
    
    async def _guided_clustering_exercise(self) -> None:
        """Guide users through experiencing semantic clustering."""
        name = self.personalization_data.get("name", "Explorer")
        
        self.console.print(f"\n[bold magenta]Phase 2 Magic Time, {name}! Let's see how Globule discovers themes in your thoughts...[/bold magenta]")
        
        clustering_intro = """
## Semantic Clustering: The Intelligence Behind Organization

Now that you've captured some thoughts, Globule can do something amazing:
**automatically discover hidden themes and connections** using AI.

### How It Works:
1. **Vector Analysis**: Each thought becomes a mathematical representation of its meaning
2. **Similarity Detection**: AI finds thoughts that are conceptually related
3. **Theme Discovery**: Groups emerge based on semantic patterns
4. **Intelligent Labeling**: Clusters get meaningful names based on their content

This isn't just keyword matching - it's genuine understanding of meaning!
        """
        
        self.console.print(Panel(Markdown(clustering_intro), title="The Magic of Semantic Understanding"))
        
        # Check if we have enough thoughts for clustering
        globules = await self.storage.get_recent_globules(limit=20)
        
        if len(globules) < 3:
            self.console.print("[yellow]You need at least 3 thoughts for clustering. Let's add one more![/yellow]")
            
            # Suggest some clustering-friendly thoughts
            suggestions = [
                "The concept of flow state applies to both coding and creative writing",
                "Local-first software gives users real ownership of their digital life",
                "Teaching someone to think is more valuable than teaching facts",
                "Progressive overload in fitness could apply to skill development",
                "The best tools become invisible - you think with them, not about them"
            ]
            
            self.console.print("\n[cyan]Here are some clustering-friendly suggestions:[/cyan]")
            for i, suggestion in enumerate(suggestions, 1):
                self.console.print(f"  {i}. {suggestion}")
            
            final_thought = Prompt.ask("\n[bold cyan]Add one more thought (or choose a number 1-5)[/bold cyan]")
            
            # Handle numeric choice
            try:
                choice_num = int(final_thought)
                if 1 <= choice_num <= 5:
                    final_thought = suggestions[choice_num - 1]
            except ValueError:
                pass  # Use their custom input
            
            # Process the final thought
            await self._demonstrate_processing_with_narration(final_thought)
            self.user_inputs.append(final_thought)
        
        # Now perform clustering analysis
        self.console.print("\n[bold cyan]Analyzing your thoughts for semantic patterns...[/bold cyan]")
        
        try:
            from globule.clustering.semantic_clustering import SemanticClusteringEngine
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                
                analysis_task = progress.add_task("Discovering semantic clusters...", total=None)
                
                # Initialize clustering engine
                clustering_engine = SemanticClusteringEngine(self.storage)
                
                # Run clustering analysis
                analysis = await clustering_engine.analyze_semantic_clusters(min_globules=2)
                
                progress.update(analysis_task, completed=True)
            
            # Display results
            self.console.print("\n[bold green]✨ Clustering Analysis Complete![/bold green]")
            
            if analysis.clusters:
                self.console.print(f"\n[bold]Discovered {len(analysis.clusters)} semantic clusters in your thoughts:[/bold]")
                
                for i, cluster in enumerate(analysis.clusters, 1):
                    self.console.print(f"\n[cyan]Cluster {i}: {cluster.label}[/cyan]")
                    self.console.print(f"  Size: {cluster.size} thoughts")
                    self.console.print(f"  Confidence: {cluster.confidence_score:.1%}")
                    self.console.print(f"  Keywords: {', '.join(cluster.keywords[:5])}")
                    self.console.print(f"  Description: {cluster.description}")
                    
                    if cluster.representative_samples:
                        self.console.print(f"  Sample: \"{cluster.representative_samples[0][:80]}...\"")
                
                # Educational moment
                clustering_insight = Panel(
                    "🧠 **What Just Happened**: Globule analyzed the *meaning* of your thoughts, not just keywords. "
                    "It found patterns in how ideas relate to each other semantically. This is how you'll "
                    "discover unexpected connections and themes in your knowledge over time.",
                    title="🎆 The Magic Explained",
                    border_style="dim blue"
                )
                self.console.print(clustering_insight)
                
            else:
                self.console.print("[yellow]No distinct clusters found - your thoughts might be very diverse, which is great for creativity![/yellow]")
                
        except Exception as e:
            self.console.print(f"[yellow]Clustering analysis not available: {e}[/yellow]")
            self.console.print("[dim]This feature requires sufficient thoughts and AI processing capabilities.[/dim]")
        
        self.learning_checkpoints.append("clustering_experienced")
    
    async def _guided_tui_experience(self) -> None:
        """Guide users through the interactive TUI experience."""
        name = self.personalization_data.get("name", "Explorer")
        
        self.console.print(f"\n[bold purple]Time for the Grand Finale, {name}! Let's experience the complete Globule interface...[/bold purple]")
        
        tui_intro = """
## The Interactive TUI: Where Everything Comes Together

Globule's **two-pane interface** is where the magic becomes practical:

### Left Pane: Semantic Palette
- **Live clustering** of your thoughts by theme
- **Expandable clusters** showing related ideas
- **Smart navigation** with keyboard shortcuts
- **Visual confidence indicators** for cluster quality

### Right Pane: Canvas Editor
- **Markdown-ready drafting** environment
- **Click-to-add** thoughts from clusters
- **Real-time synthesis** as you build your draft
- **Integrated save** functionality

### The Experience
Watch your scattered thoughts transform into organized knowledge,
then seamlessly flow into structured drafts. This is knowledge work
at the speed of thought.
        """
        
        self.console.print(Panel(Markdown(tui_intro), title="The Complete Experience"))
        
        # Show what the TUI looks like
        tui_demo = """
┌────────────────────────────────────┬───────────────────────────────────┐
│ PALETTE: Semantic Clusters        │ CANVAS: Draft Editor              │
├────────────────────────────────────┼───────────────────────────────────┤
│                                    │                                   │
│ ▶ Creative Thinking (3) [=======] │ # My Article Draft                │
│   TAGS: creativity, flow, ideas    │                                   │
│                                    │ ## Introduction                   │
│ ▶ Local-First Tech (2) [======]   │                                   │
│   TAGS: privacy, control, tools    │ The concept of flow state applies │
│                                    │ to both coding and creative       │
│ ▶ Learning Methods (2) [=====]    │ writing...                        │
│   TAGS: education, thinking        │                                   │
│                                    │ ## Core Ideas                     │
│ [Tab] Switch Focus               │                                   │
│ [Enter] Select/Add               │ [Ctrl+S] Save Draft               │
│ [Space] Toggle Expand            │ [Tab] Switch to Palette           │
│                                    │                                   │
└────────────────────────────────────┴───────────────────────────────────┘
        """
        
        self.console.print(Panel(tui_demo, title="Live TUI Interface Preview", border_style="green"))
        
        # Offer to launch the TUI
        if Confirm.ask("\nWould you like to launch the live TUI interface to try it yourself?"):
            self.console.print("\n[bold cyan]Launching interactive drafting session...[/bold cyan]")
            self.console.print("[dim]This will open the Phase 3 AI-assisted interface where you can:")
            self.console.print("[dim]- Navigate clusters with arrow keys")
            self.console.print("[dim]- Press Enter to add thoughts to your draft")
            self.console.print("[dim]- Use Tab to switch between panes")
            self.console.print("[dim]- Press Ctrl+E to expand text with AI Co-Pilot")
            self.console.print("[dim]- Press Ctrl+R to summarize text with AI Co-Pilot")
            self.console.print("[dim]- Press Ctrl+S to save as markdown file")
            self.console.print("[dim]- Press Q to return to tutorial[/dim]")
            
            try:
                # Import and launch TUI
                from globule.tui.app import SynthesisApp
                
                # Create TUI app with current storage
                tui_app = SynthesisApp(storage_manager=self.storage, topic="tutorial session")
                
                # Brief pause for user to read instructions
                await asyncio.sleep(2)
                
                # Launch TUI (this will block until user exits)
                await tui_app.run_async()
                
                # Back to tutorial
                self.console.print("\n[bold green]Welcome back to the tutorial![/bold green]")
                self.console.print("[cyan]You've now experienced the complete Phase 2 interface![/cyan]")
                
            except Exception as e:
                self.console.print(f"[yellow]Could not launch TUI: {e}[/yellow]")
                self.console.print("[dim]You can try 'globule draft' from the command line after the tutorial.[/dim]")
        else:
            self.console.print("\n[cyan]No worries! You can launch the TUI anytime with 'globule draft'[/cyan]")
        
        # Always provide the learning moment
        tui_insight = Panel(
            "✨ **The Complete Vision Realized**: You've just seen how scattered thoughts become "
            "structured knowledge. The TUI combines semantic clustering (AI understanding) with "
            "interactive drafting (human creativity). This is knowledge work evolved.",
            title="🎨 From Chaos to Creation",
            border_style="dim purple"
        )
        self.console.print(tui_insight)
        
        self.learning_checkpoints.append("tui_experienced")
    
    async def _demonstrate_vector_search(self, globules: List) -> None:
        """Demonstrate vector search capabilities."""
        search_intro = """
Vector search finds thoughts by **meaning**, not just keywords.
Try searching for concepts, themes, or ideas - even if the exact words don't appear!
        """
        
        self.console.print(Panel(search_intro, title="🔍 Semantic Search Demo"))
        
        # Get search query from user
        search_query = Prompt.ask("[bold cyan]Enter a search concept or theme[/bold cyan]", default="creativity")
        
        try:
            # Perform vector search
            self.console.print(f"\n[dim]Searching for thoughts similar to '{search_query}'...[/dim]")
            
            # Generate embedding for query
            query_embedding = await self.embedding_provider.embed(search_query)
            
            # Search for similar globules
            search_results = await self.storage.search_by_embedding(
                query_embedding, 
                limit=5, 
                similarity_threshold=0.3
            )
            
            if search_results:
                self.console.print(f"\n[bold green]Found {len(search_results)} semantically similar thoughts:[/bold green]")
                
                search_table = Table(title="Vector Search Results")
                search_table.add_column("Similarity", style="green", width=10)
                search_table.add_column("Thought", style="cyan")
                
                for globule, similarity in search_results:
                    preview = globule.text[:70] + "..." if len(globule.text) > 70 else globule.text
                    search_table.add_row(f"{similarity:.1%}", preview)
                
                self.console.print(search_table)
                
                # Explain the magic
                search_insight = Panel(
                    f"🎯 **Semantic Magic**: These results were found by meaning similarity to '{search_query}', "
                    "not keyword matching. The AI understands conceptual relationships and finds related ideas "
                    "even when different words are used.",
                    title="How Vector Search Works",
                    border_style="dim green"
                )
                self.console.print(search_insight)
                
            else:
                self.console.print("[yellow]No similar thoughts found. Try adding more diverse content![/yellow]")
                
        except Exception as e:
            self.console.print(f"[yellow]Vector search not available: {e}[/yellow]")
    
    async def _guided_ai_copilot_exercise(self) -> None:
        """
        Guide users through the Phase 3 AI Co-Pilot features with hands-on exercises.
        
        This interactive section lets users experience the AI-assisted writing
        capabilities that distinguish Globule's Phase 3 implementation.
        """
        self.console.print(
            Panel(
                "[bold cyan]🤖 Phase 3 Learning Module: AI Co-Pilot[/bold cyan]",
                title="AI-Assisted Writing Experience",
                border_style="cyan"
            )
        )
        
        # Introduction to AI Co-Pilot
        intro_panel = Panel(
            "[bold]Welcome to Globule's AI Co-Pilot![/bold]\n\n"
            "Phase 3 introduces revolutionary AI-assisted writing capabilities that enhance "
            "your knowledge work without compromising your ideas or privacy.\n\n"
            "[cyan]What you'll learn:[/cyan]\n"
            "• How AI text expansion enhances brief notes\n"
            "• How AI summarization distills complex content\n"
            "• How professional export creates publication-ready documents\n"
            "• Why local AI processing protects your intellectual property",
            title="🎯 Learning Objectives",
            border_style="green"
        )
        self.console.print(intro_panel)
        
        # AI Co-Pilot features overview
        features_table = Table(title="🤖 AI Co-Pilot Features")
        features_table.add_column("Feature", style="cyan")
        features_table.add_column("Keybinding", style="green")
        features_table.add_column("Purpose", style="yellow")
        features_table.add_column("AI Technology", style="dim")
        
        features_table.add_row(
            "Text Expansion", 
            "Ctrl+E", 
            "Elaborate and enrich brief notes", 
            "Ollama LLM with context prompting"
        )
        features_table.add_row(
            "Text Summarization", 
            "Ctrl+R", 
            "Condense complex content to essentials", 
            "Semantic analysis and key point extraction"
        )
        features_table.add_row(
            "Enhanced Export", 
            "Ctrl+S", 
            "Generate professional markdown documents", 
            "Structured formatting with metadata"
        )
        
        self.console.print(features_table)
        
        # Interactive AI demonstration
        if Confirm.ask("\nWould you like to see AI Co-Pilot in action with live examples?"):
            await self._demonstrate_ai_copilot_live()
        
        # Privacy and local processing explanation
        privacy_panel = Panel(
            "[bold]🔒 Privacy-First AI Design:[/bold]\n\n"
            "Unlike cloud-based AI tools, Globule's AI Co-Pilot:\n"
            "• Runs entirely on your local machine using Ollama\n"
            "• Never sends your ideas to external servers\n"
            "• Doesn't contribute your content to AI training datasets\n"
            "• Works completely offline once models are downloaded\n"
            "• Gives you full control over your intellectual property\n\n"
            "[cyan]This means you can confidently use AI assistance on sensitive work, "
            "proprietary research, or personal thoughts without privacy concerns.[/cyan]",
            title="🛡️ Your Ideas Stay Yours",
            border_style="blue"
        )
        self.console.print(privacy_panel)
        
        # Offer hands-on experience
        if Confirm.ask("\nWould you like to try the AI Co-Pilot features yourself in the TUI?"):
            self.console.print("\n[bold cyan]Launching AI Co-Pilot experience...[/bold cyan]")
            self.console.print("[dim]The TUI will open with Phase 3 AI features enabled.")
            self.console.print("[dim]Try selecting some text and pressing Ctrl+E or Ctrl+R!")
            self.console.print("[dim]Press Q when you're ready to continue the tutorial.[/dim]")
            
            try:
                # Import and launch TUI with AI focus
                from globule.tui.app import SynthesisApp
                
                # Create TUI app with emphasis on AI features
                tui_app = SynthesisApp(storage_manager=self.storage, topic="ai copilot demo")
                
                # Brief pause for user to read instructions
                await asyncio.sleep(2)
                
                # Launch TUI for AI Co-Pilot experience
                await tui_app.run_async()
                
                self.console.print("\n[green]Welcome back to the tutorial![/green]")
                
            except Exception as e:
                self.console.print(f"[yellow]TUI launch failed: {e}[/yellow]")
                self.console.print("[dim]Continuing with tutorial...[/dim]")
        
        # Wrap up the AI Co-Pilot section
        conclusion_panel = Panel(
            "[bold]🎯 Key Takeaways:[/bold]\n\n"
            "• AI Co-Pilot enhances your writing without replacing your voice\n"
            "• Local AI processing ensures complete privacy and offline capability\n"
            "• Professional export creates publication-ready documents\n"
            "• These features work seamlessly within your knowledge capture workflow\n\n"
            "[cyan]Next: We'll explore advanced search and retrieval capabilities![/cyan]",
            title="✅ AI Co-Pilot Mastery",
            border_style="green"
        )
        self.console.print(conclusion_panel)
    
    async def _demonstrate_ai_copilot_live(self) -> None:
        """Demonstrate AI Co-Pilot with live examples."""
        examples_panel = Panel(
            "[bold]📝 Live AI Co-Pilot Examples:[/bold]\n\n"
            "[cyan]Text Expansion Example:[/cyan]\n"
            "[dim]Original:[/dim] 'Local-first software matters.'\n"
            "[dim]AI Enhanced:[/dim] 'Local-first software matters because it fundamentally "
            "shifts power from centralized platforms back to users, ensuring data sovereignty, "
            "enabling offline productivity, and providing protection against vendor lock-in "
            "while maintaining collaborative capabilities when connectivity is available.'\n\n"
            "[cyan]Text Summarization Example:[/cyan]\n"
            "[dim]Original:[/dim] '[Long paragraph about machine learning architectures...]\n"
            "[dim]AI Summary:[/dim] 'Modern neural networks use transformer architectures "
            "with attention mechanisms to process sequential data efficiently.'\n\n"
            "[cyan]Professional Export Example:[/cyan]\n"
            "[dim]Creates:[/dim] drafts/globule_draft_ai_demo_20250128_143500.md\n"
            "[dim]With:[/dim] Proper headers, formatting, and metadata",
            title="🎬 Real AI Assistance in Action",
            border_style="magenta"
        )
        self.console.print(examples_panel)
    
    async def _guided_retrieval_exercise(self) -> None:
        """
        Guide users through advanced retrieval and Phase 2 search features.
        
        This exercise teaches users how to use vector search, clustering,
        and other Phase 2 intelligence features.
        """
        self.console.print("\n[bold green]Let's explore Globule's advanced search and retrieval capabilities![/bold green]")
        
        # Show what's in their database
        globules = await self.storage.get_recent_globules(limit=10)
        
        if not globules:
            self.console.print("[yellow]You haven't captured any thoughts yet! Try the capture exercise first.[/yellow]")
            return
        
        retrieval_intro = """
## Phase 2 Retrieval: Beyond Keywords

Globule offers multiple intelligent ways to find and connect your thoughts:

1. **Recent Thoughts** - Your latest captures with rich metadata
2. **Vector Search** - Find by semantic meaning, not just keywords
3. **Semantic Clustering** - AI-discovered thematic groups
4. **Domain Filtering** - Browse by content type and category
5. **Confidence Scoring** - See how well the AI understood each thought

### What Makes This Special
- **Meaning-based**: Finds "flow state" when you search "optimal performance"
- **Cross-domain**: Connects fitness concepts to creative work
- **Evolving**: Gets smarter as you add more thoughts

Let's explore your personal knowledge base!
        """
        
        self.console.print(Panel(Markdown(retrieval_intro), title="Retrieval Methods"))
        
        # Show their recent thoughts with Phase 2 metadata
        thoughts_table = Table(title=f"Your Intelligent Knowledge Base ({len(globules)} thoughts)")
        thoughts_table.add_column("ID", style="dim", width=8)
        thoughts_table.add_column("Thought", style="cyan")
        thoughts_table.add_column("Domain", style="green", width=12)
        thoughts_table.add_column("Category", style="yellow", width=10)
        thoughts_table.add_column("Confidence", style="magenta", width=10)
        thoughts_table.add_column("When", style="dim", width=12)
        
        for globule in globules:
            preview = globule.text[:50] + "..." if len(globule.text) > 50 else globule.text
            when = globule.created_at.strftime("%m/%d %H:%M")
            
            # Extract Phase 2 metadata
            domain = "unknown"
            category = "note"
            confidence = "unknown"
            
            if globule.parsed_data:
                domain = globule.parsed_data.get('domain', 'general')[:11]
                category = globule.parsed_data.get('category', 'note')[:9]
            
            if globule.parsing_confidence is not None:
                confidence = f"{globule.parsing_confidence:.1%}"
            
            thoughts_table.add_row(str(globule.id)[:8], preview, domain, category, confidence, when)
        
        self.console.print(thoughts_table)
        
        # Phase 2 search demonstration
        if len(globules) > 2 and Confirm.ask("\nWould you like to try semantic search?"):
            await self._demonstrate_vector_search(globules)
        
        # Interactive exploration
        if Confirm.ask("\nWould you like to explore one of these thoughts in detail?"):
            await self._detailed_thought_exploration(globules)
        
        # Advanced features teaser
        if len(globules) > 1:
            advanced_features = Panel(
                "🔍 **Available from Command Line:**\n\n"
                "• `globule search 'your query'` - Semantic search\n"
                "• `globule cluster` - View clustering analysis\n"
                "• `globule draft` - Launch interactive TUI\n"
                "• `globule tutorial --demo` - See professional showcase\n"
                "• `globule tutorial --debug` - Deep technical analysis",
                title="🚀 More to Explore",
                border_style="dim cyan"
            )
            self.console.print(advanced_features)
        
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
        
        # Learning achievement summary with Phase 2 features
        achievements = []
        if "objectives_presented" in self.learning_checkpoints:
            achievements.append("✓ Understood Globule's Phase 2 intelligence capabilities")
        if "configuration_explored" in self.learning_checkpoints:
            achievements.append("✓ Explored local-first architecture with vector storage")
        if "thought_capture_completed" in self.learning_checkpoints:
            achievements.append(f"✓ Experienced real AI processing with {len(self.user_inputs)} thoughts")
        if "pipeline_explored" in self.learning_checkpoints:
            achievements.append("✓ Learned how intelligent parsing and embedding works")
        if "clustering_experienced" in self.learning_checkpoints:
            achievements.append("✓ Witnessed semantic clustering in action")
        if "tui_experienced" in self.learning_checkpoints:
            achievements.append("✓ Experienced the complete two-pane interface")
        if "retrieval_explored" in self.learning_checkpoints:
            achievements.append("✓ Discovered advanced search and retrieval features")
        
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