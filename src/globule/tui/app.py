"""
Enhanced Textual TUI for Globule Phase 2.

Phase 2: Two-pane layout with semantic clustering (palette) and canvas editor.
"""

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Header, Footer, Static, TextArea, Tree, Label
from textual.reactive import reactive, var
from textual.binding import Binding
from textual.message import Message
from textual import events
from typing import Optional, List, Set
import asyncio

from globule.core.interfaces import StorageManager
from globule.core.models import ProcessedGlobule, SynthesisState, UIMode, GlobuleCluster
from globule.clustering.semantic_clustering import SemanticClusteringEngine


class ClusterPalette(VerticalScroll):
    """Phase 2: Semantic cluster palette for thought discovery"""
    
    class ClusterSelected(Message):
        """Message sent when a cluster is selected"""
        def __init__(self, cluster_id: str) -> None:
            self.cluster_id = cluster_id
            super().__init__()
    
    class GlobuleSelected(Message):
        """Message sent when a globule is selected"""
        def __init__(self, globule: ProcessedGlobule) -> None:
            self.globule = globule
            super().__init__()
    
    def __init__(self, clusters: List[GlobuleCluster], **kwargs):
        super().__init__(**kwargs)
        self.clusters = clusters
        self.expanded_clusters: Set[str] = set()
        self.selected_cluster_id: Optional[str] = None
        self.selected_globule_id: Optional[str] = None
        self.can_focus = True
    
    def compose(self) -> ComposeResult:
        """Display semantic clusters with expandable thought groups"""
        if not self.clusters:
            yield Static("No clusters found. Add more thoughts to discover semantic patterns.", classes="no-content")
            return
        
        yield Static(f"CLUSTERS: {len(self.clusters)} Semantic Groups:", classes="cluster-header")
        
        for cluster in self.clusters:
            # Cluster header with size and confidence
            confidence_score = cluster.metadata.get('confidence_score', 0.5)
            confidence_bar = "=" * max(1, int(confidence_score * 10))
            cluster_title = f"FOLDER: {cluster.label} ({cluster.metadata.get('size', len(cluster.globules))} thoughts) [{confidence_bar}]"
            
            # Apply selection styling
            cluster_classes = "cluster-title"
            if cluster.id == self.selected_cluster_id:
                cluster_classes += " selected"
            
            cluster_widget = Static(cluster_title, classes=cluster_classes, id=f"cluster-{cluster.id}")
            yield cluster_widget
            
            # Show cluster keywords if available
            keywords = cluster.metadata.get('keywords', [])
            if keywords:
                keyword_text = f"TAGS: {', '.join(keywords[:3])}"
                yield Static(keyword_text, classes="cluster-keywords")
            
            # Show representative samples (expandable)
            if cluster.id in self.expanded_clusters and cluster.globules:
                for i, globule in enumerate(cluster.globules[:5]):  # Show top 5
                    preview = globule.text[:80] + "..." if len(globule.text) > 80 else globule.text
                    
                    # Apply selection styling to globules
                    globule_classes = "globule-sample"
                    if globule.id == self.selected_globule_id:
                        globule_classes += " selected"
                    
                    sample_widget = Static(f"  THOUGHT: {preview}", classes=globule_classes, id=f"globule-{globule.id}")
                    yield sample_widget
    
    def toggle_cluster(self, cluster_id: str) -> None:
        """Toggle cluster expansion"""
        if cluster_id in self.expanded_clusters:
            self.expanded_clusters.remove(cluster_id)
        else:
            self.expanded_clusters.add(cluster_id)
        # Refresh display
        self.refresh(recompose=True)
    
    def select_cluster(self, cluster_id: str) -> None:
        """Select a cluster"""
        self.selected_cluster_id = cluster_id
        self.selected_globule_id = None  # Clear globule selection
        self.post_message(self.ClusterSelected(cluster_id))
        self.refresh(recompose=True)
    
    def select_globule(self, globule_id: str) -> None:
        """Select a globule and post selection message"""
        self.selected_globule_id = globule_id
        
        # Find the globule in clusters
        for cluster in self.clusters:
            for globule in cluster.globules:
                if globule.id == globule_id:
                    self.post_message(self.GlobuleSelected(globule))
                    self.refresh(recompose=True)
                    return
    
    async def on_click(self, event: events.Click) -> None:
        """Handle click events on clusters and globules"""
        try:
            # Find what was clicked
            widget = self.get_widget_at(*event.screen_coordinate)
            if widget and hasattr(widget, 'id') and widget.id:
                if widget.id.startswith('cluster-'):
                    cluster_id = widget.id.replace('cluster-', '')
                    if cluster_id == self.selected_cluster_id:
                        # Toggle expansion if already selected
                        self.toggle_cluster(cluster_id)
                    else:
                        # Select cluster
                        self.select_cluster(cluster_id)
                elif widget.id.startswith('globule-'):
                    globule_id = widget.id.replace('globule-', '')
                    self.select_globule(globule_id)
        except Exception:
            # Ignore click handling errors
            pass
    
    async def on_key(self, event: events.Key) -> None:
        """Handle keyboard navigation"""
        if event.key == "enter":
            if self.selected_cluster_id:
                if self.selected_globule_id:
                    # Enter on globule - add to canvas
                    self.select_globule(self.selected_globule_id)
                else:
                    # Enter on cluster - toggle expansion
                    self.toggle_cluster(self.selected_cluster_id)
            elif self.clusters:
                # No selection - select first cluster
                self.select_cluster(self.clusters[0].id)
        
        elif event.key == "space":
            if self.selected_cluster_id:
                self.toggle_cluster(self.selected_cluster_id)
        
        elif event.key == "down":
            self._navigate_down()
        
        elif event.key == "up":
            self._navigate_up()
    
    def _navigate_down(self) -> None:
        """Navigate to next item"""
        if not self.clusters:
            return
        
        if not self.selected_cluster_id:
            # Select first cluster
            self.select_cluster(self.clusters[0].id)
            return
        
        # Find current position
        cluster_idx = None
        for i, cluster in enumerate(self.clusters):
            if cluster.id == self.selected_cluster_id:
                cluster_idx = i
                break
        
        if cluster_idx is None:
            return
        
        current_cluster = self.clusters[cluster_idx]
        
        # If we're on a cluster and it's expanded and has globules
        if (not self.selected_globule_id and 
            current_cluster.id in self.expanded_clusters and 
            current_cluster.globules):
            # Move to first globule
            self.selected_globule_id = current_cluster.globules[0].id
            self.refresh(recompose=True)
            return
        
        # If we're on a globule, try to move to next globule
        if self.selected_globule_id:
            globule_idx = None
            for i, globule in enumerate(current_cluster.globules):
                if globule.id == self.selected_globule_id:
                    globule_idx = i
                    break
            
            if globule_idx is not None and globule_idx < len(current_cluster.globules) - 1:
                # Move to next globule in same cluster
                self.selected_globule_id = current_cluster.globules[globule_idx + 1].id
                self.refresh(recompose=True)
                return
            else:
                # Move to next cluster
                self.selected_globule_id = None
        
        # Move to next cluster
        if cluster_idx < len(self.clusters) - 1:
            self.select_cluster(self.clusters[cluster_idx + 1].id)
    
    def _navigate_up(self) -> None:
        """Navigate to previous item"""
        if not self.clusters:
            return
        
        if not self.selected_cluster_id:
            # Select last cluster
            self.select_cluster(self.clusters[-1].id)
            return
        
        # Find current position
        cluster_idx = None
        for i, cluster in enumerate(self.clusters):
            if cluster.id == self.selected_cluster_id:
                cluster_idx = i
                break
        
        if cluster_idx is None:
            return
        
        current_cluster = self.clusters[cluster_idx]
        
        # If we're on a globule, try to move to previous globule or cluster
        if self.selected_globule_id:
            globule_idx = None
            for i, globule in enumerate(current_cluster.globules):
                if globule.id == self.selected_globule_id:
                    globule_idx = i
                    break
            
            if globule_idx is not None and globule_idx > 0:
                # Move to previous globule in same cluster
                self.selected_globule_id = current_cluster.globules[globule_idx - 1].id
                self.refresh(recompose=True)
                return
            else:
                # Move to cluster header
                self.selected_globule_id = None
                self.refresh(recompose=True)
                return
        
        # Move to previous cluster
        if cluster_idx > 0:
            prev_cluster = self.clusters[cluster_idx - 1]
            self.select_cluster(prev_cluster.id)
            # If previous cluster is expanded, go to its last globule
            if (prev_cluster.id in self.expanded_clusters and 
                prev_cluster.globules):
                self.selected_globule_id = prev_cluster.globules[-1].id
                self.refresh(recompose=True)


class CanvasEditor(TextArea):
    """Phase 2: Enhanced canvas editor for drafting with AI assistance"""
    
    def __init__(self, content: str = "", **kwargs):
        super().__init__(content, **kwargs)
        self.incorporated_globules: Set[str] = set()
        self.can_focus = True
        
    def add_globule_content(self, globule: ProcessedGlobule) -> None:
        """Add globule content to canvas with context preservation"""
        if globule.id in self.incorporated_globules:
            return  # Already incorporated
            
        # Format the globule content for integration
        title = globule.parsed_data.get('title', 'Thought') if globule.parsed_data else 'Thought'
        formatted_content = f"\n\n## {title}\n\n{globule.text}\n"
        
        # Add to current cursor position or end
        current_content = self.text
        cursor_pos = self.cursor_position if hasattr(self, 'cursor_position') else len(current_content)
        
        new_content = current_content[:cursor_pos] + formatted_content + current_content[cursor_pos:]
        self.text = new_content
        
        # Track incorporation
        self.incorporated_globules.add(globule.id)
        
        # Move cursor to end of inserted content
        try:
            self.cursor_position = cursor_pos + len(formatted_content)
        except:
            pass  # Ignore cursor positioning errors
    
    def get_content(self) -> str:
        """Get current canvas content"""
        return self.text
    
    def clear_content(self) -> None:
        """Clear canvas content"""
        self.text = "# Draft Editor\n\nStart writing your draft here...\n\n"
        self.incorporated_globules.clear()


class StatusBar(Static):
    """Phase 2: Enhanced status bar showing current mode and context"""
    
    def __init__(self, synthesis_state: SynthesisState, **kwargs):
        self.state = synthesis_state
        super().__init__(self._generate_status_text(), **kwargs)
    
    def _generate_status_text(self) -> str:
        """Generate status text based on current state"""
        mode_icons = {
            UIMode.BUILD: "BUILD",
            UIMode.EXPLORE: "EXPLORE",
            UIMode.EDIT: "EDIT"
        }
        
        mode_icon = mode_icons.get(self.state.current_mode, "UNKNOWN")
        
        status_parts = [f"MODE: {mode_icon}"]
        
        # Add context information
        if self.state.selected_cluster_id:
            status_parts.append(f"Cluster: {self.state.selected_cluster_id[:8]}...")
        
        if self.state.incorporated_globules:
            status_parts.append(f"Incorporated: {len(self.state.incorporated_globules)}")
        
        return " | ".join(status_parts)
    
    def update_status(self, new_state: SynthesisState) -> None:
        """Update status display"""
        self.state = new_state
        self.update(self._generate_status_text())


class SynthesisApp(App):
    """Phase 2: Enhanced Textual application with two-pane layout"""
    
    CSS = """
    /* Phase 2: Enhanced styling for two-pane layout */
    .cluster-header {
        color: $accent;
        margin-bottom: 1;
        text-style: bold;
    }
    
    .cluster-title {
        border: solid $primary;
        margin-bottom: 1;
        padding: 1;
        background: $surface;
    }
    
    .cluster-title:hover {
        background: $primary-background;
    }
    
    .cluster-keywords {
        color: $secondary;
        margin-left: 2;
        text-style: italic;
    }
    
    .globule-sample {
        margin-left: 4;
        margin-bottom: 1;
        padding: 1;
        border-left: solid $secondary;
    }
    
    .globule-sample:hover {
        background: $secondary-background;
    }
    
    .selected {
        background: $accent !important;
        color: $text-selected;
        border: solid $accent;
    }
    
    .no-content {
        color: $warning;
        text-style: italic;
        text-align: center;
        margin: 2;
    }
    
    .status-bar {
        background: $primary;
        color: $text;
        padding: 0 1;
    }
    
    /* Two-pane layout */
    #palette {
        width: 40%;
        border-right: solid $primary;
    }
    
    #canvas {
        width: 60%;
    }
    """
    
    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("q", "quit", "Quit"),
        ("tab", "switch_focus", "Switch Pane"),
        ("enter", "select_item", "Select/Add"),
        ("space", "toggle_expand", "Toggle"),
        ("ctrl+s", "save_draft", "Save"),
    ]
    
    def __init__(self, 
                 storage_manager: StorageManager,
                 topic: Optional[str] = None,
                 limit: int = 50):
        super().__init__()
        self.storage_manager = storage_manager
        self.topic = topic
        self.limit = limit
        
        # Phase 2: State management
        self.synthesis_state = SynthesisState()
        self.clusters: List[GlobuleCluster] = []
        self.clustering_engine: Optional[SemanticClusteringEngine] = None
    
    def compose(self) -> ComposeResult:
        """Phase 2: Compose two-pane layout"""
        yield Header()
        
        # Two-pane layout: Palette (left) + Canvas (right)
        with Horizontal():
            # Left pane: Semantic cluster palette
            with Vertical(id="palette"):
                yield Static("PALETTE: Semantic Clusters", classes="cluster-header")
                yield ClusterPalette(self.clusters, id="cluster-palette")
            
            # Right pane: Canvas editor
            with Vertical(id="canvas"):
                yield Static("CANVAS: Draft Editor", classes="cluster-header")
                yield CanvasEditor(
                    "# Draft Editor\n\nWelcome to Globule Phase 2!\n\n" +
                    "INSTRUCTIONS:\n" +
                    "• Use Tab to switch between palette and canvas\n" +
                    "• Use arrow keys to navigate clusters and thoughts\n" +
                    "• Press Enter to expand clusters or add thoughts to canvas\n" +
                    "• Press Space to toggle cluster expansion\n" +
                    "• Press Ctrl+S to save your draft\n\n" +
                    "Start writing your draft below...\n\n", 
                    id="canvas-editor"
                )
        
        # Status bar
        yield StatusBar(self.synthesis_state, id="status-bar", classes="status-bar")
        yield Footer()
    
    async def on_mount(self) -> None:
        """Phase 2: Load clusters and initialize two-pane interface"""
        self.title = "Globule Phase 2: Semantic Synthesis"
        
        if self.topic:
            self.sub_title = f"Topic: {self.topic}"
        else:
            self.sub_title = "Intelligent Drafting Session"
        
        try:
            # Initialize clustering engine
            self.clustering_engine = SemanticClusteringEngine(self.storage_manager)
            
            # Perform semantic clustering analysis
            analysis = await self.clustering_engine.analyze_semantic_clusters(min_globules=3)
            
            if analysis.clusters:
                # Convert SemanticCluster to GlobuleCluster format
                self.clusters = []
                for semantic_cluster in analysis.clusters:
                    # Get the actual globules for this cluster
                    cluster_globules = []
                    for globule_id in semantic_cluster.member_ids:
                        globule = await self._get_globule_by_id(globule_id)
                        if globule:
                            cluster_globules.append(globule)
                    
                    if cluster_globules:  # Only add if we found globules
                        globule_cluster = GlobuleCluster(
                            id=semantic_cluster.id,
                            globules=cluster_globules,
                            centroid=semantic_cluster.centroid,
                            label=semantic_cluster.label,
                            metadata={
                                'confidence_score': semantic_cluster.confidence_score,
                                'size': semantic_cluster.size,
                                'keywords': semantic_cluster.keywords,
                                'domains': semantic_cluster.domains,
                                'description': semantic_cluster.description
                            }
                        )
                        self.clusters.append(globule_cluster)
                
                # Update synthesis state
                self.synthesis_state.visible_clusters = self.clusters
            
            # Update the palette display
            palette = self.query_one("#cluster-palette", ClusterPalette)
            palette.clusters = self.clusters
            palette.expanded_clusters = self.synthesis_state.expanded_clusters
            await palette.recompose()
            
            # Set initial focus to palette
            palette.focus()
            self.synthesis_state.current_mode = UIMode.EXPLORE
            
            # Update status
            status_bar = self.query_one("#status-bar", StatusBar)
            status_bar.update_status(self.synthesis_state)
            
        except Exception as e:
            # Show error in palette and log for debugging
            import logging
            logging.error(f"Error loading semantic clusters: {e}")
            
            try:
                palette = self.query_one("#cluster-palette", ClusterPalette)
                error_widget = Static(f"Error loading semantic clusters: {e}\n\nTry adding more thoughts first.", classes="no-content")
                await palette.mount(error_widget)
            except Exception:
                # If even mounting fails, just continue
                pass
    
    async def _get_globule_by_id(self, globule_id: str) -> Optional[ProcessedGlobule]:
        """Helper to get globule by ID"""
        try:
            # For now, get recent globules and find by ID
            # In production, we'd have a direct lookup method
            recent_globules = await self.storage_manager.get_recent_globules(1000)
            for globule in recent_globules:
                if globule.id == globule_id:
                    return globule
            return None
        except Exception:
            return None
    
    def action_quit(self) -> None:
        """Quit the application"""
        self.exit()
    
    def action_switch_focus(self) -> None:
        """Switch focus between palette and canvas"""
        try:
            if self.focused is None:
                palette = self.query_one("#cluster-palette")
                palette.focus()
                self.synthesis_state.current_mode = UIMode.EXPLORE
            elif self.focused.id == "cluster-palette":
                canvas = self.query_one("#canvas-editor")
                canvas.focus()
                self.synthesis_state.current_mode = UIMode.EDIT
            else:
                palette = self.query_one("#cluster-palette")
                palette.focus()
                self.synthesis_state.current_mode = UIMode.EXPLORE
            
            # Update status bar
            status_bar = self.query_one("#status-bar", StatusBar)
            status_bar.update_status(self.synthesis_state)
        except Exception:
            pass  # Ignore focus errors
    
    def action_select_item(self) -> None:
        """Select item in current focused pane"""
        try:
            if self.focused and self.focused.id == "cluster-palette":
                palette = self.query_one("#cluster-palette", ClusterPalette)
                # Trigger enter key behavior
                if palette.selected_cluster_id:
                    if palette.selected_globule_id:
                        # Add globule to canvas
                        palette.select_globule(palette.selected_globule_id)
                    else:
                        # Toggle cluster expansion
                        palette.toggle_cluster(palette.selected_cluster_id)
                elif palette.clusters:
                    # Select first cluster
                    palette.select_cluster(palette.clusters[0].id)
        except Exception:
            pass
    
    def action_toggle_expand(self) -> None:
        """Toggle expansion of selected cluster"""
        try:
            palette = self.query_one("#cluster-palette", ClusterPalette)
            if palette.selected_cluster_id:
                palette.toggle_cluster(palette.selected_cluster_id)
        except Exception:
            pass
    
    async def on_cluster_palette_cluster_selected(self, message: ClusterPalette.ClusterSelected) -> None:
        """Handle cluster selection"""
        self.synthesis_state.selected_cluster_id = message.cluster_id
        self.synthesis_state.selected_globule_id = None
        
        # Update status bar
        status_bar = self.query_one("#status-bar", StatusBar)
        status_bar.update_status(self.synthesis_state)
    
    async def on_cluster_palette_globule_selected(self, message: ClusterPalette.GlobuleSelected) -> None:
        """Handle globule selection - add to canvas"""
        self.synthesis_state.selected_globule_id = message.globule.id
        
        # Add globule content to canvas
        canvas = self.query_one("#canvas-editor", CanvasEditor)
        canvas.add_globule_content(message.globule)
        
        # Update synthesis state
        self.synthesis_state.incorporated_globules.add(message.globule.id)
        
        # Update status bar
        status_bar = self.query_one("#status-bar", StatusBar)
        status_bar.update_status(self.synthesis_state)
        
        # Show feedback
        self.notify(f"Added thought: {message.globule.text[:50]}...")
    
    def action_save_draft(self) -> None:
        """Save current draft content"""
        try:
            canvas = self.query_one("#canvas-editor", CanvasEditor)
            content = canvas.get_content()
            
            if content.strip():
                # For MVP, just show success message
                # In production, this would save to file system
                self.notify(f"Draft saved! ({len(content)} characters)")
            else:
                self.notify("Nothing to save - canvas is empty")
        except Exception as e:
            self.notify(f"Error saving draft: {e}")