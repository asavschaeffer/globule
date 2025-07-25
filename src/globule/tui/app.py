"""
Enhanced Textual TUI for Globule Phase 2.

Phase 2: Two-pane layout with semantic clustering (palette) and canvas editor.
"""

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Header, Footer, Static, TextArea, Tree, Label
from textual.reactive import reactive, var
from textual.binding import Binding
from typing import Optional, List, Set
import asyncio

from globule.core.interfaces import StorageManager
from globule.core.models import ProcessedGlobule, SynthesisState, UIMode, GlobuleCluster
from globule.clustering.semantic_clustering import SemanticClusteringEngine


class ClusterPalette(VerticalScroll):
    """Phase 2: Semantic cluster palette for thought discovery"""
    
    def __init__(self, clusters: List[GlobuleCluster], **kwargs):
        super().__init__(**kwargs)
        self.clusters = clusters
        self.expanded_clusters: Set[str] = set()
    
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
            
            cluster_widget = Static(cluster_title, classes="cluster-title", id=f"cluster-{cluster.id}")
            yield cluster_widget
            
            # Show cluster keywords if available
            keywords = cluster.metadata.get('keywords', [])
            if keywords:
                keyword_text = f"TAGS: {', '.join(keywords[:3])}"
                yield Static(keyword_text, classes="cluster-keywords")
            
            # Show representative samples (expandable)
            if cluster.id in self.expanded_clusters and cluster.globules:
                for i, globule in enumerate(cluster.globules[:3]):  # Show top 3
                    preview = globule.text[:80] + "..." if len(globule.text) > 80 else globule.text
                    sample_widget = Static(f"  THOUGHT: {preview}", classes="globule-sample", id=f"globule-{globule.id}")
                    yield sample_widget
    
    def toggle_cluster(self, cluster_id: str) -> None:
        """Toggle cluster expansion"""
        if cluster_id in self.expanded_clusters:
            self.expanded_clusters.remove(cluster_id)
        else:
            self.expanded_clusters.add(cluster_id)


class CanvasEditor(TextArea):
    """Phase 2: Enhanced canvas editor for drafting with AI assistance"""
    
    def __init__(self, content: str = "", **kwargs):
        super().__init__(content, **kwargs)
        self.incorporated_globules: Set[str] = set()
        
    def add_globule_content(self, globule: ProcessedGlobule) -> None:
        """Add globule content to canvas with context preservation"""
        if globule.id in self.incorporated_globules:
            return  # Already incorporated
            
        # Format the globule content for integration
        formatted_content = f"\n\n## {globule.parsed_data.get('title', 'Thought')}\n\n{globule.text}\n"
        
        # Add to current cursor position
        current_content = self.text
        cursor_pos = len(current_content)  # Append at end for MVP
        
        new_content = current_content[:cursor_pos] + formatted_content + current_content[cursor_pos:]
        self.text = new_content
        
        # Track incorporation
        self.incorporated_globules.add(globule.id)


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
        ("enter", "select_item", "Select"),
        ("space", "toggle_expand", "Toggle"),
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
                    "# Draft Editor\n\nStart writing your draft here...\nUse Tab to switch to palette and select thoughts to incorporate.\n\n", 
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
            await palette.recompose()
            
            # Update status
            status_bar = self.query_one("#status-bar", StatusBar)
            status_bar.update_status(self.synthesis_state)
            
        except Exception as e:
            # Show error in palette
            error_widget = Static(f"Error loading semantic clusters: {e}", classes="no-content")
            palette = self.query_one("#cluster-palette", ClusterPalette)
            await palette.mount(error_widget)
    
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
            elif self.focused.id == "cluster-palette":
                canvas = self.query_one("#canvas-editor")
                canvas.focus()
            else:
                palette = self.query_one("#cluster-palette")
                palette.focus()
        except Exception:
            pass  # Ignore focus errors
    
    def action_select_item(self) -> None:
        """Select item in current focused pane"""
        # This would handle selecting clusters or globules
        # For MVP, we'll implement basic functionality
        pass
    
    def action_toggle_expand(self) -> None:
        """Toggle expansion of selected cluster"""
        # For MVP, expand/collapse clusters in palette
        pass