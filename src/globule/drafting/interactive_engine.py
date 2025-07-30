"""
Interactive Drafting Engine for Globule.

Implements a headless TUI system with simple keyboard navigation for building
drafts from clustered thoughts. No external TUI frameworks - just a stateful
command-line loop with rich rendering.
"""

import os
import sys
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.layout import Layout
from rich.align import Align

from globule.core.models import ProcessedGlobule
from globule.clustering.semantic_clustering import ClusteringAnalysis, SemanticCluster


class DraftingView(Enum):
    """The two main views in the interactive drafting system."""
    CLUSTER_VIEW = "cluster_view"
    GLOBULE_VIEW = "globule_view"


@dataclass
class DraftingState:
    """
    Core state for the interactive drafting system.
    
    This simple data structure tracks everything needed for the headless TUI:
    - Which view we're in (clusters vs globules)
    - Selection indices for navigation
    - Accumulated draft content
    - Data from clustering analysis
    """
    
    # Navigation state
    current_view: DraftingView = DraftingView.CLUSTER_VIEW
    selected_cluster_index: int = 0
    selected_globule_index: int = 0
    
    # Content state
    draft_content: List[str] = field(default_factory=list)
    topic: Optional[str] = None
    
    # Data from analysis
    clusters: List[SemanticCluster] = field(default_factory=list)
    globules_by_cluster: Dict[str, List[ProcessedGlobule]] = field(default_factory=dict)
    all_globules: List[ProcessedGlobule] = field(default_factory=list)
    
    # UI state
    should_quit: bool = False
    status_message: str = ""
    
    def get_current_cluster(self) -> Optional[SemanticCluster]:
        """Get the currently selected cluster."""
        if 0 <= self.selected_cluster_index < len(self.clusters):
            return self.clusters[self.selected_cluster_index]
        return None
    
    def get_current_cluster_globules(self) -> List[ProcessedGlobule]:
        """Get globules for the currently selected cluster."""
        cluster = self.get_current_cluster()
        if cluster and cluster.id in self.globules_by_cluster:
            return self.globules_by_cluster[cluster.id]
        return []
    
    def get_current_globule(self) -> Optional[ProcessedGlobule]:
        """Get the currently selected globule in the current cluster."""
        globules = self.get_current_cluster_globules()
        if 0 <= self.selected_globule_index < len(globules):
            return globules[self.selected_globule_index]
        return None
    
    def add_to_draft(self, text: str) -> None:
        """Add content to the draft."""
        self.draft_content.append(text)
        self.status_message = f"Added to draft ({len(self.draft_content)} items)"
    
    def get_draft_text(self) -> str:
        """Get the complete draft as a single string."""
        header = [
            f"# Draft: {self.topic or 'My Thoughts'}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Items: {len(self.draft_content)}",
            "",
        ]
        
        content = []
        for i, item in enumerate(self.draft_content, 1):
            content.append(f"## Item {i}")
            content.append("")
            content.append(item)
            content.append("")
        
        return "\\n".join(header + content)


class InteractiveDraftingEngine:
    """
    Headless TUI engine for interactive drafting.
    
    Implements a simple keyboard-driven interface without external TUI frameworks.
    Uses a stateful loop that redraws on each keypress for maximum control and performance.
    """
    
    def __init__(self):
        self.console = Console()
        self.state = DraftingState()
        
        # Terminal handling for keypress detection
        self._old_settings = None
    
    async def run_interactive_session(
        self, 
        topic: Optional[str],
        clusters: List[SemanticCluster],
        globules_by_cluster: Dict[str, List[ProcessedGlobule]],
        all_globules: List[ProcessedGlobule]
    ) -> str:
        """
        Run the main interactive drafting session.
        
        Args:
            topic: The drafting topic
            clusters: List of semantic clusters from analysis
            globules_by_cluster: Map of cluster ID to globules
            all_globules: All globules that were clustered
            
        Returns:
            The final draft content as a string
        """
        # Initialize state
        self.state.topic = topic
        self.state.clusters = clusters
        self.state.globules_by_cluster = globules_by_cluster
        self.state.all_globules = all_globules
        self.state.status_message = f"Interactive drafting: {len(clusters)} clusters, {len(all_globules)} globules"
        
        if not clusters:
            return self._handle_no_clusters()
        
        # Setup terminal for keypress detection
        self._setup_terminal()
        
        try:
            # Main interactive loop
            while not self.state.should_quit:
                self._render_ui()
                keypress = self._get_keypress()
                self._handle_keypress(keypress)
            
            # Return final draft
            return self.state.get_draft_text()
            
        finally:
            self._restore_terminal()
    
    def _handle_no_clusters(self) -> str:
        """Handle case where no clusters were found."""
        self.console.print("[yellow]NO CLUSTERS:[/yellow] No semantic clusters found.")
        self.console.print("Using chronological listing of thoughts...")
        
        # Create simple draft from all globules
        draft_lines = [
            f"# Draft: {self.state.topic or 'My Thoughts'}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Total thoughts: {len(self.state.all_globules)}",
            "",
            "## Your Thoughts",
            ""
        ]
        
        for i, globule in enumerate(self.state.all_globules[:20], 1):
            draft_lines.append(f"{i}. {globule.text}")
            draft_lines.append("")
        
        return "\\n".join(draft_lines)
    
    def _render_ui(self) -> None:
        """Render the current UI state using rich library."""
        # Clear screen
        self.console.clear()
        
        # Create layout
        if self.state.current_view == DraftingView.CLUSTER_VIEW:
            self._render_cluster_view()
        else:
            self._render_globule_view()
        
        # Show status and help
        self._render_status_bar()
    
    def _render_cluster_view(self) -> None:
        """Render the cluster selection view."""
        # Title
        title = f"Draft: {self.state.topic or 'Interactive Mode'}"
        self.console.print(Panel(title, style="bold blue"))
        
        # Clusters table
        table = Table(title="Semantic Clusters", show_header=True)
        table.add_column("", style="dim", width=3)
        table.add_column("Cluster", style="cyan", min_width=20)
        table.add_column("Size", style="green", width=6)
        table.add_column("Confidence", style="yellow", width=10)
        table.add_column("Keywords", style="magenta")
        
        for i, cluster in enumerate(self.state.clusters):
            # Highlight selected row
            style = "bold white on blue" if i == self.state.selected_cluster_index else ""
            
            # Selection indicator
            indicator = "→" if i == self.state.selected_cluster_index else ""
            
            # Format confidence as percentage
            confidence = f"{cluster.confidence_score * 100:.1f}%"
            
            # Format keywords (limit to avoid overflow)
            keywords = ", ".join(cluster.keywords[:3])
            if len(cluster.keywords) > 3:
                keywords += "..."
            
            table.add_row(
                indicator,
                cluster.label,
                str(cluster.size),
                confidence,
                keywords,
                style=style
            )
        
        self.console.print(table)
        
        # Draft preview
        if self.state.draft_content:
            draft_preview = f"Draft: {len(self.state.draft_content)} items"
            self.console.print(Panel(draft_preview, title="Current Draft", style="green"))
    
    def _render_globule_view(self) -> None:
        """Render the globule selection view within a cluster."""
        cluster = self.state.get_current_cluster()
        if not cluster:
            return
        
        # Title with cluster info
        title = f"Cluster: {cluster.label} ({cluster.size} thoughts)"
        self.console.print(Panel(title, style="bold cyan"))
        
        # Globules list
        globules = self.state.get_current_cluster_globules()
        
        for i, globule in enumerate(globules):
            # Highlight selected globule
            style = "bold white on blue" if i == self.state.selected_globule_index else ""
            
            # Selection indicator
            indicator = "→ " if i == self.state.selected_globule_index else "  "
            
            # Truncate long text for display
            display_text = globule.text
            if len(display_text) > 80:
                display_text = display_text[:77] + "..."
            
            # Show globule with selection
            text = Text(f"{indicator}{i+1}. {display_text}")
            if style:
                text.stylize(style)
            
            self.console.print(text)
        
        # Show selected globule details
        current_globule = self.state.get_current_globule()
        if current_globule:
            details = Panel(
                current_globule.text,
                title="Selected Thought",
                style="yellow"
            )
            self.console.print(details)
    
    def _render_status_bar(self) -> None:
        """Render status bar with help and current state."""
        # Status message
        if self.state.status_message:
            self.console.print(f"Status: {self.state.status_message}", style="dim")
        
        # Help based on current view
        if self.state.current_view == DraftingView.CLUSTER_VIEW:
            help_text = "↑↓: Navigate | →/Enter: View cluster | d/q: Finish draft"
        else:
            help_text = "↑↓: Navigate | Enter: Add to draft | ←/Backspace: Back | d/q: Finish"
        
        self.console.print(f"Controls: {help_text}", style="dim")
        self.console.print("", end="")  # Ensure cursor positioning
    
    def _setup_terminal(self) -> None:
        """Setup terminal for single keypress detection."""
        if sys.platform != 'win32':  # Unix-like systems
            try:
                import termios
                import tty
                self._old_settings = termios.tcgetattr(sys.stdin)
                tty.setraw(sys.stdin.fileno())
            except ImportError:
                # Fallback for systems without termios
                pass
    
    def _restore_terminal(self) -> None:
        """Restore terminal to original state."""
        if sys.platform != 'win32' and self._old_settings:
            try:
                import termios
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
            except ImportError:
                pass
    
    def _get_keypress(self) -> str:
        """Get a single keypress from the user."""
        if sys.platform == 'win32':
            import msvcrt
            key = msvcrt.getch()
            if key == b'\\x00' or key == b'\\xe0':  # Special keys on Windows
                key = msvcrt.getch()
                # Map Windows special keys
                key_map = {b'H': 'up', b'P': 'down', b'K': 'left', b'M': 'right'}
                return key_map.get(key, 'unknown')
            return key.decode('utf-8', errors='ignore')
        else:
            key = sys.stdin.read(1)
            if key == '\\x1b':  # Escape sequence
                seq = sys.stdin.read(2)
                if seq == '[A': return 'up'
                elif seq == '[B': return 'down'
                elif seq == '[C': return 'right'
                elif seq == '[D': return 'left'
            return key
    
    def _handle_keypress(self, key: str) -> None:
        """Handle a keypress and update state accordingly."""
        self.state.status_message = ""  # Clear previous status
        
        # Global commands
        if key in ('d', 'q', '\\x03', '\\x04'):  # d, q, Ctrl+C, Ctrl+D
            self.state.should_quit = True
            return
        
        # Navigation based on current view
        if self.state.current_view == DraftingView.CLUSTER_VIEW:
            self._handle_cluster_view_keypress(key)
        else:
            self._handle_globule_view_keypress(key)
    
    def _handle_cluster_view_keypress(self, key: str) -> None:
        """Handle keypresses in cluster view."""
        if key == 'up' and self.state.selected_cluster_index > 0:
            self.state.selected_cluster_index -= 1
        
        elif key == 'down' and self.state.selected_cluster_index < len(self.state.clusters) - 1:
            self.state.selected_cluster_index += 1
        
        elif key in ('right', '\\r', '\\n'):  # Right arrow or Enter
            # Switch to globule view for selected cluster
            if self.state.get_current_cluster():
                self.state.current_view = DraftingView.GLOBULE_VIEW
                self.state.selected_globule_index = 0  # Reset selection
        
        else:
            self.state.status_message = f"Unknown key: {repr(key)}"
    
    def _handle_globule_view_keypress(self, key: str) -> None:
        """Handle keypresses in globule view."""
        globules = self.state.get_current_cluster_globules()
        
        if key == 'up' and self.state.selected_globule_index > 0:
            self.state.selected_globule_index -= 1
        
        elif key == 'down' and self.state.selected_globule_index < len(globules) - 1:
            self.state.selected_globule_index += 1
        
        elif key in ('left', '\\x08', '\\x7f'):  # Left arrow, Backspace, Delete
            # Go back to cluster view
            self.state.current_view = DraftingView.CLUSTER_VIEW
        
        elif key in ('\\r', '\\n'):  # Enter - add globule to draft
            current_globule = self.state.get_current_globule()
            if current_globule:
                self.state.add_to_draft(current_globule.text)
        
        else:
            self.state.status_message = f"Unknown key: {repr(key)}"