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
from globule.services.clustering.semantic_clustering import ClusteringAnalysis, SemanticCluster


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
        """Get the currently selected cluster, or None if invalid."""
        if not self.clusters or not (0 <= self.selected_cluster_index < len(self.clusters)):
            if not self.clusters:
                self.status_message = "No clusters available"
            else:
                self.status_message = "Invalid cluster selection"
            return None
        return self.clusters[self.selected_cluster_index]
    
    def get_current_cluster_globules(self) -> List[ProcessedGlobule]:
        """Get globules for the currently selected cluster, or empty list if invalid."""
        cluster = self.get_current_cluster()
        if not cluster:
            return []
        
        if cluster.id not in self.globules_by_cluster:
            self.status_message = "No globules mapped for selected cluster"
            return []
        
        globules = self.globules_by_cluster[cluster.id]
        if not globules:
            self.status_message = "Selected cluster has no globules"
        
        return globules
    
    def get_current_globule(self) -> Optional[ProcessedGlobule]:
        """Get the currently selected globule, or None if invalid."""
        globules = self.get_current_cluster_globules()
        if not globules:
            return None
        
        if not (0 <= self.selected_globule_index < len(globules)):
            self.status_message = "Invalid globule selection"
            return None
        
        return globules[self.selected_globule_index]
    
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
        Run the main interactive drafting session with robust error handling.
        
        Args:
            topic: The drafting topic
            clusters: List of semantic clusters from analysis
            globules_by_cluster: Map of cluster ID to globules
            all_globules: All globules that were clustered
            
        Returns:
            The final draft content as a string
            
        Raises:
            ValueError: If input arguments are invalid
        """
        # Validate input arguments to prevent crashes from malformed data
        if not isinstance(clusters, list):
            raise ValueError("clusters must be a list")
        if not isinstance(globules_by_cluster, dict):
            raise ValueError("globules_by_cluster must be a dictionary")
        if not isinstance(all_globules, list):
            raise ValueError("all_globules must be a list")
        
        # Initialize state
        self.state.topic = topic
        self.state.clusters = clusters
        self.state.globules_by_cluster = globules_by_cluster
        self.state.all_globules = all_globules
        self.state.status_message = f"Interactive drafting: {len(clusters)} clusters, {len(all_globules)} globules"
        
        if not clusters:
            return self._handle_no_clusters()
        
        # Setup terminal for keypress detection with validation
        if not self._setup_terminal():
            self.state.status_message = "Failed to initialize terminal, falling back to default draft"
            return self._handle_no_clusters()
        
        try:
            # Main interactive loop with robust error handling
            while not self.state.should_quit:
                try:
                    self._render_ui()
                    keypress = self._get_keypress()
                    self._handle_keypress(keypress)
                except KeyboardInterrupt:
                    self.state.status_message = "Session interrupted by user (Ctrl+C)"
                    self.state.should_quit = True
                except Exception as e:
                    self.state.status_message = f"Error: {str(e)}"
                    print(f"Error in interactive session: {str(e)}", file=sys.stderr)
                    # Continue running unless it's a critical error
                    if "critical" in str(e).lower():
                        self.state.should_quit = True
            
            # Return final draft
            return self.state.get_draft_text()
            
        finally:
            # CRITICAL: Always restore terminal state, even if exception occurs
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
        """Render the current UI state using rich library with size validation."""
        # Clear screen
        self.console.clear()
        
        # Check terminal size for usability
        if self.console.width < 80 or self.console.height < 20:
            self.console.print("[red]Terminal too small for optimal experience[/red]")
            self.console.print(f"Current: {self.console.width}x{self.console.height}, Recommended: 80x20 or larger")
            self.console.print("Resize terminal or press 'd' to continue anyway")
            return
        
        # Create layout based on current view
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
    
    def _setup_terminal(self) -> bool:
        """
        Setup terminal for single keypress detection.
        
        Returns:
            True if terminal setup was successful, False otherwise.
            On Unix, uses termios to enable raw mode; on Windows, relies on msvcrt.
        """
        if sys.platform != 'win32':  # Unix-like systems
            try:
                import termios
                import tty
                self._old_settings = termios.tcgetattr(sys.stdin)
                tty.setraw(sys.stdin.fileno())
                return True
            except (ImportError, OSError, termios.error) as e:
                self.state.status_message = f"Terminal setup failed: {str(e)}"
                return False
        # Windows terminal handling is simpler and generally reliable
        return True
    
    def _restore_terminal(self) -> None:
        """
        Restore terminal to original state if settings were saved.
        
        Silently handles failures as terminal state may already be broken.
        """
        if sys.platform != 'win32' and self._old_settings is not None:
            try:
                import termios
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
                self._old_settings = None  # Clear to prevent reuse
            except (ImportError, OSError, termios.error):
                # Silently fail - terminal state is already compromised
                pass
    
    def _get_keypress(self) -> str:
        """
        Get a single keypress from the user with robust error handling.
        
        Handles Windows (msvcrt) and Unix (termios) systems with fallback for errors.
        """
        if sys.platform == 'win32':
            try:
                import msvcrt
                key = msvcrt.getch()
                if key in (b'\\x00', b'\\xe0'):  # Special keys on Windows
                    key = msvcrt.getch()
                    # Map Windows special keys
                    key_map = {b'H': 'up', b'P': 'down', b'K': 'left', b'M': 'right'}
                    return key_map.get(key, 'unknown')
                return key.decode('utf-8', errors='ignore')
            except Exception as e:
                self.state.status_message = f"Keypress error on Windows: {str(e)}"
                return ''
        else:
            try:
                key = sys.stdin.read(1)
                if key == '\\x1b':  # Escape sequence
                    seq = sys.stdin.read(2)
                    if seq == '[A': return 'up'
                    elif seq == '[B': return 'down'
                    elif seq == '[C': return 'right'
                    elif seq == '[D': return 'left'
                return key
            except Exception as e:
                self.state.status_message = f"Keypress error on Unix: {str(e)}"
                return ''
    
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