"""
Basic Textual TUI for Globule MVP.

Phase 1: Simple display of recent globules in a non-interactive list.
"""

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Header, Footer, Static
from textual.reactive import reactive
from typing import Optional, List

from globule.core.interfaces import StorageManager
from globule.core.models import ProcessedGlobule


class GlobuleList(VerticalScroll):
    """Widget to display a list of globules"""
    
    def __init__(self, globules: List[ProcessedGlobule], **kwargs):
        super().__init__(**kwargs)
        self.globules = globules
    
    def compose(self) -> ComposeResult:
        """Display globules as simple text blocks"""
        if not self.globules:
            yield Static("No globules found. Try adding some with 'globule add \"your thought\"'")
            return
        
        yield Static(f"Found {len(self.globules)} recent thoughts:", classes="header")
        
        for i, globule in enumerate(self.globules):
            # Format globule display
            title = f"#{i+1}: "
            if globule.parsed_data.get("title"):
                title += globule.parsed_data["title"][:60]
            else:
                title += globule.text[:60]
            
            if len(globule.text) > 60:
                title += "..."
            
            # Show content preview
            content_preview = globule.text[:200]
            if len(globule.text) > 200:
                content_preview += "..."
            
            # Create globule widget
            globule_widget = Static(
                f"[bold]{title}[/bold]\n{content_preview}\n",
                classes="globule-item"
            )
            yield globule_widget


class SynthesisApp(App):
    """Main Textual application for synthesis"""
    
    CSS = """
    .header {
        color: $accent;
        margin-bottom: 1;
    }
    
    .globule-item {
        border: solid $primary;
        margin-bottom: 1;
        padding: 1;
    }
    
    .globule-item:hover {
        background: $primary-background;
    }
    """
    
    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("q", "quit", "Quit"),
    ]
    
    def __init__(self, 
                 storage_manager: StorageManager,
                 topic: Optional[str] = None,
                 limit: int = 50):
        super().__init__()
        self.storage_manager = storage_manager
        self.topic = topic
        self.limit = limit
        self.globules: List[ProcessedGlobule] = []
    
    def compose(self) -> ComposeResult:
        """Compose the app layout"""
        yield Header()
        
        # Main content area
        with Horizontal():
            # For Phase 1, show all globules in a single pane
            # Phase 3 will split into Palette (left) and Canvas (right)
            yield GlobuleList(self.globules, id="globule-list")
        
        yield Footer()
    
    async def on_mount(self) -> None:
        """Load globules when app starts"""
        self.title = "Globule Draft Session"
        
        if self.topic:
            self.sub_title = f"Topic: {self.topic}"
        else:
            self.sub_title = "Recent Thoughts"
        
        # Load recent globules
        try:
            self.globules = await self.storage_manager.get_recent_globules(self.limit)
            
            # Update the display
            globule_list = self.query_one("#globule-list", GlobuleList)
            globule_list.globules = self.globules
            
            # Force refresh of the list
            await globule_list.recompose()
            
        except Exception as e:
            # Show error message
            error_widget = Static(f"Error loading globules: {e}")
            globule_list = self.query_one("#globule-list", GlobuleList)
            await globule_list.mount(error_widget)
    
    def action_quit(self) -> None:
        """Quit the application"""
        self.exit()