"""
Refactored Textual TUI for Globule.

This TUI is now a pure presentation layer. All business logic, data access,
and service integrations are handled by the GlobuleAPI.
"""

import asyncio
from typing import Optional, List, Dict, Any

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Header, Footer, Static, TextArea, Tree, Label, Input
from textual.message import Message

from globule.core.api import GlobuleAPI

class ItemSelected(Message):
    """Message to pass selected content to the canvas."""
    def __init__(self, content: str) -> None:
        self.content = content
        super().__init__()

class ThoughtPalette(Vertical):
    """The palette for searching and displaying thoughts."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compose(self) -> ComposeResult:
        yield Input(placeholder="Search thoughts...", id="palette-search")
        yield Tree(label="Results", id="palette-tree")

    async def on_input_submitted(self, event: Input.Submitted):
        if event.input.id == "palette-search":
            query = event.value.strip()
            if query:
                await self.run_search(query)
                event.input.value = ""

    async def run_search(self, query: str):
        tree = self.query_one("#palette-tree", Tree)
        tree.clear()
        tree.root.label = f"Searching for '{query}'..."

        try:
            results = await self.app.api.search_semantic(query, limit=20)
            tree.root.label = f"Results for '{query}' ({len(results)})"
            if not results:
                tree.root.add_leaf("No results found.")
                return

            for globule in results:
                node_label = globule.original_globule.raw_text.split('\n')[0][:80]
                tree.root.add_leaf(node_label, data=globule.original_globule.raw_text)
        except Exception as e:
            tree.root.label = "Error"
            tree.root.add_leaf(f"Search failed: {e}")

    def on_tree_node_selected(self, event: Tree.NodeSelected):
        if event.node.data:
            self.post_message(ItemSelected(event.node.data))

class VizCanvas(Vertical):
    """The main canvas for viewing and editing content."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.text_area = TextArea(id="main-canvas-text")

    def compose(self) -> ComposeResult:
        yield self.text_area

    def on_item_selected(self, message: ItemSelected):
        # Append selected content to the text area
        self.text_area.text += f"\n\n---\n\n{message.content}"
        self.text_area.scroll_end(animate=False)

class DashboardApp(App):
    """A Textual app for Globule."""

    CSS = """
    #palette {
        width: 40%;
        border-right: solid $primary;
        padding: 1;
    }
    #canvas {
        width: 60%;
        padding: 1;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("q", "quit", "Quit"),
        ("ctrl+s", "save_draft", "Save Draft"),
    ]

    def __init__(self, api: GlobuleAPI, topic: Optional[str] = None):
        super().__init__()
        self.api = api
        self.topic = topic

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            yield ThoughtPalette(id="palette")
            yield VizCanvas(id="canvas")
        yield Footer()

    async def on_mount(self):
        """Called when the app is mounted."""
        if self.topic:
            search_input = self.query_one("#palette-search", Input)
            search_input.value = self.topic
            palette = self.query_one(ThoughtPalette)
            await palette.run_search(self.topic)

    def on_item_selected(self, message: ItemSelected):
        """Pass the message to the canvas."""
        canvas = self.query_one(VizCanvas)
        canvas.on_item_selected(message)

    async def action_save_draft(self):
        """Save the content of the canvas to a file."""
        canvas = self.query_one(VizCanvas)
        content = canvas.text_area.text
        if not content:
            self.notify("Canvas is empty. Nothing to save.", severity="warning")
            return

        # Simple file path for now
        file_path = f"draft_{self.topic or 'untitled'}.md".replace(" ", "_")
        success = await self.api.export_draft(content, file_path)
        if success:
            self.notify(f"Draft saved to {file_path}")
        else:
            self.notify(f"Error saving draft to {file_path}", severity="error")

if __name__ == '__main__':
    # This part is for standalone testing of the TUI, if needed.
    # It would require creating a mock API instance.
    class MockAPI(GlobuleAPI):
        async def search_semantic(self, query: str, limit: int = 10) -> List[Any]:
            # Mock implementation
            from globule.core.models import GlobuleV1, ProcessedGlobuleV1
            from uuid import uuid4
            from datetime import datetime
            return [
                ProcessedGlobuleV1(
                    globule_id=uuid4(),
                    original_globule=GlobuleV1(raw_text=f"Mock result for '{query}' - Item {i}"),
                    embedding=[],
                    parsed_data={},
                    processing_time_ms=10
                ) for i in range(5)
            ]
        async def export_draft(self, draft_content: str, file_path: str) -> bool:
            print(f"--- MOCK SAVE ---\nPath: {file_path}\nContent:\n{draft_content}")
            return True

    # To run this file directly for testing:
    # python -m globule.tui.app
    storage = None # MockStorageManager()
    orchestrator = None # MockOrchestrator()
    mock_api = MockAPI(storage, orchestrator)
    app = DashboardApp(api=mock_api, topic="testing")
    app.run()
