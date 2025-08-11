"""
Enhanced Textual TUI for Globule with Analytics Dashboard.

Progressive enhancement: Two-pane layout with configurable dashboards.
Palette: Data/query explorer (vector + analytics queries via schemas)
Canvas: View composer (text + visualization)
"""

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Header, Footer, Static, TextArea, Tree, Label, Input
from textual.reactive import reactive, var
from textual.binding import Binding
from textual.message import Message
from textual import events
from textual.events import MouseDown, MouseMove, MouseUp
from typing import Optional, List, Set, Dict, Any
import asyncio
import json
import base64
import sqlite3
from datetime import datetime, timedelta

from globule.core.interfaces import StorageManager
from globule.core.models import ProcessedGlobule, SynthesisState, UIMode, GlobuleCluster
from globule.services.clustering.semantic_clustering import SemanticClusteringEngine
from globule.services.parsing.ollama_parser import OllamaParser
from globule.schemas.manager import get_schema_manager, detect_schema_for_text


class AnalyticsPalette(VerticalScroll):
    """Enhanced palette: Data/query explorer with analytics capabilities"""
    
    class QueryExecuted(Message):
        """Message sent when a query is executed"""
        def __init__(self, query: str, result: Any, query_type: str = "sql") -> None:
            self.query = query
            self.result = result
            self.query_type = query_type
            super().__init__()
    
    class GlobuleSelected(Message):
        """Message sent when a globule is selected"""
        def __init__(self, globule: ProcessedGlobule) -> None:
            self.globule = globule
            super().__init__()
    
    class QueryResultDragged(Message):
        """Message sent when a query result is dragged to canvas"""
        def __init__(self, query_result: Dict[str, Any]) -> None:
            self.query_result = query_result
            super().__init__()
    
    def __init__(self, storage_manager: StorageManager, topic: str = None, **kwargs):
        super().__init__(**kwargs)
        self.storage_manager = storage_manager
        self.topic = topic
        self.schema_manager = get_schema_manager()
        self.detected_schema = None
        self.clusters: List[GlobuleCluster] = []
        self.query_results: List[Dict[str, Any]] = []
        self.expanded_sections: Set[str] = set()
        self.can_focus = True
        self.drag_data: Optional[Dict[str, Any]] = None
        self.is_dragging = False
        
        # Detect schema from topic
        if topic:
            self.detected_schema = detect_schema_for_text(topic)
    
    def compose(self) -> ComposeResult:
        """Compose analytics palette with query input and results tree"""
        yield Static("📊 ANALYTICS PALETTE", classes="palette-header")
        
        # Schema detection info
        if self.detected_schema:
            yield Static(f"🎯 Schema: {self.detected_schema}", classes="schema-info")
        
        # Query input section
        yield Static("🔍 QUERY EXPLORER:", classes="section-header")
        
        # Pre-populate with schema-specific query if available
        placeholder_query = self._get_default_query()
        yield Input(
            placeholder=placeholder_query,
            id="query-input"
        )
        
        # Pre-defined queries from schema
        schema_queries = self._get_schema_queries()
        if schema_queries:
            yield Static("📋 PRE-DEFINED QUERIES:", classes="section-header")
            queries_tree = Tree("Quick Queries", id="schema-queries")
            root = queries_tree.root
            for i, query in enumerate(schema_queries):
                query_node = root.add(f"▶️ {query['name']}")
                query_node.add_leaf(f"📝 {query['description']}")
                query_node.add_leaf(f"📊 Viz: {query.get('visualization', 'table')}")
            yield queries_tree
        
        # Query results section
        yield Static("📈 RESULTS:", classes="section-header")
        yield Tree("Analytics", id="results-tree")
        
        # Clusters section
        if self.clusters:
            yield Static(f"🗂️ CLUSTERS ({len(self.clusters)}):", classes="section-header")
            clusters_tree = Tree("Clusters", id="clusters-tree")
            root = clusters_tree.root
            for cluster in self.clusters:
                confidence_bar = "█" * max(1, int(cluster.metadata.get('confidence_score', 0.5) * 10))
                cluster_title = f"{cluster.label} ({len(cluster.globules)}) [{confidence_bar}]"
                root.add_leaf(cluster_title)
            yield clusters_tree
    
    def _get_default_query(self) -> str:
        """Get default query based on detected schema"""
        if self.detected_schema == 'valet' or self.detected_schema == 'valet_enhanced':
            return "SELECT COUNT(*) as cars_today FROM globules WHERE date(created_at) = date('now') AND parsed_data LIKE '%valet%'"
        elif self.detected_schema == 'technical':
            return "SELECT category, COUNT(*) FROM globules WHERE parsed_data LIKE '%technical%' GROUP BY category"
        else:
            return "SELECT COUNT(*) as total_thoughts FROM globules WHERE created_at >= date('now', '-7 days')"
    
    def _get_schema_queries(self) -> List[Dict[str, str]]:
        """Get pre-defined queries from schema"""
        if not self.detected_schema:
            return []
        
        schema = self.schema_manager.get_schema(self.detected_schema)
        if not schema:
            return []
        
        # Check for dashboard_queries in schema
        return schema.get('dashboard_queries', [])
    
    async def execute_query(self, query: str) -> None:
        """Execute SQL query and update results tree"""
        try:
            # Get database connection from storage manager
            if hasattr(self.storage_manager, 'db_path'):
                db_path = self.storage_manager.db_path
            else:
                # Fallback path
                import os
                db_path = os.path.expanduser("~/.globule/data/globule.db")
            
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Execute query
                cursor.execute(query)
                rows = cursor.fetchall()
                
                # Convert to list of dicts
                results = [dict(row) for row in rows]
                
                # Add to query results
                query_result = {
                    'query': query,
                    'results': results,
                    'timestamp': datetime.now().isoformat(),
                    'type': 'sql'
                }
                self.query_results.append(query_result)
                
                # Update results tree
                await self._update_results_tree()
                
                # Post message for canvas to handle
                self.post_message(self.QueryExecuted(query, results, "sql"))
                
        except Exception as e:
            # Add error result
            error_result = {
                'query': query,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'type': 'error'
            }
            self.query_results.append(error_result)
            await self._update_results_tree()
    
    async def _update_results_tree(self) -> None:
        """Update the results tree with latest query results"""
        try:
            tree = self.query_one("#results-tree", Tree)
            tree.clear()
            root = tree.root
            
            if not self.query_results:
                root.add_leaf("No queries executed yet")
                return
            
            for i, result in enumerate(self.query_results[-5:]):  # Show last 5 results
                if 'error' in result:
                    node = root.add(f"❌ Query {i+1}: Error")
                    node.add_leaf(f"Error: {result['error']}")
                else:
                    results_data = result['results']
                    node = root.add(f"✅ Query {i+1}: {len(results_data)} rows")
                    
                    # Add summary of results
                    if results_data:
                        for j, row in enumerate(results_data[:3]):  # Show first 3 rows
                            if isinstance(row, dict):
                                summary = ", ".join([f"{k}: {v}" for k, v in row.items()])
                                node.add_leaf(f"Row {j+1}: {summary}")
                            else:
                                node.add_leaf(f"Row {j+1}: {row}")
                        
                        if len(results_data) > 3:
                            node.add_leaf(f"... and {len(results_data) - 3} more rows")
        except Exception as e:
            # Ignore tree update errors
            pass
    
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle query input submission"""
        if event.input.id == "query-input":
            query = event.value.strip()
            if query:
                await self.execute_query(query)
                event.input.value = ""  # Clear input
    
    async def load_clusters(self, clusters: List[GlobuleCluster]) -> None:
        """Load clusters into the palette"""
        self.clusters = clusters
        await self.recompose()
    
    async def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """Handle tree node selection (pre-defined queries)"""
        try:
            if event.tree.id == "schema-queries":
                # Get the selected query
                node_label = str(event.node.label)
                if node_label.startswith("▶️"):
                    query_name = node_label[3:].strip()  # Remove arrow and spaces
                    
                    # Find matching query in schema
                    schema_queries = self._get_schema_queries()
                    for query in schema_queries:
                        if query['name'] == query_name:
                            # Execute the pre-defined query
                            await self.execute_query(query['sql'])
                            break
            elif event.tree.id == "results-tree":
                # Prepare data for potential drag operation
                node_label = str(event.node.label)
                if node_label.startswith("✅ Query"):
                    # Find the corresponding query result
                    try:
                        query_num = int(node_label.split(":")[0].split()[-1]) - 1
                        if 0 <= query_num < len(self.query_results):
                            self.drag_data = self.query_results[-(query_num + 1)]
                    except (ValueError, IndexError):
                        pass
        except Exception:
            # Ignore tree selection errors
            pass
    
    async def on_mouse_down(self, event: MouseDown) -> None:
        """Handle mouse down for drag initiation"""
        # Check if we're over the results tree and have drag data
        if self.drag_data and event.button == 1:  # Left click
            self.is_dragging = True
    
    async def on_mouse_up(self, event: MouseUp) -> None:
        """Handle mouse up to complete drag operation"""
        if self.is_dragging and self.drag_data:
            # Check if we're dropping onto the canvas area
            # For now, just send the message - the app will handle positioning
            self.post_message(self.QueryResultDragged(self.drag_data))
            self.is_dragging = False
            self.drag_data = None


class VizCanvas(TextArea):
    """Enhanced canvas: View composer with text + visualization support"""
    
    class AIActionRequested(Message):
        """Message sent when AI action is requested"""
        def __init__(self, action: str, text: str, context: Dict[str, Any] = None) -> None:
            self.action = action  # 'expand' or 'summarize'
            self.text = text
            self.context = context or {}
            super().__init__()
    
    def __init__(self, content: str = "", **kwargs):
        super().__init__(content, **kwargs)
        self.incorporated_results: List[Dict[str, Any]] = []
        self.dashboard_mode = False
        self.can_focus = True
        self._selected_text: Optional[str] = None
    
    def add_dragged_result(self, query_result: Dict[str, Any]) -> None:
        """Add dragged query result to canvas at cursor position"""
        query = query_result.get('query', 'Unknown Query')
        result = query_result.get('results', [])
        query_type = query_result.get('type', 'sql')
        
        # Generate visualization
        viz_content = self._generate_visualization(query, result, query_type)
        
        # Insert at current cursor position
        cursor = self.cursor_position
        current_text = self.text
        
        # Add header with drag indicator
        new_content = f"\n\n## 🎯 Dragged Query Result\n\n**Query:** `{query}`\n**Dragged at:** {datetime.now().strftime('%H:%M:%S')}\n\n{viz_content}\n"
        
        # Insert at cursor position
        before = current_text[:cursor[1] if cursor else len(current_text)]
        after = current_text[cursor[1] if cursor else len(current_text):]
        self.text = before + new_content + after
        
        # Track incorporation
        self.incorporated_results.append({
            'query': query,
            'result': result,
            'type': query_type,
            'timestamp': datetime.now().isoformat(),
            'method': 'drag_drop'
        })
    
    def get_selected_text(self) -> Optional[str]:
        """Get currently selected text from the text area"""
        # In a real implementation, this would get actual selection
        # For now, we'll simulate with cursor position context
        cursor = self.cursor_position
        if cursor:
            # Get the current line as "selected" text for demo
            lines = self.text.split('\n')
            if cursor[0] < len(lines):
                return lines[cursor[0]].strip()
        return None
    
    def expand_text_ai(self, text: str) -> None:
        """Request AI expansion of text"""
        if text:
            self.post_message(self.AIActionRequested('expand', text, {'position': self.cursor_position}))
    
    def summarize_text_ai(self, text: str) -> None:
        """Request AI summarization of text"""
        if text:
            self.post_message(self.AIActionRequested('summarize', text, {'position': self.cursor_position}))
    
    def add_query_result(self, query: str, result: Any, query_type: str = "sql") -> None:
        """Add query result to canvas with visualization"""
        # Generate visualization based on result type
        viz_content = self._generate_visualization(query, result, query_type)
        
        # Add to current cursor position
        current_content = self.text
        cursor_pos = len(current_content)  # Append at end for now
        
        new_content = current_content + f"\n\n## Query Result\n\n**Query:** `{query}`\n\n{viz_content}\n"
        self.text = new_content
        
        # Track incorporation
        self.incorporated_results.append({
            'query': query,
            'result': result,
            'type': query_type,
            'timestamp': datetime.now().isoformat()
        })
    
    def _generate_visualization(self, query: str, result: Any, query_type: str) -> str:
        """Generate visualization content (text/tables/charts) from query results"""
        if query_type == "error":
            return f"❌ **Error:** {result}\n"
        
        if not result or len(result) == 0:
            return "📊 **No data found**\n"
        
        try:
            # Detect visualization type based on query patterns
            viz_type = self._detect_visualization_type(query, result)
            
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict):
                    return self._generate_enhanced_visualization(result, viz_type, query)
                else:
                    return self._generate_simple_list(result)
            
            return f"📊 **Result:** {result}\n"
            
        except Exception as e:
            return f"❌ **Visualization error:** {e}\n"
    
    def _detect_visualization_type(self, query: str, result: Any) -> str:
        """Detect appropriate visualization type based on query and result"""
        query_lower = query.lower()
        
        # Check for specific patterns
        if 'count' in query_lower and 'group by' in query_lower:
            if isinstance(result, list) and len(result) <= 5:
                return 'pie'
            else:
                return 'bar'
        elif 'date' in query_lower and 'group by' in query_lower:
            return 'line'
        elif any(word in query_lower for word in ['summary', 'total', 'avg', 'sum']):
            return 'metrics'
        else:
            return 'table'
    
    def _generate_enhanced_visualization(self, data: List[Dict], viz_type: str, query: str) -> str:
        """Generate enhanced visualization based on type"""
        if viz_type == 'metrics':
            return self._generate_metrics_cards(data)
        elif viz_type == 'bar':
            return self._generate_bar_chart(data)
        elif viz_type == 'pie':
            return self._generate_pie_chart(data)
        elif viz_type == 'line':
            return self._generate_line_chart(data)
        else:
            return self._generate_table(data)
    
    def _generate_metrics_cards(self, data: List[Dict]) -> str:
        """Generate metrics cards visualization"""
        if not data or not isinstance(data[0], dict):
            return self._generate_table(data)
        
        cards = []
        for row in data[:1]:  # Usually metrics are single row
            for key, value in row.items():
                cards.append(f"**{key.replace('_', ' ').title()}**: {value}")
        
        return "📊 **Key Metrics**\n\n" + " | ".join(cards) + "\n\n"
    
    def _generate_bar_chart(self, data: List[Dict]) -> str:
        """Generate ASCII bar chart"""
        if not data:
            return "No data to chart"
        
        # Get first two columns as x,y
        keys = list(data[0].keys())
        if len(keys) < 2:
            return self._generate_table(data)
        
        x_key, y_key = keys[0], keys[1]
        
        # Generate ASCII bars
        chart_lines = [f"📊 **{y_key.replace('_', ' ').title()} by {x_key.replace('_', ' ').title()}**\n"]
        
        max_value = max(row[y_key] for row in data if isinstance(row[y_key], (int, float)))
        max_label_len = max(len(str(row[x_key])) for row in data)
        
        for row in data[:10]:  # Limit to 10 bars
            label = str(row[x_key]).ljust(max_label_len)
            value = row[y_key]
            
            if isinstance(value, (int, float)):
                bar_length = max(1, int((value / max_value) * 30))
                bar = "█" * bar_length
                chart_lines.append(f"`{label}` {bar} ({value})")
            else:
                chart_lines.append(f"`{label}` {value}")
        
        return "\n".join(chart_lines) + "\n\n"
    
    def _generate_pie_chart(self, data: List[Dict]) -> str:
        """Generate ASCII pie chart representation"""
        if not data:
            return "No data to chart"
        
        keys = list(data[0].keys())
        if len(keys) < 2:
            return self._generate_table(data)
        
        label_key, value_key = keys[0], keys[1]
        
        # Calculate percentages
        total = sum(row[value_key] for row in data if isinstance(row[value_key], (int, float)))
        
        chart_lines = [f"🥧 **{label_key.replace('_', ' ').title()} Distribution**\n"]
        
        pie_chars = ["●", "◐", "◑", "◒", "◓", "○"]
        
        for i, row in enumerate(data[:6]):  # Limit to 6 slices
            label = row[label_key]
            value = row[value_key]
            
            if isinstance(value, (int, float)) and total > 0:
                percentage = (value / total) * 100
                char = pie_chars[i % len(pie_chars)]
                chart_lines.append(f"{char} `{label}`: {value} ({percentage:.1f}%)")
            else:
                char = pie_chars[i % len(pie_chars)]
                chart_lines.append(f"{char} `{label}`: {value}")
        
        return "\n".join(chart_lines) + "\n\n"
    
    def _generate_line_chart(self, data: List[Dict]) -> str:
        """Generate ASCII line chart for time series"""
        if not data:
            return "No data to chart"
        
        keys = list(data[0].keys())
        if len(keys) < 2:
            return self._generate_table(data)
        
        x_key, y_key = keys[0], keys[1]
        
        chart_lines = [f"📈 **{y_key.replace('_', ' ').title()} Trend**\n"]
        
        # Simple trend visualization
        values = [row[y_key] for row in data[-7:] if isinstance(row[y_key], (int, float))]  # Last 7 points
        
        if values:
            max_val = max(values)
            min_val = min(values)
            
            for i, row in enumerate(data[-7:]):
                label = str(row[x_key])
                value = row[y_key]
                
                if isinstance(value, (int, float)) and max_val > min_val:
                    # Normalize to 0-20 range
                    normalized = int(((value - min_val) / (max_val - min_val)) * 20)
                    sparkline = "▁▂▃▄▅▆▇█"[min(normalized // 3, 7)]
                    chart_lines.append(f"`{label}` {sparkline} {value}")
                else:
                    chart_lines.append(f"`{label}` {value}")
        
        return "\n".join(chart_lines) + "\n\n"
    
    def _generate_table(self, data: List[Dict]) -> str:
        """Generate markdown table from query results"""
        if not data:
            return "No data to display"
        
        # Get headers
        headers = list(data[0].keys())
        
        # Generate table
        table_lines = []
        
        # Header row
        header_row = "| " + " | ".join(headers) + " |"
        table_lines.append(header_row)
        
        # Separator row
        separator = "| " + " | ".join(["---"] * len(headers)) + " |"
        table_lines.append(separator)
        
        # Data rows (limit to 10 for display)
        for row in data[:10]:
            values = [str(row.get(h, '')) for h in headers]
            data_row = "| " + " | ".join(values) + " |"
            table_lines.append(data_row)
        
        if len(data) > 10:
            table_lines.append(f"\n*... and {len(data) - 10} more rows*")
        
        return "\n".join(table_lines) + "\n"
    
    def _generate_simple_list(self, data: List) -> str:
        """Generate simple list from results"""
        return "\n".join([f"- {item}" for item in data[:20]]) + "\n"
    
    async def toggle_dashboard_mode(self) -> None:
        """Toggle dashboard mode with auto-run queries"""
        self.dashboard_mode = not self.dashboard_mode
        if self.dashboard_mode:
            await self._setup_dashboard_template()
        else:
            await self._setup_edit_template()
    
    async def _setup_dashboard_template(self) -> None:
        """Setup dashboard template in canvas with auto-run queries"""
        dashboard_content = """# 📊 Globule Analytics Dashboard

**Dashboard Mode: ACTIVE** | Use Ctrl+D to toggle back to edit mode

## 📈 Live Analytics

Auto-running queries based on detected schema...

---

"""
        self.text = dashboard_content
        
        # Post message to trigger auto-run of dashboard queries
        self.post_message(self.DashboardModeActivated())
    
    async def _setup_edit_template(self) -> None:
        """Setup edit template in canvas"""
        edit_content = """# 📝 Globule Canvas

**Edit Mode: ACTIVE** | Use Ctrl+D to toggle to dashboard mode

Use this space to compose your thoughts and analysis. Query results from the palette will appear here.

## Instructions

- Execute queries from the palette (left pane)
- Use Ctrl+E to expand text with AI
- Use Ctrl+R or Ctrl+U to summarize with AI
- Drag query results from palette to canvas
- Use Ctrl+S to save as Markdown

---

"""
        self.text = edit_content
    
    class DashboardModeActivated(Message):
        """Message sent when dashboard mode is activated"""
        pass


class DashboardApp(App):
    """Enhanced Globule TUI with analytics dashboard capabilities"""
    
    CSS = """
    .palette-header {
        color: $accent;
        text-style: bold;
        margin-bottom: 1;
    }
    
    .schema-info {
        color: $secondary;
        margin-bottom: 1;
        text-style: italic;
    }
    
    .section-header {
        color: $primary;
        text-style: bold;
        margin-top: 1;
        margin-bottom: 1;
    }
    
    #palette {
        width: 30%;
        border-right: solid $primary;
        padding: 1;
    }
    
    #canvas {
        width: 70%;
        padding: 1;
    }
    
    Tree {
        margin-bottom: 1;
    }
    
    Input {
        margin-bottom: 1;
    }
    """
    
    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("q", "quit", "Quit"),
        ("tab", "switch_focus", "Switch Pane"),
        ("enter", "execute_action", "Execute"),
        ("ctrl+s", "save_draft", "Save MD"),
        ("ctrl+d", "toggle_dashboard", "Dashboard"),
        ("ctrl+e", "expand_text", "AI Expand"),
        ("ctrl+r", "summarize_text", "AI Summarize"),
        ("ctrl+u", "summarize_text", "AI Summarize"),
    ]
    
    def __init__(self, storage_manager: StorageManager, topic: Optional[str] = None):
        super().__init__()
        self.storage_manager = storage_manager
        self.topic = topic
        self.synthesis_state = SynthesisState()
        
    def compose(self) -> ComposeResult:
        """Compose two-pane layout with analytics palette and viz canvas"""
        yield Header()
        
        with Horizontal():
            # Left pane: Analytics Palette (30%)
            with Vertical(id="palette"):
                yield AnalyticsPalette(
                    self.storage_manager, 
                    self.topic, 
                    id="analytics-palette"
                )
            
            # Right pane: Visualization Canvas (70%)
            with Vertical(id="canvas"):
                initial_content = f"# Globule Dashboard\n\n**Topic:** {self.topic or 'General'}\n\n"
                initial_content += "## Instructions\n\n"
                initial_content += "- Use the query explorer (left) to run analytics\n"
                initial_content += "- Results will appear here as tables and charts\n"
                initial_content += "- Press Tab to switch between panes\n"
                initial_content += "- Press Ctrl+D to toggle dashboard mode\n"
                initial_content += "- Press Ctrl+S to save as Markdown\n\n"
                initial_content += "## Analysis Results\n\n*Query results will appear here...*\n\n"
                
                yield VizCanvas(initial_content, id="viz-canvas")
        
        yield Footer()
    
    async def on_mount(self) -> None:
        """Initialize the dashboard on mount"""
        self.title = "Globule Analytics Dashboard"
        self.sub_title = f"Topic: {self.topic}" if self.topic else "Data Explorer"
        
        # Load recent globules and clusters
        try:
            # Get recent globules
            recent_globules = await self.storage_manager.get_recent_globules(100)
            
            # Filter by schema if valet topic detected
            palette = self.query_one("#analytics-palette", AnalyticsPalette)
            if palette.detected_schema in ['valet', 'valet_enhanced']:
                # Filter for valet-related globules
                valet_globules = [
                    g for g in recent_globules 
                    if g.parsed_data and ('valet' in json.dumps(g.parsed_data).lower())
                ]
                
                # Auto-execute valet analytics query
                if valet_globules:
                    await palette.execute_query(
                        "SELECT COUNT(*) as total_cars, AVG(julianday('now') - julianday(created_at)) as avg_days_ago FROM globules WHERE parsed_data LIKE '%valet%'"
                    )
            
            # Try to load clusters
            try:
                clustering_engine = SemanticClusteringEngine(self.storage_manager)
                analysis = await clustering_engine.analyze_semantic_clusters(min_globules=2)
                
                if analysis.clusters:
                    # Convert to GlobuleCluster format (simplified)
                    clusters = []
                    for cluster in analysis.clusters[:5]:  # Limit to 5 clusters
                        globule_cluster = GlobuleCluster(
                            id=cluster.id,
                            globules=[],  # We'll populate if needed
                            label=cluster.label,
                            metadata={
                                'confidence_score': cluster.confidence_score,
                                'size': cluster.size
                            }
                        )
                        clusters.append(globule_cluster)
                    
                    await palette.load_clusters(clusters)
            except Exception:
                # Ignore clustering errors
                pass
            
            # Focus on palette initially
            palette.focus()
            
        except Exception as e:
            # Show error but don't crash
            canvas = self.query_one("#viz-canvas", VizCanvas)
            canvas.text += f"\n⚠️ **Initialization Warning:** {e}\n"
    
    async def on_analytics_palette_query_executed(self, message: AnalyticsPalette.QueryExecuted) -> None:
        """Handle query execution from palette"""
        canvas = self.query_one("#viz-canvas", VizCanvas)
        canvas.add_query_result(message.query, message.result, message.query_type)
    
    async def on_analytics_palette_globule_selected(self, message: AnalyticsPalette.GlobuleSelected) -> None:
        """Handle globule selection from palette"""
        canvas = self.query_one("#viz-canvas", VizCanvas)
        # Add globule as text content
        canvas.text += f"\n\n## Selected Globule\n\n{message.globule.text}\n\n"
    
    async def on_analytics_palette_query_result_dragged(self, message: AnalyticsPalette.QueryResultDragged) -> None:
        """Handle query result dragged from palette to canvas"""
        canvas = self.query_one("#viz-canvas", VizCanvas)
        canvas.add_dragged_result(message.query_result)
        self.notify("🎯 Query result dragged to canvas!")
    
    async def on_viz_canvas_ai_action_requested(self, message: VizCanvas.AIActionRequested) -> None:
        """Handle AI action request from canvas"""
        try:
            # For now, simulate AI actions with placeholder responses
            if message.action == 'expand':
                expanded_text = f"\n\n### 🤖 AI Expansion:\n\n*[AI would expand: \"{message.text[:50]}...\" here]*\n\nThis text could be elaborated with additional context, examples, and detailed explanations. The AI system would analyze the content and provide relevant expansions based on the topic and context.\n\n"
                
                # Insert expansion at current cursor position
                canvas = self.query_one("#viz-canvas", VizCanvas)
                current_text = canvas.text
                cursor = canvas.cursor_position
                
                if cursor:
                    before = current_text[:cursor[1]]
                    after = current_text[cursor[1]:]
                    canvas.text = before + expanded_text + after
                else:
                    canvas.text += expanded_text
                    
                self.notify("🤖 AI Expansion completed!")
                
            elif message.action == 'summarize':
                summary_text = f"\n\n### 📝 AI Summary:\n\n*[AI would summarize: \"{message.text[:50]}...\" here]*\n\n• **Key Point 1**: Main concept identified\n• **Key Point 2**: Supporting details extracted\n• **Key Point 3**: Relevant conclusions drawn\n\n"
                
                # Insert summary at current cursor position
                canvas = self.query_one("#viz-canvas", VizCanvas)
                current_text = canvas.text
                cursor = canvas.cursor_position
                
                if cursor:
                    before = current_text[:cursor[1]]
                    after = current_text[cursor[1]:]
                    canvas.text = before + summary_text + after
                else:
                    canvas.text += summary_text
                    
                self.notify("📝 AI Summary completed!")
                
        except Exception as e:
            self.notify(f"❌ AI action error: {e}")
    
    def action_quit(self) -> None:
        """Quit the application"""
        self.exit()
    
    def action_switch_focus(self) -> None:
        """Switch focus between palette and canvas"""
        if self.focused and self.focused.id == "analytics-palette":
            canvas = self.query_one("#viz-canvas")
            canvas.focus()
        else:
            palette = self.query_one("#analytics-palette")
            palette.focus()
    
    def action_execute_action(self) -> None:
        """Execute action based on focused pane"""
        if self.focused and hasattr(self.focused, 'id'):
            if self.focused.id == "query-input":
                # Submit query
                query = self.focused.value.strip()
                if query:
                    # Get palette and execute query
                    palette = self.query_one("#analytics-palette", AnalyticsPalette)
                    asyncio.create_task(palette.execute_query(query))
    
    def action_save_draft(self) -> None:
        """Save canvas content as Markdown"""
        try:
            canvas = self.query_one("#viz-canvas", VizCanvas)
            content = canvas.text
            
            if not content.strip():
                self.notify("Nothing to save")
                return
            
            # Generate filename
            import os
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            topic_part = self.topic.replace(" ", "_") if self.topic else "dashboard"
            filename = f"globule_dashboard_{topic_part}_{timestamp}.md"
            
            # Save to drafts directory
            drafts_dir = "drafts"
            os.makedirs(drafts_dir, exist_ok=True)
            filepath = os.path.join(drafts_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                # Add schema metadata if available
                palette = self.query_one("#analytics-palette", AnalyticsPalette)
                if palette.detected_schema:
                    f.write(f"---\nschema: {palette.detected_schema}\ntopic: {self.topic}\ngenerated: {datetime.now().isoformat()}\n---\n\n")
                f.write(content)
            
            self.notify(f"✅ Saved to {filepath}")
            
        except Exception as e:
            self.notify(f"❌ Save error: {e}")
    
    async def action_toggle_dashboard(self) -> None:
        """Toggle dashboard mode"""
        try:
            canvas = self.query_one("#viz-canvas", VizCanvas)
            await canvas.toggle_dashboard_mode()
            
            if canvas.dashboard_mode:
                self.notify("📊 Dashboard Mode: ON - Auto-running queries...")
            else:
                self.notify("📝 Edit Mode: ON")
                
        except Exception as e:
            self.notify(f"❌ Toggle error: {e}")
    
    async def on_viz_canvas_dashboard_mode_activated(self, message: VizCanvas.DashboardModeActivated) -> None:
        """Handle dashboard mode activation - auto-run schema queries"""
        try:
            palette = self.query_one("#analytics-palette", AnalyticsPalette)
            canvas = self.query_one("#viz-canvas", VizCanvas)
            
            # Get schema-specific queries to auto-run
            schema_queries = palette._get_schema_queries()
            
            if schema_queries:
                # Run the first few key queries automatically
                key_queries = schema_queries[:3]  # Run first 3 queries
                
                for i, query in enumerate(key_queries):
                    try:
                        # Add a small delay between queries
                        if i > 0:
                            await asyncio.sleep(0.5)
                        
                        # Execute the query
                        await palette.execute_query(query['sql'])
                        
                        # Add progress update to canvas
                        canvas.text += f"🔄 Auto-executed: {query['name']}\n\n"
                        
                    except Exception as query_error:
                        canvas.text += f"❌ Failed to auto-run: {query['name']} - {query_error}\n\n"
                
                canvas.text += "\n## 📊 Dashboard Ready\n\nAll auto-queries completed. Results are shown above.\n\n"
                self.notify("✅ Dashboard auto-setup complete!")
            else:
                canvas.text += "\n⚠️ No schema queries available for auto-run.\n\n"
                self.notify("⚠️ No schema queries found for dashboard mode")
                
        except Exception as e:
            self.notify(f"❌ Dashboard auto-setup error: {e}")
    
    async def action_expand_text(self) -> None:
        """AI expand selected text"""
        try:
            canvas = self.query_one("#viz-canvas", VizCanvas)
            selected_text = canvas.get_selected_text()
            
            if selected_text and selected_text.strip():
                canvas.expand_text_ai(selected_text)
            else:
                self.notify("⚠️ No text selected or cursor on empty line")
        except Exception as e:
            self.notify(f"❌ AI expand error: {e}")
    
    async def action_summarize_text(self) -> None:
        """AI summarize selected text"""
        try:
            canvas = self.query_one("#viz-canvas", VizCanvas)
            selected_text = canvas.get_selected_text()
            
            if selected_text and selected_text.strip():
                canvas.summarize_text_ai(selected_text)
            else:
                self.notify("⚠️ No text selected or cursor on empty line")
        except Exception as e:
            self.notify(f"❌ AI summarize error: {e}")