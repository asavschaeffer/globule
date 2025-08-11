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
import io
import re
import smtplib
import os
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

try:
    import jinja2
    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import tweepy
    HAS_TWEEPY = True
except ImportError:
    HAS_TWEEPY = False

try:
    import pyperclip
    HAS_PYPERCLIP = True
except ImportError:
    HAS_PYPERCLIP = False

# WeasyPrint import moved to function level to avoid startup issues
HAS_WEASYPRINT = True

from globule.core.interfaces import StorageManager
from globule.core.models import ProcessedGlobule, SynthesisState, UIMode, GlobuleCluster
from globule.services.clustering.semantic_clustering import SemanticClusteringEngine
from globule.services.parsing.ollama_parser import OllamaParser
from globule.schemas.manager import get_schema_manager, detect_schema_for_text, detect_output_schema_for_topic, get_output_schema


class ItemSelected(Message):
    def __init__(self, content: str) -> None:
        self.content = content
        super().__init__()


class ThoughtPalette(Vertical):
    """Enhanced palette: Data/query explorer with analytics capabilities"""
    
    BINDINGS = [
        ("enter", "add_to_canvas", "Add Selection"),
        ("ctrl+s", "save_module", "Save Module to Schema"),
    ]
    
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
        self.output_schema_name = None
        self.output_schema = None
        self.clusters: List[GlobuleCluster] = []
        self.query_results: List[Dict[str, Any]] = []
        self.expanded_sections: Set[str] = set()
        self.can_focus = True
        self.drag_data: Optional[Dict[str, Any]] = None
        self.is_dragging = False
        self.parser = None  # Will be initialized for LLM queries
        
        # Detect schemas from topic
        if topic:
            self.detected_schema = detect_schema_for_text(topic)
            self.output_schema_name = detect_output_schema_for_topic(topic)
            if self.output_schema_name:
                self.output_schema = get_output_schema(self.output_schema_name)
    
    def compose(self) -> ComposeResult:
        """Compose the unified ThoughtPalette with search and tree"""
        yield Input(placeholder="Search thoughts (e.g., car make: honda parked by valet:maria)", id="palette-search")
        yield Tree(label="Results", id="palette-tree")
        
        # OLD: Removed for unified view - Multiple sections below were commented out
        # OLD: yield Static("📊 ANALYTICS PALETTE", classes="palette-header")
        # OLD: Schema detection info section
        # OLD: yield Static("🔍 QUERY EXPLORER:", classes="section-header")
        # OLD: yield Input(..., id="query-input")
        # OLD: yield Static("📋 PRE-DEFINED QUERIES:", classes="section-header")
        # OLD: queries_tree creation and population
        # OLD: yield Static("📈 RESULTS:", classes="section-header")
        # OLD: yield Tree("Analytics", id="results-tree")
        # OLD: Clusters section with confidence bars
    
    def on_input_submitted(self, event: Input.Submitted) -> None:
        pass  # Will handle search later

    async def _orchestrate_search(self, nl_query: str):
        pass  # Will add AI logic later

    def action_add_to_canvas(self):
        pass  # Will add to canvas later

    def action_save_module(self):
        pass  # Will save later
    
    def _get_default_query(self) -> str:
        """Get default query based on detected schema"""
        if self.detected_schema == 'valet' or self.detected_schema == 'valet_enhanced':
            return "SELECT COUNT(*) as cars_today FROM globules WHERE date(created_at) = date('now') AND parsed_data LIKE '%valet%'"
        elif self.detected_schema == 'technical':
            return "SELECT category, COUNT(*) FROM globules WHERE parsed_data LIKE '%technical%' GROUP BY category"
        else:
            return "SELECT COUNT(*) as total_thoughts FROM globules WHERE created_at >= date('now', '-7 days')"
    
    def _get_schema_queries(self) -> List[Dict[str, str]]:
        """Get pre-defined queries from output schema or fallback to input schema"""
        queries = []
        
        # Priority 1: Output schema queries
        if self.output_schema:
            queries = self.output_schema.get('queries', [])
        
        # Priority 2: Input schema dashboard queries (fallback)
        elif self.detected_schema:
            schema = self.schema_manager.get_schema(self.detected_schema)
            if schema:
                queries = schema.get('dashboard_queries', [])
        
        return queries
    
    async def execute_query(self, query_dict: Dict[str, Any]) -> None:
        """Execute SQL or LLM query based on query type"""
        try:
            query_name = query_dict.get('name', 'Unknown Query')
            query_type = 'sql' if 'sql' in query_dict else 'llm'
            
            if query_type == 'sql':
                await self._execute_sql_query(query_dict)
            else:
                await self._execute_llm_query(query_dict)
                
        except Exception as e:
            # Add error result
            error_result = {
                'query': query_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'type': 'error'
            }
            self.query_results.append(error_result)
            await self._update_results_tree()
    
    async def _execute_sql_query(self, query_dict: Dict[str, Any]) -> None:
        """Execute SQL query"""
        query = query_dict['sql']
        query_name = query_dict.get('name', 'SQL Query')
        viz_type = query_dict.get('viz_type', 'table')
        
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
                'name': query_name,
                'query': query,
                'results': results,
                'timestamp': datetime.now().isoformat(),
                'type': 'sql',
                'viz_type': viz_type
            }
            self.query_results.append(query_result)
            
            # Update results tree
            await self._update_results_tree()
            
            # Post message for canvas to handle
            self.post_message(self.QueryExecuted(query, results, "sql"))
    
    async def _execute_llm_query(self, query_dict: Dict[str, Any]) -> None:
        """Execute LLM-based query"""
        llm_prompt = query_dict['llm_prompt']
        query_name = query_dict.get('name', 'LLM Query')
        viz_type = query_dict.get('viz_type', 'text')
        context_query = query_dict.get('context_query')
        
        # Initialize parser if not done
        if not self.parser:
            self.parser = OllamaParser()
        
        # Get context data if context_query provided
        context_data = []
        if context_query:
            if hasattr(self.storage_manager, 'db_path'):
                db_path = self.storage_manager.db_path
            else:
                import os
                db_path = os.path.expanduser("~/.globule/data/globule.db")
            
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(context_query)
                context_data = [dict(row) for row in cursor.fetchall()]
        
        # Build LLM prompt with context
        if context_data:
            context_text = "\n".join([
                f"Entry {i+1}: {row.get('text', '')} | Data: {row.get('parsed_data', '')}"
                for i, row in enumerate(context_data[:10])  # Limit context
            ])
            full_prompt = f"{llm_prompt}\n\nContext Data:\n{context_text}"
        else:
            full_prompt = llm_prompt
        
        # Execute LLM query
        try:
            llm_result = await self.parser.parse(full_prompt, {'action': 'analyze', 'output_format': 'text'})
            
            # Extract meaningful response
            analysis_text = llm_result.get('analysis', llm_result.get('summary', 'Analysis completed'))
            
            results = [{
                'analysis': analysis_text,
                'context_entries': len(context_data),
                'query_type': 'llm'
            }]
            
        except Exception as e:
            results = [{
                'analysis': f'LLM analysis failed: {str(e)}',
                'context_entries': len(context_data),
                'query_type': 'llm'
            }]
        
        # Add to query results
        query_result = {
            'name': query_name,
            'query': llm_prompt,
            'results': results,
            'timestamp': datetime.now().isoformat(),
            'type': 'llm',
            'viz_type': viz_type
        }
        self.query_results.append(query_result)
        
        # Update results tree
        await self._update_results_tree()
        
        # Post message for canvas to handle
        self.post_message(self.QueryExecuted(llm_prompt, results, "llm"))
    
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
                # Treat manual input as SQL query
                query_dict = {
                    'name': 'Manual Query',
                    'sql': query,
                    'viz_type': 'table'
                }
                await self.execute_query(query_dict)
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
                            # Execute the pre-defined query (SQL or LLM)
                            await self.execute_query(query)
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
    
    async def on_mouse_move(self, event: MouseMove) -> None:
        """Handle mouse move during drag operation"""
        if self.is_dragging and self.drag_data:
            # Get query name for overlay
            query_name = self.drag_data.get('name', 'Query Result')
            
            # Create drag overlay (simulated - textual doesn't support real overlay)
            # We'll show a notification instead
            app = self.app
            if hasattr(app, '_drag_notification_shown') and not app._drag_notification_shown:
                app.notify(f"🎯 Dragging: {query_name}")
                app._drag_notification_shown = True
    
    async def on_mouse_up(self, event: MouseUp) -> None:
        """Handle mouse up to complete drag operation"""
        if self.is_dragging and self.drag_data:
            # Reset drag notification flag
            app = self.app
            if hasattr(app, '_drag_notification_shown'):
                app._drag_notification_shown = False
            
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
    
    def __init__(self, content: str = "", output_schema: Dict[str, Any] = None, **kwargs):
        super().__init__(content, **kwargs)
        self.incorporated_results: List[Dict[str, Any]] = []
        self.dashboard_mode = False
        self.can_focus = True
        self._selected_text: Optional[str] = None
        self.output_schema = output_schema
        self.template = output_schema.get('template', '') if output_schema else ''
        self.query_data = {}  # Store query results for template substitution
        self.parser = None  # For AI actions
    
    def add_dragged_result(self, query_result: Dict[str, Any]) -> None:
        """Add dragged query result to canvas with smart placeholder matching"""
        query_name = query_result.get('name', 'Query Result')
        query = query_result.get('query', 'Unknown Query')
        result = query_result.get('results', [])
        query_type = query_result.get('type', 'sql')
        viz_type = query_result.get('viz_type', 'table')
        
        # Check for template placeholder matching
        current_text = self.text
        cursor = self.cursor_position
        
        # Look for matching placeholders in the template
        placeholder_patterns = [
            f"{{bar: {query_name.lower().replace(' ', '_')}}}",
            f"{{table: {query_name.lower().replace(' ', '_')}}}",
            f"{{line: {query_name.lower().replace(' ', '_')}}}",
            f"{{pie: {query_name.lower().replace(' ', '_')}}}",
            f"{{metrics: {query_name.lower().replace(' ', '_')}}}",
        ]
        
        matched_placeholder = None
        for pattern in placeholder_patterns:
            if pattern in current_text:
                matched_placeholder = pattern
                break
        
        if matched_placeholder and self.template:
            # Replace the placeholder with actual visualization
            viz_content = self._generate_visualization(query_name, result, query_type, viz_type)
            self.text = current_text.replace(matched_placeholder, viz_content)
            
            # Store data for future template processing
            self.query_data[query_name.lower().replace(' ', '_')] = {
                'results': result,
                'type': query_type,
                'viz_type': viz_type,
                'timestamp': datetime.now().isoformat()
            }
        else:
            # Standard insertion at cursor position
            viz_content = self._generate_visualization(query_name, result, query_type, viz_type)
            
            # Add header with drag indicator
            new_content = f"\n\n## 🎯 Dragged Query Result\n\n**Query:** `{query}`\n**Dragged at:** {datetime.now().strftime('%H:%M:%S')}\n\n{viz_content}\n"
            
            # Insert at cursor position
            before = current_text[:cursor[1] if cursor else len(current_text)]
            after = current_text[cursor[1] if cursor else len(current_text):]
            self.text = before + new_content + after
        
        # Track incorporation
        self.incorporated_results.append({
            'query': query,
            'name': query_name,
            'result': result,
            'type': query_type,
            'viz_type': viz_type,
            'timestamp': datetime.now().isoformat(),
            'method': 'drag_drop',
            'placeholder_matched': matched_placeholder is not None
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
    
    async def expand_text_ai(self, text: str) -> None:
        """Request AI expansion of text using real parser with schema context"""
        if not text:
            return
            
        try:
            # Initialize parser if needed
            if not self.parser:
                self.parser = OllamaParser()
            
            # Build context with schema and query data
            context = {
                'action': 'expand', 
                'output_format': 'markdown'
            }
            
            # Add schema context if available
            if self.output_schema:
                context['schema'] = self.output_schema
                context['schema_title'] = self.output_schema.get('title', 'Output Schema')
                
            # Add query data context
            if self.query_data:
                context['query_data'] = self.query_data
                context['available_queries'] = list(self.query_data.keys())
            
            # Enhanced prompt with schema awareness
            base_prompt = f"Expand and elaborate on this text with additional context, examples, and insights: {text}"
            
            if self.output_schema:
                schema_info = f"\nSchema Context: {self.output_schema.get('title', 'Output Schema')} - {self.output_schema.get('description', 'No description')}"
                base_prompt += schema_info
                
            if self.query_data:
                available_data = f"\nAvailable Data: {', '.join(self.query_data.keys())}"
                base_prompt += available_data
            
            # Use parser to expand the text
            expansion_result = await self.parser.parse(base_prompt, context)
            
            # Extract the expanded content
            expanded_content = expansion_result.get('title', '') + '\n\n' + expansion_result.get('reasoning', 'AI expansion completed.')
            
            # Post result
            self.post_message(self.AIActionRequested('expand', text, {
                'position': self.cursor_position,
                'result': expanded_content
            }))
            
        except Exception as e:
            self.post_message(self.AIActionRequested('expand', text, {
                'position': self.cursor_position,
                'error': str(e)
            }))
    
    async def summarize_text_ai(self, text: str) -> None:
        """Request AI summarization of text using real parser with schema context"""
        if not text:
            return
            
        try:
            # Initialize parser if needed
            if not self.parser:
                self.parser = OllamaParser()
            
            # Build context with schema and query data
            context = {
                'action': 'summarize', 
                'output_format': 'bullet_points'
            }
            
            # Add schema context if available
            if self.output_schema:
                context['schema'] = self.output_schema
                context['schema_title'] = self.output_schema.get('title', 'Output Schema')
                
            # Add query data context
            if self.query_data:
                context['query_data'] = self.query_data
                context['available_queries'] = list(self.query_data.keys())
            
            # Enhanced prompt with schema awareness
            base_prompt = f"Summarize this text into key points and main ideas: {text}"
            
            if self.output_schema:
                schema_info = f"\nSchema Context: {self.output_schema.get('title', 'Output Schema')} - Focus on elements relevant to this schema"
                base_prompt += schema_info
                
            if self.query_data:
                data_info = f"\nData Context: Consider the available analytics data: {', '.join(self.query_data.keys())}"
                base_prompt += data_info
            
            # Use parser to summarize the text
            summary_result = await self.parser.parse(base_prompt, context)
            
            # Extract the summary content
            summary_content = summary_result.get('title', 'Summary') + '\n\n' + summary_result.get('reasoning', 'AI summarization completed.')
            
            # Post result
            self.post_message(self.AIActionRequested('summarize', text, {
                'position': self.cursor_position,
                'result': summary_content
            }))
            
        except Exception as e:
            self.post_message(self.AIActionRequested('summarize', text, {
                'position': self.cursor_position,
                'error': str(e)
            }))
    
    def add_query_result(self, query_name: str, result: Any, query_type: str = "sql", viz_type: str = "table") -> None:
        """Add query result to canvas with visualization and template integration"""
        # Store data for template substitution
        self.query_data[query_name] = {
            'results': result,
            'type': query_type,
            'viz_type': viz_type,
            'timestamp': datetime.now().isoformat()
        }
        
        # Generate visualization based on result type
        viz_content = self._generate_visualization(query_name, result, query_type, viz_type)
        
        # Check if we should auto-fill template
        if self.template and self.dashboard_mode:
            self._update_template_content()
        else:
            # Add to current cursor position (traditional mode)
            current_content = self.text
            cursor_pos = len(current_content)  # Append at end for now
            
            new_content = current_content + f"\n\n## Query Result: {query_name}\n\n{viz_content}\n"
            self.text = new_content
        
        # Track incorporation
        self.incorporated_results.append({
            'name': query_name,
            'result': result,
            'type': query_type,
            'viz_type': viz_type,
            'timestamp': datetime.now().isoformat()
        })
    
    def _update_template_content(self) -> None:
        """Update canvas content using template with current query data and jinja2 processing"""
        if not self.template:
            return
            
        content = self.template
        
        # Try jinja2 processing first if available
        if HAS_JINJA2:
            content = self._process_jinja2_template(content)
        else:
            # Fallback to simple template processing
            content = self._process_simple_template(content)
        
        self.text = content
    
    def _process_jinja2_template(self, template_content: str) -> str:
        """Process template using jinja2 for advanced substitutions"""
        try:
            # Create jinja2 environment
            env = jinja2.Environment(loader=jinja2.BaseLoader(), autoescape=False)
            template = env.from_string(template_content)
            
            # Build template context
            context = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'date': datetime.now().strftime('%Y-%m-%d'),
                'time': datetime.now().strftime('%H:%M:%S'),
                'year': datetime.now().year,
                'month': datetime.now().month,
                'day': datetime.now().day,
                'query_data': self.query_data,
                'schema': self.output_schema,
            }
            
            # Add each query's results as top-level variables
            for query_name, query_info in self.query_data.items():
                context[query_name] = query_info['results']
            
            # Add helper functions
            context['generate_viz'] = lambda query_name, viz_type='table': self._generate_visualization_for_template(query_name, viz_type)
            
            # Render template
            return template.render(**context)
            
        except Exception as e:
            # Fallback to simple processing if jinja2 fails
            return self._process_simple_template(template_content) + f"\n\n*Jinja2 processing error: {e}*"
    
    def _process_simple_template(self, template_content: str) -> str:
        """Fallback simple template processing"""
        content = template_content
        
        # Replace timestamp placeholders
        content = content.replace('{timestamp}', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        content = content.replace('{date}', datetime.now().strftime('%Y-%m-%d'))
        
        # Process template placeholders
        placeholder_pattern = r'\{(metrics|bar|table|line|pie|text|list|tweets): ([^}]+)\}'
        
        def replace_placeholder(match):
            viz_type = match.group(1)
            query_name = match.group(2)
            
            if query_name in self.query_data:
                query_data = self.query_data[query_name]
                return self._generate_visualization(query_name, query_data['results'], query_data['type'], viz_type)
            else:
                return f"*[Query '{query_name}' not executed yet]*"
        
        content = re.sub(placeholder_pattern, replace_placeholder, content)
        
        # Handle direct data access like {avg_cars.avg_cars}
        data_pattern = r'\{([^.}]+)\.([^}]+)\}'
        
        def replace_data_access(match):
            query_name = match.group(1)
            field_name = match.group(2)
            
            if query_name in self.query_data:
                results = self.query_data[query_name]['results']
                if results and len(results) > 0:
                    first_result = results[0]
                    if isinstance(first_result, dict) and field_name in first_result:
                        return str(first_result[field_name])
            
            return f"*[{query_name}.{field_name} not available]*"
        
        content = re.sub(data_pattern, replace_data_access, content)
        
        # Handle array access like {top_valets.0.valet_name}
        array_pattern = r'\{([^.}]+)\.(\d+)\.([^}]+)\}'
        
        def replace_array_access(match):
            query_name = match.group(1)
            index = int(match.group(2))
            field_name = match.group(3)
            
            if query_name in self.query_data:
                results = self.query_data[query_name]['results']
                if results and len(results) > index:
                    result_item = results[index]
                    if isinstance(result_item, dict) and field_name in result_item:
                        return str(result_item[field_name])
            
            return f"*[{query_name}.{index}.{field_name} not available]*"
        
        content = re.sub(array_pattern, replace_array_access, content)
        
        return content
    
    def _generate_visualization_for_template(self, query_name: str, viz_type: str = 'table') -> str:
        """Generate visualization for jinja2 template usage"""
        if query_name not in self.query_data:
            return f"*[Query '{query_name}' not executed yet]*"
        
        query_info = self.query_data[query_name]
        return self._generate_visualization(query_name, query_info['results'], query_info['type'], viz_type)
    
    def _generate_visualization(self, query_name: str, result: Any, query_type: str, viz_type: str = None) -> str:
        """Generate enhanced visualization content with matplotlib support"""
        if query_type == "error":
            return f"❌ **Error:** {result}\n"
        
        if not result or len(result) == 0:
            return "📊 **No data found**\n"
        
        try:
            # Use provided viz_type or auto-detect
            if not viz_type:
                viz_type = self._detect_visualization_type(query_name, result)
            
            if isinstance(result, list) and len(result) > 0:
                # Try matplotlib first if available and appropriate
                if HAS_MATPLOTLIB and viz_type in ['bar', 'pie', 'line'] and isinstance(result[0], dict):
                    matplotlib_viz = self._generate_matplotlib_visualization(result, viz_type, query_name)
                    if matplotlib_viz:
                        return matplotlib_viz
                
                # Fallback to ASCII/text
                if isinstance(result[0], dict):
                    return self._generate_enhanced_visualization(result, viz_type, query_name)
                else:
                    return self._generate_simple_list(result)
            
            return f"📊 **Result:** {result}\n"
            
        except Exception as e:
            return f"❌ **Visualization error:** {e}\n"
    
    def _generate_matplotlib_visualization(self, data: List[Dict], viz_type: str, title: str) -> Optional[str]:
        """Generate matplotlib chart and return as base64 embedded image"""
        if not HAS_MATPLOTLIB or not data:
            return None
            
        try:
            fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
            
            if viz_type == 'bar' and len(data[0]) >= 2:
                # Bar chart
                keys = list(data[0].keys())
                x_key, y_key = keys[0], keys[1]
                
                x_data = [str(row[x_key]) for row in data[:10]]
                y_data = [row[y_key] for row in data[:10] if isinstance(row[y_key], (int, float))]
                
                if len(x_data) == len(y_data):
                    ax.bar(x_data, y_data)
                    ax.set_title(f"{title} - {y_key.replace('_', ' ').title()}")
                    ax.set_xlabel(x_key.replace('_', ' ').title())
                    ax.set_ylabel(y_key.replace('_', ' ').title())
                    plt.xticks(rotation=45, ha='right')
                
            elif viz_type == 'pie' and len(data[0]) >= 2:
                # Pie chart
                keys = list(data[0].keys())
                label_key, value_key = keys[0], keys[1]
                
                labels = [str(row[label_key]) for row in data[:6]]
                values = [row[value_key] for row in data[:6] if isinstance(row[value_key], (int, float))]
                
                if len(labels) == len(values) and all(v > 0 for v in values):
                    ax.pie(values, labels=labels, autopct='%1.1f%%')
                    ax.set_title(f"{title} - Distribution")
                
            elif viz_type == 'line' and len(data[0]) >= 2:
                # Line chart with date sorting support
                keys = list(data[0].keys())
                x_key, y_key = keys[0], keys[1]
                
                # Try to sort by date if x_key looks like date/time
                sorted_data = data[:15]  # Limit to 15 points for clarity
                if any(date_word in x_key.lower() for date_word in ['date', 'time', 'created', 'day']):
                    try:
                        # Attempt to sort by date
                        from datetime import datetime
                        sorted_data = sorted(sorted_data, key=lambda row: 
                            datetime.fromisoformat(str(row[x_key]).replace('Z', '+00:00')) 
                            if 'T' in str(row[x_key]) 
                            else datetime.strptime(str(row[x_key])[:10], '%Y-%m-%d')
                        )
                    except (ValueError, KeyError):
                        pass  # Use original order if date parsing fails
                
                x_data = range(len(sorted_data))
                y_data = [row[y_key] for row in sorted_data if isinstance(row[y_key], (int, float))]
                x_labels = [str(row[x_key])[:10] if 'date' in x_key.lower() else str(row[x_key]) for row in sorted_data]
                
                if len(y_data) > 1:
                    ax.plot(x_data[:len(y_data)], y_data, marker='o', linewidth=2, markersize=6)
                    ax.set_title(f"{title} - Trend")
                    ax.set_ylabel(y_key.replace('_', ' ').title())
                    ax.set_xlabel(x_key.replace('_', ' ').title())
                    ax.set_xticks(x_data[:len(x_labels)])
                    ax.set_xticklabels(x_labels, rotation=45, ha='right')
                    ax.grid(True, alpha=0.3)
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buffer, format='png', dpi=200, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode()
            plt.close(fig)
            
            return f"\n![{title} Chart](data:image/png;base64,{image_base64})\n\n*Generated with matplotlib*\n"
            
        except Exception as e:
            plt.close('all')  # Clean up any open figures
            return None
    
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
        """Toggle dashboard mode - unified interface now auto-detects schema"""
        # In unified mode, always enable dashboard if schema is available
        if self.output_schema:
            self.dashboard_mode = True
            self._update_template_content()
        else:
            self.dashboard_mode = not self.dashboard_mode
            if self.dashboard_mode:
                await self._setup_dashboard_template()
            else:
                await self._setup_edit_template()
    
    async def export_content(self, export_type: str) -> None:
        """Export canvas content using schema export configurations"""
        if not self.output_schema:
            self.post_message(self.ExportCompleted("error", "No output schema available for export"))
            return
            
        exports = self.output_schema.get('exports', [])
        export_config = None
        
        # Find the export configuration
        for exp in exports:
            if exp.get('type') == export_type:
                export_config = exp
                break
        
        if not export_config:
            self.post_message(self.ExportCompleted("error", f"No export configuration found for {export_type}"))
            return
        
        try:
            if export_type == "x_post":
                await self._export_to_twitter(export_config)
            elif export_type == "email":
                await self._export_to_email(export_config)
            elif export_type == "print":
                await self._export_to_pdf(export_config)
            elif export_type == "copy":
                await self._export_to_clipboard(export_config)
            else:
                self.post_message(self.ExportCompleted("error", f"Unsupported export type: {export_type}"))
        
        except Exception as e:
            self.post_message(self.ExportCompleted("error", f"Export failed: {str(e)}"))
    
    async def _export_to_twitter(self, export_config: Dict[str, Any]) -> None:
        """Export to Twitter/X using tweepy"""
        template = export_config.get('template', '')
        
        # Process template with current data
        processed_content = self._process_export_template(template)
        
        if not HAS_TWEEPY:
            self.post_message(self.ExportCompleted("error", "tweepy not installed. Install with: pip install tweepy"))
            return
        
        try:
            # Get API credentials from export config or environment
            api_key = export_config.get('api_key') or os.getenv('TWITTER_API_KEY')
            api_secret = export_config.get('api_secret') or os.getenv('TWITTER_API_SECRET')
            access_token = export_config.get('access_token') or os.getenv('TWITTER_ACCESS_TOKEN')
            access_token_secret = export_config.get('access_token_secret') or os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
            
            if not all([api_key, api_secret, access_token, access_token_secret]):
                self.post_message(self.ExportCompleted("error", "Twitter API credentials not configured in schema or environment"))
                return
            
            # Authenticate to Twitter
            auth = tweepy.OAuthHandler(api_key, api_secret)
            auth.set_access_token(access_token, access_token_secret)
            api = tweepy.API(auth)
            
            # Truncate to Twitter's character limit
            if len(processed_content) > 280:
                processed_content = processed_content[:277] + "..."
            
            # Post tweet
            tweet = api.update_status(processed_content)
            
            self.post_message(self.ExportCompleted("success", f"Tweet posted successfully: {tweet.id}"))
            
        except Exception as e:
            self.post_message(self.ExportCompleted("error", f"Twitter export failed: {str(e)}"))
    
    async def _export_to_email(self, export_config: Dict[str, Any]) -> None:
        """Export via email using smtplib"""
        template = export_config.get('template', '')
        subject = export_config.get('subject', 'Globule Export')
        
        processed_content = self._process_export_template(template)
        
        try:
            # Get email configuration
            smtp_server = export_config.get('smtp_server') or os.getenv('SMTP_SERVER', 'smtp.gmail.com')
            smtp_port = export_config.get('smtp_port', 587)
            email = export_config.get('email') or os.getenv('EMAIL')
            password = export_config.get('password') or os.getenv('EMAIL_PASSWORD')
            to_email = export_config.get('to_email') or email
            
            if not all([email, password, to_email]):
                self.post_message(self.ExportCompleted("error", "Email credentials not configured in schema or environment"))
                return
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = email
            msg['To'] = to_email
            msg['Subject'] = subject
            
            # Add body
            msg.attach(MIMEText(processed_content, 'plain'))
            
            # Connect and send
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(email, password)
            text = msg.as_string()
            server.sendmail(email, to_email, text)
            server.quit()
            
            self.post_message(self.ExportCompleted("success", f"Email sent successfully to {to_email}"))
            
        except Exception as e:
            self.post_message(self.ExportCompleted("error", f"Email export failed: {str(e)}"))
    
    async def _export_to_pdf(self, export_config: Dict[str, Any]) -> None:
        """Export to PDF using weasyprint"""
        template = export_config.get('template', '{full_template}')
        
        if template == '{full_template}':
            content = self.text
        else:
            content = self._process_export_template(template)
        
        try:
            # Try to import weasyprint at runtime
            import weasyprint
        except (ImportError, OSError) as e:
            self.post_message(self.ExportCompleted("error", f"weasyprint not available. Install with: pip install weasyprint. Error: {e}"))
            return
        
        try:
            # Create filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = export_config.get('filename', f"globule_export_{timestamp}.pdf")
            
            # Ensure output directory exists
            output_dir = export_config.get('output_dir', 'exports')
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, filename)
            
            # Convert markdown to HTML (basic conversion)
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                    h1, h2, h3 {{ color: #333; }}
                    pre {{ background: #f4f4f4; padding: 10px; border-radius: 5px; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <pre>{content}</pre>
            </body>
            </html>
            """
            
            # Generate PDF
            weasyprint.HTML(string=html_content).write_pdf(filepath)
            
            self.post_message(self.ExportCompleted("success", f"PDF created: {filepath}"))
            
        except Exception as e:
            self.post_message(self.ExportCompleted("error", f"PDF export failed: {str(e)}"))
    
    async def _export_to_clipboard(self, export_config: Dict[str, Any]) -> None:
        """Export to clipboard using pyperclip"""
        template = export_config.get('template', '')
        
        if not template:
            processed_content = self.text
        else:
            processed_content = self._process_export_template(template)
        
        if not HAS_PYPERCLIP:
            self.post_message(self.ExportCompleted("error", "pyperclip not installed. Install with: pip install pyperclip"))
            return
        
        try:
            # Copy to clipboard
            pyperclip.copy(processed_content)
            
            # Get clipboard content length for confirmation
            content_length = len(processed_content)
            self.post_message(self.ExportCompleted("success", f"Content copied to clipboard ({content_length} characters)"))
            
        except Exception as e:
            self.post_message(self.ExportCompleted("error", f"Clipboard export failed: {str(e)}"))
    
    def _process_export_template(self, template: str) -> str:
        """Process export template with current query data using jinja2 or fallback"""
        if not template:
            return self.text
        
        # Try jinja2 processing first if available
        if HAS_JINJA2:
            return self._process_jinja2_export_template(template)
        else:
            # Fallback to simple processing
            return self._process_simple_export_template(template)
    
    def _process_jinja2_export_template(self, template: str) -> str:
        """Process export template using jinja2"""
        try:
            # Create jinja2 environment
            env = jinja2.Environment(loader=jinja2.BaseLoader(), autoescape=False)
            template_obj = env.from_string(template)
            
            # Build template context
            context = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'date': datetime.now().strftime('%Y-%m-%d'),
                'time': datetime.now().strftime('%H:%M:%S'),
                'year': datetime.now().year,
                'month': datetime.now().month,
                'day': datetime.now().day,
                'query_data': self.query_data,
                'schema': self.output_schema,
                'full_content': self.text,
            }
            
            # Add each query's results as top-level variables
            for query_name, query_info in self.query_data.items():
                context[query_name] = query_info['results']
            
            # Add helper functions
            context['generate_viz'] = lambda query_name, viz_type='table': self._generate_visualization_for_template(query_name, viz_type)
            context['get_summary'] = lambda data, field: self._get_summary_for_export(data, field)
            
            # Render template
            return template_obj.render(**context)
            
        except Exception as e:
            # Fallback to simple processing if jinja2 fails
            return self._process_simple_export_template(template) + f"\n\n*Jinja2 processing error: {e}*"
    
    def _process_simple_export_template(self, template: str) -> str:
        """Fallback simple export template processing"""
        content = template
        
        # Use the same template processing logic as _update_template_content
        content = content.replace('{timestamp}', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        content = content.replace('{date}', datetime.now().strftime('%Y-%m-%d'))
        content = content.replace('{full_content}', self.text)
        
        # Process data placeholders
        for query_name, query_data in self.query_data.items():
            results = query_data['results']
            if results and len(results) > 0:
                first_result = results[0]
                if isinstance(first_result, dict):
                    for field_name, value in first_result.items():
                        placeholder = f"{{{query_name}.{field_name}}}"
                        content = content.replace(placeholder, str(value))
        
        return content
    
    def _get_summary_for_export(self, data: List[Dict], field: str) -> str:
        """Helper function for export templates to get summary statistics"""
        if not data or not isinstance(data, list):
            return "No data"
        
        try:
            values = [row.get(field) for row in data if isinstance(row, dict) and field in row]
            numeric_values = [v for v in values if isinstance(v, (int, float))]
            
            if numeric_values:
                total = sum(numeric_values)
                avg = total / len(numeric_values)
                return f"Total: {total}, Average: {avg:.1f}, Count: {len(numeric_values)}"
            else:
                return f"Count: {len(values)}"
        except Exception:
            return "Summary unavailable"
    
    class ExportCompleted(Message):
        """Message sent when export is completed"""
        def __init__(self, status: str, message: str) -> None:
            self.status = status  # 'success' or 'error'
            self.message = message
            super().__init__()
    
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
        ("ctrl+e", "expand_text", "AI Expand"),
        ("ctrl+r", "summarize_text", "AI Summarize"),
        ("ctrl+u", "summarize_text", "AI Summarize"),
        ("ctrl+x", "export_content", "Export"),
        ("ctrl+t", "refresh_template", "Refresh"),
    ]
    
    def __init__(self, storage_manager: StorageManager, topic: Optional[str] = None):
        super().__init__()
        self.storage_manager = storage_manager
        self.topic = topic
        self.synthesis_state = SynthesisState()
        self.output_schema_name = None
        self.output_schema = None
        
        # Detect output schema for topic
        if topic:
            self.output_schema_name = detect_output_schema_for_topic(topic)
            if self.output_schema_name:
                self.output_schema = get_output_schema(self.output_schema_name)
        
    def compose(self) -> ComposeResult:
        """Compose two-pane layout with analytics palette and viz canvas"""
        yield Header()
        
        with Horizontal():
            # Left pane: Thought Palette (30%)
            yield ThoughtPalette(
                self.storage_manager, 
                self.topic, 
                id="thought-palette"
            )
            
            # Right pane: Visualization Canvas (70%)
            with Vertical(id="canvas"):
                # Determine initial content based on output schema - unified approach
                if self.output_schema:
                    # Schema-driven mode: Load template directly
                    initial_content = self.output_schema.get('template', f"# {self.output_schema.get('title', 'Output Dashboard')}\n\n*Template loading...*")
                else:
                    # Free exploration mode
                    initial_content = f"# 📊 Globule Analytics\n\n**Topic:** {self.topic or 'General'}\n\n"
                    initial_content += "## 🔍 Exploration Mode\n\n"
                    initial_content += "- Use the query explorer (left) to run analytics\n"
                    initial_content += "- Results will appear here as tables and charts\n"
                    initial_content += "- Drag query results to canvas\n"
                    initial_content += "- Press Ctrl+S to save as Markdown\n"
                    initial_content += "- Press Ctrl+X for export options\n\n"
                    initial_content += "## Analysis Results\n\n*Query results will appear here as you execute them...*\n\n"
                
                yield VizCanvas(initial_content, self.output_schema, id="viz-canvas")
        
        yield Footer()
    
    async def on_mount(self) -> None:
        """Initialize the dashboard on mount with enhanced schema support"""
        self.title = "Globule Analytics Dashboard"
        self.sub_title = f"Topic: {self.topic}" if self.topic else "Data Explorer"
        
        # Load recent globules and clusters
        try:
            palette = self.query_one("#thought-palette", ThoughtPalette)
            canvas = self.query_one("#viz-canvas", VizCanvas)
            
            # If output schema is detected, automatically set up dashboard mode
            if self.output_schema:
                canvas.dashboard_mode = True
                canvas._update_template_content()  # Load template immediately
                
                # Auto-run the first few schema queries
                schema_queries = palette._get_schema_queries()
                if schema_queries:
                    self.notify(f"🚀 Auto-loading {len(schema_queries)} schema queries...")
                    
                    # Execute key queries automatically
                    for i, query in enumerate(schema_queries[:3]):  # Run first 3 queries
                        try:
                            await asyncio.sleep(0.2 * i)  # Stagger queries
                            await palette.execute_query(query)
                        except Exception as query_error:
                            self.notify(f"⚠️ Query failed: {query['name']}")
            
            # Get recent globules for context
            recent_globules = await self.storage_manager.get_recent_globules(100)
            
            # Filter by detected schema if applicable  
            if palette.detected_schema in ['valet', 'valet_enhanced']:
                # Filter for valet-related globules
                valet_globules = [
                    g for g in recent_globules 
                    if g.parsed_data and ('valet' in json.dumps(g.parsed_data).lower())
                ]
                
                # If no output schema but input schema detected, run basic query
                if not self.output_schema and valet_globules:
                    await palette.execute_query({
                        'name': 'Valet Overview',
                        'sql': "SELECT COUNT(*) as total_entries, AVG(julianday('now') - julianday(created_at)) as avg_days_ago FROM globules WHERE parsed_data LIKE '%valet%'",
                        'viz_type': 'metrics'
                    })
            
            # Try to load semantic clusters
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
                # Ignore clustering errors silently
                pass
            
            # Initial focus setup
            if self.output_schema:
                canvas.focus()  # Focus on canvas if we have a schema template
            else:
                palette.focus()  # Focus on palette for exploration
                
        except Exception as e:
            # Show error but don't crash
            canvas = self.query_one("#viz-canvas", VizCanvas)
            canvas.text += f"\n⚠️ **Initialization Warning:** {e}\n"
    
    async def on_thought_palette_query_executed(self, message: ThoughtPalette.QueryExecuted) -> None:
        """Handle query execution from palette"""
        canvas = self.query_one("#viz-canvas", VizCanvas)
        # Extract query name and viz type from palette query results
        palette = self.query_one("#thought-palette", ThoughtPalette)
        query_name = "Query Result"
        viz_type = "table"
        
        # Try to find the query details from the last query result
        if palette.query_results:
            last_result = palette.query_results[-1]
            query_name = last_result.get('name', 'Query Result')
            viz_type = last_result.get('viz_type', 'table')
        
        canvas.add_query_result(query_name, message.result, message.query_type, viz_type)
    
    async def on_thought_palette_globule_selected(self, message: ThoughtPalette.GlobuleSelected) -> None:
        """Handle globule selection from palette"""
        canvas = self.query_one("#viz-canvas", VizCanvas)
        # Add globule as text content
        canvas.text += f"\n\n## Selected Globule\n\n{message.globule.text}\n\n"
    
    async def on_thought_palette_query_result_dragged(self, message: ThoughtPalette.QueryResultDragged) -> None:
        """Handle query result dragged from palette to canvas"""
        canvas = self.query_one("#viz-canvas", VizCanvas)
        canvas.add_dragged_result(message.query_result)
        self.notify("🎯 Query result dragged to canvas!")
    
    async def on_viz_canvas_ai_action_requested(self, message: VizCanvas.AIActionRequested) -> None:
        """Handle AI action request from canvas with real results"""
        try:
            canvas = self.query_one("#viz-canvas", VizCanvas)
            
            if 'result' in message.context:
                # Real AI result received
                result_text = f"\n\n### {'🤖 AI Expansion' if message.action == 'expand' else '📝 AI Summary'}:\n\n{message.context['result']}\n\n"
                
                # Insert at cursor position
                current_text = canvas.text
                cursor = canvas.cursor_position or (0, len(current_text))
                
                before = current_text[:cursor[1]] if cursor else current_text
                after = current_text[cursor[1]:] if cursor else ""
                canvas.text = before + result_text + after
                
                self.notify(f"✅ AI {message.action} completed!")
                
            elif 'error' in message.context:
                # AI processing failed
                error_text = f"\n\n### ❌ AI {message.action.title()} Error:\n\n{message.context['error']}\n\n"
                
                current_text = canvas.text
                cursor = canvas.cursor_position or (0, len(current_text))
                
                before = current_text[:cursor[1]] if cursor else current_text
                after = current_text[cursor[1]:] if cursor else ""
                canvas.text = before + error_text + after
                
                self.notify(f"❌ AI {message.action} failed")
                
        except Exception as e:
            self.notify(f"❌ AI action error: {e}")
    
    async def on_viz_canvas_export_completed(self, message: VizCanvas.ExportCompleted) -> None:
        """Handle export completion from canvas"""
        if message.status == "success":
            self.notify(f"✅ Export: {message.message}")
        else:
            self.notify(f"❌ Export failed: {message.message}")
    
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
                    palette = self.query_one("#thought-palette", ThoughtPalette)
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
                palette = self.query_one("#thought-palette", ThoughtPalette)
                if palette.detected_schema:
                    f.write(f"---\nschema: {palette.detected_schema}\ntopic: {self.topic}\ngenerated: {datetime.now().isoformat()}\n---\n\n")
                f.write(content)
            
            self.notify(f"✅ Saved to {filepath}")
            
        except Exception as e:
            self.notify(f"❌ Save error: {e}")
    
    async def action_refresh_template(self) -> None:
        """Refresh template with current query data"""
        try:
            canvas = self.query_one("#viz-canvas", VizCanvas)
            
            if canvas.output_schema:
                canvas._update_template_content()
                self.notify("✅ Template refreshed with latest data")
            else:
                self.notify("⚠️ No output schema template available")
                
        except Exception as e:
            self.notify(f"❌ Refresh error: {e}")
    
    async def on_viz_canvas_dashboard_mode_activated(self, message: VizCanvas.DashboardModeActivated) -> None:
        """Handle dashboard mode activation - auto-run schema queries"""
        try:
            palette = self.query_one("#thought-palette", ThoughtPalette)
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
                        await palette.execute_query(query)
                        
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
                await canvas.expand_text_ai(selected_text)
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
                await canvas.summarize_text_ai(selected_text)
            else:
                self.notify("⚠️ No text selected or cursor on empty line")
        except Exception as e:
            self.notify(f"❌ AI summarize error: {e}")
    
    async def action_export_content(self) -> None:
        """Export canvas content using schema configurations"""
        try:
            canvas = self.query_one("#viz-canvas", VizCanvas)
            
            if not canvas.output_schema:
                self.notify("⚠️ No output schema available for export")
                return
            
            # Get available export types
            exports = canvas.output_schema.get('exports', [])
            
            if not exports:
                self.notify("⚠️ No export configurations found in schema")
                return
            
            # For now, just use the first export type
            # In a real implementation, we'd show a selection dialog
            export_type = exports[0].get('type', 'copy')
            
            await canvas.export_content(export_type)
            
        except Exception as e:
            self.notify(f"❌ Export error: {e}")