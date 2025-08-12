"""
Web Frontend for Globule using FastAPI.

This is a basic web interface that mirrors the TUI and CLI functionality
in a browser-accessible format. Currently a prototype/placeholder.
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime

try:
    from fastapi import FastAPI, HTTPException, Request, Form
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.templating import Jinja2Templates
    from fastapi.staticfiles import StaticFiles
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from globule.core.api import GlobuleAPI
from globule.storage.sqlite_manager import SQLiteStorageManager

logger = logging.getLogger(__name__)


def create_app() -> "FastAPI":
    """
    Create and configure the FastAPI web application.
    
    Returns:
        FastAPI app instance
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI not available. Install with: pip install fastapi uvicorn jinja2")
    
    app = FastAPI(
        title="Globule Web Interface",
        description="Web frontend for Globule thought management system",
        version="1.0.0"
    )
    
    # Initialize core components
    storage_manager = None
    api = None
    
    @app.on_event("startup")
    async def startup_event():
        nonlocal storage_manager, api
        try:
            # Initialize storage (using default path)
            storage_manager = SQLiteStorageManager()
            api = GlobuleAPI(storage_manager)
            logger.info("Web app initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize web app: {e}")
    
    @app.get("/", response_class=HTMLResponse)
    async def index():
        """Main page with search and draft functionality."""
        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Globule Web Interface</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background: #f5f5f5;
                }
                .header {
                    background: #2c3e50;
                    color: white;
                    padding: 20px;
                    border-radius: 8px;
                    margin-bottom: 20px;
                }
                .container {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 20px;
                }
                .panel {
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .search-box, .draft-box {
                    margin-bottom: 15px;
                }
                input, textarea, button {
                    width: 100%;
                    padding: 10px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    font-size: 14px;
                    margin-bottom: 10px;
                }
                button {
                    background: #3498db;
                    color: white;
                    border: none;
                    cursor: pointer;
                    transition: background 0.3s;
                }
                button:hover {
                    background: #2980b9;
                }
                .results {
                    background: #ecf0f1;
                    padding: 15px;
                    border-radius: 4px;
                    min-height: 200px;
                    max-height: 400px;
                    overflow-y: auto;
                    font-family: monospace;
                    white-space: pre-wrap;
                }
                .status {
                    padding: 10px;
                    border-radius: 4px;
                    margin: 10px 0;
                }
                .success { background: #d4edda; color: #155724; }
                .error { background: #f8d7da; color: #721c24; }
                .info { background: #d1ecf1; color: #0c5460; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🌐 Globule Web Interface</h1>
                <p>Search globules and manage drafts through your web browser</p>
            </div>
            
            <div class="container">
                <div class="panel">
                    <h2>🔍 Search Globules</h2>
                    <div class="search-box">
                        <input type="text" id="searchQuery" placeholder="Enter natural language query (e.g., 'valet maria honda')" />
                        <button onclick="performSearch()">Search</button>
                    </div>
                    <div id="searchResults" class="results">Search results will appear here...</div>
                </div>
                
                <div class="panel">
                    <h2>📝 Draft Management</h2>
                    <div class="draft-box">
                        <textarea id="draftContent" rows="4" placeholder="Add content to draft..."></textarea>
                        <button onclick="addToDraft()">Add to Draft</button>
                        <button onclick="getDraftStats()">Get Draft Stats</button>
                        <button onclick="exportDraft()">Export Draft</button>
                    </div>
                    <div id="draftResults" class="results">Draft operations will show here...</div>
                </div>
            </div>
            
            <div class="panel" style="margin-top: 20px;">
                <h2>ℹ️ Available Operations</h2>
                <p><strong>Search:</strong> Use natural language queries like "valet maria honda" or "car parked yesterday"</p>
                <p><strong>Draft:</strong> Add search results or custom content to your current draft</p>
                <p><strong>Export:</strong> Save your draft to a file for later use</p>
                <p><strong>CLI Equivalent:</strong> All operations mirror the CLI commands (globule search, globule add-to-draft, etc.)</p>
            </div>
            
            <script>
                async function performSearch() {
                    const query = document.getElementById('searchQuery').value;
                    const resultsDiv = document.getElementById('searchResults');
                    
                    if (!query.trim()) {
                        resultsDiv.innerHTML = '<div class="error">Please enter a search query</div>';
                        return;
                    }
                    
                    resultsDiv.innerHTML = '<div class="info">Searching...</div>';
                    
                    try {
                        const response = await fetch('/api/search', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ query: query })
                        });
                        
                        const result = await response.json();
                        
                        if (result.success) {
                            resultsDiv.innerHTML = `<div class="success">Search completed</div><pre>${result.data}</pre>`;
                        } else {
                            resultsDiv.innerHTML = `<div class="error">Search failed: ${result.message}</div>`;
                        }
                    } catch (error) {
                        resultsDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
                    }
                }
                
                async function addToDraft() {
                    const content = document.getElementById('draftContent').value;
                    const resultsDiv = document.getElementById('draftResults');
                    
                    if (!content.trim()) {
                        resultsDiv.innerHTML = '<div class="error">Please enter content to add</div>';
                        return;
                    }
                    
                    resultsDiv.innerHTML = '<div class="info">Adding to draft...</div>';
                    
                    try {
                        const response = await fetch('/api/draft/add', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ content: content })
                        });
                        
                        const result = await response.json();
                        
                        if (result.success) {
                            resultsDiv.innerHTML = `<div class="success">Added to draft successfully</div>`;
                            document.getElementById('draftContent').value = '';
                        } else {
                            resultsDiv.innerHTML = `<div class="error">Failed to add: ${result.message}</div>`;
                        }
                    } catch (error) {
                        resultsDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
                    }
                }
                
                async function getDraftStats() {
                    const resultsDiv = document.getElementById('draftResults');
                    resultsDiv.innerHTML = '<div class="info">Getting draft stats...</div>';
                    
                    try {
                        const response = await fetch('/api/draft/stats');
                        const result = await response.json();
                        
                        if (result.success) {
                            const stats = result.data;
                            resultsDiv.innerHTML = `
                                <div class="success">Draft Statistics</div>
                                <pre>Path: ${stats.path}
Size: ${stats.size_bytes} bytes
Lines: ${stats.line_count}
Words: ${stats.word_count}
Modified: ${stats.last_modified}</pre>`;
                        } else {
                            resultsDiv.innerHTML = `<div class="error">Failed to get stats: ${result.message}</div>`;
                        }
                    } catch (error) {
                        resultsDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
                    }
                }
                
                async function exportDraft() {
                    const resultsDiv = document.getElementById('draftResults');
                    resultsDiv.innerHTML = '<div class="info">Exporting draft...</div>';
                    
                    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
                    const filename = `draft_export_${timestamp}.md`;
                    
                    try {
                        const response = await fetch('/api/draft/export', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ output_path: filename })
                        });
                        
                        const result = await response.json();
                        
                        if (result.success) {
                            resultsDiv.innerHTML = `<div class="success">Exported to: ${result.data.export_path}</div>`;
                        } else {
                            resultsDiv.innerHTML = `<div class="error">Export failed: ${result.message}</div>`;
                        }
                    } catch (error) {
                        resultsDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
                    }
                }
                
                // Allow Enter key in search box
                document.getElementById('searchQuery').addEventListener('keypress', function(e) {
                    if (e.key === 'Enter') {
                        performSearch();
                    }
                });
            </script>
        </body>
        </html>
        """
        return html_content
    
    @app.post("/api/search")
    async def api_search(request: Request):
        """API endpoint for search functionality."""
        try:
            data = await request.json()
            query = data.get('query', '')
            
            if not query:
                raise HTTPException(status_code=400, detail="Query is required")
            
            if api is None:
                raise HTTPException(status_code=500, detail="API not initialized")
                
            result = await api.search(query)
            return JSONResponse(result)
            
        except Exception as e:
            logger.error(f"Search API error: {e}")
            return JSONResponse({
                "success": False,
                "message": f"Search failed: {str(e)}",
                "data": None
            }, status_code=500)
    
    @app.post("/api/draft/add")
    async def api_draft_add(request: Request):
        """API endpoint for adding content to draft."""
        try:
            data = await request.json()
            content = data.get('content', '')
            
            if not content:
                raise HTTPException(status_code=400, detail="Content is required")
                
            if api is None:
                raise HTTPException(status_code=500, detail="API not initialized")
                
            result = await api.add_to_draft(content)
            return JSONResponse(result)
            
        except Exception as e:
            logger.error(f"Draft add API error: {e}")
            return JSONResponse({
                "success": False,
                "message": f"Add to draft failed: {str(e)}",
                "data": None
            }, status_code=500)
    
    @app.get("/api/draft/stats")
    async def api_draft_stats():
        """API endpoint for getting draft statistics."""
        try:
            if api is None:
                raise HTTPException(status_code=500, detail="API not initialized")
                
            result = await api.get_draft_stats()
            return JSONResponse(result)
            
        except Exception as e:
            logger.error(f"Draft stats API error: {e}")
            return JSONResponse({
                "success": False,
                "message": f"Get draft stats failed: {str(e)}",
                "data": None
            }, status_code=500)
    
    @app.post("/api/draft/export")
    async def api_draft_export(request: Request):
        """API endpoint for exporting draft."""
        try:
            data = await request.json()
            output_path = data.get('output_path', f'draft_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md')
            
            if api is None:
                raise HTTPException(status_code=500, detail="API not initialized")
                
            result = await api.export_draft(output_path)
            return JSONResponse(result)
            
        except Exception as e:
            logger.error(f"Draft export API error: {e}")
            return JSONResponse({
                "success": False,
                "message": f"Export draft failed: {str(e)}",
                "data": None
            }, status_code=500)
    
    return app


# For development/testing
if __name__ == "__main__":
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="127.0.0.1", port=8000)