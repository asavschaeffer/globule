"""Query engine for Globule - intelligent retrieval of thoughts."""

import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np
from rich.console import Console
from rich.table import Table

from .embedding_engine import Embedder, cosine_similarity
from .storage import Globule, Storage


class QueryEngine:
    """Engine for querying and retrieving globules intelligently."""
    
    def __init__(self, storage: Storage, embedder: Embedder):
        self.storage = storage
        self.embedder = embedder
        self.console = Console()
    
    async def search_semantic(self, query: str, limit: int = 10) -> List[Tuple[Globule, float]]:
        """Search for globules using semantic similarity."""
        try:
            # Generate embedding for the query
            query_embedding = await self.embedder.embed_text(query)
            
            # Get all globules with embeddings
            all_globules = await self.storage.search_semantic(query_embedding, limit=100)
            
            # Calculate similarities and rank results
            results = []
            for globule in all_globules:
                if globule.embedding is not None:
                    similarity = cosine_similarity(query_embedding, globule.embedding)
                    results.append((globule, similarity))
            
            # Sort by similarity (descending) and return top results
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:limit]
            
        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []
    
    async def search_temporal(self, timeframe: str = "today") -> List[Globule]:
        """Search for globules within a specific timeframe."""
        now = datetime.now()
        
        if timeframe == "today":
            start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = now.replace(hour=23, minute=59, second=59, microsecond=999999)
        elif timeframe == "yesterday":
            yesterday = now - timedelta(days=1)
            start_date = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)
        elif timeframe == "this_week":
            # Start of the week (Monday)
            start_date = now - timedelta(days=now.weekday())
            start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = now
        elif timeframe == "last_week":
            # Last week (Monday to Sunday)
            start_date = now - timedelta(days=now.weekday() + 7)
            start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = now - timedelta(days=now.weekday() + 1)
            end_date = end_date.replace(hour=23, minute=59, second=59, microsecond=999999)
        else:
            # Default to today
            start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = now
        
        return await self.storage.search_temporal(start_date, end_date)
    
    async def search_by_domain(self, domain: str, limit: int = 10) -> List[Globule]:
        """Search for globules in a specific domain."""
        # This would require a domain-specific search method in storage
        # For now, we'll do a simple filter on temporal results
        all_globules = await self.search_temporal("this_week")
        return [g for g in all_globules if g.domain == domain][:limit]
    
    async def search_combined(self, query: str, domain: Optional[str] = None, 
                            timeframe: Optional[str] = None, limit: int = 10) -> List[Tuple[Globule, float]]:
        """Combined search using semantic similarity with optional filters."""
        # Start with semantic search
        semantic_results = await self.search_semantic(query, limit=50)
        
        # Apply domain filter if specified
        if domain:
            semantic_results = [(g, s) for g, s in semantic_results if g.domain == domain]
        
        # Apply timeframe filter if specified
        if timeframe:
            temporal_globules = await self.search_temporal(timeframe)
            temporal_ids = {g.id for g in temporal_globules}
            semantic_results = [(g, s) for g, s in semantic_results if g.id in temporal_ids]
        
        return semantic_results[:limit]
    
    def display_results(self, results: List[Tuple[Globule, float]], title: str = "Search Results"):
        """Display search results in a formatted table."""
        if not results:
            self.console.print(f"[yellow]No results found.[/yellow]")
            return
        
        table = Table(title=title)
        table.add_column("Date", style="cyan", no_wrap=True)
        table.add_column("Content", style="white")
        table.add_column("Domain", style="magenta")
        table.add_column("Similarity", style="green")
        
        for globule, similarity in results:
            date_str = globule.created_at.strftime("%Y-%m-%d %H:%M")
            content = globule.content[:80] + "..." if len(globule.content) > 80 else globule.content
            domain = globule.domain or "unknown"
            similarity_str = f"{similarity:.3f}"
            
            table.add_row(date_str, content, domain, similarity_str)
        
        self.console.print(table)
    
    def display_temporal_results(self, results: List[Globule], title: str = "Temporal Results"):
        """Display temporal search results in a formatted table."""
        if not results:
            self.console.print(f"[yellow]No results found.[/yellow]")
            return
        
        table = Table(title=title)
        table.add_column("Time", style="cyan", no_wrap=True)
        table.add_column("Content", style="white")
        table.add_column("Domain", style="magenta")
        table.add_column("Category", style="blue")
        
        for globule in results:
            time_str = globule.created_at.strftime("%H:%M")
            content = globule.content[:80] + "..." if len(globule.content) > 80 else globule.content
            domain = globule.domain or "unknown"
            category = ""
            if globule.parsed_data and globule.parsed_data.get("category"):
                category = globule.parsed_data["category"]
            
            table.add_row(time_str, content, domain, category)
        
        self.console.print(table)
    
    async def get_stats(self) -> dict:
        """Get database statistics."""
        # This would require additional methods in storage
        # For now, return basic stats
        today_globules = await self.search_temporal("today")
        this_week_globules = await self.search_temporal("this_week")
        
        return {
            "today_count": len(today_globules),
            "this_week_count": len(this_week_globules),
            "domains": self._count_domains(this_week_globules),
            "categories": self._count_categories(this_week_globules)
        }
    
    def _count_domains(self, globules: List[Globule]) -> dict:
        """Count globules by domain."""
        domain_counts = {}
        for globule in globules:
            domain = globule.domain or "unknown"
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        return domain_counts
    
    def _count_categories(self, globules: List[Globule]) -> dict:
        """Count globules by category."""
        category_counts = {}
        for globule in globules:
            category = "unknown"
            if globule.parsed_data and globule.parsed_data.get("category"):
                category = globule.parsed_data["category"]
            category_counts[category] = category_counts.get(category, 0) + 1
        return category_counts
    
    def display_stats(self, stats: dict):
        """Display database statistics."""
        self.console.print(f"[bold green]Database Statistics[/bold green]")
        self.console.print(f"Today: {stats['today_count']} globules")
        self.console.print(f"This week: {stats['this_week_count']} globules")
        
        if stats['domains']:
            self.console.print(f"\n[bold blue]Domains:[/bold blue]")
            for domain, count in stats['domains'].items():
                self.console.print(f"  {domain}: {count}")
        
        if stats['categories']:
            self.console.print(f"\n[bold magenta]Categories:[/bold magenta]")
            for category, count in stats['categories'].items():
                self.console.print(f"  {category}: {count}")


def parse_query(query: str) -> dict:
    """Parse a natural language query to extract search parameters."""
    query_lower = query.lower()
    
    # Extract domain hints
    domain = None
    if any(word in query_lower for word in ["work", "meeting", "project", "client"]):
        domain = "work"
    elif any(word in query_lower for word in ["personal", "family", "home", "friend"]):
        domain = "personal"
    
    # Extract timeframe hints
    timeframe = None
    if "today" in query_lower:
        timeframe = "today"
    elif "yesterday" in query_lower:
        timeframe = "yesterday"
    elif "this week" in query_lower:
        timeframe = "this_week"
    elif "last week" in query_lower:
        timeframe = "last_week"
    
    return {
        "domain": domain,
        "timeframe": timeframe,
        "clean_query": query  # Could clean this up further
    }