"""Synthesis engine for Globule - generates reports and summaries."""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional

from rich.console import Console
from rich.markdown import Markdown

from .parser_engine import create_parser
from .query_engine import QueryEngine
from .storage import Globule


class SynthesisEngine:
    """Engine for synthesizing thoughts into reports and summaries."""
    
    def __init__(self, query_engine: QueryEngine, use_llm: bool = True):
        self.query_engine = query_engine
        self.use_llm = use_llm
        self.parser = None
        self.console = Console()
    
    async def initialize(self):
        """Initialize the synthesis engine."""
        if self.use_llm:
            self.parser = await create_parser(use_ollama=True)
    
    async def generate_daily_summary(self, date: Optional[datetime] = None) -> str:
        """Generate a daily summary of thoughts."""
        if date is None:
            date = datetime.now()
        
        # Get today's globules
        today_globules = await self.query_engine.search_temporal("today")
        
        if not today_globules:
            return "No thoughts captured today."
        
        if self.use_llm and self.parser:
            return await self._generate_llm_summary(today_globules, "daily")
        else:
            return self._generate_simple_summary(today_globules, "daily")
    
    async def generate_weekly_summary(self) -> str:
        """Generate a weekly summary of thoughts."""
        week_globules = await self.query_engine.search_temporal("this_week")
        
        if not week_globules:
            return "No thoughts captured this week."
        
        if self.use_llm and self.parser:
            return await self._generate_llm_summary(week_globules, "weekly")
        else:
            return self._generate_simple_summary(week_globules, "weekly")
    
    async def generate_domain_summary(self, domain: str) -> str:
        """Generate a summary for a specific domain."""
        domain_globules = await self.query_engine.search_by_domain(domain)
        
        if not domain_globules:
            return f"No thoughts found for domain: {domain}"
        
        if self.use_llm and self.parser:
            return await self._generate_llm_summary(domain_globules, f"{domain} domain")
        else:
            return self._generate_simple_summary(domain_globules, f"{domain} domain")
    
    async def _generate_llm_summary(self, globules: List[Globule], summary_type: str) -> str:
        """Generate a summary using LLM."""
        if not globules:
            return "No thoughts to summarize."
        
        # Prepare the context
        thoughts_text = "\n".join([f"- {g.content}" for g in globules])
        
        prompt = f"""Please create a {summary_type} summary based on these thoughts:

{thoughts_text}

Create a well-structured summary that:
1. Groups similar thoughts together
2. Highlights key themes and insights
3. Identifies any action items or important notes
4. Uses markdown formatting for readability
5. Keeps it concise but comprehensive

Summary:"""
        
        try:
            # Use the parser's LLM functionality
            if hasattr(self.parser, 'client') and hasattr(self.parser, 'model_name'):
                response = await self.parser.client.post(
                    f"{self.parser.base_url}/api/generate",
                    json={
                        "model": self.parser.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,
                            "max_tokens": 800
                        }
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("response", "Failed to generate summary")
            
            # Fallback to simple summary
            return self._generate_simple_summary(globules, summary_type)
            
        except Exception as e:
            print(f"Failed to generate LLM summary: {e}")
            return self._generate_simple_summary(globules, summary_type)
    
    def _generate_simple_summary(self, globules: List[Globule], summary_type: str) -> str:
        """Generate a simple summary without LLM."""
        if not globules:
            return "No thoughts to summarize."
        
        # Group by domain
        domain_groups = {}
        for globule in globules:
            domain = globule.domain or "other"
            if domain not in domain_groups:
                domain_groups[domain] = []
            domain_groups[domain].append(globule)
        
        # Group by category
        category_groups = {}
        for globule in globules:
            category = "note"
            if globule.parsed_data and globule.parsed_data.get("category"):
                category = globule.parsed_data["category"]
            
            if category not in category_groups:
                category_groups[category] = []
            category_groups[category].append(globule)
        
        # Build the summary
        summary_lines = [
            f"# {summary_type.title()} Summary",
            f"",
            f"**Total thoughts:** {len(globules)}",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d')}",
            f"",
        ]
        
        # Add domain breakdown
        if len(domain_groups) > 1:
            summary_lines.extend([
                "## By Domain",
                ""
            ])
            
            for domain, domain_globules in domain_groups.items():
                summary_lines.extend([
                    f"### {domain.title()} ({len(domain_globules)} thoughts)",
                    ""
                ])
                
                for globule in domain_globules[:3]:  # Show first 3
                    time_str = globule.created_at.strftime("%H:%M")
                    content = globule.content[:100] + "..." if len(globule.content) > 100 else globule.content
                    summary_lines.append(f"- [{time_str}] {content}")
                
                if len(domain_globules) > 3:
                    summary_lines.append(f"- ... and {len(domain_globules) - 3} more")
                
                summary_lines.append("")
        
        # Add category breakdown
        if len(category_groups) > 1:
            summary_lines.extend([
                "## By Category",
                ""
            ])
            
            for category, cat_globules in category_groups.items():
                summary_lines.extend([
                    f"### {category.title()} ({len(cat_globules)} thoughts)",
                    ""
                ])
                
                for globule in cat_globules[:2]:  # Show first 2
                    time_str = globule.created_at.strftime("%H:%M")
                    content = globule.content[:80] + "..." if len(globule.content) > 80 else globule.content
                    summary_lines.append(f"- [{time_str}] {content}")
                
                if len(cat_globules) > 2:
                    summary_lines.append(f"- ... and {len(cat_globules) - 2} more")
                
                summary_lines.append("")
        
        # Add recent thoughts
        recent_thoughts = sorted(globules, key=lambda g: g.created_at, reverse=True)[:5]
        summary_lines.extend([
            "## Recent Thoughts",
            ""
        ])
        
        for globule in recent_thoughts:
            time_str = globule.created_at.strftime("%H:%M")
            content = globule.content[:120] + "..." if len(globule.content) > 120 else globule.content
            summary_lines.append(f"- [{time_str}] {content}")
        
        return "\n".join(summary_lines)
    
    def display_summary(self, summary: str):
        """Display a summary using rich formatting."""
        self.console.print(Markdown(summary))
    
    async def export_summary(self, summary: str, filename: str):
        """Export summary to a file."""
        try:
            with open(filename, 'w') as f:
                f.write(summary)
            self.console.print(f"[green]Summary exported to {filename}[/green]")
        except Exception as e:
            self.console.print(f"[red]Failed to export summary: {e}[/red]")
    
    def _get_insights(self, globules: List[Globule]) -> List[str]:
        """Extract insights from globules."""
        insights = []
        
        # Count domains
        domain_counts = {}
        for globule in globules:
            domain = globule.domain or "other"
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        if domain_counts:
            most_common_domain = max(domain_counts, key=domain_counts.get)
            insights.append(f"Most active domain: {most_common_domain} ({domain_counts[most_common_domain]} thoughts)")
        
        # Count categories
        category_counts = {}
        for globule in globules:
            category = "note"
            if globule.parsed_data and globule.parsed_data.get("category"):
                category = globule.parsed_data["category"]
            category_counts[category] = category_counts.get(category, 0) + 1
        
        if category_counts:
            most_common_category = max(category_counts, key=category_counts.get)
            insights.append(f"Most common type: {most_common_category} ({category_counts[most_common_category]} thoughts)")
        
        # Time patterns
        hours = [g.created_at.hour for g in globules]
        if hours:
            avg_hour = sum(hours) / len(hours)
            if avg_hour < 12:
                insights.append("You tend to capture thoughts in the morning")
            elif avg_hour < 17:
                insights.append("You tend to capture thoughts in the afternoon")
            else:
                insights.append("You tend to capture thoughts in the evening")
        
        return insights