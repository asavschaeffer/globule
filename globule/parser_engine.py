"""Parser engine for Globule - extracts meaning from text using LLM."""

import json
import asyncio
from typing import Any, Dict, List, Optional, Protocol

import httpx
from pydantic import BaseModel


class ParsedResult(BaseModel):
    """Result of parsing text with LLM."""
    entities: List[Dict[str, Any]] = []
    category: Optional[str] = None
    topics: List[str] = []
    temporal: Optional[str] = None
    domain: Optional[str] = None
    sentiment: Optional[str] = None
    keywords: List[str] = []


class Parser(Protocol):
    """Abstract interface for text parsing engines."""
    
    async def parse_text(self, text: str) -> ParsedResult:
        """Parse text and extract structured information."""
        ...


class OllamaParser:
    """Ollama-based parser using llama3.2:3b model."""
    
    def __init__(self, model_name: str = "llama3.2:3b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def parse_text(self, text: str) -> ParsedResult:
        """Parse text using Ollama LLM."""
        prompt = self._build_prompt(text)
        
        try:
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "max_tokens": 500
                    }
                }
            )
            response.raise_for_status()
            
            result = response.json()
            llm_output = result.get("response", "")
            
            # Parse the LLM output
            parsed_result = self._parse_llm_output(llm_output)
            return parsed_result
            
        except Exception as e:
            # Return empty result if parsing fails
            print(f"Warning: Failed to parse text with LLM: {e}")
            return ParsedResult()
    
    def _build_prompt(self, text: str) -> str:
        """Build the prompt for the LLM."""
        return f"""Analyze this text and extract structured information. Return your response as valid JSON with the following structure:

{{
  "entities": [
    {{"name": "entity_name", "type": "person|organization|location|concept"}}
  ],
  "category": "meeting|idea|task|note|complaint|question|observation",
  "topics": ["topic1", "topic2"],
  "temporal": "today|yesterday|this_week|last_week|specific_date|null",
  "domain": "work|personal|finance|health|family|travel|other",
  "sentiment": "positive|negative|neutral",
  "keywords": ["keyword1", "keyword2"]
}}

Text to analyze: "{text}"

JSON response:"""
    
    def _parse_llm_output(self, output: str) -> ParsedResult:
        """Parse the LLM output and return structured data."""
        try:
            # Extract JSON from the output
            json_start = output.find('{')
            json_end = output.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                return ParsedResult()
            
            json_str = output[json_start:json_end]
            parsed_data = json.loads(json_str)
            
            # Clean and validate the data
            return ParsedResult(
                entities=parsed_data.get("entities", []),
                category=parsed_data.get("category"),
                topics=parsed_data.get("topics", []),
                temporal=parsed_data.get("temporal"),
                domain=parsed_data.get("domain"),
                sentiment=parsed_data.get("sentiment"),
                keywords=parsed_data.get("keywords", [])
            )
            
        except (json.JSONDecodeError, ValueError) as e:
            # Try to extract basic info with fallback logic
            return self._fallback_parse(output)
    
    def _fallback_parse(self, output: str) -> ParsedResult:
        """Fallback parsing when JSON parsing fails."""
        # Simple keyword-based parsing
        result = ParsedResult()
        
        # Basic domain detection
        work_keywords = ["meeting", "project", "deadline", "client", "boss", "salary", "work"]
        personal_keywords = ["family", "friend", "home", "personal", "shopping", "vacation"]
        
        text_lower = output.lower()
        
        if any(keyword in text_lower for keyword in work_keywords):
            result.domain = "work"
        elif any(keyword in text_lower for keyword in personal_keywords):
            result.domain = "personal"
        else:
            result.domain = "other"
        
        # Basic sentiment detection
        positive_keywords = ["good", "great", "happy", "excited", "love", "awesome", "amazing"]
        negative_keywords = ["bad", "terrible", "sad", "angry", "hate", "awful", "problem"]
        
        if any(keyword in text_lower for keyword in positive_keywords):
            result.sentiment = "positive"
        elif any(keyword in text_lower for keyword in negative_keywords):
            result.sentiment = "negative"
        else:
            result.sentiment = "neutral"
        
        return result
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()


class SimpleParser:
    """Simple rule-based parser as fallback."""
    
    async def parse_text(self, text: str) -> ParsedResult:
        """Parse text using simple rules."""
        result = ParsedResult()
        text_lower = text.lower()
        
        # Basic category detection
        if any(word in text_lower for word in ["meeting", "met", "discussed", "talked"]):
            result.category = "meeting"
        elif any(word in text_lower for word in ["idea", "think", "maybe", "could"]):
            result.category = "idea"
        elif any(word in text_lower for word in ["need", "todo", "must", "should"]):
            result.category = "task"
        elif any(word in text_lower for word in ["problem", "issue", "complaint", "wrong"]):
            result.category = "complaint"
        else:
            result.category = "note"
        
        # Basic domain detection
        work_keywords = ["meeting", "project", "deadline", "client", "boss", "salary", "work", "office"]
        personal_keywords = ["family", "friend", "home", "personal", "shopping", "vacation", "dinner"]
        
        if any(keyword in text_lower for keyword in work_keywords):
            result.domain = "work"
        elif any(keyword in text_lower for keyword in personal_keywords):
            result.domain = "personal"
        else:
            result.domain = "other"
        
        # Basic sentiment
        positive_keywords = ["good", "great", "happy", "excited", "love", "awesome", "amazing", "excellent"]
        negative_keywords = ["bad", "terrible", "sad", "angry", "hate", "awful", "problem", "issue"]
        
        if any(keyword in text_lower for keyword in positive_keywords):
            result.sentiment = "positive"
        elif any(keyword in text_lower for keyword in negative_keywords):
            result.sentiment = "negative"
        else:
            result.sentiment = "neutral"
        
        # Extract basic keywords (words longer than 3 characters)
        words = text.split()
        keywords = [word.strip('.,!?;:()[]{}') for word in words if len(word) > 3]
        result.keywords = keywords[:5]  # Take first 5 keywords
        
        return result


async def create_parser(use_ollama: bool = True) -> Parser:
    """Factory function to create the appropriate parser."""
    if use_ollama:
        try:
            parser = OllamaParser()
            # Test if Ollama is available
            await parser.parse_text("test")
            return parser
        except Exception:
            print("Ollama not available, using simple parser")
    
    return SimpleParser()


def detect_domain(parsed_data: ParsedResult) -> str:
    """Detect the domain from parsed data."""
    if parsed_data.domain:
        return parsed_data.domain
    
    # Fallback domain detection
    category = parsed_data.category or ""
    topics = " ".join(parsed_data.topics)
    
    work_indicators = ["meeting", "project", "client", "deadline", "budget", "report"]
    personal_indicators = ["family", "friend", "home", "shopping", "vacation", "dinner"]
    
    combined_text = f"{category} {topics}".lower()
    
    if any(indicator in combined_text for indicator in work_indicators):
        return "work"
    elif any(indicator in combined_text for indicator in personal_indicators):
        return "personal"
    else:
        return "other"