"""
Real Ollama Parser for Phase 2 Intelligence.

This module implements intelligent content analysis using Ollama LLMs,
replacing the mock parser with genuine AI-powered text understanding.

Features:
- Semantic title generation
- Domain and category classification
- Keyword and entity extraction
- Sentiment and content type detection
- Structured metadata generation

Author: Globule Team
Version: 2.0.0
"""

import json
import logging
import asyncio
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

import aiohttp
from globule.core.interfaces import ParsingProvider
from globule.config.settings import get_config


@dataclass
class ParsedContent:
    """Structured representation of parsed content."""
    title: str
    category: str
    domain: str
    keywords: List[str]
    entities: List[str]
    sentiment: str
    content_type: str
    confidence_score: float
    metadata: Dict[str, Any]


class OllamaParser(ParsingProvider):
    """
    Production Ollama parser implementing intelligent content analysis.
    
    This parser uses sophisticated LLM prompting to extract meaningful
    structure and metadata from unstructured text input.
    """
    
    def __init__(self):
        """Initialize the Ollama parser with configuration."""
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Compiled regex patterns for fast path classification
        self.URL_REGEX = re.compile(r'^https?://\S+')
        self.TODO_REGEX = re.compile(r'^(TODO|TASK):', re.IGNORECASE)
        self.CODE_REGEX = re.compile(r'(def\s+\w+|function\s+\w+|class\s+\w+|SELECT\s+.*FROM)', re.IGNORECASE)
        self.EMAIL_REGEX = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.QUESTION_REGEX = re.compile(r'.+\?\s*$')
        self.BACKTICK_CODE_REGEX = re.compile(r'```[\s\S]*?```|`[^`]+`')
        self.FILE_PATH_REGEX = re.compile(r'^[a-zA-Z]:\\|^/[a-zA-Z0-9_./\-]+|^\./[a-zA-Z0-9_./\-]+|\.[a-zA-Z]{2,4}$')
        
        # JSON schema for LLM prompts
        self.JSON_SCHEMA = '''
{
  "title": "A concise, meaningful title (max 80 chars)",
  "category": "one of: note, idea, question, task, reference, draft, quote, observation",
  "domain": "one of: creative, technical, personal, academic, business, philosophy, other",
  "keywords": ["key", "terms", "from", "text"],
  "entities": ["people", "places", "concepts", "mentioned"],
  "sentiment": "one of: positive, negative, neutral, mixed",
  "content_type": "one of: prose, list, code, data, dialogue, poetry, instructions",
  "confidence_score": 0.85,
  "reasoning": "Brief explanation of your classification decisions"
}
'''
        
        # Parsing prompt template for structured extraction
        self.parsing_prompt = """
You are an expert content analyst. Analyze the following text and extract structured information.

Text to analyze:
{text}

Return your analysis as valid JSON with this exact structure:
{{
    "title": "A concise, meaningful title (max 80 chars)",
    "category": "one of: note, idea, question, task, reference, draft, quote, observation",
    "domain": "one of: creative, technical, personal, academic, business, philosophy, other",
    "keywords": ["key", "terms", "from", "text"],
    "entities": ["people", "places", "concepts", "mentioned"],
    "sentiment": "one of: positive, negative, neutral, mixed",
    "content_type": "one of: prose, list, code, data, dialogue, poetry, instructions",
    "confidence_score": 0.85,
    "reasoning": "Brief explanation of your classification decisions"
}}

Be precise and analytical. Focus on semantic meaning over surface features.
"""

    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def _ensure_session(self) -> None:
        """Ensure HTTP session is initialized."""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=self.config.ollama_timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)

    async def close(self) -> None:
        """Clean up HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None

    async def health_check(self) -> bool:
        """Check if Ollama service is available."""
        try:
            await self._ensure_session()
            url = f"{self.config.ollama_base_url}/api/tags"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    # Check if our parsing model is available
                    models = [model["name"] for model in data.get("models", [])]
                    return self.config.default_parsing_model in models
                    
        except Exception as e:
            self.logger.warning(f"Ollama health check failed: {e}")
            
        return False

    async def get_cpu_safe_model(self) -> str:
        """
        Get CPU-safe model for systems without GPU acceleration.
        
        Automatically detects available lightweight models and returns
        the most appropriate one for CPU-only execution.
        """
        try:
            await self._ensure_session()
            url = f"{self.config.ollama_base_url}/api/tags"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    models = [model["name"] for model in data.get("models", [])]
                    
                    # Priority order: fastest to slowest for CPU
                    cpu_safe_models = ["tinyllama", "phi3:mini", "gemma2:2b", "llama3.2:1b"]
                    
                    for model in cpu_safe_models:
                        if any(model in available for available in models):
                            self.logger.info(f"CPU-safe mode: Using {model} for faster processing")
                            return model
                    
                    # Fallback to configured model if no CPU-safe alternatives
                    return self.config.default_parsing_model
                    
        except Exception as e:
            self.logger.warning(f"CPU-safe model detection failed: {e}")
            return self.config.default_parsing_model

    async def health_check_with_cpu_fallback(self) -> tuple[bool, str]:
        """
        Enhanced health check that detects optimal model for current system.
        
        Returns:
            tuple: (is_healthy, optimal_model_name)
        """
        try:
            await self._ensure_session()
            url = f"{self.config.ollama_base_url}/api/tags"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    models = [model["name"] for model in data.get("models", [])]
                    
                    # First try configured model
                    if self.config.default_parsing_model in models:
                        # Test if model loads quickly (indicates GPU acceleration)
                        quick_model = await self._test_model_speed(self.config.default_parsing_model)
                        if quick_model:
                            return True, self.config.default_parsing_model
                    
                    # If slow or unavailable, find CPU-safe alternative
                    cpu_model = await self.get_cpu_safe_model()
                    cpu_model_available = any(cpu_model in available for available in models)
                    
                    return cpu_model_available, cpu_model
                    
        except Exception as e:
            self.logger.warning(f"Enhanced health check failed: {e}")
            
        return False, self.config.default_parsing_model

    async def _test_model_speed(self, model_name: str) -> bool:
        """Test if model loads/responds quickly (indicates GPU acceleration)."""
        try:
            test_payload = {
                "model": model_name,
                "prompt": "Test",
                "stream": False,
                "options": {"max_tokens": 1}
            }
            
            url = f"{self.config.ollama_base_url}/api/generate"
            
            # Set aggressive timeout for speed test
            import aiohttp
            timeout = aiohttp.ClientTimeout(total=5.0)  # 5 second max
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=test_payload) as response:
                    return response.status == 200
                    
        except Exception:
            return False

    async def parse(self, text: str, schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Parse text using resilient fast/slow path architecture.
        
        Args:
            text: Input text to analyze
            schema: Optional schema hints (not used in this implementation)
            
        Returns:
            Dict containing structured parsing results
        """
        # 1. Attempt the fast path first.
        fast_result = self._fast_path_parse(text)
        if fast_result:
            self.logger.info(f"Handled by fast path: {fast_result['metadata']['parser_type']}")
            return fast_result

        # 2. If fast path fails, check if Ollama is available before proceeding to slow path.
        try:
            # Check if Ollama service is healthy before attempting LLM parsing
            if await self.health_check():
                return await self._slow_path_parse_with_retry(text)
            else:
                # Ollama not available, skip directly to fallback
                self.logger.info("Ollama service unavailable, using fallback parsing")
                return self._create_fallback_result(text, "Ollama service unavailable")
        except Exception as e:
            self.logger.error(f"LLM parsing failed after all retries: {e}")
            # 3. If all else fails, return a safe, minimal structure.
            return self._create_fallback_result(text, str(e))

    def _fast_path_parse(self, text: str) -> Optional[Dict]:
        """
        Fast path: Cheap, deterministic regex-based classifiers.
        
        Returns complete structured result if high-confidence match found,
        None if no match (proceed to slow path).
        """
        text_stripped = text.strip()
        
        # URL Pattern
        if self.URL_REGEX.match(text_stripped):
            return {
                "title": text_stripped,
                "category": "reference",
                "domain": "weblink",
                "keywords": ["url", "link"],
                "entities": [],
                "sentiment": "neutral",
                "content_type": "data",
                "confidence_score": 1.0,
                "reasoning": "Detected URL pattern",
                "metadata": {"parser_type": "fast_path_url", "confidence_score": 1.0}
            }
        
        # TODO Pattern
        elif self.TODO_REGEX.match(text_stripped):
            task_text = text_stripped[5:].strip()  # Remove "TODO:" prefix
            return {
                "title": task_text,
                "category": "task",
                "domain": "personal",
                "keywords": ["task", "todo"],
                "entities": [],
                "sentiment": "neutral",
                "content_type": "instructions",
                "confidence_score": 1.0,
                "reasoning": "Detected TODO pattern",
                "metadata": {"parser_type": "fast_path_task", "confidence_score": 1.0}
            }
        
        # Code Pattern
        elif self.CODE_REGEX.search(text_stripped):
            return {
                "title": "Code snippet",
                "category": "reference",
                "domain": "technical",
                "keywords": ["code", "programming"],
                "entities": [],
                "sentiment": "neutral",
                "content_type": "code",
                "confidence_score": 1.0,
                "reasoning": "Detected code pattern",
                "metadata": {"parser_type": "fast_path_code", "confidence_score": 1.0}
            }
        
        # Email Pattern
        elif self.EMAIL_REGEX.search(text_stripped):
            return {
                "title": "Email address",
                "category": "reference",
                "domain": "personal",
                "keywords": ["email", "contact"],
                "entities": [],
                "sentiment": "neutral",
                "content_type": "data",
                "confidence_score": 1.0,
                "reasoning": "Detected email pattern",
                "metadata": {"parser_type": "fast_path_email", "confidence_score": 1.0}
            }
        
        # Question Pattern
        elif self.QUESTION_REGEX.match(text_stripped) and len(text_stripped.split()) <= 20:
            return {
                "title": text_stripped,
                "category": "question",
                "domain": "other",
                "keywords": ["question", "inquiry"],
                "entities": [],
                "sentiment": "neutral",
                "content_type": "prose",
                "confidence_score": 1.0,
                "reasoning": "Detected short question pattern",
                "metadata": {"parser_type": "fast_path_question", "confidence_score": 1.0}
            }
        
        # Backtick Code Pattern
        elif self.BACKTICK_CODE_REGEX.search(text_stripped):
            return {
                "title": "Code block",
                "category": "reference",
                "domain": "technical",
                "keywords": ["code", "programming", "snippet"],
                "entities": [],
                "sentiment": "neutral",
                "content_type": "code",
                "confidence_score": 1.0,
                "reasoning": "Detected backtick code block pattern",
                "metadata": {"parser_type": "fast_path_code_block", "confidence_score": 1.0}
            }
        
        # File Path Pattern
        elif self.FILE_PATH_REGEX.match(text_stripped):
            return {
                "title": f"File path: {text_stripped}",
                "category": "reference",
                "domain": "technical",
                "keywords": ["file", "path", "system"],
                "entities": [],
                "sentiment": "neutral",
                "content_type": "data",
                "confidence_score": 1.0,
                "reasoning": "Detected file path pattern",
                "metadata": {"parser_type": "fast_path_file_path", "confidence_score": 1.0}
            }
        
        # If no high-confidence match, return None to proceed to slow path
        return None

    async def _slow_path_parse_with_retry(self, text: str, max_retries: int = 2) -> Dict[str, Any]:
        """
        Slow path: LLM parsing with self-healing retry mechanism.
        
        Args:
            text: Text to parse
            max_retries: Maximum number of retry attempts
            
        Returns:
            Parsed result dictionary
            
        Raises:
            Exception: If all retries fail
        """
        await self._ensure_session()
        last_exception = None

        for attempt in range(max_retries):
            prompt = self._build_llm_prompt(text, last_exception)
            
            raw_response = await self._call_llm(prompt)

            try:
                # Attempt to parse the JSON
                parsed_json = json.loads(raw_response)
                # If successful, validate and return immediately
                self._validate_parsed_result(parsed_json)
                return self._format_result(parsed_json)
            except json.JSONDecodeError as e:
                self.logger.warning(f"LLM returned malformed JSON on attempt {attempt + 1}. Retrying.")
                last_exception = f"JSONDecodeError: {e}. The malformed text was: {raw_response}"
                # The loop will continue to the next attempt

        # If the loop finishes without returning, all retries have failed
        raise Exception(f"Failed to parse LLM response after {max_retries} attempts. Last error: {last_exception}")

    def _build_llm_prompt(self, text: str, error_context: Optional[str] = None) -> str:
        """
        Build LLM prompt with optional error context for self-healing.
        
        Args:
            text: Original text to analyze  
            error_context: Previous error for self-healing prompt
            
        Returns:
            Formatted prompt string
        """
        if error_context:
            # This is the self-healing prompt
            return f"""
You previously failed to generate valid JSON. Correct your mistake.
The error was: {error_context}

Provide ONLY the valid JSON object. Do not include any other text or apologies.
"""
        else:
            # This is the standard initial prompt  
            return f"""
Analyze the text and return a valid JSON object conforming to the following schema:
{self.JSON_SCHEMA}

Text to analyze:
{text}
"""

    async def _call_llm(self, prompt: str) -> str:
        """Make the actual LLM API call."""
        payload = {
            "model": self.config.default_parsing_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for consistent structured output
                "top_p": 0.9,
                "top_k": 40,
            }
        }
        
        url = f"{self.config.ollama_base_url}/api/generate"
        
        async with self.session.post(url, json=payload) as response:
            if response.status != 200:
                raise Exception(f"Ollama request failed with status {response.status}")
                
            data = await response.json()
            return data.get("response", "").strip()

    def _create_fallback_result(self, text: str, error_message: Optional[str] = None) -> Dict[str, Any]:
        """
        Final safety net: Create minimal safe result when all else fails.
        
        Args:
            text: Original input text
            error_message: Optional error details for debugging
            
        Returns:
            Safe fallback result dictionary with error context
        """
        reasoning = "Fallback parser - all other methods failed"
        if error_message:
            reasoning += f". Error: {error_message}"
            
        return {
            "title": text[:50],
            "category": "note",
            "domain": "general",
            "keywords": [],
            "entities": [],
            "sentiment": "neutral",
            "content_type": "prose",
            "confidence_score": 0.1,
            "reasoning": reasoning,
            "metadata": {
                "parser_type": "fatal_fallback", 
                "confidence_score": 0.1,
                "error_details": error_message,
                "processing_notes": [f"Parser failure: {error_message}"] if error_message else ["Parser failure: unknown error"]
            }
        }


    def _validate_parsed_result(self, result: Dict[str, Any]) -> None:
        """Validate that parsed result has required fields."""
        required_fields = ["title", "category", "domain", "keywords", "entities", "sentiment", "content_type", "confidence_score"]
        
        for field in required_fields:
            if field not in result:
                raise ValueError(f"Missing required field: {field}")

    def _format_result(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format parsing result for consistency with interface."""
        return {
            "title": parsed_data.get("title", "Untitled"),
            "category": parsed_data.get("category", "note"),
            "domain": parsed_data.get("domain", "other"),
            "keywords": parsed_data.get("keywords", []),
            "entities": parsed_data.get("entities", []),
            "metadata": {
                "parser_type": "ollama_llm",
                "parser_version": "3.0.0",
                "sentiment": parsed_data.get("sentiment", "neutral"),
                "content_type": parsed_data.get("content_type", "prose"),
                "confidence_score": parsed_data.get("confidence_score", 0.0),
                "reasoning": parsed_data.get("reasoning", "")
            }
        }