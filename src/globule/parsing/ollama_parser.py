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
        Parse text using Ollama LLM to extract structured information.
        
        Args:
            text: Input text to analyze
            schema: Optional schema hints (not used in this implementation)
            
        Returns:
            Dict containing structured parsing results
            
        Raises:
            Exception: If parsing fails and fallback is not possible
        """
        if not text.strip():
            return self._create_empty_result(text)
            
        try:
            await self._ensure_session()
            
            # ATTEMPT: Intelligent model selection with CPU-safe fallback
            self.logger.info(f"ATTEMPT: Using 'ollama_parser' with model '{self.config.default_parsing_model}'...")
            
            # Enhanced health check with automatic CPU-safe detection
            is_healthy, optimal_model = await self.health_check_with_cpu_fallback()
            
            if not is_healthy:
                self.logger.warning(f"FAILURE: Ollama service unavailable at {self.config.ollama_base_url}")
                self.logger.info("ACTION: Engaging fallback parser 'enhanced_fallback'")
                result = await self._enhanced_fallback_parse(text)
                self.logger.info(f"SUCCESS: Parsed with fallback. Confidence: {result['metadata']['confidence_score']:.2f}")
                return result
            
            # Use optimal model (might be CPU-safe alternative)
            if optimal_model != self.config.default_parsing_model:
                self.logger.info(f"ACTION: CPU-safe mode detected, switching to '{optimal_model}' for better performance")
            
            # Perform LLM-based parsing with optimal model
            result = await self._llm_parse(text, model_override=optimal_model)
            confidence = result.get('confidence_score', 0)
            self.logger.info(f"SUCCESS: LLM parsing completed with '{optimal_model}'. Confidence: {confidence:.2f}")
            
            return self._format_result(result)
            
        except Exception as e:
            self.logger.warning(f"FAILURE: LLM parsing error - {type(e).__name__}: {str(e)}")
            self.logger.info("ACTION: Engaging fallback parser 'enhanced_fallback'")
            result = await self._enhanced_fallback_parse(text)
            self.logger.info(f"SUCCESS: Parsed with fallback. Confidence: {result['metadata']['confidence_score']:.2f}")
            return result

    async def _llm_parse(self, text: str, model_override: str = None) -> Dict[str, Any]:
        """Perform LLM-based parsing using Ollama."""
        model_to_use = model_override or self.config.default_parsing_model
        prompt = self.parsing_prompt.format(text=text[:2000])  # Limit context length
        
        payload = {
            "model": model_to_use,
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
            llm_response = data.get("response", "").strip()
            
            # Parse JSON response from LLM
            try:
                # Extract JSON from response (LLM might include extra text)
                json_start = llm_response.find("{")
                json_end = llm_response.rfind("}") + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = llm_response[json_start:json_end]
                    parsed_result = json.loads(json_str)
                    
                    # Validate required fields
                    self._validate_parsed_result(parsed_result)
                    return parsed_result
                else:
                    raise ValueError("No valid JSON found in LLM response")
                    
            except (json.JSONDecodeError, ValueError) as e:
                self.logger.warning(f"Failed to parse LLM JSON response: {e}")
                raise Exception(f"Invalid LLM response format: {e}")

    async def _enhanced_fallback_parse(self, text: str) -> Dict[str, Any]:
        """
        Enhanced fallback parser using heuristics when LLM is unavailable.
        
        This provides intelligent analysis without requiring Ollama.
        """
        # Simulate processing time
        await asyncio.sleep(0.05)
        
        # Analyze text characteristics
        word_count = len(text.split())
        has_question = "?" in text
        has_code = any(keyword in text.lower() for keyword in ["def ", "function", "class ", "import", "select"])
        has_numbers = any(char.isdigit() for char in text)
        has_urls = "http" in text.lower() or "www." in text.lower()
        
        # Generate intelligent title
        title = self._generate_title(text)
        
        # Classify category based on content analysis
        category = self._classify_category(text, has_question, has_code)
        
        # Classify domain
        domain = self._classify_domain(text, has_code, has_numbers)
        
        # Extract keywords using simple NLP
        keywords = self._extract_keywords(text)
        
        # Detect entities
        entities = self._extract_entities(text)
        
        # Analyze sentiment
        sentiment = self._analyze_sentiment(text)
        
        # Determine content type
        content_type = self._classify_content_type(text, has_code, has_urls)
        
        return {
            "title": title,
            "category": category,
            "domain": domain,
            "keywords": keywords,
            "entities": entities,
            "metadata": {
                "parser_type": "enhanced_fallback",
                "parser_version": "2.0.0",
                "sentiment": sentiment,
                "content_type": content_type,
                "confidence_score": 0.75,  # Reasonable confidence for heuristic analysis
                "word_count": word_count,
                "analysis_features": {
                    "has_question": has_question,
                    "has_code": has_code,
                    "has_numbers": has_numbers,
                    "has_urls": has_urls
                }
            }
        }

    def _generate_title(self, text: str) -> str:
        """Generate an intelligent title from text."""
        # Use first sentence or meaningful portion
        sentences = text.split(".")
        first_sentence = sentences[0].strip()
        
        if len(first_sentence) <= 80:
            return first_sentence
        
        # Truncate intelligently at word boundary
        words = first_sentence.split()
        title_words = []
        char_count = 0
        
        for word in words:
            if char_count + len(word) + 1 > 77:  # Leave room for "..."
                break
            title_words.append(word)
            char_count += len(word) + 1
            
        return " ".join(title_words) + "..."

    def _classify_category(self, text: str, has_question: bool, has_code: bool) -> str:
        """Classify text into content category."""
        text_lower = text.lower()
        
        if has_question:
            return "question"
        elif has_code:
            return "reference"
        elif any(word in text_lower for word in ["todo", "task", "need to", "should", "must"]):
            return "task"
        elif any(word in text_lower for word in ["idea", "concept", "what if", "perhaps"]):
            return "idea"
        elif text.startswith('"') or "said" in text_lower:
            return "quote"
        elif len(text.split()) > 50:
            return "draft"
        else:
            return "note"

    def _classify_domain(self, text: str, has_code: bool, has_numbers: bool) -> str:
        """Classify text domain based on content analysis."""
        text_lower = text.lower()
        
        if has_code or any(word in text_lower for word in ["algorithm", "database", "api", "programming"]):
            return "technical"
        elif any(word in text_lower for word in ["story", "creative", "imagine", "character", "poetry"]):
            return "creative"
        elif any(word in text_lower for word in ["feel", "emotion", "personal", "my", "journal"]):
            return "personal"
        elif any(word in text_lower for word in ["research", "study", "theory", "academic", "paper"]):
            return "academic"
        elif any(word in text_lower for word in ["business", "strategy", "market", "customer", "revenue"]):
            return "business"
        elif any(word in text_lower for word in ["philosophy", "meaning", "existence", "ethics", "moral"]):
            return "philosophy"
        else:
            return "other"

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords using simple frequency analysis."""
        import re
        
        # Simple tokenization and filtering
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        common_words = {"the", "and", "that", "have", "for", "not", "with", "you", "this", "but", "his", "from", "they", "she", "her", "been", "than", "its", "were", "said", "each", "which", "their", "time", "will", "about", "would", "there", "could", "other", "more", "very", "what", "know", "just", "first", "get", "has", "had", "let", "put", "say", "set", "run", "made"}
        
        # Filter out common words and get unique keywords
        keywords = [word for word in set(words) if word not in common_words and len(word) > 3]
        
        # Return top 5 by length (longer words tend to be more specific)
        return sorted(keywords, key=len, reverse=True)[:5]

    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities using simple pattern matching."""
        import re
        
        entities = []
        
        # Capitalized words (potential proper nouns)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', text)
        entities.extend(proper_nouns[:3])  # Top 3
        
        # URLs
        urls = re.findall(r'https?://[^\s]+', text)
        entities.extend([url[:30] + "..." if len(url) > 30 else url for url in urls])
        
        return list(set(entities))[:5]  # Max 5 unique entities

    def _analyze_sentiment(self, text: str) -> str:
        """Analyze sentiment using keyword-based approach."""
        text_lower = text.lower()
        
        positive_words = ["good", "great", "excellent", "amazing", "wonderful", "love", "like", "happy", "excited", "beautiful", "perfect", "best"]
        negative_words = ["bad", "terrible", "awful", "hate", "dislike", "sad", "angry", "frustrated", "worst", "horrible", "disgusting"]
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        elif positive_count > 0 or negative_count > 0:
            return "mixed"
        else:
            return "neutral"

    def _classify_content_type(self, text: str, has_code: bool, has_urls: bool) -> str:
        """Classify the structural type of content."""
        lines = text.split('\n')
        
        if has_code:
            return "code"
        elif has_urls and len(lines) > 3:
            return "data"
        elif text.count('\n') > 5 and any(line.strip().startswith(('-', '*', '1.', '2.')) for line in lines):
            return "list"
        elif '"' in text and text.count('"') >= 4:
            return "dialogue"
        elif any(word in text.lower() for word in ["step", "first", "then", "next", "finally"]):
            return "instructions"
        elif len(text.split()) > 100 and text.count('.') > 5:
            return "prose"
        else:
            return "prose"

    def _validate_parsed_result(self, result: Dict[str, Any]) -> None:
        """Validate that parsed result has required fields."""
        required_fields = ["title", "category", "domain", "keywords", "entities", "sentiment", "content_type"]
        
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
                "parser_version": "2.0.0",
                "sentiment": parsed_data.get("sentiment", "neutral"),
                "content_type": parsed_data.get("content_type", "prose"),
                "confidence_score": parsed_data.get("confidence_score", 0.0),
                "reasoning": parsed_data.get("reasoning", "")
            }
        }

    def _create_empty_result(self, text: str) -> Dict[str, Any]:
        """Create result for empty input."""
        return {
            "title": "Empty Input",
            "category": "note",
            "domain": "other",
            "keywords": [],
            "entities": [],
            "metadata": {
                "parser_type": "empty_input",
                "parser_version": "2.0.0",
                "sentiment": "neutral",
                "content_type": "prose",
                "confidence_score": 0.0
            }
        }