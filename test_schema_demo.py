"""
Demonstration of the new schema functionality in Globule parsing system.

This script shows how the enhanced parser now uses dynamic schemas for
better structured content parsing and validation.
"""

import asyncio
import json
from src.globule.services.parsing.ollama_parser import OllamaParser
from src.globule.schemas.manager import get_schema_manager


async def demonstrate_schema_functionality():
    """Demonstrate the enhanced schema functionality."""
    
    print("GLOBULE SCHEMA SYSTEM DEMONSTRATION")
    print("=" * 50)
    
    # Initialize components
    parser = OllamaParser()
    schema_manager = get_schema_manager()
    
    print(f"\nAvailable Schemas: {schema_manager.get_available_schemas()}")
    
    # Sample texts for different domains
    sample_texts = {
        "technical": """
        def fibonacci(n):
            if n <= 1:
                return n
            return fibonacci(n-1) + fibonacci(n-2)
        
        This recursive implementation has O(2^n) complexity.
        """,
        
        "academic": """
        According to Smith et al. (2023), machine learning models demonstrate 
        improved performance when trained on diverse datasets. The study 
        utilized a randomized controlled trial methodology with 1000 participants.
        """,
        
        "creative": """
        The old lighthouse keeper had seen many storms, but none quite like this.
        As the waves crashed against the rocks below, he wondered if tonight 
        would be his last vigil in this tower that had been his home for thirty years.
        """
    }
    
    print(f"\nTESTING SCHEMA SELECTION")
    print("-" * 30)
    
    # Test schema determination
    test_cases = [
        (None, "Default behavior"),
        ({"name": "technical"}, "Named schema selection"),
        ({"domain": "academic"}, "Domain-based selection"),
        ({"name": "nonexistent"}, "Fallback for unknown schema")
    ]
    
    for schema_param, description in test_cases:
        schema_name = parser._determine_schema_name(schema_param)
        print(f"- {description}: '{schema_name}'")
    
    print(f"\nSCHEMA-AWARE PARSING DEMONSTRATION")
    print("-" * 40)
    
    # Test parsing with different schemas
    for domain, text in sample_texts.items():
        print(f"\nParsing {domain.upper()} content:")
        print(f"Text: {text.strip()[:80]}...")
        
        # Parse with domain-specific schema
        schema_param = {"domain": domain}
        result = await parser.parse(text, schema_param)
        
        print(f"Schema used: {parser._determine_schema_name(schema_param)}")
        print(f"Fast path: {'Yes' if result.get('metadata', {}).get('parser_type', '').startswith('fast_path') else 'No'}")
        print(f"Title: {result.get('title', 'N/A')}")
        print(f"Category: {result.get('category', 'N/A')}")
        print(f"Domain: {result.get('metadata', {}).get('domain', result.get('domain', 'N/A'))}")
        
        # Show additional domain-specific fields
        if domain == "technical" and 'metadata' in result:
            technologies = result['metadata'].get('technologies', [])
            if technologies:
                print(f"Technologies: {technologies}")
        
        elif domain == "academic" and 'metadata' in result:
            authors = result['metadata'].get('authors', [])
            if authors:
                print(f"Authors: {authors}")
        
        elif domain == "creative" and 'metadata' in result:
            mood = result['metadata'].get('mood', '')
            if mood:
                print(f"Mood: {mood}")
    
    print(f"\nSCHEMA VALIDATION DEMONSTRATION")
    print("-" * 35)
    
    # Test validation with good and bad data
    test_data = [
        {
            "name": "Valid default schema data",
            "schema": "default", 
            "data": {
                "title": "Test Title",
                "category": "note",
                "domain": "technical",
                "keywords": ["test", "demo"],
                "entities": ["Python", "code"],
                "sentiment": "neutral",
                "content_type": "prose",
                "confidence_score": 0.85
            }
        },
        {
            "name": "Invalid data (missing required fields)",
            "schema": "default",
            "data": {
                "title": "Incomplete Title",
                "category": "note"
                # Missing required fields
            }
        },
        {
            "name": "Valid technical schema data",
            "schema": "technical",
            "data": {
                "title": "Python Function",
                "category": "code_snippet",
                "domain": "software_engineering", 
                "technologies": ["Python"],
                "keywords": ["function", "recursion"],
                "complexity_level": "intermediate",
                "confidence_score": 0.9
            }
        }
    ]
    
    for test_case in test_data:
        print(f"\n- Testing: {test_case['name']}")
        is_valid, errors = schema_manager.validate_data(test_case['data'], test_case['schema'])
        
        if is_valid:
            print("  [PASS] Validation passed")
        else:
            print("  [FAIL] Validation failed:")
            for error in errors[:3]:  # Show first 3 errors
                print(f"    - {error}")
            if len(errors) > 3:
                print(f"    ... and {len(errors) - 3} more errors")
    
    print(f"\nSCHEMA FORMATTING FOR LLM")
    print("-" * 30)
    
    # Show how schemas are formatted for LLM prompts
    for schema_name in ["default", "technical", "academic"]:
        print(f"\n{schema_name.upper()} Schema for LLM:")
        formatted = schema_manager.format_schema_for_llm(schema_name)
        # Show first few lines
        lines = formatted.split('\n')[:6]
        for line in lines:
            print(f"  {line}")
        if len(formatted.split('\n')) > 6:
            print("  ...")
    
    print(f"\nSCHEMA SYSTEM READY")
    print("=" * 50)
    print("The enhanced parser now supports:")
    print("- Dynamic schema loading from JSON files")
    print("- Schema selection by name or domain")
    print("- Comprehensive validation with jsonschema")
    print("- LLM prompt customization per schema")
    print("- Progressive enhancement for specialized content")
    
    await parser.close()


if __name__ == "__main__":
    asyncio.run(demonstrate_schema_functionality())