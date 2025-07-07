#!/usr/bin/env python3
"""Basic test script to verify Globule functionality."""

import asyncio
import sys
from pathlib import Path

# Add the globule package to the path
sys.path.insert(0, str(Path(__file__).parent))

from globule.config import Config, create_default_config
from globule.storage import SQLiteStorage, Globule, generate_id
from datetime import datetime

async def test_basic_functionality():
    """Test basic functionality without heavy dependencies."""
    print("🧪 Testing Globule Basic Functionality")
    print("=" * 50)
    
    # Test 1: Config creation
    print("\n1. Testing configuration...")
    try:
        config = create_default_config()
        print("✓ Config created successfully")
        print(f"  - LLM Provider: {config.llm_provider}")
        print(f"  - Embedding Model: {config.embedding_model}")
        print(f"  - Database Path: {config.db_path}")
    except Exception as e:
        print(f"✗ Config creation failed: {e}")
        return False
    
    # Test 2: Storage functionality
    print("\n2. Testing storage...")
    try:
        storage = SQLiteStorage("test_globule.db")
        
        # Create a test globule
        globule = Globule(
            id=generate_id(),
            content="This is a test thought about machine learning",
            created_at=datetime.now(),
            domain="work",
            metadata={"test": True}
        )
        
        # Store it
        await storage.store_globule(globule)
        print("✓ Storage test passed")
        print(f"  - Stored globule with ID: {globule.id}")
        
        # Retrieve it
        retrieved = await storage.retrieve_by_id(globule.id)
        if retrieved and retrieved.content == globule.content:
            print("✓ Retrieval test passed")
        else:
            print("✗ Retrieval test failed")
            return False
        
        # Test temporal search
        from datetime import datetime, timedelta
        now = datetime.now()
        start_date = now - timedelta(minutes=1)
        end_date = now + timedelta(minutes=1)
        
        temporal_results = await storage.search_temporal(start_date, end_date)
        if temporal_results:
            print("✓ Temporal search test passed")
            print(f"  - Found {len(temporal_results)} recent globules")
        else:
            print("✗ Temporal search test failed")
            return False
            
    except Exception as e:
        print(f"✗ Storage test failed: {e}")
        return False
    
    # Test 3: Simple parser (without LLM)
    print("\n3. Testing simple parser...")
    try:
        from globule.parser_engine import SimpleParser
        
        parser = SimpleParser()
        result = await parser.parse_text("Meeting with Sarah about budget cuts")
        
        print("✓ Simple parser test passed")
        print(f"  - Category: {result.category}")
        print(f"  - Domain: {result.domain}")
        print(f"  - Sentiment: {result.sentiment}")
        print(f"  - Keywords: {result.keywords[:3]}")  # First 3 keywords
        
    except Exception as e:
        print(f"✗ Simple parser test failed: {e}")
        return False
    
    # Test 4: CLI module import
    print("\n4. Testing CLI module...")
    try:
        from globule.cli import cli
        print("✓ CLI module imported successfully")
    except Exception as e:
        print(f"✗ CLI module import failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 All basic tests passed!")
    print("\nNext steps:")
    print("1. Install dependencies: poetry install")
    print("2. Test with Ollama: globule add 'test thought'")
    print("3. Search thoughts: globule search 'test'")
    print("4. View today's thoughts: globule today")
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_basic_functionality())
    sys.exit(0 if success else 1)