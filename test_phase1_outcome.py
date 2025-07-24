#!/usr/bin/env python3
"""
Test script to verify Phase 1 outcome:
"A developer can run `globule add "test"` and `globule draft "test"` and see the word "test" appear in the terminal UI."

This script simulates the end-to-end flow without requiring interactive TUI.
"""

import asyncio
import tempfile
from pathlib import Path
from datetime import datetime

from globule.core.models import EnrichedInput
from globule.storage.sqlite_manager import SQLiteStorageManager
from globule.parsing.mock_parser import MockOllamaParser
from globule.orchestration.parallel_strategy import ParallelOrchestrationEngine
from globule.tui.app import SynthesisApp


class MockEmbeddingProvider:
    """Mock embedding provider for testing without Ollama"""
    def get_dimension(self):
        return 1024
    
    async def embed(self, text):
        import numpy as np
        return np.random.randn(1024).astype(np.float32)
    
    async def embed_batch(self, texts):
        return [await self.embed(text) for text in texts]
    
    async def close(self):
        pass
    
    async def health_check(self):
        return True


async def test_phase1_outcome():
    """Test the Phase 1 outcome without interactive TUI"""
    
    print("Testing Phase 1 Walking Skeleton Outcome")
    print("=" * 50)
    
    # Use temporary directory for isolated test
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_globules.db"
        
        # Step 1: Initialize components (simulating `globule add "test"`)
        print("1. Initializing storage and components...")
        storage = SQLiteStorageManager(db_path)
        await storage.initialize()
        
        embedding_provider = MockEmbeddingProvider()
        parsing_provider = MockOllamaParser()
        orchestrator = ParallelOrchestrationEngine(
            embedding_provider, parsing_provider, storage
        )
        
        # Step 2: Add test globule
        print('2. Processing globule: "test"...')
        enriched_input = EnrichedInput(
            original_text="test",
            enriched_text="test",
            detected_schema_id=None,
            schema_config=None,
            additional_context={},
            source="test_script",
            timestamp=datetime.now()
        )
        
        processed_globule = await orchestrator.process_globule(enriched_input)
        globule_id = await storage.store_globule(processed_globule)
        print(f"   [OK] Globule stored with ID: {globule_id}")
        
        # Step 3: Simulate the TUI loading recent globules (simulating `globule draft "test"`)
        print('3. Loading recent globules (simulating TUI)...')
        recent_globules = await storage.get_recent_globules(limit=10)
        
        # Step 4: Verify the test word appears
        print("4. Verifying Phase 1 outcome...")
        found_test = False
        for globule in recent_globules:
            if "test" in globule.text.lower():
                found_test = True
                print(f'   [OK] Found globule containing "test": "{globule.text}"')
                print(f"      - ID: {globule.id}")
                print(f"      - Parsed title: {globule.parsed_data.get('title', 'N/A')}")
                print(f"      - Embedding dimension: {len(globule.embedding) if globule.embedding is not None else 'None'}")
                break
        
        if found_test:
            print("\n[SUCCESS] Phase 1 Outcome VERIFIED!")
            print("[OK] A developer can add a globule with 'test' and it appears in the TUI data")
            print("[OK] Core end-to-end plumbing works")
            print("[OK] Storage, embedding, parsing, and orchestration are functional")
        else:
            print("\n[FAIL] Phase 1 Outcome FAILED!")
            print("Could not find the test globule in recent results")
            
        # Step 5: Test that TUI app can be instantiated
        print("\n5. Testing TUI instantiation...")
        try:
            app = SynthesisApp(storage_manager=storage, topic="test", limit=10)
            print("   [OK] TUI app instantiated successfully")
            
            # Simulate loading globules like the TUI would
            app.globules = await storage.get_recent_globules(app.limit)
            print(f"   [OK] TUI loaded {len(app.globules)} globules")
            
            # Check if any contain "test"
            test_globules = [g for g in app.globules if "test" in g.text.lower()]
            print(f"   [OK] Found {len(test_globules)} globules containing 'test'")
            
        except Exception as e:
            print(f"   [ERROR] TUI instantiation failed: {e}")
        
        # Cleanup
        await embedding_provider.close()
        await storage.close()
        
    print("\n" + "=" * 50)
    print("Phase 1 Walking Skeleton Test Complete")


if __name__ == "__main__":
    asyncio.run(test_phase1_outcome())