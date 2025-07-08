"""Glass Engine Test Hooks System for Setup/Teardown and Failure Simulation."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

from .glass import TestContext, MockParser, MockEmbedder


class TestHook(Protocol):
    """Protocol for test hooks."""
    
    async def execute(self, context: TestContext) -> None:
        """Execute the hook with the given test context."""
        ...


class SetupHook:
    """Base class for setup hooks."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"glass.hooks.{name}")
    
    async def execute(self, context: TestContext) -> None:
        """Execute the setup hook."""
        context.trace_manager.log_trace(f"Executing setup hook: {self.name}")
        await self._execute_hook(context)
        context.trace_manager.log_trace(f"Setup hook completed: {self.name}")
    
    async def _execute_hook(self, context: TestContext) -> None:
        """Override this method to implement hook logic."""
        pass


class TeardownHook:
    """Base class for teardown hooks."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"glass.hooks.{name}")
    
    async def execute(self, context: TestContext) -> None:
        """Execute the teardown hook."""
        context.trace_manager.log_trace(f"Executing teardown hook: {self.name}")
        await self._execute_hook(context)
        context.trace_manager.log_trace(f"Teardown hook completed: {self.name}")
    
    async def _execute_hook(self, context: TestContext) -> None:
        """Override this method to implement hook logic."""
        pass


# Specific Hook Implementations
class MockParserFailureHook(SetupHook):
    """Hook to simulate parser failure."""
    
    def __init__(self):
        super().__init__("mock_parser_failure")
    
    async def _execute_hook(self, context: TestContext) -> None:
        """Set up a failing parser mock."""
        failing_parser = MockParser(should_fail=True)
        context.mock_registry.register_mock("parser", failing_parser)
        context.trace_manager.log_trace("Registered failing parser mock")


class MockEmbedderFailureHook(SetupHook):
    """Hook to simulate embedder failure."""
    
    def __init__(self):
        super().__init__("mock_embedder_failure")
    
    async def _execute_hook(self, context: TestContext) -> None:
        """Set up a failing embedder mock."""
        class FailingEmbedder:
            async def embed_text(self, text: str):
                raise RuntimeError("Simulated embedder failure")
        
        failing_embedder = FailingEmbedder()
        context.mock_registry.register_mock("embedder", failing_embedder)
        context.trace_manager.log_trace("Registered failing embedder mock")


class MockDatabaseFullHook(SetupHook):
    """Hook to simulate database full condition."""
    
    def __init__(self):
        super().__init__("mock_database_full")
    
    async def _execute_hook(self, context: TestContext) -> None:
        """Set up a database mock that simulates disk full."""
        class FullDatabaseMock:
            async def store_globule(self, globule):
                raise RuntimeError("Database full: No space left on device")
            
            async def retrieve_by_id(self, globule_id):
                return None
            
            async def search_semantic(self, query_embedding, limit=10):
                return []
            
            async def search_temporal(self, start_date, end_date):
                return []
        
        full_db_mock = FullDatabaseMock()
        context.mock_registry.register_mock("storage", full_db_mock)
        context.trace_manager.log_trace("Registered full database mock")


class NetworkTimeoutHook(SetupHook):
    """Hook to simulate network timeout conditions."""
    
    def __init__(self, timeout_seconds: int = 1):
        super().__init__("network_timeout")
        self.timeout_seconds = timeout_seconds
    
    async def _execute_hook(self, context: TestContext) -> None:
        """Set up components that simulate network timeouts."""
        class TimeoutParser:
            async def parse_text(self, text: str):
                await asyncio.sleep(self.timeout_seconds + 1)  # Exceed timeout
                raise asyncio.TimeoutError("Network timeout")
        
        timeout_parser = TimeoutParser()
        context.mock_registry.register_mock("parser", timeout_parser)
        context.trace_manager.log_trace(f"Registered timeout parser mock ({self.timeout_seconds}s)")


class TestDataSetupHook(SetupHook):
    """Hook to set up test data in the database."""
    
    def __init__(self, test_data_file: str):
        super().__init__("test_data_setup")
        self.test_data_file = test_data_file
    
    async def _execute_hook(self, context: TestContext) -> None:
        """Load test data from file into the test database."""
        test_data_path = Path(f"tests/glass_engine/fixtures/{self.test_data_file}")
        
        if not test_data_path.exists():
            context.trace_manager.log_trace(f"Test data file not found: {test_data_path}")
            return
        
        with open(test_data_path, 'r') as f:
            test_data = json.load(f)
        
        # Load test data into the database
        # This would require extending the storage interface
        # For now, we'll just log the action
        context.trace_manager.log_trace(f"Loaded test data from {self.test_data_file}")
        context.trace_manager.log_trace(f"Test data contains {len(test_data)} items")


class CleanupDatabaseHook(TeardownHook):
    """Hook to clean up test database after test completion."""
    
    def __init__(self):
        super().__init__("cleanup_database")
    
    async def _execute_hook(self, context: TestContext) -> None:
        """Clean up the test database."""
        # The database cleanup is already handled by TestContext.__aexit__
        # But we can add additional cleanup logic here if needed
        context.trace_manager.log_trace("Database cleanup completed")


class CollectArtifactsHook(TeardownHook):
    """Hook to collect additional debug artifacts."""
    
    def __init__(self):
        super().__init__("collect_artifacts")
    
    async def _execute_hook(self, context: TestContext) -> None:
        """Collect additional debug artifacts."""
        artifacts_dir = Path(context.artifacts_dir)
        
        # Save trace entries
        trace_file = artifacts_dir / "trace_entries.json"
        with open(trace_file, 'w') as f:
            json.dump(context.trace_manager.get_trace_entries(), f, indent=2)
        
        # Save test context metadata
        context_file = artifacts_dir / "test_context.json"
        context_metadata = {
            "trace_id": context.trace_id,
            "test_case_id": context.test_case_id,
            "mode": context.mode,
            "start_time": context.start_time.isoformat(),
            "temp_db_path": context.temp_db_path,
            "temp_config_path": context.temp_config_path
        }
        
        with open(context_file, 'w') as f:
            json.dump(context_metadata, f, indent=2)
        
        context.trace_manager.log_trace("Additional artifacts collected")


class LogLevelHook(SetupHook):
    """Hook to adjust logging level for test execution."""
    
    def __init__(self, log_level: str = "DEBUG"):
        super().__init__("log_level")
        self.log_level = log_level
    
    async def _execute_hook(self, context: TestContext) -> None:
        """Set logging level for test execution."""
        # Store original log level for restoration
        original_level = logging.getLogger().level
        context.resource_manager.register_cleanup_task(
            lambda: logging.getLogger().setLevel(original_level)
        )
        
        # Set new log level
        logging.getLogger().setLevel(getattr(logging, self.log_level))
        context.trace_manager.log_trace(f"Log level set to {self.log_level}")


# Hook Registry
class HookRegistry:
    """Registry for managing test hooks."""
    
    def __init__(self):
        self.setup_hooks: Dict[str, SetupHook] = {}
        self.teardown_hooks: Dict[str, TeardownHook] = {}
        self._register_default_hooks()
    
    def _register_default_hooks(self):
        """Register default hooks."""
        # Setup hooks
        self.setup_hooks["mock_parser_failure"] = MockParserFailureHook()
        self.setup_hooks["mock_embedder_failure"] = MockEmbedderFailureHook()
        self.setup_hooks["mock_database_full"] = MockDatabaseFullHook()
        self.setup_hooks["network_timeout"] = NetworkTimeoutHook()
        self.setup_hooks["log_level_debug"] = LogLevelHook("DEBUG")
        self.setup_hooks["log_level_info"] = LogLevelHook("INFO")
        
        # Teardown hooks
        self.teardown_hooks["cleanup_database"] = CleanupDatabaseHook()
        self.teardown_hooks["collect_artifacts"] = CollectArtifactsHook()
    
    def register_setup_hook(self, name: str, hook: SetupHook):
        """Register a setup hook."""
        self.setup_hooks[name] = hook
    
    def register_teardown_hook(self, name: str, hook: TeardownHook):
        """Register a teardown hook."""
        self.teardown_hooks[name] = hook
    
    def get_setup_hook(self, name: str) -> Optional[SetupHook]:
        """Get a setup hook by name."""
        return self.setup_hooks.get(name)
    
    def get_teardown_hook(self, name: str) -> Optional[TeardownHook]:
        """Get a teardown hook by name."""
        return self.teardown_hooks.get(name)
    
    def list_setup_hooks(self) -> list[str]:
        """List all available setup hooks."""
        return list(self.setup_hooks.keys())
    
    def list_teardown_hooks(self) -> list[str]:
        """List all available teardown hooks."""
        return list(self.teardown_hooks.keys())


# Hook execution utilities
async def execute_setup_hooks(hook_names: list[str], context: TestContext, 
                            registry: HookRegistry) -> None:
    """Execute a list of setup hooks."""
    for hook_name in hook_names:
        hook = registry.get_setup_hook(hook_name)
        if hook:
            await hook.execute(context)
        else:
            context.trace_manager.log_trace(f"Setup hook not found: {hook_name}")


async def execute_teardown_hooks(hook_names: list[str], context: TestContext, 
                               registry: HookRegistry) -> None:
    """Execute a list of teardown hooks."""
    # Execute teardown hooks in reverse order
    for hook_name in reversed(hook_names):
        hook = registry.get_teardown_hook(hook_name)
        if hook:
            try:
                await hook.execute(context)
            except Exception as e:
                context.trace_manager.log_trace(f"Teardown hook failed: {hook_name} - {e}")
        else:
            context.trace_manager.log_trace(f"Teardown hook not found: {hook_name}")


# Global hook registry instance
hook_registry = HookRegistry()