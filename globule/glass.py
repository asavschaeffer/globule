"""Glass Engine - UDE Framework for Globule Testing."""

import asyncio
import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import aiosqlite
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .config import Config, load_config
from .embedding_engine import Embedder, create_embedder
from .parser_engine import Parser, ParsedResult, create_parser
from .storage import Globule, SQLiteStorage, Storage


class TestStatus(Enum):
    """Status of test execution."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class ModeSettings:
    """Settings for a specific test mode."""
    enabled: bool
    priority: str  # essential, high, medium, low
    narration_level: str = "basic"  # basic, detailed, verbose


@dataclass
class ModeConfig:
    """Configuration for tutorial and showcase modes."""
    tutorial: ModeSettings
    showcase: ModeSettings


@dataclass
class TestStep:
    """Individual test step definition."""
    action: str  # cli_command, api_call, state_check
    command: Optional[str] = None
    args: Optional[List[str]] = None
    expected_output_contains: Optional[List[str]] = None
    expected_output_not_contains: Optional[List[str]] = None
    timeout: int = 10
    retry_on_failure: bool = False
    setup_hook: Optional[str] = None
    teardown_hook: Optional[str] = None


@dataclass
class TestCase:
    """Complete test case definition."""
    id: str
    name: str
    description: str
    category: str
    mode_config: ModeConfig
    steps: List[TestStep]
    assertions: Dict[str, List[Dict[str, Any]]]  # mode -> assertion configs
    edge_cases: Optional[List[Dict[str, Any]]] = None
    setup_hooks: Optional[List[str]] = None
    teardown_hooks: Optional[List[str]] = None
    timeout: int = 30
    retry_count: int = 0


@dataclass
class StepResult:
    """Result of executing a test step."""
    step_index: int
    status: TestStatus
    duration: float
    output: str
    error: Optional[str] = None


@dataclass
class AssertionResult:
    """Result of assertion validation."""
    assertion: Optional['Assertion']
    status: TestStatus
    message: str
    duration: float
    actual_value: Optional[str] = None
    expected_value: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class TestResult:
    """Complete test execution result."""
    test_case_id: str
    status: TestStatus
    duration: float
    trace_id: str
    start_time: datetime
    end_time: datetime
    step_results: List[StepResult]
    assertion_results: List[AssertionResult]
    artifacts_path: str
    error_message: Optional[str] = None
    failure_details: Optional[str] = None


@dataclass
class TestRunResult:
    """Result of running multiple tests."""
    run_id: str
    mode: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    error_tests: int
    skipped_tests: int
    total_duration: float
    test_results: List[TestResult]
    artifacts_dir: str


# Assertion System (Polymorphic)
class Assertion(ABC):
    """Base class for all assertions."""
    
    def __init__(self, description: str, timeout: int = 5):
        self.description = description
        self.timeout = timeout
    
    @abstractmethod
    async def execute(self, context: 'TestContext') -> AssertionResult:
        """Execute the assertion and return result."""
        pass
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {self.description}"


class CliOutputAssertion(Assertion):
    """Assert CLI output contains specific text."""
    
    def __init__(self, description: str, contains: List[str], 
                 not_contains: Optional[List[str]] = None, **kwargs):
        super().__init__(description, **kwargs)
        self.contains = contains
        self.not_contains = not_contains or []
    
    async def execute(self, context: 'TestContext') -> AssertionResult:
        start_time = time.time()
        
        try:
            last_output = context.state_capture.get_last_command_output()
            
            # Check required content
            for expected in self.contains:
                if expected not in last_output:
                    return AssertionResult(
                        assertion=self,
                        status=TestStatus.FAILED,
                        message=f"Expected '{expected}' not found in CLI output",
                        duration=time.time() - start_time,
                        actual_value=last_output[:200] + "..." if len(last_output) > 200 else last_output,
                        expected_value=expected
                    )
            
            # Check forbidden content
            for forbidden in self.not_contains:
                if forbidden in last_output:
                    return AssertionResult(
                        assertion=self,
                        status=TestStatus.FAILED,
                        message=f"Forbidden text '{forbidden}' found in CLI output",
                        duration=time.time() - start_time,
                        actual_value=last_output,
                        expected_value=f"NOT {forbidden}"
                    )
            
            return AssertionResult(
                assertion=self,
                status=TestStatus.PASSED,
                message="CLI output assertion passed",
                duration=time.time() - start_time
            )
            
        except Exception as e:
            return AssertionResult(
                assertion=self,
                status=TestStatus.ERROR,
                message=f"CLI output assertion failed: {str(e)}",
                duration=time.time() - start_time
            )


class DatabaseRecordAssertion(Assertion):
    """Assert database record exists with specific conditions."""
    
    def __init__(self, description: str, table: str, condition: str, 
                 expected_count: Optional[int] = None, **kwargs):
        super().__init__(description, **kwargs)
        self.table = table
        self.condition = condition
        self.expected_count = expected_count
    
    async def execute(self, context: 'TestContext') -> AssertionResult:
        start_time = time.time()
        
        try:
            # Build query
            query = f"SELECT COUNT(*) FROM {self.table} WHERE {self.condition}"
            
            # Execute query
            async with aiosqlite.connect(context.temp_db_path) as db:
                cursor = await db.execute(query)
                row = await cursor.fetchone()
                actual_count = row[0] if row else 0
            
            # Validate count
            if self.expected_count is not None:
                if actual_count != self.expected_count:
                    return AssertionResult(
                        assertion=self,
                        status=TestStatus.FAILED,
                        message=f"Expected {self.expected_count} records, found {actual_count}",
                        duration=time.time() - start_time,
                        actual_value=str(actual_count),
                        expected_value=str(self.expected_count),
                        details={"query": query}
                    )
            else:
                # Default: expect at least one record
                if actual_count == 0:
                    return AssertionResult(
                        assertion=self,
                        status=TestStatus.FAILED,
                        message=f"No records found matching condition: {self.condition}",
                        duration=time.time() - start_time,
                        actual_value="0",
                        expected_value="> 0",
                        details={"query": query}
                    )
            
            return AssertionResult(
                assertion=self,
                status=TestStatus.PASSED,
                message=f"Database assertion passed: found {actual_count} records",
                duration=time.time() - start_time,
                details={"query": query, "count": actual_count}
            )
            
        except Exception as e:
            return AssertionResult(
                assertion=self,
                status=TestStatus.ERROR,
                message=f"Database assertion failed: {str(e)}",
                duration=time.time() - start_time
            )


class LogEntryAssertion(Assertion):
    """Assert log contains specific entries."""
    
    def __init__(self, description: str, contains: List[str], 
                 log_level: Optional[str] = None, **kwargs):
        super().__init__(description, **kwargs)
        self.contains = contains
        self.log_level = log_level
    
    async def execute(self, context: 'TestContext') -> AssertionResult:
        start_time = time.time()
        
        try:
            log_content = context.state_capture.get_log_content()
            
            # Filter by log level if specified
            if self.log_level:
                log_lines = [line for line in log_content.split('\n') 
                           if self.log_level.upper() in line]
                search_content = '\n'.join(log_lines)
            else:
                search_content = log_content
            
            # Check for required content
            for expected in self.contains:
                if expected not in search_content:
                    return AssertionResult(
                        assertion=self,
                        status=TestStatus.FAILED,
                        message=f"Expected log entry '{expected}' not found",
                        duration=time.time() - start_time,
                        actual_value=search_content[-500:] if len(search_content) > 500 else search_content,
                        expected_value=expected
                    )
            
            return AssertionResult(
                assertion=self,
                status=TestStatus.PASSED,
                message="Log entry assertion passed",
                duration=time.time() - start_time
            )
            
        except Exception as e:
            return AssertionResult(
                assertion=self,
                status=TestStatus.ERROR,
                message=f"Log assertion failed: {str(e)}",
                duration=time.time() - start_time
            )


class EmbeddingAssertion(Assertion):
    """Assert embedding was generated with correct properties."""
    
    def __init__(self, description: str, expected_shape: Optional[Tuple[int, ...]] = None, 
                 not_null: bool = True, **kwargs):
        super().__init__(description, **kwargs)
        self.expected_shape = expected_shape
        self.not_null = not_null
    
    async def execute(self, context: 'TestContext') -> AssertionResult:
        start_time = time.time()
        
        try:
            # Get the last created globule
            last_globule = context.state_capture.get_last_created_globule()
            
            if not last_globule:
                return AssertionResult(
                    assertion=self,
                    status=TestStatus.FAILED,
                    message="No globule found to check embedding",
                    duration=time.time() - start_time
                )
            
            embedding = last_globule.embedding
            
            if self.not_null and embedding is None:
                return AssertionResult(
                    assertion=self,
                    status=TestStatus.FAILED,
                    message="Expected embedding to be generated, but it was None",
                    duration=time.time() - start_time,
                    actual_value="None",
                    expected_value="not None"
                )
            
            if self.expected_shape and embedding is not None:
                actual_shape = embedding.shape
                if actual_shape != self.expected_shape:
                    return AssertionResult(
                        assertion=self,
                        status=TestStatus.FAILED,
                        message=f"Embedding shape mismatch",
                        duration=time.time() - start_time,
                        actual_value=str(actual_shape),
                        expected_value=str(self.expected_shape)
                    )
            
            return AssertionResult(
                assertion=self,
                status=TestStatus.PASSED,
                message="Embedding assertion passed",
                duration=time.time() - start_time,
                details={"embedding_shape": str(embedding.shape) if embedding is not None else None}
            )
            
        except Exception as e:
            return AssertionResult(
                assertion=self,
                status=TestStatus.ERROR,
                message=f"Embedding assertion failed: {str(e)}",
                duration=time.time() - start_time
            )


class AssertionFactory:
    """Factory for creating assertion instances from configuration."""
    
    ASSERTION_TYPES = {
        'cli_output': CliOutputAssertion,
        'database_record': DatabaseRecordAssertion,
        'log_entry': LogEntryAssertion,
        'embedding': EmbeddingAssertion,
    }
    
    @classmethod
    def create_assertion(cls, assertion_config: Dict[str, Any]) -> Assertion:
        """Create assertion instance from configuration."""
        assertion_type = assertion_config.get('type')
        
        if assertion_type not in cls.ASSERTION_TYPES:
            raise ValueError(f"Unknown assertion type: {assertion_type}")
        
        assertion_class = cls.ASSERTION_TYPES[assertion_type]
        
        # Remove 'type' from config and pass remaining as kwargs
        config_copy = assertion_config.copy()
        config_copy.pop('type')
        
        return assertion_class(**config_copy)


def generate_trace_id() -> str:
    """Generate a unique trace ID for test execution."""
    return str(uuid.uuid4())[:8]


def generate_run_id() -> str:
    """Generate a unique run ID for test run."""
    return f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"


# Mock System for Dependency Injection
class MockRegistry:
    """Clean dependency injection-based mocking."""
    
    def __init__(self):
        self.mocks: Dict[str, Any] = {}
        self.active_mocks: Set[str] = set()
    
    def register_mock(self, component_name: str, mock_instance: Any):
        """Register a mock instance for a component."""
        self.mocks[component_name] = mock_instance
        self.active_mocks.add(component_name)
    
    def has_mock(self, component_name: str) -> bool:
        """Check if a mock is registered for a component."""
        return component_name in self.active_mocks
    
    def get_mock(self, component_name: str) -> Any:
        """Get mock instance for a component."""
        if not self.has_mock(component_name):
            raise ValueError(f"No mock registered for component: {component_name}")
        return self.mocks[component_name]
    
    def clear_mocks(self):
        """Clear all registered mocks."""
        self.mocks.clear()
        self.active_mocks.clear()


class ComponentFactory:
    """Factory for creating components with dependency injection."""
    
    def __init__(self, config: Config, mock_registry: Optional[MockRegistry] = None):
        self.config = config
        self.mock_registry = mock_registry or MockRegistry()
    
    async def create_parser(self) -> Parser:
        """Create parser with optional mocking."""
        if self.mock_registry.has_mock('parser'):
            return self.mock_registry.get_mock('parser')
        return await create_parser(use_ollama=self.config.llm_provider == "local")
    
    async def create_embedder(self) -> Embedder:
        """Create embedder with optional mocking."""
        if self.mock_registry.has_mock('embedder'):
            return self.mock_registry.get_mock('embedder')
        return await create_embedder(use_ollama=self.config.embedding_provider == "local")
    
    def create_storage(self) -> Storage:
        """Create storage with optional mocking."""
        if self.mock_registry.has_mock('storage'):
            return self.mock_registry.get_mock('storage')
        return SQLiteStorage(self.config.db_path)


# Example Mock Implementations
class MockParser:
    """Mock parser for testing failure scenarios."""
    
    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
    
    async def parse_text(self, text: str) -> ParsedResult:
        if self.should_fail:
            raise RuntimeError("Simulated parser failure")
        return ParsedResult(
            domain="test",
            category="note",
            sentiment="neutral"
        )


class MockEmbedder:
    """Mock embedder for testing scenarios."""
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
    
    async def embed_text(self, text: str):
        # Return deterministic embedding for testing
        import numpy as np
        return np.random.RandomState(hash(text) % 2**32).random(self.embedding_dim)


# State Capture System
class StateCapture:
    """Minimal, assertion-driven state capture."""
    
    def __init__(self, artifacts_dir: str):
        self.artifacts_dir = Path(artifacts_dir)
        self.command_outputs: List[str] = []
        self.log_buffer: List[str] = []
        self.database_snapshots: Dict[str, Any] = {}
        self.last_created_globule: Optional[Globule] = None
        self.required_db_queries: Set[str] = set()
        self.required_log_levels: Set[str] = set()
        self.needs_embedding_check: bool = False
    
    def register_assertion_requirements(self, assertions: List[Assertion]):
        """Register what state needs to be captured based on assertions."""
        self.required_db_queries = set()
        self.required_log_levels = set()
        self.needs_embedding_check = False
        
        for assertion in assertions:
            if isinstance(assertion, DatabaseRecordAssertion):
                # Build the exact query needed for this assertion
                query = f"SELECT * FROM {assertion.table} WHERE {assertion.condition}"
                self.required_db_queries.add(query)
            
            elif isinstance(assertion, LogEntryAssertion):
                if assertion.log_level:
                    self.required_log_levels.add(assertion.log_level)
            
            elif isinstance(assertion, EmbeddingAssertion):
                self.needs_embedding_check = True
    
    async def capture_initial_state(self):
        """Capture initial targeted state."""
        # Only capture state if we have specific requirements
        pass
    
    async def capture_final_state(self):
        """Capture final targeted state."""
        # Only capture state if we have specific requirements
        pass
    
    async def capture_targeted_state(self, context: 'TestContext'):
        """Capture only the state required by assertions."""
        # Capture database state for specific queries
        if self.required_db_queries:
            await self._capture_database_queries(context)
        
        # Capture log entries for specific levels
        if self.required_log_levels:
            await self._capture_log_entries()
        
        # Capture embedding state if needed
        if self.needs_embedding_check:
            await self._capture_embedding_state(context)
    
    async def _capture_database_queries(self, context: 'TestContext'):
        """Execute and capture results of specific database queries."""
        async with aiosqlite.connect(context.temp_db_path) as db:
            for query in self.required_db_queries:
                cursor = await db.execute(query)
                rows = await cursor.fetchall()
                
                # Store query results
                self.database_snapshots[query] = {
                    'timestamp': datetime.now().isoformat(),
                    'row_count': len(rows),
                    'rows': rows[:10]  # Store first 10 rows for debugging
                }
    
    async def _capture_log_entries(self):
        """Capture log entries for specific levels."""
        log_file = Path("globule.log")
        if log_file.exists():
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            # Filter by required log levels
            for level in self.required_log_levels:
                filtered_lines = [line for line in lines if level in line]
                self.log_buffer.extend(filtered_lines)
    
    async def _capture_embedding_state(self, context: 'TestContext'):
        """Capture embedding state for the last created globule."""
        # Get the most recent globule
        async with aiosqlite.connect(context.temp_db_path) as db:
            cursor = await db.execute(
                "SELECT * FROM globules ORDER BY created_at DESC LIMIT 1"
            )
            row = await cursor.fetchone()
            
            if row:
                storage = context.component_factory.create_storage()
                self.last_created_globule = storage._row_to_globule(row)
    
    def log_command_execution(self, command: str, args: List[str], 
                             output: str, return_code: int):
        """Log command execution details."""
        self.command_outputs.append(output)
        
        # Save to artifacts
        command_log = {
            'timestamp': datetime.now().isoformat(),
            'command': command,
            'args': args,
            'output': output,
            'return_code': return_code
        }
        
        command_file = self.artifacts_dir / "command_history.json"
        
        # Ensure directory exists
        command_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Append to command history
        if command_file.exists():
            with open(command_file, 'r') as f:
                history = json.load(f)
        else:
            history = []
        
        history.append(command_log)
        
        with open(command_file, 'w') as f:
            json.dump(history, f, indent=2)
    
    def get_last_command_output(self) -> str:
        """Get output from the last executed command."""
        return self.command_outputs[-1] if self.command_outputs else ""
    
    def get_log_content(self) -> str:
        """Get captured log content."""
        return '\n'.join(self.log_buffer)
    
    def get_last_created_globule(self) -> Optional[Globule]:
        """Get the last created globule."""
        return self.last_created_globule


# Test Context System
class TestContext:
    """Isolated test environment manager."""
    
    def __init__(self, trace_id: str, test_case_id: str, mode: str):
        self.trace_id = trace_id
        self.test_case_id = test_case_id
        self.mode = mode
        self.temp_db_path = f"temp_globule_test_{trace_id}.db"
        self.temp_config_path = f"temp_config_{trace_id}.yaml"
        self.artifacts_dir = f"test_runs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{mode}/{test_case_id}"
        self.start_time = datetime.now()
        
        # Initialize sub-components
        self.mock_registry = MockRegistry()
        self.component_factory = None  # Will be set in __aenter__
        self.state_capture = StateCapture(self.artifacts_dir)
        self.resource_manager = ResourceManager()
        self.trace_manager = TraceManager(trace_id)
        self.config = None
    
    async def __aenter__(self):
        """Setup isolated test environment."""
        # Create artifacts directory
        Path(self.artifacts_dir).mkdir(parents=True, exist_ok=True)
        
        # Setup temporary database
        await self._setup_temp_database()
        
        # Setup temporary configuration
        self.config = await self._setup_temp_config()
        
        # Initialize dependency injection
        self.component_factory = ComponentFactory(self.config, self.mock_registry)
        
        # Initialize tracing
        await self.trace_manager.initialize()
        
        # Capture initial state
        await self.state_capture.capture_initial_state()
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup test environment."""
        # Capture final state
        await self.state_capture.capture_final_state()
        
        # Clean up resources
        await self.resource_manager.cleanup()
        
        # Remove temporary files
        Path(self.temp_db_path).unlink(missing_ok=True)
        Path(self.temp_config_path).unlink(missing_ok=True)
        
        # Close tracing
        await self.trace_manager.close()
    
    async def _setup_temp_database(self):
        """Create isolated database for test."""
        # Copy schema from main database or create fresh
        storage = SQLiteStorage(self.temp_db_path)
        await storage._init_db()
    
    async def _setup_temp_config(self) -> Config:
        """Create sandboxed configuration."""
        # Load base config
        base_config = load_config()
        
        # Override with test-specific settings
        test_config = base_config.model_copy()
        test_config.db_path = self.temp_db_path
        
        # Save to temporary file
        test_config.save_to_file(self.temp_config_path)
        
        return test_config
    
    async def setup_for_test_case(self, test_case: TestCase, mode: str):
        """Setup context for specific test case."""
        # Get assertions for this mode
        assertions_config = test_case.assertions.get(mode, [])
        
        # Create assertion objects
        assertions = []
        for assertion_config in assertions_config:
            assertion = AssertionFactory.create_assertion(assertion_config)
            assertions.append(assertion)
        
        # Register state capture requirements
        self.state_capture.register_assertion_requirements(assertions)
        
        # Capture initial targeted state
        await self.state_capture.capture_targeted_state(self)


class ResourceManager:
    """Manager for test resources and cleanup."""
    
    def __init__(self):
        self.resources = []
        self.cleanup_tasks = []
    
    def register_resource(self, resource: Any):
        """Register a resource for cleanup."""
        self.resources.append(resource)
    
    def register_cleanup_task(self, task: callable):
        """Register a cleanup task."""
        self.cleanup_tasks.append(task)
    
    async def cleanup(self):
        """Clean up all registered resources."""
        # Run cleanup tasks
        for task in reversed(self.cleanup_tasks):
            try:
                if asyncio.iscoroutinefunction(task):
                    await task()
                else:
                    task()
            except Exception as e:
                print(f"Cleanup task failed: {e}")
        
        # Clean up resources
        for resource in reversed(self.resources):
            try:
                if hasattr(resource, 'close'):
                    if asyncio.iscoroutinefunction(resource.close):
                        await resource.close()
                    else:
                        resource.close()
            except Exception as e:
                print(f"Resource cleanup failed: {e}")
        
        self.resources.clear()
        self.cleanup_tasks.clear()


class TraceManager:
    """Manager for test execution tracing."""
    
    def __init__(self, trace_id: str):
        self.trace_id = trace_id
        self.trace_entries = []
    
    async def initialize(self):
        """Initialize tracing."""
        self.log_trace("Glass Engine initialized", {"trace_id": self.trace_id})
    
    async def close(self):
        """Close tracing."""
        self.log_trace("Glass Engine closed", {"trace_id": self.trace_id})
    
    def log_trace(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Log a trace entry."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "trace_id": self.trace_id,
            "message": message,
            "details": details or {}
        }
        self.trace_entries.append(entry)
        
        # Also log to console in showcase mode
        print(f"[GLASS ENGINE][trace_id: {self.trace_id}] {message}")
    
    def get_trace_entries(self) -> List[Dict[str, Any]]:
        """Get all trace entries."""
        return self.trace_entries.copy()