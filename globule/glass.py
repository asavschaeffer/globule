"""Glass Engine - UDE Framework for Globule Testing."""

import asyncio
import json
import os
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

# Import Glass Engine components
try:
    from .glass_hooks import HookRegistry, get_hook_registry
    from .glass_errors import ErrorHandler, ErrorRecovery, error_handler, error_recovery
    from .glass_reports import ReportGenerator, save_test_reports
    GLASS_COMPONENTS_AVAILABLE = True
except ImportError:
    GLASS_COMPONENTS_AVAILABLE = False
    # Create stub classes for missing components
    class HookRegistry:
        def __init__(self): pass
        def execute_setup_hooks(self, *args, **kwargs): pass
        def execute_teardown_hooks(self, *args, **kwargs): pass
    
    class ErrorHandler:
        def __init__(self): pass
        def handle_error(self, error, **kwargs): return None
        def should_retry(self, error, attempt): return False
    
    class ErrorRecovery:
        def __init__(self, handler): pass
        async def execute_with_retry(self, func, *args, **kwargs): return await func(*args, **kwargs)
    
    def get_hook_registry(): return HookRegistry()
    error_handler = ErrorHandler()
    error_recovery = ErrorRecovery(error_handler)


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


# Test Execution System
class CommandExecutor:
    """Executor for CLI commands with proper environment setup."""
    
    async def execute_cli_command(self, command: str, args: List[str], 
                                 context: TestContext) -> str:
        """Execute CLI command with proper environment setup."""
        # Set up environment with test context
        env = dict(os.environ)
        env['GLOBULE_DB_PATH'] = context.temp_db_path
        env['GLOBULE_CONFIG_PATH'] = context.temp_config_path
        env['GLASS_ENGINE_TRACE_ID'] = context.trace_id
        
        # Build command
        cmd = [command] + args
        
        context.trace_manager.log_trace(f"Executing command: {' '.join(cmd)}")
        
        # Execute with timeout
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), 
                timeout=30.0
            )
            
            output = stdout.decode() + stderr.decode()
            
            # Log execution details
            context.state_capture.log_command_execution(
                command, args, output, process.returncode
            )
            
            context.trace_manager.log_trace(
                f"Command completed with return code: {process.returncode}",
                {"output_length": len(output)}
            )
            
            return output
            
        except asyncio.TimeoutError:
            process.kill()
            raise asyncio.TimeoutError(f"Command '{command}' timed out")


class TestExecutor:
    """Executor for test steps."""
    
    def __init__(self):
        self.command_executor = CommandExecutor()
    
    async def execute_step(self, step: TestStep, context: TestContext, 
                          step_index: int) -> StepResult:
        """Execute a single test step."""
        start_time = time.time()
        
        context.trace_manager.log_trace(f"=== STEP {step_index + 1}: {step.action} ===")
        
        try:
            if step.action == "cli_command":
                output = await self.command_executor.execute_cli_command(
                    step.command, step.args, context
                )
            else:
                raise ValueError(f"Unknown action type: {step.action}")
            
            # Validate expected output
            if step.expected_output_contains:
                for expected in step.expected_output_contains:
                    if expected not in output:
                        return StepResult(
                            step_index=step_index,
                            status=TestStatus.FAILED,
                            duration=time.time() - start_time,
                            output=output,
                            error=f"Expected output '{expected}' not found"
                        )
            
            if step.expected_output_not_contains:
                for forbidden in step.expected_output_not_contains:
                    if forbidden in output:
                        return StepResult(
                            step_index=step_index,
                            status=TestStatus.FAILED,
                            duration=time.time() - start_time,
                            output=output,
                            error=f"Forbidden output '{forbidden}' found"
                        )
            
            return StepResult(
                step_index=step_index,
                status=TestStatus.PASSED,
                duration=time.time() - start_time,
                output=output
            )
            
        except asyncio.TimeoutError:
            return StepResult(
                step_index=step_index,
                status=TestStatus.TIMEOUT,
                duration=time.time() - start_time,
                output="",
                error="Step execution timed out"
            )
        except Exception as e:
            return StepResult(
                step_index=step_index,
                status=TestStatus.ERROR,
                duration=time.time() - start_time,
                output="",
                error=str(e)
            )


class AssertionValidator:
    """Validator for test assertions using polymorphic execution."""
    
    def __init__(self):
        self.assertion_factory = AssertionFactory()
    
    async def validate_assertions(self, assertion_configs: List[Dict[str, Any]], 
                                 context: TestContext) -> List[AssertionResult]:
        """Validate all assertions using polymorphic execution."""
        results = []
        
        context.trace_manager.log_trace("=== ASSERTIONS ===")
        
        for assertion_config in assertion_configs:
            try:
                assertion = self.assertion_factory.create_assertion(assertion_config)
                context.trace_manager.log_trace(f"Validating: {assertion}")
                
                result = await assertion.execute(context)
                results.append(result)
                
                status_symbol = "✓" if result.status == TestStatus.PASSED else "✗"
                context.trace_manager.log_trace(f"{status_symbol} {result.message}")
                
            except Exception as e:
                # Create error result for failed assertion creation
                error_result = AssertionResult(
                    assertion=None,
                    status=TestStatus.ERROR,
                    message=f"Failed to create assertion: {str(e)}",
                    duration=0.0
                )
                results.append(error_result)
                context.trace_manager.log_trace(f"✗ {error_result.message}")
        
        return results


# Test Case Loading System
class TestCaseLoader:
    """Load test cases from directory structure."""
    
    def __init__(self, test_cases_dir: str = "tests/glass_engine/test_cases"):
        self.test_cases_dir = Path(test_cases_dir)
        self.loaded_cases: Dict[str, TestCase] = {}
        self.categories: Dict[str, List[str]] = {}
    
    async def load_test_cases(self, filter_mode: str = None, 
                             test_id: str = None, 
                             category: str = None) -> List[TestCase]:
        """Load test cases with filtering."""
        if not self.loaded_cases:
            await self._load_all_test_cases()
        
        # Filter by specific test ID
        if test_id:
            if test_id not in self.loaded_cases:
                raise ValueError(f"Test case {test_id} not found")
            return [self.loaded_cases[test_id]]
        
        # Filter by category
        if category:
            if category not in self.categories:
                raise ValueError(f"Category {category} not found")
            test_ids = self.categories[category]
            filtered_cases = [self.loaded_cases[tid] for tid in test_ids]
        else:
            filtered_cases = list(self.loaded_cases.values())
        
        # Filter by mode
        if filter_mode:
            mode_filtered = []
            for test_case in filtered_cases:
                mode_config = getattr(test_case.mode_config, filter_mode, None)
                if mode_config and mode_config.enabled:
                    mode_filtered.append(test_case)
            filtered_cases = mode_filtered
        
        return filtered_cases
    
    async def _load_all_test_cases(self):
        """Load all test cases from directory structure."""
        if not self.test_cases_dir.exists():
            raise FileNotFoundError(f"Test cases directory not found: {self.test_cases_dir}")
        
        for yaml_file in self.test_cases_dir.rglob("*.yaml"):
            category = yaml_file.parent.name
            test_cases = await self._load_test_cases_from_file(yaml_file)
            
            for test_case in test_cases:
                self.loaded_cases[test_case.id] = test_case
                
                # Track categories
                if category not in self.categories:
                    self.categories[category] = []
                self.categories[category].append(test_case.id)
    
    async def _load_test_cases_from_file(self, yaml_file: Path) -> List[TestCase]:
        """Load test cases from a single YAML file."""
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        
        test_cases = []
        for case_data in data.get('test_cases', []):
            test_case = self._parse_test_case(case_data)
            test_cases.append(test_case)
        
        return test_cases
    
    def _parse_test_case(self, case_data: Dict[str, Any]) -> TestCase:
        """Parse test case data into TestCase object."""
        # Parse mode config
        mode_config_data = case_data.get('mode_config', {})
        mode_config = ModeConfig(
            tutorial=ModeSettings(**mode_config_data.get('tutorial', {})),
            showcase=ModeSettings(**mode_config_data.get('showcase', {}))
        )
        
        # Parse steps
        steps = []
        for step_data in case_data.get('steps', []):
            step = TestStep(**step_data)
            steps.append(step)
        
        return TestCase(
            id=case_data['id'],
            name=case_data['name'],
            description=case_data['description'],
            category=case_data.get('category', 'general'),
            mode_config=mode_config,
            steps=steps,
            assertions=case_data.get('assertions', {}),
            edge_cases=case_data.get('edge_cases'),
            setup_hooks=case_data.get('setup_hooks'),
            teardown_hooks=case_data.get('teardown_hooks'),
            timeout=case_data.get('timeout', 30),
            retry_count=case_data.get('retry_count', 0)
        )


# Main Glass Engine Orchestrator
class GlassOrchestrator:
    """Main orchestrator for Glass Engine test execution."""
    
    def __init__(self, config: Config):
        self.config = config
        self.test_loader = TestCaseLoader()
        self.test_executor = TestExecutor()
        self.assertion_validator = AssertionValidator()
        self.console = Console()
        
        # Initialize Glass Engine components
        self.hook_registry = get_hook_registry() if GLASS_COMPONENTS_AVAILABLE else HookRegistry()
        self.error_handler = error_handler
        self.error_recovery = error_recovery
        self.report_generator = ReportGenerator() if GLASS_COMPONENTS_AVAILABLE else None
        
    async def run(self, mode: str, test_id: Optional[str] = None, 
                  category: Optional[str] = None) -> TestRunResult:
        """Main orchestration method."""
        run_id = generate_run_id()
        start_time = time.time()
        
        self.console.print(Panel(
            f"🧪 Glass Engine Test Run - {mode.title()} Mode",
            title="Glass Engine",
            border_style="blue"
        ))
        
        try:
            # Load test cases
            test_cases = await self.test_loader.load_test_cases(
                filter_mode=mode, 
                test_id=test_id, 
                category=category
            )
            
            if not test_cases:
                self.console.print("[yellow]No test cases found matching criteria[/yellow]")
                return TestRunResult(
                    run_id=run_id,
                    mode=mode,
                    total_tests=0,
                    passed_tests=0,
                    failed_tests=0,
                    error_tests=0,
                    skipped_tests=0,
                    total_duration=time.time() - start_time,
                    test_results=[],
                    artifacts_dir=""
                )
            
            self.console.print(f"Found {len(test_cases)} test case(s) to run")
            
            # Execute tests
            results = []
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                
                for test_case in test_cases:
                    task = progress.add_task(f"Running {test_case.id}: {test_case.name}")
                    
                    result = await self._execute_single_test(test_case, mode)
                    results.append(result)
                    
                    progress.update(task, completed=True)
            
            # Calculate summary
            total_tests = len(results)
            passed_tests = sum(1 for r in results if r.status == TestStatus.PASSED)
            failed_tests = sum(1 for r in results if r.status == TestStatus.FAILED)
            error_tests = sum(1 for r in results if r.status == TestStatus.ERROR)
            skipped_tests = sum(1 for r in results if r.status == TestStatus.SKIPPED)
            
            # Generate report
            await self._display_results(results, mode)
            
            return TestRunResult(
                run_id=run_id,
                mode=mode,
                total_tests=total_tests,
                passed_tests=passed_tests,
                failed_tests=failed_tests,
                error_tests=error_tests,
                skipped_tests=skipped_tests,
                total_duration=time.time() - start_time,
                test_results=results,
                artifacts_dir=f"test_runs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{mode}"
            )
            
        except Exception as e:
            # Handle orchestrator-level errors
            error_info = self.error_handler.handle_error(e)
            self.console.print(f"[red]Error during test execution: {e}[/red]")
            if error_info and error_info.recovery_suggestion:
                self.console.print(f"[yellow]Recovery suggestion: {error_info.recovery_suggestion}[/yellow]")
            raise
    
    async def _execute_single_test(self, test_case: TestCase, mode: str) -> TestResult:
        """Execute a single test case with full isolation and error handling."""
        trace_id = generate_trace_id()
        error_message = None
        failure_details = None
        
        try:
            async with TestContext(trace_id, test_case.id, mode) as context:
                # Setup context for this test case
                await context.setup_for_test_case(test_case, mode)
                
                # Execute setup hooks
                if test_case.setup_hooks and GLASS_COMPONENTS_AVAILABLE:
                    await self.hook_registry.execute_setup_hooks(
                        test_case.setup_hooks, context
                    )
                
                # Execute test steps with error recovery
                step_results = []
                for i, step in enumerate(test_case.steps):
                    try:
                        step_result = await self.error_recovery.execute_with_retry(
                            self.test_executor.execute_step,
                            step, context, i,
                            max_retries=test_case.retry_count
                        )
                        step_results.append(step_result)
                        
                        if step_result.status == TestStatus.FAILED:
                            break
                            
                    except Exception as e:
                        error_info = self.error_handler.handle_error(
                            e, test_case_id=test_case.id, trace_id=trace_id
                        )
                        
                        step_results.append(StepResult(
                            step_index=i,
                            status=TestStatus.ERROR,
                            duration=0.0,
                            output="",
                            error=str(e)
                        ))
                        
                        error_message = str(e)
                        failure_details = error_info.recovery_suggestion
                        break
                
                # Validate assertions with error handling
                assertion_results = []
                try:
                    assertion_results = await self.assertion_validator.validate_assertions(
                        test_case.assertions.get(mode, []), context
                    )
                except Exception as e:
                    error_info = self.error_handler.handle_error(
                        e, test_case_id=test_case.id, trace_id=trace_id
                    )
                    error_message = str(e)
                    failure_details = error_info.recovery_suggestion
                
                # Execute teardown hooks
                if test_case.teardown_hooks and GLASS_COMPONENTS_AVAILABLE:
                    try:
                        await self.hook_registry.execute_teardown_hooks(
                            test_case.teardown_hooks, context
                        )
                    except Exception as e:
                        # Log teardown errors but don't fail the test
                        self.error_handler.handle_error(
                            e, test_case_id=test_case.id, trace_id=trace_id
                        )
                
                # Determine overall test status
                test_status = self._calculate_test_status(step_results, assertion_results)
                
                return TestResult(
                    test_case_id=test_case.id,
                    status=test_status,
                    duration=time.time() - context.start_time.timestamp(),
                    trace_id=trace_id,
                    start_time=context.start_time,
                    end_time=datetime.now(),
                    step_results=step_results,
                    assertion_results=assertion_results,
                    artifacts_path=context.artifacts_dir,
                    error_message=error_message,
                    failure_details=failure_details
                )
                
        except Exception as e:
            # Handle catastrophic test failures
            error_info = self.error_handler.handle_error(
                e, test_case_id=test_case.id, trace_id=trace_id
            )
            
            return TestResult(
                test_case_id=test_case.id,
                status=TestStatus.ERROR,
                duration=0.0,
                trace_id=trace_id,
                start_time=datetime.now(),
                end_time=datetime.now(),
                step_results=[],
                assertion_results=[],
                artifacts_path=f"test_runs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{mode}/{test_case.id}",
                error_message=str(e),
                failure_details=error_info.recovery_suggestion if error_info else None
            )
    
    def _calculate_test_status(self, step_results: List[StepResult], 
                              assertion_results: List[AssertionResult]) -> TestStatus:
        """Calculate overall test status from step and assertion results."""
        # Check for errors first
        if any(r.status == TestStatus.ERROR for r in step_results + assertion_results):
            return TestStatus.ERROR
        
        # Check for timeouts
        if any(r.status == TestStatus.TIMEOUT for r in step_results):
            return TestStatus.TIMEOUT
        
        # Check for failures
        if any(r.status == TestStatus.FAILED for r in step_results + assertion_results):
            return TestStatus.FAILED
        
        # All passed
        return TestStatus.PASSED
    
    async def _display_results(self, results: List[TestResult], mode: str):
        """Display test results in a formatted table."""
        if not results:
            return
        
        table = Table(title=f"Test Results - {mode.title()} Mode")
        table.add_column("Test ID", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Duration", style="yellow")
        table.add_column("Steps", style="blue")
        table.add_column("Assertions", style="green")
        table.add_column("Artifacts", style="dim")
        
        for result in results:
            # Status with color
            if result.status == TestStatus.PASSED:
                status = "[green]✓ PASSED[/green]"
            elif result.status == TestStatus.FAILED:
                status = "[red]✗ FAILED[/red]"
            elif result.status == TestStatus.ERROR:
                status = "[red]⚠ ERROR[/red]"
            elif result.status == TestStatus.TIMEOUT:
                status = "[yellow]⏱ TIMEOUT[/yellow]"
            else:
                status = "[dim]- SKIPPED[/dim]"
            
            # Step summary
            passed_steps = sum(1 for r in result.step_results if r.status == TestStatus.PASSED)
            total_steps = len(result.step_results)
            steps_summary = f"{passed_steps}/{total_steps}"
            
            # Assertion summary
            passed_assertions = sum(1 for r in result.assertion_results if r.status == TestStatus.PASSED)
            total_assertions = len(result.assertion_results)
            assertions_summary = f"{passed_assertions}/{total_assertions}"
            
            table.add_row(
                result.test_case_id,
                status,
                f"{result.duration:.2f}s",
                steps_summary,
                assertions_summary,
                result.artifacts_path
            )
        
        self.console.print(table)
        
        # Summary
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.status == TestStatus.PASSED)
        failed_tests = sum(1 for r in results if r.status == TestStatus.FAILED)
        error_tests = sum(1 for r in results if r.status == TestStatus.ERROR)
        
        summary_text = f"[bold]Summary:[/bold] {passed_tests}/{total_tests} passed"
        if failed_tests > 0:
            summary_text += f", {failed_tests} failed"
        if error_tests > 0:
            summary_text += f", {error_tests} errors"
        
        self.console.print(summary_text)
        
        # Generate and save reports
        if self.report_generator and GLASS_COMPONENTS_AVAILABLE:
            try:
                artifacts_dir = f"test_runs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{mode}"
                run_result = TestRunResult(
                    run_id=generate_run_id(),
                    mode=mode,
                    total_tests=len(results),
                    passed_tests=passed_tests,
                    failed_tests=failed_tests,
                    error_tests=error_tests,
                    skipped_tests=0,
                    total_duration=sum(r.duration for r in results),
                    test_results=results,
                    artifacts_dir=artifacts_dir
                )
                
                saved_files = save_test_reports(run_result, artifacts_dir)
                self.console.print(f"[dim]Reports saved to: {artifacts_dir}[/dim]")
                
                # Display error statistics if available
                if hasattr(self.error_handler, 'get_error_statistics'):
                    error_stats = self.error_handler.get_error_statistics()
                    if error_stats['total_errors'] > 0:
                        self.console.print(f"[yellow]Total errors encountered: {error_stats['total_errors']}[/yellow]")
                        if error_stats.get('most_common_error'):
                            most_common = error_stats['most_common_error']
                            self.console.print(f"[yellow]Most common error: {most_common['type']} ({most_common['count']} occurrences)[/yellow]")
                
            except Exception as e:
                self.console.print(f"[yellow]Warning: Could not generate reports: {e}[/yellow]")
        else:
            self.console.print(f"[dim]Reports not available (Glass components not loaded)[/dim]")