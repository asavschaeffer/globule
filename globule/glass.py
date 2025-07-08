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