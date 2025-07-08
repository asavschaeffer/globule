# Glass Engine Low Level Design (LLD)

## 1. Architecture Overview

### Component Hierarchy
```
GlassOrchestrator
├── TestCaseLoader (supports directory-based loading)
├── TestExecutor
├── AssertionValidator (polymorphic assertion system)
├── ArtifactCollector
└── ReportGenerator

TestContext
├── TraceManager
├── ComponentFactory (dependency injection)
├── MockRegistry (clean DI-based mocking)
├── ResourceManager
└── StateCapture (assertion-driven)

TestHooks
├── SetupHooks
├── TeardownHooks
├── MockHooks
└── FailureSimulators
```

### Integration Points
```
Existing Globule Components     Glass Engine Components
├── cli.py                 ←→  GlassOrchestrator
├── processor.py           ←→  ComponentFactory (DI), TestContext (trace injection)
├── storage.py             ←→  ComponentFactory (DI), StateCapture
├── parser_engine.py       ←→  ComponentFactory (DI), TraceManager
├── embedding_engine.py    ←→  ComponentFactory (DI), TraceManager
└── query_engine.py        ←→  ComponentFactory (DI), TraceManager
```

## 2. Core Data Structures

### TestCase Model
```python
@dataclass
class TestCase:
    id: str
    name: str
    description: str
    category: str
    mode_config: ModeConfig
    steps: List[TestStep]
    assertions: Dict[str, List[Assertion]]
    edge_cases: Optional[List[EdgeCase]] = None
    setup_hooks: Optional[List[str]] = None
    teardown_hooks: Optional[List[str]] = None
    timeout: int = 30
    retry_count: int = 0

@dataclass
class ModeConfig:
    tutorial: ModeSettings
    showcase: ModeSettings

@dataclass
class ModeSettings:
    enabled: bool
    priority: str  # essential, high, medium, low
    narration_level: str  # basic, detailed, verbose
```

### TestStep Model
```python
@dataclass
class TestStep:
    action: str  # cli_command, api_call, state_check
    command: Optional[str] = None
    args: Optional[List[str]] = None
    expected_output_contains: Optional[List[str]] = None
    expected_output_not_contains: Optional[List[str]] = None
    timeout: int = 10
    retry_on_failure: bool = False
    setup_hook: Optional[str] = None
    teardown_hook: Optional[str] = None
```

### Assertion Model (Object-Oriented)
```python
from abc import ABC, abstractmethod

class Assertion(ABC):
    """Base class for all assertions"""
    
    def __init__(self, description: str, timeout: int = 5):
        self.description = description
        self.timeout = timeout
    
    @abstractmethod
    async def execute(self, context: TestContext) -> AssertionResult:
        """Execute the assertion and return result"""
        pass
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {self.description}"

# Specific assertion implementations
class CliOutputAssertion(Assertion):
    def __init__(self, description: str, contains: List[str], 
                 not_contains: Optional[List[str]] = None, **kwargs):
        super().__init__(description, **kwargs)
        self.contains = contains
        self.not_contains = not_contains or []

class DatabaseRecordAssertion(Assertion):
    def __init__(self, description: str, table: str, condition: str, 
                 expected_count: Optional[int] = None, **kwargs):
        super().__init__(description, **kwargs)
        self.table = table
        self.condition = condition
        self.expected_count = expected_count

class LogEntryAssertion(Assertion):
    def __init__(self, description: str, contains: List[str], 
                 log_level: Optional[str] = None, **kwargs):
        super().__init__(description, **kwargs)
        self.contains = contains
        self.log_level = log_level

class EmbeddingAssertion(Assertion):
    def __init__(self, description: str, expected_shape: Optional[Tuple[int, ...]] = None, 
                 not_null: bool = True, **kwargs):
        super().__init__(description, **kwargs)
        self.expected_shape = expected_shape
        self.not_null = not_null
```

### TestResult Model
```python
@dataclass
class TestResult:
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
class StepResult:
    step_index: int
    status: TestStatus
    duration: float
    output: str
    error: Optional[str] = None
    
@dataclass
class AssertionResult:
    assertion: Assertion
    status: TestStatus
    message: str
    duration: float
    actual_value: Optional[str] = None
    expected_value: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

enum TestStatus:
    PASSED
    FAILED
    SKIPPED
    ERROR
    TIMEOUT
```

### TestContext Model
```python
@dataclass
class TestContext:
    trace_id: str
    test_case_id: str
    mode: str
    temp_db_path: str
    temp_config_path: str
    artifacts_dir: str
    component_factory: ComponentFactory
    mock_registry: MockRegistry
    state_capture: StateCapture
    resource_manager: ResourceManager
    start_time: datetime
    metadata: Dict[str, Any]
```

## 3. Core Classes Implementation

### GlassOrchestrator
```python
class GlassOrchestrator:
    def __init__(self, config: Config):
        self.config = config
        self.test_loader = TestCaseLoader()
        self.test_executor = TestExecutor()
        self.assertion_validator = AssertionValidator()
        self.artifact_collector = ArtifactCollector()
        self.report_generator = ReportGenerator()
        self.console = Console()
        
    async def run(self, mode: str, test_id: Optional[str] = None, 
                  category: Optional[str] = None) -> TestRunResult:
        """Main orchestration method"""
        # Load test cases
        test_cases = await self.test_loader.load_test_cases(
            filter_mode=mode, 
            test_id=test_id, 
            category=category
        )
        
        # Create test run context
        run_id = generate_run_id()
        run_context = TestRunContext(run_id, mode, test_cases)
        
        # Execute tests
        results = []
        for test_case in test_cases:
            result = await self._execute_single_test(test_case, mode, run_context)
            results.append(result)
            
        # Generate report
        report = await self.report_generator.generate_report(results, run_context)
        
        return TestRunResult(run_id, results, report)
    
    async def _execute_single_test(self, test_case: TestCase, mode: str, 
                                   run_context: TestRunContext) -> TestResult:
        """Execute a single test case with full isolation"""
        trace_id = generate_trace_id()
        
        async with TestContext(trace_id, test_case.id, mode) as context:
            # Setup hooks
            if test_case.setup_hooks:
                await self._execute_hooks(test_case.setup_hooks, context)
            
            # Execute test steps
            step_results = []
            for i, step in enumerate(test_case.steps):
                step_result = await self.test_executor.execute_step(
                    step, context, i
                )
                step_results.append(step_result)
                
                if step_result.status == TestStatus.FAILED:
                    break
            
            # Validate assertions
            assertion_results = await self.assertion_validator.validate_assertions(
                test_case.assertions.get(mode, []), context
            )
            
            # Collect artifacts
            await self.artifact_collector.collect_artifacts(context)
            
            # Teardown hooks
            if test_case.teardown_hooks:
                await self._execute_hooks(test_case.teardown_hooks, context)
            
            return TestResult(
                test_case_id=test_case.id,
                status=self._calculate_test_status(step_results, assertion_results),
                duration=time.time() - context.start_time.timestamp(),
                trace_id=trace_id,
                start_time=context.start_time,
                end_time=datetime.now(),
                step_results=step_results,
                assertion_results=assertion_results,
                artifacts_path=context.artifacts_dir
            )
```

### TestExecutor
```python
class TestExecutor:
    def __init__(self):
        self.command_executor = CommandExecutor()
        self.api_executor = APIExecutor()
        
    async def execute_step(self, step: TestStep, context: TestContext, 
                          step_index: int) -> StepResult:
        """Execute a single test step"""
        start_time = time.time()
        
        try:
            if step.action == "cli_command":
                output = await self.command_executor.execute_cli_command(
                    step.command, step.args, context
                )
            elif step.action == "api_call":
                output = await self.api_executor.execute_api_call(
                    step, context
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
```

### CommandExecutor
```python
class CommandExecutor:
    async def execute_cli_command(self, command: str, args: List[str], 
                                 context: TestContext) -> str:
        """Execute CLI command with proper environment setup"""
        # Set up environment with test context
        env = os.environ.copy()
        env['GLOBULE_DB_PATH'] = context.temp_db_path
        env['GLOBULE_CONFIG_PATH'] = context.temp_config_path
        env['GLASS_ENGINE_TRACE_ID'] = context.trace_id
        
        # Build command
        cmd = [command] + args
        
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
            
            return output
            
        except asyncio.TimeoutError:
            process.kill()
            raise asyncio.TimeoutError(f"Command '{command}' timed out")
```

### AssertionValidator (Simplified with Polymorphism)
```python
class AssertionValidator:
    """Much simpler validation with polymorphic assertions"""
    
    def __init__(self):
        self.assertion_factory = AssertionFactory()
    
    async def validate_assertions(self, assertion_configs: List[Dict[str, Any]], 
                                 context: TestContext) -> List[AssertionResult]:
        """Validate all assertions using polymorphic execution"""
        results = []
        
        for assertion_config in assertion_configs:
            try:
                assertion = self.assertion_factory.create_assertion(assertion_config)
                result = await assertion.execute(context)
                results.append(result)
                
            except Exception as e:
                # Create error result for failed assertion creation
                error_result = AssertionResult(
                    assertion=None,
                    status=TestStatus.ERROR,
                    message=f"Failed to create assertion: {str(e)}",
                    duration=0.0
                )
                results.append(error_result)
        
        return results

class AssertionFactory:
    """Factory for creating assertion instances from configuration"""
    
    ASSERTION_TYPES = {
        'cli_output': CliOutputAssertion,
        'database_record': DatabaseRecordAssertion,
        'log_entry': LogEntryAssertion,
        'embedding': EmbeddingAssertion,
    }
    
    @classmethod
    def create_assertion(cls, assertion_config: Dict[str, Any]) -> Assertion:
        """Create assertion instance from configuration"""
        assertion_type = assertion_config.get('type')
        
        if assertion_type not in cls.ASSERTION_TYPES:
            raise ValueError(f"Unknown assertion type: {assertion_type}")
        
        assertion_class = cls.ASSERTION_TYPES[assertion_type]
        
        # Remove 'type' from config and pass remaining as kwargs
        config_copy = assertion_config.copy()
        config_copy.pop('type')
        
        return assertion_class(**config_copy)
```

### TestContext Implementation
```python
class TestContext:
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
        
    async def __aenter__(self):
        """Setup isolated test environment"""
        # Create artifacts directory
        Path(self.artifacts_dir).mkdir(parents=True, exist_ok=True)
        
        # Setup temporary database
        await self._setup_temp_database()
        
        # Setup temporary configuration
        config = await self._setup_temp_config()
        
        # Initialize dependency injection
        self.component_factory = ComponentFactory(config, self.mock_registry)
        
        # Initialize tracing
        await self.trace_manager.initialize()
        
        # Capture initial state
        await self.state_capture.capture_initial_state()
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup test environment"""
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
        """Create isolated database for test"""
        # Copy schema from main database or create fresh
        storage = SQLiteStorage(self.temp_db_path)
        await storage._init_db()
        
    async def _setup_temp_config(self) -> Config:
        """Create sandboxed configuration"""
        # Load base config
        base_config = load_config()
        
        # Override with test-specific settings
        test_config = base_config.model_copy()
        test_config.db_path = self.temp_db_path
        
        # Save to temporary file
        test_config.save_to_file(self.temp_config_path)
        
        return test_config
```

### ComponentFactory & MockRegistry (Dependency Injection)
```python
class ComponentFactory:
    """Factory for creating components with dependency injection"""
    
    def __init__(self, config: Config, mock_registry: Optional[MockRegistry] = None):
        self.config = config
        self.mock_registry = mock_registry or MockRegistry()
    
    async def create_parser(self) -> Parser:
        """Create parser with optional mocking"""
        if self.mock_registry.has_mock('parser'):
            return self.mock_registry.get_mock('parser')
        return await create_parser(use_ollama=self.config.llm_provider == "local")
    
    async def create_embedder(self) -> Embedder:
        """Create embedder with optional mocking"""
        if self.mock_registry.has_mock('embedder'):
            return self.mock_registry.get_mock('embedder')
        return await create_embedder(use_ollama=self.config.embedding_provider == "local")
    
    def create_storage(self) -> Storage:
        """Create storage with optional mocking"""
        if self.mock_registry.has_mock('storage'):
            return self.mock_registry.get_mock('storage')
        return SQLiteStorage(self.config.db_path)

class MockRegistry:
    """Clean dependency injection-based mocking"""
    
    def __init__(self):
        self.mocks: Dict[str, Any] = {}
        self.active_mocks: Set[str] = set()
    
    def register_mock(self, component_name: str, mock_instance: Any):
        """Register a mock instance for a component"""
        self.mocks[component_name] = mock_instance
        self.active_mocks.add(component_name)
    
    def has_mock(self, component_name: str) -> bool:
        """Check if a mock is registered for a component"""
        return component_name in self.active_mocks
    
    def get_mock(self, component_name: str) -> Any:
        """Get mock instance for a component"""
        if not self.has_mock(component_name):
            raise ValueError(f"No mock registered for component: {component_name}")
        return self.mocks[component_name]
    
    def clear_mocks(self):
        """Clear all registered mocks"""
        self.mocks.clear()
        self.active_mocks.clear()

# Example mock implementations
class MockParser:
    """Mock parser for testing failure scenarios"""
    
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
    """Mock embedder for testing scenarios"""
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
    
    async def embed_text(self, text: str) -> np.ndarray:
        # Return deterministic embedding for testing
        return np.random.RandomState(hash(text) % 2**32).random(self.embedding_dim)
```

### StateCapture (Assertion-Driven)
```python
class StateCapture:
    """Minimal, assertion-driven state capture"""
    
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
        """Register what state needs to be captured based on assertions"""
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
        """Capture initial targeted state"""
        # Only capture state if we have specific requirements
        pass
    
    async def capture_final_state(self):
        """Capture final targeted state"""
        # Only capture state if we have specific requirements
        pass
    
    async def capture_targeted_state(self, context: TestContext):
        """Capture only the state required by assertions"""
        # Capture database state for specific queries
        if self.required_db_queries:
            await self._capture_database_queries(context)
        
        # Capture log entries for specific levels
        if self.required_log_levels:
            await self._capture_log_entries()
        
        # Capture embedding state if needed
        if self.needs_embedding_check:
            await self._capture_embedding_state(context)
    
    async def _capture_database_queries(self, context: TestContext):
        """Execute and capture results of specific database queries"""
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
    
    def log_command_execution(self, command: str, args: List[str], 
                             output: str, return_code: int):
        """Log command execution details"""
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
        """Get output from the last executed command"""
        return self.command_outputs[-1] if self.command_outputs else ""
    
    def get_log_content(self) -> str:
        """Get captured log content"""
        return '\n'.join(self.log_buffer)
    
    def get_last_created_globule(self) -> Optional[Globule]:
        """Get the last created globule"""
        return self.last_created_globule
```

## 4. Error Handling Strategy

### Exception Hierarchy
```python
class GlassEngineError(Exception):
    """Base exception for Glass Engine"""
    pass

class TestCaseLoadError(GlassEngineError):
    """Error loading test case definitions"""
    pass

class TestExecutionError(GlassEngineError):
    """Error during test execution"""
    pass

class AssertionError(GlassEngineError):
    """Assertion validation failed"""
    pass

class TestTimeoutError(GlassEngineError):
    """Test execution timed out"""
    pass

class MockSetupError(GlassEngineError):
    """Error setting up mocks"""
    pass
```

### Error Recovery Mechanisms
1. **Graceful Degradation**: Continue with remaining tests if one fails
2. **Retry Logic**: Configurable retry for flaky tests
3. **Fallback Modes**: Simple assertions if complex ones fail
4. **Isolation**: Ensure test failures don't affect other tests
5. **Cleanup**: Always clean up resources even on failure

## 5. Performance Considerations

### Optimization Strategies
1. **Parallel Test Execution**: Run independent tests concurrently
2. **Resource Pooling**: Reuse database connections and components
3. **Lazy Loading**: Load test cases and dependencies on demand
4. **Caching**: Cache parsed test definitions and compiled assertions
5. **Streaming**: Stream large outputs instead of loading in memory

### Memory Management
- Use temporary files for large outputs
- Implement proper cleanup in context managers
- Monitor memory usage during test execution
- Garbage collection after each test

## 6. Security Considerations

### Isolation Mechanisms
1. **Process Isolation**: Each test runs in isolated environment
2. **File System Isolation**: Use temporary directories
3. **Network Isolation**: Mock external services
4. **Database Isolation**: Separate database per test
5. **Configuration Isolation**: Sandboxed configuration files

### Security Best Practices
- Never log sensitive information
- Validate all inputs to prevent injection
- Use secure temporary file creation
- Implement proper cleanup of sensitive data
- Audit trail for all test executions

## 7. Testing Strategy for Glass Engine

### Unit Tests
- Test individual components in isolation
- Mock external dependencies
- Cover edge cases and error conditions
- Validate assertion logic

### Integration Tests
- Test component interactions
- Validate end-to-end workflows
- Test with real Globule components
- Verify isolation mechanisms

### Performance Tests
- Measure test execution time
- Test with large numbers of test cases
- Memory usage validation
- Concurrent execution testing

### Dogfooding
- Use Glass Engine to test itself
- Create test cases for Glass Engine functionality
- Validate meta-testing scenarios
- Ensure framework reliability

## 8. Directory-Based Test Case Organization

### Directory Structure
```
tests/glass_engine/
├── test_cases/
│   ├── core_functionality/
│   │   ├── add_thought.yaml
│   │   ├── basic_validation.yaml
│   │   └── error_handling.yaml
│   ├── search/
│   │   ├── semantic_search.yaml
│   │   ├── temporal_search.yaml
│   │   └── combined_search.yaml
│   ├── reporting/
│   │   ├── daily_report.yaml
│   │   └── domain_report.yaml
│   └── edge_cases/
│       ├── parser_failures.yaml
│       ├── database_errors.yaml
│       └── network_issues.yaml
├── hooks/
│   ├── setup_hooks.py
│   └── failure_simulators.py
└── fixtures/
    ├── sample_data.yaml
    └── test_configs.yaml
```

### Enhanced TestCaseLoader
```python
class TestCaseLoader:
    """Load test cases from directory structure"""
    
    def __init__(self, test_cases_dir: str = "tests/glass_engine/test_cases"):
        self.test_cases_dir = Path(test_cases_dir)
        self.loaded_cases: Dict[str, TestCase] = {}
        self.categories: Dict[str, List[str]] = {}
    
    async def load_test_cases(self, filter_mode: str = None, 
                             test_id: str = None, 
                             category: str = None) -> List[TestCase]:
        """Load test cases with filtering"""
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
        """Load all test cases from directory structure"""
        for yaml_file in self.test_cases_dir.rglob("*.yaml"):
            category = yaml_file.parent.name
            test_cases = await self._load_test_cases_from_file(yaml_file)
            
            for test_case in test_cases:
                self.loaded_cases[test_case.id] = test_case
                
                # Track categories
                if category not in self.categories:
                    self.categories[category] = []
                self.categories[category].append(test_case.id)
```

## 9. Architectural Decisions & Design History

### Mocking Strategy Evolution
**Initial Design**: Dynamic `MockRegistry` using `importlib` and `setattr` for monkey-patching functions at runtime.

**Problem**: This approach was brittle and error-prone:
- String-based `target_path` could break silently during refactoring
- Timing issues with module imports
- Difficult to reason about and debug
- No compile-time validation

**Final Design**: **Dependency Injection** pattern via `ComponentFactory`.
- Dependencies are explicit and type-safe
- `TestContext` manages component creation with optional mocking
- `ThoughtProcessor` accepts injected dependencies instead of creating them
- More maintainable and testable architecture
- Requires minor refactoring of core components but significantly improves robustness

### Assertion Model Evolution
**Initial Design**: Single `Assertion` dataclass with numerous optional fields (`contains`, `equals`, `regex_match`, etc.).

**Problem**: This approach led to:
- Bloated dataclass with many optional fields
- Complex `AssertionValidator` with growing `if/elif/else` chains
- Difficult to add new assertion types without modifying existing code
- Violated the Open/Closed Principle

**Final Design**: **Object-oriented model** with polymorphic assertion classes.
- Base `Assertion` class with `execute()` method
- Specific subclasses: `CliOutputAssertion`, `DatabaseRecordAssertion`, etc.
- `AssertionValidator` becomes simple: iterate and call `execute()` on each assertion
- Easy to add new assertion types without modifying existing code
- Cleaner, more maintainable architecture

### State Capture Strategy Evolution
**Initial Design**: Generic state capture that snapshots everything (database, file system, process state).

**Problem**: This approach was:
- Slow and resource-intensive
- Generated noisy diffs that were hard to analyze
- Captured unnecessary data that wasn't used for validation
- Poor signal-to-noise ratio for debugging

**Final Design**: **Assertion-driven state capture** that only collects what's needed.
- `StateCapture.register_assertion_requirements()` analyzes assertions to determine what state to capture
- Database queries are specific to assertion requirements
- Log capture is filtered by relevant log levels
- Embedding state is only captured when needed
- Minimal, targeted data collection improves performance and debugging effectiveness

### Test Organization Evolution
**Initial Design**: Single monolithic `test_cases.yaml` file.

**Problem**: This approach would:
- Create merge conflicts in large teams
- Become difficult to navigate as test suite grows
- Mix different categories of tests together
- Be hard to run specific subsets of tests

**Final Design**: **Directory-based organization** with category support.
- `TestCaseLoader` supports loading from directory structure
- Tests organized by logical categories (core_functionality, search, reporting, etc.)
- Each category can be run independently
- Better scalability and maintainability
- Easier collaboration and code reviews

These architectural decisions prioritize maintainability, robustness, and developer experience while ensuring the framework can scale as the test suite grows.

This LLD provides the detailed technical specifications needed to implement the Glass Engine framework with proper architecture, error handling, and performance considerations.