# Glass Engine Low Level Design (LLD)

## 1. Architecture Overview

### Component Hierarchy
```
GlassOrchestrator
├── TestCaseLoader
├── TestExecutor
├── AssertionValidator
├── ArtifactCollector
└── ReportGenerator

TestContext
├── TraceManager
├── MockRegistry
├── ResourceManager
└── StateCapture

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
├── processor.py           ←→  TestContext (trace injection)
├── storage.py             ←→  MockRegistry, StateCapture
├── parser_engine.py       ←→  MockRegistry, TraceManager
├── embedding_engine.py    ←→  MockRegistry, TraceManager
└── query_engine.py        ←→  MockRegistry, TraceManager
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

### Assertion Model
```python
@dataclass
class Assertion:
    type: str  # cli_output, database_record, log_entry, file_exists, etc.
    description: str
    condition: Optional[str] = None
    contains: Optional[List[str]] = None
    not_contains: Optional[List[str]] = None
    equals: Optional[str] = None
    not_equals: Optional[str] = None
    regex_match: Optional[str] = None
    count: Optional[int] = None
    table: Optional[str] = None
    field: Optional[str] = None
    not_null: Optional[bool] = None
    timeout: int = 5
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
    actual_value: Optional[str] = None
    expected_value: Optional[str] = None

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

### AssertionValidator
```python
class AssertionValidator:
    def __init__(self):
        self.db_validator = DatabaseValidator()
        self.log_validator = LogValidator()
        self.file_validator = FileValidator()
        
    async def validate_assertions(self, assertions: List[Assertion], 
                                 context: TestContext) -> List[AssertionResult]:
        """Validate all assertions for a test case"""
        results = []
        
        for assertion in assertions:
            try:
                result = await self._validate_single_assertion(assertion, context)
                results.append(result)
            except Exception as e:
                results.append(AssertionResult(
                    assertion=assertion,
                    status=TestStatus.ERROR,
                    message=f"Assertion validation failed: {str(e)}"
                ))
        
        return results
    
    async def _validate_single_assertion(self, assertion: Assertion, 
                                        context: TestContext) -> AssertionResult:
        """Validate a single assertion"""
        if assertion.type == "cli_output":
            return await self._validate_cli_output(assertion, context)
        elif assertion.type == "database_record":
            return await self.db_validator.validate_database_record(assertion, context)
        elif assertion.type == "log_entry":
            return await self.log_validator.validate_log_entry(assertion, context)
        elif assertion.type == "file_exists":
            return await self.file_validator.validate_file_exists(assertion, context)
        else:
            raise ValueError(f"Unknown assertion type: {assertion.type}")
    
    async def _validate_cli_output(self, assertion: Assertion, 
                                  context: TestContext) -> AssertionResult:
        """Validate CLI output assertion"""
        last_output = context.state_capture.get_last_command_output()
        
        if assertion.contains:
            for expected in assertion.contains:
                if expected not in last_output:
                    return AssertionResult(
                        assertion=assertion,
                        status=TestStatus.FAILED,
                        message=f"Expected '{expected}' not found in output",
                        actual_value=last_output[:200] + "..." if len(last_output) > 200 else last_output,
                        expected_value=expected
                    )
        
        return AssertionResult(
            assertion=assertion,
            status=TestStatus.PASSED,
            message="CLI output assertion passed"
        )
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
        await self._setup_temp_config()
        
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
        
    async def _setup_temp_config(self):
        """Create sandboxed configuration"""
        # Load base config
        base_config = load_config()
        
        # Override with test-specific settings
        test_config = base_config.model_copy()
        test_config.db_path = self.temp_db_path
        
        # Save to temporary file
        test_config.save_to_file(self.temp_config_path)
```

### MockRegistry
```python
class MockRegistry:
    def __init__(self):
        self.mocks = {}
        self.original_functions = {}
        
    def register_mock(self, target_path: str, mock_function: Callable):
        """Register a mock function for a specific target"""
        self.mocks[target_path] = mock_function
        
    def activate_mocks(self):
        """Activate all registered mocks"""
        for target_path, mock_func in self.mocks.items():
            # Store original function
            original = self._get_function_by_path(target_path)
            self.original_functions[target_path] = original
            
            # Replace with mock
            self._set_function_by_path(target_path, mock_func)
    
    def deactivate_mocks(self):
        """Restore original functions"""
        for target_path, original_func in self.original_functions.items():
            self._set_function_by_path(target_path, original_func)
        
        self.mocks.clear()
        self.original_functions.clear()
    
    def _get_function_by_path(self, path: str) -> Callable:
        """Get function by import path"""
        module_path, func_name = path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, func_name)
    
    def _set_function_by_path(self, path: str, func: Callable):
        """Set function by import path"""
        module_path, func_name = path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        setattr(module, func_name, func)
```

### StateCapture
```python
class StateCapture:
    def __init__(self, artifacts_dir: str):
        self.artifacts_dir = artifacts_dir
        self.command_history = []
        self.state_snapshots = {}
        
    async def capture_initial_state(self):
        """Capture initial system state"""
        self.state_snapshots['initial'] = {
            'timestamp': datetime.now().isoformat(),
            'database_state': await self._capture_database_state(),
            'file_system_state': await self._capture_file_system_state(),
            'process_state': await self._capture_process_state()
        }
        
    async def capture_final_state(self):
        """Capture final system state"""
        self.state_snapshots['final'] = {
            'timestamp': datetime.now().isoformat(),
            'database_state': await self._capture_database_state(),
            'file_system_state': await self._capture_file_system_state(),
            'process_state': await self._capture_process_state()
        }
        
        # Save snapshots to artifacts
        await self._save_state_snapshots()
    
    async def _capture_database_state(self) -> dict:
        """Capture current database state"""
        # Implementation to capture database state
        pass
    
    async def _save_state_snapshots(self):
        """Save state snapshots to artifacts directory"""
        snapshots_file = Path(self.artifacts_dir) / "state_snapshots.json"
        with open(snapshots_file, 'w') as f:
            json.dump(self.state_snapshots, f, indent=2)
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

This LLD provides the detailed technical specifications needed to implement the Glass Engine framework with proper architecture, error handling, and performance considerations.