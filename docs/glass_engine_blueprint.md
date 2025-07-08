# UDE "Glass Engine" Implementation Plan for Globule

Based on my analysis of the Globule codebase, here's a comprehensive implementation plan for integrating the Unified Development Experience (UDE) Methodology framework:

## 1. Test Case Definitions

### Structure & Storage
**Location:** `tests/glass_engine/test_cases.yaml`

**YAML Structure:**
```yaml
test_cases:
  - id: "TC-001"
    name: "Add Basic Thought"
    description: "User adds a simple thought to the system"
    category: "core_functionality"
    mode_config:
      tutorial:
        enabled: true
        priority: "essential"
      showcase:
        enabled: true
        priority: "high"
    
    steps:
      - action: "cli_command"
        command: "globule add"
        args: ["I need to finish the quarterly report"]
        expected_output_contains: ["✓ Thought captured successfully!", "ID:"]
        
    assertions:
      tutorial:
        - type: "cli_output"
          contains: ["✓ Thought captured successfully!"]
        - type: "database_record"
          table: "globules"
          condition: "content = 'I need to finish the quarterly report'"
      
      showcase:
        - type: "cli_output"
          contains: ["✓ Thought captured successfully!"]
        - type: "database_record"
          table: "globules"
          condition: "content = 'I need to finish the quarterly report'"
        - type: "log_entry"
          contains: ["[GLASS ENGINE]", "ThoughtProcessor initialized"]
        - type: "embedding_generated"
          field: "embedding"
          not_null: true
    
    edge_cases:
      - name: "Empty Input"
        setup: null
        action: 
          command: "globule add"
          args: [""]
        expected_behavior: "graceful_failure"
        assertions:
          - type: "error_message"
            contains: ["Error: Empty input"]
      
      - name: "Parser Failure"
        setup: "mock_parser_failure"
        action:
          command: "globule add"
          args: ["Test thought"]
        expected_behavior: "fallback_to_simple_parser"
        assertions:
          - type: "log_entry"
            contains: ["Parser failed", "fallback"]
          - type: "database_record"
            table: "globules"
            condition: "content = 'Test thought'"

  - id: "TC-002"
    name: "Search Thoughts"
    # ... similar structure
```

**Essential Fields:**
- `id`: Unique test case identifier
- `name`: Human-readable test name
- `description`: Purpose of the test
- `category`: Groups related tests (core_functionality, search, reporting, etc.)
- `mode_config`: Controls which modes include this test
- `steps`: Sequence of actions to perform
- `assertions`: Validations for tutorial vs showcase modes
- `edge_cases`: Showcase-specific scenarios with setup hooks

## 2. Glass Orchestrator Design

### Integration with CLI
**New Module:** `globule/glass.py`

**CLI Integration in `cli.py`:**
```python
@cli.command()
@click.argument('mode', type=click.Choice(['tutorial', 'showcase']))
@click.option('--test', help='Run specific test case')
@click.option('--category', help='Run tests in specific category')
@click.pass_context
def glass(ctx: click.Context, mode: str, test: str = None, category: str = None):
    """Run Glass Engine tests in tutorial or showcase mode."""
    orchestrator = GlassOrchestrator(ctx.obj.config)
    asyncio.run(orchestrator.run(mode, test_id=test, category=category))
```

### Core Components

**GlassOrchestrator Class:**
- **Responsibility:** Load test cases, manage execution, coordinate with existing components
- **Key Methods:**
  - `load_test_cases()`: Parse YAML definitions
  - `run(mode, test_id=None, category=None)`: Execute tests based on mode
  - `execute_test_case(test_case, mode)`: Run individual test with appropriate narration
  - `validate_assertions(test_case, mode, results)`: Check test outcomes

**TestContext Class:**
- **Responsibility:** Manage isolated test environment
- **Key Attributes:**
  - `trace_id`: Unique identifier for traceability
  - `temp_db_path`: Isolated database for each test
  - `temp_config_path`: Sandboxed configuration
  - `mock_registry`: Manage mocked components
  - `artifacts_dir`: Directory for test artifacts

**Integration Points:**
- Hook into `ThoughtProcessor` to inject trace_id into metadata
- Extend `SQLiteStorage` to support temporary databases
- Add instrumentation to `OllamaParser`, `OllamaEmbedder`, `QueryEngine`

## 3. Observability and Traceability

### Trace ID Implementation
**Injection Strategy:**
- Generate unique `trace_id` for each test execution
- Add to `TestContext` and pass through component chain
- Inject into:
  - Log messages: `[GLASS ENGINE][trace_id: xyz-123] Parser invoked`
  - Database metadata: `{"trace_id": "xyz-123", "glass_test": "TC-001"}`
  - Component initialization

### Enhanced Logging
**Instrumentation Points:**
- `processor.py:28`: `self.logger.info(f"[GLASS ENGINE][trace_id: {trace_id}] ThoughtProcessor initialized")`
- `parser_engine.py:38`: Log before/after LLM calls
- `embedding_engine.py:32`: Log embedding generation
- `storage.py:82`: Log database operations
- `query_engine.py:23`: Log search operations

### State Capture
**Before/After Snapshots:**
- Database state: Row counts, recent entries
- Configuration state: Active settings
- Component state: Initialized embedders, parsers
- File system state: Generated artifacts

**Glass Engine Block View (Showcase Mode):**
```
[GLASS ENGINE] Test: TC-001 - Add Basic Thought
[GLASS ENGINE][trace_id: xyz-123] === STEP 1: CLI Command ===
[GLASS ENGINE][trace_id: xyz-123] Command: globule add "I need to finish the quarterly report"
[GLASS ENGINE][trace_id: xyz-123] ThoughtProcessor initialized
[GLASS ENGINE][trace_id: xyz-123] Generated globule_id: abc-456
[GLASS ENGINE][trace_id: xyz-123] Cached input for globule abc-456
[GLASS ENGINE][trace_id: xyz-123] Starting background processing
[GLASS ENGINE][trace_id: xyz-123] OllamaParser invoked
[GLASS ENGINE][trace_id: xyz-123] OllamaEmbedder generating embedding
[GLASS ENGINE][trace_id: xyz-123] Embedding shape: (1024,)
[GLASS ENGINE][trace_id: xyz-123] Database updated with processed globule
[GLASS ENGINE][trace_id: xyz-123] === ASSERTIONS ===
[GLASS ENGINE][trace_id: xyz-123] ✓ CLI output contains success message
[GLASS ENGINE][trace_id: xyz-123] ✓ Database record created
[GLASS ENGINE][trace_id: xyz-123] ✓ Embedding generated
```

## 4. Assertions and Edge Cases

### Tutorial Mode Assertions
**Focus:** Essential user-facing validations
- CLI output contains expected messages
- Database records created successfully
- Basic functionality works as expected

**Example Assertions:**
- `globule add` produces "✓ Thought captured successfully!"
- `globule search` returns results in table format
- `globule report` generates markdown summary

### Showcase Mode Assertions
**Focus:** Comprehensive internal validation
- All tutorial assertions plus:
- Log entries confirm internal processes
- Component fallbacks work correctly
- Database schema integrity maintained
- Embedding dimensions correct

**Edge Case Categories:**
1. **Input Validation:** Empty inputs, malformed data
2. **Component Failures:** Parser timeout, embedding service down
3. **Database Issues:** Connection failures, disk full
4. **Configuration Problems:** Invalid URLs, missing models
5. **Performance:** Large inputs, concurrent operations

## 5. Debug Artifacts

### Test Run Directory Structure
```
test_runs/
├── 2025-07-08_14-30-15_tutorial/
│   ├── summary.md
│   ├── TC-001_add_basic_thought/
│   │   ├── test_context.json
│   │   ├── cli_output.txt
│   │   ├── app.log
│   │   ├── db_before.sql
│   │   ├── db_after.sql
│   │   └── assertions.json
│   └── TC-002_search_thoughts/
│       └── ... similar structure
└── 2025-07-08_14-35-22_showcase/
    ├── summary.md
    ├── TC-001_add_basic_thought/
    │   ├── test_context.json
    │   ├── cli_output.txt
    │   ├── app.log
    │   ├── glass_engine.log
    │   ├── db_before.sql
    │   ├── db_after.sql
    │   ├── embedding_analysis.json
    │   └── assertions.json
    └── edge_cases/
        └── TC-001_empty_input/
            └── ... artifacts
```

### Summary Report Format
```markdown
# Glass Engine Test Run Summary

**Mode:** Tutorial
**Date:** 2025-07-08 14:30:15
**Total Tests:** 5
**Passed:** 4
**Failed:** 1

## Test Results

### ✓ TC-001: Add Basic Thought
- **Status:** PASSED
- **Duration:** 2.3s
- **Assertions:** 2/2 passed

### ✗ TC-002: Search Thoughts
- **Status:** FAILED  
- **Duration:** 5.1s
- **Assertions:** 1/3 passed
- **Failure:** Database query timeout
- **Artifacts:** See `TC-002_search_thoughts/app.log`

## Recommendations
- Check database connection settings
- Review query performance for large datasets
```

## 6. Test State Management and Isolation

### Database Isolation
**Strategy:** Temporary SQLite databases for each test
- Create fresh `temp_globule_test_{trace_id}.db` for each test
- Modify `SQLiteStorage` to accept custom database path
- Ensure complete cleanup after test completion

**Implementation:**
```python
class TestContext:
    def __init__(self, test_case_id: str):
        self.trace_id = generate_trace_id()
        self.temp_db_path = f"temp_globule_test_{self.trace_id}.db"
        self.temp_config_path = f"temp_config_{self.trace_id}.yaml"
        # ... other setup
    
    async def __aenter__(self):
        # Setup isolated environment
        self.config = await self.create_sandboxed_config()
        self.storage = SQLiteStorage(self.temp_db_path)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Cleanup resources
        await self.storage.close()
        Path(self.temp_db_path).unlink(missing_ok=True)
        Path(self.temp_config_path).unlink(missing_ok=True)
```

### Configuration Sandboxing
**Approach:** Generate temporary config for each test
- Copy base config to `temp_config_{trace_id}.yaml`
- Modify paths to use temporary directories
- Override with test-specific settings

### Idempotency Guarantees
- Fresh database for each test run
- Deterministic test data generation
- Cleanup of all temporary artifacts
- Reset of singleton components (`ProcessingManager`)

## 7. Setup Hooks and Test Context

### Hook System
**Hook Types:**
- `before_test`: Setup test environment
- `mock_component`: Replace component with mock
- `simulate_failure`: Inject controlled failures
- `after_test`: Cleanup and validation

**Example Hook Implementation:**
```python
class TestHooks:
    @staticmethod
    async def mock_parser_failure(context: TestContext):
        """Simulate parser failure by mocking OllamaParser"""
        async def failing_parse(text: str):
            raise RuntimeError("Simulated parser failure")
        
        context.mock_registry.register("parser.parse_text", failing_parse)
    
    @staticmethod
    async def simulate_database_full(context: TestContext):
        """Simulate database storage failure"""
        # Mock SQLiteStorage to raise disk full error
        pass
```

### TestContext Management
**Responsibilities:**
- Manage temporary resources (database, config, logs)
- Coordinate component mocking
- Handle trace ID propagation
- Collect artifacts for debugging

**Key Methods:**
- `setup_isolated_environment()`: Create clean test environment
- `inject_trace_id()`: Add traceability to all operations
- `collect_artifacts()`: Gather logs, database dumps, outputs
- `cleanup()`: Remove temporary resources

## 8. Developer Experience

### Test Development Workflow
1. **Add New Test Case:**
   - Edit `test_cases.yaml`
   - Define steps, assertions, edge cases
   - Run single test: `globule glass tutorial --test TC-005`

2. **Debug Failed Tests:**
   - Check `test_runs/latest/summary.md` for overview
   - Navigate to specific test artifact directory
   - Review `app.log` for detailed execution trace
   - Examine `db_before.sql` and `db_after.sql` for state changes

3. **Test Categories:**
   - Run by category: `globule glass showcase --category core_functionality`
   - Categories: `core_functionality`, `search`, `reporting`, `edge_cases`

### Framework Commands
```bash
# Run all tutorial tests
globule glass tutorial

# Run all showcase tests (includes edge cases)
globule glass showcase

# Run specific test
globule glass tutorial --test TC-001

# Run category
globule glass showcase --category search

# Continuous mode (watch for changes)
globule glass tutorial --watch
```

### Documentation Structure
```
docs/glass_engine/
├── README.md              # Overview and quick start
├── test_case_authoring.md # How to write test cases
├── edge_case_guide.md     # Edge case patterns
├── debugging_guide.md     # Troubleshooting failed tests
└── architecture.md        # Internal framework design
```

## Implementation Phases

### Phase 1: Core Infrastructure
- Implement `GlassOrchestrator` and `TestContext`
- Basic test case loading and execution
- Simple CLI integration

### Phase 2: Observability
- Add trace ID injection system
- Implement enhanced logging
- Create artifact collection system

### Phase 3: Advanced Features
- Edge case support with hooks
- Mock system for component failures
- Performance testing capabilities

### Phase 4: Developer Experience
- Rich CLI interface with progress indicators
- Comprehensive documentation
- IDE integration for test authoring

This plan provides a robust foundation for implementing the UDE methodology in Globule, ensuring comprehensive testing coverage while maintaining developer productivity and system transparency.