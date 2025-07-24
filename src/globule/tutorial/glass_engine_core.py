"""
Glass Engine Core Architecture

This module implements the Glass Engine philosophy for Globule Phase 1, providing
a unified testing/tutorial/showcase system with three distinct operational modes.

The Glass Engine Philosophy:
    "Let the user see exactly how the pistons fire while teaching them to drive."
    
    Tests become tutorials, tutorials become showcases, showcases become tests.
    No black boxes - complete transparency in system operation.

Author: Globule Team
Date: 2025-07-24
Version: 1.0.0

Architecture Overview:
    - AbstractGlassEngine: Base class defining the Glass Engine interface
    - InteractiveMode: Pedagogical tutorial with guided user input
    - DemoMode: Professional technical showcase with automated examples  
    - DebugMode: Raw execution traces for deep system introspection
    - GlassEngineFactory: Factory pattern for mode instantiation
    - GlassEngineMetrics: Performance and validation metrics collection

Design Patterns:
    - Strategy Pattern: Different execution modes with common interface
    - Template Method: Shared tutorial flow with mode-specific implementations
    - Factory Pattern: Clean mode instantiation and configuration
    - Observer Pattern: Metrics collection and event logging

Professional Protocols:
    - Comprehensive docstrings following Google style
    - Type hints for all public interfaces
    - Logging with structured output
    - Error handling with proper exception hierarchy
    - Unit test coverage targets (>90%)
    - Performance profiling and optimization
"""

import abc
import asyncio
import logging
import time
import traceback
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Protocol, TypeVar, Generic
from contextlib import asynccontextmanager

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.tree import Tree
from rich.json import JSON
from rich.syntax import Syntax

from globule.config.settings import get_config
from globule.storage.sqlite_manager import SQLiteStorageManager
from globule.embedding.ollama_provider import OllamaEmbeddingProvider
from globule.parsing.ollama_parser import OllamaParser
from globule.orchestration.parallel_strategy import ParallelOrchestrationEngine
from globule.core.models import EnrichedInput, ProcessedGlobule


class GlassEngineMode(Enum):
    """
    Enumeration of Glass Engine operational modes.
    
    Each mode serves a distinct purpose while maintaining the core Glass Engine
    philosophy of unified testing/teaching/showcasing.
    """
    INTERACTIVE = "interactive"  # Pedagogical tutorial with guided user input
    DEMO = "demo"               # Professional technical showcase
    DEBUG = "debug"             # Raw execution traces and deep introspection


class GlassEngineError(Exception):
    """Base exception class for Glass Engine operations."""
    pass


class GlassEngineValidationError(GlassEngineError):
    """Raised when Glass Engine validation fails."""
    pass


class GlassEngineExecutionError(GlassEngineError):
    """Raised when Glass Engine execution encounters an error."""
    pass


@dataclass
class GlassEngineMetrics:
    """
    Comprehensive metrics collection for Glass Engine operations.
    
    Attributes:
        mode: The Glass Engine mode that generated these metrics
        start_time: Execution start timestamp
        end_time: Execution end timestamp
        total_duration_ms: Total execution time in milliseconds
        test_results: Results from validation tests
        performance_data: Performance profiling data
        validation_status: Overall validation status
        error_log: Any errors encountered during execution
        user_interactions: Count of user interactions (Interactive mode)
        showcase_components: Components demonstrated (Demo mode)
        trace_depth: Depth of execution traces (Debug mode)
    """
    mode: GlassEngineMode
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    total_duration_ms: float = 0.0
    test_results: List[Dict[str, Any]] = field(default_factory=list)
    performance_data: Dict[str, float] = field(default_factory=dict)
    validation_status: str = "PENDING"
    error_log: List[str] = field(default_factory=list)
    user_interactions: int = 0
    showcase_components: List[str] = field(default_factory=list)
    trace_depth: int = 0
    
    def mark_complete(self) -> None:
        """Mark metrics collection as complete and calculate final statistics."""
        self.end_time = datetime.now()
        if self.start_time:
            delta = self.end_time - self.start_time
            self.total_duration_ms = delta.total_seconds() * 1000
        
        # Determine validation status based on test results
        if not self.test_results:
            self.validation_status = "NO_TESTS"
        elif all(result.get("success", False) for result in self.test_results):
            self.validation_status = "PASS"
        else:
            self.validation_status = "FAIL"
    
    def add_error(self, error: Exception, context: str = "") -> None:
        """Add an error to the error log with context information."""
        error_msg = f"{context}: {type(error).__name__}: {str(error)}"
        self.error_log.append(error_msg)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            "mode": self.mode.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_duration_ms": self.total_duration_ms,
            "test_results": self.test_results,
            "performance_data": self.performance_data,
            "validation_status": self.validation_status,
            "error_count": len(self.error_log),
            "user_interactions": self.user_interactions,
            "showcase_components": self.showcase_components,
            "trace_depth": self.trace_depth
        }


class AbstractGlassEngine(abc.ABC):
    """
    Abstract base class for all Glass Engine implementations.
    
    This class defines the common interface and shared functionality for all
    Glass Engine modes, implementing the Template Method pattern to ensure
    consistent behavior while allowing mode-specific customization.
    
    The Glass Engine operates in four phases:
        1. Initialization: Set up components and validate system state
        2. Execution: Run the mode-specific tutorial/showcase/debug flow
        3. Validation: Verify system functionality and collect metrics
        4. Reporting: Present results in mode-appropriate format
    """
    
    def __init__(self, console: Optional[Console] = None):
        """
        Initialize the Glass Engine with required components.
        
        Args:
            console: Rich console for output formatting. If None, creates new console.
        """
        self.console = console or Console()
        self.logger = self._setup_logging()
        self.config = get_config()
        self.metrics = GlassEngineMetrics(mode=self.get_mode())
        
        # Core Globule components - initialized in async context
        self.storage: Optional[SQLiteStorageManager] = None
        self.embedding_provider: Optional[OllamaEmbeddingProvider] = None
        self.parser: Optional[OllamaParser] = None  
        self.orchestrator: Optional[ParallelOrchestrationEngine] = None
        
        # State tracking
        self._initialized = False
        self._test_data: List[Dict[str, Any]] = []
    
    @abc.abstractmethod
    def get_mode(self) -> GlassEngineMode:
        """Return the Glass Engine mode for this implementation."""
        pass
    
    @abc.abstractmethod
    async def execute_tutorial_flow(self) -> None:
        """
        Execute the mode-specific tutorial flow.
        
        This is the core method that each mode must implement to define
        its unique behavior while maintaining Glass Engine principles.
        """
        pass
    
    @abc.abstractmethod
    def present_results(self) -> None:
        """
        Present execution results in mode-appropriate format.
        
        Each mode should format and display results according to its
        target audience and use case.
        """
        pass
    
    async def run(self) -> GlassEngineMetrics:
        """
        Execute the complete Glass Engine workflow.
        
        This is the main entry point that orchestrates the four-phase
        Glass Engine execution: Initialize → Execute → Validate → Report.
        
        Returns:
            GlassEngineMetrics: Comprehensive execution metrics and results
            
        Raises:
            GlassEngineError: If any phase fails critically
        """
        self.logger.info(f"Starting Glass Engine in {self.get_mode().value} mode")
        
        try:
            # Phase 1: Initialization
            await self._initialize_components()
            
            # Phase 2: Execution  
            await self.execute_tutorial_flow()
            
            # Phase 3: Validation
            await self._validate_system_state()
            
            # Phase 4: Reporting
            self.present_results()
            
            self.metrics.mark_complete()
            self.logger.info(f"Glass Engine completed successfully in {self.metrics.total_duration_ms:.1f}ms")
            
        except Exception as e:
            self.metrics.add_error(e, "Glass Engine execution")
            self.metrics.mark_complete()
            self.logger.error(f"Glass Engine failed: {e}")
            raise GlassEngineExecutionError(f"Glass Engine execution failed: {e}") from e
        
        finally:
            await self._cleanup_components()
        
        return self.metrics
    
    async def _initialize_components(self) -> None:
        """
        Initialize all Globule components required for Glass Engine operation.
        
        This method sets up the core Globule architecture components and
        validates that they can communicate properly.
        
        Raises:
            GlassEngineError: If component initialization fails
        """
        self.logger.debug("Initializing Globule components")
        
        try:
            # Initialize storage layer
            self.storage = SQLiteStorageManager()
            await self.storage.initialize()
            
            # Initialize AI providers
            self.embedding_provider = OllamaEmbeddingProvider()
            self.parser = OllamaParser()
            
            # Initialize orchestration engine
            self.orchestrator = ParallelOrchestrationEngine(
                embedding_provider=self.embedding_provider,
                parsing_provider=self.parser,
                storage_manager=self.storage
            )
            
            self._initialized = True
            self.logger.debug("Component initialization completed successfully")
            
        except Exception as e:
            raise GlassEngineError(f"Component initialization failed: {e}") from e
    
    async def _validate_system_state(self) -> None:
        """
        Validate that all system components are functioning correctly.
        
        This method performs comprehensive validation of the Globule system
        state and records results in metrics for later analysis.
        """
        self.logger.debug("Validating system state")
        
        validation_results = []
        
        # Validate storage layer
        try:
            await self.storage.get_recent_globules(limit=1)
            validation_results.append({"component": "storage", "status": "PASS", "message": "Storage accessible"})
        except Exception as e:
            validation_results.append({"component": "storage", "status": "FAIL", "message": str(e)})
        
        # Validate embedding provider
        try:
            health_ok = await self.embedding_provider.health_check()
            status = "PASS" if health_ok else "WARN"
            message = "Embedding provider healthy" if health_ok else "Embedding provider unavailable (using mock)"
            validation_results.append({"component": "embedding", "status": status, "message": message})
        except Exception as e:
            validation_results.append({"component": "embedding", "status": "FAIL", "message": str(e)})
        
        # Validate parser
        try:
            test_result = await self.parser.parse("test input")
            validation_results.append({"component": "parser", "status": "PASS", "message": "Parser responding"})
        except Exception as e:
            validation_results.append({"component": "parser", "status": "FAIL", "message": str(e)})
        
        # Record validation results
        self.metrics.test_results.extend(validation_results)
        
        # Check if any critical components failed
        critical_failures = [r for r in validation_results if r["status"] == "FAIL"]
        if critical_failures:
            raise GlassEngineValidationError(f"Critical component validation failed: {critical_failures}")
    
    async def _cleanup_components(self) -> None:
        """Clean up resources and close connections."""
        self.logger.debug("Cleaning up components")
        
        if self.embedding_provider:
            await self.embedding_provider.close()
        
        if self.parser:
            await self.parser.close()
        
        if self.storage:
            await self.storage.close()
    
    def _setup_logging(self) -> logging.Logger:
        """
        Set up structured logging for Glass Engine operations.
        
        Returns:
            logging.Logger: Configured logger instance
        """
        logger = logging.getLogger(f"glass_engine.{self.get_mode().value}")
        
        # Avoid duplicate handlers
        if not logger.handlers:
            handler = RichHandler(console=self.console, show_path=False)
            handler.setFormatter(logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            ))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        
        return logger
    
    # Utility methods for common Glass Engine operations
    
    def create_test_input(self, text: str, source: str = "glass_engine") -> EnrichedInput:
        """
        Create a standardized EnrichedInput for testing purposes.
        
        Args:
            text: The input text to process
            source: Source identifier for tracking
            
        Returns:
            EnrichedInput: Properly formatted input object
        """
        return EnrichedInput(
            original_text=text,
            enriched_text=text,
            detected_schema_id=None,
            schema_config=None,
            additional_context={},
            source=source,
            timestamp=datetime.now(),
            verbosity="verbose"
        )
    
    @asynccontextmanager
    async def performance_timer(self, operation_name: str):
        """
        Context manager for timing operations and recording performance data.
        
        Args:
            operation_name: Name of the operation being timed
            
        Yields:
            None
        """
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            self.metrics.performance_data[operation_name] = duration_ms
            self.logger.debug(f"{operation_name} completed in {duration_ms:.2f}ms")
    
    def log_user_interaction(self, interaction_type: str, data: Any = None) -> None:
        """
        Log user interactions for metrics collection.
        
        Args:
            interaction_type: Type of interaction (input, selection, etc.)
            data: Additional interaction data
        """
        self.metrics.user_interactions += 1
        self.logger.debug(f"User interaction: {interaction_type}", extra={"data": data})
    
    def add_showcase_component(self, component_name: str) -> None:
        """
        Record a component being showcased for metrics.
        
        Args:
            component_name: Name of the component being demonstrated
        """
        self.metrics.showcase_components.append(component_name)
        self.logger.debug(f"Showcasing component: {component_name}")


class GlassEngineFactory:
    """
    Factory class for creating Glass Engine instances.
    
    Implements the Factory pattern to provide clean instantiation of different
    Glass Engine modes with proper configuration and dependency injection.
    """
    
    @staticmethod
    def create(mode: GlassEngineMode, console: Optional[Console] = None) -> AbstractGlassEngine:
        """
        Create a Glass Engine instance for the specified mode.
        
        Args:
            mode: The Glass Engine mode to create
            console: Optional Rich console for output formatting
            
        Returns:
            AbstractGlassEngine: Configured Glass Engine instance
            
        Raises:
            ValueError: If the specified mode is not supported
        """
        # Import mode implementations here to avoid circular imports
        from globule.tutorial.modes.interactive_mode import InteractiveGlassEngine
        from globule.tutorial.modes.simple_demo import SimpleDemoGlassEngine  
        from globule.tutorial.modes.debug_mode import DebugGlassEngine
        
        mode_map = {
            GlassEngineMode.INTERACTIVE: InteractiveGlassEngine,
            GlassEngineMode.DEMO: SimpleDemoGlassEngine,
            GlassEngineMode.DEBUG: DebugGlassEngine
        }
        
        if mode not in mode_map:
            raise ValueError(f"Unsupported Glass Engine mode: {mode}")
        
        return mode_map[mode](console=console)


# Main entry point for Glass Engine execution
async def run_glass_engine(mode: GlassEngineMode = GlassEngineMode.DEMO, 
                          console: Optional[Console] = None) -> GlassEngineMetrics:
    """
    Main entry point for Glass Engine execution.
    
    This function provides a clean, high-level interface for running the
    Glass Engine in any mode with proper error handling and metrics collection.
    
    Args:
        mode: The Glass Engine mode to execute
        console: Optional Rich console for output formatting
        
    Returns:
        GlassEngineMetrics: Comprehensive execution metrics and results
        
    Raises:
        GlassEngineError: If execution fails critically
    """
    engine = GlassEngineFactory.create(mode, console)
    return await engine.run()


# Convenience functions for each mode
async def run_interactive_tutorial(console: Optional[Console] = None) -> GlassEngineMetrics:
    """Run Glass Engine in Interactive mode."""
    return await run_glass_engine(GlassEngineMode.INTERACTIVE, console)


async def run_demo_showcase(console: Optional[Console] = None) -> GlassEngineMetrics:
    """Run Glass Engine in Demo mode."""
    return await run_glass_engine(GlassEngineMode.DEMO, console)


async def run_debug_trace(console: Optional[Console] = None) -> GlassEngineMetrics:
    """Run Glass Engine in Debug mode.""" 
    return await run_glass_engine(GlassEngineMode.DEBUG, console)