"""
Debug Glass Engine Mode

This module implements the Debug mode of the Glass Engine, designed as a direct
LLM/human-to-code interface that provides raw execution traces, deep system
introspection, and maximum data fidelity for immediate understanding.

The Debug mode embodies the Glass Engine philosophy by:
- Sacrificing pretty formatting for data depth and fidelity
- Providing granular execution traces with variable states
- Offering direct access to internal data structures
- Enabling immediate understanding of system behavior
- Facilitating rapid debugging and system analysis

Target Audience: Engineers, LLMs, system debugging, deep analysis
Primary Purpose: Raw system introspection and debugging interface
User Experience: Maximum information density, technical focus, efficiency over aesthetics

Author: Globule Team
Date: 2025-07-24
Version: 1.0.0
"""

import asyncio
import json
import sys
import traceback
import inspect
import time
import psutil
import os
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.json import JSON
from rich.syntax import Syntax
from rich.tree import Tree

from globule.tutorial.glass_engine_core import AbstractGlassEngine, GlassEngineMode
from globule.core.models import EnrichedInput


def rich_json_default(o: Any) -> Any:
    """Custom JSON serializer for rich.json.JSON that handles special types."""
    if isinstance(o, datetime):
        return o.isoformat()
    if isinstance(o, Path):
        return o.as_posix()
    if hasattr(o, 'as_dict'):
        return o.as_dict()
    if hasattr(o, '__dict__'):
        return o.__dict__
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


class ExecutionTrace:
    """
    Detailed execution trace collector for debug analysis.
    
    Captures function calls, variable states, timing information,
    and system resources at each execution step.
    """
    
    def __init__(self):
        self.traces: List[Dict[str, Any]] = []
        self.call_stack: List[str] = []
        self.start_time = time.perf_counter()
        
    def trace_call(self, func_name: str, args: tuple = (), kwargs: Dict = None, 
                   locals_snapshot: Dict = None, memory_delta: float = 0):
        """Record a function call with full context."""
        current_time = time.perf_counter()
        
        trace_entry = {
            "timestamp": current_time,
            "elapsed_ms": (current_time - self.start_time) * 1000,
            "function": func_name,
            "call_depth": len(self.call_stack),
            "args": str(args)[:200] + "..." if len(str(args)) > 200 else str(args),
            "kwargs": str(kwargs or {})[:200] + "..." if len(str(kwargs or {})) > 200 else str(kwargs or {}),
            "locals_snapshot": locals_snapshot or {},
            "memory_delta_mb": memory_delta,
            "stack_trace": self.call_stack.copy()
        }
        
        self.traces.append(trace_entry)
        self.call_stack.append(func_name)
    
    def trace_return(self, func_name: str, return_value: Any = None, 
                     locals_snapshot: Dict = None, memory_delta: float = 0):
        """Record a function return with context."""
        current_time = time.perf_counter()
        
        if self.call_stack and self.call_stack[-1] == func_name:
            self.call_stack.pop()
        
        trace_entry = {
            "timestamp": current_time,
            "elapsed_ms": (current_time - self.start_time) * 1000,
            "function": f"{func_name}_return",
            "call_depth": len(self.call_stack),
            "return_value": str(return_value)[:200] + "..." if len(str(return_value)) > 200 else str(return_value),
            "locals_snapshot": locals_snapshot or {},
            "memory_delta_mb": memory_delta,
            "stack_trace": self.call_stack.copy()
        }
        
        self.traces.append(trace_entry)
    
    def get_trace_summary(self) -> Dict[str, Any]:
        """Get comprehensive trace summary."""
        if not self.traces:
            return {"error": "No traces recorded"}
        
        total_time = self.traces[-1]["elapsed_ms"]
        unique_functions = set(t["function"] for t in self.traces)
        max_depth = max(t["call_depth"] for t in self.traces)
        
        return {
            "total_execution_time_ms": total_time,
            "total_trace_points": len(self.traces),
            "unique_functions_called": len(unique_functions),
            "max_call_depth": max_depth,
            "functions_list": list(unique_functions),
            "memory_usage_pattern": [t["memory_delta_mb"] for t in self.traces if t["memory_delta_mb"] != 0]
        }


class DebugGlassEngine(AbstractGlassEngine):
    """
    Debug Glass Engine implementation for deep system introspection.
    
    This class provides raw execution traces, variable state dumps, performance
    profiling, and maximum data fidelity for engineers and LLMs who need to
    understand exactly what the system is doing at each step.
    
    Unlike other modes, Debug mode prioritizes:
    - Data completeness over visual appeal
    - Technical accuracy over user-friendliness  
    - Granular detail over high-level summaries
    - Raw structures over formatted presentations
    - Efficiency over aesthetics
    
    Attributes:
        execution_trace: Detailed execution trace collector
        memory_profiler: Memory usage tracking
        performance_counters: Granular performance measurements
        variable_dumps: Complete variable state snapshots
        debug_data: Raw debug information collection
    """
    
    def __init__(self, console: Optional[Console] = None):
        """
        Initialize the Debug Glass Engine.
        
        Args:
            console: Rich console for output. Debug mode uses minimal formatting.
        """
        super().__init__(console)
        self.execution_trace = ExecutionTrace()
        self.memory_profiler: Dict[str, float] = {}
        self.performance_counters: Dict[str, List[float]] = {}
        self.variable_dumps: Dict[str, Dict[str, Any]] = {}
        self.debug_data: Dict[str, Any] = {}
        
        # Initialize system monitoring
        self.process = psutil.Process(os.getpid())
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024
        
    def get_mode(self) -> GlassEngineMode:
        """Return the Debug Glass Engine mode."""
        return GlassEngineMode.DEBUG
    
    @contextmanager
    def trace_execution(self, operation_name: str, capture_locals: bool = True):
        """
        Context manager for tracing operation execution with full detail.
        
        Args:
            operation_name: Name of the operation being traced
            capture_locals: Whether to capture local variable states
        """
        # Get current memory usage
        current_memory = self.process.memory_info().rss / 1024 / 1024
        memory_delta = current_memory - self.initial_memory
        
        # Capture local variables if requested
        locals_snapshot = {}
        if capture_locals:
            frame = inspect.currentframe().f_back
            if frame:
                locals_snapshot = {
                    k: str(v)[:100] + "..." if len(str(v)) > 100 else str(v)
                    for k, v in frame.f_locals.items()
                    if not k.startswith('_') and not callable(v)
                }
        
        # Start timing
        start_time = time.perf_counter()
        self.execution_trace.trace_call(operation_name, locals_snapshot=locals_snapshot, memory_delta=memory_delta)
        
        try:
            yield
        except Exception as e:
            # Record exception in trace
            error_info = {
                "exception_type": type(e).__name__,
                "exception_message": str(e),
                "traceback": traceback.format_exc()
            }
            self.execution_trace.trace_return(operation_name, return_value=f"EXCEPTION: {error_info}", memory_delta=memory_delta)
            raise
        finally:
            # End timing and record
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            
            if operation_name not in self.performance_counters:
                self.performance_counters[operation_name] = []
            self.performance_counters[operation_name].append(duration_ms)
            
            final_memory = self.process.memory_info().rss / 1024 / 1024
            memory_delta = final_memory - current_memory
            
            self.execution_trace.trace_return(operation_name, memory_delta=memory_delta)
    
    async def execute_tutorial_flow(self) -> None:
        """
        Execute the debug tutorial flow with maximum introspection.
        
        This method provides raw, unfiltered access to system execution
        with comprehensive tracing and data collection.
        """
        self.logger.info("Starting debug mode with full system introspection")
        
        with self.trace_execution("debug_tutorial_initialization"):
            self.debug_data["mode_start_time"] = datetime.now().isoformat()
            self.debug_data["initial_system_state"] = self._capture_system_state()
        
        # Phase 1: System State Inspection
        await self._deep_system_analysis()
        
        # Phase 2: Component-Level Debugging
        await self._debug_individual_components()
        
        # Phase 3: End-to-End Execution Tracing
        await self._trace_complete_pipeline()
        
        # Phase 4: Performance Profiling
        await self._comprehensive_performance_analysis()
        
        # Phase 5: Memory and Resource Analysis
        await self._analyze_resource_consumption()
        
        # Phase 6: Data Structure Inspection
        await self._inspect_data_structures()
        
        self.logger.info("Debug mode analysis completed")
    
    def _capture_system_state(self) -> Dict[str, Any]:
        """Capture comprehensive system state for analysis."""
        try:
            return {
                "python_version": sys.version,
                "platform": sys.platform,
                "memory_rss_mb": self.process.memory_info().rss / 1024 / 1024,
                "memory_vms_mb": self.process.memory_info().vms / 1024 / 1024,
                "cpu_percent": self.process.cpu_percent(),
                "thread_count": self.process.num_threads(),
                "open_files": len(self.process.open_files()),
                "cwd": os.getcwd(),
                "environment_vars": {k: v for k, v in os.environ.items() if not k.startswith('_')},
                "loaded_modules": list(sys.modules.keys())[:20]  # First 20 modules
            }
        except Exception as e:
            return {"error": f"Failed to capture system state: {e}"}
    
    async def _deep_system_analysis(self) -> None:
        """Perform deep analysis of system components and configuration."""
        self.console.print("=== DEBUG MODE: DEEP SYSTEM ANALYSIS ===")
        self.console.print(f"TIMESTAMP: {datetime.now().isoformat()}")
        self.console.print(f"MODE: {self.get_mode().value}")
        self.console.print(f"TRACE_DEPTH: MAXIMUM")
        
        with self.trace_execution("system_configuration_analysis"):
            # Raw configuration dump
            config_dict = {
                "storage_path": self.config.storage_path,
                "default_embedding_model": self.config.default_embedding_model,
                "default_parsing_model": self.config.default_parsing_model,
                "ollama_base_url": self.config.ollama_base_url,
                "ollama_timeout": self.config.ollama_timeout,
                "embedding_cache_size": self.config.embedding_cache_size,
                "max_concurrent_requests": self.config.max_concurrent_requests
            }
            
            self.console.print("\n--- RAW CONFIGURATION DATA ---")
            self.console.print(JSON.from_data(config_dict, default=rich_json_default))
            
            # Component initialization states
            self.console.print("\n--- COMPONENT INITIALIZATION STATES ---")
            component_states = {
                "storage_initialized": self.storage is not None,
                "embedding_provider_initialized": self.embedding_provider is not None,
                "parser_initialized": self.parser is not None,
                "orchestrator_initialized": self.orchestrator is not None,
                "storage_class": str(type(self.storage)) if self.storage else None,
                "embedding_class": str(type(self.embedding_provider)) if self.embedding_provider else None,
                "parser_class": str(type(self.parser)) if self.parser else None,
                "orchestrator_class": str(type(self.orchestrator)) if self.orchestrator else None
            }
            
            for component, state in component_states.items():
                self.console.print(f"{component}: {state}")
            
            self.variable_dumps["system_analysis"] = {
                "config": config_dict,
                "components": component_states,
                "system_state": self.debug_data.get("initial_system_state", {})
            }
    
    async def _debug_individual_components(self) -> None:
        """Debug each system component with detailed introspection."""
        self.console.print("\n=== COMPONENT-LEVEL DEBUG ANALYSIS ===")
        
        # Storage component debugging
        await self._debug_storage_component()
        
        # Embedding provider debugging
        await self._debug_embedding_component()
        
        # Parser component debugging
        await self._debug_parser_component()
        
        # Ollama service debugging (Priority 3 enhancement)
        await self._debug_ollama_service()
        
        # Orchestrator debugging
        await self._debug_orchestrator_component()
    
    async def _debug_storage_component(self) -> None:
        """Deep debug analysis of storage component."""
        self.console.print("\n--- STORAGE COMPONENT DEBUG ---")
        
        with self.trace_execution("storage_component_debug"):
            try:
                # Test basic storage operations with timing
                start_time = time.perf_counter()
                
                # Database connection test
                conn_test_start = time.perf_counter()
                recent_globules = await self.storage.get_recent_globules(limit=1)
                conn_test_time = (time.perf_counter() - conn_test_start) * 1000
                
                # Storage info
                storage_dir = self.config.get_storage_dir()
                db_path = storage_dir / "globules.db"
                
                storage_debug_info = {
                    "database_path": str(db_path),
                    "database_exists": db_path.exists(),
                    "database_size_bytes": db_path.stat().st_size if db_path.exists() else 0,
                    "connection_test_time_ms": conn_test_time,
                    "recent_globules_count": len(recent_globules),
                    "storage_directory_exists": storage_dir.exists(),
                    "storage_directory_permissions": oct(storage_dir.stat().st_mode)[-3:] if storage_dir.exists() else None
                }
                
                self.console.print("STORAGE_DEBUG_DATA:")
                self.console.print(JSON.from_data(storage_debug_info, default=rich_json_default))
                
                # Raw SQL schema inspection (if possible)
                if hasattr(self.storage, '_connection') and self.storage._connection:
                    try:
                        # This is implementation-specific debugging
                        schema_info = await self._inspect_database_schema()
                        self.console.print("DATABASE_SCHEMA:")
                        self.console.print(JSON.from_data(schema_info, default=rich_json_default))
                    except Exception as e:
                        self.console.print(f"SCHEMA_INSPECTION_ERROR: {e}")
                
                self.variable_dumps["storage_debug"] = storage_debug_info
                
            except Exception as e:
                error_info = {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "traceback": traceback.format_exc()
                }
                self.console.print("STORAGE_DEBUG_ERROR:")
                self.console.print(JSON.from_data(error_info, default=rich_json_default))
                self.metrics.add_error(e, "storage_component_debug")
    
    async def _inspect_database_schema(self) -> Dict[str, Any]:
        """Inspect database schema for debugging."""
        # This is a debug-specific method to inspect the actual database
        try:
            # Note: This is implementation-specific and may need adjustment
            # based on the actual SQLiteStorageManager implementation
            schema_info = {
                "tables": [],
                "indexes": [],
                "table_info": {}
            }
            
            # Try to get table information
            if hasattr(self.storage, 'get_recent_globules'):
                # Indirect schema inspection through available methods
                sample_globules = await self.storage.get_recent_globules(limit=1)
                if sample_globules:
                    sample_globule = sample_globules[0]
                    schema_info["globule_structure"] = {
                        "id": str(type(sample_globule.id)),
                        "text": str(type(getattr(sample_globule, 'text', None))),
                        "created_at": str(type(sample_globule.created_at)),
                        "embedding": f"numpy_array_shape_{getattr(sample_globule.embedding, 'shape', 'unknown')}" if hasattr(sample_globule, 'embedding') and sample_globule.embedding is not None else None,
                        "parsed_data": str(type(getattr(sample_globule, 'parsed_data', None))),
                    }
            
            return schema_info
        except Exception as e:
            return {"schema_inspection_error": str(e)}
    
    async def _debug_embedding_component(self) -> None:
        """Deep debug analysis of embedding component."""
        self.console.print("\n--- EMBEDDING COMPONENT DEBUG ---")
        
        with self.trace_execution("embedding_component_debug"):
            try:
                # Health check with timing
                health_start = time.perf_counter()
                health_status = await self.embedding_provider.health_check()
                health_time = (time.perf_counter() - health_start) * 1000
                
                # Component introspection
                embedding_debug_info = {
                    "provider_class": str(type(self.embedding_provider)),
                    "health_check_status": health_status,
                    "health_check_time_ms": health_time,
                    "provider_attributes": {
                        attr: str(getattr(self.embedding_provider, attr))[:100]
                        for attr in dir(self.embedding_provider)
                        if not attr.startswith('_') and not callable(getattr(self.embedding_provider, attr))
                    }
                }
                
                # Test embedding generation with detailed timing
                if health_status:
                    test_input = "Debug test embedding generation"
                    embed_start = time.perf_counter()
                    
                    try:
                        test_embedding = await self.embedding_provider.embed(test_input)
                        embed_time = (time.perf_counter() - embed_start) * 1000
                        
                        embedding_debug_info.update({
                            "test_embedding_time_ms": embed_time,
                            "test_embedding_shape": getattr(test_embedding, 'shape', 'unknown') if test_embedding is not None else None,
                            "test_embedding_dtype": str(getattr(test_embedding, 'dtype', 'unknown')) if test_embedding is not None else None,
                            "test_embedding_first_5_values": test_embedding[:5].tolist() if test_embedding is not None and hasattr(test_embedding, 'tolist') else None
                        })
                    except Exception as embed_error:
                        embedding_debug_info["test_embedding_error"] = str(embed_error)
                
                self.console.print("EMBEDDING_DEBUG_DATA:")
                self.console.print(JSON.from_data(embedding_debug_info, default=rich_json_default))
                
                self.variable_dumps["embedding_debug"] = embedding_debug_info
                
            except Exception as e:
                error_info = {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "traceback": traceback.format_exc()
                }
                self.console.print("EMBEDDING_DEBUG_ERROR:")
                self.console.print(JSON.from_data(error_info, default=rich_json_default))
                self.metrics.add_error(e, "embedding_component_debug")
    
    async def _debug_parser_component(self) -> None:
        """Deep debug analysis of parser component."""
        self.console.print("\n--- PARSER COMPONENT DEBUG ---")
        
        with self.trace_execution("parser_component_debug"):
            try:
                # Component introspection
                parser_debug_info = {
                    "parser_class": str(type(self.parser)),
                    "parser_attributes": {
                        attr: str(getattr(self.parser, attr))[:100]
                        for attr in dir(self.parser)
                        if not attr.startswith('_') and not callable(getattr(self.parser, attr))
                    }
                }
                
                # Test parsing with detailed timing
                test_input = "Debug test parsing analysis with multiple concepts and entities"
                parse_start = time.perf_counter()
                
                try:
                    test_result = await self.parser.parse(test_input)
                    parse_time = (time.perf_counter() - parse_start) * 1000
                    
                    parser_debug_info.update({
                        "test_parsing_time_ms": parse_time,
                        "test_result_type": str(type(test_result)),
                        "test_result_keys": list(test_result.keys()) if isinstance(test_result, dict) else None,
                        "test_result_content": test_result if len(str(test_result)) < 500 else str(test_result)[:500] + "..."
                    })
                except Exception as parse_error:
                    parser_debug_info["test_parsing_error"] = str(parse_error)
                
                self.console.print("PARSER_DEBUG_DATA:")
                self.console.print(JSON.from_data(parser_debug_info, default=rich_json_default))
                
                self.variable_dumps["parser_debug"] = parser_debug_info
                
            except Exception as e:
                error_info = {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "traceback": traceback.format_exc()
                }
                self.console.print("PARSER_DEBUG_ERROR:")
                self.console.print(JSON.from_data(error_info, default=rich_json_default))
                self.metrics.add_error(e, "parser_component_debug")
    
    async def _debug_ollama_service(self) -> None:
        """
        Debug Ollama service status and configuration.
        
        Priority 3 enhancement: Provides essential diagnostics for the core dependency
        that would allow instant root cause analysis of parsing timeouts.
        """
        with self.trace_execution("ollama_service_debug", capture_locals=True):
            try:
                self.console.print("\n--- OLLAMA SERVICE DEBUG ---")
                
                import aiohttp
                import subprocess
                import json as json_lib
                
                ollama_debug_info = {
                    "service_url": self.config.ollama_base_url,
                    "configured_timeout": self.config.ollama_timeout,
                    "parsing_model": self.config.default_parsing_model,
                    "embedding_model": self.config.default_embedding_model
                }
                
                # Test service connectivity
                connectivity_start = time.perf_counter()
                try:
                    timeout = aiohttp.ClientTimeout(total=10.0)
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.get(f"{self.config.ollama_base_url}/api/tags") as response:
                            connectivity_time = (time.perf_counter() - connectivity_start) * 1000
                            
                            if response.status == 200:
                                data = await response.json()
                                models = data.get("models", [])
                                
                                ollama_debug_info.update({
                                    "service_status": "HEALTHY",
                                    "connectivity_time_ms": connectivity_time,
                                    "available_models": [model["name"] for model in models],
                                    "model_details": {
                                        model["name"]: {
                                            "size_bytes": model.get("size", 0),
                                            "modified": model.get("modified_at", "unknown"),
                                            "family": model.get("details", {}).get("family", "unknown")
                                        } for model in models
                                    },
                                    "parsing_model_available": self.config.default_parsing_model in [m["name"] for m in models],
                                    "embedding_model_available": self.config.default_embedding_model in [m["name"] for m in models]
                                })
                            else:
                                ollama_debug_info.update({
                                    "service_status": "UNHEALTHY",
                                    "http_status": response.status,
                                    "connectivity_time_ms": connectivity_time
                                })
                                
                except Exception as conn_error:
                    ollama_debug_info.update({
                        "service_status": "UNREACHABLE", 
                        "connection_error": str(conn_error),
                        "connectivity_time_ms": (time.perf_counter() - connectivity_start) * 1000
                    })
                
                # Test Docker container status (if available)
                try:
                    docker_result = subprocess.run(
                        ["docker", "ps", "-a", "--filter", "name=globule-ollama", "--format", "json"],
                        capture_output=True, text=True, timeout=5
                    )
                    
                    if docker_result.returncode == 0 and docker_result.stdout.strip():
                        container_info = json_lib.loads(docker_result.stdout.strip())
                        ollama_debug_info["docker_status"] = {
                            "container_state": container_info.get("State", "unknown"),
                            "container_status": container_info.get("Status", "unknown"),
                            "image": container_info.get("Image", "unknown"),
                            "ports": container_info.get("Ports", "unknown")
                        }
                        
                        # Get recent Docker logs for diagnosis
                        logs_result = subprocess.run(
                            ["docker", "logs", "globule-ollama", "--tail", "10"],
                            capture_output=True, text=True, timeout=5
                        )
                        
                        if logs_result.returncode == 0:
                            recent_logs = logs_result.stdout.strip().split('\n')[-5:]  # Last 5 lines
                            ollama_debug_info["recent_docker_logs"] = recent_logs
                            
                            # Analyze logs for common issues
                            log_analysis = []
                            log_text = ' '.join(recent_logs).lower()
                            
                            if "vram" in log_text and "timeout" in log_text:
                                log_analysis.append("DETECTED: VRAM recovery timeout (suggests model loading issues)")
                            if "context canceled" in log_text:
                                log_analysis.append("DETECTED: Client disconnection during model loading")
                            if "timed out waiting" in log_text:
                                log_analysis.append("DETECTED: Service timeout (possibly CPU-bound or resource constrained)")
                            if "error" in log_text:
                                log_analysis.append("DETECTED: Error conditions in recent logs")
                                
                            ollama_debug_info["log_analysis"] = log_analysis
                            
                except Exception as docker_error:
                    ollama_debug_info["docker_debug_error"] = str(docker_error)
                
                # CPU-safe model detection
                if ollama_debug_info.get("service_status") == "HEALTHY":
                    try:
                        cpu_safe_model = await self.parser.get_cpu_safe_model()
                        model_speed_test = await self.parser._test_model_speed(self.config.default_parsing_model)
                        
                        ollama_debug_info.update({
                            "cpu_safe_model_recommendation": cpu_safe_model,
                            "current_model_speed_test": "FAST" if model_speed_test else "SLOW",
                            "cpu_optimization_needed": not model_speed_test and cpu_safe_model != self.config.default_parsing_model
                        })
                        
                    except Exception as cpu_test_error:
                        ollama_debug_info["cpu_detection_error"] = str(cpu_test_error)
                
                self.console.print("OLLAMA_SERVICE_DEBUG:")
                self.console.print(JSON.from_data(ollama_debug_info, default=rich_json_default))
                
                # Glass Engine diagnostic summary
                if ollama_debug_info.get("service_status") == "UNHEALTHY":
                    self.console.print("\n[red]🚨 DIAGNOSIS: Ollama service issues detected[/red]")
                    if "log_analysis" in ollama_debug_info:
                        for analysis in ollama_debug_info["log_analysis"]:
                            self.console.print(f"[yellow]   {analysis}[/yellow]")
                
                self.variable_dumps["ollama_service_debug"] = ollama_debug_info
                
            except Exception as e:
                error_info = {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "traceback": traceback.format_exc()
                }
                self.console.print("OLLAMA_DEBUG_ERROR:")
                self.console.print(JSON.from_data(error_info, default=rich_json_default))
                self.metrics.add_error(e, "ollama_service_debug")
    
    async def _debug_orchestrator_component(self) -> None:
        """Deep debug analysis of orchestrator component."""
        self.console.print("\n--- ORCHESTRATOR COMPONENT DEBUG ---")
        
        with self.trace_execution("orchestrator_component_debug"):
            try:
                # Component introspection
                orchestrator_debug_info = {
                    "orchestrator_class": str(type(self.orchestrator)),
                    "orchestrator_attributes": {
                        attr: str(getattr(self.orchestrator, attr))[:100]
                        for attr in dir(self.orchestrator)
                        if not attr.startswith('_') and not callable(getattr(self.orchestrator, attr))
                    }
                }
                
                self.console.print("ORCHESTRATOR_DEBUG_DATA:")
                self.console.print(JSON.from_data(orchestrator_debug_info, default=rich_json_default))
                
                self.variable_dumps["orchestrator_debug"] = orchestrator_debug_info
                
            except Exception as e:
                error_info = {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "traceback": traceback.format_exc()
                }
                self.console.print("ORCHESTRATOR_DEBUG_ERROR:")
                self.console.print(JSON.from_data(error_info, default=rich_json_default))
                self.metrics.add_error(e, "orchestrator_component_debug")
    
    async def _trace_complete_pipeline(self) -> None:
        """Trace complete pipeline execution with maximum detail."""
        self.console.print("\n=== COMPLETE PIPELINE EXECUTION TRACE ===")
        
        test_input = "Debug pipeline trace: exploring the intersection of artificial intelligence and human creativity in modern knowledge work"
        
        self.console.print(f"TRACING_INPUT: {test_input}")
        
        def json_default(o):
            if isinstance(o, datetime):
                return o.isoformat()
            if hasattr(o, 'as_posix'): # Handle pathlib.Path objects
                return o.as_posix()
            raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

        with self.trace_execution("complete_pipeline_trace", capture_locals=True):
            # Create enriched input with tracing
            enriched_input = self.create_test_input(test_input, "debug_pipeline_trace")
            
            self.console.print("\nENRICHED_INPUT_STRUCTURE:")
            enriched_input_dict = asdict(enriched_input)
            self.console.print(JSON.from_data(enriched_input_dict, default=rich_json_default))
            
            # Process with detailed tracing
            pipeline_start = time.perf_counter()
            
            try:
                # Trace the orchestrator processing
                with self.trace_execution("orchestrator_process_globule"):
                    processed_globule = await self.orchestrator.process_globule(enriched_input)
                
                pipeline_time = (time.perf_counter() - pipeline_start) * 1000
                
                # Store with tracing
                with self.trace_execution("storage_save_globule"):
                    globule_id = await self.storage.store_globule(processed_globule)
                
                # Output complete results
                self.console.print(f"\nPIPELINE_EXECUTION_TIME_MS: {pipeline_time:.2f}")
                self.console.print(f"GENERATED_GLOBULE_ID: {globule_id}")
                
                # Deep inspection of processed globule
                globule_debug_info = {
                    "id": str(processed_globule.id),
                    "text_length": len(processed_globule.text),
                    "embedding_shape": getattr(processed_globule.embedding, 'shape', None) if processed_globule.embedding is not None else None,
                    "embedding_dtype": str(getattr(processed_globule.embedding, 'dtype', None)) if processed_globule.embedding is not None else None,
                    "embedding_confidence": processed_globule.embedding_confidence,
                    "parsed_data": processed_globule.parsed_data,
                    "parsing_confidence": processed_globule.parsing_confidence,
                    "file_decision": asdict(processed_globule.file_decision) if processed_globule.file_decision else None,
                    "processing_time_ms": processed_globule.processing_time_ms,
                    "orchestration_strategy": processed_globule.orchestration_strategy,
                    "confidence_scores": processed_globule.confidence_scores,
                    "created_at": processed_globule.created_at.isoformat(),
                    "modified_at": processed_globule.modified_at.isoformat()
                }
                
                self.console.print("\nPROCESSED_GLOBULE_COMPLETE_STRUCTURE:")
                self.console.print(JSON.from_data(globule_debug_info, default=rich_json_default))
                
                self.variable_dumps["pipeline_trace"] = {
                    "input": enriched_input_dict,
                    "output": globule_debug_info,
                    "timing": pipeline_time,
                    "globule_id": str(globule_id)
                }
                
                # Record successful test
                self.metrics.test_results.append({
                    "test": "debug_pipeline_trace",
                    "input": test_input,
                    "success": True,
                    "processing_time_ms": pipeline_time,
                    "globule_id": str(globule_id)
                })
                
            except Exception as e:
                error_info = {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "traceback": traceback.format_exc(),
                    "input_that_failed": test_input
                }
                
                self.console.print("PIPELINE_EXECUTION_ERROR:")
                self.console.print(JSON.from_data(error_info, default=rich_json_default))
                
                self.metrics.add_error(e, "complete_pipeline_trace")
                self.metrics.test_results.append({
                    "test": "debug_pipeline_trace",
                    "input": test_input,
                    "success": False,
                    "error": str(e)
                })
    
    async def _comprehensive_performance_analysis(self) -> None:
        """Perform comprehensive performance analysis with granular metrics."""
        self.console.print("\n=== COMPREHENSIVE PERFORMANCE ANALYSIS ===")
        
        with self.trace_execution("performance_analysis"):
            # Aggregate performance counter data
            perf_summary = {}
            for operation, times in self.performance_counters.items():
                if times:
                    perf_summary[operation] = {
                        "call_count": len(times),
                        "total_time_ms": sum(times),
                        "average_time_ms": sum(times) / len(times),
                        "min_time_ms": min(times),
                        "max_time_ms": max(times),
                        "std_deviation": self._calculate_std_dev(times),
                        "percentile_95": self._calculate_percentile(times, 95),
                        "percentile_99": self._calculate_percentile(times, 99)
                    }
            
            self.console.print("PERFORMANCE_COUNTERS_SUMMARY:")
            self.console.print(JSON.from_data(perf_summary, default=rich_json_default))
            
            # Execution trace summary
            trace_summary = self.execution_trace.get_trace_summary()
            self.console.print("\nEXECUTION_TRACE_SUMMARY:")
            self.console.print(JSON.from_data(trace_summary, default=rich_json_default))
            
            # Memory analysis
            current_memory = self.process.memory_info().rss / 1024 / 1024
            memory_analysis = {
                "initial_memory_mb": self.initial_memory,
                "current_memory_mb": current_memory,
                "memory_delta_mb": current_memory - self.initial_memory,
                "peak_memory_usage": max(self.memory_profiler.values()) if self.memory_profiler else current_memory
            }
            
            self.console.print("\nMEMORY_ANALYSIS:")
            self.console.print(JSON.from_data(memory_analysis, default=rich_json_default))
            
            self.variable_dumps["performance_analysis"] = {
                "performance_counters": perf_summary,
                "execution_trace": trace_summary,
                "memory_analysis": memory_analysis
            }
    
    def _calculate_std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation of values."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def _calculate_percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = (percentile / 100) * (len(sorted_values) - 1)
        if index.is_integer():
            return sorted_values[int(index)]
        else:
            lower = sorted_values[int(index)]
            upper = sorted_values[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    async def _analyze_resource_consumption(self) -> None:
        """Analyze detailed resource consumption patterns."""
        self.console.print("\n=== RESOURCE CONSUMPTION ANALYSIS ===")
        
        with self.trace_execution("resource_analysis"):
            # CPU analysis
            cpu_times = self.process.cpu_times()
            cpu_analysis = {
                "user_time": cpu_times.user,
                "system_time": cpu_times.system,
                "cpu_percent": self.process.cpu_percent(interval=0.1),
                "num_threads": self.process.num_threads(),
                "num_fds": self.process.num_fds() if hasattr(self.process, 'num_fds') else None
            }
            
            # Memory detailed analysis
            memory_info = self.process.memory_info()
            memory_analysis = {
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
                "shared_mb": getattr(memory_info, 'shared', 0) / 1024 / 1024,
                "text_mb": getattr(memory_info, 'text', 0) / 1024 / 1024,
                "data_mb": getattr(memory_info, 'data', 0) / 1024 / 1024
            }
            
            # I/O analysis (if available)
            io_analysis = {}
            try:
                io_counters = self.process.io_counters()
                io_analysis = {
                    "read_count": io_counters.read_count,
                    "write_count": io_counters.write_count,
                    "read_bytes": io_counters.read_bytes,
                    "write_bytes": io_counters.write_bytes
                }
            except (AttributeError, psutil.AccessDenied):
                io_analysis = {"error": "I/O counters not available"}
            
            resource_data = {
                "cpu_analysis": cpu_analysis,
                "memory_analysis": memory_analysis,
                "io_analysis": io_analysis
            }
            
            self.console.print("RESOURCE_CONSUMPTION_DATA:")
            self.console.print(JSON.from_data(resource_data, default=rich_json_default))
            
            self.variable_dumps["resource_analysis"] = resource_data
    
    async def _inspect_data_structures(self) -> None:
        """Inspect internal data structures and object relationships."""
        self.console.print("\n=== DATA STRUCTURE INSPECTION ===")
        
        with self.trace_execution("data_structure_inspection"):
            # Inspect configuration object
            config_inspection = {
                "config_class": str(type(self.config)),
                "config_dict": asdict(self.config) if hasattr(self.config, '__dataclass_fields__') else vars(self.config),
                "config_methods": [method for method in dir(self.config) if not method.startswith('_')]
            }
            
            # Inspect storage object
            storage_inspection = {
                "storage_class": str(type(self.storage)),
                "storage_attributes": {
                    attr: str(type(getattr(self.storage, attr)))
                    for attr in dir(self.storage)
                    if not attr.startswith('_') and not callable(getattr(self.storage, attr))
                },
                "storage_methods": [method for method in dir(self.storage) if not method.startswith('_') and callable(getattr(self.storage, method))]
            }
            
            # Inspect embedding provider
            embedding_inspection = {
                "embedding_class": str(type(self.embedding_provider)),
                "embedding_attributes": {
                    attr: str(type(getattr(self.embedding_provider, attr)))
                    for attr in dir(self.embedding_provider)
                    if not attr.startswith('_') and not callable(getattr(self.embedding_provider, attr))
                },
                "embedding_methods": [method for method in dir(self.embedding_provider) if not method.startswith('_') and callable(getattr(self.embedding_provider, method))]
            }
            
            structure_data = {
                "config_inspection": config_inspection,
                "storage_inspection": storage_inspection,
                "embedding_inspection": embedding_inspection
            }
            
            self.console.print("DATA_STRUCTURE_INSPECTION:")
            self.console.print(JSON.from_data(structure_data, default=rich_json_default))
            
            self.variable_dumps["data_structure_inspection"] = structure_data
    
    def present_results(self) -> None:
        """
        Present debug results in raw, high-fidelity format.
        
        Unlike other modes, debug mode prioritizes completeness and accuracy
        over visual appeal, providing maximum information for analysis.
        """
        self.console.print("\n" + "=" * 80)
        self.console.print("=== DEBUG MODE: COMPREHENSIVE RESULTS DUMP ===")
        self.console.print(f"TIMESTAMP: {datetime.now().isoformat()}")
        
        # Execution trace complete dump
        self.console.print("\n--- COMPLETE EXECUTION TRACE ---")
        self.console.print(f"TOTAL_TRACE_POINTS: {len(self.execution_trace.traces)}")
        
        for i, trace in enumerate(self.execution_trace.traces):
            self.console.print(f"TRACE_{i:03d}: {JSON.from_data(trace, default=rich_json_default)}")
        
        # Performance counters complete dump
        self.console.print("\n--- PERFORMANCE COUNTERS COMPLETE DUMP ---")
        for operation, times in self.performance_counters.items():
            self.console.print(f"{operation.upper()}_TIMES_MS: {times}")
        
        # Variable dumps complete
        self.console.print("\n--- VARIABLE DUMPS COMPLETE ---")
        for dump_name, dump_data in self.variable_dumps.items():
            self.console.print(f"{dump_name.upper()}_DUMP:")
            self.console.print(JSON.from_data(dump_data, default=rich_json_default))
        
        # Metrics summary
        self.console.print("\n--- METRICS SUMMARY ---")
        metrics_dict = self.metrics.to_dict()
        self.console.print(JSON.from_data(metrics_dict, default=rich_json_default))
        
        # Test results raw
        self.console.print("\n--- TEST RESULTS RAW ---")
        for test_result in self.metrics.test_results:
            self.console.print(JSON.from_data(test_result, default=rich_json_default))
        
        # Final system state
        final_system_state = self._capture_system_state()
        self.console.print("\n--- FINAL SYSTEM STATE ---")
        self.console.print(JSON.from_data(final_system_state, default=rich_json_default))
        
        # Debug mode specific summary
        debug_summary = {
            "debug_mode_version": "1.0.0",
            "total_execution_time_ms": self.metrics.total_duration_ms,
            "trace_points_captured": len(self.execution_trace.traces),
            "performance_operations_tracked": len(self.performance_counters),
            "variable_dumps_collected": len(self.variable_dumps),
            "test_results_count": len(self.metrics.test_results),
            "errors_encountered": len(self.metrics.error_log),
            "max_memory_usage_mb": max(self.memory_profiler.values()) if self.memory_profiler else 0
        }
        
        self.console.print("\n--- DEBUG_MODE_SUMMARY ---")
        self.console.print(JSON.from_data(debug_summary, default=rich_json_default))
        
        self.console.print("\n=== DEBUG MODE COMPLETE ===")
        self.console.print("RAW_DATA_FIDELITY: MAXIMUM")
        self.console.print("ANALYSIS_DEPTH: COMPREHENSIVE")
        self.console.print("DEBUG_INTERFACE: DIRECT_CODE_ACCESS")