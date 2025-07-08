"""Glass Engine Error Handling and Recovery System."""

import asyncio
import logging
import traceback
from typing import Any, Dict, List, Optional, Type, Union
from dataclasses import dataclass
from enum import Enum

from .glass import TestStatus


class ErrorSeverity(Enum):
    """Severity levels for Glass Engine errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of Glass Engine errors."""
    CONFIGURATION = "configuration"
    TEST_CASE = "test_case"
    EXECUTION = "execution"
    ASSERTION = "assertion"
    INFRASTRUCTURE = "infrastructure"
    DEPENDENCY = "dependency"
    TIMEOUT = "timeout"
    RESOURCE = "resource"


@dataclass
class ErrorInfo:
    """Information about an error that occurred during test execution."""
    error_type: str
    message: str
    severity: ErrorSeverity
    category: ErrorCategory
    test_case_id: Optional[str] = None
    trace_id: Optional[str] = None
    timestamp: Optional[str] = None
    stack_trace: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    recovery_suggestion: Optional[str] = None


class GlassEngineError(Exception):
    """Base exception for Glass Engine errors."""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 category: ErrorCategory = ErrorCategory.EXECUTION,
                 recovery_suggestion: Optional[str] = None,
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.category = category
        self.recovery_suggestion = recovery_suggestion
        self.context = context or {}
    
    def to_error_info(self, test_case_id: Optional[str] = None,
                     trace_id: Optional[str] = None) -> ErrorInfo:
        """Convert exception to ErrorInfo."""
        return ErrorInfo(
            error_type=self.__class__.__name__,
            message=self.message,
            severity=self.severity,
            category=self.category,
            test_case_id=test_case_id,
            trace_id=trace_id,
            timestamp=None,  # Will be set by error handler
            stack_trace=traceback.format_exc() if traceback.format_exc() != "NoneType: None\n" else None,
            context=self.context,
            recovery_suggestion=self.recovery_suggestion
        )


class TestCaseLoadError(GlassEngineError):
    """Error loading test case definitions."""
    
    def __init__(self, message: str, file_path: Optional[str] = None):
        super().__init__(
            message=message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.TEST_CASE,
            recovery_suggestion="Check test case YAML syntax and file permissions",
            context={"file_path": file_path} if file_path else None
        )


class TestExecutionError(GlassEngineError):
    """Error during test execution."""
    
    def __init__(self, message: str, step_index: Optional[int] = None):
        super().__init__(
            message=message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.EXECUTION,
            recovery_suggestion="Check test step configuration and system state",
            context={"step_index": step_index} if step_index is not None else None
        )


class AssertionError(GlassEngineError):
    """Assertion validation failed."""
    
    def __init__(self, message: str, assertion_type: Optional[str] = None,
                 expected: Optional[str] = None, actual: Optional[str] = None):
        super().__init__(
            message=message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.ASSERTION,
            recovery_suggestion="Review assertion configuration and expected values",
            context={
                "assertion_type": assertion_type,
                "expected": expected,
                "actual": actual
            }
        )


class TestTimeoutError(GlassEngineError):
    """Test execution timed out."""
    
    def __init__(self, message: str, timeout_seconds: Optional[int] = None):
        super().__init__(
            message=message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.TIMEOUT,
            recovery_suggestion="Consider increasing timeout values or check for hanging processes",
            context={"timeout_seconds": timeout_seconds} if timeout_seconds else None
        )


class MockSetupError(GlassEngineError):
    """Error setting up mocks."""
    
    def __init__(self, message: str, mock_name: Optional[str] = None):
        super().__init__(
            message=message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.INFRASTRUCTURE,
            recovery_suggestion="Check mock configuration and dependencies",
            context={"mock_name": mock_name} if mock_name else None
        )


class ConfigurationError(GlassEngineError):
    """Configuration-related error."""
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        super().__init__(
            message=message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.CONFIGURATION,
            recovery_suggestion="Check configuration file and environment variables",
            context={"config_key": config_key} if config_key else None
        )


class DependencyError(GlassEngineError):
    """Missing or incompatible dependency."""
    
    def __init__(self, message: str, dependency_name: Optional[str] = None):
        super().__init__(
            message=message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.DEPENDENCY,
            recovery_suggestion="Install missing dependencies or check versions",
            context={"dependency_name": dependency_name} if dependency_name else None
        )


class ResourceError(GlassEngineError):
    """Resource-related error (memory, disk, etc.)."""
    
    def __init__(self, message: str, resource_type: Optional[str] = None):
        super().__init__(
            message=message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.RESOURCE,
            recovery_suggestion="Check system resources and clean up temporary files",
            context={"resource_type": resource_type} if resource_type else None
        )


class ErrorHandler:
    """Centralized error handling for Glass Engine."""
    
    def __init__(self):
        self.logger = logging.getLogger("glass.errors")
        self.error_history: List[ErrorInfo] = []
        self.retry_strategies: Dict[Type[Exception], int] = {
            TestTimeoutError: 2,
            ResourceError: 1,
            TestExecutionError: 1
        }
    
    def handle_error(self, error: Exception, test_case_id: Optional[str] = None,
                    trace_id: Optional[str] = None) -> ErrorInfo:
        """Handle an error and return error information."""
        # Convert to GlassEngineError if needed
        if isinstance(error, GlassEngineError):
            glass_error = error
        else:
            glass_error = self._convert_to_glass_error(error)
        
        # Create error info
        error_info = glass_error.to_error_info(test_case_id, trace_id)
        error_info.timestamp = self._get_timestamp()
        
        # Log the error
        self._log_error(error_info)
        
        # Store in history
        self.error_history.append(error_info)
        
        return error_info
    
    def should_retry(self, error: Exception, attempt: int) -> bool:
        """Determine if an error should be retried."""
        error_type = type(error)
        max_retries = self.retry_strategies.get(error_type, 0)
        
        if attempt <= max_retries:
            self.logger.info(f"Retrying after {error_type.__name__} (attempt {attempt}/{max_retries})")
            return True
        
        return False
    
    def get_recovery_suggestion(self, error: Exception) -> Optional[str]:
        """Get recovery suggestion for an error."""
        if isinstance(error, GlassEngineError):
            return error.recovery_suggestion
        
        # Default suggestions for common errors
        error_type = type(error).__name__
        
        suggestions = {
            "TimeoutError": "Consider increasing timeout values",
            "ConnectionError": "Check network connectivity",
            "FileNotFoundError": "Verify file paths and permissions",
            "PermissionError": "Check file and directory permissions",
            "ImportError": "Install missing dependencies",
            "MemoryError": "Increase available memory or reduce test scope",
            "KeyError": "Check configuration keys and data structure",
            "ValueError": "Validate input parameters and data types"
        }
        
        return suggestions.get(error_type)
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get statistics about errors encountered."""
        if not self.error_history:
            return {"total_errors": 0}
        
        stats = {
            "total_errors": len(self.error_history),
            "by_severity": {},
            "by_category": {},
            "by_type": {},
            "most_common_error": None,
            "recent_errors": self.error_history[-5:]  # Last 5 errors
        }
        
        # Count by severity
        for error in self.error_history:
            severity = error.severity.value
            stats["by_severity"][severity] = stats["by_severity"].get(severity, 0) + 1
        
        # Count by category
        for error in self.error_history:
            category = error.category.value
            stats["by_category"][category] = stats["by_category"].get(category, 0) + 1
        
        # Count by type
        type_counts = {}
        for error in self.error_history:
            error_type = error.error_type
            type_counts[error_type] = type_counts.get(error_type, 0) + 1
            stats["by_type"][error_type] = type_counts[error_type]
        
        # Most common error
        if type_counts:
            most_common = max(type_counts.items(), key=lambda x: x[1])
            stats["most_common_error"] = {
                "type": most_common[0],
                "count": most_common[1]
            }
        
        return stats
    
    def clear_error_history(self):
        """Clear the error history."""
        self.error_history.clear()
    
    def _convert_to_glass_error(self, error: Exception) -> GlassEngineError:
        """Convert a regular exception to a GlassEngineError."""
        error_type = type(error).__name__
        message = str(error)
        
        # Map common exception types to Glass Engine errors
        if isinstance(error, asyncio.TimeoutError):
            return TestTimeoutError(message)
        elif isinstance(error, FileNotFoundError):
            return TestCaseLoadError(message)
        elif isinstance(error, ImportError):
            return DependencyError(message, getattr(error, 'name', None))
        elif isinstance(error, PermissionError):
            return ResourceError(message, "permission")
        elif isinstance(error, MemoryError):
            return ResourceError(message, "memory")
        elif isinstance(error, ConnectionError):
            return TestExecutionError(message)
        else:
            return GlassEngineError(
                message=f"{error_type}: {message}",
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.EXECUTION
            )
    
    def _log_error(self, error_info: ErrorInfo):
        """Log an error with appropriate level."""
        log_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }.get(error_info.severity, logging.ERROR)
        
        message = f"[{error_info.category.value.upper()}] {error_info.error_type}: {error_info.message}"
        
        if error_info.test_case_id:
            message += f" (Test: {error_info.test_case_id})"
        
        if error_info.trace_id:
            message += f" (Trace: {error_info.trace_id})"
        
        self.logger.log(log_level, message)
        
        if error_info.recovery_suggestion:
            self.logger.log(log_level, f"Recovery suggestion: {error_info.recovery_suggestion}")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()


class ErrorRecovery:
    """Error recovery strategies for Glass Engine."""
    
    def __init__(self, error_handler: ErrorHandler):
        self.error_handler = error_handler
        self.logger = logging.getLogger("glass.recovery")
    
    async def execute_with_retry(self, func, *args, max_retries: int = 3, **kwargs):
        """Execute a function with retry logic."""
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as error:
                last_error = error
                
                if attempt < max_retries and self.error_handler.should_retry(error, attempt + 1):
                    # Wait before retry (exponential backoff)
                    wait_time = 2 ** attempt
                    self.logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    # Max retries reached or error shouldn't be retried
                    break
        
        # If we get here, all retries failed
        if last_error:
            raise last_error
    
    async def execute_with_fallback(self, primary_func, fallback_func, *args, **kwargs):
        """Execute a function with fallback on error."""
        try:
            return await primary_func(*args, **kwargs)
        except Exception as error:
            error_info = self.error_handler.handle_error(error)
            
            if error_info.severity in [ErrorSeverity.LOW, ErrorSeverity.MEDIUM]:
                self.logger.warning(f"Primary function failed, trying fallback: {error_info.message}")
                return await fallback_func(*args, **kwargs)
            else:
                # Critical errors shouldn't use fallback
                raise error
    
    def create_circuit_breaker(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        """Create a circuit breaker for error recovery."""
        return CircuitBreaker(failure_threshold, recovery_timeout, self.error_handler)


class CircuitBreaker:
    """Circuit breaker pattern for error recovery."""
    
    def __init__(self, failure_threshold: int, recovery_timeout: int, error_handler: ErrorHandler):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.error_handler = error_handler
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.logger = logging.getLogger("glass.circuit_breaker")
    
    async def call(self, func, *args, **kwargs):
        """Call a function through the circuit breaker."""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
                self.logger.info("Circuit breaker moving to HALF_OPEN state")
            else:
                raise GlassEngineError(
                    "Circuit breaker is OPEN - too many failures",
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.INFRASTRUCTURE
                )
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as error:
            self._on_failure(error)
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt to reset."""
        if self.last_failure_time is None:
            return True
        
        import time
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful execution."""
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            self.failure_count = 0
            self.logger.info("Circuit breaker reset to CLOSED state")
    
    def _on_failure(self, error: Exception):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            self.logger.error(f"Circuit breaker opened after {self.failure_count} failures")
        
        self.error_handler.handle_error(error)


# Global error handler instance
error_handler = ErrorHandler()
error_recovery = ErrorRecovery(error_handler)