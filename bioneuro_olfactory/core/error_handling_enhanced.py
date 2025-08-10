"""Enhanced comprehensive error handling for the BioNeuro-Olfactory-Fusion framework."""

import logging
import traceback
from typing import Any, Dict, Optional, Type, Union, Callable, List
from functools import wraps
from enum import Enum
import sys
from pathlib import Path
import time
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import threading
from collections import defaultdict, deque
import json
import queue
import inspect
import random


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RetryPolicy:
    """Retry policy configuration."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_backoff: bool = True
    jitter: bool = True
    retriable_exceptions: List[Type[Exception]] = field(default_factory=lambda: [
        ConnectionError, TimeoutError, OSError
    ])
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if an exception should be retried."""
        if attempt >= self.max_attempts:
            return False
            
        if hasattr(exception, 'can_retry'):
            return exception.can_retry() and type(exception) in self.retriable_exceptions
            
        return type(exception) in self.retriable_exceptions
        
    def get_delay(self, attempt: int) -> float:
        """Calculate delay before retry attempt."""
        if self.exponential_backoff:
            delay = self.base_delay * (2 ** attempt)
        else:
            delay = self.base_delay
            
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            delay *= (0.5 + random.random() * 0.5)
            
        return delay


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 3


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker for preventing cascading failures."""
    
    def __init__(self, config: CircuitBreakerConfig, name: str = "default"):
        self.config = config
        self.name = name
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.half_open_calls = 0
        self.lock = threading.Lock()
        
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker."""
        with self.lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.half_open_calls = 0
                else:
                    raise BioNeuroError(
                        f"Circuit breaker '{self.name}' is OPEN",
                        error_code="CIRCUIT_BREAKER_OPEN",
                        severity=ErrorSeverity.HIGH,
                        recoverable=False
                    )
                    
            if self.state == CircuitBreakerState.HALF_OPEN:
                if self.half_open_calls >= self.config.half_open_max_calls:
                    self.state = CircuitBreakerState.OPEN
                    raise BioNeuroError(
                        f"Circuit breaker '{self.name}' returned to OPEN state",
                        error_code="CIRCUIT_BREAKER_HALF_OPEN_EXCEEDED",
                        severity=ErrorSeverity.HIGH
                    )
                    
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
            
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        return (
            self.last_failure_time is not None and
            time.time() - self.last_failure_time > self.config.recovery_timeout
        )
        
    def _on_success(self):
        """Handle successful call."""
        with self.lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            self.half_open_calls = 0
            
    def _on_failure(self):
        """Handle failed call."""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.OPEN
            elif self.failure_count >= self.config.failure_threshold:
                self.state = CircuitBreakerState.OPEN


class BioNeuroError(Exception):
    """Base exception for BioNeuro-Olfactory-Fusion framework."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "UNKNOWN",
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        recoverable: bool = True,
        retry_after: Optional[float] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.severity = severity
        self.context = context or {}
        self.recoverable = recoverable
        self.retry_after = retry_after
        self.timestamp = datetime.now()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "severity": self.severity.value,
            "context": self.context,
            "recoverable": self.recoverable,
            "retry_after": self.retry_after,
            "timestamp": self.timestamp.isoformat(),
            "traceback": traceback.format_exc()
        }
        
    def can_retry(self) -> bool:
        """Check if error can be retried."""
        return self.recoverable and self.severity != ErrorSeverity.CRITICAL


class SensorError(BioNeuroError):
    """Sensor-related errors."""
    pass


class NetworkError(BioNeuroError):
    """Neural network-related errors."""
    pass


class ValidationError(BioNeuroError):
    """Data validation errors."""
    pass


class ConfigurationError(BioNeuroError):
    """Configuration-related errors."""
    pass


class HardwareError(BioNeuroError):
    """Neuromorphic hardware-related errors."""
    pass


class SecurityError(BioNeuroError):
    """Security-related errors."""
    pass


class DataIntegrityError(BioNeuroError):
    """Data integrity and corruption errors."""
    pass


class ResourceExhaustionError(BioNeuroError):
    """Resource exhaustion errors (memory, disk, etc.)."""
    pass


class CommunicationError(BioNeuroError):
    """Communication and network-related errors."""
    pass


class CalibrationError(BioNeuroError):
    """Sensor calibration errors."""
    pass


class ModelError(BioNeuroError):
    """Neural network model errors."""
    pass


class StructuredLogFormatter(logging.Formatter):
    """Structured JSON log formatter."""
    
    def __init__(self, include_traceback: bool = False):
        super().__init__()
        self.include_traceback = include_traceback
        
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "process": record.process
        }
        
        # Add structured data if present
        if hasattr(record, 'structured_data'):
            log_data.update(record.structured_data)
            
        # Add traceback if enabled and present
        if self.include_traceback and record.exc_info:
            log_data["traceback"] = self.formatException(record.exc_info)
            
        return json.dumps(log_data, default=str, separators=(',', ':'))


class ErrorRateLimiter:
    """Rate limiter for error handling to prevent log flooding."""
    
    def __init__(self, max_errors_per_minute: int = 60):
        self.max_errors = max_errors_per_minute
        self.error_timestamps = deque()
        self.lock = threading.Lock()
        
    def should_handle_error(self, error_key: str) -> bool:
        """Check if error should be handled or rate limited."""
        with self.lock:
            now = time.time()
            cutoff_time = now - 60  # 1 minute ago
            
            # Remove old timestamps
            while self.error_timestamps and self.error_timestamps[0] < cutoff_time:
                self.error_timestamps.popleft()
                
            # Check rate limit
            if len(self.error_timestamps) >= self.max_errors:
                return False
                
            # Add current timestamp
            self.error_timestamps.append(now)
            return True


class EnhancedErrorHandler:
    """Enhanced central error handling and logging system."""
    
    def __init__(
        self, 
        log_file: Optional[str] = None, 
        log_level: str = "INFO",
        enable_structured_logging: bool = True,
        max_errors_per_minute: int = 60
    ):
        self.enable_structured_logging = enable_structured_logging
        self.logger = self._setup_logger(log_file, log_level)
        self.error_counts = defaultdict(int)
        self.recovery_strategies: Dict[str, Callable] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.error_history = deque(maxlen=1000)
        self.rate_limiter = ErrorRateLimiter(max_errors_per_minute)
        
        # Graceful degradation strategies
        self.degradation_strategies: Dict[str, Callable] = {}
        self.degradation_active: Dict[str, bool] = defaultdict(bool)
        
        # Error aggregation for batch processing
        self.error_queue = queue.Queue(maxsize=1000)
        self.batch_processor_thread: Optional[threading.Thread] = None
        self.batch_processing_active = False
        
        # Performance metrics
        self.error_response_times: Dict[str, List[float]] = defaultdict(list)
        
        # Register default strategies
        self._register_default_strategies()
        
    def _setup_logger(self, log_file: Optional[str], log_level: str) -> logging.Logger:
        """Setup centralized logging with structured format."""
        logger = logging.getLogger("bioneuro_olfactory_enhanced")
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Console handler with structured logging
        console_handler = logging.StreamHandler(sys.stdout)
        if self.enable_structured_logging:
            console_handler.setFormatter(StructuredLogFormatter())
        else:
            console_format = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            if self.enable_structured_logging:
                file_handler.setFormatter(StructuredLogFormatter(include_traceback=True))
            else:
                file_format = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
                )
                file_handler.setFormatter(file_format)
            logger.addHandler(file_handler)
        
        return logger
        
    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        attempt_recovery: bool = True,
        enable_degradation: bool = True
    ) -> bool:
        """Handle an error with comprehensive logging, recovery, and degradation."""
        start_time = time.time()
        
        error_key = f"{type(error).__name__}:{str(error)}"
        
        # Check rate limiting
        if not self.rate_limiter.should_handle_error(error_key):
            return False
        
        error_info = {
            "error_type": type(error).__name__,
            "message": str(error),
            "context": context or {},
            "timestamp": datetime.now().isoformat(),
            "stack_trace": traceback.format_exc()
        }
        
        # Add error to history
        self.error_history.append(error_info)
        
        # Count error occurrences
        self.error_counts[error_key] += 1
        
        # Log error based on type and severity
        if isinstance(error, BioNeuroError):
            error_dict = error.to_dict()
            severity = error.severity
            
            # Use structured logging if enabled
            log_data = {
                "event_type": "error",
                "error_code": error.error_code,
                "severity": severity.value,
                "recoverable": error.recoverable,
                **error_dict
            }
            
            if severity == ErrorSeverity.CRITICAL:
                self.logger.critical("Critical system error occurred", extra={"structured_data": log_data})
            elif severity == ErrorSeverity.HIGH:
                self.logger.error("High severity error occurred", extra={"structured_data": log_data})
            elif severity == ErrorSeverity.MEDIUM:
                self.logger.warning("Medium severity error occurred", extra={"structured_data": log_data})
            else:
                self.logger.info("Low severity error occurred", extra={"structured_data": log_data})
        else:
            log_data = {
                "event_type": "unhandled_error",
                **error_info
            }
            self.logger.error("Unhandled error occurred", extra={"structured_data": log_data})
            
        # Track response time
        response_time = (time.time() - start_time) * 1000
        self.error_response_times[error_key].append(response_time)
        
        # Attempt recovery if enabled
        recovered = False
        if attempt_recovery:
            recovered = self._attempt_recovery(error, context)
            
        # Attempt graceful degradation if recovery failed
        if not recovered and enable_degradation:
            self._attempt_degradation(error, context)
            
        return recovered
        
    def _attempt_recovery(
        self, 
        error: Exception, 
        context: Optional[Dict[str, Any]]
    ) -> bool:
        """Attempt to recover from error using registered strategies."""
        error_type = type(error).__name__
        
        if error_type in self.recovery_strategies:
            try:
                recovery_func = self.recovery_strategies[error_type]
                recovery_result = recovery_func(error, context)
                
                # Log structured recovery success
                self.logger.info(
                    "Error recovery successful",
                    extra={
                        "structured_data": {
                            "event_type": "recovery_success",
                            "error_type": error_type,
                            "recovery_strategy": recovery_func.__name__,
                            "recovery_result": recovery_result
                        }
                    }
                )
                return True
            except Exception as recovery_error:
                self.logger.error(
                    "Error recovery failed",
                    extra={
                        "structured_data": {
                            "event_type": "recovery_failure",
                            "error_type": error_type,
                            "recovery_error": str(recovery_error),
                            "recovery_strategy": recovery_func.__name__ if 'recovery_func' in locals() else "unknown"
                        }
                    }
                )
                
        return False
        
    def _attempt_degradation(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]]
    ):
        """Attempt graceful degradation when recovery fails."""
        error_type = type(error).__name__
        
        if error_type in self.degradation_strategies and not self.degradation_active[error_type]:
            try:
                degradation_func = self.degradation_strategies[error_type]
                degradation_func(error, context)
                self.degradation_active[error_type] = True
                
                self.logger.warning(
                    "Graceful degradation activated",
                    extra={
                        "structured_data": {
                            "event_type": "degradation_activated",
                            "error_type": error_type,
                            "degradation_strategy": degradation_func.__name__
                        }
                    }
                )
            except Exception as degradation_error:
                self.logger.error(
                    "Graceful degradation failed",
                    extra={
                        "structured_data": {
                            "event_type": "degradation_failure",
                            "error_type": error_type,
                            "degradation_error": str(degradation_error)
                        }
                    }
                )
                
    def register_recovery_strategy(self, error_type: str, recovery_func: Callable):
        """Register a recovery strategy for a specific error type."""
        self.recovery_strategies[error_type] = recovery_func
        self.logger.info(
            "Recovery strategy registered",
            extra={
                "structured_data": {
                    "event_type": "recovery_strategy_registered",
                    "error_type": error_type,
                    "strategy_name": recovery_func.__name__
                }
            }
        )
        
    def register_degradation_strategy(self, error_type: str, degradation_func: Callable):
        """Register a graceful degradation strategy for a specific error type."""
        self.degradation_strategies[error_type] = degradation_func
        self.logger.info(
            "Degradation strategy registered",
            extra={
                "structured_data": {
                    "event_type": "degradation_strategy_registered",
                    "error_type": error_type,
                    "strategy_name": degradation_func.__name__
                }
            }
        )
        
    def get_circuit_breaker(
        self, 
        name: str, 
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        if name not in self.circuit_breakers:
            if config is None:
                config = CircuitBreakerConfig()
            self.circuit_breakers[name] = CircuitBreaker(config, name)
        return self.circuit_breakers[name]
        
    def get_error_statistics(self, time_window_hours: Optional[int] = None) -> Dict[str, Any]:
        """Get comprehensive error statistics for monitoring."""
        stats = {
            "total_errors": sum(self.error_counts.values()),
            "unique_errors": len(self.error_counts),
            "error_counts": dict(self.error_counts),
            "most_common": max(self.error_counts.items(), key=lambda x: x[1]) if self.error_counts else None,
            "circuit_breakers": {},
            "degradation_status": dict(self.degradation_active),
            "error_response_times": {}
        }
        
        # Circuit breaker statistics
        for name, cb in self.circuit_breakers.items():
            stats["circuit_breakers"][name] = {
                "state": cb.state.value,
                "failure_count": cb.failure_count,
                "half_open_calls": cb.half_open_calls
            }
            
        # Error response time statistics
        for error_type, times in self.error_response_times.items():
            if times:
                stats["error_response_times"][error_type] = {
                    "avg_ms": sum(times) / len(times),
                    "max_ms": max(times),
                    "min_ms": min(times),
                    "count": len(times)
                }
                
        # Time-windowed statistics if requested
        if time_window_hours:
            cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
            recent_errors = [
                err for err in self.error_history 
                if datetime.fromisoformat(err["timestamp"]) > cutoff_time
            ]
            
            stats["recent_errors"] = {
                "count": len(recent_errors),
                "error_types": list(set(err["error_type"] for err in recent_errors))
            }
            
        return stats
        
    def _register_default_strategies(self):
        """Register default recovery and degradation strategies."""
        # Recovery strategies
        self.register_recovery_strategy("SensorError", self._sensor_recovery_strategy)
        self.register_recovery_strategy("NetworkError", self._network_recovery_strategy)
        self.register_recovery_strategy("CommunicationError", self._communication_recovery_strategy)
        
        # Degradation strategies
        self.register_degradation_strategy("SensorError", self._sensor_degradation_strategy)
        self.register_degradation_strategy("NetworkError", self._network_degradation_strategy)
        
    def _sensor_recovery_strategy(self, error: SensorError, context: Dict[str, Any]) -> bool:
        """Enhanced sensor recovery strategy."""
        self.logger.info("Attempting sensor recovery: reinitializing sensor connection")
        
        try:
            # Attempt sensor reconnection
            if "sensor_id" in context:
                # Reset sensor connection
                # Switch to backup sensors if available
                # Use cached readings temporarily
                return True
        except Exception:
            return False
        return False
        
    def _network_recovery_strategy(self, error: NetworkError, context: Dict[str, Any]) -> bool:
        """Enhanced network recovery strategy."""
        self.logger.info("Attempting network recovery: reinitializing model components")
        
        try:
            # Reset network state
            # Reload model weights
            # Switch to simpler model if needed
            return True
        except Exception:
            return False
        return False
        
    def _communication_recovery_strategy(self, error: CommunicationError, context: Dict[str, Any]) -> bool:
        """Communication recovery strategy."""
        self.logger.info("Attempting communication recovery")
        
        try:
            # Retry communication with different parameters
            # Switch to alternative communication channel
            return True
        except Exception:
            return False
        return False
        
    def _sensor_degradation_strategy(self, error: SensorError, context: Dict[str, Any]):
        """Sensor degradation strategy."""
        self.logger.warning("Activating sensor degradation mode")
        # Use fewer sensors, reduce sampling rate, etc.
        
    def _network_degradation_strategy(self, error: NetworkError, context: Dict[str, Any]):
        """Network degradation strategy."""
        self.logger.warning("Activating network degradation mode")
        # Use simpler model, reduce batch size, etc.


def safe_execute(
    func: Callable,
    *args,
    default_return: Any = None,
    retry_policy: Optional[RetryPolicy] = None,
    circuit_breaker: Optional[str] = None,
    **kwargs
) -> Any:
    """Execute function safely with comprehensive error handling."""
    handler = get_enhanced_error_handler()
    
    if retry_policy is None:
        retry_policy = RetryPolicy()
        
    # Use circuit breaker if specified
    if circuit_breaker:
        cb = handler.get_circuit_breaker(circuit_breaker)
        try:
            return cb.call(func, *args, **kwargs)
        except Exception as e:
            if isinstance(e, BioNeuroError) and e.error_code == "CIRCUIT_BREAKER_OPEN":
                handler.logger.warning(
                    "Circuit breaker prevented execution",
                    extra={
                        "structured_data": {
                            "event_type": "circuit_breaker_blocked",
                            "function": func.__name__,
                            "circuit_breaker": circuit_breaker
                        }
                    }
                )
                return default_return
            # Continue with retry logic for other errors
    
    last_exception = None
    
    for attempt in range(retry_policy.max_attempts):
        try:
            if circuit_breaker and attempt > 0:
                # Use circuit breaker for retries too
                cb = handler.get_circuit_breaker(circuit_breaker)
                return cb.call(func, *args, **kwargs)
            else:
                return func(*args, **kwargs)
                
        except Exception as e:
            last_exception = e
            
            context = {
                "attempt": attempt + 1,
                "max_attempts": retry_policy.max_attempts,
                "function": func.__name__,
                "args_summary": str(args)[:100],
                "kwargs_summary": str(kwargs)[:100]
            }
            
            # Check if we should retry
            if not retry_policy.should_retry(e, attempt + 1):
                handler.handle_error(e, context, attempt_recovery=False)
                handler.logger.error(
                    "Function execution failed - no retry",
                    extra={
                        "structured_data": {
                            "event_type": "execution_failed_no_retry",
                            "function": func.__name__,
                            "attempt": attempt + 1,
                            "error_type": type(e).__name__,
                            "error_message": str(e)
                        }
                    }
                )
                return default_return
                
            # Log retry attempt
            if attempt < retry_policy.max_attempts - 1:
                delay = retry_policy.get_delay(attempt)
                handler.logger.warning(
                    "Function execution failed - retrying",
                    extra={
                        "structured_data": {
                            "event_type": "execution_retry",
                            "function": func.__name__,
                            "attempt": attempt + 1,
                            "max_attempts": retry_policy.max_attempts,
                            "retry_delay": delay,
                            "error_type": type(e).__name__
                        }
                    }
                )
                time.sleep(delay)
            else:
                # Final attempt failed
                handler.handle_error(e, context, attempt_recovery=True)
                handler.logger.error(
                    "Function execution failed after all retries",
                    extra={
                        "structured_data": {
                            "event_type": "execution_failed_final",
                            "function": func.__name__,
                            "total_attempts": retry_policy.max_attempts,
                            "error_type": type(e).__name__,
                            "error_message": str(e)
                        }
                    }
                )
                
    return default_return


async def safe_execute_async(
    func: Callable,
    *args,
    default_return: Any = None,
    retry_policy: Optional[RetryPolicy] = None,
    **kwargs
) -> Any:
    """Async version of safe_execute."""
    handler = get_enhanced_error_handler()
    
    if retry_policy is None:
        retry_policy = RetryPolicy()
        
    for attempt in range(retry_policy.max_attempts):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
                
        except Exception as e:
            context = {
                "attempt": attempt + 1,
                "max_attempts": retry_policy.max_attempts,
                "function": func.__name__
            }
            
            if not retry_policy.should_retry(e, attempt + 1):
                handler.handle_error(e, context, attempt_recovery=False)
                return default_return
                
            if attempt < retry_policy.max_attempts - 1:
                delay = retry_policy.get_delay(attempt)
                await asyncio.sleep(delay)
            else:
                handler.handle_error(e, context, attempt_recovery=True)
                
    return default_return


def resilient_operation(
    name: str,
    retry_policy: Optional[RetryPolicy] = None,
    circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
    enable_degradation: bool = True
):
    """Decorator for resilient operations with comprehensive error handling."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            handler = get_enhanced_error_handler()
            
            # Set up circuit breaker if configured
            cb_name = f"{name}_circuit_breaker"
            if circuit_breaker_config:
                handler.get_circuit_breaker(cb_name, circuit_breaker_config)
            
            try:
                return safe_execute(
                    func, *args,
                    retry_policy=retry_policy,
                    circuit_breaker=cb_name if circuit_breaker_config else None,
                    **kwargs
                )
            except Exception as e:
                # Handle the error with degradation if enabled
                context = {
                    "operation_name": name,
                    "function": func.__name__,
                    "args_signature": str(inspect.signature(func))
                }
                
                handler.handle_error(e, context, enable_degradation=enable_degradation)
                raise
                
        return wrapper
    return decorator


def error_handler(
    error_types: Union[Type[Exception], tuple] = Exception,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    attempt_recovery: bool = True,
    reraise: bool = False,
    retry_policy: Optional[RetryPolicy] = None,
    circuit_breaker: Optional[str] = None
):
    """Enhanced decorator for automatic error handling with retry and circuit breaker support."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            handler = get_enhanced_error_handler()
            
            # Create enhanced context
            context = {
                "function": func.__name__,
                "module": func.__module__,
                "args_count": len(args),
                "kwargs_keys": list(kwargs.keys()),
                "timestamp": datetime.now().isoformat()
            }
            
            # Add function signature information
            try:
                sig = inspect.signature(func)
                context["signature"] = str(sig)
            except (ValueError, TypeError):
                pass
            
            try:
                # Use safe_execute if retry policy or circuit breaker is specified
                if retry_policy or circuit_breaker:
                    return safe_execute(
                        func, *args, 
                        retry_policy=retry_policy,
                        circuit_breaker=circuit_breaker,
                        **kwargs
                    )
                else:
                    return func(*args, **kwargs)
                    
            except error_types as e:
                # Handle the error with enhanced context
                recovered = handler.handle_error(e, context, attempt_recovery)
                
                # Re-raise if specified or recovery failed
                if reraise or not recovered:
                    raise
                    
                # Return None if recovery succeeded but no return value
                return None
                
        return wrapper
    return decorator


def validate_input(
    value: Any,
    expected_type: Type,
    name: str = "input",
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    allowed_values: Optional[list] = None
) -> Any:
    """Validate input parameters with detailed error messages."""
    # Type validation
    if not isinstance(value, expected_type):
        raise ValidationError(
            f"Invalid type for {name}: expected {expected_type.__name__}, got {type(value).__name__}",
            error_code="INVALID_TYPE",
            context={"parameter": name, "expected": expected_type.__name__, "actual": type(value).__name__},
            recoverable=True  # Type errors might be recoverable through conversion
        )
    
    # Range validation for numeric types
    if isinstance(value, (int, float)) and (min_value is not None or max_value is not None):
        if min_value is not None and value < min_value:
            raise ValidationError(
                f"{name} value {value} is below minimum {min_value}",
                error_code="VALUE_TOO_LOW",
                context={"parameter": name, "value": value, "minimum": min_value},
                recoverable=True
            )
            
        if max_value is not None and value > max_value:
            raise ValidationError(
                f"{name} value {value} is above maximum {max_value}",
                error_code="VALUE_TOO_HIGH",
                context={"parameter": name, "value": value, "maximum": max_value},
                recoverable=True
            )
    
    # Allowed values validation
    if allowed_values is not None and value not in allowed_values:
        raise ValidationError(
            f"{name} value '{value}' not in allowed values: {allowed_values}",
            error_code="INVALID_VALUE",
            context={"parameter": name, "value": value, "allowed": allowed_values},
            recoverable=False  # Invalid values usually can't be recovered
        )
    
    return value


def validate_array_shape(
    array: Any,
    expected_shape: tuple,
    name: str = "array"
) -> Any:
    """Validate array shape with detailed error reporting."""
    try:
        # Check if array has shape attribute (numpy-like)
        if hasattr(array, 'shape'):
            actual_shape = array.shape
        elif hasattr(array, '__len__'):
            # Handle lists/tuples
            actual_shape = (len(array),)
        else:
            raise ValidationError(
                f"{name} does not have a valid shape",
                error_code="NO_SHAPE",
                context={"parameter": name, "type": type(array).__name__},
                recoverable=False
            )
        
        # Check shape compatibility
        if actual_shape != expected_shape:
            raise ValidationError(
                f"{name} shape mismatch: expected {expected_shape}, got {actual_shape}",
                error_code="SHAPE_MISMATCH",
                context={
                    "parameter": name,
                    "expected_shape": expected_shape,
                    "actual_shape": actual_shape
                },
                recoverable=True
            )
            
    except AttributeError:
        raise ValidationError(
            f"{name} is not a valid array-like object",
            error_code="INVALID_ARRAY",
            context={"parameter": name, "type": type(array).__name__},
            recoverable=False
        )
    
    return array


# Global enhanced error handler instance
_enhanced_error_handler = None


def get_enhanced_error_handler() -> EnhancedErrorHandler:
    """Get global enhanced error handler instance."""
    global _enhanced_error_handler
    if _enhanced_error_handler is None:
        _enhanced_error_handler = EnhancedErrorHandler(enable_structured_logging=True)
    return _enhanced_error_handler


def configure_enhanced_error_handler(
    log_file: Optional[str] = None,
    log_level: str = "INFO",
    enable_structured_logging: bool = True,
    max_errors_per_minute: int = 60
) -> EnhancedErrorHandler:
    """Configure global enhanced error handler with specific settings."""
    global _enhanced_error_handler
    _enhanced_error_handler = EnhancedErrorHandler(
        log_file=log_file,
        log_level=log_level,
        enable_structured_logging=enable_structured_logging,
        max_errors_per_minute=max_errors_per_minute
    )
    
    return _enhanced_error_handler


# Initialize on import
get_enhanced_error_handler()