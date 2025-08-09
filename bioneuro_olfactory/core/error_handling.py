"""Comprehensive error handling for the BioNeuro-Olfactory-Fusion framework."""

import logging
import traceback
from typing import Any, Dict, Optional, Type, Union
from functools import wraps
from enum import Enum
import sys
from pathlib import Path


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class BioNeuroError(Exception):
    """Base exception for BioNeuro-Olfactory-Fusion framework."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "UNKNOWN",
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.severity = severity
        self.context = context or {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "severity": self.severity.value,
            "context": self.context,
            "traceback": traceback.format_exc()
        }


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


class ErrorHandler:
    """Central error handling and logging system."""
    
    def __init__(self, log_file: Optional[str] = None, log_level: str = "INFO"):
        self.logger = self._setup_logger(log_file, log_level)
        self.error_counts = {}
        self.recovery_strategies = {}
        
    def _setup_logger(self, log_file: Optional[str], log_level: str) -> logging.Logger:
        """Setup centralized logging."""
        logger = logging.getLogger("bioneuro_olfactory")
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
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
        attempt_recovery: bool = True
    ) -> bool:
        """Handle an error with appropriate logging and recovery."""
        error_info = {
            "error_type": type(error).__name__,
            "message": str(error),
            "context": context or {}
        }
        
        # Count error occurrences
        error_key = f"{type(error).__name__}:{str(error)}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Log error based on type and severity
        if isinstance(error, BioNeuroError):
            error_dict = error.to_dict()
            severity = error.severity
            
            if severity == ErrorSeverity.CRITICAL:
                self.logger.critical(f"CRITICAL ERROR: {error_dict}")
            elif severity == ErrorSeverity.HIGH:
                self.logger.error(f"HIGH SEVERITY: {error_dict}")
            elif severity == ErrorSeverity.MEDIUM:
                self.logger.warning(f"MEDIUM SEVERITY: {error_dict}")
            else:
                self.logger.info(f"LOW SEVERITY: {error_dict}")
        else:
            self.logger.error(f"UNHANDLED ERROR: {error_info}")
            
        # Attempt recovery if enabled
        if attempt_recovery:
            return self._attempt_recovery(error, context)
            
        return False
        
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
                recovery_func(error, context)
                self.logger.info(f"Successfully recovered from {error_type}")
                return True
            except Exception as recovery_error:
                self.logger.error(f"Recovery failed for {error_type}: {recovery_error}")
                
        return False
        
    def register_recovery_strategy(
        self, 
        error_type: str, 
        recovery_func: callable
    ):
        """Register a recovery strategy for a specific error type."""
        self.recovery_strategies[error_type] = recovery_func
        self.logger.info(f"Registered recovery strategy for {error_type}")
        
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        return {
            "total_errors": sum(self.error_counts.values()),
            "unique_errors": len(self.error_counts),
            "error_counts": dict(self.error_counts),
            "most_common": max(self.error_counts.items(), key=lambda x: x[1]) if self.error_counts else None
        }


# Global error handler instance
_error_handler = None


def get_error_handler() -> ErrorHandler:
    """Get global error handler instance."""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler


def error_handler(
    error_types: Union[Type[Exception], tuple] = Exception,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    attempt_recovery: bool = True,
    reraise: bool = False
):
    """Decorator for automatic error handling."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except error_types as e:
                handler = get_error_handler()
                
                # Create context from function arguments
                context = {
                    "function": func.__name__,
                    "args": str(args)[:200],  # Limit length
                    "kwargs": str(kwargs)[:200]
                }
                
                # Handle the error
                recovered = handler.handle_error(e, context, attempt_recovery)
                
                # Re-raise if specified or recovery failed
                if reraise or not recovered:
                    raise
                    
                # Return None if recovery succeeded but no return value
                return None
                
        return wrapper
    return decorator


def safe_execute(
    func: callable,
    *args,
    default_return: Any = None,
    max_retries: int = 3,
    **kwargs
) -> Any:
    """Execute function safely with automatic retry."""
    handler = get_error_handler()
    
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            context = {
                "attempt": attempt + 1,
                "max_retries": max_retries,
                "function": func.__name__
            }
            
            if attempt == max_retries:
                # Final attempt failed
                handler.handle_error(e, context, attempt_recovery=False)
                handler.logger.error(f"Function {func.__name__} failed after {max_retries + 1} attempts")
                return default_return
            else:
                # Log retry attempt
                handler.logger.warning(f"Retry {attempt + 1}/{max_retries} for {func.__name__}: {e}")
                
    return default_return


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
            context={"parameter": name, "expected": expected_type.__name__, "actual": type(value).__name__}
        )
    
    # Range validation for numeric types
    if isinstance(value, (int, float)) and (min_value is not None or max_value is not None):
        if min_value is not None and value < min_value:
            raise ValidationError(
                f"{name} value {value} is below minimum {min_value}",
                error_code="VALUE_TOO_LOW",
                context={"parameter": name, "value": value, "minimum": min_value}
            )
            
        if max_value is not None and value > max_value:
            raise ValidationError(
                f"{name} value {value} is above maximum {max_value}",
                error_code="VALUE_TOO_HIGH",
                context={"parameter": name, "value": value, "maximum": max_value}
            )
    
    # Allowed values validation
    if allowed_values is not None and value not in allowed_values:
        raise ValidationError(
            f"{name} value '{value}' not in allowed values: {allowed_values}",
            error_code="INVALID_VALUE",
            context={"parameter": name, "value": value, "allowed": allowed_values}
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
                context={"parameter": name, "type": type(array).__name__}
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
                }
            )
            
    except AttributeError:
        raise ValidationError(
            f"{name} is not a valid array-like object",
            error_code="INVALID_ARRAY",
            context={"parameter": name, "type": type(array).__name__}
        )
    
    return array


# Recovery strategy examples
def sensor_recovery_strategy(error: SensorError, context: Dict[str, Any]):
    """Recovery strategy for sensor errors."""
    handler = get_error_handler()
    handler.logger.info("Attempting sensor recovery: reinitializing sensor connection")
    
    # Example recovery actions:
    # 1. Reset sensor connection
    # 2. Switch to backup sensors
    # 3. Use cached readings temporarily
    
    # This would be implemented with actual hardware interfaces


def network_recovery_strategy(error: NetworkError, context: Dict[str, Any]):
    """Recovery strategy for network errors."""
    handler = get_error_handler()
    handler.logger.info("Attempting network recovery: reinitializing model components")
    
    # Example recovery actions:
    # 1. Reset network state
    # 2. Reload model weights
    # 3. Switch to simpler model


# Register default recovery strategies
def setup_default_recovery_strategies():
    """Setup default recovery strategies."""
    handler = get_error_handler()
    handler.register_recovery_strategy("SensorError", sensor_recovery_strategy)
    handler.register_recovery_strategy("NetworkError", network_recovery_strategy)


# Initialize default strategies
setup_default_recovery_strategies()