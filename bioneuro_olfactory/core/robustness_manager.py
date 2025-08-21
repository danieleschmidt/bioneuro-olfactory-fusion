"""
Robustness Manager for Generation 2: MAKE IT ROBUST
Comprehensive error handling, validation, and fault tolerance for neuromorphic systems
"""

import sys
import traceback
import logging
import time
import functools
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass
from enum import Enum
import hashlib
import json


class SeverityLevel(Enum):
    """Severity levels for robustness events."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RobustnessEvent:
    """Event data for robustness monitoring."""
    timestamp: float
    event_type: str
    severity: SeverityLevel
    component: str
    message: str
    details: Optional[Dict] = None
    recovery_action: Optional[str] = None
    resolved: bool = False


class RobustnessLogger:
    """Enhanced logging for robustness events."""
    
    def __init__(self, name: str = "bioneuro_robustness"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Create console handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.events = []
    
    def log_event(self, event: RobustnessEvent):
        """Log a robustness event."""
        self.events.append(event)
        
        # Map severity to logging level
        level_map = {
            SeverityLevel.LOW: logging.INFO,
            SeverityLevel.MEDIUM: logging.WARNING,
            SeverityLevel.HIGH: logging.ERROR,
            SeverityLevel.CRITICAL: logging.CRITICAL
        }
        
        level = level_map.get(event.severity, logging.INFO)
        message = f"[{event.component}] {event.event_type}: {event.message}"
        
        if event.details:
            message += f" | Details: {event.details}"
        
        self.logger.log(level, message)
    
    def get_recent_events(self, max_events: int = 100) -> List[RobustnessEvent]:
        """Get recent robustness events."""
        return self.events[-max_events:]
    
    def get_critical_events(self) -> List[RobustnessEvent]:
        """Get all unresolved critical events."""
        return [e for e in self.events 
                if e.severity == SeverityLevel.CRITICAL and not e.resolved]


class InputValidator:
    """Comprehensive input validation for neuromorphic components."""
    
    @staticmethod
    def validate_sensor_data(data: Any, expected_shape: Optional[tuple] = None) -> bool:
        """Validate sensor input data."""
        try:
            # Check if data exists
            if data is None:
                raise ValueError("Sensor data cannot be None")
            
            # Check for numpy array or tensor-like structure
            if hasattr(data, 'shape'):
                if expected_shape and data.shape != expected_shape:
                    raise ValueError(f"Data shape {data.shape} doesn't match expected {expected_shape}")
                
                # Check for NaN or infinite values
                if hasattr(data, 'isfinite'):
                    if not data.isfinite().all():
                        raise ValueError("Sensor data contains NaN or infinite values")
            
            # Check for list/array structure
            elif isinstance(data, (list, tuple)):
                if len(data) == 0:
                    raise ValueError("Sensor data cannot be empty")
                
                # Check for numeric values
                for i, value in enumerate(data):
                    if not isinstance(value, (int, float)):
                        raise ValueError(f"Non-numeric value at index {i}: {value}")
                    
                    if not (-1000 <= value <= 1000):  # Reasonable sensor range
                        raise ValueError(f"Sensor value {value} outside reasonable range [-1000, 1000]")
            
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
            
            return True
            
        except Exception as e:
            return False
    
    @staticmethod
    def validate_network_config(config: Dict[str, Any]) -> bool:
        """Validate network configuration parameters."""
        try:
            required_fields = [
                'num_receptors', 'num_projection_neurons', 
                'num_kenyon_cells', 'num_gas_classes'
            ]
            
            # Check required fields
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Missing required field: {field}")
                
                value = config[field]
                if not isinstance(value, int) or value <= 0:
                    raise ValueError(f"Field {field} must be positive integer, got {value}")
            
            # Check temporal parameters
            if 'tau_membrane' in config:
                tau = config['tau_membrane']
                if not (1.0 <= tau <= 200.0):
                    raise ValueError(f"tau_membrane {tau} outside reasonable range [1.0, 200.0]")
            
            # Check sparsity
            if 'sparsity_target' in config:
                sparsity = config['sparsity_target']
                if not (0.001 <= sparsity <= 0.5):
                    raise ValueError(f"sparsity_target {sparsity} outside range [0.001, 0.5]")
            
            return True
            
        except Exception as e:
            return False
    
    @staticmethod
    def sanitize_file_path(path: str) -> str:
        """Sanitize file paths to prevent directory traversal."""
        import os
        
        # Remove any path traversal attempts
        path = path.replace('..', '')
        path = path.replace('//', '/')
        
        # Ensure path is relative and safe
        path = os.path.normpath(path)
        
        # Remove leading slashes to make it relative
        while path.startswith('/'):
            path = path[1:]
        
        return path


class ErrorRecoveryManager:
    """Manages error recovery strategies for neuromorphic components."""
    
    def __init__(self, logger: RobustnessLogger):
        self.logger = logger
        self.recovery_strategies = {}
        self.fallback_configs = {}
    
    def register_recovery_strategy(self, component: str, strategy: Callable):
        """Register a recovery strategy for a component."""
        self.recovery_strategies[component] = strategy
    
    def register_fallback_config(self, component: str, config: Dict[str, Any]):
        """Register a fallback configuration for a component."""
        self.fallback_configs[component] = config
    
    def attempt_recovery(self, component: str, error: Exception) -> bool:
        """Attempt to recover from an error in a component."""
        try:
            # Log the error
            event = RobustnessEvent(
                timestamp=time.time(),
                event_type="ERROR_RECOVERY_ATTEMPT",
                severity=SeverityLevel.HIGH,
                component=component,
                message=f"Attempting recovery from {type(error).__name__}: {str(error)}",
                details={'error_type': type(error).__name__}
            )
            self.logger.log_event(event)
            
            # Try component-specific recovery strategy
            if component in self.recovery_strategies:
                strategy = self.recovery_strategies[component]
                result = strategy(error)
                
                if result:
                    self.logger.log_event(RobustnessEvent(
                        timestamp=time.time(),
                        event_type="ERROR_RECOVERY_SUCCESS",
                        severity=SeverityLevel.MEDIUM,
                        component=component,
                        message="Recovery strategy successful",
                        recovery_action="component_specific_strategy"
                    ))
                    return True
            
            # Try fallback configuration
            if component in self.fallback_configs:
                self.logger.log_event(RobustnessEvent(
                    timestamp=time.time(),
                    event_type="FALLBACK_CONFIG_APPLIED",
                    severity=SeverityLevel.MEDIUM,
                    component=component,
                    message="Applied fallback configuration",
                    recovery_action="fallback_configuration"
                ))
                return True
            
            return False
            
        except Exception as recovery_error:
            self.logger.log_event(RobustnessEvent(
                timestamp=time.time(),
                event_type="RECOVERY_FAILED",
                severity=SeverityLevel.CRITICAL,
                component=component,
                message=f"Recovery attempt failed: {str(recovery_error)}",
                details={'original_error': str(error), 'recovery_error': str(recovery_error)}
            ))
            return False


class SecurityManager:
    """Security management for neuromorphic systems."""
    
    def __init__(self, logger: RobustnessLogger):
        self.logger = logger
        self.access_log = []
        self.rate_limits = {}
        self.max_rate_per_minute = 1000  # Maximum operations per minute
    
    def validate_data_integrity(self, data: Any, expected_hash: Optional[str] = None) -> bool:
        """Validate data integrity using checksums."""
        try:
            # Convert data to string for hashing
            if hasattr(data, 'tobytes'):
                data_str = data.tobytes()
            else:
                data_str = str(data).encode('utf-8')
            
            # Calculate hash
            current_hash = hashlib.sha256(data_str).hexdigest()
            
            # If expected hash provided, compare
            if expected_hash:
                if current_hash != expected_hash:
                    self.logger.log_event(RobustnessEvent(
                        timestamp=time.time(),
                        event_type="DATA_INTEGRITY_VIOLATION",
                        severity=SeverityLevel.HIGH,
                        component="SecurityManager",
                        message="Data integrity check failed",
                        details={'expected_hash': expected_hash, 'actual_hash': current_hash}
                    ))
                    return False
            
            return True
            
        except Exception as e:
            self.logger.log_event(RobustnessEvent(
                timestamp=time.time(),
                event_type="INTEGRITY_CHECK_ERROR",
                severity=SeverityLevel.MEDIUM,
                component="SecurityManager",
                message=f"Failed to verify data integrity: {str(e)}"
            ))
            return False
    
    def check_rate_limit(self, component: str) -> bool:
        """Check if component is within rate limits."""
        current_time = time.time()
        
        # Clean old entries (older than 1 minute)
        if component in self.rate_limits:
            self.rate_limits[component] = [
                timestamp for timestamp in self.rate_limits[component]
                if current_time - timestamp < 60.0
            ]
        else:
            self.rate_limits[component] = []
        
        # Check current rate
        current_rate = len(self.rate_limits[component])
        
        if current_rate >= self.max_rate_per_minute:
            self.logger.log_event(RobustnessEvent(
                timestamp=current_time,
                event_type="RATE_LIMIT_EXCEEDED",
                severity=SeverityLevel.HIGH,
                component=component,
                message=f"Rate limit exceeded: {current_rate} ops/min",
                details={'rate_limit': self.max_rate_per_minute}
            ))
            return False
        
        # Add current timestamp
        self.rate_limits[component].append(current_time)
        return True
    
    def sanitize_user_input(self, user_input: str) -> str:
        """Sanitize user input to prevent injection attacks."""
        if not isinstance(user_input, str):
            user_input = str(user_input)
        
        # Remove potentially dangerous characters
        dangerous_chars = ['<', '>', '&', '"', "'", '\\', '/', '`', ';', '|']
        sanitized = user_input
        
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        
        # Limit length
        max_length = 1000
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
            
            self.logger.log_event(RobustnessEvent(
                timestamp=time.time(),
                event_type="INPUT_TRUNCATED",
                severity=SeverityLevel.LOW,
                component="SecurityManager",
                message=f"User input truncated to {max_length} characters"
            ))
        
        return sanitized


def robust_execution(component_name: str, logger: Optional[RobustnessLogger] = None):
    """Decorator for robust execution of neuromorphic functions."""
    if logger is None:
        logger = RobustnessLogger()
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Log function entry
                logger.log_event(RobustnessEvent(
                    timestamp=time.time(),
                    event_type="FUNCTION_ENTRY",
                    severity=SeverityLevel.LOW,
                    component=component_name,
                    message=f"Entering function {func.__name__}"
                ))
                
                # Execute function
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Log successful execution
                logger.log_event(RobustnessEvent(
                    timestamp=time.time(),
                    event_type="FUNCTION_SUCCESS",
                    severity=SeverityLevel.LOW,
                    component=component_name,
                    message=f"Function {func.__name__} completed successfully",
                    details={'execution_time_ms': execution_time * 1000}
                ))
                
                return result
                
            except Exception as e:
                # Log error
                logger.log_event(RobustnessEvent(
                    timestamp=time.time(),
                    event_type="FUNCTION_ERROR",
                    severity=SeverityLevel.HIGH,
                    component=component_name,
                    message=f"Function {func.__name__} failed: {str(e)}",
                    details={
                        'error_type': type(e).__name__,
                        'traceback': traceback.format_exc()
                    }
                ))
                
                # Re-raise the exception
                raise
        
        return wrapper
    return decorator


class RobustnessManager:
    """Central manager for all robustness features."""
    
    def __init__(self):
        self.logger = RobustnessLogger()
        self.input_validator = InputValidator()
        self.error_recovery = ErrorRecoveryManager(self.logger)
        self.security_manager = SecurityManager(self.logger)
        
        # Initialize with default recovery strategies
        self._setup_default_recovery_strategies()
    
    def _setup_default_recovery_strategies(self):
        """Setup default recovery strategies."""
        
        def sensor_recovery(error: Exception) -> bool:
            """Default sensor recovery strategy."""
            if "NaN" in str(error) or "infinite" in str(error):
                # Could implement data interpolation or use last known good values
                return True
            return False
        
        def network_recovery(error: Exception) -> bool:
            """Default network recovery strategy."""
            if "dimension" in str(error).lower():
                # Could implement automatic dimension adjustment
                return True
            return False
        
        self.error_recovery.register_recovery_strategy("sensor", sensor_recovery)
        self.error_recovery.register_recovery_strategy("network", network_recovery)
        
        # Register fallback configurations
        self.error_recovery.register_fallback_config("projection_neurons", {
            'num_receptors': 6,
            'num_projection_neurons': 100,  # Reduced for safety
            'tau_membrane': 20.0
        })
        
        self.error_recovery.register_fallback_config("kenyon_cells", {
            'num_projection_inputs': 100,
            'num_kenyon_cells': 500,  # Reduced for safety
            'sparsity_target': 0.1
        })
    
    def validate_and_execute(self, component: str, func: Callable, 
                           data: Any = None, config: Dict = None) -> Any:
        """Validate inputs and execute function with robustness features."""
        
        # Security: Check rate limits
        if not self.security_manager.check_rate_limit(component):
            raise RuntimeError(f"Rate limit exceeded for component {component}")
        
        # Validation: Check inputs
        if data is not None:
            if not self.input_validator.validate_sensor_data(data):
                raise ValueError(f"Invalid sensor data for component {component}")
        
        if config is not None:
            if not self.input_validator.validate_network_config(config):
                raise ValueError(f"Invalid configuration for component {component}")
        
        # Execute with error handling
        try:
            return func()
        except Exception as e:
            # Attempt recovery
            if self.error_recovery.attempt_recovery(component, e):
                self.logger.log_event(RobustnessEvent(
                    timestamp=time.time(),
                    event_type="RECOVERED_EXECUTION",
                    severity=SeverityLevel.MEDIUM,
                    component=component,
                    message="Execution continued after recovery"
                ))
                # Could retry with fallback configuration here
                return None  # Or appropriate fallback result
            else:
                # Recovery failed, re-raise
                raise
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        recent_events = self.logger.get_recent_events(50)
        critical_events = self.logger.get_critical_events()
        
        # Calculate health metrics
        error_count = len([e for e in recent_events if e.event_type.endswith("ERROR")])
        success_count = len([e for e in recent_events if e.event_type.endswith("SUCCESS")])
        
        health_score = 100.0
        if error_count + success_count > 0:
            health_score = (success_count / (error_count + success_count)) * 100
        
        # Reduce score for critical events
        health_score -= len(critical_events) * 20
        health_score = max(0, health_score)
        
        return {
            'health_score': health_score,
            'recent_errors': error_count,
            'recent_successes': success_count,
            'critical_events': len(critical_events),
            'status': 'HEALTHY' if health_score >= 80 else 'DEGRADED' if health_score >= 50 else 'CRITICAL'
        }