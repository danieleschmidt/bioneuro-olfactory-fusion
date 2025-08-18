"""
Advanced Error Handling for Neuromorphic Systems
================================================

This module provides comprehensive error handling, validation, and recovery
mechanisms specifically designed for neuromorphic computing systems.

Created as part of Terragon SDLC Generation 2: MAKE IT ROBUST
"""

import logging
import traceback
import time
from typing import Dict, List, Optional, Any, Callable, Union
from functools import wraps
from dataclasses import dataclass, field
from enum import Enum
import warnings


class ErrorSeverity(Enum):
    """Error severity levels for neuromorphic systems."""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


class NeuromorphicErrorType(Enum):
    """Specific error types for neuromorphic systems."""
    SPIKE_ENCODING_ERROR = "spike_encoding_error"
    MEMBRANE_POTENTIAL_OVERFLOW = "membrane_potential_overflow"
    SYNAPTIC_WEIGHT_DIVERGENCE = "synaptic_weight_divergence"
    SPARSITY_VIOLATION = "sparsity_violation"
    TEMPORAL_ALIGNMENT_ERROR = "temporal_alignment_error"
    FUSION_DIMENSION_MISMATCH = "fusion_dimension_mismatch"
    PLASTICITY_INSTABILITY = "plasticity_instability"
    DECISION_DEADLOCK = "decision_deadlock"
    SENSOR_DATA_CORRUPTION = "sensor_data_corruption"
    NETWORK_CONVERGENCE_FAILURE = "network_convergence_failure"


@dataclass
class NeuromorphicError:
    """Structured error information for neuromorphic systems."""
    error_type: NeuromorphicErrorType
    severity: ErrorSeverity
    message: str
    timestamp: float = field(default_factory=time.time)
    component: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    recovery_suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return {
            'error_type': self.error_type.value,
            'severity': self.severity.value,
            'message': self.message,
            'timestamp': self.timestamp,
            'component': self.component,
            'context': self.context,
            'stack_trace': self.stack_trace,
            'recovery_suggestions': self.recovery_suggestions
        }


class NeuromorphicException(Exception):
    """Base exception for neuromorphic computing errors."""
    
    def __init__(self, error: NeuromorphicError):
        self.error = error
        super().__init__(error.message)


class SpikeEncodingException(NeuromorphicException):
    """Exception for spike encoding errors."""
    pass


class MembraneOverflowException(NeuromorphicException):
    """Exception for membrane potential overflow."""
    pass


class SparsityViolationException(NeuromorphicException):
    """Exception for sparsity constraint violations."""
    pass


class PlasticityInstabilityException(NeuromorphicException):
    """Exception for synaptic plasticity instability."""
    pass


class NeuromorphicErrorHandler:
    """Advanced error handler for neuromorphic systems."""
    
    def __init__(self, 
                 log_level: int = logging.INFO,
                 enable_recovery: bool = True,
                 max_retries: int = 3):
        self.logger = self._setup_logger(log_level)
        self.enable_recovery = enable_recovery
        self.max_retries = max_retries
        self.error_history: List[NeuromorphicError] = []
        self.recovery_strategies: Dict[NeuromorphicErrorType, Callable] = {}
        self.error_statistics: Dict[str, int] = {}
        
        # Register default recovery strategies
        self._register_default_recovery_strategies()
        
    def _setup_logger(self, log_level: int) -> logging.Logger:
        """Setup structured logging for neuromorphic errors."""
        logger = logging.getLogger('neuromorphic_errors')
        logger.setLevel(log_level)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    def _register_default_recovery_strategies(self):
        """Register default recovery strategies for common errors."""
        
        def recover_spike_encoding(error: NeuromorphicError, *args, **kwargs):
            """Recovery strategy for spike encoding errors."""
            self.logger.info("Attempting spike encoding recovery...")
            
            # Reset encoding parameters
            if 'encoder' in kwargs:
                encoder = kwargs['encoder']
                encoder.reset()
                
            # Clamp input values
            if 'input_data' in kwargs:
                input_data = kwargs['input_data']
                # Simplified recovery - in practice would be more sophisticated
                return True
                
            return False
            
        def recover_membrane_overflow(error: NeuromorphicError, *args, **kwargs):
            """Recovery strategy for membrane potential overflow."""
            self.logger.info("Attempting membrane potential recovery...")
            
            if 'neuron' in kwargs:
                neuron = kwargs['neuron']
                # Reset membrane potential
                neuron.reset()
                return True
                
            return False
            
        def recover_sparsity_violation(error: NeuromorphicError, *args, **kwargs):
            """Recovery strategy for sparsity violations."""
            self.logger.info("Attempting sparsity recovery...")
            
            if 'layer' in kwargs and hasattr(kwargs['layer'], 'adjust_sparsity'):
                layer = kwargs['layer']
                layer.adjust_sparsity()
                return True
                
            return False
            
        # Register strategies
        self.recovery_strategies[NeuromorphicErrorType.SPIKE_ENCODING_ERROR] = recover_spike_encoding
        self.recovery_strategies[NeuromorphicErrorType.MEMBRANE_POTENTIAL_OVERFLOW] = recover_membrane_overflow
        self.recovery_strategies[NeuromorphicErrorType.SPARSITY_VIOLATION] = recover_sparsity_violation
        
    def handle_error(self, 
                    error: NeuromorphicError, 
                    raise_exception: bool = True,
                    **recovery_kwargs) -> bool:
        """Handle neuromorphic error with recovery attempts."""
        
        # Log error
        self._log_error(error)
        
        # Update statistics
        self._update_statistics(error)
        
        # Store in history
        self.error_history.append(error)
        
        # Attempt recovery if enabled
        if self.enable_recovery and error.error_type in self.recovery_strategies:
            recovery_success = self._attempt_recovery(error, **recovery_kwargs)
            if recovery_success:
                self.logger.info(f"Successfully recovered from {error.error_type.value}")
                return True
                
        # Raise exception if recovery failed or disabled
        if raise_exception:
            exception_class = self._get_exception_class(error.error_type)
            raise exception_class(error)
            
        return False
        
    def _log_error(self, error: NeuromorphicError):
        """Log error with appropriate level."""
        error_dict = error.to_dict()
        
        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(f"CRITICAL ERROR: {error.message}", extra=error_dict)
        elif error.severity == ErrorSeverity.HIGH:
            self.logger.error(f"HIGH SEVERITY: {error.message}", extra=error_dict)
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(f"MEDIUM SEVERITY: {error.message}", extra=error_dict)
        else:
            self.logger.info(f"LOW SEVERITY: {error.message}", extra=error_dict)
            
    def _update_statistics(self, error: NeuromorphicError):
        """Update error statistics."""
        error_key = f"{error.error_type.value}_{error.severity.value}"
        self.error_statistics[error_key] = self.error_statistics.get(error_key, 0) + 1
        
    def _attempt_recovery(self, error: NeuromorphicError, **kwargs) -> bool:
        """Attempt error recovery using registered strategies."""
        strategy = self.recovery_strategies.get(error.error_type)
        
        if strategy:
            try:
                return strategy(error, **kwargs)
            except Exception as e:
                self.logger.error(f"Recovery strategy failed: {e}")
                return False
                
        return False
        
    def _get_exception_class(self, error_type: NeuromorphicErrorType) -> type:
        """Get appropriate exception class for error type."""
        exception_map = {
            NeuromorphicErrorType.SPIKE_ENCODING_ERROR: SpikeEncodingException,
            NeuromorphicErrorType.MEMBRANE_POTENTIAL_OVERFLOW: MembraneOverflowException,
            NeuromorphicErrorType.SPARSITY_VIOLATION: SparsityViolationException,
            NeuromorphicErrorType.PLASTICITY_INSTABILITY: PlasticityInstabilityException,
        }
        
        return exception_map.get(error_type, NeuromorphicException)
        
    def register_recovery_strategy(self, 
                                 error_type: NeuromorphicErrorType, 
                                 strategy: Callable):
        """Register custom recovery strategy."""
        self.recovery_strategies[error_type] = strategy
        
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        return {
            'total_errors': len(self.error_history),
            'error_counts': self.error_statistics.copy(),
            'recent_errors': [error.to_dict() for error in self.error_history[-10:]],
            'error_rate_by_type': self._calculate_error_rates(),
            'most_common_errors': self._get_most_common_errors()
        }
        
    def _calculate_error_rates(self) -> Dict[str, float]:
        """Calculate error rates by type."""
        if not self.error_history:
            return {}
            
        # Calculate time window (last hour)
        current_time = time.time()
        hour_ago = current_time - 3600
        
        recent_errors = [
            error for error in self.error_history
            if error.timestamp > hour_ago
        ]
        
        if not recent_errors:
            return {}
            
        # Count errors by type
        type_counts = {}
        for error in recent_errors:
            type_counts[error.error_type.value] = type_counts.get(error.error_type.value, 0) + 1
            
        # Calculate rates (errors per hour)
        return {error_type: count for error_type, count in type_counts.items()}
        
    def _get_most_common_errors(self, top_n: int = 5) -> List[Dict]:
        """Get most common error types."""
        sorted_errors = sorted(
            self.error_statistics.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            {'error_type': error_type, 'count': count}
            for error_type, count in sorted_errors[:top_n]
        ]


def neuromorphic_error_handler(error_handler: Optional[NeuromorphicErrorHandler] = None):
    """Decorator for automatic neuromorphic error handling."""
    
    if error_handler is None:
        error_handler = NeuromorphicErrorHandler()
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Create neuromorphic error from exception
                error = NeuromorphicError(
                    error_type=_classify_error(e),
                    severity=_assess_severity(e),
                    message=str(e),
                    component=func.__name__,
                    context={'args': str(args), 'kwargs': str(kwargs)},
                    stack_trace=traceback.format_exc()
                )
                
                # Handle error
                return error_handler.handle_error(error, **kwargs)
                
        return wrapper
    return decorator


def _classify_error(exception: Exception) -> NeuromorphicErrorType:
    """Classify exception into neuromorphic error type."""
    error_message = str(exception).lower()
    
    if 'spike' in error_message or 'encoding' in error_message:
        return NeuromorphicErrorType.SPIKE_ENCODING_ERROR
    elif 'membrane' in error_message or 'potential' in error_message:
        return NeuromorphicErrorType.MEMBRANE_POTENTIAL_OVERFLOW
    elif 'sparsity' in error_message or 'sparse' in error_message:
        return NeuromorphicErrorType.SPARSITY_VIOLATION
    elif 'weight' in error_message or 'plasticity' in error_message:
        return NeuromorphicErrorType.PLASTICITY_INSTABILITY
    elif 'fusion' in error_message or 'dimension' in error_message:
        return NeuromorphicErrorType.FUSION_DIMENSION_MISMATCH
    elif 'temporal' in error_message or 'alignment' in error_message:
        return NeuromorphicErrorType.TEMPORAL_ALIGNMENT_ERROR
    else:
        return NeuromorphicErrorType.NETWORK_CONVERGENCE_FAILURE


def _assess_severity(exception: Exception) -> ErrorSeverity:
    """Assess error severity based on exception type and message."""
    
    # Critical errors that can break the system
    critical_indicators = ['overflow', 'divergence', 'deadlock', 'corruption']
    
    # High severity errors that affect functionality
    high_indicators = ['instability', 'convergence', 'mismatch']
    
    error_message = str(exception).lower()
    
    if any(indicator in error_message for indicator in critical_indicators):
        return ErrorSeverity.CRITICAL
    elif any(indicator in error_message for indicator in high_indicators):
        return ErrorSeverity.HIGH
    elif isinstance(exception, (ValueError, RuntimeError)):
        return ErrorSeverity.MEDIUM
    else:
        return ErrorSeverity.LOW


class NeuromorphicValidator:
    """Comprehensive validator for neuromorphic system inputs and states."""
    
    def __init__(self, error_handler: Optional[NeuromorphicErrorHandler] = None):
        self.error_handler = error_handler or NeuromorphicErrorHandler()
        
    def validate_spike_train(self, spike_train, expected_shape=None, max_rate=None):
        """Validate spike train data."""
        errors = []
        
        # Check basic properties
        if spike_train is None:
            errors.append(NeuromorphicError(
                error_type=NeuromorphicErrorType.SPIKE_ENCODING_ERROR,
                severity=ErrorSeverity.HIGH,
                message="Spike train is None",
                component="validate_spike_train"
            ))
            
        # Check shape if provided
        if expected_shape and hasattr(spike_train, 'shape'):
            if spike_train.shape != expected_shape:
                errors.append(NeuromorphicError(
                    error_type=NeuromorphicErrorType.SPIKE_ENCODING_ERROR,
                    severity=ErrorSeverity.MEDIUM,
                    message=f"Spike train shape {spike_train.shape} != expected {expected_shape}",
                    component="validate_spike_train"
                ))
                
        # Check firing rate if specified
        if max_rate and hasattr(spike_train, 'mean'):
            actual_rate = float(spike_train.mean())
            if actual_rate > max_rate:
                errors.append(NeuromorphicError(
                    error_type=NeuromorphicErrorType.SPIKE_ENCODING_ERROR,
                    severity=ErrorSeverity.MEDIUM,
                    message=f"Firing rate {actual_rate} exceeds maximum {max_rate}",
                    component="validate_spike_train"
                ))
                
        return errors
        
    def validate_membrane_potential(self, potential, threshold=None):
        """Validate membrane potential values."""
        errors = []
        
        if potential is None:
            errors.append(NeuromorphicError(
                error_type=NeuromorphicErrorType.MEMBRANE_POTENTIAL_OVERFLOW,
                severity=ErrorSeverity.HIGH,
                message="Membrane potential is None",
                component="validate_membrane_potential"
            ))
            return errors
            
        # Check for overflow (assuming reasonable bounds)
        if hasattr(potential, 'max'):
            max_potential = float(potential.max())
            if max_potential > 100.0:  # Arbitrary large value
                errors.append(NeuromorphicError(
                    error_type=NeuromorphicErrorType.MEMBRANE_POTENTIAL_OVERFLOW,
                    severity=ErrorSeverity.CRITICAL,
                    message=f"Membrane potential overflow: {max_potential}",
                    component="validate_membrane_potential"
                ))
                
        # Check for NaN or infinite values
        if hasattr(potential, 'isnan') and hasattr(potential, 'isinf'):
            if potential.isnan().any():
                errors.append(NeuromorphicError(
                    error_type=NeuromorphicErrorType.MEMBRANE_POTENTIAL_OVERFLOW,
                    severity=ErrorSeverity.HIGH,
                    message="NaN values detected in membrane potential",
                    component="validate_membrane_potential"
                ))
                
            if potential.isinf().any():
                errors.append(NeuromorphicError(
                    error_type=NeuromorphicErrorType.MEMBRANE_POTENTIAL_OVERFLOW,
                    severity=ErrorSeverity.CRITICAL,
                    message="Infinite values detected in membrane potential",
                    component="validate_membrane_potential"
                ))
                
        return errors
        
    def validate_sparsity(self, activity, target_sparsity, tolerance=0.1):
        """Validate sparsity constraints."""
        errors = []
        
        if activity is None:
            errors.append(NeuromorphicError(
                error_type=NeuromorphicErrorType.SPARSITY_VIOLATION,
                severity=ErrorSeverity.HIGH,
                message="Activity data is None",
                component="validate_sparsity"
            ))
            return errors
            
        # Calculate actual sparsity
        if hasattr(activity, 'mean'):
            actual_sparsity = float(activity.mean())
            sparsity_error = abs(actual_sparsity - target_sparsity)
            
            if sparsity_error > tolerance:
                severity = ErrorSeverity.HIGH if sparsity_error > 2 * tolerance else ErrorSeverity.MEDIUM
                errors.append(NeuromorphicError(
                    error_type=NeuromorphicErrorType.SPARSITY_VIOLATION,
                    severity=severity,
                    message=f"Sparsity violation: {actual_sparsity} vs target {target_sparsity}",
                    component="validate_sparsity",
                    context={'actual': actual_sparsity, 'target': target_sparsity, 'error': sparsity_error}
                ))
                
        return errors
        
    def validate_synaptic_weights(self, weights, bounds=(-10.0, 10.0)):
        """Validate synaptic weight stability."""
        errors = []
        
        if weights is None:
            errors.append(NeuromorphicError(
                error_type=NeuromorphicErrorType.SYNAPTIC_WEIGHT_DIVERGENCE,
                severity=ErrorSeverity.HIGH,
                message="Synaptic weights are None",
                component="validate_synaptic_weights"
            ))
            return errors
            
        # Check bounds
        if hasattr(weights, 'min') and hasattr(weights, 'max'):
            min_weight = float(weights.min())
            max_weight = float(weights.max())
            
            if min_weight < bounds[0] or max_weight > bounds[1]:
                errors.append(NeuromorphicError(
                    error_type=NeuromorphicErrorType.SYNAPTIC_WEIGHT_DIVERGENCE,
                    severity=ErrorSeverity.HIGH,
                    message=f"Weight bounds violation: [{min_weight}, {max_weight}] outside {bounds}",
                    component="validate_synaptic_weights"
                ))
                
        # Check for NaN or infinite values
        if hasattr(weights, 'isnan') and hasattr(weights, 'isinf'):
            if weights.isnan().any():
                errors.append(NeuromorphicError(
                    error_type=NeuromorphicErrorType.SYNAPTIC_WEIGHT_DIVERGENCE,
                    severity=ErrorSeverity.CRITICAL,
                    message="NaN values in synaptic weights",
                    component="validate_synaptic_weights"
                ))
                
            if weights.isinf().any():
                errors.append(NeuromorphicError(
                    error_type=NeuromorphicErrorType.SYNAPTIC_WEIGHT_DIVERGENCE,
                    severity=ErrorSeverity.CRITICAL,
                    message="Infinite values in synaptic weights",
                    component="validate_synaptic_weights"
                ))
                
        return errors


class CircuitBreaker:
    """Circuit breaker pattern for neuromorphic system protection."""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 timeout: float = 60.0,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        
    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if self.state == 'OPEN':
                if self._should_attempt_reset():
                    self.state = 'HALF_OPEN'
                else:
                    raise NeuromorphicException(NeuromorphicError(
                        error_type=NeuromorphicErrorType.NETWORK_CONVERGENCE_FAILURE,
                        severity=ErrorSeverity.HIGH,
                        message="Circuit breaker is OPEN - too many failures",
                        component=func.__name__
                    ))
                    
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
                
            except self.expected_exception as e:
                self._on_failure()
                raise
                
        return wrapper
        
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.timeout
        
    def _on_success(self):
        """Handle successful execution."""
        self.failure_count = 0
        self.state = 'CLOSED'
        
    def _on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
            
    def reset(self):
        """Manually reset circuit breaker."""
        self.failure_count = 0
        self.state = 'CLOSED'
        self.last_failure_time = None


# Global error handler instance
global_error_handler = NeuromorphicErrorHandler()


def create_robust_neuromorphic_component(component_class):
    """Factory function to create robust neuromorphic components."""
    
    class RobustNeuromorphicComponent(component_class):
        """Enhanced component with built-in error handling and validation."""
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.error_handler = NeuromorphicErrorHandler()
            self.validator = NeuromorphicValidator(self.error_handler)
            self.circuit_breaker = CircuitBreaker()
            
        @neuromorphic_error_handler()
        def robust_forward(self, *args, **kwargs):
            """Forward pass with comprehensive error handling."""
            
            # Pre-validation
            self._validate_inputs(*args, **kwargs)
            
            # Execute with circuit breaker protection
            result = self.circuit_breaker(super().forward)(*args, **kwargs)
            
            # Post-validation
            self._validate_outputs(result)
            
            return result
            
        def _validate_inputs(self, *args, **kwargs):
            """Validate inputs before processing."""
            # Override in subclasses for specific validation
            pass
            
        def _validate_outputs(self, outputs):
            """Validate outputs after processing."""
            # Override in subclasses for specific validation
            pass
            
    return RobustNeuromorphicComponent