"""Advanced robustness decorators and wrappers for neuromorphic models.

This module provides comprehensive decorators for adding robustness features
to neuromorphic spiking neural network models, including error handling,
logging, health monitoring, and input validation.
"""

import functools
import time
import traceback
from typing import Dict, Any, Optional, List, Callable, Union
import logging
from contextlib import contextmanager

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

from .error_handling_enhanced import (
    EnhancedErrorHandler, BioNeuroError, NetworkError, 
    ValidationError, SecurityError, ErrorSeverity
)
from .logging_enhanced import info, error, warning, critical, security, performance
from .health_monitoring_enhanced import HealthMonitor
from .validation_enhanced import InputValidator
from ..security.input_validation import get_input_validator, get_sensor_validator, get_network_validator


class RobustnessConfig:
    """Configuration for robustness features."""
    
    def __init__(
        self,
        enable_error_handling: bool = True,
        enable_logging: bool = True,
        enable_health_monitoring: bool = True,
        enable_input_validation: bool = True,
        enable_security_checks: bool = True,
        fallback_strategy: str = "zero_output",
        max_retries: int = 3,
        timeout_seconds: float = 30.0,
        memory_threshold_mb: float = 1000.0,
        log_level: str = "INFO"
    ):
        self.enable_error_handling = enable_error_handling
        self.enable_logging = enable_logging
        self.enable_health_monitoring = enable_health_monitoring
        self.enable_input_validation = enable_input_validation
        self.enable_security_checks = enable_security_checks
        self.fallback_strategy = fallback_strategy
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.memory_threshold_mb = memory_threshold_mb
        self.log_level = log_level


# Global robustness configuration
_global_config = RobustnessConfig()


def set_global_robustness_config(config: RobustnessConfig):
    """Set global robustness configuration."""
    global _global_config
    _global_config = config


def get_global_robustness_config() -> RobustnessConfig:
    """Get global robustness configuration."""
    return _global_config


@contextmanager
def robust_context(component_name: str, config: Optional[RobustnessConfig] = None):
    """Context manager for robust processing with comprehensive monitoring."""
    if config is None:
        config = _global_config
    
    health_monitor = None
    start_time = None
    
    try:
        # Initialize health monitoring
        if config.enable_health_monitoring:
            health_monitor = HealthMonitor(component_name)
            health_monitor.check_memory_usage()
        
        # Log processing start
        if config.enable_logging:
            info(f"Starting processing for {component_name}")
            start_time = time.time()
        
        yield health_monitor
        
        # Log successful completion
        if config.enable_logging and start_time is not None:
            processing_time = time.time() - start_time
            performance(f"Processing completed for {component_name}", structured_data={"processing_time": processing_time})
            
    except Exception as e:
        if health_monitor:
            health_monitor.record_error(str(e))
        
        if config.enable_logging:
            critical(
                f"{component_name} processing failed: {str(e)}", structured_data={"component": component_name, "severity": ErrorSeverity.CRITICAL}
            )
        
        raise
    
    finally:
        if health_monitor and config.enable_health_monitoring:
            final_health = health_monitor.get_health_status()
            if config.enable_logging:
                info(f"Health status for {component_name}", structured_data={"component": component_name, "health": final_health})


def robust_neuromorphic_model(
    fallback_strategy: str = "zero_output",
    enable_retries: bool = True,
    max_retries: int = 3,
    config: Optional[RobustnessConfig] = None
):
    """
    Comprehensive robustness decorator for neuromorphic models.
    
    Args:
        fallback_strategy: Strategy for handling complete failures
        enable_retries: Whether to retry on recoverable errors
        max_retries: Maximum number of retry attempts
        config: Custom robustness configuration
    """
    if config is None:
        config = _global_config
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            component_name = f"{self.__class__.__name__}.{func.__name__}"
            
            with robust_context(component_name, config) as health_monitor:
                
                # Input validation
                if config.enable_input_validation:
                    try:
                        _validate_neuromorphic_inputs(args, kwargs)
                    except ValidationError as e:
                        if config.enable_logging:
                            error(f"Input validation failed: {str(e)}")
                        if fallback_strategy == "raise":
                            raise
                        return _create_fallback_output(self, fallback_strategy, args, kwargs)
                
                # Security checks
                if config.enable_security_checks:
                    try:
                        _perform_security_checks(args, kwargs)
                    except SecurityError as e:
                        if config.enable_logging:
                            security(str(e))
                        raise  # Always raise security violations
                
                # Main processing with retries
                last_exception = None
                for attempt in range(max_retries if enable_retries else 1):
                    try:
                        # Memory check before processing
                        if health_monitor:
                            health_monitor.check_memory_usage()
                        
                        # Execute the function
                        with _timeout_context(config.timeout_seconds):
                            result = func(self, *args, **kwargs)
                        
                        # Validate output
                        if config.enable_input_validation:
                            _validate_neuromorphic_output(result)
                        
                        # Add robustness metadata
                        if isinstance(result, dict):
                            result['robustness_metadata'] = {
                                'attempts': attempt + 1,
                                'health_status': health_monitor.get_health_status() if health_monitor else None,
                                'processing_successful': True
                            }
                        
                        return result
                        
                    except Exception as e:
                        last_exception = e
                        
                        if health_monitor:
                            health_monitor.record_error(str(e))
                        
                        if config.enable_logging:
                            error(
                                f"Attempt {attempt + 1} failed: {str(e)}", structured_data={"attempt": attempt + 1, "component": component_name}
                            )
                        
                        # Don't retry on critical errors
                        if isinstance(e, (SecurityError, SystemExit, KeyboardInterrupt)):
                            break
                        
                        # Don't retry on final attempt
                        if attempt == max_retries - 1:
                            break
                        
                        # Wait before retry
                        time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
                
                # All retries failed - use fallback strategy
                if config.enable_logging:
                    critical(
                        f"All {max_retries} attempts failed for {component_name}: {str(last_exception)}", structured_data={"component": component_name, "max_retries": max_retries, "severity": ErrorSeverity.CRITICAL}
                    )
                
                if fallback_strategy == "raise":
                    raise last_exception
                else:
                    return _create_fallback_output(self, fallback_strategy, args, kwargs, last_exception)
        
        return wrapper
    return decorator


def sensor_input_validator(
    sensor_type: str = "chemical",
    value_range: tuple = (0.0, 5.0),
    required_shape: Optional[tuple] = None
):
    """
    Decorator for validating sensor inputs.
    
    Args:
        sensor_type: Type of sensor ("chemical", "audio", etc.)
        value_range: Expected value range for sensor readings
        required_shape: Expected tensor shape (if any)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            
            # Find sensor input in arguments
            sensor_input = None
            for arg in args:
                if TORCH_AVAILABLE and isinstance(arg, torch.Tensor):
                    sensor_input = arg
                    break
            
            if sensor_input is None:
                for key, value in kwargs.items():
                    if TORCH_AVAILABLE and isinstance(value, torch.Tensor):
                        sensor_input = value
                        break
            
            if sensor_input is not None:
                try:
                    # Validate sensor data
                    if sensor_type == "chemical":
                        validator = get_sensor_validator()
                        # Convert to list for validation
                        sensor_data = sensor_input.detach().cpu().numpy()
                        for batch_idx in range(sensor_data.shape[0]):
                            validated = validator.validate_sensor_array(
                                sensor_data[batch_idx].flatten().tolist(),
                                sensor_data.shape[-1],
                                "voltage"
                            )
                    
                    # Range validation
                    if value_range:
                        min_val, max_val = value_range
                        if sensor_input.min() < min_val or sensor_input.max() > max_val:
                            raise ValidationError(
                                f"Sensor values outside expected range [{min_val}, {max_val}]",
                                error_code="SENSOR_RANGE_VIOLATION"
                            )
                    
                    # Shape validation
                    if required_shape and sensor_input.shape != required_shape:
                        raise ValidationError(
                            f"Sensor shape {sensor_input.shape} != expected {required_shape}",
                            error_code="SENSOR_SHAPE_MISMATCH"
                        )
                        
                except ValidationError as e:
                    error(f"Sensor validation failed: {str(e)}")
                    raise
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def network_health_monitor(
    convergence_threshold: float = 0.01,
    stability_window: int = 100,
    spike_rate_limits: tuple = (0.0, 1.0)
):
    """
    Decorator for monitoring network health and convergence.
    
    Args:
        convergence_threshold: Threshold for considering network converged
        stability_window: Number of timesteps to check for stability
        spike_rate_limits: Expected spike rate range
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            
            result = func(self, *args, **kwargs)
            
            # Analyze network health from result
            if isinstance(result, dict):
                health_status = _analyze_network_health(
                    result, convergence_threshold, stability_window, spike_rate_limits
                )
                result['network_health'] = health_status
                
                # Log health warnings
                if health_status.get('overall_health') in ['poor', 'critical']:
                    warning(
                        f"Network health degraded: {health_status}", structured_data={"health_status": health_status, "component": self.__class__.__name__}
                    )
            
            return result
        
        return wrapper
    return decorator


def performance_monitor(
    memory_limit_mb: float = 1000.0,
    time_limit_seconds: float = 30.0,
    log_performance_metrics: bool = True
):
    """
    Decorator for monitoring performance metrics.
    
    Args:
        memory_limit_mb: Memory usage limit in MB
        time_limit_seconds: Processing time limit in seconds
        log_performance_metrics: Whether to log performance metrics
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = _get_memory_usage()
            
            try:
                result = func(*args, **kwargs)
                
                # Calculate performance metrics
                processing_time = time.time() - start_time
                peak_memory = _get_memory_usage()
                memory_delta = peak_memory - start_memory
                
                # Check limits
                if processing_time > time_limit_seconds:
                    warning(
                        f"Processing time {processing_time:.2f}s exceeded limit {time_limit_seconds}s", structured_data={"processing_time": processing_time, "time_limit": time_limit_seconds}
                    )
                
                if peak_memory > memory_limit_mb:
                    warning(
                        f"Memory usage {peak_memory:.2f}MB exceeded limit {memory_limit_mb}MB", structured_data={"memory_usage": peak_memory, "memory_limit": memory_limit_mb}
                    )
                
                # Log performance metrics
                if log_performance_metrics:
                    performance(
                        f"Performance metrics for {func.__name__}", structured_data={"func_name": func.__name__, "processing_time": processing_time, "memory_usage": peak_memory, "memory_delta": memory_delta}
                    )
                
                # Add performance metadata to result
                if isinstance(result, dict):
                    result['performance_metrics'] = {
                        'processing_time': processing_time,
                        'peak_memory_mb': peak_memory,
                        'memory_delta_mb': memory_delta,
                        'within_limits': (processing_time <= time_limit_seconds and 
                                        peak_memory <= memory_limit_mb)
                    }
                
                return result
                
            except Exception as e:
                processing_time = time.time() - start_time
                error(
                    f"Function failed after {processing_time:.2f}s: {str(e)}", structured_data={"processing_time": processing_time, "memory_at_failure": _get_memory_usage()}
                )
                raise
        
        return wrapper
    return decorator


def graceful_degradation(
    fallback_strategies: List[str] = ["retry", "reduce_precision", "fallback_model", "zero_output"],
    max_degradation_steps: int = 3
):
    """
    Decorator implementing graceful degradation strategies.
    
    Args:
        fallback_strategies: Ordered list of fallback strategies to try
        max_degradation_steps: Maximum number of degradation steps
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            
            original_args = args
            original_kwargs = kwargs
            
            for step, strategy in enumerate(fallback_strategies[:max_degradation_steps]):
                try:
                    if strategy == "retry":
                        # Simple retry
                        result = func(self, *args, **kwargs)
                        
                    elif strategy == "reduce_precision":
                        # Reduce numerical precision
                        reduced_args, reduced_kwargs = _reduce_precision(args, kwargs)
                        result = func(self, *reduced_args, **reduced_kwargs)
                        
                    elif strategy == "fallback_model":
                        # Use simpler model version
                        result = _call_fallback_model(self, func, args, kwargs)
                        
                    elif strategy == "zero_output":
                        # Return safe zero output
                        result = _create_fallback_output(self, "zero_output", args, kwargs)
                        
                    else:
                        # Unknown strategy - skip
                        continue
                    
                    # Success - add degradation metadata
                    if isinstance(result, dict):
                        result['degradation_metadata'] = {
                            'degradation_level': step,
                            'strategy_used': strategy,
                            'degradation_successful': True
                        }
                    
                    if step > 0:  # Log degradation use
                        warning(
                            f"Used degradation strategy '{strategy}' at level {step}", structured_data={"strategy": strategy, "level": step, "component": self.__class__.__name__}
                        )
                    
                    return result
                    
                except Exception as e:
                    error(f"Degradation strategy '{strategy}' failed: {str(e)}", structured_data={"strategy": strategy})
                    if step == len(fallback_strategies) - 1:  # Last strategy failed
                        raise
                    continue
            
            # All strategies failed
            raise RuntimeError(f"All degradation strategies failed for {func.__name__}")
        
        return wrapper
    return decorator


# Helper functions

def _validate_neuromorphic_inputs(args, kwargs):
    """Validate inputs for neuromorphic models."""
    validator = get_input_validator()
    
    # Check torch tensors in arguments
    for i, arg in enumerate(args):
        if TORCH_AVAILABLE and isinstance(arg, torch.Tensor):
            if torch.isnan(arg).any() or torch.isinf(arg).any():
                raise ValidationError(
                    f"Invalid values (NaN/Inf) in argument {i}",
                    error_code="INVALID_TENSOR_VALUES"
                )
    
    # Check torch tensors in keyword arguments
    for key, value in kwargs.items():
        if TORCH_AVAILABLE and isinstance(value, torch.Tensor):
            if torch.isnan(value).any() or torch.isinf(value).any():
                raise ValidationError(
                    f"Invalid values (NaN/Inf) in parameter {key}",
                    error_code="INVALID_TENSOR_VALUES"
                )


def _perform_security_checks(args, kwargs):
    """Perform security checks on inputs."""
    # Check for suspiciously large tensors
    for arg in args:
        if TORCH_AVAILABLE and isinstance(arg, torch.Tensor):
            if arg.numel() > 1e8:  # 100M elements
                raise SecurityError(
                    f"Tensor too large: {arg.numel()} elements",
                    error_code="TENSOR_TOO_LARGE"
                )
    
    for key, value in kwargs.items():
        if TORCH_AVAILABLE and isinstance(value, torch.Tensor):
            if value.numel() > 1e8:
                raise SecurityError(
                    f"Parameter {key} tensor too large: {value.numel()} elements",
                    error_code="TENSOR_TOO_LARGE"
                )


def _validate_neuromorphic_output(result):
    """Validate neuromorphic model outputs."""
    if isinstance(result, dict):
        # Check for required keys and valid values
        if 'decision_output' in result:
            decision_output = result['decision_output']
            if 'class_probabilities' in decision_output:
                probs = decision_output['class_probabilities']
                if TORCH_AVAILABLE and isinstance(probs, torch.Tensor):
                    if torch.isnan(probs).any() or torch.isinf(probs).any():
                        raise ValidationError(
                            "Invalid values in class probabilities",
                            error_code="INVALID_OUTPUT_PROBS"
                        )
                    if (probs < 0).any() or (probs > 1).any():
                        raise ValidationError(
                            "Probabilities outside [0,1] range",
                            error_code="PROB_OUT_OF_RANGE"
                        )


@contextmanager
def _timeout_context(timeout_seconds: float):
    """Context manager for timeout handling."""
    # Simple timeout implementation - in production, use more sophisticated approaches
    start_time = time.time()
    try:
        yield
        if time.time() - start_time > timeout_seconds:
            raise TimeoutError(f"Processing exceeded {timeout_seconds}s timeout")
    except Exception as e:
        if time.time() - start_time > timeout_seconds:
            raise TimeoutError(f"Processing timed out: {str(e)}")
        raise


def _create_fallback_output(model_instance, strategy: str, args, kwargs, exception=None):
    """Create fallback output based on strategy."""
    
    # Try to determine output structure from model
    if hasattr(model_instance, 'config'):
        config = model_instance.config
    else:
        config = None
    
    # Determine batch size and device from inputs
    batch_size = 1
    device = torch.device('cpu') if TORCH_AVAILABLE else None
    
    for arg in args:
        if TORCH_AVAILABLE and isinstance(arg, torch.Tensor):
            batch_size = arg.shape[0]
            device = arg.device
            break
    
    if strategy == "zero_output":
        if config and hasattr(config, 'num_output_classes'):
            num_classes = config.num_output_classes
            if TORCH_AVAILABLE:
                return {
                    'decision_output': {
                        'class_probabilities': torch.ones(batch_size, num_classes, device=device) / num_classes
                    },
                    'fallback_mode': True,
                    'error_info': str(exception) if exception else "Fallback mode activated"
                }
        
        # Generic fallback
        return {
            'output': None,
            'fallback_mode': True,
            'error_info': str(exception) if exception else "Fallback mode activated"
        }
    
    elif strategy == "uniform_output":
        if config and hasattr(config, 'num_output_classes') and TORCH_AVAILABLE:
            num_classes = config.num_output_classes
            return {
                'decision_output': {
                    'class_probabilities': torch.ones(batch_size, num_classes, device=device) / num_classes
                },
                'uniform_fallback': True
            }
    
    # Default fallback
    return {'error': True, 'fallback_strategy': strategy}


def _analyze_network_health(result: dict, convergence_threshold: float, 
                          stability_window: int, spike_rate_limits: tuple) -> dict:
    """Analyze network health from processing results."""
    health_status = {
        'overall_health': 'good',
        'convergence_status': 'unknown',
        'spike_rate_health': 'unknown',
        'issues': []
    }
    
    try:
        # Check convergence if available
        if 'convergence_status' in result:
            conv_status = result['convergence_status']
            if isinstance(conv_status, dict):
                if conv_status.get('convergence_metric', 1.0) < convergence_threshold:
                    health_status['convergence_status'] = 'converged'
                else:
                    health_status['convergence_status'] = 'not_converged'
                    health_status['issues'].append('Network not converged')
        
        # Check spike rates
        spike_data = None
        if 'kenyon_spikes' in result:
            spike_data = result['kenyon_spikes']
        elif 'excitatory_spikes' in result:
            spike_data = result['excitatory_spikes']
        
        if spike_data is not None and TORCH_AVAILABLE and isinstance(spike_data, torch.Tensor):
            mean_rate = spike_data.mean().item()
            min_rate, max_rate = spike_rate_limits
            
            if min_rate <= mean_rate <= max_rate:
                health_status['spike_rate_health'] = 'healthy'
            else:
                health_status['spike_rate_health'] = 'unhealthy'
                health_status['issues'].append(f'Spike rate {mean_rate:.3f} outside [{min_rate}, {max_rate}]')
        
        # Overall health assessment
        if len(health_status['issues']) == 0:
            health_status['overall_health'] = 'good'
        elif len(health_status['issues']) <= 2:
            health_status['overall_health'] = 'fair' 
        else:
            health_status['overall_health'] = 'poor'
            
    except Exception as e:
        health_status['overall_health'] = 'critical'
        health_status['issues'].append(f'Health analysis failed: {str(e)}')
    
    return health_status


def _get_memory_usage() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    except ImportError:
        return 0.0  # psutil not available


def _reduce_precision(args, kwargs):
    """Reduce numerical precision of tensor arguments."""
    reduced_args = []
    reduced_kwargs = {}
    
    for arg in args:
        if TORCH_AVAILABLE and isinstance(arg, torch.Tensor) and arg.dtype == torch.float64:
            reduced_args.append(arg.float())  # Convert to float32
        else:
            reduced_args.append(arg)
    
    for key, value in kwargs.items():
        if TORCH_AVAILABLE and isinstance(value, torch.Tensor) and value.dtype == torch.float64:
            reduced_kwargs[key] = value.float()
        else:
            reduced_kwargs[key] = value
    
    return tuple(reduced_args), reduced_kwargs


def _call_fallback_model(model_instance, func, args, kwargs):
    """Call a simplified version of the model."""
    # This is a placeholder - in practice, you'd implement model-specific fallbacks
    warning(f"Fallback model not implemented for {model_instance.__class__.__name__}", structured_data={"model_class": model_instance.__class__.__name__})
    return _create_fallback_output(model_instance, "zero_output", args, kwargs)


# Convenience decorators combining multiple robustness features

def robust_snn_layer(config: Optional[RobustnessConfig] = None):
    """Complete robustness decorator for SNN layers."""
    if config is None:
        config = _global_config
    
    def decorator(func):
        @robust_neuromorphic_model(config=config)
        @network_health_monitor()
        @performance_monitor()
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


def robust_sensor_processor(sensor_type: str = "chemical"):
    """Complete robustness decorator for sensor processing."""
    def decorator(func):
        @robust_neuromorphic_model()
        @sensor_input_validator(sensor_type=sensor_type)
        @performance_monitor()
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


def robust_decision_maker():
    """Complete robustness decorator for decision-making components."""
    def decorator(func):
        @robust_neuromorphic_model(fallback_strategy="uniform_output")
        @graceful_degradation(["retry", "reduce_precision", "zero_output"])
        @performance_monitor()
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator