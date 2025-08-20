"""Enterprise-grade robustness system for neuromorphic gas detection.

This module implements comprehensive error handling, fault tolerance,
circuit breakers, and monitoring for production deployment readiness.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import logging
from contextlib import contextmanager
import threading
from collections import defaultdict, deque
import warnings


class ErrorSeverity(Enum):
    """Error severity levels for classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SystemState(Enum):
    """System operational states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    OFFLINE = "offline"
    RECOVERY = "recovery"


@dataclass
class RobustnessConfig:
    """Configuration for robustness system."""
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 30.0
    enable_retry_logic: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0
    enable_fallback_mode: bool = True
    enable_health_monitoring: bool = True
    health_check_interval: float = 10.0
    enable_graceful_degradation: bool = True
    performance_threshold: float = 0.8
    memory_threshold: float = 0.9
    enable_logging: bool = True
    log_level: str = "INFO"


class CircuitBreaker:
    """Circuit breaker pattern implementation for fault tolerance."""
    
    def __init__(
        self, 
        threshold: int = 5, 
        timeout: float = 30.0,
        name: str = "default"
    ):
        self.threshold = threshold
        self.timeout = timeout
        self.name = name
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
        self._lock = threading.Lock()
        
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap functions with circuit breaker."""
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
        
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == "open":
                if self._should_attempt_reset():
                    self.state = "half-open"
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker '{self.name}' is open"
                    )
                    
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except Exception as e:
            self._on_failure()
            raise e
            
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.timeout
        
    def _on_success(self):
        """Handle successful execution."""
        with self._lock:
            self.failure_count = 0
            if self.state == "half-open":
                self.state = "closed"
                
    def _on_failure(self):
        """Handle failed execution."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.threshold:
                self.state = "open"
                
    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        return {
            "name": self.name,
            "state": self.state,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time,
            "threshold": self.threshold
        }


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class RetryManager:
    """Intelligent retry manager with exponential backoff."""
    
    def __init__(
        self, 
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        
    def __call__(self, func: Callable) -> Callable:
        """Decorator to add retry logic to functions."""
        def wrapper(*args, **kwargs):
            return self.execute_with_retry(func, *args, **kwargs)
        return wrapper
        
    def execute_with_retry(
        self, 
        func: Callable, 
        *args, 
        **kwargs
    ) -> Any:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
                
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    break
                    
                # Calculate delay with exponential backoff
                delay = min(
                    self.base_delay * (self.backoff_factor ** attempt),
                    self.max_delay
                )
                
                logging.warning(
                    f"Attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {delay:.2f} seconds..."
                )
                
                time.sleep(delay)
                
        raise last_exception


class HealthMonitor:
    """System health monitoring and alerting."""
    
    def __init__(self, check_interval: float = 10.0):
        self.check_interval = check_interval
        self.metrics = defaultdict(list)
        self.alerts = deque(maxlen=100)
        self.running = False
        self._monitor_thread = None
        
    def start_monitoring(self):
        """Start health monitoring in background thread."""
        if not self.running:
            self.running = True
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop, daemon=True
            )
            self._monitor_thread.start()
            
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.running = False
        if self._monitor_thread:
            self._monitor_thread.join()
            
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                self._collect_metrics()
                self._check_health()
                time.sleep(self.check_interval)
            except Exception as e:
                logging.error(f"Health monitoring error: {e}")
                
    def _collect_metrics(self):
        """Collect system metrics."""
        import psutil
        
        # CPU and memory usage
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent / 100.0
        
        # GPU usage if available
        gpu_usage = 0.0
        if torch.cuda.is_available():
            gpu_usage = torch.cuda.utilization() / 100.0
            
        timestamp = time.time()
        
        self.metrics['cpu_usage'].append((timestamp, cpu_usage))
        self.metrics['memory_usage'].append((timestamp, memory_usage))
        self.metrics['gpu_usage'].append((timestamp, gpu_usage))
        
        # Keep only last 1000 entries
        for key in self.metrics:
            if len(self.metrics[key]) > 1000:
                self.metrics[key] = self.metrics[key][-1000:]
                
    def _check_health(self):
        """Check system health and generate alerts."""
        if not self.metrics:
            return
            
        # Check recent metrics
        recent_cpu = [v for t, v in self.metrics['cpu_usage'][-10:]]
        recent_memory = [v for t, v in self.metrics['memory_usage'][-10:]]
        
        # CPU usage alerts
        if recent_cpu and np.mean(recent_cpu) > 90:
            self._add_alert(
                "HIGH_CPU_USAGE", 
                f"CPU usage: {np.mean(recent_cpu):.1f}%",
                ErrorSeverity.HIGH
            )
            
        # Memory usage alerts
        if recent_memory and np.mean(recent_memory) > 90:
            self._add_alert(
                "HIGH_MEMORY_USAGE",
                f"Memory usage: {np.mean(recent_memory):.1f}%", 
                ErrorSeverity.HIGH
            )
            
    def _add_alert(self, alert_type: str, message: str, severity: ErrorSeverity):
        """Add alert to queue."""
        alert = {
            'timestamp': time.time(),
            'type': alert_type,
            'message': message,
            'severity': severity.value
        }
        self.alerts.append(alert)
        
        if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            logging.warning(f"ALERT [{severity.value}]: {message}")
            
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        if not self.metrics:
            return {"status": "unknown", "message": "No metrics available"}
            
        # Get recent averages
        recent_cpu = [v for t, v in self.metrics['cpu_usage'][-10:]]
        recent_memory = [v for t, v in self.metrics['memory_usage'][-10:]]
        
        avg_cpu = np.mean(recent_cpu) if recent_cpu else 0
        avg_memory = np.mean(recent_memory) if recent_memory else 0
        
        # Determine status
        if avg_cpu > 95 or avg_memory > 95:
            status = SystemState.FAILING
        elif avg_cpu > 80 or avg_memory > 80:
            status = SystemState.DEGRADED
        else:
            status = SystemState.HEALTHY
            
        return {
            "status": status.value,
            "cpu_usage": avg_cpu,
            "memory_usage": avg_memory,
            "alerts_count": len(self.alerts),
            "recent_alerts": list(self.alerts)[-5:]  # Last 5 alerts
        }


class FallbackProcessor:
    """Fallback processing for degraded system states."""
    
    def __init__(self):
        self.fallback_models = {}
        self.simple_classifiers = {}
        
    def register_fallback(self, name: str, model: nn.Module):
        """Register a fallback model."""
        self.fallback_models[name] = model
        
    def create_simple_classifier(
        self, 
        input_dim: int, 
        output_dim: int
    ) -> nn.Module:
        """Create simple fallback classifier."""
        classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.Softmax(dim=-1)
        )
        return classifier
        
    def process_with_fallback(
        self, 
        input_data: torch.Tensor,
        fallback_name: str = "simple"
    ) -> torch.Tensor:
        """Process data using fallback method."""
        if fallback_name in self.fallback_models:
            return self.fallback_models[fallback_name](input_data)
        else:
            # Ultra-simple processing
            return torch.softmax(torch.mean(input_data, dim=-1, keepdim=True), dim=-1)


class RobustNeuromorphicSystem:
    """Main robustness system wrapper for neuromorphic networks."""
    
    def __init__(
        self, 
        base_model: nn.Module,
        config: RobustnessConfig = None
    ):
        self.base_model = base_model
        self.config = config or RobustnessConfig()
        
        # Initialize robustness components
        self.circuit_breaker = CircuitBreaker(
            threshold=self.config.circuit_breaker_threshold,
            timeout=self.config.circuit_breaker_timeout,
            name="neuromorphic_system"
        ) if self.config.enable_circuit_breaker else None
        
        self.retry_manager = RetryManager(
            max_retries=self.config.max_retries,
            base_delay=self.config.retry_delay
        ) if self.config.enable_retry_logic else None
        
        self.health_monitor = HealthMonitor(
            check_interval=self.config.health_check_interval
        ) if self.config.enable_health_monitoring else None
        
        self.fallback_processor = FallbackProcessor() if self.config.enable_fallback_mode else None
        
        # Performance tracking
        self.performance_metrics = deque(maxlen=100)
        self.error_log = deque(maxlen=500)
        
        # Setup logging
        if self.config.enable_logging:
            self._setup_logging()
            
        # Start monitoring
        if self.health_monitor:
            self.health_monitor.start_monitoring()
            
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('neuromorphic_system.log')
            ]
        )
        
    def forward(
        self, 
        *args, 
        **kwargs
    ) -> Dict[str, Any]:
        """Robust forward pass with error handling."""
        start_time = time.time()
        
        try:
            # Health check
            if self.health_monitor:
                health_status = self.health_monitor.get_health_status()
                if health_status['status'] == SystemState.FAILING.value:
                    return self._fallback_processing(*args, **kwargs)
                    
            # Main processing with circuit breaker and retries
            if self.circuit_breaker and self.retry_manager:
                result = self.circuit_breaker.call(
                    self.retry_manager.execute_with_retry,
                    self._safe_forward,
                    *args, **kwargs
                )
            elif self.circuit_breaker:
                result = self.circuit_breaker.call(self._safe_forward, *args, **kwargs)
            elif self.retry_manager:
                result = self.retry_manager.execute_with_retry(
                    self._safe_forward, *args, **kwargs
                )
            else:
                result = self._safe_forward(*args, **kwargs)
                
            # Record performance
            processing_time = time.time() - start_time
            self._record_performance(processing_time, success=True)
            
            return {
                'result': result,
                'processing_time': processing_time,
                'status': 'success',
                'fallback_used': False
            }
            
        except Exception as e:
            # Log error
            self._log_error(e, *args, **kwargs)
            
            # Attempt fallback processing
            if self.fallback_processor:
                try:
                    fallback_result = self._fallback_processing(*args, **kwargs)
                    processing_time = time.time() - start_time
                    
                    return {
                        'result': fallback_result,
                        'processing_time': processing_time,
                        'status': 'fallback_success',
                        'fallback_used': True,
                        'original_error': str(e)
                    }
                except Exception as fallback_error:
                    self._log_error(fallback_error, *args, **kwargs)
                    
            # If all else fails
            processing_time = time.time() - start_time
            self._record_performance(processing_time, success=False)
            
            return {
                'result': None,
                'processing_time': processing_time,
                'status': 'error',
                'fallback_used': False,
                'error': str(e)
            }
            
    def _safe_forward(self, *args, **kwargs) -> Any:
        """Safe forward pass with input validation."""
        # Input validation
        self._validate_inputs(*args, **kwargs)
        
        # Memory check
        self._check_memory_usage()
        
        # Execute model
        with torch.no_grad():
            result = self.base_model(*args, **kwargs)
            
        # Output validation
        self._validate_outputs(result)
        
        return result
        
    def _validate_inputs(self, *args, **kwargs):
        """Validate input tensors."""
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                if torch.isnan(arg).any():
                    raise ValueError(f"NaN detected in input tensor {i}")
                if torch.isinf(arg).any():
                    raise ValueError(f"Inf detected in input tensor {i}")
                    
    def _validate_outputs(self, outputs):
        """Validate output tensors."""
        if isinstance(outputs, torch.Tensor):
            if torch.isnan(outputs).any():
                raise ValueError("NaN detected in model output")
            if torch.isinf(outputs).any():
                raise ValueError("Inf detected in model output")
        elif isinstance(outputs, dict):
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    if torch.isnan(value).any():
                        raise ValueError(f"NaN detected in output '{key}'")
                    if torch.isinf(value).any():
                        raise ValueError(f"Inf detected in output '{key}'")
                        
    def _check_memory_usage(self):
        """Check memory usage and warn if too high."""
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            if memory_used > self.config.memory_threshold:
                warnings.warn(
                    f"High GPU memory usage: {memory_used:.1%}",
                    ResourceWarning
                )
                
    def _fallback_processing(self, *args, **kwargs) -> Any:
        """Execute fallback processing."""
        if not self.fallback_processor:
            raise RuntimeError("No fallback processor available")
            
        # Simple fallback: return safe default
        if args and isinstance(args[0], torch.Tensor):
            batch_size = args[0].shape[0]
            # Return uniform probability distribution
            return torch.ones(batch_size, 4) / 4.0  # Assume 4 classes
        else:
            return torch.tensor([0.25, 0.25, 0.25, 0.25])  # Default probabilities
            
    def _record_performance(self, processing_time: float, success: bool):
        """Record performance metrics."""
        metric = {
            'timestamp': time.time(),
            'processing_time': processing_time,
            'success': success
        }
        self.performance_metrics.append(metric)
        
    def _log_error(self, error: Exception, *args, **kwargs):
        """Log error with context."""
        error_info = {
            'timestamp': time.time(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'input_shapes': [arg.shape if isinstance(arg, torch.Tensor) else type(arg) for arg in args]
        }
        self.error_log.append(error_info)
        
        logging.error(f"Neuromorphic system error: {error_info}")
        
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            'timestamp': time.time(),
            'model_name': self.base_model.__class__.__name__,
        }
        
        # Circuit breaker status
        if self.circuit_breaker:
            status['circuit_breaker'] = self.circuit_breaker.get_status()
            
        # Health monitor status
        if self.health_monitor:
            status['health'] = self.health_monitor.get_health_status()
            
        # Performance metrics
        if self.performance_metrics:
            recent_metrics = list(self.performance_metrics)[-10:]
            success_rate = sum(1 for m in recent_metrics if m['success']) / len(recent_metrics)
            avg_processing_time = np.mean([m['processing_time'] for m in recent_metrics])
            
            status['performance'] = {
                'success_rate': success_rate,
                'avg_processing_time': avg_processing_time,
                'total_requests': len(self.performance_metrics)
            }
            
        # Error summary
        if self.error_log:
            recent_errors = list(self.error_log)[-5:]
            error_types = defaultdict(int)
            for error in recent_errors:
                error_types[error['error_type']] += 1
                
            status['errors'] = {
                'total_errors': len(self.error_log),
                'recent_error_types': dict(error_types),
                'last_error_time': recent_errors[-1]['timestamp'] if recent_errors else None
            }
            
        return status
        
    def shutdown(self):
        """Graceful shutdown of robustness system."""
        if self.health_monitor:
            self.health_monitor.stop_monitoring()
            
        logging.info("Robustness system shutdown complete")


@contextmanager
def robust_execution(
    model: nn.Module,
    config: RobustnessConfig = None
):
    """Context manager for robust model execution."""
    robust_system = RobustNeuromorphicSystem(model, config)
    
    try:
        yield robust_system
    finally:
        robust_system.shutdown()


def create_robust_neuromorphic_system(
    base_model: nn.Module,
    enable_all_features: bool = True
) -> RobustNeuromorphicSystem:
    """Create a fully-featured robust neuromorphic system.
    
    Args:
        base_model: Base neuromorphic model to wrap
        enable_all_features: Whether to enable all robustness features
        
    Returns:
        Configured robust system
    """
    config = RobustnessConfig(
        enable_circuit_breaker=enable_all_features,
        enable_retry_logic=enable_all_features,
        enable_fallback_mode=enable_all_features,
        enable_health_monitoring=enable_all_features,
        enable_graceful_degradation=enable_all_features,
        enable_logging=True
    )
    
    return RobustNeuromorphicSystem(base_model, config)