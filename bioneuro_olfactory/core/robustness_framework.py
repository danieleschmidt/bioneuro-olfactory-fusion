"""Comprehensive robustness and error handling framework.

This module provides robust error handling, graceful degradation,
automatic recovery, and comprehensive monitoring for the gas detection system.
"""

import logging
import traceback
import time
import threading
from typing import Dict, Any, Callable, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SystemState(Enum):
    """System operational states."""
    NORMAL = "normal"
    DEGRADED = "degraded"
    FAULT_TOLERANT = "fault_tolerant"
    EMERGENCY = "emergency"
    OFFLINE = "offline"


@dataclass
class ErrorInfo:
    """Information about an error event."""
    timestamp: float
    error_type: str
    error_message: str
    severity: ErrorSeverity
    component: str
    context: Dict[str, Any] = field(default_factory=dict)
    traceback_info: Optional[str] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False


@dataclass
class ComponentHealth:
    """Health status of a system component."""
    name: str
    is_operational: bool = True
    error_count: int = 0
    last_error: Optional[ErrorInfo] = None
    degraded_mode: bool = False
    fallback_active: bool = False
    uptime: float = 0.0
    last_check: float = field(default_factory=time.time)


class RobustnessManager:
    """Central manager for system robustness and error handling."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.system_state = SystemState.NORMAL
        self.component_health = {}
        self.error_history = []
        self.recovery_strategies = {}
        self.monitoring_active = False
        self.max_error_history = 1000
        
        # Statistics
        self.stats = {
            'total_errors': 0,
            'recovered_errors': 0,
            'system_restarts': 0,
            'uptime_start': time.time()
        }
        
        # Initialize monitoring
        self._setup_monitoring()
        
    def _default_config(self) -> Dict[str, Any]:
        """Default robustness configuration."""
        return {
            'max_error_rate': 10,  # errors per minute
            'degraded_mode_threshold': 5,  # consecutive errors
            'emergency_threshold': 20,  # total errors in window
            'recovery_timeout': 30.0,  # seconds
            'health_check_interval': 5.0,  # seconds
            'error_window_size': 300.0,  # seconds (5 minutes)
            'enable_auto_recovery': True,
            'enable_fallback_modes': True,
            'log_errors': True,
            'save_error_reports': True
        }
        
    def register_component(self, name: str, health_check: Callable = None):
        """Register a system component for monitoring.
        
        Args:
            name: Component name
            health_check: Optional function to check component health
        """
        self.component_health[name] = ComponentHealth(name=name)
        
        if health_check:
            self.recovery_strategies[name] = {
                'health_check': health_check,
                'recovery_function': None,
                'fallback_function': None
            }
            
        logger.info(f"Registered component: {name}")
        
    def register_recovery_strategy(
        self, 
        component: str, 
        recovery_func: Callable,
        fallback_func: Callable = None
    ):
        """Register recovery and fallback strategies for a component.
        
        Args:
            component: Component name
            recovery_func: Function to attempt recovery
            fallback_func: Function to enable fallback mode
        """
        if component not in self.recovery_strategies:
            self.recovery_strategies[component] = {}
            
        self.recovery_strategies[component]['recovery_function'] = recovery_func
        if fallback_func:
            self.recovery_strategies[component]['fallback_function'] = fallback_func
            
        logger.info(f"Registered recovery strategy for: {component}")
        
    def handle_error(
        self, 
        error: Exception, 
        component: str,
        context: Dict[str, Any] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM
    ) -> bool:
        """Handle an error with appropriate recovery actions.
        
        Args:
            error: The exception that occurred
            component: Component where error occurred
            context: Additional context information
            severity: Error severity level
            
        Returns:
            True if error was handled successfully
        """
        error_info = ErrorInfo(
            timestamp=time.time(),
            error_type=type(error).__name__,
            error_message=str(error),
            severity=severity,
            component=component,
            context=context or {},
            traceback_info=traceback.format_exc()
        )
        
        # Update statistics
        self.stats['total_errors'] += 1
        
        # Log error
        if self.config['log_errors']:
            log_level = {
                ErrorSeverity.LOW: logging.INFO,
                ErrorSeverity.MEDIUM: logging.WARNING,
                ErrorSeverity.HIGH: logging.ERROR,
                ErrorSeverity.CRITICAL: logging.CRITICAL
            }[severity]
            
            logger.log(
                log_level,
                f"Error in {component}: {error_info.error_message}"
            )
            
        # Update component health
        if component in self.component_health:
            health = self.component_health[component]
            health.error_count += 1
            health.last_error = error_info
            
            # Check if component should be marked as degraded
            if health.error_count >= self.config['degraded_mode_threshold']:
                health.degraded_mode = True
                health.is_operational = False
                
        # Add to error history
        self.error_history.append(error_info)
        if len(self.error_history) > self.max_error_history:
            self.error_history.pop(0)
            
        # Attempt recovery
        recovery_success = False
        if self.config['enable_auto_recovery']:
            recovery_success = self._attempt_recovery(component, error_info)
            
        # Update system state
        self._update_system_state()
        
        # Save error report if configured
        if self.config['save_error_reports']:
            self._save_error_report(error_info)
            
        return recovery_success
        
    def _attempt_recovery(self, component: str, error_info: ErrorInfo) -> bool:
        """Attempt to recover from an error.
        
        Args:
            component: Component name
            error_info: Error information
            
        Returns:
            True if recovery was successful
        """
        if component not in self.recovery_strategies:
            return False
            
        strategy = self.recovery_strategies[component]
        error_info.recovery_attempted = True
        
        try:
            # Try recovery function
            if 'recovery_function' in strategy and strategy['recovery_function']:
                logger.info(f"Attempting recovery for {component}")
                
                recovery_func = strategy['recovery_function']
                success = recovery_func(error_info)
                
                if success:
                    error_info.recovery_successful = True
                    self.stats['recovered_errors'] += 1
                    
                    # Restore component health
                    if component in self.component_health:
                        health = self.component_health[component]
                        health.is_operational = True
                        health.degraded_mode = False
                        health.error_count = max(0, health.error_count - 1)
                        
                    logger.info(f"Recovery successful for {component}")
                    return True
                    
            # If recovery fails, try fallback mode
            if (self.config['enable_fallback_modes'] and 
                'fallback_function' in strategy and 
                strategy['fallback_function']):
                
                logger.info(f"Enabling fallback mode for {component}")
                fallback_func = strategy['fallback_function']
                fallback_success = fallback_func(error_info)
                
                if fallback_success and component in self.component_health:
                    health = self.component_health[component]
                    health.fallback_active = True
                    health.degraded_mode = True
                    health.is_operational = True  # Still operational, but degraded
                    
                return fallback_success
                
        except Exception as recovery_error:
            logger.error(f"Recovery attempt failed for {component}: {recovery_error}")
            
        return False
        
    def _update_system_state(self):
        """Update overall system state based on component health."""
        operational_components = sum(
            1 for health in self.component_health.values() 
            if health.is_operational
        )
        total_components = len(self.component_health)
        
        if total_components == 0:
            return
            
        operational_ratio = operational_components / total_components
        
        # Count recent errors
        recent_errors = self._count_recent_errors()
        
        # Determine system state
        previous_state = self.system_state
        
        if recent_errors >= self.config['emergency_threshold']:
            self.system_state = SystemState.EMERGENCY
        elif operational_ratio < 0.5:
            self.system_state = SystemState.FAULT_TOLERANT
        elif operational_ratio < 0.8 or any(h.degraded_mode for h in self.component_health.values()):
            self.system_state = SystemState.DEGRADED
        else:
            self.system_state = SystemState.NORMAL
            
        # Log state changes
        if self.system_state != previous_state:
            logger.warning(f"System state changed: {previous_state.value} -> {self.system_state.value}")
            
    def _count_recent_errors(self) -> int:
        """Count errors in recent time window."""
        current_time = time.time()
        window_start = current_time - self.config['error_window_size']
        
        return sum(
            1 for error in self.error_history 
            if error.timestamp >= window_start
        )
        
    def _setup_monitoring(self):
        """Setup background monitoring thread."""
        def monitor_loop():
            while self.monitoring_active:
                try:
                    self._perform_health_checks()
                    time.sleep(self.config['health_check_interval'])
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        
    def _perform_health_checks(self):
        """Perform health checks on all registered components."""
        current_time = time.time()
        
        for component_name, strategy in self.recovery_strategies.items():
            if 'health_check' not in strategy or not strategy['health_check']:
                continue
                
            try:
                health_check = strategy['health_check']
                is_healthy = health_check()
                
                if component_name in self.component_health:
                    health = self.component_health[component_name]
                    health.last_check = current_time
                    
                    if not is_healthy and health.is_operational:
                        # Component became unhealthy
                        error = Exception(f"Health check failed for {component_name}")
                        self.handle_error(error, component_name, severity=ErrorSeverity.HIGH)
                        
            except Exception as e:
                logger.error(f"Health check error for {component_name}: {e}")
                
    def _save_error_report(self, error_info: ErrorInfo):
        """Save error report to file."""
        try:
            reports_dir = Path("error_reports")
            reports_dir.mkdir(exist_ok=True)
            
            timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(error_info.timestamp))
            filename = f"error_{timestamp_str}_{error_info.component}.json"
            
            report = {
                'timestamp': error_info.timestamp,
                'error_type': error_info.error_type,
                'error_message': error_info.error_message,
                'severity': error_info.severity.value,
                'component': error_info.component,
                'context': error_info.context,
                'traceback': error_info.traceback_info,
                'recovery_attempted': error_info.recovery_attempted,
                'recovery_successful': error_info.recovery_successful,
                'system_state': self.system_state.value
            }
            
            with open(reports_dir / filename, 'w') as f:
                json.dump(report, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save error report: {e}")
            
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        current_time = time.time()
        uptime = current_time - self.stats['uptime_start']
        
        return {
            'system_state': self.system_state.value,
            'uptime_seconds': uptime,
            'component_health': {
                name: {
                    'operational': health.is_operational,
                    'degraded': health.degraded_mode,
                    'fallback_active': health.fallback_active,
                    'error_count': health.error_count,
                    'last_error': health.last_error.error_message if health.last_error else None
                }
                for name, health in self.component_health.items()
            },
            'statistics': self.stats.copy(),
            'recent_error_count': self._count_recent_errors()
        }
        
    def reset_component(self, component: str) -> bool:
        """Reset a component's error state."""
        if component in self.component_health:
            health = self.component_health[component]
            health.error_count = 0
            health.is_operational = True
            health.degraded_mode = False
            health.fallback_active = False
            health.last_error = None
            
            logger.info(f"Reset component: {component}")
            self._update_system_state()
            return True
            
        return False
        
    def shutdown(self):
        """Shutdown the robustness manager."""
        self.monitoring_active = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=5.0)
            
        logger.info("Robustness manager shutdown complete")


# Global robustness manager instance
robustness_manager = RobustnessManager()


def robust_operation(
    component: str, 
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    context: Dict[str, Any] = None
):
    """Decorator for robust operation handling.
    
    Args:
        component: Component name for error tracking
        severity: Default error severity
        context: Additional context information
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                success = robustness_manager.handle_error(
                    e, component, context, severity
                )
                if not success:
                    raise  # Re-raise if not handled
                    
                # Return safe default if handled
                return None
                
        return wrapper
    return decorator


def require_operational_component(component: str):
    """Decorator to require a component to be operational.
    
    Args:
        component: Component name to check
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if component in robustness_manager.component_health:
                health = robustness_manager.component_health[component]
                if not health.is_operational:
                    raise RuntimeError(f"Component {component} is not operational")
                    
            return func(*args, **kwargs)
        return wrapper
    return decorator


def safe_execute(
    func: Callable, 
    component: str,
    default_return=None,
    *args, 
    **kwargs
) -> Any:
    """Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        component: Component name for error tracking
        default_return: Default return value on error
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Function result or default_return on error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        robustness_manager.handle_error(e, component)
        return default_return


# Health check utilities
def create_component_health_check(component_name: str, check_func: Callable) -> Callable:
    """Create a standardized health check function.
    
    Args:
        component_name: Name of the component
        check_func: Function that returns True if healthy
        
    Returns:
        Health check function
    """
    def health_check():
        try:
            return bool(check_func())
        except Exception as e:
            logger.error(f"Health check failed for {component_name}: {e}")
            return False
    
    return health_check


if __name__ == "__main__":
    # Demo the robustness framework
    print("🛡️  Robustness Framework Demo")
    
    # Register a test component
    def test_health_check():
        return True
        
    def test_recovery(error_info):
        print(f"Recovering from: {error_info.error_message}")
        return True
        
    robustness_manager.register_component("test_component", test_health_check)
    robustness_manager.register_recovery_strategy("test_component", test_recovery)
    
    # Simulate some errors
    try:
        raise ValueError("Test error")
    except Exception as e:
        robustness_manager.handle_error(e, "test_component")
        
    # Print status
    status = robustness_manager.get_system_status()
    print(f"System state: {status['system_state']}")
    print(f"Components: {len(status['component_health'])}")
    print(f"Total errors: {status['statistics']['total_errors']}")
    
    robustness_manager.shutdown()