"""
Advanced Neuromorphic System Monitoring
=======================================

This module provides comprehensive monitoring, health checking, and performance
analysis for neuromorphic computing systems.

Created as part of Terragon SDLC Generation 2: MAKE IT ROBUST
"""

import time
import threading
import statistics
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import json
import warnings


class HealthStatus(Enum):
    """Health status levels for neuromorphic components."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"
    UNKNOWN = "unknown"


class MetricType(Enum):
    """Types of metrics for neuromorphic monitoring."""
    SPIKE_RATE = "spike_rate"
    MEMBRANE_POTENTIAL = "membrane_potential"
    SYNAPTIC_WEIGHT = "synaptic_weight"
    SPARSITY = "sparsity"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ACCURACY = "accuracy"
    ENERGY_CONSUMPTION = "energy_consumption"
    PLASTICITY_RATE = "plasticity_rate"
    COMPETITION_STRENGTH = "competition_strength"


@dataclass
class Metric:
    """Individual metric measurement."""
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    unit: str = ""
    component: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert metric to dictionary."""
        return {
            'name': self.name,
            'value': self.value,
            'timestamp': self.timestamp,
            'unit': self.unit,
            'component': self.component,
            'metadata': self.metadata
        }


@dataclass
class HealthCheck:
    """Health check configuration and results."""
    name: str
    check_function: Callable
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    enabled: bool = True
    last_status: HealthStatus = HealthStatus.UNKNOWN
    last_check_time: float = 0
    check_interval: float = 60.0  # seconds
    failure_count: int = 0
    max_failures: int = 3
    
    def should_run(self) -> bool:
        """Check if health check should run based on interval."""
        return time.time() - self.last_check_time >= self.check_interval
        
    def run_check(self, system_state: Dict) -> HealthStatus:
        """Execute the health check."""
        if not self.enabled:
            return self.last_status
            
        try:
            result = self.check_function(system_state)
            
            # Evaluate result against thresholds
            if isinstance(result, (int, float)):
                if self.threshold_critical is not None and result >= self.threshold_critical:
                    status = HealthStatus.CRITICAL
                elif self.threshold_warning is not None and result >= self.threshold_warning:
                    status = HealthStatus.WARNING
                else:
                    status = HealthStatus.HEALTHY
            elif isinstance(result, bool):
                status = HealthStatus.HEALTHY if result else HealthStatus.CRITICAL
            else:
                status = HealthStatus.UNKNOWN
                
            # Update state
            self.last_status = status
            self.last_check_time = time.time()
            
            if status in [HealthStatus.CRITICAL, HealthStatus.FAILED]:
                self.failure_count += 1
            else:
                self.failure_count = 0
                
            return status
            
        except Exception as e:
            self.failure_count += 1
            self.last_status = HealthStatus.FAILED
            self.last_check_time = time.time()
            return HealthStatus.FAILED


class MetricsCollector:
    """Advanced metrics collection system for neuromorphic components."""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.current_metrics: Dict[str, Metric] = {}
        self.collection_lock = threading.Lock()
        self.collection_enabled = True
        
    def record_metric(self, 
                     name: str, 
                     value: float, 
                     component: str = "", 
                     unit: str = "",
                     metadata: Optional[Dict] = None):
        """Record a new metric value."""
        if not self.collection_enabled:
            return
            
        metric = Metric(
            name=name,
            value=value,
            component=component,
            unit=unit,
            metadata=metadata or {}
        )
        
        with self.collection_lock:
            metric_key = f"{component}.{name}" if component else name
            self.current_metrics[metric_key] = metric
            self.metrics_history[metric_key].append(metric)
            
    def get_current_metrics(self) -> Dict[str, Metric]:
        """Get current metric values."""
        with self.collection_lock:
            return self.current_metrics.copy()
            
    def get_metric_history(self, metric_name: str, duration: Optional[float] = None) -> List[Metric]:
        """Get metric history for specified duration."""
        with self.collection_lock:
            history = list(self.metrics_history[metric_name])
            
        if duration is not None:
            cutoff_time = time.time() - duration
            history = [m for m in history if m.timestamp >= cutoff_time]
            
        return history
        
    def calculate_statistics(self, metric_name: str, duration: Optional[float] = None) -> Dict:
        """Calculate statistics for a metric over specified duration."""
        history = self.get_metric_history(metric_name, duration)
        
        if not history:
            return {}
            
        values = [m.value for m in history]
        
        stats = {
            'count': len(values),
            'mean': statistics.mean(values),
            'min': min(values),
            'max': max(values),
            'latest': values[-1] if values else None,
            'trend': self._calculate_trend(values)
        }
        
        if len(values) > 1:
            stats['std'] = statistics.stdev(values)
            stats['median'] = statistics.median(values)
            
        return stats
        
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for metric values."""
        if len(values) < 2:
            return "stable"
            
        # Simple linear regression slope
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = statistics.mean(values)
        
        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return "stable"
            
        slope = numerator / denominator
        
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"
            
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format."""
        current = self.get_current_metrics()
        
        if format == "json":
            return json.dumps({
                k: v.to_dict() for k, v in current.items()
            }, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")


class NeuromorphicHealthMonitor:
    """Comprehensive health monitoring for neuromorphic systems."""
    
    def __init__(self):
        self.health_checks: Dict[str, HealthCheck] = {}
        self.metrics_collector = MetricsCollector()
        self.system_state: Dict[str, Any] = {}
        self.monitoring_enabled = True
        self.last_full_check = 0
        self.check_interval = 30.0  # Full system check every 30 seconds
        
        # Register default health checks
        self._register_default_health_checks()
        
    def _register_default_health_checks(self):
        """Register default health checks for neuromorphic systems."""
        
        def check_spike_rate_health(state: Dict) -> float:
            """Check if spike rates are within healthy bounds."""
            spike_rates = state.get('spike_rates', [])
            if not spike_rates:
                return 0.0
            return max(spike_rates) if isinstance(spike_rates, list) else spike_rates
            
        def check_membrane_stability(state: Dict) -> float:
            """Check membrane potential stability."""
            potentials = state.get('membrane_potentials', [])
            if not potentials:
                return 0.0
            # Return coefficient of variation as instability measure
            if len(potentials) > 1:
                mean_pot = statistics.mean(potentials)
                std_pot = statistics.stdev(potentials)
                return std_pot / (abs(mean_pot) + 1e-8)
            return 0.0
            
        def check_sparsity_compliance(state: Dict) -> float:
            """Check sparsity constraint compliance."""
            target_sparsity = state.get('target_sparsity', 0.05)
            actual_sparsity = state.get('actual_sparsity', 0.0)
            return abs(actual_sparsity - target_sparsity)
            
        def check_weight_stability(state: Dict) -> float:
            """Check synaptic weight stability."""
            weights = state.get('synaptic_weights', [])
            if not weights:
                return 0.0
            # Check for weight divergence
            return max(abs(w) for w in weights) if isinstance(weights, list) else abs(weights)
            
        def check_plasticity_rate(state: Dict) -> float:
            """Check plasticity learning rate."""
            plasticity_rate = state.get('plasticity_rate', 0.0)
            return plasticity_rate
            
        # Register health checks
        self.register_health_check(
            "spike_rate",
            check_spike_rate_health,
            threshold_warning=0.8,
            threshold_critical=1.0
        )
        
        self.register_health_check(
            "membrane_stability",
            check_membrane_stability,
            threshold_warning=0.3,
            threshold_critical=0.5
        )
        
        self.register_health_check(
            "sparsity_compliance",
            check_sparsity_compliance,
            threshold_warning=0.1,
            threshold_critical=0.2
        )
        
        self.register_health_check(
            "weight_stability",
            check_weight_stability,
            threshold_warning=5.0,
            threshold_critical=10.0
        )
        
        self.register_health_check(
            "plasticity_rate",
            check_plasticity_rate,
            threshold_warning=0.1,
            threshold_critical=0.2
        )
        
    def register_health_check(self,
                            name: str,
                            check_function: Callable,
                            threshold_warning: Optional[float] = None,
                            threshold_critical: Optional[float] = None,
                            check_interval: float = 60.0):
        """Register a new health check."""
        health_check = HealthCheck(
            name=name,
            check_function=check_function,
            threshold_warning=threshold_warning,
            threshold_critical=threshold_critical,
            check_interval=check_interval
        )
        
        self.health_checks[name] = health_check
        
    def update_system_state(self, state_updates: Dict[str, Any]):
        """Update system state for health monitoring."""
        self.system_state.update(state_updates)
        
        # Record metrics
        for key, value in state_updates.items():
            if isinstance(value, (int, float)):
                self.metrics_collector.record_metric(
                    name=key,
                    value=float(value),
                    component="system"
                )
                
    def run_health_checks(self, force: bool = False) -> Dict[str, HealthStatus]:
        """Run all enabled health checks."""
        if not self.monitoring_enabled and not force:
            return {}
            
        results = {}
        
        for name, health_check in self.health_checks.items():
            if health_check.should_run() or force:
                status = health_check.run_check(self.system_state)
                results[name] = status
                
                # Record health status as metric
                status_value = {
                    HealthStatus.HEALTHY: 1.0,
                    HealthStatus.WARNING: 0.5,
                    HealthStatus.CRITICAL: 0.2,
                    HealthStatus.FAILED: 0.0,
                    HealthStatus.UNKNOWN: -1.0
                }[status]
                
                self.metrics_collector.record_metric(
                    name=f"health_{name}",
                    value=status_value,
                    component="health_monitor"
                )
                
        return results
        
    def get_overall_health(self) -> HealthStatus:
        """Get overall system health status."""
        health_results = self.run_health_checks()
        
        if not health_results:
            return HealthStatus.UNKNOWN
            
        statuses = list(health_results.values())
        
        # Overall health is worst individual status
        if HealthStatus.FAILED in statuses:
            return HealthStatus.FAILED
        elif HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING
        elif HealthStatus.HEALTHY in statuses:
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
            
    def get_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        health_results = self.run_health_checks()
        overall_health = self.get_overall_health()
        
        # Component health details
        component_details = {}
        for name, health_check in self.health_checks.items():
            component_details[name] = {
                'status': health_results.get(name, HealthStatus.UNKNOWN).value,
                'last_check': health_check.last_check_time,
                'failure_count': health_check.failure_count,
                'thresholds': {
                    'warning': health_check.threshold_warning,
                    'critical': health_check.threshold_critical
                }
            }
            
        # Recent metrics summary
        current_metrics = self.metrics_collector.get_current_metrics()
        metrics_summary = {}
        for metric_name, metric in current_metrics.items():
            if metric_name.startswith('health_'):
                continue  # Skip health metrics in summary
            stats = self.metrics_collector.calculate_statistics(metric_name, duration=3600)  # Last hour
            metrics_summary[metric_name] = {
                'current': metric.value,
                'stats': stats
            }
            
        return {
            'timestamp': time.time(),
            'overall_health': overall_health.value,
            'component_health': component_details,
            'metrics_summary': metrics_summary,
            'system_uptime': time.time() - self.last_full_check if self.last_full_check else 0
        }
        
    def export_health_data(self, format: str = "json") -> str:
        """Export health monitoring data."""
        report = self.get_health_report()
        
        if format == "json":
            return json.dumps(report, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")


class PerformanceProfiler:
    """Performance profiling for neuromorphic computations."""
    
    def __init__(self):
        self.profiles: Dict[str, List[float]] = defaultdict(list)
        self.active_profiles: Dict[str, float] = {}
        
    def start_profile(self, name: str):
        """Start timing a computation."""
        self.active_profiles[name] = time.time()
        
    def end_profile(self, name: str) -> float:
        """End timing and record duration."""
        if name not in self.active_profiles:
            warnings.warn(f"Profile '{name}' was not started")
            return 0.0
            
        duration = time.time() - self.active_profiles[name]
        self.profiles[name].append(duration)
        del self.active_profiles[name]
        
        return duration
        
    def profile_function(self, name: str):
        """Decorator for profiling function execution time."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                self.start_profile(name)
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    self.end_profile(name)
            return wrapper
        return decorator
        
    def get_profile_stats(self, name: str) -> Dict:
        """Get performance statistics for a profile."""
        if name not in self.profiles or not self.profiles[name]:
            return {}
            
        durations = self.profiles[name]
        
        return {
            'count': len(durations),
            'total_time': sum(durations),
            'mean_time': statistics.mean(durations),
            'min_time': min(durations),
            'max_time': max(durations),
            'median_time': statistics.median(durations) if len(durations) > 1 else durations[0],
            'std_time': statistics.stdev(durations) if len(durations) > 1 else 0.0
        }
        
    def get_all_profiles(self) -> Dict[str, Dict]:
        """Get statistics for all profiles."""
        return {name: self.get_profile_stats(name) for name in self.profiles.keys()}


class NeuromorphicSystemMonitor:
    """Main monitoring system combining health, metrics, and performance."""
    
    def __init__(self):
        self.health_monitor = NeuromorphicHealthMonitor()
        self.performance_profiler = PerformanceProfiler()
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_active = False
        self.monitor_interval = 10.0  # seconds
        
    def start_monitoring(self):
        """Start background monitoring thread."""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
            
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                # Run health checks
                self.health_monitor.run_health_checks()
                
                # Record system metrics
                self._collect_system_metrics()
                
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                warnings.warn(f"Monitoring loop error: {e}")
                time.sleep(1.0)  # Short delay before retry
                
    def _collect_system_metrics(self):
        """Collect system-level metrics."""
        # Memory usage, CPU usage, etc. would be implemented here
        # For now, just record a timestamp
        self.health_monitor.metrics_collector.record_metric(
            name="monitoring_heartbeat",
            value=time.time(),
            component="system_monitor"
        )
        
    def update_component_state(self, component_name: str, state: Dict[str, Any]):
        """Update state for a specific component."""
        prefixed_state = {f"{component_name}.{k}": v for k, v in state.items()}
        self.health_monitor.update_system_state(prefixed_state)
        
    def profile_component(self, component_name: str):
        """Get profiling decorator for a component."""
        return self.performance_profiler.profile_function(component_name)
        
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data."""
        return {
            'health_report': self.health_monitor.get_health_report(),
            'performance_profiles': self.performance_profiler.get_all_profiles(),
            'monitoring_status': {
                'active': self.monitoring_active,
                'interval': self.monitor_interval,
                'thread_alive': self.monitoring_thread.is_alive() if self.monitoring_thread else False
            }
        }
        
    def export_monitoring_data(self, format: str = "json") -> str:
        """Export all monitoring data."""
        dashboard = self.get_monitoring_dashboard()
        
        if format == "json":
            return json.dumps(dashboard, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Global monitoring instance
global_monitor = NeuromorphicSystemMonitor()


def monitored_component(component_name: str):
    """Decorator to add monitoring to neuromorphic components."""
    def decorator(cls):
        class MonitoredComponent(cls):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.component_name = component_name
                self.monitor = global_monitor
                
            def forward(self, *args, **kwargs):
                # Profile execution
                with self.monitor.performance_profiler.profile_function(f"{self.component_name}.forward"):
                    result = super().forward(*args, **kwargs)
                    
                # Update health state
                self._update_health_state(result)
                
                return result
                
            def _update_health_state(self, result):
                """Update component health state based on results."""
                state_updates = {}
                
                # Extract relevant metrics from result
                if hasattr(result, 'shape'):
                    state_updates['output_shape'] = list(result.shape)
                    
                if hasattr(result, 'mean'):
                    state_updates['output_mean'] = float(result.mean())
                    
                if hasattr(result, 'std'):
                    state_updates['output_std'] = float(result.std())
                    
                self.monitor.update_component_state(self.component_name, state_updates)
                
        return MonitoredComponent
    return decorator