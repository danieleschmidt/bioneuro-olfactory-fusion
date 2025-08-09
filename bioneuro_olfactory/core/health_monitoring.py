"""Health monitoring and system diagnostics for neuromorphic gas detection."""

import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import json
from pathlib import Path

from .error_handling import get_error_handler, ErrorSeverity, BioNeuroError


class HealthStatus:
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthMetric:
    """Individual health metric."""
    name: str
    value: float
    unit: str
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_status(self) -> str:
        """Get health status based on thresholds."""
        if self.threshold_critical is not None and self.value >= self.threshold_critical:
            return HealthStatus.CRITICAL
        elif self.threshold_warning is not None and self.value >= self.threshold_warning:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY


@dataclass
class SystemHealth:
    """Overall system health summary."""
    overall_status: str
    components: Dict[str, str]
    metrics: Dict[str, HealthMetric]
    alerts: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "overall_status": self.overall_status,
            "components": self.components,
            "metrics": {name: {
                "value": metric.value,
                "unit": metric.unit,
                "status": metric.get_status(),
                "timestamp": metric.timestamp.isoformat()
            } for name, metric in self.metrics.items()},
            "alerts": self.alerts,
            "timestamp": self.timestamp.isoformat()
        }


class HealthMonitor:
    """Comprehensive health monitoring system."""
    
    def __init__(
        self,
        check_interval: float = 30.0,
        history_size: int = 1000,
        alert_callback: Optional[Callable] = None
    ):
        self.check_interval = check_interval
        self.history_size = history_size
        self.alert_callback = alert_callback
        
        # Health data storage
        self.current_metrics: Dict[str, HealthMetric] = {}
        self.metric_history: Dict[str, deque] = {}
        self.component_status: Dict[str, str] = {}
        self.active_alerts: List[str] = []
        
        # Monitoring thread
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Health checkers
        self.health_checkers: Dict[str, Callable] = {}
        
        # Error handler
        self.error_handler = get_error_handler()
        
        # Register default health checks
        self._register_default_checks()
        
    def _register_default_checks(self):
        """Register default health checks."""
        self.register_health_check("system_uptime", self._check_system_uptime)
        self.register_health_check("memory_usage", self._check_memory_usage)
        self.register_health_check("error_rate", self._check_error_rate)
        self.register_health_check("response_time", self._check_response_time)
        
    def register_health_check(self, name: str, checker_func: Callable):
        """Register a health check function."""
        self.health_checkers[name] = checker_func
        self.error_handler.logger.info(f"Registered health check: {name}")
        
    def start_monitoring(self):
        """Start continuous health monitoring."""
        if self.monitoring_active:
            self.error_handler.logger.warning("Health monitoring already active")
            return
            
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        self.error_handler.logger.info("Health monitoring started")
        
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        self.error_handler.logger.info("Health monitoring stopped")
        
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self.perform_health_check()
                time.sleep(self.check_interval)
            except Exception as e:
                self.error_handler.handle_error(
                    BioNeuroError(
                        f"Health monitoring loop error: {e}",
                        error_code="MONITOR_LOOP_ERROR",
                        severity=ErrorSeverity.HIGH
                    )
                )
                time.sleep(self.check_interval)  # Continue monitoring despite errors
                
    def perform_health_check(self) -> SystemHealth:
        """Perform comprehensive health check."""
        new_metrics = {}
        component_statuses = {}
        alerts = []
        
        # Run all registered health checks
        for check_name, checker_func in self.health_checkers.items():
            try:
                metric = checker_func()
                if metric:
                    new_metrics[check_name] = metric
                    
                    # Update component status based on metric
                    component_statuses[check_name] = metric.get_status()
                    
                    # Generate alerts if needed
                    if metric.get_status() in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                        alert_msg = f"{check_name}: {metric.value} {metric.unit} ({metric.get_status()})"
                        alerts.append(alert_msg)
                        
            except Exception as e:
                self.error_handler.logger.error(f"Health check failed for {check_name}: {e}")
                component_statuses[check_name] = HealthStatus.UNKNOWN
                
        # Update current metrics
        self.current_metrics = new_metrics
        self.component_status = component_statuses
        
        # Update metric history
        for metric_name, metric in new_metrics.items():
            if metric_name not in self.metric_history:
                self.metric_history[metric_name] = deque(maxlen=self.history_size)
            self.metric_history[metric_name].append(metric)
            
        # Determine overall status
        overall_status = self._calculate_overall_status(component_statuses)
        
        # Update active alerts
        self.active_alerts = alerts
        
        # Create system health summary
        system_health = SystemHealth(
            overall_status=overall_status,
            components=component_statuses,
            metrics=new_metrics,
            alerts=alerts
        )
        
        # Trigger alert callback if needed
        if self.alert_callback and alerts:
            try:
                self.alert_callback(system_health)
            except Exception as e:
                self.error_handler.logger.error(f"Alert callback failed: {e}")
                
        return system_health
        
    def _calculate_overall_status(self, component_statuses: Dict[str, str]) -> str:
        """Calculate overall system status from component statuses."""
        if not component_statuses:
            return HealthStatus.UNKNOWN
            
        # Priority: CRITICAL > DEGRADED > WARNING > HEALTHY
        if HealthStatus.CRITICAL in component_statuses.values():
            return HealthStatus.CRITICAL
        elif HealthStatus.DEGRADED in component_statuses.values():
            return HealthStatus.DEGRADED
        elif HealthStatus.WARNING in component_statuses.values():
            return HealthStatus.WARNING
        elif all(status == HealthStatus.HEALTHY for status in component_statuses.values()):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
            
    def get_current_health(self) -> SystemHealth:
        """Get current system health status."""
        return SystemHealth(
            overall_status=self._calculate_overall_status(self.component_status),
            components=dict(self.component_status),
            metrics=dict(self.current_metrics),
            alerts=list(self.active_alerts)
        )
        
    def get_metric_history(self, metric_name: str, duration: Optional[timedelta] = None) -> List[HealthMetric]:
        """Get metric history for a specific metric."""
        if metric_name not in self.metric_history:
            return []
            
        history = list(self.metric_history[metric_name])
        
        if duration:
            cutoff_time = datetime.now() - duration
            history = [metric for metric in history if metric.timestamp >= cutoff_time]
            
        return history
        
    def export_health_report(self, file_path: str):
        """Export comprehensive health report to file."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "current_health": self.get_current_health().to_dict(),
            "metric_history": {},
            "error_statistics": self.error_handler.get_error_statistics()
        }
        
        # Add metric history
        for metric_name, history in self.metric_history.items():
            report["metric_history"][metric_name] = [
                {
                    "value": metric.value,
                    "timestamp": metric.timestamp.isoformat()
                }
                for metric in history
            ]
            
        # Write to file
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        self.error_handler.logger.info(f"Health report exported to {file_path}")
        
    # Default health check implementations
    def _check_system_uptime(self) -> HealthMetric:
        """Check system uptime."""
        # Simple uptime check - in production would use actual system metrics
        try:
            with open('/proc/uptime', 'r') as f:
                uptime_seconds = float(f.read().split()[0])
        except:
            # Fallback for systems without /proc/uptime
            uptime_seconds = time.time()  # Approximate
            
        return HealthMetric(
            name="system_uptime",
            value=uptime_seconds / 3600,  # Convert to hours
            unit="hours"
        )
        
    def _check_memory_usage(self) -> HealthMetric:
        """Check memory usage."""
        try:
            # Try to get actual memory usage
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
                
            mem_total = 0
            mem_available = 0
            
            for line in meminfo.split('\n'):
                if line.startswith('MemTotal:'):
                    mem_total = int(line.split()[1]) * 1024  # Convert kB to bytes
                elif line.startswith('MemAvailable:'):
                    mem_available = int(line.split()[1]) * 1024
                    
            if mem_total > 0:
                usage_percent = ((mem_total - mem_available) / mem_total) * 100
            else:
                usage_percent = 0
                
        except:
            # Fallback - simulate memory usage
            usage_percent = 45.0  # Simulated value
            
        return HealthMetric(
            name="memory_usage",
            value=usage_percent,
            unit="percent",
            threshold_warning=75.0,
            threshold_critical=90.0
        )
        
    def _check_error_rate(self) -> HealthMetric:
        """Check error rate from error handler."""
        stats = self.error_handler.get_error_statistics()
        
        # Calculate errors per minute (simplified)
        total_errors = stats.get("total_errors", 0)
        error_rate = total_errors / max(1, time.time() / 60)  # Errors per minute
        
        return HealthMetric(
            name="error_rate",
            value=error_rate,
            unit="errors/min",
            threshold_warning=1.0,
            threshold_critical=5.0
        )
        
    def _check_response_time(self) -> HealthMetric:
        """Check system response time."""
        # Simulate response time measurement
        start_time = time.time()
        
        # Simple operation to measure
        try:
            # Simulate some computation
            result = sum(i * i for i in range(1000))
            response_time = (time.time() - start_time) * 1000  # Convert to ms
        except:
            response_time = 999.0  # High value to indicate problem
            
        return HealthMetric(
            name="response_time",
            value=response_time,
            unit="ms",
            threshold_warning=100.0,
            threshold_critical=500.0
        )


# Specialized health checks for neuromorphic components
class NeuromorphicHealthMonitor(HealthMonitor):
    """Specialized health monitor for neuromorphic components."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Register neuromorphic-specific checks
        self.register_health_check("spike_rate", self._check_spike_rate)
        self.register_health_check("network_sparsity", self._check_network_sparsity)
        self.register_health_check("learning_rate", self._check_learning_rate)
        self.register_health_check("sensor_calibration", self._check_sensor_calibration)
        
    def _check_spike_rate(self) -> HealthMetric:
        """Check neural network spike rate."""
        # Simulate spike rate monitoring
        # In production, this would monitor actual network activity
        spike_rate = 25.0 + (time.time() % 10) * 2  # Simulated varying spike rate
        
        return HealthMetric(
            name="spike_rate",
            value=spike_rate,
            unit="Hz",
            threshold_warning=100.0,  # Too high spike rate
            threshold_critical=200.0
        )
        
    def _check_network_sparsity(self) -> HealthMetric:
        """Check network sparsity level."""
        # Simulate sparsity monitoring
        # Ideal sparsity for Kenyon cells is around 5%
        sparsity = 0.05 + (time.time() % 20) * 0.001  # Simulated sparsity
        
        # Convert to percentage
        sparsity_percent = sparsity * 100
        
        return HealthMetric(
            name="network_sparsity",
            value=sparsity_percent,
            unit="percent",
            threshold_warning=10.0,  # Too dense
            threshold_critical=20.0
        )
        
    def _check_learning_rate(self) -> HealthMetric:
        """Check learning adaptation rate."""
        # Simulate learning rate monitoring
        learning_rate = 0.01 * (1 + 0.1 * (time.time() % 30) / 30)  # Simulated learning rate
        
        return HealthMetric(
            name="learning_rate",
            value=learning_rate,
            unit="rate",
            threshold_warning=0.1,  # Learning rate too high
            threshold_critical=0.5
        )
        
    def _check_sensor_calibration(self) -> HealthMetric:
        """Check sensor calibration drift."""
        # Simulate calibration drift monitoring
        # Drift as percentage from baseline
        drift = abs(1.0 + 0.05 * (time.time() % 60) / 60)  # Simulated drift
        drift_percent = (drift - 1.0) * 100
        
        return HealthMetric(
            name="sensor_calibration",
            value=drift_percent,
            unit="percent_drift",
            threshold_warning=5.0,  # 5% drift
            threshold_critical=10.0  # 10% drift
        )


# Global health monitor instance
_health_monitor = None


def get_health_monitor() -> HealthMonitor:
    """Get global health monitor instance."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = NeuromorphicHealthMonitor()
    return _health_monitor


# Health check decorator
def health_check(metric_name: str):
    """Decorator to automatically register function as health check."""
    def decorator(func):
        monitor = get_health_monitor()
        monitor.register_health_check(metric_name, func)
        return func
    return decorator