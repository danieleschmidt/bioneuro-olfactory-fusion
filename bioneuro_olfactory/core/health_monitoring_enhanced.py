"""Enhanced health monitoring and system diagnostics for neuromorphic gas detection."""

import time
import threading
import psutil
import GPUtil
from typing import Dict, Any, List, Optional, Callable, Union, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import deque, defaultdict
import json
from pathlib import Path
import logging
import asyncio
import statistics
from enum import Enum
import uuid
import socket
from concurrent.futures import ThreadPoolExecutor
import gc

from .error_handling_enhanced import get_enhanced_error_handler, BioNeuroError, ErrorSeverity

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"
    MAINTENANCE = "maintenance"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthMetric:
    """Enhanced health metric with trends and predictions."""
    name: str
    value: float
    unit: str
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    trend: Optional[str] = None  # "increasing", "decreasing", "stable"
    prediction: Optional[float] = None  # Predicted value for next measurement
    confidence: float = 1.0  # Confidence in the measurement (0-1)
    
    def get_status(self) -> HealthStatus:
        """Get health status based on thresholds with trend consideration."""
        status = HealthStatus.HEALTHY
        
        if self.threshold_critical is not None and self.value >= self.threshold_critical:
            status = HealthStatus.CRITICAL
        elif self.threshold_warning is not None and self.value >= self.threshold_warning:
            status = HealthStatus.WARNING
        
        # Upgrade status if trend is concerning
        if self.trend == "increasing" and status != HealthStatus.HEALTHY:
            if status == HealthStatus.WARNING:
                status = HealthStatus.DEGRADED
            elif status == HealthStatus.DEGRADED:
                status = HealthStatus.CRITICAL
                
        return status
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "threshold_warning": self.threshold_warning,
            "threshold_critical": self.threshold_critical,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
            "trend": self.trend,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "status": self.get_status().value
        }


@dataclass
class SystemAlert:
    """System alert with correlation and suppression support."""
    alert_id: str
    alert_type: str
    severity: AlertSeverity
    message: str
    component: str
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    correlation_id: Optional[str] = None
    suppressed: bool = False
    suppression_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type,
            "severity": self.severity.value,
            "message": self.message,
            "component": self.component,
            "timestamp": self.timestamp.isoformat(),
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "correlation_id": self.correlation_id,
            "suppressed": self.suppressed,
            "suppression_reason": self.suppression_reason,
            "metadata": self.metadata
        }


@dataclass
class SystemHealth:
    """Enhanced system health summary with predictions and recommendations."""
    overall_status: HealthStatus
    components: Dict[str, HealthStatus]
    metrics: Dict[str, HealthMetric]
    alerts: List[SystemAlert]
    timestamp: datetime = field(default_factory=datetime.now)
    system_load: float = 0.0
    uptime_seconds: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    performance_score: float = 1.0  # Overall performance score (0-1)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "overall_status": self.overall_status.value,
            "components": {name: status.value for name, status in self.components.items()},
            "metrics": {name: metric.to_dict() for name, metric in self.metrics.items()},
            "alerts": [alert.to_dict() for alert in self.alerts],
            "timestamp": self.timestamp.isoformat(),
            "system_load": self.system_load,
            "uptime_seconds": self.uptime_seconds,
            "recommendations": self.recommendations,
            "performance_score": self.performance_score
        }


class MetricCollector:
    """Enhanced metric collector with trend analysis and prediction."""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))
        self.trend_analyzers: Dict[str, Callable] = {}
        self.prediction_models: Dict[str, Callable] = {}
        
    def add_metric(self, metric: HealthMetric):
        """Add metric with trend analysis."""
        # Store in history
        self.metrics_history[metric.name].append(metric)
        
        # Analyze trend if we have enough history
        if len(self.metrics_history[metric.name]) >= 3:
            metric.trend = self._analyze_trend(metric.name)
            
        # Make prediction if model available
        if metric.name in self.prediction_models:
            try:
                metric.prediction = self.prediction_models[metric.name](
                    list(self.metrics_history[metric.name])
                )
            except Exception as e:
                logger.warning(f"Prediction failed for {metric.name}: {e}")
                
    def _analyze_trend(self, metric_name: str) -> Optional[str]:
        """Analyze trend for a metric."""
        if metric_name in self.trend_analyzers:
            return self.trend_analyzers[metric_name](list(self.metrics_history[metric_name]))
        else:
            return self._default_trend_analysis(metric_name)
            
    def _default_trend_analysis(self, metric_name: str) -> Optional[str]:
        """Default trend analysis using simple moving average."""
        history = list(self.metrics_history[metric_name])
        if len(history) < 3:
            return None
            
        values = [m.value for m in history[-3:]]
        
        if len(values) >= 3:
            # Simple trend detection
            recent_avg = statistics.mean(values[-2:])
            older_avg = statistics.mean(values[:-2])
            
            threshold = 0.05 * abs(older_avg) if older_avg != 0 else 0.05
            
            if recent_avg > older_avg + threshold:
                return "increasing"
            elif recent_avg < older_avg - threshold:
                return "decreasing"
            else:
                return "stable"
        
        return None
        
    def get_metric_statistics(self, metric_name: str, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get statistics for a metric."""
        if metric_name not in self.metrics_history:
            return {}
            
        history = list(self.metrics_history[metric_name])
        
        # Filter by time window if specified
        if time_window:
            cutoff_time = datetime.now() - time_window
            history = [m for m in history if m.timestamp >= cutoff_time]
            
        if not history:
            return {}
            
        values = [m.value for m in history]
        
        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "first_value": values[0],
            "last_value": values[-1],
            "trend": history[-1].trend if history else None,
            "time_range": {
                "start": history[0].timestamp.isoformat(),
                "end": history[-1].timestamp.isoformat()
            } if history else None
        }


class AlertManager:
    """Enhanced alert manager with correlation, suppression, and escalation."""
    
    def __init__(self, max_alerts: int = 10000):
        self.active_alerts: Dict[str, SystemAlert] = {}
        self.alert_history = deque(maxlen=max_alerts)
        self.suppression_rules: List[Dict[str, Any]] = []
        self.correlation_rules: List[Dict[str, Any]] = []
        self.escalation_policies: Dict[str, Dict[str, Any]] = {}
        self.callbacks: List[Callable[[SystemAlert], None]] = []
        
    def create_alert(
        self,
        alert_type: str,
        severity: AlertSeverity,
        message: str,
        component: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SystemAlert:
        """Create a new alert with correlation and suppression checking."""
        alert = SystemAlert(
            alert_id=str(uuid.uuid4()),
            alert_type=alert_type,
            severity=severity,
            message=message,
            component=component,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        # Check suppression rules
        if self._should_suppress_alert(alert):
            alert.suppressed = True
            alert.suppression_reason = "Matched suppression rule"
            
        # Check correlation rules
        correlation_id = self._find_correlation(alert)
        if correlation_id:
            alert.correlation_id = correlation_id
            
        # Store alert
        if not alert.suppressed:
            self.active_alerts[alert.alert_id] = alert
            
        self.alert_history.append(alert)
        
        # Trigger callbacks
        for callback in self.callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
                
        return alert
        
    def resolve_alert(self, alert_id: str, resolution_note: Optional[str] = None):
        """Resolve an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.now()
            
            if resolution_note:
                alert.metadata["resolution_note"] = resolution_note
                
            del self.active_alerts[alert_id]
            logger.info(f"Alert {alert_id} resolved: {alert.message}")
            
    def _should_suppress_alert(self, alert: SystemAlert) -> bool:
        """Check if alert should be suppressed."""
        for rule in self.suppression_rules:
            if self._match_rule(alert, rule):
                return True
        return False
        
    def _find_correlation(self, alert: SystemAlert) -> Optional[str]:
        """Find correlation ID for alert."""
        for rule in self.correlation_rules:
            if self._match_rule(alert, rule):
                # Simple correlation - use component as correlation ID
                return f"correlation_{alert.component}_{alert.alert_type}"
        return None
        
    def _match_rule(self, alert: SystemAlert, rule: Dict[str, Any]) -> bool:
        """Check if alert matches a rule."""
        for key, pattern in rule.items():
            alert_value = getattr(alert, key, None)
            if alert_value is None:
                return False
            if isinstance(pattern, str) and pattern not in str(alert_value):
                return False
            if isinstance(pattern, list) and alert_value not in pattern:
                return False
        return True
        
    def add_suppression_rule(
        self,
        alert_type: Optional[str] = None,
        component: Optional[str] = None,
        severity: Optional[AlertSeverity] = None,
        duration_minutes: int = 60
    ):
        """Add alert suppression rule."""
        rule = {}
        if alert_type:
            rule["alert_type"] = alert_type
        if component:
            rule["component"] = component
        if severity:
            rule["severity"] = severity
            
        rule["duration_minutes"] = duration_minutes
        rule["created_at"] = datetime.now()
        
        self.suppression_rules.append(rule)
        logger.info(f"Added suppression rule: {rule}")
        
    def get_alert_statistics(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get alert statistics."""
        alerts_to_analyze = list(self.alert_history)
        
        if time_window:
            cutoff_time = datetime.now() - time_window
            alerts_to_analyze = [a for a in alerts_to_analyze if a.timestamp >= cutoff_time]
            
        severity_counts = defaultdict(int)
        component_counts = defaultdict(int)
        type_counts = defaultdict(int)
        
        for alert in alerts_to_analyze:
            severity_counts[alert.severity.value] += 1
            component_counts[alert.component] += 1
            type_counts[alert.alert_type] += 1
            
        return {
            "total_alerts": len(alerts_to_analyze),
            "active_alerts": len(self.active_alerts),
            "severity_distribution": dict(severity_counts),
            "component_distribution": dict(component_counts),
            "type_distribution": dict(type_counts),
            "suppressed_alerts": sum(1 for a in alerts_to_analyze if a.suppressed),
            "resolved_alerts": sum(1 for a in alerts_to_analyze if a.resolved)
        }


class PerformanceProfiler:
    """Enhanced performance profiler with detailed system analysis."""
    
    def __init__(self):
        self.profiles: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.active_profiles: Dict[str, datetime] = {}
        self.system_baseline: Optional[Dict[str, float]] = None
        self.gpu_available = self._check_gpu_availability()
        
    def _check_gpu_availability(self) -> bool:
        """Check if GPU monitoring is available."""
        try:
            GPUtil.getGPUs()
            return True
        except Exception:
            return False
            
    def start_profiling(self, operation_name: str):
        """Start profiling an operation."""
        self.active_profiles[operation_name] = datetime.now()
        
    def end_profiling(self, operation_name: str, additional_metrics: Optional[Dict[str, Any]] = None):
        """End profiling and record metrics."""
        if operation_name not in self.active_profiles:
            logger.warning(f"No active profile found for {operation_name}")
            return
            
        start_time = self.active_profiles[operation_name]
        duration = (datetime.now() - start_time).total_seconds()
        
        profile_data = {
            "duration_seconds": duration,
            "timestamp": start_time.isoformat(),
            "system_metrics": self._collect_system_metrics()
        }
        
        if additional_metrics:
            profile_data.update(additional_metrics)
            
        self.profiles[operation_name].append(profile_data)
        del self.active_profiles[operation_name]
        
        # Keep only recent profiles (last 1000 per operation)
        if len(self.profiles[operation_name]) > 1000:
            self.profiles[operation_name] = self.profiles[operation_name][-1000:]
            
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics."""
        metrics = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_io": dict(psutil.disk_io_counters()._asdict()) if psutil.disk_io_counters() else {},
            "network_io": dict(psutil.net_io_counters()._asdict()) if psutil.net_io_counters() else {},
            "open_files": len(psutil.Process().open_files()),
            "threads": psutil.Process().num_threads()
        }
        
        # Add GPU metrics if available
        if self.gpu_available:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    metrics["gpu"] = {
                        "utilization": gpu.load * 100,
                        "memory_percent": gpu.memoryUtil * 100,
                        "temperature": gpu.temperature,
                        "memory_used_mb": gpu.memoryUsed,
                        "memory_total_mb": gpu.memoryTotal
                    }
            except Exception as e:
                logger.debug(f"GPU metrics collection failed: {e}")
                
        return metrics
        
    def get_operation_statistics(self, operation_name: str) -> Dict[str, Any]:
        """Get statistics for a specific operation."""
        if operation_name not in self.profiles:
            return {}
            
        profiles = self.profiles[operation_name]
        durations = [p["duration_seconds"] for p in profiles]
        
        stats = {
            "total_runs": len(profiles),
            "avg_duration": statistics.mean(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "median_duration": statistics.median(durations),
            "std_duration": statistics.stdev(durations) if len(durations) > 1 else 0.0
        }
        
        # Add system resource statistics
        if profiles:
            cpu_values = [p["system_metrics"].get("cpu_percent", 0) for p in profiles]
            memory_values = [p["system_metrics"].get("memory_percent", 0) for p in profiles]
            
            stats["resource_usage"] = {
                "avg_cpu_percent": statistics.mean(cpu_values),
                "avg_memory_percent": statistics.mean(memory_values),
                "max_cpu_percent": max(cpu_values),
                "max_memory_percent": max(memory_values)
            }
            
        return stats
        
    def set_baseline(self):
        """Set current system state as baseline for comparison."""
        self.system_baseline = self._collect_system_metrics()
        logger.info("Performance baseline set")
        
    def compare_to_baseline(self) -> Dict[str, float]:
        """Compare current metrics to baseline."""
        if not self.system_baseline:
            return {}
            
        current = self._collect_system_metrics()
        comparison = {}
        
        for key in ["cpu_percent", "memory_percent"]:
            if key in self.system_baseline and key in current:
                baseline_val = self.system_baseline[key]
                current_val = current[key]
                if baseline_val > 0:
                    comparison[f"{key}_change_percent"] = ((current_val - baseline_val) / baseline_val) * 100
                    
        return comparison


class EnhancedHealthMonitor:
    """Comprehensive health monitoring system with advanced features."""
    
    def __init__(
        self,
        check_interval: float = 30.0,
        history_size: int = 1000,
        alert_callback: Optional[Callable] = None,
        enable_predictions: bool = True,
        enable_gpu_monitoring: bool = True
    ):
        self.check_interval = check_interval
        self.history_size = history_size
        self.alert_callback = alert_callback
        self.enable_predictions = enable_predictions
        self.enable_gpu_monitoring = enable_gpu_monitoring
        
        # Core components
        self.metric_collector = MetricCollector(history_size)
        self.alert_manager = AlertManager()
        self.performance_profiler = PerformanceProfiler()
        
        # Health data storage
        self.current_metrics: Dict[str, HealthMetric] = {}
        self.component_status: Dict[str, HealthStatus] = {}
        
        # Monitoring control
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="health_monitor")
        
        # Health checkers
        self.health_checkers: Dict[str, Callable] = {}
        self.async_checkers: Dict[str, Callable] = {}
        
        # Advanced features
        self.predictive_alerts: Set[str] = set()
        self.maintenance_mode = False
        self.maintenance_start: Optional[datetime] = None
        
        # Error handler
        self.error_handler = get_enhanced_error_handler()
        
        # Register default health checks
        self._register_default_checks()
        
        # Set up alert callbacks
        if alert_callback:
            self.alert_manager.callbacks.append(alert_callback)
            
    def _register_default_checks(self):
        """Register comprehensive default health checks."""
        self.register_health_check("system_resources", self._check_system_resources)
        self.register_health_check("memory_health", self._check_memory_health)
        self.register_health_check("disk_health", self._check_disk_health)
        self.register_health_check("network_health", self._check_network_health)
        self.register_health_check("process_health", self._check_process_health)
        
        if self.enable_gpu_monitoring:
            self.register_health_check("gpu_health", self._check_gpu_health)
            
        # Neuromorphic-specific checks
        self.register_health_check("spike_rate", self._check_spike_rate)
        self.register_health_check("network_sparsity", self._check_network_sparsity)
        self.register_health_check("learning_stability", self._check_learning_stability)
        self.register_health_check("sensor_calibration", self._check_sensor_calibration)
        
    def register_health_check(self, name: str, checker_func: Callable):
        """Register a health check function."""
        if asyncio.iscoroutinefunction(checker_func):
            self.async_checkers[name] = checker_func
        else:
            self.health_checkers[name] = checker_func
            
        self.error_handler.logger.info(f"Registered health check: {name}")
        
    def start_monitoring(self):
        """Start comprehensive health monitoring."""
        if self.monitoring_active:
            self.error_handler.logger.warning("Health monitoring already active")
            return
            
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        # Set performance baseline
        self.performance_profiler.set_baseline()
        
        self.error_handler.logger.info("Enhanced health monitoring started")
        
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
            
        self.executor.shutdown(wait=True)
        self.error_handler.logger.info("Health monitoring stopped")
        
    def _monitoring_loop(self):
        """Main monitoring loop with error handling."""
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
        """Perform comprehensive health check with parallel execution."""
        start_time = datetime.now()
        
        # Execute health checks in parallel
        futures = []
        for check_name, checker_func in self.health_checkers.items():
            future = self.executor.submit(self._execute_health_check, check_name, checker_func)
            futures.append((check_name, future))
            
        # Collect results
        new_metrics = {}
        component_statuses = {}
        alerts = []
        
        for check_name, future in futures:
            try:
                result = future.result(timeout=30.0)  # 30 second timeout per check
                if result:
                    if isinstance(result, HealthMetric):
                        new_metrics[check_name] = result
                        self.metric_collector.add_metric(result)
                        component_statuses[check_name] = result.get_status()
                        
                        # Generate alerts if needed
                        self._check_metric_alerts(result)
                        
            except Exception as e:
                self.error_handler.logger.error(f"Health check failed for {check_name}: {e}")
                component_statuses[check_name] = HealthStatus.UNKNOWN
                
                # Create alert for failed health check
                alert = self.alert_manager.create_alert(
                    alert_type="health_check_failed",
                    severity=AlertSeverity.WARNING,
                    message=f"Health check failed: {check_name} - {str(e)}",
                    component=check_name,
                    metadata={"exception": str(e)}
                )
                alerts.append(alert)
                
        # Update current state
        self.current_metrics = new_metrics
        self.component_status = component_statuses
        
        # Calculate overall status and performance score
        overall_status = self._calculate_overall_status(component_statuses)
        performance_score = self._calculate_performance_score()
        
        # Get active alerts
        alerts.extend(list(self.alert_manager.active_alerts.values()))
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        # Create system health summary
        system_health = SystemHealth(
            overall_status=overall_status,
            components=component_statuses,
            metrics=new_metrics,
            alerts=alerts,
            system_load=psutil.cpu_percent(interval=0.1),
            uptime_seconds=time.time() - psutil.boot_time(),
            recommendations=recommendations,
            performance_score=performance_score
        )
        
        # Log performance metrics
        check_duration = (datetime.now() - start_time).total_seconds()
        self.error_handler.logger.debug(
            f"Health check completed in {check_duration:.2f}s",
            extra={
                "structured_data": {
                    "event_type": "health_check_completed",
                    "duration_seconds": check_duration,
                    "metrics_count": len(new_metrics),
                    "alerts_count": len(alerts),
                    "overall_status": overall_status.value,
                    "performance_score": performance_score
                }
            }
        )
        
        return system_health
        
    def _execute_health_check(self, check_name: str, checker_func: Callable) -> Optional[HealthMetric]:
        """Execute a single health check with profiling."""
        self.performance_profiler.start_profiling(f"health_check_{check_name}")
        
        try:
            result = checker_func()
            return result
        finally:
            self.performance_profiler.end_profiling(f"health_check_{check_name}")
            
    def _check_metric_alerts(self, metric: HealthMetric):
        """Check if metric should trigger alerts."""
        status = metric.get_status()
        
        if status in [HealthStatus.WARNING, HealthStatus.DEGRADED, HealthStatus.CRITICAL]:
            severity_map = {
                HealthStatus.WARNING: AlertSeverity.WARNING,
                HealthStatus.DEGRADED: AlertSeverity.ERROR,
                HealthStatus.CRITICAL: AlertSeverity.CRITICAL
            }
            
            alert = self.alert_manager.create_alert(
                alert_type="metric_threshold_exceeded",
                severity=severity_map[status],
                message=f"Metric {metric.name} exceeded threshold: {metric.value} {metric.unit}",
                component=metric.name,
                metadata={
                    "metric_value": metric.value,
                    "threshold_warning": metric.threshold_warning,
                    "threshold_critical": metric.threshold_critical,
                    "trend": metric.trend
                }
            )
            
        # Predictive alerts based on trends and predictions
        if self.enable_predictions and metric.prediction is not None:
            if (metric.threshold_critical is not None and 
                metric.prediction >= metric.threshold_critical * 0.9):  # 90% of critical threshold
                
                alert_key = f"predictive_{metric.name}"
                if alert_key not in self.predictive_alerts:
                    self.alert_manager.create_alert(
                        alert_type="predictive_threshold_warning",
                        severity=AlertSeverity.WARNING,
                        message=f"Metric {metric.name} predicted to exceed threshold: {metric.prediction:.2f} {metric.unit}",
                        component=metric.name,
                        metadata={
                            "predicted_value": metric.prediction,
                            "current_value": metric.value,
                            "threshold_critical": metric.threshold_critical,
                            "confidence": metric.confidence
                        }
                    )
                    self.predictive_alerts.add(alert_key)
            else:
                # Remove from predictive alerts if no longer needed
                self.predictive_alerts.discard(f"predictive_{metric.name}")
                
    def _calculate_overall_status(self, component_statuses: Dict[str, HealthStatus]) -> HealthStatus:
        """Calculate overall system status with maintenance mode consideration."""
        if self.maintenance_mode:
            return HealthStatus.MAINTENANCE
            
        if not component_statuses:
            return HealthStatus.UNKNOWN
            
        # Priority: CRITICAL > DEGRADED > WARNING > HEALTHY
        status_priority = {
            HealthStatus.CRITICAL: 4,
            HealthStatus.DEGRADED: 3,
            HealthStatus.WARNING: 2,
            HealthStatus.HEALTHY: 1,
            HealthStatus.UNKNOWN: 0
        }
        
        max_priority = max(status_priority.get(status, 0) for status in component_statuses.values())
        
        for status, priority in status_priority.items():
            if priority == max_priority:
                return status
                
        return HealthStatus.UNKNOWN
        
    def _calculate_performance_score(self) -> float:
        """Calculate overall system performance score."""
        if not self.current_metrics:
            return 1.0
            
        scores = []
        
        # CPU performance score
        if "system_resources" in self.current_metrics:
            cpu_metric = self.current_metrics["system_resources"]
            if hasattr(cpu_metric, 'tags') and 'cpu_percent' in cpu_metric.tags:
                cpu_usage = float(cpu_metric.tags['cpu_percent'])
                cpu_score = max(0.0, 1.0 - cpu_usage / 100.0)
                scores.append(cpu_score)
                
        # Memory performance score
        if "memory_health" in self.current_metrics:
            memory_metric = self.current_metrics["memory_health"]
            memory_score = max(0.0, 1.0 - memory_metric.value / 100.0)
            scores.append(memory_score)
            
        # Component health score
        healthy_components = sum(1 for status in self.component_status.values() 
                               if status == HealthStatus.HEALTHY)
        total_components = len(self.component_status)
        
        if total_components > 0:
            health_score = healthy_components / total_components
            scores.append(health_score)
            
        return statistics.mean(scores) if scores else 1.0
        
    def _generate_recommendations(self) -> List[str]:
        """Generate system recommendations based on current state."""
        recommendations = []
        
        # Memory recommendations
        if "memory_health" in self.current_metrics:
            memory_usage = self.current_metrics["memory_health"].value
            if memory_usage > 80:
                recommendations.append("High memory usage detected. Consider restarting services or adding more memory.")
            elif memory_usage > 70:
                recommendations.append("Memory usage is elevated. Monitor closely and consider optimization.")
                
        # CPU recommendations
        if "system_resources" in self.current_metrics:
            cpu_metric = self.current_metrics["system_resources"]
            if hasattr(cpu_metric, 'tags') and 'cpu_percent' in cpu_metric.tags:
                cpu_usage = float(cpu_metric.tags['cpu_percent'])
                if cpu_usage > 90:
                    recommendations.append("High CPU usage detected. Consider scaling or optimizing workloads.")
                    
        # Alert-based recommendations
        critical_alerts = [a for a in self.alert_manager.active_alerts.values() 
                          if a.severity == AlertSeverity.CRITICAL]
        if critical_alerts:
            recommendations.append(f"Address {len(critical_alerts)} critical alerts immediately.")
            
        # Trend-based recommendations
        for metric_name, metric in self.current_metrics.items():
            if metric.trend == "increasing" and metric.get_status() == HealthStatus.WARNING:
                recommendations.append(f"Monitor {metric_name} closely - showing increasing trend.")
                
        return recommendations
        
    def enter_maintenance_mode(self, reason: str = "Scheduled maintenance"):
        """Enter maintenance mode."""
        self.maintenance_mode = True
        self.maintenance_start = datetime.now()
        
        self.alert_manager.create_alert(
            alert_type="maintenance_mode_started",
            severity=AlertSeverity.INFO,
            message=f"System entered maintenance mode: {reason}",
            component="system",
            metadata={"reason": reason}
        )
        
        self.error_handler.logger.info(f"Entered maintenance mode: {reason}")
        
    def exit_maintenance_mode(self):
        """Exit maintenance mode."""
        duration = None
        if self.maintenance_start:
            duration = (datetime.now() - self.maintenance_start).total_seconds()
            
        self.maintenance_mode = False
        self.maintenance_start = None
        
        self.alert_manager.create_alert(
            alert_type="maintenance_mode_ended",
            severity=AlertSeverity.INFO,
            message=f"System exited maintenance mode. Duration: {duration:.0f}s" if duration else "System exited maintenance mode",
            component="system",
            metadata={"duration_seconds": duration}
        )
        
        self.error_handler.logger.info("Exited maintenance mode")
        
    # Enhanced health check implementations
    def _check_system_resources(self) -> HealthMetric:
        """Comprehensive system resource check."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        return HealthMetric(
            name="system_resources",
            value=cpu_percent,
            unit="percent",
            threshold_warning=70.0,
            threshold_critical=90.0,
            tags={
                "cpu_percent": str(cpu_percent),
                "cpu_count": str(psutil.cpu_count()),
                "load_avg": str(psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0)
            }
        )
        
    def _check_memory_health(self) -> HealthMetric:
        """Enhanced memory health check."""
        memory = psutil.virtual_memory()
        
        return HealthMetric(
            name="memory_health",
            value=memory.percent,
            unit="percent",
            threshold_warning=75.0,
            threshold_critical=90.0,
            tags={
                "total_gb": str(round(memory.total / (1024**3), 2)),
                "available_gb": str(round(memory.available / (1024**3), 2)),
                "cached_gb": str(round(getattr(memory, 'cached', 0) / (1024**3), 2))
            }
        )
        
    def _check_disk_health(self) -> HealthMetric:
        """Disk health and I/O check."""
        disk_usage = psutil.disk_usage('/')
        usage_percent = (disk_usage.used / disk_usage.total) * 100
        
        # Get disk I/O stats
        disk_io = psutil.disk_io_counters()
        
        return HealthMetric(
            name="disk_health",
            value=usage_percent,
            unit="percent",
            threshold_warning=80.0,
            threshold_critical=95.0,
            tags={
                "total_gb": str(round(disk_usage.total / (1024**3), 2)),
                "free_gb": str(round(disk_usage.free / (1024**3), 2)),
                "read_count": str(disk_io.read_count if disk_io else 0),
                "write_count": str(disk_io.write_count if disk_io else 0)
            }
        )
        
    def _check_network_health(self) -> HealthMetric:
        """Network connectivity and I/O check."""
        network_io = psutil.net_io_counters()
        
        # Simple connectivity test
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            connectivity_score = 100.0
        except (socket.error, socket.timeout):
            connectivity_score = 0.0
            
        return HealthMetric(
            name="network_health",
            value=connectivity_score,
            unit="percent",
            threshold_warning=50.0,
            threshold_critical=0.0,
            tags={
                "bytes_sent": str(network_io.bytes_sent if network_io else 0),
                "bytes_recv": str(network_io.bytes_recv if network_io else 0),
                "packets_sent": str(network_io.packets_sent if network_io else 0),
                "packets_recv": str(network_io.packets_recv if network_io else 0)
            }
        )
        
    def _check_process_health(self) -> HealthMetric:
        """Current process health check."""
        process = psutil.Process()
        
        # Calculate process health score based on various factors
        health_score = 100.0
        
        # Memory usage factor
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        if memory_mb > 1024:  # > 1GB
            health_score -= 20.0
        elif memory_mb > 512:  # > 512MB
            health_score -= 10.0
            
        # Thread count factor
        num_threads = process.num_threads()
        if num_threads > 50:
            health_score -= 15.0
        elif num_threads > 20:
            health_score -= 5.0
            
        # Open files factor
        try:
            open_files = len(process.open_files())
            if open_files > 100:
                health_score -= 10.0
        except psutil.AccessDenied:
            open_files = 0
            
        health_score = max(0.0, health_score)
        
        return HealthMetric(
            name="process_health",
            value=health_score,
            unit="score",
            threshold_warning=70.0,
            threshold_critical=40.0,
            tags={
                "memory_mb": str(round(memory_mb, 2)),
                "threads": str(num_threads),
                "open_files": str(open_files),
                "cpu_percent": str(process.cpu_percent())
            }
        )
        
    def _check_gpu_health(self) -> Optional[HealthMetric]:
        """GPU health check if available."""
        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                return None
                
            gpu = gpus[0]  # Monitor first GPU
            
            # Calculate GPU health score
            health_score = 100.0
            
            # Temperature factor
            if gpu.temperature > 85:
                health_score -= 30.0
            elif gpu.temperature > 75:
                health_score -= 15.0
                
            # Memory utilization factor
            if gpu.memoryUtil > 0.9:
                health_score -= 20.0
            elif gpu.memoryUtil > 0.8:
                health_score -= 10.0
                
            # GPU utilization factor (high utilization is actually good for ML workloads)
            if gpu.load < 0.1:  # Very low utilization might indicate issues
                health_score -= 10.0
                
            health_score = max(0.0, health_score)
            
            return HealthMetric(
                name="gpu_health",
                value=health_score,
                unit="score",
                threshold_warning=70.0,
                threshold_critical=40.0,
                tags={
                    "gpu_name": gpu.name,
                    "utilization": str(round(gpu.load * 100, 1)),
                    "memory_util": str(round(gpu.memoryUtil * 100, 1)),
                    "temperature": str(gpu.temperature),
                    "memory_used_mb": str(gpu.memoryUsed),
                    "memory_total_mb": str(gpu.memoryTotal)
                }
            )
            
        except Exception as e:
            logger.debug(f"GPU health check failed: {e}")
            return None
            
    # Neuromorphic-specific health checks
    def _check_spike_rate(self) -> HealthMetric:
        """Check neural network spike rate."""
        # Simulate spike rate monitoring with realistic variation
        base_rate = 25.0
        variation = 5.0 * (0.5 - abs((time.time() % 20) / 20 - 0.5))
        spike_rate = base_rate + variation
        
        return HealthMetric(
            name="spike_rate",
            value=spike_rate,
            unit="Hz",
            threshold_warning=100.0,
            threshold_critical=200.0,
            tags={
                "network_layer": "kenyon_cells",
                "monitoring_window": "1s"
            }
        )
        
    def _check_network_sparsity(self) -> HealthMetric:
        """Check network sparsity level."""
        # Simulate sparsity with realistic drift
        target_sparsity = 5.0  # 5% for Kenyon cells
        drift = 1.0 * (time.time() % 30) / 30  # Slow drift over 30s
        current_sparsity = target_sparsity + drift
        
        return HealthMetric(
            name="network_sparsity",
            value=current_sparsity,
            unit="percent",
            threshold_warning=10.0,
            threshold_critical=20.0,
            tags={
                "target_sparsity": str(target_sparsity),
                "network_type": "mushroom_body"
            }
        )
        
    def _check_learning_stability(self) -> HealthMetric:
        """Check learning process stability."""
        # Simulate learning stability with periodic instability
        base_stability = 90.0
        instability = 20.0 * max(0, (time.time() % 60) - 55) / 5  # Unstable for 5s every minute
        stability_score = max(0, base_stability - instability)
        
        return HealthMetric(
            name="learning_stability",
            value=stability_score,
            unit="score",
            threshold_warning=70.0,
            threshold_critical=40.0,
            tags={
                "learning_algorithm": "stdp",
                "convergence_rate": "0.95"
            }
        )
        
    def _check_sensor_calibration(self) -> HealthMetric:
        """Check sensor calibration drift."""
        # Simulate calibration drift
        max_drift = 5.0
        current_drift = max_drift * (time.time() % 120) / 120  # Drift over 2 minutes
        
        return HealthMetric(
            name="sensor_calibration",
            value=current_drift,
            unit="percent_drift",
            threshold_warning=3.0,
            threshold_critical=7.0,
            tags={
                "sensor_type": "enose_array",
                "last_calibration": "2024-01-01T00:00:00Z",
                "drift_direction": "positive" if current_drift > 2.5 else "negative"
            }
        )
        
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get comprehensive system health report."""
        current_health = self.get_current_health()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "system_health": current_health.to_dict(),
            "performance_profile": {
                name: self.performance_profiler.get_operation_statistics(name)
                for name in self.performance_profiler.profiles.keys()
            },
            "alert_statistics": self.alert_manager.get_alert_statistics(timedelta(hours=24)),
            "metric_trends": {
                name: self.metric_collector.get_metric_statistics(name, timedelta(hours=1))
                for name in self.current_metrics.keys()
            },
            "system_configuration": {
                "check_interval": self.check_interval,
                "history_size": self.history_size,
                "predictions_enabled": self.enable_predictions,
                "gpu_monitoring_enabled": self.enable_gpu_monitoring,
                "maintenance_mode": self.maintenance_mode
            }
        }
        
        return report
        
    def get_current_health(self) -> SystemHealth:
        """Get current system health status."""
        return SystemHealth(
            overall_status=self._calculate_overall_status(self.component_status),
            components=dict(self.component_status),
            metrics=dict(self.current_metrics),
            alerts=list(self.alert_manager.active_alerts.values()),
            system_load=psutil.cpu_percent(interval=0.1),
            uptime_seconds=time.time() - psutil.boot_time() if psutil else 0,
            recommendations=self._generate_recommendations(),
            performance_score=self._calculate_performance_score()
        )


# Global enhanced health monitor instance
_health_monitor = None
_health_monitor_lock = threading.Lock()


def get_health_monitor() -> EnhancedHealthMonitor:
    """Get global enhanced health monitor instance."""
    global _health_monitor
    
    if _health_monitor is None:
        with _health_monitor_lock:
            if _health_monitor is None:
                _health_monitor = EnhancedHealthMonitor()
                
    return _health_monitor


def configure_health_monitor(
    check_interval: float = 30.0,
    history_size: int = 1000,
    alert_callback: Optional[Callable] = None,
    enable_predictions: bool = True,
    enable_gpu_monitoring: bool = True
) -> EnhancedHealthMonitor:
    """Configure global health monitor with specific settings."""
    global _health_monitor
    
    with _health_monitor_lock:
        _health_monitor = EnhancedHealthMonitor(
            check_interval=check_interval,
            history_size=history_size,
            alert_callback=alert_callback,
            enable_predictions=enable_predictions,
            enable_gpu_monitoring=enable_gpu_monitoring
        )
        
    return _health_monitor


# Health check decorator with performance profiling
def health_check(metric_name: str, profile_operation: bool = True):
    """Decorator to automatically register function as health check with profiling."""
    def decorator(func):
        monitor = get_health_monitor()
        
        if profile_operation:
            @wraps(func)
            def profiled_wrapper(*args, **kwargs):
                monitor.performance_profiler.start_profiling(f"health_check_{metric_name}")
                try:
                    return func(*args, **kwargs)
                finally:
                    monitor.performance_profiler.end_profiling(f"health_check_{metric_name}")
            
            monitor.register_health_check(metric_name, profiled_wrapper)
            return profiled_wrapper
        else:
            monitor.register_health_check(metric_name, func)
            return func
            
    return decorator


# Initialize on import
get_health_monitor()