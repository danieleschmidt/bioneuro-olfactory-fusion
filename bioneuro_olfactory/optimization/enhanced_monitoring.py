"""
Enhanced performance monitoring with real-time metrics.

This module provides comprehensive real-time monitoring capabilities for
the BioNeuro-Olfactory-Fusion system with advanced analytics, alerting,
and visualization support.
"""

import asyncio
import time
import threading
import logging
import json
import queue
import numpy as np
import psutil
import torch
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
from collections import deque, defaultdict
from contextlib import asynccontextmanager, contextmanager
import pickle
import gzip
import statistics
from concurrent.futures import ThreadPoolExecutor
import websockets
import aiohttp

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"           # Monotonically increasing
    GAUGE = "gauge"              # Can go up and down
    HISTOGRAM = "histogram"       # Distribution of values
    SUMMARY = "summary"          # Quantiles over time windows
    RATE = "rate"               # Rate of change


@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: float
    value: Union[int, float, Dict[str, Any]]
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp,
            'value': self.value,
            'labels': self.labels
        }


@dataclass
class Alert:
    """Performance alert."""
    alert_id: str
    severity: AlertSeverity
    message: str
    metric_name: str
    threshold_value: float
    current_value: float
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    resolved_timestamp: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class MonitoringConfig:
    """Configuration for monitoring system."""
    collection_interval_seconds: float = 1.0
    retention_hours: int = 24
    enable_alerting: bool = True
    enable_websocket_streaming: bool = True
    enable_metric_aggregation: bool = True
    enable_anomaly_detection: bool = True
    websocket_port: int = 8765
    max_metric_history: int = 10000
    alert_cooldown_seconds: float = 300.0  # 5 minutes
    enable_compression: bool = True
    enable_persistence: bool = True
    persistence_interval_seconds: float = 60.0


class MetricCollector:
    """Collects and stores metrics with configurable retention."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.metrics: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=config.max_metric_history)
        )
        self.metric_types: Dict[str, MetricType] = {}
        self.metric_metadata: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        
        # Aggregated metrics
        self.aggregated_metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.last_aggregation = time.time()
        self.aggregation_interval = 60.0  # 1 minute
        
    def record_metric(self, name: str, value: Union[int, float], 
                     labels: Dict[str, str] = None, 
                     metric_type: MetricType = MetricType.GAUGE):
        """Record a metric value."""
        timestamp = time.time()
        labels = labels or {}
        
        point = MetricPoint(timestamp=timestamp, value=value, labels=labels)
        
        with self._lock:
            self.metrics[name].append(point)
            self.metric_types[name] = metric_type
            
            # Update metadata
            if name not in self.metric_metadata:
                self.metric_metadata[name] = {
                    'first_seen': timestamp,
                    'type': metric_type.value,
                    'description': '',
                    'unit': '',
                    'label_keys': set()
                }
                
            self.metric_metadata[name]['last_seen'] = timestamp
            self.metric_metadata[name]['label_keys'].update(labels.keys())
            
        # Trigger aggregation if needed
        if (self.config.enable_metric_aggregation and 
            time.time() - self.last_aggregation > self.aggregation_interval):
            self._aggregate_metrics()
            
    def record_histogram(self, name: str, value: float, buckets: List[float] = None,
                        labels: Dict[str, str] = None):
        """Record a histogram metric."""
        if buckets is None:
            buckets = [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0]
            
        # Find appropriate bucket
        bucket_counts = {str(bucket): 0 for bucket in buckets}
        bucket_counts['+Inf'] = 0
        
        for bucket in buckets:
            if value <= bucket:
                bucket_counts[str(bucket)] += 1
        bucket_counts['+Inf'] += 1
        
        histogram_data = {
            'buckets': bucket_counts,
            'count': 1,
            'sum': value
        }
        
        self.record_metric(name, histogram_data, labels, MetricType.HISTOGRAM)
        
    def get_metric_values(self, name: str, start_time: Optional[float] = None, 
                         end_time: Optional[float] = None) -> List[MetricPoint]:
        """Get metric values within time range."""
        with self._lock:
            points = list(self.metrics.get(name, []))
            
        if start_time is None and end_time is None:
            return points
            
        filtered_points = []
        for point in points:
            if start_time and point.timestamp < start_time:
                continue
            if end_time and point.timestamp > end_time:
                continue
            filtered_points.append(point)
            
        return filtered_points
        
    def get_latest_value(self, name: str) -> Optional[MetricPoint]:
        """Get most recent value for metric."""
        with self._lock:
            points = self.metrics.get(name, [])
            return points[-1] if points else None
            
    def get_metric_summary(self, name: str, window_seconds: float = 300.0) -> Dict[str, Any]:
        """Get statistical summary of metric over time window."""
        end_time = time.time()
        start_time = end_time - window_seconds
        
        points = self.get_metric_values(name, start_time, end_time)
        
        if not points:
            return {'count': 0}
            
        values = [p.value for p in points if isinstance(p.value, (int, float))]
        
        if not values:
            return {'count': len(points)}
            
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0,
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99)
        }
        
    def _aggregate_metrics(self):
        """Aggregate metrics for efficient querying."""
        current_time = time.time()
        
        with self._lock:
            for metric_name, points in self.metrics.items():
                if not points:
                    continue
                    
                # Aggregate last minute of data
                minute_start = current_time - 60
                recent_points = [p for p in points if p.timestamp >= minute_start]
                
                if recent_points:
                    values = [p.value for p in recent_points 
                             if isinstance(p.value, (int, float))]
                    
                    if values:
                        self.aggregated_metrics[metric_name] = {
                            'timestamp': current_time,
                            'count': len(values),
                            'sum': sum(values),
                            'min': min(values),
                            'max': max(values),
                            'avg': statistics.mean(values)
                        }
                        
        self.last_aggregation = current_time
        
    def cleanup_old_metrics(self):
        """Remove metrics older than retention period."""
        cutoff_time = time.time() - (self.config.retention_hours * 3600)
        
        with self._lock:
            for metric_name in list(self.metrics.keys()):
                points = self.metrics[metric_name]
                
                # Remove old points
                while points and points[0].timestamp < cutoff_time:
                    points.popleft()
                    
                # Remove empty metrics
                if not points:
                    del self.metrics[metric_name]
                    if metric_name in self.metric_types:
                        del self.metric_types[metric_name]
                    if metric_name in self.metric_metadata:
                        del self.metric_metadata[metric_name]
                        
    def get_all_metric_names(self) -> List[str]:
        """Get list of all metric names."""
        with self._lock:
            return list(self.metrics.keys())
            
    def export_metrics(self, format: str = 'json') -> str:
        """Export metrics in specified format."""
        with self._lock:
            data = {
                'metrics': {
                    name: [point.to_dict() for point in points]
                    for name, points in self.metrics.items()
                },
                'metadata': self.metric_metadata,
                'aggregated': self.aggregated_metrics,
                'exported_at': time.time()
            }
            
        if format == 'json':
            return json.dumps(data, indent=2)
        elif format == 'compressed':
            json_data = json.dumps(data)
            return gzip.compress(json_data.encode()).hex()
        else:
            raise ValueError(f"Unsupported export format: {format}")


class AlertManager:
    """Manages performance alerts and notifications."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        self.last_alert_times: Dict[str, float] = {}
        self._lock = threading.Lock()
        
    def add_alert_rule(self, metric_name: str, threshold: float, 
                      condition: str = 'greater_than', severity: AlertSeverity = AlertSeverity.MEDIUM,
                      message_template: str = None):
        """Add an alert rule for a metric."""
        rule_id = f"{metric_name}_{condition}_{threshold}"
        
        message_template = message_template or f"{metric_name} {condition} {threshold}"
        
        self.alert_rules[rule_id] = {
            'metric_name': metric_name,
            'threshold': threshold,
            'condition': condition,
            'severity': severity,
            'message_template': message_template,
            'enabled': True
        }
        
        logger.info(f"Added alert rule: {rule_id}")
        
    def remove_alert_rule(self, rule_id: str) -> bool:
        """Remove an alert rule."""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            logger.info(f"Removed alert rule: {rule_id}")
            return True
        return False
        
    def check_alerts(self, metric_collector: MetricCollector):
        """Check all alert rules against current metrics."""
        current_time = time.time()
        
        for rule_id, rule in self.alert_rules.items():
            if not rule['enabled']:
                continue
                
            # Check cooldown period
            last_alert_time = self.last_alert_times.get(rule_id, 0)
            if current_time - last_alert_time < self.config.alert_cooldown_seconds:
                continue
                
            metric_name = rule['metric_name']
            latest_point = metric_collector.get_latest_value(metric_name)
            
            if latest_point is None or not isinstance(latest_point.value, (int, float)):
                continue
                
            current_value = latest_point.value
            threshold = rule['threshold']
            condition = rule['condition']
            
            # Evaluate condition
            triggered = False
            if condition == 'greater_than' and current_value > threshold:
                triggered = True
            elif condition == 'less_than' and current_value < threshold:
                triggered = True
            elif condition == 'equal' and abs(current_value - threshold) < 1e-6:
                triggered = True
            elif condition == 'not_equal' and abs(current_value - threshold) >= 1e-6:
                triggered = True
                
            if triggered:
                self._trigger_alert(rule_id, rule, current_value, current_time)
            elif rule_id in self.active_alerts:
                self._resolve_alert(rule_id, current_time)
                
    def _trigger_alert(self, rule_id: str, rule: Dict[str, Any], 
                      current_value: float, timestamp: float):
        """Trigger an alert."""
        alert = Alert(
            alert_id=f"{rule_id}_{int(timestamp)}",
            severity=rule['severity'],
            message=rule['message_template'].format(
                metric=rule['metric_name'],
                value=current_value,
                threshold=rule['threshold']
            ),
            metric_name=rule['metric_name'],
            threshold_value=rule['threshold'],
            current_value=current_value,
            timestamp=timestamp
        )
        
        with self._lock:
            self.active_alerts[rule_id] = alert
            self.alert_history.append(alert)
            self.last_alert_times[rule_id] = timestamp
            
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
                
        logger.warning(f"ALERT [{alert.severity.value.upper()}]: {alert.message}")
        
    def _resolve_alert(self, rule_id: str, timestamp: float):
        """Resolve an active alert."""
        with self._lock:
            if rule_id in self.active_alerts:
                alert = self.active_alerts[rule_id]
                alert.resolved = True
                alert.resolved_timestamp = timestamp
                del self.active_alerts[rule_id]
                
                logger.info(f"RESOLVED: Alert {alert.alert_id}")
                
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add callback function for alert notifications."""
        self.alert_callbacks.append(callback)
        
    def get_active_alerts(self) -> List[Alert]:
        """Get list of active alerts."""
        with self._lock:
            return list(self.active_alerts.values())
            
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history."""
        with self._lock:
            return list(self.alert_history)[-limit:]


class AnomalyDetector:
    """Detects anomalies in metric data using statistical methods."""
    
    def __init__(self, sensitivity: float = 2.0):
        self.sensitivity = sensitivity  # Number of standard deviations
        self.baseline_windows: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)  # Keep 1000 points for baseline
        )
        self.anomaly_scores: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )
        
    def add_baseline_point(self, metric_name: str, value: float):
        """Add point to baseline for anomaly detection."""
        self.baseline_windows[metric_name].append(value)
        
    def detect_anomaly(self, metric_name: str, value: float) -> Dict[str, Any]:
        """Detect if value is anomalous compared to baseline."""
        baseline = self.baseline_windows[metric_name]
        
        if len(baseline) < 10:
            # Need more baseline data
            self.add_baseline_point(metric_name, value)
            return {'is_anomaly': False, 'confidence': 0.0}
            
        baseline_values = list(baseline)
        mean = statistics.mean(baseline_values)
        std = statistics.stdev(baseline_values)
        
        if std == 0:
            # No variance in baseline
            anomaly_score = 0.0 if value == mean else 1.0
        else:
            # Z-score based detection
            z_score = abs(value - mean) / std
            anomaly_score = max(0, (z_score - self.sensitivity) / self.sensitivity)
            
        is_anomaly = anomaly_score > 0.5
        
        # Update scores
        self.anomaly_scores[metric_name].append(anomaly_score)
        
        # Add to baseline if not anomalous (for adaptive baseline)
        if not is_anomaly:
            self.add_baseline_point(metric_name, value)
            
        return {
            'is_anomaly': is_anomaly,
            'anomaly_score': anomaly_score,
            'z_score': z_score if std > 0 else 0,
            'baseline_mean': mean,
            'baseline_std': std,
            'confidence': min(1.0, anomaly_score)
        }
        
    def get_anomaly_summary(self, metric_name: str) -> Dict[str, Any]:
        """Get anomaly detection summary for metric."""
        scores = list(self.anomaly_scores[metric_name])
        baseline = list(self.baseline_windows[metric_name])
        
        if not scores:
            return {'anomaly_rate': 0.0, 'avg_score': 0.0}
            
        anomaly_count = sum(1 for score in scores if score > 0.5)
        
        return {
            'anomaly_rate': anomaly_count / len(scores),
            'avg_anomaly_score': statistics.mean(scores),
            'max_anomaly_score': max(scores),
            'baseline_points': len(baseline),
            'recent_anomalies': anomaly_count
        }


class RealTimeMonitor:
    """Real-time monitoring system with WebSocket streaming."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.metric_collector = MetricCollector(config)
        self.alert_manager = AlertManager(config) if config.enable_alerting else None
        self.anomaly_detector = AnomalyDetector() if config.enable_anomaly_detection else None
        
        # System monitoring
        self.system_monitor = SystemMonitor()
        self.neuromorphic_monitor = NeuromorphicMonitor()
        
        # WebSocket connections
        self.websocket_clients: set = set()
        self.websocket_server = None
        
        # Background tasks
        self._monitoring = False
        self._monitor_thread = None
        self._cleanup_thread = None
        self._persistence_thread = None
        
        # Performance tracking
        self.start_time = time.time()
        self.metrics_collected = 0
        self.alerts_triggered = 0
        self.anomalies_detected = 0
        
    async def start_monitoring(self):
        """Start the monitoring system."""
        if self._monitoring:
            return
            
        self._monitoring = True
        
        # Start background threads
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        
        if self.config.enable_persistence:
            self._persistence_thread = threading.Thread(target=self._persistence_loop, daemon=True)
            self._persistence_thread.start()
            
        # Start WebSocket server
        if self.config.enable_websocket_streaming:
            await self._start_websocket_server()
            
        # Initialize alert rules
        if self.alert_manager:
            self._setup_default_alerts()
            
        logger.info("Real-time monitoring system started")
        
    async def stop_monitoring(self):
        """Stop the monitoring system."""
        self._monitoring = False
        
        # Stop WebSocket server
        if self.websocket_server:
            self.websocket_server.close()
            await self.websocket_server.wait_closed()
            
        # Wait for threads to finish
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5.0)
        if self._persistence_thread:
            self._persistence_thread.join(timeout=5.0)
            
        logger.info("Real-time monitoring system stopped")
        
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._monitoring:
            try:
                start_time = time.perf_counter()
                
                # Collect system metrics
                system_metrics = self.system_monitor.collect_metrics()
                for name, value in system_metrics.items():
                    self.metric_collector.record_metric(f"system.{name}", value)
                    
                # Collect neuromorphic metrics  
                neuro_metrics = self.neuromorphic_monitor.collect_metrics()
                for name, value in neuro_metrics.items():
                    self.metric_collector.record_metric(f"neuromorphic.{name}", value)
                    
                # Update statistics
                self.metrics_collected += len(system_metrics) + len(neuro_metrics)
                
                # Check for anomalies
                if self.anomaly_detector:
                    for name, value in {**system_metrics, **neuro_metrics}.items():
                        if isinstance(value, (int, float)):
                            anomaly_result = self.anomaly_detector.detect_anomaly(name, value)
                            if anomaly_result['is_anomaly']:
                                self.anomalies_detected += 1
                                logger.warning(f"Anomaly detected in {name}: {value} (score: {anomaly_result['anomaly_score']:.2f})")
                                
                # Check alerts
                if self.alert_manager:
                    self.alert_manager.check_alerts(self.metric_collector)
                    
                # Stream to WebSocket clients
                if self.websocket_clients:
                    asyncio.create_task(self._stream_to_websockets(system_metrics, neuro_metrics))
                    
                # Control loop timing
                elapsed = time.perf_counter() - start_time
                sleep_time = max(0, self.config.collection_interval_seconds - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1.0)
                
    def _cleanup_loop(self):
        """Background cleanup loop."""
        while self._monitoring:
            try:
                self.metric_collector.cleanup_old_metrics()
                time.sleep(3600)  # Cleanup every hour
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                time.sleep(300)  # Back off on error
                
    def _persistence_loop(self):
        """Background persistence loop."""
        while self._monitoring:
            try:
                # Export metrics to file
                timestamp = int(time.time())
                filename = f"metrics_{timestamp}.json"
                
                metrics_data = self.metric_collector.export_metrics('json')
                with open(f"/tmp/{filename}", 'w') as f:
                    f.write(metrics_data)
                    
                time.sleep(self.config.persistence_interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in persistence loop: {e}")
                time.sleep(300)
                
    async def _start_websocket_server(self):
        """Start WebSocket server for real-time streaming."""
        try:
            self.websocket_server = await websockets.serve(
                self._handle_websocket_connection,
                "localhost",
                self.config.websocket_port
            )
            logger.info(f"WebSocket server started on port {self.config.websocket_port}")
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
            
    async def _handle_websocket_connection(self, websocket, path):
        """Handle WebSocket client connections."""
        self.websocket_clients.add(websocket)
        logger.info(f"WebSocket client connected: {websocket.remote_address}")
        
        try:
            # Send initial data
            initial_data = {
                'type': 'initial',
                'metrics': {
                    name: [point.to_dict() for point in points[-100:]]  # Last 100 points
                    for name, points in self.metric_collector.metrics.items()
                },
                'alert_rules': self.alert_manager.alert_rules if self.alert_manager else {},
                'active_alerts': [alert.to_dict() for alert in self.alert_manager.get_active_alerts()] if self.alert_manager else []
            }
            
            await websocket.send(json.dumps(initial_data))
            
            # Keep connection alive
            await websocket.wait_closed()
            
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            self.websocket_clients.discard(websocket)
            logger.info(f"WebSocket client disconnected: {websocket.remote_address}")
            
    async def _stream_to_websockets(self, system_metrics: Dict[str, Any], 
                                   neuro_metrics: Dict[str, Any]):
        """Stream metrics to WebSocket clients."""
        if not self.websocket_clients:
            return
            
        data = {
            'type': 'metrics_update',
            'timestamp': time.time(),
            'system_metrics': system_metrics,
            'neuromorphic_metrics': neuro_metrics
        }
        
        # Add active alerts if any
        if self.alert_manager:
            active_alerts = self.alert_manager.get_active_alerts()
            if active_alerts:
                data['active_alerts'] = [alert.to_dict() for alert in active_alerts]
                
        message = json.dumps(data)
        
        # Send to all connected clients
        disconnected_clients = set()
        for websocket in self.websocket_clients:
            try:
                await websocket.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(websocket)
            except Exception as e:
                logger.error(f"Failed to send to WebSocket client: {e}")
                disconnected_clients.add(websocket)
                
        # Remove disconnected clients
        self.websocket_clients -= disconnected_clients
        
    def _setup_default_alerts(self):
        """Setup default alert rules."""
        if not self.alert_manager:
            return
            
        # System alerts
        self.alert_manager.add_alert_rule(
            "system.cpu_percent", 90.0, "greater_than", AlertSeverity.HIGH,
            "High CPU usage: {value:.1f}% > {threshold}%"
        )
        
        self.alert_manager.add_alert_rule(
            "system.memory_percent", 85.0, "greater_than", AlertSeverity.MEDIUM,
            "High memory usage: {value:.1f}% > {threshold}%"
        )
        
        self.alert_manager.add_alert_rule(
            "system.disk_percent", 90.0, "greater_than", AlertSeverity.MEDIUM,
            "High disk usage: {value:.1f}% > {threshold}%"
        )
        
        # Neuromorphic alerts
        self.alert_manager.add_alert_rule(
            "neuromorphic.temperature_celsius", 80.0, "greater_than", AlertSeverity.CRITICAL,
            "High neuromorphic temperature: {value:.1f}°C > {threshold}°C"
        )
        
        self.alert_manager.add_alert_rule(
            "neuromorphic.power_watts", 15.0, "greater_than", AlertSeverity.HIGH,
            "High neuromorphic power consumption: {value:.1f}W > {threshold}W"
        )
        
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring system statistics."""
        uptime = time.time() - self.start_time
        
        stats = {
            'uptime_seconds': uptime,
            'metrics_collected': self.metrics_collected,
            'alerts_triggered': self.alerts_triggered,
            'anomalies_detected': self.anomalies_detected,
            'websocket_clients': len(self.websocket_clients),
            'metrics_per_second': self.metrics_collected / max(uptime, 1),
            'total_metric_series': len(self.metric_collector.get_all_metric_names()),
            'memory_usage_mb': psutil.Process().memory_info().rss / (1024 * 1024)
        }
        
        if self.alert_manager:
            stats['active_alerts'] = len(self.alert_manager.get_active_alerts())
            stats['alert_rules'] = len(self.alert_manager.alert_rules)
            
        return stats


class SystemMonitor:
    """Monitors system-level metrics."""
    
    def __init__(self):
        self.last_network_stats = None
        self.last_disk_stats = None
        self.last_cpu_times = None
        
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect system metrics."""
        metrics = {}
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_times = psutil.cpu_times()
        metrics['cpu_percent'] = cpu_percent
        metrics['cpu_count'] = psutil.cpu_count()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        metrics['memory_percent'] = memory.percent
        metrics['memory_available_gb'] = memory.available / (1024**3)
        metrics['memory_used_gb'] = memory.used / (1024**3)
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        metrics['disk_percent'] = (disk.used / disk.total) * 100
        metrics['disk_free_gb'] = disk.free / (1024**3)
        
        # Network metrics (rates)
        network = psutil.net_io_counters()
        if self.last_network_stats:
            time_delta = 1.0  # Assume 1 second between calls
            bytes_sent_rate = (network.bytes_sent - self.last_network_stats.bytes_sent) / time_delta
            bytes_recv_rate = (network.bytes_recv - self.last_network_stats.bytes_recv) / time_delta
            
            metrics['network_bytes_sent_per_sec'] = bytes_sent_rate
            metrics['network_bytes_recv_per_sec'] = bytes_recv_rate
            
        self.last_network_stats = network
        
        # Process count
        metrics['process_count'] = len(psutil.pids())
        
        # Load average (Unix only)
        try:
            load_avg = psutil.getloadavg()
            metrics['load_average_1m'] = load_avg[0]
            metrics['load_average_5m'] = load_avg[1]
            metrics['load_average_15m'] = load_avg[2]
        except AttributeError:
            pass  # Windows doesn't have load average
            
        return metrics


class NeuromorphicMonitor:
    """Monitors neuromorphic hardware metrics."""
    
    def __init__(self):
        self.simulated_temp = 25.0  # Base temperature
        self.simulated_power = 2.0  # Base power consumption
        
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect neuromorphic hardware metrics."""
        # In a real implementation, these would come from actual hardware
        
        # Simulate temperature variations
        temp_variation = np.random.normal(0, 2)  # ±2°C variation
        self.simulated_temp = max(20, min(100, self.simulated_temp + temp_variation * 0.1))
        
        # Simulate power variations
        power_variation = np.random.normal(0, 0.5)  # ±0.5W variation  
        self.simulated_power = max(0.5, min(20, self.simulated_power + power_variation * 0.1))
        
        metrics = {
            'temperature_celsius': self.simulated_temp,
            'power_watts': self.simulated_power,
            'spike_rate_hz': np.random.exponential(100),  # Exponential distribution
            'inference_latency_ms': np.random.gamma(2, 5),  # Gamma distribution
            'accuracy_score': 0.9 + np.random.normal(0, 0.02),  # Around 90% ± 2%
            'energy_efficiency': self.simulated_power / max(metrics.get('spike_rate_hz', 1), 1),
            'utilization_percent': np.random.uniform(30, 95)
        }
        
        # Ensure accuracy is between 0 and 1
        metrics['accuracy_score'] = max(0, min(1, metrics['accuracy_score']))
        
        return metrics


# Global monitoring instance
_global_monitor: Optional[RealTimeMonitor] = None


async def get_monitor(config: MonitoringConfig = None) -> RealTimeMonitor:
    """Get global monitoring instance."""
    global _global_monitor
    
    if _global_monitor is None:
        config = config or MonitoringConfig()
        _global_monitor = RealTimeMonitor(config)
        await _global_monitor.start_monitoring()
        
    return _global_monitor


@contextmanager
def monitoring_context(config: MonitoringConfig = None):
    """Context manager for monitoring lifecycle."""
    import asyncio
    
    monitor = RealTimeMonitor(config or MonitoringConfig())
    
    try:
        # Start monitoring
        loop = asyncio.get_event_loop()
        loop.run_until_complete(monitor.start_monitoring())
        
        yield monitor
        
    finally:
        # Clean up
        loop = asyncio.get_event_loop()
        loop.run_until_complete(monitor.stop_monitoring())


def record_metric(name: str, value: Union[int, float], labels: Dict[str, str] = None):
    """Convenience function to record a metric."""
    global _global_monitor
    
    if _global_monitor:
        _global_monitor.metric_collector.record_metric(name, value, labels)
    else:
        logger.warning("No global monitor instance available")


# Decorator for method performance monitoring
def monitor_performance(metric_name: str = None, labels: Dict[str, str] = None):
    """Decorator to monitor method performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            name = metric_name or f"{func.__module__}.{func.__name__}"
            start_time = time.perf_counter()
            
            try:
                result = func(*args, **kwargs)
                execution_time = (time.perf_counter() - start_time) * 1000  # ms
                record_metric(f"{name}.execution_time_ms", execution_time, labels)
                record_metric(f"{name}.success_count", 1, labels)
                return result
            except Exception as e:
                execution_time = (time.perf_counter() - start_time) * 1000  # ms
                record_metric(f"{name}.execution_time_ms", execution_time, labels)
                record_metric(f"{name}.error_count", 1, labels)
                raise
                
        return wrapper
    return decorator