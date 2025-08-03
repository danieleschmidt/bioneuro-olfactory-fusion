"""
Neuromorphic-specific metrics collection for gas detection system.
Monitors spiking neural network performance, detection accuracy, and system health.
"""

import time
import logging
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import numpy as np
import torch
from prometheus_client import Counter, Histogram, Gauge, Summary
import psutil
import threading

from ..core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class NeuromorphicMetrics:
    """Metrics specific to neuromorphic gas detection system."""
    spike_rate: float
    membrane_potential_variance: float
    kenyon_cell_sparsity: float
    detection_latency_ms: float
    confidence_score: float
    gas_concentration: float
    sensor_drift: float
    false_positive_rate: float
    detection_accuracy: float
    energy_consumption_watts: float


@dataclass
class SystemHealthMetrics:
    """System health and performance metrics."""
    cpu_usage_percent: float
    memory_usage_percent: float
    gpu_usage_percent: Optional[float]
    gpu_memory_percent: Optional[float]
    disk_usage_percent: float
    network_io_bytes: int
    temperature_celsius: float
    uptime_seconds: float


# Prometheus metrics
DETECTION_COUNTER = Counter('gas_detections_total', 'Total gas detections', ['gas_type', 'severity'])
DETECTION_LATENCY = Histogram('detection_latency_seconds', 'Detection latency in seconds')
SPIKE_RATE_GAUGE = Gauge('neural_spike_rate_hz', 'Neural spike rate in Hz')
MEMBRANE_POTENTIAL = Summary('membrane_potential_variance', 'Membrane potential variance')
KENYON_SPARSITY = Gauge('kenyon_cell_sparsity_ratio', 'Kenyon cell sparsity ratio')
CONFIDENCE_SCORE = Histogram('detection_confidence', 'Detection confidence score')
SENSOR_DRIFT = Gauge('sensor_drift_percentage', 'Sensor drift percentage')
FALSE_POSITIVE_RATE = Gauge('false_positive_rate', 'False positive rate')
SYSTEM_CPU = Gauge('system_cpu_usage_percent', 'CPU usage percentage')
SYSTEM_MEMORY = Gauge('system_memory_usage_percent', 'Memory usage percentage')
SYSTEM_TEMPERATURE = Gauge('system_temperature_celsius', 'System temperature')


class MetricsCollector:
    """Comprehensive metrics collection for neuromorphic gas detection."""
    
    def __init__(self):
        self.running = False
        self.collection_interval = settings.METRICS_COLLECTION_INTERVAL
        self._metrics_history: List[Dict[str, Any]] = []
        self._max_history_size = 1000
        self._collection_task: Optional[asyncio.Task] = None
        
    async def start_collection(self):
        """Start continuous metrics collection."""
        if self.running:
            logger.warning("Metrics collection already running")
            return
            
        self.running = True
        self._collection_task = asyncio.create_task(self._collection_loop())
        logger.info("Started metrics collection")
        
    async def stop_collection(self):
        """Stop metrics collection."""
        self.running = False
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped metrics collection")
        
    async def _collection_loop(self):
        """Main collection loop."""
        while self.running:
            try:
                await self._collect_all_metrics()
                await asyncio.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(5)  # Back off on error
                
    async def _collect_all_metrics(self):
        """Collect all system and neuromorphic metrics."""
        timestamp = datetime.utcnow()
        
        # Collect system health
        system_metrics = self._collect_system_health()
        
        # Update Prometheus gauges
        SYSTEM_CPU.set(system_metrics.cpu_usage_percent)
        SYSTEM_MEMORY.set(system_metrics.memory_usage_percent)
        SYSTEM_TEMPERATURE.set(system_metrics.temperature_celsius)
        
        # Store in history
        metrics_snapshot = {
            'timestamp': timestamp.isoformat(),
            'system': asdict(system_metrics),
        }
        
        self._add_to_history(metrics_snapshot)
        
    def _collect_system_health(self) -> SystemHealthMetrics:
        """Collect system health metrics."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = disk.used / disk.total * 100
        
        # Network I/O
        net_io = psutil.net_io_counters()
        network_bytes = net_io.bytes_sent + net_io.bytes_recv
        
        # Temperature (if available)
        temperature = 0.0
        try:
            temps = psutil.sensors_temperatures()
            if 'coretemp' in temps:
                temperature = temps['coretemp'][0].current
        except (AttributeError, KeyError, IndexError):
            pass
            
        # GPU metrics (if CUDA available)
        gpu_usage = None
        gpu_memory = None
        if torch.cuda.is_available():
            try:
                gpu_usage = torch.cuda.utilization()
                gpu_memory = torch.cuda.memory_usage() * 100
            except:
                pass
                
        # System uptime
        uptime = time.time() - psutil.boot_time()
        
        return SystemHealthMetrics(
            cpu_usage_percent=cpu_percent,
            memory_usage_percent=memory.percent,
            gpu_usage_percent=gpu_usage,
            gpu_memory_percent=gpu_memory,
            disk_usage_percent=disk_percent,
            network_io_bytes=network_bytes,
            temperature_celsius=temperature,
            uptime_seconds=uptime
        )
        
    def record_detection(self, gas_type: str, concentration: float, 
                        confidence: float, latency: float, severity: str = 'medium'):
        """Record a gas detection event."""
        DETECTION_COUNTER.labels(gas_type=gas_type, severity=severity).inc()
        DETECTION_LATENCY.observe(latency)
        CONFIDENCE_SCORE.observe(confidence)
        
        logger.info(f"Recorded detection: {gas_type} at {concentration}ppm "
                   f"(confidence: {confidence:.3f}, latency: {latency:.3f}s)")
        
    def record_neuromorphic_metrics(self, metrics: NeuromorphicMetrics):
        """Record neuromorphic-specific metrics."""
        SPIKE_RATE_GAUGE.set(metrics.spike_rate)
        MEMBRANE_POTENTIAL.observe(metrics.membrane_potential_variance)
        KENYON_SPARSITY.set(metrics.kenyon_cell_sparsity)
        SENSOR_DRIFT.set(metrics.sensor_drift)
        FALSE_POSITIVE_RATE.set(metrics.false_positive_rate)
        
    def _add_to_history(self, metrics: Dict[str, Any]):
        """Add metrics to history with size management."""
        self._metrics_history.append(metrics)
        if len(self._metrics_history) > self._max_history_size:
            self._metrics_history.pop(0)
            
    def get_metrics_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get metrics history for specified hours."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        return [
            metrics for metrics in self._metrics_history
            if datetime.fromisoformat(metrics['timestamp']) > cutoff_time
        ]
        
    def get_health_status(self) -> Dict[str, Any]:
        """Get current system health status."""
        if not self._metrics_history:
            return {'status': 'unknown', 'message': 'No metrics collected yet'}
            
        latest = self._metrics_history[-1]['system']
        
        # Health assessment
        issues = []
        if latest['cpu_usage_percent'] > 80:
            issues.append('High CPU usage')
        if latest['memory_usage_percent'] > 85:
            issues.append('High memory usage')
        if latest['temperature_celsius'] > 70:
            issues.append('High temperature')
        if latest['disk_usage_percent'] > 90:
            issues.append('Low disk space')
            
        if not issues:
            status = 'healthy'
            message = 'All systems operating normally'
        elif len(issues) == 1:
            status = 'warning'
            message = f"Warning: {issues[0]}"
        else:
            status = 'critical'
            message = f"Critical issues: {', '.join(issues)}"
            
        return {
            'status': status,
            'message': message,
            'issues': issues,
            'metrics': latest
        }


class PerformanceProfiler:
    """Performance profiling for neuromorphic algorithms."""
    
    def __init__(self):
        self._profiles: Dict[str, List[float]] = {}
        
    def profile_function(self, func_name: str):
        """Decorator for profiling function execution time."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    end_time = time.perf_counter()
                    execution_time = end_time - start_time
                    self.record_execution_time(func_name, execution_time)
            return wrapper
        return decorator
        
    def record_execution_time(self, operation: str, execution_time: float):
        """Record execution time for an operation."""
        if operation not in self._profiles:
            self._profiles[operation] = []
            
        self._profiles[operation].append(execution_time)
        
        # Keep only last 100 measurements
        if len(self._profiles[operation]) > 100:
            self._profiles[operation].pop(0)
            
    def get_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """Get performance summary for all profiled operations."""
        summary = {}
        
        for operation, times in self._profiles.items():
            if times:
                summary[operation] = {
                    'mean_ms': np.mean(times) * 1000,
                    'std_ms': np.std(times) * 1000,
                    'min_ms': np.min(times) * 1000,
                    'max_ms': np.max(times) * 1000,
                    'p95_ms': np.percentile(times, 95) * 1000,
                    'p99_ms': np.percentile(times, 99) * 1000,
                    'count': len(times)
                }
                
        return summary


# Global instances
metrics_collector = MetricsCollector()
performance_profiler = PerformanceProfiler()