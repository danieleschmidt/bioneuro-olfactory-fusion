"""Performance profiling and optimization system."""

import time
import threading
import functools
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import json
from pathlib import Path
import statistics

from ..core.error_handling import get_error_handler, BioNeuroError, ErrorSeverity


@dataclass
class PerformanceMetric:
    """Individual performance measurement."""
    operation: str
    duration: float
    timestamp: datetime
    memory_delta: float = 0.0
    cpu_percent: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class PerformanceProfile:
    """Performance profile for an operation."""
    operation: str
    call_count: int = 0
    total_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0
    avg_duration: float = 0.0
    std_duration: float = 0.0
    percentile_95: float = 0.0
    percentile_99: float = 0.0
    recent_durations: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def update(self, duration: float):
        """Update profile with new measurement."""
        self.call_count += 1
        self.total_duration += duration
        self.min_duration = min(self.min_duration, duration)
        self.max_duration = max(self.max_duration, duration)
        self.recent_durations.append(duration)
        
        # Calculate statistics
        if self.recent_durations:
            durations_list = list(self.recent_durations)
            self.avg_duration = statistics.mean(durations_list)
            
            if len(durations_list) > 1:
                self.std_duration = statistics.stdev(durations_list)
            else:
                self.std_duration = 0.0
                
            # Calculate percentiles
            if len(durations_list) >= 20:  # Need enough samples
                sorted_durations = sorted(durations_list)
                self.percentile_95 = self._percentile(sorted_durations, 95)
                self.percentile_99 = self._percentile(sorted_durations, 99)
                
    def _percentile(self, sorted_data: List[float], percentile: int) -> float:
        """Calculate percentile value."""
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "operation": self.operation,
            "call_count": self.call_count,
            "total_duration": self.total_duration,
            "min_duration": self.min_duration,
            "max_duration": self.max_duration,
            "avg_duration": self.avg_duration,
            "std_duration": self.std_duration,
            "percentile_95": self.percentile_95,
            "percentile_99": self.percentile_99,
        }


class PerformanceProfiler:
    """Advanced performance profiling system."""
    
    def __init__(
        self,
        enable_memory_tracking: bool = True,
        enable_cpu_tracking: bool = True,
        max_metrics: int = 10000
    ):
        self.enable_memory_tracking = enable_memory_tracking
        self.enable_cpu_tracking = enable_cpu_tracking
        self.max_metrics = max_metrics
        
        # Metric storage
        self.profiles: Dict[str, PerformanceProfile] = {}
        self.recent_metrics: deque = deque(maxlen=max_metrics)
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Hot path detection
        self.hot_paths: Dict[str, int] = defaultdict(int)
        self.bottlenecks: Dict[str, float] = {}
        
        # Error handler
        self.error_handler = get_error_handler()
        
        # System resource tracking
        self._setup_system_tracking()
        
    def _setup_system_tracking(self):
        """Setup system resource tracking."""
        try:
            import psutil
            self.psutil_available = True
            self.process = psutil.Process()
        except ImportError:
            self.psutil_available = False
            self.process = None
            
    def profile_function(self, operation_name: Optional[str] = None):
        """Decorator for profiling function performance."""
        def decorator(func: Callable):
            name = operation_name or f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return self.profile_execution(name, func, *args, **kwargs)
            return wrapper
        return decorator
        
    def profile_execution(
        self, 
        operation_name: str, 
        func: Callable, 
        *args, 
        **kwargs
    ) -> Any:
        """Profile execution of a function."""
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()
        start_cpu = self._get_cpu_percent()
        
        try:
            result = func(*args, **kwargs)
            success = True
            error_info = None
        except Exception as e:
            success = False
            error_info = str(e)
            raise
        finally:
            end_time = time.perf_counter()
            end_memory = self._get_memory_usage()
            end_cpu = self._get_cpu_percent()
            
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            cpu_percent = (end_cpu + start_cpu) / 2  # Average
            
            # Record metric
            metric = PerformanceMetric(
                operation=operation_name,
                duration=duration,
                timestamp=datetime.now(),
                memory_delta=memory_delta,
                cpu_percent=cpu_percent,
                metadata={
                    'success': success,
                    'error': error_info,
                    'args_count': len(args),
                    'kwargs_count': len(kwargs)
                }
            )
            
            self.record_metric(metric)
            
        return result
        
    def record_metric(self, metric: PerformanceMetric):
        """Record a performance metric."""
        with self.lock:
            # Update profile
            if metric.operation not in self.profiles:
                self.profiles[metric.operation] = PerformanceProfile(metric.operation)
                
            self.profiles[metric.operation].update(metric.duration)
            
            # Add to recent metrics
            self.recent_metrics.append(metric)
            
            # Track hot paths
            self.hot_paths[metric.operation] += 1
            
            # Update bottleneck detection
            if metric.duration > 0.1:  # Slow operations (>100ms)
                self.bottlenecks[metric.operation] = max(
                    self.bottlenecks.get(metric.operation, 0),
                    metric.duration
                )
                
    def get_profile(self, operation: str) -> Optional[PerformanceProfile]:
        """Get performance profile for an operation."""
        with self.lock:
            return self.profiles.get(operation)
            
    def get_all_profiles(self) -> Dict[str, PerformanceProfile]:
        """Get all performance profiles."""
        with self.lock:
            return dict(self.profiles)
            
    def get_hot_paths(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """Get most frequently called operations."""
        with self.lock:
            return sorted(
                self.hot_paths.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:top_n]
            
    def get_bottlenecks(self, top_n: int = 10) -> List[Tuple[str, float]]:
        """Get operations with highest peak latency."""
        with self.lock:
            return sorted(
                self.bottlenecks.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_n]
            
    def get_slow_operations(self, threshold: float = 0.1) -> List[PerformanceProfile]:
        """Get operations slower than threshold."""
        with self.lock:
            return [
                profile for profile in self.profiles.values()
                if profile.avg_duration > threshold
            ]
            
    def analyze_performance_trends(
        self, 
        operation: str,
        window_minutes: int = 60
    ) -> Dict[str, Any]:
        """Analyze performance trends for an operation."""
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        
        with self.lock:
            # Filter recent metrics for this operation
            recent_metrics = [
                m for m in self.recent_metrics
                if m.operation == operation and m.timestamp >= cutoff_time
            ]
            
            if not recent_metrics:
                return {"error": "No recent metrics found"}
                
            # Calculate trends
            durations = [m.duration for m in recent_metrics]
            memory_deltas = [m.memory_delta for m in recent_metrics]
            timestamps = [m.timestamp for m in recent_metrics]
            
            # Time series analysis (simplified)
            if len(durations) >= 10:
                early_avg = statistics.mean(durations[:len(durations)//2])
                late_avg = statistics.mean(durations[len(durations)//2:])
                trend = "improving" if late_avg < early_avg else "degrading"
                change_percent = ((late_avg - early_avg) / early_avg) * 100
            else:
                trend = "insufficient_data"
                change_percent = 0.0
                
            return {
                "operation": operation,
                "sample_count": len(recent_metrics),
                "avg_duration": statistics.mean(durations),
                "std_duration": statistics.stdev(durations) if len(durations) > 1 else 0,
                "trend": trend,
                "change_percent": change_percent,
                "avg_memory_delta": statistics.mean(memory_deltas),
                "error_rate": len([m for m in recent_metrics if not m.metadata.get('success', True)]) / len(recent_metrics) * 100
            }
            
    def generate_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        with self.lock:
            # Recommend caching for frequently called slow operations
            hot_paths = self.get_hot_paths(20)
            slow_ops = {op.operation: op for op in self.get_slow_operations(0.05)}
            
            for operation, call_count in hot_paths:
                if operation in slow_ops:
                    profile = slow_ops[operation]
                    potential_savings = profile.avg_duration * call_count * 0.5  # Assume 50% improvement
                    
                    recommendations.append({
                        "type": "caching",
                        "operation": operation,
                        "priority": "high" if potential_savings > 10.0 else "medium",
                        "description": f"Add caching for frequently called operation (called {call_count} times, avg {profile.avg_duration:.3f}s)",
                        "potential_savings": potential_savings,
                        "implementation": "Consider memoization or Redis caching"
                    })
                    
            # Recommend async processing for I/O bound operations
            for operation, profile in self.profiles.items():
                if profile.avg_duration > 0.5 and "io" in operation.lower():
                    recommendations.append({
                        "type": "async_processing",
                        "operation": operation,
                        "priority": "medium",
                        "description": f"Convert to async processing (avg {profile.avg_duration:.3f}s)",
                        "potential_savings": profile.avg_duration * 0.7,
                        "implementation": "Use asyncio or thread pool"
                    })
                    
            # Recommend optimization for high variance operations
            for operation, profile in self.profiles.items():
                if profile.std_duration > profile.avg_duration * 0.5:  # High variance
                    recommendations.append({
                        "type": "consistency_optimization",
                        "operation": operation,
                        "priority": "low",
                        "description": f"High variance operation (std: {profile.std_duration:.3f}s, avg: {profile.avg_duration:.3f}s)",
                        "potential_savings": profile.std_duration * 0.3,
                        "implementation": "Profile for algorithmic improvements or resource contention"
                    })
                    
        return sorted(recommendations, key=lambda x: x.get("potential_savings", 0), reverse=True)
        
    def export_performance_report(self, file_path: str):
        """Export comprehensive performance report."""
        with self.lock:
            report = {
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "total_operations": len(self.profiles),
                    "total_calls": sum(p.call_count for p in self.profiles.values()),
                    "total_time": sum(p.total_duration for p in self.profiles.values()),
                    "avg_call_duration": statistics.mean([p.avg_duration for p in self.profiles.values()]) if self.profiles else 0
                },
                "profiles": {name: profile.to_dict() for name, profile in self.profiles.items()},
                "hot_paths": self.get_hot_paths(20),
                "bottlenecks": self.get_bottlenecks(20),
                "recommendations": self.generate_optimization_recommendations()
            }
            
        # Write to file
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        self.error_handler.logger.info(f"Performance report exported to {file_path}")
        
    def _get_memory_usage(self) -> float:
        """Get current memory usage."""
        if not self.enable_memory_tracking or not self.psutil_available:
            return 0.0
            
        try:
            return self.process.memory_info().rss / 1024 / 1024  # MB
        except:
            return 0.0
            
    def _get_cpu_percent(self) -> float:
        """Get current CPU usage."""
        if not self.enable_cpu_tracking or not self.psutil_available:
            return 0.0
            
        try:
            return self.process.cpu_percent()
        except:
            return 0.0
            
    def reset_metrics(self):
        """Reset all collected metrics."""
        with self.lock:
            self.profiles.clear()
            self.recent_metrics.clear()
            self.hot_paths.clear()
            self.bottlenecks.clear()
            
        self.error_handler.logger.info("Performance metrics reset")


# Context manager for profiling code blocks
class ProfileContext:
    """Context manager for profiling code blocks."""
    
    def __init__(self, profiler: PerformanceProfiler, operation_name: str):
        self.profiler = profiler
        self.operation_name = operation_name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.perf_counter() - self.start_time
        
        metric = PerformanceMetric(
            operation=self.operation_name,
            duration=duration,
            timestamp=datetime.now(),
            metadata={
                'success': exc_type is None,
                'error': str(exc_val) if exc_val else None
            }
        )
        
        self.profiler.record_metric(metric)


# Global profiler instance
_profiler = None


def get_profiler() -> PerformanceProfiler:
    """Get global performance profiler instance."""
    global _profiler
    if _profiler is None:
        _profiler = PerformanceProfiler()
    return _profiler


# Convenience decorators
def profile(operation_name: Optional[str] = None):
    """Decorator for profiling function performance."""
    return get_profiler().profile_function(operation_name)


def profile_context(operation_name: str):
    """Context manager for profiling code blocks."""
    return ProfileContext(get_profiler(), operation_name)


# Specialized profilers for different components
class NeuralNetworkProfiler(PerformanceProfiler):
    """Specialized profiler for neural network operations."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Neural network specific metrics
        self.spike_counts: Dict[str, List[int]] = defaultdict(list)
        self.layer_timings: Dict[str, Dict[str, float]] = defaultdict(dict)
        
    def profile_layer(self, layer_name: str, spike_count: int = 0):
        """Profile neural network layer execution."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                
                try:
                    result = func(*args, **kwargs)
                    success = True
                except Exception as e:
                    success = False
                    raise
                finally:
                    duration = time.perf_counter() - start_time
                    
                    # Record layer timing
                    self.layer_timings[layer_name]["last_duration"] = duration
                    self.layer_timings[layer_name]["total_duration"] = (
                        self.layer_timings[layer_name].get("total_duration", 0) + duration
                    )
                    self.layer_timings[layer_name]["call_count"] = (
                        self.layer_timings[layer_name].get("call_count", 0) + 1
                    )
                    
                    # Record spike count if provided
                    if spike_count > 0:
                        self.spike_counts[layer_name].append(spike_count)
                    
                    # Record general metric
                    metric = PerformanceMetric(
                        operation=f"neural_layer.{layer_name}",
                        duration=duration,
                        timestamp=datetime.now(),
                        metadata={
                            'success': success,
                            'spike_count': spike_count,
                            'layer_type': layer_name
                        }
                    )
                    
                    self.record_metric(metric)
                    
                return result
            return wrapper
        return decorator
        
    def get_layer_statistics(self) -> Dict[str, Any]:
        """Get neural network layer performance statistics."""
        stats = {}
        
        for layer_name, timings in self.layer_timings.items():
            call_count = timings.get("call_count", 0)
            total_duration = timings.get("total_duration", 0)
            
            layer_stats = {
                "call_count": call_count,
                "total_duration": total_duration,
                "avg_duration": total_duration / call_count if call_count > 0 else 0,
                "last_duration": timings.get("last_duration", 0)
            }
            
            # Add spike statistics if available
            if layer_name in self.spike_counts and self.spike_counts[layer_name]:
                spikes = self.spike_counts[layer_name]
                layer_stats.update({
                    "avg_spike_count": statistics.mean(spikes),
                    "max_spike_count": max(spikes),
                    "spike_rate_hz": statistics.mean(spikes) / layer_stats["avg_duration"] if layer_stats["avg_duration"] > 0 else 0
                })
                
            stats[layer_name] = layer_stats
            
        return stats