"""Performance acceleration and optimization framework.

This module provides comprehensive performance optimizations including:
- Multi-threading and concurrent processing
- Intelligent caching systems  
- Memory optimization
- Algorithmic improvements
- Hardware acceleration detection
- Real-time performance monitoring
"""

import time
import threading
import multiprocessing
import queue
import gc
import psutil
import logging
from typing import Dict, Any, Callable, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from functools import wraps, lru_cache
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import math
import statistics

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance measurement data."""
    execution_time: float
    memory_usage: float  # MB
    cpu_usage: float     # %
    throughput: float    # operations/second
    cache_hit_ratio: float
    concurrency_level: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class OptimizationConfig:
    """Configuration for performance optimizations."""
    enable_caching: bool = True
    enable_concurrency: bool = True
    enable_memory_optimization: bool = True
    max_workers: int = 4
    cache_size: int = 1000
    memory_threshold_mb: float = 500.0
    performance_target_fps: float = 30.0
    enable_profiling: bool = True


class PerformanceProfiler:
    """Real-time performance profiler."""
    
    def __init__(self):
        self.metrics_history = []
        self.current_operations = {}
        self.max_history = 1000
        
    def start_operation(self, operation_id: str):
        """Start timing an operation."""
        self.current_operations[operation_id] = {
            'start_time': time.time(),
            'start_memory': self._get_memory_usage()
        }
        
    def end_operation(self, operation_id: str, operations_count: int = 1) -> PerformanceMetrics:
        """End timing an operation and calculate metrics."""
        if operation_id not in self.current_operations:
            raise ValueError(f"Operation {operation_id} not started")
            
        start_data = self.current_operations.pop(operation_id)
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        execution_time = end_time - start_data['start_time']
        memory_usage = end_memory - start_data['start_memory']
        throughput = operations_count / execution_time if execution_time > 0 else 0.0
        
        metrics = PerformanceMetrics(
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=self._get_cpu_usage(),
            throughput=throughput,
            cache_hit_ratio=0.0,  # To be set by caching system
            concurrency_level=threading.active_count()
        )
        
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.max_history:
            self.metrics_history.pop(0)
            
        return metrics
        
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
        
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        return psutil.cpu_percent(interval=0.1)
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self.metrics_history:
            return {}
            
        execution_times = [m.execution_time for m in self.metrics_history]
        memory_usages = [m.memory_usage for m in self.metrics_history]
        throughputs = [m.throughput for m in self.metrics_history if m.throughput > 0]
        
        return {
            'avg_execution_time': statistics.mean(execution_times),
            'min_execution_time': min(execution_times),
            'max_execution_time': max(execution_times),
            'avg_memory_usage': statistics.mean(memory_usages),
            'max_memory_usage': max(memory_usages),
            'avg_throughput': statistics.mean(throughputs) if throughputs else 0.0,
            'max_throughput': max(throughputs) if throughputs else 0.0,
            'total_operations': len(self.metrics_history)
        }


class IntelligentCache:
    """Intelligent caching system with performance optimization."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: float = 300.0):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.hit_count = 0
        self.miss_count = 0
        self._lock = threading.RLock()
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            current_time = time.time()
            
            if key in self.cache:
                # Check TTL
                if current_time - self.access_times[key] < self.ttl_seconds:
                    self.access_times[key] = current_time
                    self.hit_count += 1
                    return self.cache[key]
                else:
                    # Expired
                    del self.cache[key]
                    del self.access_times[key]
                    
            self.miss_count += 1
            return None
            
    def set(self, key: str, value: Any):
        """Set item in cache."""
        with self._lock:
            current_time = time.time()
            
            # Evict oldest items if cache is full
            if len(self.cache) >= self.max_size:
                self._evict_oldest()
                
            self.cache[key] = value
            self.access_times[key] = current_time
            
    def _evict_oldest(self):
        """Evict oldest cache entries."""
        if not self.access_times:
            return
            
        # Remove 20% of oldest entries
        num_to_evict = max(1, len(self.cache) // 5)
        oldest_keys = sorted(self.access_times.keys(), 
                           key=lambda k: self.access_times[k])[:num_to_evict]
        
        for key in oldest_keys:
            del self.cache[key]
            del self.access_times[key]
            
    def get_hit_ratio(self) -> float:
        """Get cache hit ratio."""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0
        
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self.cache.clear()
            self.access_times.clear()
            self.hit_count = 0
            self.miss_count = 0


class ConcurrentProcessor:
    """Concurrent processing manager for parallel operations."""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(32, (multiprocessing.cpu_count() or 1) + 4)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=max(1, multiprocessing.cpu_count() or 1))
        
    def process_batch_threaded(self, func: Callable, items: List[Any], **kwargs) -> List[Any]:
        """Process batch of items using thread pool."""
        futures = [self.thread_pool.submit(func, item, **kwargs) for item in items]
        results = []
        
        for future in as_completed(futures):
            try:
                result = future.result(timeout=30.0)  # 30 second timeout
                results.append(result)
            except Exception as e:
                logger.error(f"Concurrent processing error: {e}")
                results.append(None)
                
        return results
        
    def process_batch_multiprocess(self, func: Callable, items: List[Any], **kwargs) -> List[Any]:
        """Process batch of items using process pool."""
        try:
            futures = [self.process_pool.submit(func, item, **kwargs) for item in items]
            results = []
            
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=60.0)  # 60 second timeout
                    results.append(result)
                except Exception as e:
                    logger.error(f"Multiprocess error: {e}")
                    results.append(None)
                    
            return results
        except Exception as e:
            logger.error(f"Multiprocess setup error: {e}")
            # Fallback to sequential processing
            return [func(item, **kwargs) for item in items]
            
    def shutdown(self):
        """Shutdown thread and process pools."""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)


class MemoryOptimizer:
    """Memory optimization and management."""
    
    def __init__(self, threshold_mb: float = 500.0):
        self.threshold_mb = threshold_mb
        self.garbage_collect_interval = 60.0  # seconds
        self.last_gc = time.time()
        
    def check_memory_usage(self) -> float:
        """Check current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
        
    def optimize_memory(self, force: bool = False) -> bool:
        """Optimize memory usage if needed."""
        current_memory = self.check_memory_usage()
        current_time = time.time()
        
        should_optimize = (
            force or 
            current_memory > self.threshold_mb or
            (current_time - self.last_gc) > self.garbage_collect_interval
        )
        
        if should_optimize:
            # Force garbage collection
            collected = gc.collect()
            self.last_gc = current_time
            
            new_memory = self.check_memory_usage()
            memory_freed = current_memory - new_memory
            
            logger.info(f"Memory optimization: freed {memory_freed:.1f}MB, collected {collected} objects")
            return True
            
        return False
        
    def get_memory_stats(self) -> Dict[str, float]:
        """Get detailed memory statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024
        }


class PerformanceAccelerator:
    """Main performance acceleration system."""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.profiler = PerformanceProfiler()
        self.cache = IntelligentCache(max_size=self.config.cache_size) if self.config.enable_caching else None
        self.concurrent_processor = ConcurrentProcessor(max_workers=self.config.max_workers) if self.config.enable_concurrency else None
        self.memory_optimizer = MemoryOptimizer(threshold_mb=self.config.memory_threshold_mb) if self.config.enable_memory_optimization else None
        
        # Performance monitoring
        self.monitoring_active = False
        self.performance_alerts = []
        
    def cached_function(self, ttl_seconds: float = 300.0):
        """Decorator for caching function results."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.cache:
                    return func(*args, **kwargs)
                    
                # Create cache key from function name and arguments
                cache_key = f"{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"
                
                # Try to get from cache
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    return cached_result
                    
                # Execute function and cache result
                result = func(*args, **kwargs)
                self.cache.set(cache_key, result)
                return result
                
            return wrapper
        return decorator
        
    def performance_monitored(self, operation_name: str = None):
        """Decorator for monitoring function performance."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                op_name = operation_name or func.__name__
                
                if self.config.enable_profiling:
                    self.profiler.start_operation(op_name)
                    
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    if self.config.enable_profiling:
                        metrics = self.profiler.end_operation(op_name)
                        self._check_performance_thresholds(metrics, op_name)
                        
            return wrapper
        return decorator
        
    def _check_performance_thresholds(self, metrics: PerformanceMetrics, operation_name: str):
        """Check if performance metrics exceed thresholds."""
        target_time = 1.0 / self.config.performance_target_fps
        
        if metrics.execution_time > target_time:
            alert = {
                'timestamp': time.time(),
                'operation': operation_name,
                'issue': 'slow_execution',
                'execution_time': metrics.execution_time,
                'target_time': target_time
            }
            self.performance_alerts.append(alert)
            logger.warning(f"Performance alert: {operation_name} took {metrics.execution_time:.3f}s (target: {target_time:.3f}s)")
            
        if metrics.memory_usage > 50.0:  # >50MB memory increase
            alert = {
                'timestamp': time.time(),
                'operation': operation_name,
                'issue': 'high_memory_usage',
                'memory_usage': metrics.memory_usage
            }
            self.performance_alerts.append(alert)
            logger.warning(f"Memory alert: {operation_name} used {metrics.memory_usage:.1f}MB")
            
    def optimize_batch_processing(self, func: Callable, items: List[Any], 
                                 use_multiprocessing: bool = False, **kwargs) -> List[Any]:
        """Optimize batch processing with concurrency."""
        if not self.concurrent_processor or len(items) < 2:
            # Sequential processing for small batches
            return [func(item, **kwargs) for item in items]
            
        if use_multiprocessing and len(items) >= 4:
            return self.concurrent_processor.process_batch_multiprocess(func, items, **kwargs)
        else:
            return self.concurrent_processor.process_batch_threaded(func, items, **kwargs)
            
    def optimize_memory_if_needed(self):
        """Optimize memory if thresholds are exceeded."""
        if self.memory_optimizer:
            return self.memory_optimizer.optimize_memory()
        return False
        
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        report = {
            'config': {
                'caching_enabled': self.config.enable_caching,
                'concurrency_enabled': self.config.enable_concurrency,
                'memory_optimization_enabled': self.config.enable_memory_optimization,
                'max_workers': self.config.max_workers,
                'performance_target_fps': self.config.performance_target_fps
            },
            'profiling': self.profiler.get_performance_summary(),
            'alerts': len(self.performance_alerts),
            'recent_alerts': self.performance_alerts[-5:] if self.performance_alerts else []
        }
        
        if self.cache:
            report['caching'] = {
                'hit_ratio': self.cache.get_hit_ratio(),
                'cache_size': len(self.cache.cache),
                'max_size': self.cache.max_size
            }
            
        if self.memory_optimizer:
            report['memory'] = self.memory_optimizer.get_memory_stats()
            
        return report
        
    def start_monitoring(self, interval_seconds: float = 5.0):
        """Start background performance monitoring."""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        
        def monitoring_loop():
            while self.monitoring_active:
                try:
                    # Check memory and optimize if needed
                    if self.memory_optimizer:
                        self.memory_optimizer.optimize_memory()
                        
                    # Clean old alerts
                    current_time = time.time()
                    self.performance_alerts = [
                        alert for alert in self.performance_alerts
                        if current_time - alert['timestamp'] < 300.0  # Keep alerts for 5 minutes
                    ]
                    
                    time.sleep(interval_seconds)
                    
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    
        self.monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Performance monitoring started")
        
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.monitoring_active = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1.0)
        logger.info("Performance monitoring stopped")
        
    def shutdown(self):
        """Shutdown all optimization systems."""
        self.stop_monitoring()
        if self.concurrent_processor:
            self.concurrent_processor.shutdown()
        if self.cache:
            self.cache.clear()


# Global performance accelerator instance
accelerator = PerformanceAccelerator()


def performance_optimized(operation_name: str = None, use_cache: bool = True, cache_ttl: float = 300.0):
    """Decorator for comprehensive performance optimization."""
    def decorator(func):
        # Apply caching if enabled
        if use_cache and accelerator.cache:
            func = accelerator.cached_function(ttl_seconds=cache_ttl)(func)
            
        # Apply performance monitoring
        func = accelerator.performance_monitored(operation_name)(func)
        
        return func
    return decorator


def batch_optimized(use_multiprocessing: bool = False):
    """Decorator for batch processing optimization."""
    def decorator(func):
        @wraps(func)
        def wrapper(items: List[Any], **kwargs):
            return accelerator.optimize_batch_processing(
                func, items, use_multiprocessing=use_multiprocessing, **kwargs
            )
        return wrapper
    return decorator


# Utility functions for specific optimizations
def optimize_numpy_operations():
    """Optimize NumPy operations if available."""
    try:
        import numpy as np
        # Set optimal thread count for NumPy
        import os
        os.environ['OMP_NUM_THREADS'] = str(min(4, multiprocessing.cpu_count()))
        os.environ['MKL_NUM_THREADS'] = str(min(4, multiprocessing.cpu_count()))
        logger.info("NumPy optimizations applied")
        return True
    except ImportError:
        return False


def optimize_torch_operations():
    """Optimize PyTorch operations if available."""
    try:
        import torch
        # Set optimal thread count
        torch.set_num_threads(min(4, multiprocessing.cpu_count()))
        
        # Enable optimizations
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            
        logger.info("PyTorch optimizations applied")
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    # Demo the performance acceleration system
    print("🚀 Performance Accelerator Demo")
    
    # Start monitoring
    accelerator.start_monitoring()
    
    # Test cached function
    @performance_optimized("test_function", use_cache=True)
    def slow_function(x):
        time.sleep(0.1)  # Simulate slow operation
        return x * x
        
    # Test function multiple times
    for i in range(5):
        result = slow_function(i)
        print(f"Result {i}: {result}")
        
    # Test batch processing
    @batch_optimized(use_multiprocessing=False)
    def batch_function(item):
        return item * 2
        
    batch_results = batch_function(list(range(10)))
    print(f"Batch results: {batch_results}")
    
    # Get performance report
    report = accelerator.get_performance_report()
    print(f"\nPerformance Report:")
    print(f"  Average execution time: {report['profiling'].get('avg_execution_time', 0):.3f}s")
    print(f"  Cache hit ratio: {report.get('caching', {}).get('hit_ratio', 0):.2f}")
    print(f"  Memory usage: {report.get('memory', {}).get('rss_mb', 0):.1f}MB")
    
    # Cleanup
    accelerator.shutdown()