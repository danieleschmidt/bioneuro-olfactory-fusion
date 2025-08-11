"""Concurrent processing and resource pooling for neuromorphic systems.

This module implements high-performance concurrent processing optimized
for spiking neural networks and real-time gas detection workloads.
"""

import torch
import torch.multiprocessing as mp
import numpy as np
import asyncio
import threading
import queue
import time
import psutil
from typing import Dict, Any, List, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future, as_completed
from abc import ABC, abstractmethod
import logging
from enum import Enum
from collections import deque
import weakref
import gc
from contextlib import contextmanager
import resource

logger = logging.getLogger(__name__)


class AdaptiveTimeout:
    """Adaptive timeout management based on execution history."""
    
    def __init__(self, initial_timeout: float = 30.0, min_timeout: float = 1.0, 
                 max_timeout: float = 300.0):
        self.min_timeout = min_timeout
        self.max_timeout = max_timeout
        self.current_timeout = initial_timeout
        
        self.execution_times: deque = deque(maxlen=100)
        self.timeout_count = 0
        self.total_tasks = 0
        
    def get_timeout(self) -> float:
        """Get current adaptive timeout."""
        return self.current_timeout
        
    def update_timing(self, execution_time: float):
        """Update timing statistics."""
        self.execution_times.append(execution_time)
        self.total_tasks += 1
        
        if len(self.execution_times) >= 10:
            avg_time = sum(self.execution_times) / len(self.execution_times)
            p95_time = sorted(self.execution_times)[int(0.95 * len(self.execution_times))]
            
            new_timeout = min(self.max_timeout, max(self.min_timeout, p95_time * 2))
            self.current_timeout = 0.8 * self.current_timeout + 0.2 * new_timeout
            
    def record_timeout(self):
        """Record timeout occurrence."""
        self.timeout_count += 1
        self.current_timeout = min(self.max_timeout, self.current_timeout * 1.1)
        
    def get_average_time(self) -> float:
        """Get average execution time."""
        return sum(self.execution_times) / len(self.execution_times) if self.execution_times else 0
        
    def get_timeout_rate(self) -> float:
        """Get timeout rate."""
        return self.timeout_count / max(self.total_tasks, 1)


class ProcessingMode(Enum):
    """Processing execution modes."""
    SEQUENTIAL = "sequential"
    THREADED = "threaded"
    MULTIPROCESS = "multiprocess"
    ASYNC = "async"
    HYBRID = "hybrid"


@dataclass
class ResourceLimits:
    """Resource usage limits for processing."""
    max_memory_mb: int = 2048
    max_cpu_percent: float = 80.0
    max_gpu_memory_mb: int = 4096
    max_concurrent_tasks: int = 8
    timeout_seconds: float = 30.0
    enable_adaptive_batching: bool = True
    enable_priority_scheduling: bool = True
    enable_backpressure: bool = True


@dataclass
class ProcessingTask:
    """Processing task definition."""
    task_id: str
    function: Callable
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    timeout: Optional[float] = None
    created_at: float = field(default_factory=time.time)
    estimated_duration: Optional[float] = None
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    
    def __lt__(self, other):
        """Priority comparison."""
        return self.priority > other.priority


class ResourceMonitor:
    """Monitor system resource usage."""
    
    def __init__(self, limits: ResourceLimits):
        self.limits = limits
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._violations: List[str] = []
        
    def start_monitoring(self):
        """Start resource monitoring."""
        if not self._monitoring:
            self._monitoring = True
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop,
                daemon=True
            )
            self._monitor_thread.start()
            
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
            
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._monitoring:
            try:
                self._check_resources()
                time.sleep(1.0)
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                
    def _check_resources(self):
        """Check resource usage."""
        violations = []
        
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            violations.append(f"Critical memory usage: {memory.percent:.1f}%")
            
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > self.limits.max_cpu_percent:
            violations.append(f"High CPU usage: {cpu_percent:.1f}%")
            
        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.memory_allocated() / (1024 ** 3)
                max_gpu_memory = self.limits.max_gpu_memory_mb / 1024
                if gpu_memory > max_gpu_memory:
                    violations.append(f"High GPU memory usage: {gpu_memory:.1f}GB")
            except Exception:
                pass
                
        if violations:
            self._violations.extend(violations)
            logger.warning(f"Resource violations: {violations}")
            
    def get_current_usage(self) -> Dict[str, Any]:
        """Get current resource usage."""
        memory = psutil.virtual_memory()
        usage = {
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024 ** 3),
            'cpu_percent': psutil.cpu_percent(),
            'disk_usage_percent': psutil.disk_usage('/').percent
        }
        
        if torch.cuda.is_available():
            try:
                usage.update({
                    'gpu_memory_allocated_gb': torch.cuda.memory_allocated() / (1024 ** 3),
                    'gpu_memory_reserved_gb': torch.cuda.memory_reserved() / (1024 ** 3)
                })
            except Exception:
                pass
                
        return usage
        
    def has_violations(self) -> bool:
        """Check if there are resource violations."""
        return len(self._violations) > 0
        
    def clear_violations(self):
        """Clear violation history."""
        self._violations.clear()


class ConcurrentProcessor:
    """Main concurrent processing coordinator."""
    
    def __init__(self, resource_limits: ResourceLimits = None):
        self.resource_limits = resource_limits or ResourceLimits()
        self.resource_monitor = ResourceMonitor(self.resource_limits)
        
        self.thread_pool = ThreadPoolExecutor(max_workers=psutil.cpu_count() or 1)
        self.process_pool = ProcessPoolExecutor(max_workers=psutil.cpu_count() or 1)
        
        self._task_queue = queue.PriorityQueue()
        self._scheduler_thread = None
        self._running = False
        
        self.resource_monitor.start_monitoring()
        
    def submit_task(self, function: Callable, args: Tuple = (), 
                   kwargs: Dict[str, Any] = None, priority: int = 0) -> Future:
        """Submit task for execution."""
        task = ProcessingTask(
            task_id=f"task_{int(time.time() * 1000000)}",
            function=function,
            args=args,
            kwargs=kwargs or {},
            priority=priority
        )
        
        return self.thread_pool.submit(task.function, *task.args, **task.kwargs)
        
    def shutdown(self):
        """Shutdown processor."""
        self.resource_monitor.stop_monitoring()
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            'resource_usage': self.resource_monitor.get_current_usage(),
            'violations': self.resource_monitor.has_violations()
        }


# Global concurrent processor instance
default_processor = ConcurrentProcessor()


def parallel_map(func: Callable, iterable, max_workers: int = None) -> List[Any]:
    """Parallel map function."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(func, item) for item in iterable]
        return [future.result() for future in as_completed(futures)]