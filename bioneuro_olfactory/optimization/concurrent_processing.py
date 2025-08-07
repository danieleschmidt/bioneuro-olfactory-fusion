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
import weakref
import gc
from contextlib import contextmanager
import resource

logger = logging.getLogger(__name__)


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


@dataclass
class ProcessingTask:
    """A processing task for concurrent execution."""
    task_id: str
    function: Callable
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # Higher priority = executed first
    timeout: Optional[float] = None
    created_at: float = field(default_factory=time.time)
    
    def __lt__(self, other):
        """For priority queue sorting."""
        return self.priority > other.priority  # Higher priority first


class ResourceMonitor:
    """Monitors system resources during processing."""
    
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
                time.sleep(1.0)  # Check every second
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                
    def _check_resources(self):
        """Check current resource usage against limits."""
        violations = []
        
        # Check memory usage
        memory = psutil.virtual_memory()
        if memory.percent > 90:  # Critical memory usage
            violations.append(f"Critical memory usage: {memory.percent:.1f}%")
            
        # Check CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > self.limits.max_cpu_percent:
            violations.append(f"High CPU usage: {cpu_percent:.1f}%")
            
        # Check GPU memory if available
        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
                max_gpu_memory = self.limits.max_gpu_memory_mb / 1024  # GB
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
                    'gpu_memory_reserved_gb': torch.cuda.memory_reserved() / (1024 ** 3),
                    'gpu_utilization': torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
                })
            except Exception:
                pass
                
        return usage
        
    def has_violations(self) -> bool:
        """Check if there are recent resource violations."""
        return len(self._violations) > 0
        
    def clear_violations(self):
        """Clear recorded violations."""
        self._violations.clear()


class WorkerPool(ABC):
    """Abstract base class for worker pools."""
    
    def __init__(self, max_workers: int, resource_limits: ResourceLimits):
        self.max_workers = max_workers
        self.resource_limits = resource_limits
        self.resource_monitor = ResourceMonitor(resource_limits)
        self._active_tasks = 0
        self._completed_tasks = 0
        self._failed_tasks = 0
        self._start_time = time.time()
        
    @abstractmethod
    def submit(self, task: ProcessingTask) -> Future:
        """Submit task for execution."""
        pass
        
    @abstractmethod
    def shutdown(self, wait: bool = True):
        """Shutdown the worker pool."""
        pass
        
    def get_stats(self) -> Dict[str, Any]:
        """Get worker pool statistics."""
        uptime = time.time() - self._start_time
        total_tasks = self._completed_tasks + self._failed_tasks
        
        return {
            'max_workers': self.max_workers,
            'active_tasks': self._active_tasks,
            'completed_tasks': self._completed_tasks,
            'failed_tasks': self._failed_tasks,
            'success_rate': self._completed_tasks / max(total_tasks, 1),
            'tasks_per_second': total_tasks / max(uptime, 1),
            'uptime_seconds': uptime,
            'resource_usage': self.resource_monitor.get_current_usage()
        }


class ThreadWorkerPool(WorkerPool):
    """Thread-based worker pool for I/O intensive tasks."""
    
    def __init__(self, max_workers: int = None, resource_limits: ResourceLimits = None):
        if max_workers is None:
            max_workers = min(32, (psutil.cpu_count() or 1) + 4)
        if resource_limits is None:
            resource_limits = ResourceLimits()
            
        super().__init__(max_workers, resource_limits)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.resource_monitor.start_monitoring()
        
    def submit(self, task: ProcessingTask) -> Future:
        """Submit task to thread pool."""
        if self._active_tasks >= self.resource_limits.max_concurrent_tasks:
            raise RuntimeError("Maximum concurrent tasks exceeded")
            
        def wrapped_task():
            self._active_tasks += 1
            try:
                start_time = time.time()
                result = task.function(*task.args, **task.kwargs)
                execution_time = time.time() - start_time
                
                if task.timeout and execution_time > task.timeout:
                    logger.warning(f"Task {task.task_id} exceeded timeout: {execution_time:.2f}s")
                    
                self._completed_tasks += 1
                return result
            except Exception as e:
                self._failed_tasks += 1
                logger.error(f"Task {task.task_id} failed: {e}")
                raise
            finally:
                self._active_tasks -= 1
                
        return self.executor.submit(wrapped_task)
        
    def shutdown(self, wait: bool = True):
        """Shutdown thread pool."""
        self.resource_monitor.stop_monitoring()
        self.executor.shutdown(wait=wait)


class ProcessWorkerPool(WorkerPool):
    """Process-based worker pool for CPU intensive tasks."""
    
    def __init__(self, max_workers: int = None, resource_limits: ResourceLimits = None):
        if max_workers is None:
            max_workers = psutil.cpu_count() or 1
        if resource_limits is None:
            resource_limits = ResourceLimits()
            
        super().__init__(max_workers, resource_limits)
        
        # Set start method for multiprocessing
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set
            
        self.executor = ProcessPoolExecutor(max_workers=max_workers)
        self.resource_monitor.start_monitoring()
        
    def submit(self, task: ProcessingTask) -> Future:
        """Submit task to process pool."""
        if self._active_tasks >= self.resource_limits.max_concurrent_tasks:
            raise RuntimeError("Maximum concurrent tasks exceeded")
            
        # Wrapper for process execution
        def process_wrapper(func, args, kwargs, task_id, timeout):
            try:
                # Set resource limits in child process
                self._set_process_limits()
                
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                if timeout and execution_time > timeout:
                    logger.warning(f"Process task {task_id} exceeded timeout: {execution_time:.2f}s")
                    
                return result
            except Exception as e:
                logger.error(f"Process task {task_id} failed: {e}")
                raise
                
        self._active_tasks += 1
        
        future = self.executor.submit(
            process_wrapper,
            task.function,
            task.args,
            task.kwargs,
            task.task_id,
            task.timeout
        )
        
        # Add callback to update counters
        def task_completed(fut):
            self._active_tasks -= 1
            try:
                fut.result()  # This will raise if task failed
                self._completed_tasks += 1
            except Exception:
                self._failed_tasks += 1
                
        future.add_done_callback(task_completed)
        return future
        
    def _set_process_limits(self):
        """Set resource limits for child processes."""
        try:
            # Set memory limit
            memory_limit = self.resource_limits.max_memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
            
            # Set CPU time limit  
            cpu_limit = int(self.resource_limits.timeout_seconds)
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
        except Exception as e:
            logger.debug(f"Failed to set process limits: {e}")
            
    def shutdown(self, wait: bool = True):
        """Shutdown process pool."""
        self.resource_monitor.stop_monitoring()
        self.executor.shutdown(wait=wait)


class AsyncWorkerPool:
    """Async-based worker pool for concurrent I/O operations."""
    
    def __init__(self, max_concurrent: int = 100):
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active_tasks = 0
        self._completed_tasks = 0
        self._failed_tasks = 0
        
    async def submit(self, coro) -> Any:
        """Submit coroutine for execution."""
        async with self._semaphore:
            self._active_tasks += 1
            try:
                result = await coro
                self._completed_tasks += 1
                return result
            except Exception as e:
                self._failed_tasks += 1
                logger.error(f"Async task failed: {e}")
                raise
            finally:
                self._active_tasks -= 1
                
    def get_stats(self) -> Dict[str, Any]:
        """Get async pool statistics."""
        total_tasks = self._completed_tasks + self._failed_tasks
        return {
            'max_concurrent': self.max_concurrent,
            'active_tasks': self._active_tasks,
            'completed_tasks': self._completed_tasks,
            'failed_tasks': self._failed_tasks,
            'success_rate': self._completed_tasks / max(total_tasks, 1)
        }


class NeuralProcessingPool:
    """Specialized processing pool for neural network operations."""
    
    def __init__(
        self,
        max_workers: int = None,
        device_pool: List[str] = None,
        resource_limits: ResourceLimits = None
    ):
        self.max_workers = max_workers or psutil.cpu_count()
        self.resource_limits = resource_limits or ResourceLimits()
        
        # Setup device pool
        if device_pool is None:
            device_pool = ['cpu']\n        if torch.cuda.is_available():\n            device_pool.extend([f'cuda:{i}' for i in range(torch.cuda.device_count())])\n            \n        self.device_pool = device_pool\n        self.device_queue = queue.Queue()\n        \n        # Initialize device queue\n        for device in device_pool:\n            self.device_queue.put(device)\n            \n        self.thread_pool = ThreadWorkerPool(max_workers, resource_limits)\n        self._model_cache = weakref.WeakValueDictionary()\n        \n    @contextmanager\n    def get_device(self):\n        \"\"\"Context manager for device allocation.\"\"\"\n        device = self.device_queue.get()\n        try:\n            if device.startswith('cuda'):\n                torch.cuda.set_device(device)\n            yield device\n        finally:\n            self.device_queue.put(device)\n            \n    def submit_neural_task(\n        self, \n        model_fn: Callable,\n        input_data: torch.Tensor,\n        task_id: str = None,\n        **kwargs\n    ) -> Future:\n        \"\"\"Submit neural network processing task.\"\"\"\n        if task_id is None:\n            task_id = f\"neural_{int(time.time() * 1000000)}\"\n            \n        def neural_wrapper():\n            with self.get_device() as device:\n                try:\n                    # Move data to device\n                    if isinstance(input_data, torch.Tensor):\n                        input_data_device = input_data.to(device)\n                    else:\n                        input_data_device = input_data\n                        \n                    # Execute model\n                    with torch.no_grad():  # Disable gradients for inference\n                        result = model_fn(input_data_device, **kwargs)\n                        \n                    # Move result back to CPU if needed\n                    if isinstance(result, torch.Tensor) and result.device != torch.device('cpu'):\n                        result = result.cpu()\n                        \n                    # Clean up GPU memory\n                    if device.startswith('cuda'):\n                        torch.cuda.empty_cache()\n                        \n                    return result\n                    \n                except Exception as e:\n                    logger.error(f\"Neural task {task_id} failed on device {device}: {e}\")\n                    raise\n                    \n        task = ProcessingTask(\n            task_id=task_id,\n            function=neural_wrapper\n        )\n        \n        return self.thread_pool.submit(task)\n        \n    def batch_process(\n        self,\n        model_fn: Callable,\n        data_batch: List[torch.Tensor],\n        batch_size: int = None\n    ) -> List[Any]:\n        \"\"\"Process batch of data in parallel.\"\"\"\n        if batch_size is None:\n            batch_size = min(len(data_batch), self.max_workers)\n            \n        # Split data into chunks\n        chunks = [data_batch[i:i + batch_size] for i in range(0, len(data_batch), batch_size)]\n        \n        futures = []\n        for i, chunk in enumerate(chunks):\n            # Combine chunk into single tensor if possible\n            try:\n                if all(isinstance(item, torch.Tensor) and item.shape[1:] == chunk[0].shape[1:] for item in chunk):\n                    batch_tensor = torch.stack(chunk)\n                else:\n                    batch_tensor = chunk\n            except Exception:\n                batch_tensor = chunk\n                \n            future = self.submit_neural_task(\n                model_fn,\n                batch_tensor,\n                task_id=f\"batch_{i}\"\n            )\n            futures.append(future)\n            \n        # Collect results\n        results = []\n        for future in as_completed(futures):\n            try:\n                result = future.result()\n                if isinstance(result, torch.Tensor) and result.dim() > 1:\n                    results.extend(result.unbind(0))  # Split batch results\n                else:\n                    results.append(result)\n            except Exception as e:\n                logger.error(f\"Batch processing failed: {e}\")\n                results.append(None)\n                \n        return results\n        \n    def shutdown(self):\n        \"\"\"Shutdown neural processing pool.\"\"\"\n        self.thread_pool.shutdown()\n        \n        # Clear model cache\n        self._model_cache.clear()\n        \n        # Clean up GPU memory\n        if torch.cuda.is_available():\n            torch.cuda.empty_cache()\n            \n    def get_stats(self) -> Dict[str, Any]:\n        \"\"\"Get neural processing pool statistics.\"\"\"\n        stats = self.thread_pool.get_stats()\n        stats.update({\n            'available_devices': len(self.device_pool),\n            'device_pool': self.device_pool,\n            'cached_models': len(self._model_cache)\n        })\n        return stats\n\n\nclass ConcurrentProcessor:\n    \"\"\"Main concurrent processing coordinator.\"\"\"\n    \n    def __init__(self, resource_limits: ResourceLimits = None):\n        self.resource_limits = resource_limits or ResourceLimits()\n        \n        # Initialize worker pools\n        self.thread_pool = ThreadWorkerPool(resource_limits=self.resource_limits)\n        self.process_pool = ProcessWorkerPool(resource_limits=self.resource_limits)\n        self.neural_pool = NeuralProcessingPool(resource_limits=self.resource_limits)\n        self.async_pool = AsyncWorkerPool()\n        \n        self._task_queue = queue.PriorityQueue()\n        self._scheduler_thread = None\n        self._running = False\n        \n    def start_scheduler(self):\n        \"\"\"Start task scheduler.\"\"\"\n        if not self._running:\n            self._running = True\n            self._scheduler_thread = threading.Thread(\n                target=self._scheduler_loop,\n                daemon=True\n            )\n            self._scheduler_thread.start()\n            logger.info(\"Concurrent processor scheduler started\")\n            \n    def stop_scheduler(self):\n        \"\"\"Stop task scheduler.\"\"\"\n        self._running = False\n        if self._scheduler_thread:\n            self._scheduler_thread.join(timeout=5.0)\n        logger.info(\"Concurrent processor scheduler stopped\")\n        \n    def _scheduler_loop(self):\n        \"\"\"Main scheduler loop.\"\"\"\n        while self._running:\n            try:\n                # Get next task from priority queue\n                task = self._task_queue.get(timeout=1.0)\n                \n                # Determine optimal execution strategy\n                mode = self._select_processing_mode(task)\n                \n                # Execute task\n                self._execute_task(task, mode)\n                \n            except queue.Empty:\n                continue\n            except Exception as e:\n                logger.error(f\"Scheduler error: {e}\")\n                \n    def _select_processing_mode(self, task: ProcessingTask) -> ProcessingMode:\n        \"\"\"Select optimal processing mode for task.\"\"\"\n        # Simple heuristics - can be made more sophisticated\n        \n        # Check if function is async\n        if asyncio.iscoroutinefunction(task.function):\n            return ProcessingMode.ASYNC\n            \n        # Check for neural network operations\n        if hasattr(task.function, '__name__') and 'neural' in task.function.__name__.lower():\n            return ProcessingMode.THREADED  # Neural pool uses threads\n            \n        # Check resource usage to decide between threads and processes\n        resource_usage = self.thread_pool.resource_monitor.get_current_usage()\n        \n        if resource_usage['cpu_percent'] > 70:\n            return ProcessingMode.THREADED  # I/O bound when CPU is busy\n        else:\n            return ProcessingMode.MULTIPROCESS  # CPU bound when CPU is free\n            \n    def _execute_task(self, task: ProcessingTask, mode: ProcessingMode):\n        \"\"\"Execute task using specified processing mode.\"\"\"\n        try:\n            if mode == ProcessingMode.THREADED:\n                future = self.thread_pool.submit(task)\n            elif mode == ProcessingMode.MULTIPROCESS:\n                future = self.process_pool.submit(task)\n            elif mode == ProcessingMode.ASYNC:\n                # For async tasks, we need to run them in the event loop\n                # This is a simplified version - in practice you'd want proper async handling\n                future = self.thread_pool.submit(task)  # Fallback to thread\n            else:\n                future = self.thread_pool.submit(task)  # Default fallback\n                \n        except Exception as e:\n            logger.error(f\"Failed to execute task {task.task_id}: {e}\")\n            \n    def submit_task(\n        self,\n        function: Callable,\n        args: Tuple = (),\n        kwargs: Dict[str, Any] = None,\n        priority: int = 0,\n        timeout: Optional[float] = None,\n        task_id: str = None\n    ) -> str:\n        \"\"\"Submit task for concurrent execution.\"\"\"\n        if task_id is None:\n            task_id = f\"task_{int(time.time() * 1000000)}\"\n            \n        task = ProcessingTask(\n            task_id=task_id,\n            function=function,\n            args=args,\n            kwargs=kwargs or {},\n            priority=priority,\n            timeout=timeout\n        )\n        \n        self._task_queue.put(task)\n        return task_id\n        \n    def submit_neural_batch(\n        self,\n        model_fn: Callable,\n        data_batch: List[torch.Tensor],\n        batch_size: int = None\n    ) -> List[Any]:\n        \"\"\"Submit neural network batch processing.\"\"\"\n        return self.neural_pool.batch_process(model_fn, data_batch, batch_size)\n        \n    def wait_for_completion(self, timeout: Optional[float] = None):\n        \"\"\"Wait for all queued tasks to complete.\"\"\"\n        start_time = time.time()\n        \n        while not self._task_queue.empty():\n            if timeout and (time.time() - start_time) > timeout:\n                logger.warning(f\"Wait for completion timed out after {timeout}s\")\n                break\n            time.sleep(0.1)\n            \n    def get_global_stats(self) -> Dict[str, Any]:\n        \"\"\"Get statistics for all processing pools.\"\"\"\n        return {\n            'queued_tasks': self._task_queue.qsize(),\n            'thread_pool': self.thread_pool.get_stats(),\n            'process_pool': self.process_pool.get_stats(),\n            'neural_pool': self.neural_pool.get_stats(),\n            'async_pool': self.async_pool.get_stats()\n        }\n        \n    def shutdown(self):\n        \"\"\"Shutdown all processing pools.\"\"\"\n        self.stop_scheduler()\n        \n        self.thread_pool.shutdown()\n        self.process_pool.shutdown()\n        self.neural_pool.shutdown()\n        \n        logger.info(\"Concurrent processor shutdown complete\")\n\n\n# Global concurrent processor instance\ndefault_processor = ConcurrentProcessor()\n\n\ndef parallel_map(\n    func: Callable,\n    iterable,\n    mode: ProcessingMode = ProcessingMode.THREADED,\n    max_workers: int = None,\n    timeout: Optional[float] = None\n) -> List[Any]:\n    \"\"\"Parallel map function with configurable execution mode.\"\"\"\n    if mode == ProcessingMode.THREADED:\n        pool = ThreadWorkerPool(max_workers or 4)\n    elif mode == ProcessingMode.MULTIPROCESS:\n        pool = ProcessWorkerPool(max_workers or psutil.cpu_count())\n    else:\n        raise ValueError(f\"Unsupported mode for parallel_map: {mode}\")\n        \n    try:\n        # Submit all tasks\n        futures = []\n        for i, item in enumerate(iterable):\n            task = ProcessingTask(\n                task_id=f\"map_{i}\",\n                function=func,\n                args=(item,),\n                timeout=timeout\n            )\n            future = pool.submit(task)\n            futures.append(future)\n            \n        # Collect results\n        results = []\n        for future in as_completed(futures, timeout=timeout):\n            try:\n                result = future.result()\n                results.append(result)\n            except Exception as e:\n                logger.error(f\"Parallel map task failed: {e}\")\n                results.append(None)\n                \n        return results\n        \n    finally:\n        pool.shutdown()\n\n\ndef concurrent_task(\n    mode: ProcessingMode = ProcessingMode.THREADED,\n    priority: int = 0,\n    timeout: Optional[float] = None\n):\n    \"\"\"Decorator for concurrent task execution.\"\"\"\n    def decorator(func: Callable):\n        def wrapper(*args, **kwargs):\n            task_id = default_processor.submit_task(\n                func, args, kwargs, priority, timeout\n            )\n            return task_id\n        return wrapper\n    return decorator