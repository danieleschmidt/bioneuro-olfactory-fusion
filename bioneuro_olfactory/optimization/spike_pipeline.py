"""
Advanced concurrent spike processing pipelines with priority scheduling.

This module implements high-performance concurrent processing pipelines specifically
optimized for neuromorphic spike data with priority-based scheduling and backpressure control.
"""

import asyncio
import threading
import queue
import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from abc import ABC, abstractmethod
from collections import deque
import heapq
import weakref
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

logger = logging.getLogger(__name__)


class Priority(IntEnum):
    """Processing priority levels (lower number = higher priority)."""
    CRITICAL = 0    # Real-time safety-critical processing
    HIGH = 1        # Time-sensitive inference
    NORMAL = 2      # Standard processing
    LOW = 3         # Background tasks
    BATCH = 4       # Batch processing


class PipelineStage(Enum):
    """Pipeline processing stages."""
    INPUT = "input"
    PREPROCESS = "preprocess"
    ENCODE = "encode"
    PROCESS = "process"
    DECODE = "decode"
    OUTPUT = "output"


@dataclass
class SpikeTask:
    """Task for spike processing pipeline."""
    task_id: str
    priority: Priority
    spike_data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    deadline_ms: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def __lt__(self, other):
        """Priority comparison for heap queue."""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.created_at < other.created_at
    
    def is_expired(self) -> bool:
        """Check if task has exceeded its deadline."""
        if self.deadline_ms is None:
            return False
        return (time.time() * 1000 - self.created_at * 1000) > self.deadline_ms


class ConcurrentSpikeProcessor:
    """High-performance concurrent spike processor with priority scheduling."""
    
    def __init__(self, 
                 num_workers: int = 4,
                 queue_size: int = 10000,
                 enable_backpressure: bool = True,
                 priority_boost_threshold_ms: float = 100.0):
        self.num_workers = num_workers
        self.queue_size = queue_size
        self.enable_backpressure = enable_backpressure
        self.priority_boost_threshold_ms = priority_boost_threshold_ms
        
        # Task queues (one per priority level)
        self.task_queues: Dict[Priority, asyncio.PriorityQueue] = {
            priority: asyncio.PriorityQueue(maxsize=queue_size // len(Priority))
            for priority in Priority
        }
        
        # Worker management
        self.workers: List[asyncio.Task] = []
        self.worker_stats: List[Dict[str, Any]] = []
        self.running = False
        
        # Performance tracking
        self.total_tasks_processed = 0
        self.total_processing_time_ms = 0.0
        self.priority_stats = {p: {"processed": 0, "avg_time_ms": 0.0} for p in Priority}
        
        # Backpressure control
        self.backpressure_threshold = int(queue_size * 0.8)
        self.drop_rate = 0.0
        
        # Dynamic priority adjustment
        self.priority_boost_enabled = True
        
    async def start(self):
        """Start the concurrent processor."""
        if self.running:
            return
            
        self.running = True
        
        # Start worker tasks
        for i in range(self.num_workers):
            worker = asyncio.create_task(self._worker_loop(i))
            self.workers.append(worker)
            self.worker_stats.append({
                "worker_id": i,
                "tasks_processed": 0,
                "avg_processing_time_ms": 0.0,
                "current_priority": None
            })
        
        # Start monitoring task
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        
        logger.info(f"Started concurrent spike processor with {self.num_workers} workers")
    
    async def stop(self):
        """Stop the concurrent processor."""
        self.running = False
        
        # Cancel workers
        for worker in self.workers:
            worker.cancel()
        
        # Cancel monitor
        if hasattr(self, 'monitor_task'):
            self.monitor_task.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        
        self.workers.clear()
        logger.info("Stopped concurrent spike processor")
    
    async def submit_task(self, task: SpikeTask) -> bool:
        """Submit task for processing."""
        # Check for backpressure
        if self.enable_backpressure:
            total_queued = sum(q.qsize() for q in self.task_queues.values())
            if total_queued > self.backpressure_threshold:
                # Drop low priority tasks
                if task.priority in [Priority.LOW, Priority.BATCH]:
                    self.drop_rate = (self.drop_rate * 0.9) + (1.0 * 0.1)  # Exponential moving average
                    logger.warning(f"Dropping task {task.task_id} due to backpressure")
                    return False
        
        # Dynamic priority boosting for aging tasks
        if self.priority_boost_enabled:
            age_ms = (time.time() - task.created_at) * 1000
            if age_ms > self.priority_boost_threshold_ms and task.priority > Priority.CRITICAL:
                original_priority = task.priority
                task.priority = Priority(max(0, task.priority - 1))
                logger.debug(f"Boosted task {task.task_id} priority from {original_priority} to {task.priority}")
        
        # Submit to appropriate queue
        try:
            await self.task_queues[task.priority].put(task)
            return True
        except asyncio.QueueFull:
            logger.warning(f"Queue full for priority {task.priority}, dropping task {task.task_id}")
            return False
    
    async def _worker_loop(self, worker_id: int):
        """Main worker loop with priority-based task selection."""
        logger.info(f"Started worker {worker_id}")
        
        while self.running:
            try:
                # Select highest priority task available
                task = await self._get_next_task()
                if task is None:
                    await asyncio.sleep(0.001)  # Brief sleep if no tasks
                    continue
                
                # Check if task has expired
                if task.is_expired():
                    logger.warning(f"Task {task.task_id} expired, skipping")
                    continue
                
                # Process task
                start_time = time.perf_counter()
                
                try:
                    result = await self._process_task(task)
                    processing_time = (time.perf_counter() - start_time) * 1000
                    
                    # Update statistics
                    self._update_worker_stats(worker_id, processing_time, task.priority)
                    self._update_priority_stats(task.priority, processing_time)
                    
                    logger.debug(f"Worker {worker_id} completed task {task.task_id} in {processing_time:.2f}ms")
                    
                except Exception as e:
                    logger.error(f"Worker {worker_id} failed to process task {task.task_id}: {e}")
                    
                    # Retry if possible
                    if task.retry_count < task.max_retries:
                        task.retry_count += 1
                        await self.submit_task(task)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(0.1)
        
        logger.info(f"Worker {worker_id} stopped")
    
    async def _get_next_task(self) -> Optional[SpikeTask]:
        """Get next task based on priority."""
        # Try each priority level in order
        for priority in Priority:
            queue = self.task_queues[priority]
            if not queue.empty():
                try:
                    task = queue.get_nowait()
                    return task
                except asyncio.QueueEmpty:
                    continue
        
        return None
    
    async def _process_task(self, task: SpikeTask) -> Any:
        """Process a spike task."""
        # This would contain the actual spike processing logic
        # For now, simulate processing
        await asyncio.sleep(0.001)  # Simulate processing time
        return f"Processed {task.task_id}"
    
    async def _monitor_loop(self):
        """Monitor system performance and adjust parameters."""
        while self.running:
            try:
                # Calculate current load
                total_queued = sum(q.qsize() for q in self.task_queues.values())
                load_factor = total_queued / (self.queue_size * len(Priority))
                
                # Adjust backpressure threshold based on load
                if load_factor > 0.9:
                    self.backpressure_threshold = int(self.backpressure_threshold * 0.95)
                elif load_factor < 0.3:
                    self.backpressure_threshold = int(self.backpressure_threshold * 1.05)
                
                # Log performance metrics
                if self.total_tasks_processed > 0:
                    avg_time = self.total_processing_time_ms / self.total_tasks_processed
                    throughput = self.total_tasks_processed / (time.time() - getattr(self, '_start_time', time.time()))
                    
                    logger.debug(f"Pipeline stats - Queued: {total_queued}, "
                               f"Avg time: {avg_time:.2f}ms, "
                               f"Throughput: {throughput:.1f} tasks/s, "
                               f"Drop rate: {self.drop_rate:.3f}")
                
                await asyncio.sleep(5.0)  # Monitor every 5 seconds
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(1.0)
    
    def _update_worker_stats(self, worker_id: int, processing_time_ms: float, priority: Priority):
        """Update worker statistics."""
        stats = self.worker_stats[worker_id]
        stats["tasks_processed"] += 1
        
        # Update moving average
        if stats["avg_processing_time_ms"] == 0:
            stats["avg_processing_time_ms"] = processing_time_ms
        else:
            stats["avg_processing_time_ms"] = (
                stats["avg_processing_time_ms"] * 0.9 + processing_time_ms * 0.1
            )
        
        stats["current_priority"] = priority.name
    
    def _update_priority_stats(self, priority: Priority, processing_time_ms: float):
        """Update priority-level statistics."""
        stats = self.priority_stats[priority]
        stats["processed"] += 1
        
        if stats["avg_time_ms"] == 0:
            stats["avg_time_ms"] = processing_time_ms
        else:
            stats["avg_time_ms"] = stats["avg_time_ms"] * 0.9 + processing_time_ms * 0.1
        
        self.total_tasks_processed += 1
        self.total_processing_time_ms += processing_time_ms
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        total_queued = sum(q.qsize() for q in self.task_queues.values())
        
        queue_stats = {
            priority.name: queue.qsize() 
            for priority, queue in self.task_queues.items()
        }
        
        return {
            "total_tasks_processed": self.total_tasks_processed,
            "avg_processing_time_ms": (
                self.total_processing_time_ms / max(self.total_tasks_processed, 1)
            ),
            "current_queue_sizes": queue_stats,
            "total_queued": total_queued,
            "backpressure_threshold": self.backpressure_threshold,
            "drop_rate": self.drop_rate,
            "worker_stats": self.worker_stats,
            "priority_stats": {
                priority.name: stats for priority, stats in self.priority_stats.items()
            },
            "throughput_tasks_per_second": (
                self.total_tasks_processed / max(time.time() - getattr(self, '_start_time', time.time()), 1)
            )
        }


# Global processor instance
_spike_processor: Optional[ConcurrentSpikeProcessor] = None


async def get_spike_processor(num_workers: int = 4, queue_size: int = 10000) -> ConcurrentSpikeProcessor:
    """Get global spike processor instance."""
    global _spike_processor
    
    if _spike_processor is None:
        _spike_processor = ConcurrentSpikeProcessor(num_workers, queue_size)
        await _spike_processor.start()
    
    return _spike_processor