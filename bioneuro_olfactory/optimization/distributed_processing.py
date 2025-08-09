"""Distributed processing and scaling system for neuromorphic computing."""

import time
import threading
import concurrent.futures
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import queue
import multiprocessing as mp
from collections import defaultdict

from ..core.error_handling import get_error_handler, BioNeuroError, ErrorSeverity
from .performance_profiler import get_profiler


class ProcessingMode(Enum):
    """Processing mode options."""
    SEQUENTIAL = "sequential"
    THREADED = "threaded"
    PROCESS_POOL = "process_pool"
    DISTRIBUTED = "distributed"


@dataclass
class WorkerNode:
    """Worker node information."""
    node_id: str
    host: str
    port: int
    capabilities: List[str]
    current_load: float = 0.0
    max_capacity: int = 4
    status: str = "idle"  # idle, busy, offline
    last_heartbeat: float = field(default_factory=time.time)
    
    def is_available(self) -> bool:
        """Check if worker is available for new tasks."""
        return (
            self.status in ["idle", "busy"] and 
            self.current_load < self.max_capacity and
            time.time() - self.last_heartbeat < 30  # 30 second timeout
        )


@dataclass
class ProcessingTask:
    """Task for distributed processing."""
    task_id: str
    operation: str
    data: Any
    priority: int = 1  # 1=low, 5=high
    timeout: float = 300.0  # 5 minutes default
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.task_id:
            self.task_id = f"{self.operation}_{int(time.time())}"


class TaskScheduler:
    """Intelligent task scheduler for distributed processing."""
    
    def __init__(self):
        self.workers: Dict[str, WorkerNode] = {}
        self.task_queue = queue.PriorityQueue()
        self.active_tasks: Dict[str, ProcessingTask] = {}
        self.completed_tasks: Dict[str, Any] = {}
        self.failed_tasks: List[ProcessingTask] = []
        
        # Load balancing
        self.load_balancer = LoadBalancer()
        self.error_handler = get_error_handler()
        self.profiler = get_profiler()
        
        # Scheduler thread
        self.scheduler_active = False
        self.scheduler_thread: Optional[threading.Thread] = None
        
    def register_worker(self, worker: WorkerNode):
        """Register a new worker node."""
        self.workers[worker.node_id] = worker
        self.error_handler.logger.info(f"Registered worker: {worker.node_id}")
        
    def unregister_worker(self, node_id: str):
        """Unregister a worker node."""
        if node_id in self.workers:
            del self.workers[node_id]
            self.error_handler.logger.info(f"Unregistered worker: {node_id}")
            
    def submit_task(self, task: ProcessingTask) -> str:
        """Submit a task for processing."""
        # Add to priority queue (negative priority for max-heap behavior)
        self.task_queue.put((-task.priority, time.time(), task))
        self.error_handler.logger.debug(f"Task submitted: {task.task_id}")
        return task.task_id
        
    def start_scheduler(self):
        """Start the task scheduler."""
        if self.scheduler_active:
            return
            
        self.scheduler_active = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        self.error_handler.logger.info("Task scheduler started")
        
    def stop_scheduler(self):
        """Stop the task scheduler."""
        self.scheduler_active = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5.0)
        self.error_handler.logger.info("Task scheduler stopped")
        
    def _scheduler_loop(self):
        """Main scheduler loop."""
        while self.scheduler_active:
            try:
                # Get next task (with timeout to allow periodic checks)
                try:
                    _, _, task = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                    
                # Find best worker for task
                worker = self.load_balancer.select_worker(
                    self.workers, 
                    task.operation
                )
                
                if worker:
                    # Assign task to worker
                    self._assign_task(task, worker)
                else:
                    # No available workers, put task back in queue
                    self.task_queue.put((-task.priority, time.time(), task))
                    time.sleep(0.1)  # Brief pause to avoid busy waiting
                    
            except Exception as e:
                self.error_handler.handle_error(
                    BioNeuroError(
                        f"Scheduler error: {str(e)}",
                        error_code="SCHEDULER_ERROR",
                        severity=ErrorSeverity.HIGH
                    )
                )
                
    def _assign_task(self, task: ProcessingTask, worker: WorkerNode):
        """Assign task to a specific worker."""
        try:
            # Update worker status
            worker.current_load += 1
            worker.status = "busy"
            
            # Track active task
            self.active_tasks[task.task_id] = task
            
            # Execute task (in production, this would be sent to remote worker)
            threading.Thread(
                target=self._execute_task,
                args=(task, worker),
                daemon=True
            ).start()
            
        except Exception as e:
            self.error_handler.handle_error(e)
            
    def _execute_task(self, task: ProcessingTask, worker: WorkerNode):
        """Execute task on worker (simulated)."""
        start_time = time.time()
        
        try:
            # Simulate task execution
            result = self._simulate_task_execution(task)
            
            # Record completion
            self.completed_tasks[task.task_id] = {
                "result": result,
                "duration": time.time() - start_time,
                "worker": worker.node_id,
                "timestamp": time.time()
            }
            
            self.error_handler.logger.debug(f"Task completed: {task.task_id}")
            
        except Exception as e:
            # Handle task failure
            task.retry_count += 1
            
            if task.retry_count <= task.max_retries:
                # Retry task
                self.task_queue.put((-task.priority, time.time(), task))
                self.error_handler.logger.warning(f"Task retry {task.retry_count}: {task.task_id}")
            else:
                # Task failed permanently
                self.failed_tasks.append(task)
                self.error_handler.logger.error(f"Task failed: {task.task_id}, {str(e)}")
                
        finally:
            # Update worker status
            worker.current_load -= 1
            worker.status = "idle" if worker.current_load == 0 else "busy"
            
            # Remove from active tasks
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
                
    def _simulate_task_execution(self, task: ProcessingTask) -> Any:
        """Simulate task execution (replace with actual processing)."""
        operation = task.operation
        data = task.data
        
        # Simulate different operation types
        if operation == "neural_inference":
            time.sleep(0.1 + len(str(data)) * 0.001)  # Simulate processing time
            return {"prediction": [0.1, 0.9, 0.2, 0.1], "confidence": 0.85}
            
        elif operation == "sensor_processing":
            time.sleep(0.05)
            return {"processed_readings": [x * 1.1 for x in data.get("readings", [])]}
            
        elif operation == "spike_encoding":
            time.sleep(0.02)
            return {"spike_train": [1, 0, 1, 0, 1] * len(str(data))}
            
        else:
            # Generic processing
            time.sleep(0.1)
            return {"status": "processed", "data_size": len(str(data))}
            
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status."""
        return {
            "active": self.scheduler_active,
            "workers": len(self.workers),
            "available_workers": len([w for w in self.workers.values() if w.is_available()]),
            "queued_tasks": self.task_queue.qsize(),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "worker_details": {
                node_id: {
                    "load": worker.current_load,
                    "capacity": worker.max_capacity,
                    "status": worker.status,
                    "utilization": worker.current_load / worker.max_capacity * 100
                }
                for node_id, worker in self.workers.items()
            }
        }


class LoadBalancer:
    """Load balancer for distributing tasks across workers."""
    
    def __init__(self, strategy: str = "least_loaded"):
        self.strategy = strategy
        self.worker_weights: Dict[str, float] = defaultdict(lambda: 1.0)
        
    def select_worker(
        self, 
        workers: Dict[str, WorkerNode], 
        operation: str
    ) -> Optional[WorkerNode]:
        """Select best worker for operation."""
        available_workers = [w for w in workers.values() if w.is_available()]
        
        if not available_workers:
            return None
            
        if self.strategy == "least_loaded":
            return min(available_workers, key=lambda w: w.current_load / w.max_capacity)
            
        elif self.strategy == "round_robin":
            # Simple round robin (in practice, would maintain state)
            return available_workers[int(time.time()) % len(available_workers)]
            
        elif self.strategy == "capability_based":
            # Filter by capability
            capable_workers = [
                w for w in available_workers 
                if operation in w.capabilities or not w.capabilities
            ]
            if capable_workers:
                return min(capable_workers, key=lambda w: w.current_load / w.max_capacity)
            else:
                return min(available_workers, key=lambda w: w.current_load / w.max_capacity)
                
        elif self.strategy == "weighted":
            # Weighted selection based on performance
            if available_workers:
                weights = [self.worker_weights[w.node_id] for w in available_workers]
                # Select worker with highest weight and lowest load
                best_score = -1
                best_worker = None
                
                for worker, weight in zip(available_workers, weights):
                    load_factor = 1.0 - (worker.current_load / worker.max_capacity)
                    score = weight * load_factor
                    
                    if score > best_score:
                        best_score = score
                        best_worker = worker
                        
                return best_worker
                
        else:
            # Default to least loaded
            return min(available_workers, key=lambda w: w.current_load / w.max_capacity)
            
    def update_worker_weight(self, node_id: str, weight: float):
        """Update worker weight for weighted load balancing."""
        self.worker_weights[node_id] = weight


class DistributedProcessor:
    """Main distributed processing coordinator."""
    
    def __init__(
        self,
        mode: ProcessingMode = ProcessingMode.THREADED,
        max_workers: int = None
    ):
        self.mode = mode
        self.max_workers = max_workers or mp.cpu_count()
        
        # Components
        self.scheduler = TaskScheduler()
        self.error_handler = get_error_handler()
        self.profiler = get_profiler()
        
        # Thread/Process pools
        self.thread_pool: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self.process_pool: Optional[concurrent.futures.ProcessPoolExecutor] = None
        
        # Auto-scaling
        self.auto_scaler = AutoScaler(self)
        
        self._initialize_processing_mode()
        
    def _initialize_processing_mode(self):
        """Initialize processing based on selected mode."""
        if self.mode == ProcessingMode.THREADED:
            self.thread_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="bioneuro_worker"
            )
            
        elif self.mode == ProcessingMode.PROCESS_POOL:
            self.process_pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.max_workers
            )
            
        elif self.mode == ProcessingMode.DISTRIBUTED:
            # Initialize distributed workers
            self._setup_distributed_workers()
            self.scheduler.start_scheduler()
            
    def _setup_distributed_workers(self):
        """Setup distributed worker nodes."""
        # Create local worker nodes (in production, these would be remote)
        for i in range(self.max_workers):
            worker = WorkerNode(
                node_id=f"local_worker_{i}",
                host="localhost",
                port=8000 + i,
                capabilities=["neural_inference", "sensor_processing", "spike_encoding"],
                max_capacity=2
            )
            self.scheduler.register_worker(worker)
            
    def process_batch(
        self,
        tasks: List[Tuple[str, Any]],  # (operation, data) pairs
        timeout: float = 300.0
    ) -> List[Any]:
        """Process a batch of tasks."""
        if self.mode == ProcessingMode.SEQUENTIAL:
            return self._process_sequential(tasks)
            
        elif self.mode == ProcessingMode.THREADED:
            return self._process_threaded(tasks, timeout)
            
        elif self.mode == ProcessingMode.PROCESS_POOL:
            return self._process_multiprocess(tasks, timeout)
            
        elif self.mode == ProcessingMode.DISTRIBUTED:
            return self._process_distributed(tasks, timeout)
            
        else:
            raise ValueError(f"Unknown processing mode: {self.mode}")
            
    def _process_sequential(self, tasks: List[Tuple[str, Any]]) -> List[Any]:
        """Process tasks sequentially."""
        results = []
        
        for operation, data in tasks:
            with self.profiler.profile_context(f"sequential.{operation}"):
                result = self._execute_operation(operation, data)
                results.append(result)
                
        return results
        
    def _process_threaded(
        self, 
        tasks: List[Tuple[str, Any]], 
        timeout: float
    ) -> List[Any]:
        """Process tasks using thread pool."""
        futures = []
        
        for operation, data in tasks:
            future = self.thread_pool.submit(self._execute_operation, operation, data)
            futures.append(future)
            
        # Wait for completion
        results = []
        for i, future in enumerate(futures):
            try:
                result = future.result(timeout=timeout)
                results.append(result)
            except Exception as e:
                self.error_handler.logger.error(f"Task {i} failed: {str(e)}")
                results.append(None)
                
        return results
        
    def _process_multiprocess(
        self, 
        tasks: List[Tuple[str, Any]], 
        timeout: float
    ) -> List[Any]:
        """Process tasks using process pool."""
        futures = []
        
        for operation, data in tasks:
            future = self.process_pool.submit(self._execute_operation, operation, data)
            futures.append(future)
            
        # Wait for completion
        results = []
        for i, future in enumerate(futures):
            try:
                result = future.result(timeout=timeout)
                results.append(result)
            except Exception as e:
                self.error_handler.logger.error(f"Process task {i} failed: {str(e)}")
                results.append(None)
                
        return results
        
    def _process_distributed(
        self, 
        tasks: List[Tuple[str, Any]], 
        timeout: float
    ) -> List[Any]:
        """Process tasks using distributed scheduler."""
        task_ids = []
        
        # Submit all tasks
        for i, (operation, data) in enumerate(tasks):
            task = ProcessingTask(
                task_id=f"batch_{int(time.time())}_{i}",
                operation=operation,
                data=data,
                timeout=timeout
            )
            task_id = self.scheduler.submit_task(task)
            task_ids.append(task_id)
            
        # Wait for completion
        results = []
        start_time = time.time()
        
        for task_id in task_ids:
            while time.time() - start_time < timeout:
                if task_id in self.scheduler.completed_tasks:
                    result = self.scheduler.completed_tasks[task_id]["result"]
                    results.append(result)
                    break
                elif any(t.task_id == task_id for t in self.scheduler.failed_tasks):
                    results.append(None)
                    break
                else:
                    time.sleep(0.01)  # Brief pause
            else:
                # Timeout
                results.append(None)
                
        return results
        
    def _execute_operation(self, operation: str, data: Any) -> Any:
        """Execute a single operation (same as TaskScheduler simulation)."""
        task = ProcessingTask(task_id="", operation=operation, data=data)
        return self.scheduler._simulate_task_execution(task)
        
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = {
            "mode": self.mode.value,
            "max_workers": self.max_workers,
        }
        
        if self.mode == ProcessingMode.DISTRIBUTED:
            stats.update(self.scheduler.get_status())
        elif self.mode == ProcessingMode.THREADED and self.thread_pool:
            stats.update({
                "thread_pool_size": self.thread_pool._max_workers
            })
        elif self.mode == ProcessingMode.PROCESS_POOL and self.process_pool:
            stats.update({
                "process_pool_size": self.process_pool._max_workers
            })
            
        return stats
        
    def shutdown(self):
        """Shutdown the processor."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
        if self.mode == ProcessingMode.DISTRIBUTED:
            self.scheduler.stop_scheduler()
            
        self.error_handler.logger.info("Distributed processor shutdown")


class AutoScaler:
    """Auto-scaling system for dynamic resource allocation."""
    
    def __init__(self, processor: DistributedProcessor):
        self.processor = processor
        self.scaling_enabled = True
        self.min_workers = 1
        self.max_workers = mp.cpu_count() * 2
        self.scale_up_threshold = 0.8  # CPU utilization
        self.scale_down_threshold = 0.3
        self.scale_cooldown = 60  # seconds
        self.last_scale_time = 0
        
    def check_scaling_conditions(self) -> Optional[str]:
        """Check if scaling is needed."""
        if not self.scaling_enabled:
            return None
            
        # Check cooldown period
        if time.time() - self.last_scale_time < self.scale_cooldown:
            return None
            
        # Get current metrics
        stats = self.processor.get_processing_stats()
        
        if self.processor.mode == ProcessingMode.DISTRIBUTED:
            # Check worker utilization
            workers = stats.get("worker_details", {})
            if not workers:
                return None
                
            avg_utilization = sum(
                w.get("utilization", 0) for w in workers.values()
            ) / len(workers) / 100.0
            
            current_workers = len(workers)
            
            # Scale up if high utilization and can add workers
            if (avg_utilization > self.scale_up_threshold and 
                current_workers < self.max_workers):
                return "scale_up"
                
            # Scale down if low utilization and have extra workers
            elif (avg_utilization < self.scale_down_threshold and 
                  current_workers > self.min_workers):
                return "scale_down"
                
        return None
        
    def execute_scaling(self, action: str) -> bool:
        """Execute scaling action."""
        try:
            if action == "scale_up":
                return self._scale_up()
            elif action == "scale_down":
                return self._scale_down()
            else:
                return False
        except Exception as e:
            self.processor.error_handler.handle_error(e)
            return False
            
    def _scale_up(self) -> bool:
        """Add more workers."""
        current_workers = len(self.processor.scheduler.workers)
        if current_workers >= self.max_workers:
            return False
            
        # Add new worker
        new_worker = WorkerNode(
            node_id=f"auto_worker_{current_workers}",
            host="localhost",
            port=8100 + current_workers,
            capabilities=["neural_inference", "sensor_processing"],
            max_capacity=2
        )
        
        self.processor.scheduler.register_worker(new_worker)
        self.last_scale_time = time.time()
        
        self.processor.error_handler.logger.info(f"Scaled up: added worker {new_worker.node_id}")
        return True
        
    def _scale_down(self) -> bool:
        """Remove workers."""
        workers = self.processor.scheduler.workers
        if len(workers) <= self.min_workers:
            return False
            
        # Find least utilized worker
        least_utilized = min(
            workers.values(), 
            key=lambda w: w.current_load / w.max_capacity
        )
        
        # Only remove if not busy
        if least_utilized.current_load == 0:
            self.processor.scheduler.unregister_worker(least_utilized.node_id)
            self.last_scale_time = time.time()
            
            self.processor.error_handler.logger.info(f"Scaled down: removed worker {least_utilized.node_id}")
            return True
            
        return False


# Global processor instance
_distributed_processor = None


def get_distributed_processor() -> DistributedProcessor:
    """Get global distributed processor instance."""
    global _distributed_processor
    if _distributed_processor is None:
        _distributed_processor = DistributedProcessor()
    return _distributed_processor