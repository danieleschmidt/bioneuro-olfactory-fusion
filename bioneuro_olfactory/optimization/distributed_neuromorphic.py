"""
Distributed Neuromorphic Computing System
=========================================

This module provides distributed processing capabilities for large-scale
neuromorphic systems, including cluster management, load balancing, and
fault tolerance.

Created as part of Terragon SDLC Generation 3: MAKE IT SCALE
"""

import time
import threading
import queue
import json
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings


class NodeType(Enum):
    """Types of nodes in distributed system."""
    COORDINATOR = "coordinator"
    WORKER = "worker"
    STORAGE = "storage"
    MONITOR = "monitor"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ComputeNode:
    """Represents a compute node in the distributed system."""
    node_id: str
    node_type: NodeType
    capacity: int = 100
    current_load: int = 0
    is_active: bool = True
    last_heartbeat: float = field(default_factory=time.time)
    capabilities: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    def get_load_percentage(self) -> float:
        """Get current load as percentage of capacity."""
        return (self.current_load / self.capacity) * 100 if self.capacity > 0 else 0.0
        
    def can_accept_task(self, task_size: int) -> bool:
        """Check if node can accept a task of given size."""
        return self.is_active and (self.current_load + task_size) <= self.capacity
        
    def update_heartbeat(self):
        """Update last heartbeat timestamp."""
        self.last_heartbeat = time.time()
        
    def is_healthy(self, timeout: float = 30.0) -> bool:
        """Check if node is healthy based on heartbeat."""
        return time.time() - self.last_heartbeat < timeout


@dataclass
class DistributedTask:
    """Represents a task in the distributed system."""
    task_id: str
    task_type: str
    priority: TaskPriority
    data: Any
    computation_func: Callable
    estimated_size: int = 1
    required_capabilities: List[str] = field(default_factory=list)
    timeout: float = 300.0
    retry_count: int = 0
    max_retries: int = 3
    created_time: float = field(default_factory=time.time)
    assigned_node: Optional[str] = None
    result: Any = None
    status: str = "pending"  # pending, running, completed, failed
    error_message: Optional[str] = None


class LoadBalancer:
    """Load balancer for distributing tasks across nodes."""
    
    def __init__(self, strategy: str = "least_loaded"):
        self.strategy = strategy
        self.node_performance_history: Dict[str, List[float]] = defaultdict(list)
        
    def select_node(self, nodes: List[ComputeNode], task: DistributedTask) -> Optional[ComputeNode]:
        """Select the best node for a task based on the strategy."""
        
        # Filter nodes that can handle the task
        eligible_nodes = [
            node for node in nodes 
            if (node.is_healthy() and 
                node.can_accept_task(task.estimated_size) and
                self._has_required_capabilities(node, task))
        ]
        
        if not eligible_nodes:
            return None
            
        # Apply load balancing strategy
        if self.strategy == "least_loaded":
            return min(eligible_nodes, key=lambda n: n.get_load_percentage())
        elif self.strategy == "round_robin":
            return self._round_robin_selection(eligible_nodes)
        elif self.strategy == "performance_based":
            return self._performance_based_selection(eligible_nodes)
        else:
            return eligible_nodes[0]  # Default to first available
            
    def _has_required_capabilities(self, node: ComputeNode, task: DistributedTask) -> bool:
        """Check if node has required capabilities for task."""
        return all(cap in node.capabilities for cap in task.required_capabilities)
        
    def _round_robin_selection(self, nodes: List[ComputeNode]) -> ComputeNode:
        """Round-robin node selection."""
        # Simple round-robin based on node_id
        return min(nodes, key=lambda n: n.node_id)
        
    def _performance_based_selection(self, nodes: List[ComputeNode]) -> ComputeNode:
        """Performance-based node selection."""
        # Select node with best historical performance
        best_node = nodes[0]
        best_score = 0.0
        
        for node in nodes:
            if node.node_id in self.node_performance_history:
                performance_data = self.node_performance_history[node.node_id]
                if performance_data:
                    avg_performance = sum(performance_data) / len(performance_data)
                    # Adjust for current load
                    load_factor = 1.0 - (node.get_load_percentage() / 100.0)
                    score = avg_performance * load_factor
                    
                    if score > best_score:
                        best_score = score
                        best_node = node
                        
        return best_node
        
    def update_node_performance(self, node_id: str, performance_score: float):
        """Update performance history for a node."""
        history = self.node_performance_history[node_id]
        history.append(performance_score)
        
        # Keep only recent history
        if len(history) > 100:
            history.pop(0)


class TaskScheduler:
    """Advanced task scheduler with priority queues."""
    
    def __init__(self):
        self.task_queues: Dict[TaskPriority, queue.PriorityQueue] = {
            priority: queue.PriorityQueue() for priority in TaskPriority
        }
        self.pending_tasks: Dict[str, DistributedTask] = {}
        self.completed_tasks: Dict[str, DistributedTask] = {}
        self.failed_tasks: Dict[str, DistributedTask] = {}
        self.scheduler_lock = threading.Lock()
        
    def submit_task(self, task: DistributedTask):
        """Submit a task to the scheduler."""
        with self.scheduler_lock:
            self.pending_tasks[task.task_id] = task
            
            # Add to priority queue (negative timestamp for earliest first)
            priority_score = (task.priority.value, -task.created_time)
            self.task_queues[task.priority].put((priority_score, task.task_id))
            
    def get_next_task(self) -> Optional[DistributedTask]:
        """Get the next task to be processed."""
        with self.scheduler_lock:
            # Check priority queues from highest to lowest priority
            for priority in sorted(TaskPriority, key=lambda p: p.value, reverse=True):
                if not self.task_queues[priority].empty():
                    try:
                        _, task_id = self.task_queues[priority].get_nowait()
                        if task_id in self.pending_tasks:
                            task = self.pending_tasks[task_id]
                            task.status = "running"
                            return task
                    except queue.Empty:
                        continue
                        
        return None
        
    def complete_task(self, task_id: str, result: Any):
        """Mark task as completed."""
        with self.scheduler_lock:
            if task_id in self.pending_tasks:
                task = self.pending_tasks[task_id]
                task.result = result
                task.status = "completed"
                
                self.completed_tasks[task_id] = task
                del self.pending_tasks[task_id]
                
    def fail_task(self, task_id: str, error_message: str):
        """Mark task as failed."""
        with self.scheduler_lock:
            if task_id in self.pending_tasks:
                task = self.pending_tasks[task_id]
                task.error_message = error_message
                task.retry_count += 1
                
                if task.retry_count >= task.max_retries:
                    task.status = "failed"
                    self.failed_tasks[task_id] = task
                    del self.pending_tasks[task_id]
                else:
                    # Retry task
                    task.status = "pending"
                    priority_score = (task.priority.value, -time.time())
                    self.task_queues[task.priority].put((priority_score, task.task_id))
                    
    def get_statistics(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        with self.scheduler_lock:
            total_tasks = len(self.pending_tasks) + len(self.completed_tasks) + len(self.failed_tasks)
            
            queue_sizes = {
                priority.name: self.task_queues[priority].qsize()
                for priority in TaskPriority
            }
            
            return {
                'pending_tasks': len(self.pending_tasks),
                'completed_tasks': len(self.completed_tasks),
                'failed_tasks': len(self.failed_tasks),
                'total_tasks': total_tasks,
                'success_rate': len(self.completed_tasks) / total_tasks if total_tasks > 0 else 0.0,
                'queue_sizes': queue_sizes
            }


class DistributedNeuromorphicSystem:
    """Main distributed neuromorphic computing system."""
    
    def __init__(self, node_id: str = "coordinator"):
        self.node_id = node_id
        self.nodes: Dict[str, ComputeNode] = {}
        self.load_balancer = LoadBalancer()
        self.task_scheduler = TaskScheduler()
        self.is_running = False
        self.worker_pool = ThreadPoolExecutor(max_workers=4)
        
        # Performance tracking
        self.system_metrics = {
            'total_tasks_processed': 0,
            'total_processing_time': 0.0,
            'average_task_time': 0.0,
            'system_throughput': 0.0,
            'node_utilization': 0.0
        }
        
        # Initialize coordinator node
        self.register_node(ComputeNode(
            node_id=node_id,
            node_type=NodeType.COORDINATOR,
            capacity=1000,
            capabilities=["coordination", "scheduling", "monitoring"]
        ))
        
    def register_node(self, node: ComputeNode):
        """Register a new compute node."""
        self.nodes[node.node_id] = node
        
    def create_worker_node(self, node_id: str, capacity: int = 100, capabilities: List[str] = None) -> ComputeNode:
        """Create and register a worker node."""
        if capabilities is None:
            capabilities = ["neuromorphic_compute", "spike_processing", "pattern_recognition"]
            
        worker_node = ComputeNode(
            node_id=node_id,
            node_type=NodeType.WORKER,
            capacity=capacity,
            capabilities=capabilities
        )
        
        self.register_node(worker_node)
        return worker_node
        
    def submit_neuromorphic_task(self,
                                task_id: str,
                                computation_func: Callable,
                                data: Any,
                                priority: TaskPriority = TaskPriority.NORMAL,
                                required_capabilities: List[str] = None) -> DistributedTask:
        """Submit a neuromorphic computation task."""
        
        if required_capabilities is None:
            required_capabilities = ["neuromorphic_compute"]
            
        task = DistributedTask(
            task_id=task_id,
            task_type="neuromorphic_computation",
            priority=priority,
            data=data,
            computation_func=computation_func,
            estimated_size=self._estimate_task_size(data),
            required_capabilities=required_capabilities
        )
        
        self.task_scheduler.submit_task(task)
        return task
        
    def _estimate_task_size(self, data: Any) -> int:
        """Estimate computational size of task based on data."""
        # Simplified estimation
        if hasattr(data, '__len__'):
            return len(data)
        elif isinstance(data, (int, float)):
            return 1
        else:
            return 10  # Default size
            
    def start_system(self):
        """Start the distributed system."""
        self.is_running = True
        
        # Start task processing threads
        for _ in range(4):  # 4 processing threads
            self.worker_pool.submit(self._task_processing_loop)
            
        # Start monitoring thread
        self.worker_pool.submit(self._monitoring_loop)
        
    def stop_system(self):
        """Stop the distributed system."""
        self.is_running = False
        self.worker_pool.shutdown(wait=True)
        
    def _task_processing_loop(self):
        """Main task processing loop."""
        while self.is_running:
            try:
                # Get next task
                task = self.task_scheduler.get_next_task()
                
                if task is None:
                    time.sleep(0.1)  # Short sleep if no tasks
                    continue
                    
                # Select node for task
                available_nodes = [node for node in self.nodes.values() if node.node_type == NodeType.WORKER]
                selected_node = self.load_balancer.select_node(available_nodes, task)
                
                if selected_node is None:
                    # No available nodes, put task back
                    self.task_scheduler.fail_task(task.task_id, "No available nodes")
                    continue
                    
                # Execute task
                self._execute_task(task, selected_node)
                
            except Exception as e:
                warnings.warn(f"Task processing error: {e}")
                time.sleep(1.0)
                
    def _execute_task(self, task: DistributedTask, node: ComputeNode):
        """Execute a task on a specific node."""
        start_time = time.time()
        
        try:
            # Update node load
            node.current_load += task.estimated_size
            task.assigned_node = node.node_id
            
            # Execute computation
            result = task.computation_func(task.data)
            
            # Record success
            execution_time = time.time() - start_time
            self.task_scheduler.complete_task(task.task_id, result)
            
            # Update performance metrics
            self._update_performance_metrics(execution_time)
            self.load_balancer.update_node_performance(node.node_id, 1.0 / execution_time)
            
        except Exception as e:
            # Record failure
            self.task_scheduler.fail_task(task.task_id, str(e))
            
        finally:
            # Update node load
            node.current_load = max(0, node.current_load - task.estimated_size)
            
    def _update_performance_metrics(self, execution_time: float):
        """Update system performance metrics."""
        self.system_metrics['total_tasks_processed'] += 1
        self.system_metrics['total_processing_time'] += execution_time
        
        # Calculate averages
        total_tasks = self.system_metrics['total_tasks_processed']
        self.system_metrics['average_task_time'] = (
            self.system_metrics['total_processing_time'] / total_tasks
        )
        
        # Calculate throughput (tasks per second over last minute)
        if execution_time > 0:
            self.system_metrics['system_throughput'] = 1.0 / execution_time
            
    def _monitoring_loop(self):
        """System monitoring loop."""
        while self.is_running:
            try:
                # Update node heartbeats and check health
                current_time = time.time()
                
                for node in self.nodes.values():
                    if node.node_type == NodeType.WORKER:
                        # Simulate heartbeat update
                        node.update_heartbeat()
                        
                        # Calculate utilization
                        node.performance_metrics['utilization'] = node.get_load_percentage()
                        
                # Calculate system-wide utilization
                worker_nodes = [n for n in self.nodes.values() if n.node_type == NodeType.WORKER]
                if worker_nodes:
                    total_utilization = sum(n.get_load_percentage() for n in worker_nodes)
                    self.system_metrics['node_utilization'] = total_utilization / len(worker_nodes)
                    
                time.sleep(5.0)  # Monitor every 5 seconds
                
            except Exception as e:
                warnings.warn(f"Monitoring error: {e}")
                time.sleep(1.0)
                
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        
        # Node status
        node_status = {}
        for node_id, node in self.nodes.items():
            node_status[node_id] = {
                'type': node.node_type.value,
                'load_percentage': node.get_load_percentage(),
                'is_healthy': node.is_healthy(),
                'capabilities': node.capabilities,
                'performance_metrics': node.performance_metrics
            }
            
        # Scheduler status
        scheduler_stats = self.task_scheduler.get_statistics()
        
        return {
            'system_running': self.is_running,
            'node_count': len(self.nodes),
            'worker_nodes': len([n for n in self.nodes.values() if n.node_type == NodeType.WORKER]),
            'system_metrics': self.system_metrics,
            'scheduler_statistics': scheduler_stats,
            'node_status': node_status
        }
        
    def process_neuromorphic_dataset(self,
                                   dataset: List[Any],
                                   processing_func: Callable,
                                   batch_size: int = 32) -> List[Any]:
        """Process a large neuromorphic dataset across the distributed system."""
        
        # Submit tasks for dataset processing
        task_ids = []
        
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            task_id = f"batch_{i // batch_size}_{time.time()}"
            
            task = self.submit_neuromorphic_task(
                task_id=task_id,
                computation_func=lambda data: [processing_func(item) for item in data],
                data=batch,
                priority=TaskPriority.NORMAL
            )
            
            task_ids.append(task_id)
            
        # Wait for all tasks to complete
        results = []
        completed_tasks = 0
        timeout = time.time() + 300.0  # 5 minute timeout
        
        while completed_tasks < len(task_ids) and time.time() < timeout:
            for task_id in task_ids:
                if task_id in self.task_scheduler.completed_tasks:
                    task = self.task_scheduler.completed_tasks[task_id]
                    if task.result is not None:
                        results.extend(task.result)
                        completed_tasks += 1
                        
            time.sleep(0.1)
            
        return results


def demonstrate_distributed_neuromorphic():
    """Demonstrate distributed neuromorphic computing capabilities."""
    
    print("🌐 Demonstrating Distributed Neuromorphic System...")
    
    # Create distributed system
    system = DistributedNeuromorphicSystem("main_coordinator")
    
    # Add worker nodes
    for i in range(4):
        system.create_worker_node(f"worker_{i}", capacity=50)
        
    print(f"  Created system with {len(system.nodes)} nodes")
    
    # Start system
    system.start_system()
    
    # Define test computation
    def neuromorphic_computation(data):
        """Simulate neuromorphic computation."""
        # Simulate spike processing
        time.sleep(0.001)  # 1ms processing time
        return sum(x ** 2 for x in data) if isinstance(data, list) else data ** 2
        
    # Create test dataset
    test_dataset = [list(range(i, i + 10)) for i in range(0, 100, 10)]
    
    # Process dataset
    start_time = time.time()
    results = system.process_neuromorphic_dataset(
        test_dataset, 
        neuromorphic_computation,
        batch_size=4
    )
    processing_time = time.time() - start_time
    
    # Get system status
    status = system.get_system_status()
    
    print(f"  Processed {len(test_dataset)} items in {processing_time:.3f}s")
    print(f"  Throughput: {len(test_dataset) / processing_time:.1f} items/sec")
    print(f"  Tasks completed: {status['scheduler_statistics']['completed_tasks']}")
    print(f"  Success rate: {status['scheduler_statistics']['success_rate']:.1%}")
    print(f"  Average node utilization: {status['system_metrics']['node_utilization']:.1f}%")
    
    # Stop system
    system.stop_system()
    
    # Validate results
    expected_results = len(test_dataset)
    actual_results = len(results)
    success = actual_results >= expected_results * 0.8  # 80% success rate
    
    return success, {
        'processing_time': processing_time,
        'throughput': len(test_dataset) / processing_time,
        'success_rate': status['scheduler_statistics']['success_rate'],
        'node_utilization': status['system_metrics']['node_utilization']
    }


if __name__ == "__main__":
    success, metrics = demonstrate_distributed_neuromorphic()
    
    print(f"\n🏆 Distributed Processing: {'✅ SUCCESS' if success else '❌ FAILED'}")
    print(f"  Throughput: {metrics['throughput']:.1f} items/sec")
    print(f"  Success Rate: {metrics['success_rate']:.1%}")
    
    exit(0 if success else 1)