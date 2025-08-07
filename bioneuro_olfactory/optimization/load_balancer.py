"""Advanced load balancing and auto-scaling for neuromorphic systems.

This module implements intelligent load balancing and auto-scaling
mechanisms for distributed neuromorphic gas detection systems.
"""

import asyncio
import time
import threading
import queue
import heapq
import psutil
import torch
import numpy as np
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from enum import Enum
import logging
import json
import socket
import uuid
import weakref

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    ADAPTIVE_WEIGHTED = "adaptive_weighted"
    RESOURCE_AWARE = "resource_aware"
    NEURAL_OPTIMIZED = "neural_optimized"


class ScalingPolicy(Enum):
    """Auto-scaling policies."""
    REACTIVE = "reactive"          # Scale based on current metrics
    PREDICTIVE = "predictive"      # Scale based on predicted load
    HYBRID = "hybrid"             # Combination of reactive and predictive
    TIME_BASED = "time_based"     # Scale based on time patterns
    NEURAL_ADAPTIVE = "neural_adaptive"  # ML-based scaling decisions


@dataclass
class WorkerNode:
    """Represents a worker node in the system."""
    node_id: str
    address: str
    port: int
    capabilities: List[str] = field(default_factory=list)
    current_load: float = 0.0
    max_capacity: int = 100
    active_connections: int = 0
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    last_heartbeat: float = field(default_factory=time.time)
    health_score: float = 1.0
    weight: float = 1.0
    
    # Resource metrics
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: Optional[float] = None
    
    # Neuromorphic-specific metrics
    avg_spike_rate: float = 0.0
    processing_latency: float = 0.0
    accuracy_score: float = 1.0
    
    @property
    def is_healthy(self) -> bool:
        """Check if node is healthy."""
        return (time.time() - self.last_heartbeat) < 30 and self.health_score > 0.5
        
    @property
    def availability_score(self) -> float:
        """Calculate node availability score."""
        load_factor = 1.0 - (self.current_load / 100.0)
        capacity_factor = 1.0 - (self.active_connections / max(self.max_capacity, 1))
        response_factor = 1.0 / (1.0 + np.mean(self.response_times) if self.response_times else 1.0)
        
        return (load_factor * 0.4 + capacity_factor * 0.3 + 
                response_factor * 0.2 + self.health_score * 0.1)
        
    def update_metrics(self, metrics: Dict[str, Any]):
        """Update node metrics."""
        self.current_load = metrics.get('cpu_usage', self.current_load)
        self.cpu_usage = metrics.get('cpu_usage', self.cpu_usage)
        self.memory_usage = metrics.get('memory_usage', self.memory_usage)
        self.gpu_usage = metrics.get('gpu_usage', self.gpu_usage)
        self.active_connections = metrics.get('active_connections', self.active_connections)
        self.health_score = metrics.get('health_score', self.health_score)
        self.last_heartbeat = time.time()
        
        # Neuromorphic metrics
        if 'avg_spike_rate' in metrics:
            self.avg_spike_rate = metrics['avg_spike_rate']
        if 'processing_latency' in metrics:
            self.processing_latency = metrics['processing_latency']
        if 'accuracy_score' in metrics:
            self.accuracy_score = metrics['accuracy_score']


@dataclass
class LoadMetrics:
    """System load metrics for scaling decisions."""
    timestamp: float
    total_requests: int
    active_connections: int
    avg_response_time: float
    cpu_utilization: float
    memory_utilization: float
    gpu_utilization: Optional[float]
    queue_depth: int
    error_rate: float
    
    # Neuromorphic-specific metrics
    inference_throughput: float = 0.0
    spike_processing_rate: float = 0.0
    detection_accuracy: float = 0.0


class LoadBalancer:
    """Advanced load balancer with multiple strategies."""
    
    def __init__(
        self,
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE_WEIGHTED,
        health_check_interval: float = 10.0
    ):
        self.strategy = strategy
        self.health_check_interval = health_check_interval
        
        self.nodes: Dict[str, WorkerNode] = {}
        self.request_history: deque = deque(maxlen=1000)
        self.metrics_history: deque = deque(maxlen=100)
        
        # Strategy-specific state
        self.rr_index = 0  # Round robin index
        self.adaptive_weights = defaultdict(float)
        
        # Health monitoring
        self._health_monitor_thread = None
        self._monitoring = False
        
        # Performance tracking
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
    def add_node(self, node: WorkerNode):
        """Add worker node to load balancer."""
        self.nodes[node.node_id] = node
        self.adaptive_weights[node.node_id] = node.weight
        logger.info(f"Added worker node: {node.node_id} at {node.address}:{node.port}")
        
    def remove_node(self, node_id: str) -> bool:
        """Remove worker node from load balancer."""
        if node_id in self.nodes:
            del self.nodes[node_id]
            if node_id in self.adaptive_weights:
                del self.adaptive_weights[node_id]
            logger.info(f"Removed worker node: {node_id}")
            return True
        return False
        
    def select_node(self, request_metadata: Dict[str, Any] = None) -> Optional[WorkerNode]:
        """Select optimal node based on current strategy."""
        healthy_nodes = [node for node in self.nodes.values() if node.is_healthy]
        
        if not healthy_nodes:
            return None
            
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(healthy_nodes)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_selection(healthy_nodes)
        elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return self._least_response_time_selection(healthy_nodes)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_selection(healthy_nodes)
        elif self.strategy == LoadBalancingStrategy.ADAPTIVE_WEIGHTED:
            return self._adaptive_weighted_selection(healthy_nodes)
        elif self.strategy == LoadBalancingStrategy.RESOURCE_AWARE:
            return self._resource_aware_selection(healthy_nodes)
        elif self.strategy == LoadBalancingStrategy.NEURAL_OPTIMIZED:
            return self._neural_optimized_selection(healthy_nodes, request_metadata)
        else:
            return self._round_robin_selection(healthy_nodes)  # Fallback
            
    def _round_robin_selection(self, nodes: List[WorkerNode]) -> WorkerNode:
        """Simple round robin selection."""
        if not nodes:
            return None
            
        selected = nodes[self.rr_index % len(nodes)]
        self.rr_index = (self.rr_index + 1) % len(nodes)
        return selected
        
    def _least_connections_selection(self, nodes: List[WorkerNode]) -> WorkerNode:
        """Select node with least active connections."""
        return min(nodes, key=lambda n: n.active_connections)
        
    def _least_response_time_selection(self, nodes: List[WorkerNode]) -> WorkerNode:
        """Select node with lowest average response time."""
        def avg_response_time(node):
            return np.mean(node.response_times) if node.response_times else 0
            
        return min(nodes, key=avg_response_time)
        
    def _weighted_round_robin_selection(self, nodes: List[WorkerNode]) -> WorkerNode:
        """Weighted round robin based on node weights."""
        # Calculate cumulative weights
        total_weight = sum(node.weight for node in nodes)
        if total_weight == 0:
            return self._round_robin_selection(nodes)
            
        # Use weighted random selection
        random_weight = np.random.random() * total_weight
        cumulative_weight = 0
        
        for node in nodes:
            cumulative_weight += node.weight
            if random_weight <= cumulative_weight:
                return node
                \n        return nodes[0]  # Fallback\n        \n    def _adaptive_weighted_selection(self, nodes: List[WorkerNode]) -> WorkerNode:\n        \"\"\"Adaptive weighted selection based on performance history.\"\"\"\n        # Update adaptive weights based on recent performance\n        for node in nodes:\n            # Base weight on availability score and performance metrics\n            performance_score = (\n                node.availability_score * 0.4 +\n                (1.0 / (node.processing_latency + 0.1)) * 0.3 +\n                node.accuracy_score * 0.2 +\n                node.health_score * 0.1\n            )\n            \n            # Update adaptive weight with exponential moving average\n            alpha = 0.1  # Learning rate\n            self.adaptive_weights[node.node_id] = (\n                (1 - alpha) * self.adaptive_weights[node.node_id] +\n                alpha * performance_score\n            )\n            \n        # Select based on adaptive weights\n        weights = [self.adaptive_weights[node.node_id] for node in nodes]\n        total_weight = sum(weights)\n        \n        if total_weight == 0:\n            return self._round_robin_selection(nodes)\n            \n        # Weighted random selection\n        random_weight = np.random.random() * total_weight\n        cumulative_weight = 0\n        \n        for i, node in enumerate(nodes):\n            cumulative_weight += weights[i]\n            if random_weight <= cumulative_weight:\n                return node\n                \n        return nodes[0]  # Fallback\n        \n    def _resource_aware_selection(self, nodes: List[WorkerNode]) -> WorkerNode:\n        \"\"\"Selection based on current resource utilization.\"\"\"\n        def resource_score(node):\n            cpu_factor = 1.0 - (node.cpu_usage / 100.0)\n            memory_factor = 1.0 - (node.memory_usage / 100.0)\n            connection_factor = 1.0 - (node.active_connections / max(node.max_capacity, 1))\n            \n            score = cpu_factor * 0.4 + memory_factor * 0.3 + connection_factor * 0.3\n            \n            # Bonus for GPU availability if relevant\n            if node.gpu_usage is not None:\n                gpu_factor = 1.0 - (node.gpu_usage / 100.0)\n                score = score * 0.7 + gpu_factor * 0.3\n                \n            return score\n            \n        return max(nodes, key=resource_score)\n        \n    def _neural_optimized_selection(self, nodes: List[WorkerNode], metadata: Dict[str, Any]) -> WorkerNode:\n        \"\"\"Selection optimized for neuromorphic processing.\"\"\"\n        if not metadata:\n            return self._adaptive_weighted_selection(nodes)\n            \n        # Consider request-specific requirements\n        requires_gpu = metadata.get('requires_gpu', False)\n        expected_spike_rate = metadata.get('expected_spike_rate', 0.0)\n        required_accuracy = metadata.get('required_accuracy', 0.8)\n        \n        # Filter nodes based on requirements\n        suitable_nodes = []\n        for node in nodes:\n            if requires_gpu and node.gpu_usage is None:\n                continue\n            if node.accuracy_score < required_accuracy:\n                continue\n            suitable_nodes.append(node)\n            \n        if not suitable_nodes:\n            suitable_nodes = nodes  # Fallback to all healthy nodes\n            \n        # Score nodes based on neural processing capabilities\n        def neural_score(node):\n            # Base availability score\n            base_score = node.availability_score\n            \n            # Accuracy bonus\n            accuracy_bonus = node.accuracy_score * 0.3\n            \n            # Processing speed bonus\n            speed_bonus = 1.0 / (node.processing_latency + 0.01) * 0.2\n            \n            # Spike rate compatibility\n            spike_compatibility = 1.0 - abs(node.avg_spike_rate - expected_spike_rate) / max(expected_spike_rate, 1.0)\n            spike_bonus = spike_compatibility * 0.2\n            \n            return base_score + accuracy_bonus + speed_bonus + spike_bonus\n            \n        return max(suitable_nodes, key=neural_score)\n        \n    def record_request(self, node_id: str, response_time: float, success: bool):\n        \"\"\"Record request metrics for adaptive learning.\"\"\"\n        self.total_requests += 1\n        \n        if success:\n            self.successful_requests += 1\n        else:\n            self.failed_requests += 1\n            \n        # Update node metrics\n        if node_id in self.nodes:\n            node = self.nodes[node_id]\n            node.response_times.append(response_time)\n            \n            # Adjust health score based on success rate\n            if not success:\n                node.health_score = max(0.1, node.health_score * 0.95)\n            else:\n                node.health_score = min(1.0, node.health_score * 1.01)\n                \n        # Record in request history\n        self.request_history.append({\n            'timestamp': time.time(),\n            'node_id': node_id,\n            'response_time': response_time,\n            'success': success\n        })\n        \n    def start_health_monitoring(self):\n        \"\"\"Start health monitoring for all nodes.\"\"\"\n        if not self._monitoring:\n            self._monitoring = True\n            self._health_monitor_thread = threading.Thread(\n                target=self._health_monitor_loop,\n                daemon=True\n            )\n            self._health_monitor_thread.start()\n            logger.info(\"Load balancer health monitoring started\")\n            \n    def stop_health_monitoring(self):\n        \"\"\"Stop health monitoring.\"\"\"\n        self._monitoring = False\n        if self._health_monitor_thread:\n            self._health_monitor_thread.join(timeout=5.0)\n        logger.info(\"Load balancer health monitoring stopped\")\n        \n    def _health_monitor_loop(self):\n        \"\"\"Health monitoring loop.\"\"\"\n        while self._monitoring:\n            try:\n                self._check_node_health()\n                time.sleep(self.health_check_interval)\n            except Exception as e:\n                logger.error(f\"Health monitoring error: {e}\")\n                \n    def _check_node_health(self):\n        \"\"\"Check health of all nodes.\"\"\"\n        current_time = time.time()\n        unhealthy_nodes = []\n        \n        for node_id, node in self.nodes.items():\n            # Check heartbeat timeout\n            if (current_time - node.last_heartbeat) > 30:\n                node.health_score = max(0.0, node.health_score - 0.1)\n                if node.health_score < 0.3:\n                    unhealthy_nodes.append(node_id)\n                    \n        # Log unhealthy nodes\n        for node_id in unhealthy_nodes:\n            logger.warning(f\"Node {node_id} is unhealthy\")\n            \n    def get_stats(self) -> Dict[str, Any]:\n        \"\"\"Get load balancer statistics.\"\"\"\n        total_requests = self.total_requests\n        success_rate = self.successful_requests / max(total_requests, 1)\n        \n        # Calculate average response time\n        recent_requests = list(self.request_history)[-100:]  # Last 100 requests\n        avg_response_time = np.mean([r['response_time'] for r in recent_requests]) if recent_requests else 0\n        \n        # Node statistics\n        node_stats = {}\n        for node_id, node in self.nodes.items():\n            node_stats[node_id] = {\n                'healthy': node.is_healthy,\n                'current_load': node.current_load,\n                'active_connections': node.active_connections,\n                'health_score': node.health_score,\n                'availability_score': node.availability_score,\n                'avg_response_time': np.mean(node.response_times) if node.response_times else 0\n            }\n            \n        return {\n            'strategy': self.strategy.value,\n            'total_nodes': len(self.nodes),\n            'healthy_nodes': sum(1 for n in self.nodes.values() if n.is_healthy),\n            'total_requests': total_requests,\n            'success_rate': success_rate,\n            'avg_response_time': avg_response_time,\n            'nodes': node_stats\n        }\n\n\nclass AutoScaler:\n    \"\"\"Intelligent auto-scaler for dynamic resource management.\"\"\"\n    \n    def __init__(\n        self,\n        load_balancer: LoadBalancer,\n        policy: ScalingPolicy = ScalingPolicy.HYBRID,\n        min_nodes: int = 1,\n        max_nodes: int = 10,\n        scale_up_threshold: float = 80.0,\n        scale_down_threshold: float = 30.0,\n        scale_up_cooldown: float = 300.0,  # 5 minutes\n        scale_down_cooldown: float = 600.0  # 10 minutes\n    ):\n        self.load_balancer = load_balancer\n        self.policy = policy\n        self.min_nodes = min_nodes\n        self.max_nodes = max_nodes\n        self.scale_up_threshold = scale_up_threshold\n        self.scale_down_threshold = scale_down_threshold\n        self.scale_up_cooldown = scale_up_cooldown\n        self.scale_down_cooldown = scale_down_cooldown\n        \n        # Scaling history\n        self.scaling_history: deque = deque(maxlen=100)\n        self.last_scale_up = 0.0\n        self.last_scale_down = 0.0\n        \n        # Predictive models (simplified)\n        self.load_predictor = LoadPredictor()\n        \n        # Auto-scaling thread\n        self._scaling_thread = None\n        self._scaling = False\n        self._scaling_interval = 30.0  # Check every 30 seconds\n        \n        # Node provisioning callbacks\n        self._node_provisioner: Optional[Callable] = None\n        self._node_terminator: Optional[Callable] = None\n        \n    def set_node_provisioner(self, provisioner: Callable[[int], List[WorkerNode]]):\n        \"\"\"Set callback for provisioning new nodes.\"\"\"\n        self._node_provisioner = provisioner\n        \n    def set_node_terminator(self, terminator: Callable[[List[str]], bool]):\n        \"\"\"Set callback for terminating nodes.\"\"\"\n        self._node_terminator = terminator\n        \n    def start_scaling(self):\n        \"\"\"Start auto-scaling monitoring.\"\"\"\n        if not self._scaling:\n            self._scaling = True\n            self._scaling_thread = threading.Thread(\n                target=self._scaling_loop,\n                daemon=True\n            )\n            self._scaling_thread.start()\n            logger.info(\"Auto-scaler started\")\n            \n    def stop_scaling(self):\n        \"\"\"Stop auto-scaling monitoring.\"\"\"\n        self._scaling = False\n        if self._scaling_thread:\n            self._scaling_thread.join(timeout=5.0)\n        logger.info(\"Auto-scaler stopped\")\n        \n    def _scaling_loop(self):\n        \"\"\"Main scaling loop.\"\"\"\n        while self._scaling:\n            try:\n                self._evaluate_scaling_decision()\n                time.sleep(self._scaling_interval)\n            except Exception as e:\n                logger.error(f\"Auto-scaling error: {e}\")\n                time.sleep(60)  # Back off on error\n                \n    def _evaluate_scaling_decision(self):\n        \"\"\"Evaluate whether to scale up or down.\"\"\"\n        current_time = time.time()\n        current_metrics = self._collect_current_metrics()\n        \n        # Store metrics for prediction\n        self.load_predictor.add_metrics(current_metrics)\n        \n        # Determine scaling action based on policy\n        if self.policy == ScalingPolicy.REACTIVE:\n            action = self._reactive_scaling_decision(current_metrics)\n        elif self.policy == ScalingPolicy.PREDICTIVE:\n            action = self._predictive_scaling_decision(current_metrics)\n        elif self.policy == ScalingPolicy.HYBRID:\n            action = self._hybrid_scaling_decision(current_metrics)\n        elif self.policy == ScalingPolicy.TIME_BASED:\n            action = self._time_based_scaling_decision(current_metrics)\n        elif self.policy == ScalingPolicy.NEURAL_ADAPTIVE:\n            action = self._neural_adaptive_scaling_decision(current_metrics)\n        else:\n            action = None\n            \n        # Execute scaling action\n        if action:\n            self._execute_scaling_action(action, current_metrics)\n            \n    def _collect_current_metrics(self) -> LoadMetrics:\n        \"\"\"Collect current system load metrics.\"\"\"\n        lb_stats = self.load_balancer.get_stats()\n        \n        # Calculate aggregate metrics\n        total_cpu = 0.0\n        total_memory = 0.0\n        total_connections = 0\n        total_gpu = 0.0\n        gpu_nodes = 0\n        \n        for node in self.load_balancer.nodes.values():\n            if node.is_healthy:\n                total_cpu += node.cpu_usage\n                total_memory += node.memory_usage\n                total_connections += node.active_connections\n                \n                if node.gpu_usage is not None:\n                    total_gpu += node.gpu_usage\n                    gpu_nodes += 1\n                    \n        healthy_nodes = max(lb_stats['healthy_nodes'], 1)\n        avg_cpu = total_cpu / healthy_nodes\n        avg_memory = total_memory / healthy_nodes\n        avg_gpu = total_gpu / max(gpu_nodes, 1) if gpu_nodes > 0 else None\n        \n        return LoadMetrics(\n            timestamp=time.time(),\n            total_requests=lb_stats['total_requests'],\n            active_connections=total_connections,\n            avg_response_time=lb_stats['avg_response_time'],\n            cpu_utilization=avg_cpu,\n            memory_utilization=avg_memory,\n            gpu_utilization=avg_gpu,\n            queue_depth=0,  # Would need to be implemented\n            error_rate=1.0 - lb_stats['success_rate']\n        )\n        \n    def _reactive_scaling_decision(self, metrics: LoadMetrics) -> Optional[str]:\n        \"\"\"Reactive scaling based on current metrics.\"\"\"\n        current_time = time.time()\n        \n        # Check scale up conditions\n        if (metrics.cpu_utilization > self.scale_up_threshold and\n            current_time - self.last_scale_up > self.scale_up_cooldown and\n            len(self.load_balancer.nodes) < self.max_nodes):\n            return \"scale_up\"\n            \n        # Check scale down conditions\n        if (metrics.cpu_utilization < self.scale_down_threshold and\n            current_time - self.last_scale_down > self.scale_down_cooldown and\n            len(self.load_balancer.nodes) > self.min_nodes):\n            return \"scale_down\"\n            \n        return None\n        \n    def _predictive_scaling_decision(self, metrics: LoadMetrics) -> Optional[str]:\n        \"\"\"Predictive scaling based on load forecasting.\"\"\"\n        # Predict load for next few minutes\n        predicted_load = self.load_predictor.predict_load(horizon_minutes=10)\n        \n        if predicted_load > self.scale_up_threshold * 0.8:  # Scale up proactively\n            return \"scale_up\"\n        elif predicted_load < self.scale_down_threshold * 1.2:  # Scale down conservatively\n            return \"scale_down\"\n            \n        return None\n        \n    def _hybrid_scaling_decision(self, metrics: LoadMetrics) -> Optional[str]:\n        \"\"\"Hybrid scaling combining reactive and predictive approaches.\"\"\"\n        reactive_decision = self._reactive_scaling_decision(metrics)\n        predictive_decision = self._predictive_scaling_decision(metrics)\n        \n        # Prioritize scale up decisions\n        if reactive_decision == \"scale_up\" or predictive_decision == \"scale_up\":\n            return \"scale_up\"\n            \n        # Be more conservative with scale down\n        if reactive_decision == \"scale_down\" and predictive_decision == \"scale_down\":\n            return \"scale_down\"\n            \n        return None\n        \n    def _time_based_scaling_decision(self, metrics: LoadMetrics) -> Optional[str]:\n        \"\"\"Time-based scaling for known traffic patterns.\"\"\"\n        # This would be customized based on known traffic patterns\n        # For now, just use reactive scaling\n        return self._reactive_scaling_decision(metrics)\n        \n    def _neural_adaptive_scaling_decision(self, metrics: LoadMetrics) -> Optional[str]:\n        \"\"\"Neural network-based scaling decisions.\"\"\"\n        # This would use a trained neural network to make scaling decisions\n        # For now, use hybrid approach\n        return self._hybrid_scaling_decision(metrics)\n        \n    def _execute_scaling_action(self, action: str, metrics: LoadMetrics):\n        \"\"\"Execute the scaling action.\"\"\"\n        current_time = time.time()\n        \n        if action == \"scale_up\":\n            self._scale_up(metrics)\n            self.last_scale_up = current_time\n        elif action == \"scale_down\":\n            self._scale_down(metrics)\n            self.last_scale_down = current_time\n            \n        # Record scaling event\n        self.scaling_history.append({\n            'timestamp': current_time,\n            'action': action,\n            'metrics': metrics,\n            'node_count_before': len(self.load_balancer.nodes),\n            'node_count_after': len(self.load_balancer.nodes)  # Will be updated after scaling\n        })\n        \n    def _scale_up(self, metrics: LoadMetrics):\n        \"\"\"Scale up by adding new nodes.\"\"\"\n        if not self._node_provisioner:\n            logger.warning(\"Cannot scale up: no node provisioner configured\")\n            return\n            \n        # Determine how many nodes to add\n        current_nodes = len(self.load_balancer.nodes)\n        target_nodes = min(current_nodes + 1, self.max_nodes)  # Add one node at a time\n        nodes_to_add = target_nodes - current_nodes\n        \n        if nodes_to_add > 0:\n            try:\n                new_nodes = self._node_provisioner(nodes_to_add)\n                for node in new_nodes:\n                    self.load_balancer.add_node(node)\n                    \n                logger.info(f\"Scaled up: added {len(new_nodes)} nodes\")\n            except Exception as e:\n                logger.error(f\"Scale up failed: {e}\")\n                \n    def _scale_down(self, metrics: LoadMetrics):\n        \"\"\"Scale down by removing nodes.\"\"\"\n        if not self._node_terminator:\n            logger.warning(\"Cannot scale down: no node terminator configured\")\n            return\n            \n        current_nodes = len(self.load_balancer.nodes)\n        target_nodes = max(current_nodes - 1, self.min_nodes)  # Remove one node at a time\n        nodes_to_remove = current_nodes - target_nodes\n        \n        if nodes_to_remove > 0:\n            # Select nodes to remove (prefer nodes with least connections)\n            candidates = sorted(\n                self.load_balancer.nodes.values(),\n                key=lambda n: n.active_connections\n            )\n            \n            nodes_to_terminate = [n.node_id for n in candidates[:nodes_to_remove]]\n            \n            try:\n                success = self._node_terminator(nodes_to_terminate)\n                if success:\n                    for node_id in nodes_to_terminate:\n                        self.load_balancer.remove_node(node_id)\n                    logger.info(f\"Scaled down: removed {len(nodes_to_terminate)} nodes\")\n            except Exception as e:\n                logger.error(f\"Scale down failed: {e}\")\n                \n    def get_stats(self) -> Dict[str, Any]:\n        \"\"\"Get auto-scaler statistics.\"\"\"\n        recent_scaling = [s for s in self.scaling_history if time.time() - s['timestamp'] < 3600]  # Last hour\n        \n        scale_up_count = sum(1 for s in recent_scaling if s['action'] == 'scale_up')\n        scale_down_count = sum(1 for s in recent_scaling if s['action'] == 'scale_down')\n        \n        return {\n            'policy': self.policy.value,\n            'current_nodes': len(self.load_balancer.nodes),\n            'min_nodes': self.min_nodes,\n            'max_nodes': self.max_nodes,\n            'scale_up_threshold': self.scale_up_threshold,\n            'scale_down_threshold': self.scale_down_threshold,\n            'recent_scale_ups': scale_up_count,\n            'recent_scale_downs': scale_down_count,\n            'last_scale_up': self.last_scale_up,\n            'last_scale_down': self.last_scale_down\n        }\n\n\nclass LoadPredictor:\n    \"\"\"Simple load predictor for auto-scaling decisions.\"\"\"\n    \n    def __init__(self, history_size: int = 100):\n        self.history_size = history_size\n        self.metrics_history: deque = deque(maxlen=history_size)\n        \n    def add_metrics(self, metrics: LoadMetrics):\n        \"\"\"Add metrics to history.\"\"\"\n        self.metrics_history.append(metrics)\n        \n    def predict_load(self, horizon_minutes: int = 10) -> float:\n        \"\"\"Predict load for given time horizon.\"\"\"\n        if len(self.metrics_history) < 10:\n            # Not enough data, return current load\n            return self.metrics_history[-1].cpu_utilization if self.metrics_history else 50.0\n            \n        # Simple trend-based prediction\n        recent_metrics = list(self.metrics_history)[-10:]\n        cpu_values = [m.cpu_utilization for m in recent_metrics]\n        \n        # Calculate trend (simple linear regression)\n        x = np.arange(len(cpu_values))\n        if len(cpu_values) > 1:\n            slope, intercept = np.polyfit(x, cpu_values, 1)\n            \n            # Predict for horizon_minutes from now\n            prediction_x = len(cpu_values) + (horizon_minutes / 5)  # Assuming 5-minute intervals\n            predicted_load = slope * prediction_x + intercept\n            \n            # Clamp prediction to reasonable range\n            return max(0, min(100, predicted_load))\n        else:\n            return cpu_values[0]