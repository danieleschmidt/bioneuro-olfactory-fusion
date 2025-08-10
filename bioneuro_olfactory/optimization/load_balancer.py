"""Load balancing and auto-scaling for distributed neuromorphic processing.

This module implements intelligent load balancing, circuit breaker patterns,
and auto-scaling capabilities optimized for spiking neural networks.
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
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, requests blocked
    HALF_OPEN = "half_open" # Testing recovery


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, timeout_seconds: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        
    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time > self.timeout_seconds:
                self.state = CircuitBreakerState.HALF_OPEN
                return True
            return False
        else:  # HALF_OPEN
            return True
            
    def record_success(self):
        """Record successful execution."""
        self.failure_count = 0
        self.state = CircuitBreakerState.CLOSED
        
    def record_failure(self):
        """Record failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN


@dataclass
class WorkerNode:
    """Worker node definition."""
    node_id: str
    address: str
    port: int
    capacity: int
    current_load: int = 0
    health_score: float = 1.0
    last_heartbeat: float = field(default_factory=time.time)
    circuit_breaker: CircuitBreaker = field(default_factory=CircuitBreaker)
    
    @property
    def load_factor(self) -> float:
        """Calculate current load factor."""
        return self.current_load / max(self.capacity, 1)
        
    @property
    def is_healthy(self) -> bool:
        """Check if node is healthy."""
        return (time.time() - self.last_heartbeat < 30.0 and 
                self.health_score > 0.5 and
                self.circuit_breaker.can_execute())


class LoadBalancer:
    """Intelligent load balancer."""
    
    def __init__(self):
        self.workers: Dict[str, WorkerNode] = {}
        self._lock = threading.RLock()
        
    def add_worker(self, worker: WorkerNode):
        """Add worker node."""
        with self._lock:
            self.workers[worker.node_id] = worker
            
    def remove_worker(self, node_id: str):
        """Remove worker node."""
        with self._lock:
            if node_id in self.workers:
                del self.workers[node_id]
                
    def select_worker(self) -> Optional[WorkerNode]:
        """Select best available worker."""
        with self._lock:
            healthy_workers = [w for w in self.workers.values() if w.is_healthy]
            
            if not healthy_workers:
                return None
                
            # Select worker with lowest load factor
            return min(healthy_workers, key=lambda w: w.load_factor)
            
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        with self._lock:
            healthy_count = sum(1 for w in self.workers.values() if w.is_healthy)
            total_capacity = sum(w.capacity for w in self.workers.values())
            total_load = sum(w.current_load for w in self.workers.values())
            
            return {
                'total_workers': len(self.workers),
                'healthy_workers': healthy_count,
                'total_capacity': total_capacity,
                'total_load': total_load,
                'load_factor': total_load / max(total_capacity, 1),
                'workers': {
                    w.node_id: {
                        'load_factor': w.load_factor,
                        'health_score': w.health_score,
                        'circuit_state': w.circuit_breaker.state.value
                    } for w in self.workers.values()
                }
            }


def parallel_map(func: Callable, iterable, max_workers: int = None) -> List[Any]:
    """Simple parallel map implementation."""
    from concurrent.futures import ThreadPoolExecutor
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(func, iterable))