"""Performance integration demonstration.

This module demonstrates the integration of all performance optimization
components working together in a real-world scenario.
"""

import asyncio
import time
import torch
import numpy as np
from typing import Dict, Any, List
import logging

from .performance_optimizer import PerformanceOptimizer
from .concurrent_processing import ConcurrentProcessor
from .advanced_caching import MultiLevelCache
from .load_balancer import LoadBalancer, WorkerNode
from .enhanced_monitoring import EnhancedMonitoringSystem

logger = logging.getLogger(__name__)


class IntegratedPerformanceSystem:
    """Integrated performance optimization system."""
    
    def __init__(self):
        self.optimizer = PerformanceOptimizer()
        self.processor = ConcurrentProcessor()
        self.cache = MultiLevelCache()
        self.load_balancer = LoadBalancer()
        self.monitor = EnhancedMonitoringSystem()
        
        # Add some sample worker nodes
        self._setup_workers()
        
    def _setup_workers(self):
        """Setup sample worker nodes."""
        for i in range(3):
            worker = WorkerNode(
                node_id=f"worker_{i}",
                address=f"192.168.1.{100+i}",
                port=8080,
                capacity=10
            )
            self.load_balancer.add_worker(worker)
            
    async def process_batch(self, data_batch: List[torch.Tensor]) -> List[Any]:
        """Process batch with full optimization pipeline."""
        start_time = time.time()
        
        try:
            # 1. Check cache first
            cache_key = f"batch_{hash(str(data_batch))}"
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                logger.info("Cache hit - returning cached result")
                return cached_result
                
            # 2. Select optimal worker
            worker = self.load_balancer.select_worker()
            if not worker:
                raise RuntimeError("No healthy workers available")
                
            # 3. Process with optimization
            def process_item(item):
                return torch.sum(item).item()
                
            # Use concurrent processing
            future = self.processor.submit_task(
                function=lambda: [process_item(item) for item in data_batch],
                priority=1
            )
            
            results = future.result(timeout=30.0)
            
            # 4. Cache results
            self.cache.put(cache_key, results, ttl=300)
            
            # 5. Record metrics
            processing_time = time.time() - start_time
            await self.monitor.record_metric("batch_processing_time", processing_time)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            raise
            
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        return {
            'processor_stats': self.processor.get_stats(),
            'cache_stats': self.cache.get_stats(),
            'load_balancer_stats': self.load_balancer.get_stats(),
            'monitor_stats': self.monitor.get_current_metrics()
        }
        
    def shutdown(self):
        """Shutdown all systems."""
        self.processor.shutdown()
        self.monitor.stop()


async def demo_integrated_system():
    """Demonstrate integrated performance system."""
    system = IntegratedPerformanceSystem()
    
    try:
        # Generate sample data
        data_batch = [torch.randn(100, 100) for _ in range(10)]
        
        # Process batch
        results = await system.process_batch(data_batch)
        
        print(f"Processed {len(results)} items")
        print(f"Sample result: {results[0]}")
        
        # Show system stats
        stats = system.get_system_stats()
        print(f"System stats: {stats}")
        
    finally:
        system.shutdown()


if __name__ == "__main__":
    asyncio.run(demo_integrated_system())