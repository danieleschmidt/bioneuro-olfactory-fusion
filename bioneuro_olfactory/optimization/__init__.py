"""
Optimization package for neuromorphic gas detection system.
"""

# Generation 3 optimization systems
from .performance_profiler import (
    PerformanceProfiler,
    PerformanceMetric,
    profile,
    profile_context,
    get_profiler
)

from .distributed_processing import (
    DistributedProcessor,
    ProcessingMode,
    TaskScheduler,
    WorkerNode,
    get_distributed_processor
)

from .adaptive_caching import (
    SmartCache,
    MultiLevelCache,
    CachePolicy,
    cached,
    get_cache as get_smart_cache,
    get_multilevel_cache
)

# Existing optimization systems
from .performance_optimizer import (
    PerformanceOptimizer,
    OptimizationConfig,
    PerformanceMetrics,
    ModelOptimizer,
    MemoryOptimizer,
    ParallelProcessingManager,
    AutoTuner,
    performance_optimizer
)
from .neuromorphic_accelerator import (
    NeuromorphicAcceleratorManager,
    NeuromorphicPlatform,
    NeuromorphicConfig,
    SpikeData,
    LoihiAccelerator,
    BrainScaleSAccelerator,
    SpiNNakerAccelerator,
    SoftwareSimulator,
    neuromorphic_manager
)

__all__ = [
    # Generation 3 optimization
    'PerformanceProfiler',
    'PerformanceMetric', 
    'profile',
    'profile_context',
    'get_profiler',
    'DistributedProcessor',
    'ProcessingMode',
    'TaskScheduler',
    'WorkerNode',
    'get_distributed_processor',
    'SmartCache',
    'MultiLevelCache',
    'CachePolicy',
    'cached',
    'get_smart_cache',
    'get_multilevel_cache',
    
    # Existing optimization
    'PerformanceOptimizer',
    'OptimizationConfig',
    'PerformanceMetrics',
    'ModelOptimizer',
    'MemoryOptimizer',
    'ParallelProcessingManager',
    'AutoTuner',
    'performance_optimizer',
    'NeuromorphicAcceleratorManager',
    'NeuromorphicPlatform',
    'NeuromorphicConfig',
    'SpikeData',
    'LoihiAccelerator',
    'BrainScaleSAccelerator',
    'SpiNNakerAccelerator',
    'SoftwareSimulator',
    'neuromorphic_manager'
]