"""
Optimization package for neuromorphic gas detection system.
"""

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