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

from .advanced_caching import (
    MultiLevelCache as AdvancedMultiLevelCache,
    CacheLevel,
    EvictionPolicy,
    cached as advanced_cached,
    get_cache as get_advanced_cache
)

from .enhanced_monitoring import (
    RealTimeMonitor,
    MetricCollector,
    AlertManager,
    AnomalyDetector,
    MonitoringConfig,
    AlertSeverity,
    MetricType,
    get_monitor,
    monitoring_context,
    record_metric,
    monitor_performance
)

# Generation 3 Neuromorphic-Specific Scaling Features
from .neuromorphic_cache import (
    NeuromorphicCache,
    WeightMatrixCache,
    CacheType,
    SpikePattern,
    get_neuromorphic_cache,
    get_weight_cache,
    cache_neuromorphic_computation
)

from .sensor_connection_pool import (
    SensorConnectionPool,
    SensorConnection,
    ConnectionConfig,
    ConnectionType,
    SensorDataType,
    SensorReading,
    AdaptiveBuffer,
    get_sensor_pool,
    sensor_pool_context
)

from .spike_io import (
    MappedSpikeFile,
    SpikeDatasetManager,
    CompressionType,
    SpikeDataFormat,
    SpikeDataCompressor,
    get_spike_dataset_manager
)

from .vectorized_ops import (
    VectorizedLIFNeurons,
    VectorizedSynapses,
    VectorizedNeuromorphicProcessor,
    OptimizationLevel,
    AdaptiveOptimizer,
    get_vectorized_processor,
    vectorized_lif_update,
    vectorized_spike_encoding
)

from .gpu_acceleration import (
    GPUAcceleratedNeurons,
    GPUAcceleratedSynapses,
    MultiGPUNeuromorphicNetwork,
    GPUResourceManager,
    GPUFallbackManager,
    PrecisionMode,
    get_gpu_manager,
    get_fallback_manager,
    create_gpu_accelerated_network,
    gpu_memory_summary
)

from .spike_pipeline import (
    ConcurrentSpikeProcessor,
    SpikeTask,
    Priority,
    PipelineStage,
    get_spike_processor
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
    
    # Advanced caching
    'AdvancedMultiLevelCache',
    'CacheLevel',
    'EvictionPolicy', 
    'advanced_cached',
    'get_advanced_cache',
    
    # Enhanced monitoring
    'RealTimeMonitor',
    'MetricCollector',
    'AlertManager',
    'AnomalyDetector',
    'MonitoringConfig',
    'AlertSeverity',
    'MetricType',
    'get_monitor',
    'monitoring_context',
    'record_metric',
    'monitor_performance',
    
    # Generation 3 Neuromorphic Scaling Features
    'NeuromorphicCache',
    'WeightMatrixCache',
    'CacheType',
    'SpikePattern',
    'get_neuromorphic_cache',
    'get_weight_cache',
    'cache_neuromorphic_computation',
    'SensorConnectionPool',
    'SensorConnection',
    'ConnectionConfig',
    'ConnectionType',
    'SensorDataType',
    'SensorReading',
    'AdaptiveBuffer',
    'get_sensor_pool',
    'sensor_pool_context',
    'MappedSpikeFile',
    'SpikeDatasetManager',
    'CompressionType',
    'SpikeDataFormat',
    'SpikeDataCompressor',
    'get_spike_dataset_manager',
    'VectorizedLIFNeurons',
    'VectorizedSynapses',
    'VectorizedNeuromorphicProcessor',
    'OptimizationLevel',
    'AdaptiveOptimizer',
    'get_vectorized_processor',
    'vectorized_lif_update',
    'vectorized_spike_encoding',
    'GPUAcceleratedNeurons',
    'GPUAcceleratedSynapses',
    'MultiGPUNeuromorphicNetwork',
    'GPUResourceManager',
    'GPUFallbackManager',
    'PrecisionMode',
    'get_gpu_manager',
    'get_fallback_manager',
    'create_gpu_accelerated_network',
    'gpu_memory_summary',
    'ConcurrentSpikeProcessor',
    'SpikeTask',
    'Priority',
    'PipelineStage',
    'get_spike_processor',
    
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