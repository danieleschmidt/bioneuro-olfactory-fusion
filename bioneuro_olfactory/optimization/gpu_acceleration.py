"""
GPU/CUDA acceleration with fallback mechanisms for neuromorphic processing.

This module provides comprehensive GPU acceleration for neuromorphic computations
with intelligent fallback to CPU implementations and adaptive resource management.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
import cupy as cp
import logging
import time
import threading
import queue
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import psutil
import gc
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import warnings

logger = logging.getLogger(__name__)

# CUDA availability checks
CUDA_AVAILABLE = torch.cuda.is_available()
CUPY_AVAILABLE = False
try:
    import cupy as cp
    CUPY_AVAILABLE = cp.cuda.is_available()
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

# Multi-GPU support
NUM_GPUS = torch.cuda.device_count() if CUDA_AVAILABLE else 0
GPU_MEMORY_TOTAL = []
GPU_MEMORY_FREE = []

if CUDA_AVAILABLE:
    for i in range(NUM_GPUS):
        try:
            torch.cuda.set_device(i)
            memory_total = torch.cuda.get_device_properties(i).total_memory
            memory_free = torch.cuda.memory_reserved(i)
            GPU_MEMORY_TOTAL.append(memory_total)
            GPU_MEMORY_FREE.append(memory_free)
        except Exception as e:
            logger.warning(f"Error checking GPU {i} memory: {e}")


class GPUBackend(Enum):
    """Available GPU backends."""
    PYTORCH = "pytorch"
    CUPY = "cupy"
    AUTO = "auto"


class PrecisionMode(Enum):
    """Precision modes for GPU computation."""
    FP32 = "fp32"
    FP16 = "fp16"
    MIXED = "mixed"
    INT8 = "int8"


@dataclass
class GPUDeviceInfo:
    """Information about GPU device."""
    device_id: int
    name: str
    memory_total_gb: float
    memory_free_gb: float
    compute_capability: Tuple[int, int]
    multiprocessor_count: int
    max_threads_per_block: int
    max_block_dimensions: Tuple[int, int, int]
    max_grid_dimensions: Tuple[int, int, int]
    warp_size: int
    
    def is_suitable_for_task(self, required_memory_gb: float = 0.0,
                           min_compute_capability: Tuple[int, int] = (3, 5)) -> bool:
        """Check if device is suitable for given task."""
        memory_ok = self.memory_free_gb >= required_memory_gb
        compute_ok = (self.compute_capability[0] > min_compute_capability[0] or 
                     (self.compute_capability[0] == min_compute_capability[0] and 
                      self.compute_capability[1] >= min_compute_capability[1]))
        return memory_ok and compute_ok


@dataclass 
class GPUPerformanceMetrics:
    """GPU performance tracking."""
    kernel_launch_overhead_ms: float = 0.0
    memory_transfer_overhead_ms: float = 0.0
    compute_time_ms: float = 0.0
    gpu_utilization_percent: float = 0.0
    memory_utilization_percent: float = 0.0
    tensor_core_utilization_percent: float = 0.0
    power_consumption_watts: float = 0.0
    temperature_celsius: float = 0.0


class GPUResourceManager:
    """Manages GPU resources and memory allocation."""
    
    def __init__(self):
        self.device_info: Dict[int, GPUDeviceInfo] = {}
        self.memory_pools: Dict[int, torch.cuda.memory.CachingAllocator] = {}
        self.active_streams: Dict[int, List[torch.cuda.Stream]] = {}
        self.performance_history: deque = deque(maxlen=1000)
        
        # Initialize device information
        self._initialize_devices()
        
        # Memory management
        self.memory_fragmentation_threshold = 0.3
        self.auto_memory_cleanup = True
        
    def _initialize_devices(self):
        """Initialize GPU device information."""
        if not CUDA_AVAILABLE:
            return
            
        for device_id in range(NUM_GPUS):
            try:
                props = torch.cuda.get_device_properties(device_id)
                
                self.device_info[device_id] = GPUDeviceInfo(
                    device_id=device_id,
                    name=props.name,
                    memory_total_gb=props.total_memory / (1024**3),
                    memory_free_gb=torch.cuda.memory_reserved(device_id) / (1024**3),
                    compute_capability=(props.major, props.minor),
                    multiprocessor_count=props.multi_processor_count,
                    max_threads_per_block=props.max_threads_per_block,
                    max_block_dimensions=(props.max_block_dim_x, props.max_block_dim_y, props.max_block_dim_z),
                    max_grid_dimensions=(props.max_grid_dim_x, props.max_grid_dim_y, props.max_grid_dim_z),
                    warp_size=props.warp_size
                )
                
                # Initialize streams for this device
                self.active_streams[device_id] = [
                    torch.cuda.Stream(device=device_id) for _ in range(4)
                ]
                
            except Exception as e:
                logger.error(f"Failed to initialize GPU {device_id}: {e}")
    
    def select_best_device(self, required_memory_gb: float = 0.0,
                          prefer_memory_over_compute: bool = False) -> Optional[int]:
        """Select best available GPU device."""
        if not self.device_info:
            return None
        
        suitable_devices = [
            (device_id, info) for device_id, info in self.device_info.items()
            if info.is_suitable_for_task(required_memory_gb)
        ]
        
        if not suitable_devices:
            return None
        
        # Sort by preference
        if prefer_memory_over_compute:
            best_device = max(suitable_devices, key=lambda x: x[1].memory_free_gb)
        else:
            # Balance compute capability and available memory
            best_device = max(suitable_devices, 
                            key=lambda x: x[1].compute_capability[0] * 10 + x[1].memory_free_gb)
        
        return best_device[0]
    
    def get_memory_stats(self, device_id: int) -> Dict[str, Any]:
        """Get detailed memory statistics for device."""
        if not CUDA_AVAILABLE or device_id not in self.device_info:
            return {}
        
        torch.cuda.set_device(device_id)
        
        return {
            'allocated_gb': torch.cuda.memory_allocated(device_id) / (1024**3),
            'reserved_gb': torch.cuda.memory_reserved(device_id) / (1024**3),
            'max_allocated_gb': torch.cuda.max_memory_allocated(device_id) / (1024**3),
            'max_reserved_gb': torch.cuda.max_memory_reserved(device_id) / (1024**3),
            'total_gb': self.device_info[device_id].memory_total_gb,
            'fragmentation_ratio': self._calculate_fragmentation_ratio(device_id)
        }
    
    def _calculate_fragmentation_ratio(self, device_id: int) -> float:
        """Calculate memory fragmentation ratio."""
        allocated = torch.cuda.memory_allocated(device_id)
        reserved = torch.cuda.memory_reserved(device_id)
        
        if reserved == 0:
            return 0.0
        
        return 1.0 - (allocated / reserved)
    
    def cleanup_memory(self, device_id: Optional[int] = None, force: bool = False):
        """Clean up GPU memory."""
        if not CUDA_AVAILABLE:
            return
        
        devices_to_clean = [device_id] if device_id is not None else list(self.device_info.keys())
        
        for dev_id in devices_to_clean:
            if dev_id in self.device_info:
                torch.cuda.set_device(dev_id)
                
                # Check if cleanup is needed
                fragmentation = self._calculate_fragmentation_ratio(dev_id)
                if force or fragmentation > self.memory_fragmentation_threshold:
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    logger.info(f"Cleaned up GPU {dev_id} memory, fragmentation: {fragmentation:.2f}")
    
    def create_stream(self, device_id: int, priority: int = 0) -> torch.cuda.Stream:
        """Create new CUDA stream."""
        if device_id not in self.device_info:
            raise ValueError(f"Invalid device ID: {device_id}")
        
        stream = torch.cuda.Stream(device=device_id, priority=priority)
        self.active_streams[device_id].append(stream)
        
        return stream


class GPUAcceleratedNeurons(nn.Module):
    """GPU-accelerated LIF neurons with automatic fallback."""
    
    def __init__(self, num_neurons: int, batch_size: int = 1,
                 precision_mode: PrecisionMode = PrecisionMode.FP32,
                 device_id: Optional[int] = None,
                 enable_fallback: bool = True):
        super().__init__()
        
        self.num_neurons = num_neurons
        self.batch_size = batch_size
        self.precision_mode = precision_mode
        self.enable_fallback = enable_fallback
        
        # Device management
        self.gpu_manager = GPUResourceManager()
        self.device_id = device_id or self.gpu_manager.select_best_device()
        self.device = torch.device(f'cuda:{self.device_id}' if self.device_id is not None else 'cpu')
        
        # Neuron parameters
        self.tau_mem = nn.Parameter(torch.full((num_neurons,), 20.0))
        self.threshold = nn.Parameter(torch.full((num_neurons,), 1.0))
        self.reset_voltage = nn.Parameter(torch.full((num_neurons,), 0.0))
        self.refractory_period = torch.full((num_neurons,), 2, dtype=torch.int32)
        
        # State variables
        self.register_buffer('membrane_potential', torch.zeros(batch_size, num_neurons))
        self.register_buffer('refractory_counter', torch.zeros(batch_size, num_neurons, dtype=torch.int32))
        
        # Precomputed values
        self.register_buffer('decay_factor', torch.exp(-1.0 / self.tau_mem))
        
        # Move to device
        self.to(self.device)
        
        # Mixed precision scaler
        if precision_mode == PrecisionMode.MIXED:
            self.scaler = amp.GradScaler()
        else:
            self.scaler = None
        
        # Performance tracking
        self.performance_metrics = GPUPerformanceMetrics()
        self.fallback_count = 0
        
    def forward(self, input_current: torch.Tensor) -> torch.Tensor:
        """Forward pass with automatic fallback."""
        try:
            # Ensure input is on correct device
            if input_current.device != self.device:
                input_current = input_current.to(self.device)
            
            # Choose precision path
            if self.precision_mode == PrecisionMode.MIXED:
                return self._forward_mixed_precision(input_current)
            elif self.precision_mode == PrecisionMode.FP16:
                return self._forward_fp16(input_current.half())
            else:
                return self._forward_fp32(input_current)
                
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if self.enable_fallback:
                logger.warning(f"GPU computation failed, falling back to CPU: {e}")
                self.fallback_count += 1
                return self._fallback_to_cpu(input_current)
            else:
                raise
    
    def _forward_fp32(self, input_current: torch.Tensor) -> torch.Tensor:
        """Standard FP32 forward pass."""
        return self._lif_dynamics(input_current)
    
    def _forward_fp16(self, input_current: torch.Tensor) -> torch.Tensor:
        """FP16 forward pass."""
        # Convert states to FP16 temporarily
        with torch.cuda.amp.autocast(dtype=torch.float16):
            return self._lif_dynamics(input_current)
    
    def _forward_mixed_precision(self, input_current: torch.Tensor) -> torch.Tensor:
        """Mixed precision forward pass."""
        with torch.cuda.amp.autocast():
            return self._lif_dynamics(input_current)
    
    def _lif_dynamics(self, input_current: torch.Tensor) -> torch.Tensor:
        """Core LIF neuron dynamics."""
        start_time = time.perf_counter()
        
        # Update membrane potential (vectorized)
        self.membrane_potential = (
            self.decay_factor * self.membrane_potential + input_current
        )
        
        # Spike detection (not in refractory period)
        not_refractory = (self.refractory_counter == 0)
        spike_mask = (self.membrane_potential >= self.threshold) & not_refractory
        
        # Generate spikes
        spikes = spike_mask.float()
        
        # Reset spiking neurons
        reset_mask = spike_mask.unsqueeze(-1).expand_as(self.membrane_potential)
        self.membrane_potential = torch.where(
            reset_mask, 
            self.reset_voltage.expand_as(self.membrane_potential),
            self.membrane_potential
        )
        
        # Update refractory counters
        self.refractory_counter = torch.where(
            spike_mask,
            self.refractory_period.expand_as(self.refractory_counter),
            torch.clamp(self.refractory_counter - 1, min=0)
        )
        
        # Record performance
        self.performance_metrics.compute_time_ms = (time.perf_counter() - start_time) * 1000
        
        return spikes
    
    def _fallback_to_cpu(self, input_current: torch.Tensor) -> torch.Tensor:
        """Fallback to CPU computation."""
        # Move everything to CPU
        cpu_device = torch.device('cpu')
        
        input_cpu = input_current.cpu()
        membrane_cpu = self.membrane_potential.cpu()
        refractory_cpu = self.refractory_counter.cpu()
        
        # CPU computation
        decay_factor = torch.exp(-1.0 / self.tau_mem.cpu())
        
        membrane_cpu = decay_factor * membrane_cpu + input_cpu
        
        not_refractory = (refractory_cpu == 0)
        spike_mask = (membrane_cpu >= self.threshold.cpu()) & not_refractory
        spikes = spike_mask.float()
        
        # Reset and update on CPU
        membrane_cpu[spike_mask] = self.reset_voltage.cpu()[spike_mask]
        refractory_cpu[spike_mask] = self.refractory_period[spike_mask]
        refractory_cpu = torch.clamp(refractory_cpu - 1, min=0)
        
        # Update states (move back to GPU if possible)
        try:
            self.membrane_potential.copy_(membrane_cpu.to(self.device))
            self.refractory_counter.copy_(refractory_cpu.to(self.device))
            return spikes.to(self.device)
        except:
            # Stay on CPU
            self.membrane_potential = membrane_cpu
            self.refractory_counter = refractory_cpu
            return spikes
    
    def reset_states(self):
        """Reset neuron states."""
        self.membrane_potential.zero_()
        self.refractory_counter.zero_()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {
            'device': str(self.device),
            'precision_mode': self.precision_mode.value,
            'fallback_count': self.fallback_count,
            'compute_time_ms': self.performance_metrics.compute_time_ms
        }
        
        if self.device.type == 'cuda':
            memory_stats = self.gpu_manager.get_memory_stats(self.device.index)
            stats.update(memory_stats)
        
        return stats


class GPUAcceleratedSynapses(nn.Module):
    """GPU-accelerated synaptic computations."""
    
    def __init__(self, pre_neurons: int, post_neurons: int,
                 sparsity: float = 0.1,
                 precision_mode: PrecisionMode = PrecisionMode.FP32,
                 device_id: Optional[int] = None,
                 enable_sparse: bool = True):
        super().__init__()
        
        self.pre_neurons = pre_neurons
        self.post_neurons = post_neurons
        self.sparsity = sparsity
        self.precision_mode = precision_mode
        self.enable_sparse = enable_sparse
        
        # Device management
        self.gpu_manager = GPUResourceManager()
        self.device_id = device_id or self.gpu_manager.select_best_device()
        self.device = torch.device(f'cuda:{self.device_id}' if self.device_id is not None else 'cpu')
        
        # Weight initialization
        if enable_sparse:
            self._initialize_sparse_weights()
        else:
            self.register_parameter('weights', nn.Parameter(
                torch.randn(post_neurons, pre_neurons) * 0.1
            ))
        
        # Synaptic dynamics parameters
        self.register_parameter('tau_syn', nn.Parameter(torch.full((post_neurons,), 5.0)))
        self.register_buffer('syn_decay', torch.exp(-1.0 / self.tau_syn))
        
        # State variables
        self.register_buffer('synaptic_current', torch.zeros(post_neurons))
        
        # Move to device
        self.to(self.device)
    
    def _initialize_sparse_weights(self):
        """Initialize sparse weight matrix."""
        num_connections = int(self.pre_neurons * self.post_neurons * self.sparsity)
        
        # Generate random connections
        pre_indices = torch.randint(0, self.pre_neurons, (num_connections,))
        post_indices = torch.randint(0, self.post_neurons, (num_connections,))
        values = torch.randn(num_connections) * 0.1
        
        # Create sparse tensor
        indices = torch.stack([post_indices, pre_indices])
        self.sparse_weights = torch.sparse_coo_tensor(
            indices, values, (self.post_neurons, self.pre_neurons)
        ).coalesce()
        
        # Register as buffer
        self.register_buffer('weights_indices', self.sparse_weights.indices())
        self.register_buffer('weights_values', self.sparse_weights.values())
    
    def forward(self, input_spikes: torch.Tensor) -> torch.Tensor:
        """Forward pass through synapses."""
        try:
            if input_spikes.device != self.device:
                input_spikes = input_spikes.to(self.device)
            
            # Synaptic decay
            self.synaptic_current *= self.syn_decay
            
            # Add new contributions
            if self.enable_sparse:
                # Sparse matrix multiplication
                sparse_weights = torch.sparse_coo_tensor(
                    self.weights_indices, self.weights_values,
                    (self.post_neurons, self.pre_neurons)
                )
                contribution = torch.sparse.mm(sparse_weights, input_spikes.unsqueeze(-1)).squeeze()
            else:
                # Dense matrix multiplication
                contribution = torch.matmul(self.weights, input_spikes)
            
            self.synaptic_current += contribution
            
            return self.synaptic_current
            
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            logger.warning(f"GPU synapse computation failed, using CPU fallback: {e}")
            return self._fallback_synapse_computation(input_spikes)
    
    def _fallback_synapse_computation(self, input_spikes: torch.Tensor) -> torch.Tensor:
        """CPU fallback for synapse computation."""
        input_cpu = input_spikes.cpu()
        current_cpu = self.synaptic_current.cpu()
        decay_cpu = self.syn_decay.cpu()
        
        # CPU computation
        current_cpu *= decay_cpu
        
        if self.enable_sparse:
            weights_cpu = torch.sparse_coo_tensor(
                self.weights_indices.cpu(), self.weights_values.cpu(),
                (self.post_neurons, self.pre_neurons)
            )
            contribution = torch.sparse.mm(weights_cpu, input_cpu.unsqueeze(-1)).squeeze()
        else:
            contribution = torch.matmul(self.weights.cpu(), input_cpu)
        
        current_cpu += contribution
        
        # Update state
        try:
            self.synaptic_current.copy_(current_cpu.to(self.device))
            return self.synaptic_current
        except:
            self.synaptic_current = current_cpu
            return current_cpu


class MultiGPUNeuromorphicNetwork(nn.Module):
    """Multi-GPU neuromorphic network with load balancing."""
    
    def __init__(self, layer_configs: List[Dict[str, Any]],
                 enable_data_parallel: bool = True,
                 enable_model_parallel: bool = False):
        super().__init__()
        
        self.layer_configs = layer_configs
        self.enable_data_parallel = enable_data_parallel
        self.enable_model_parallel = enable_model_parallel
        
        # GPU resource management
        self.gpu_manager = GPUResourceManager()
        self.available_devices = list(self.gpu_manager.device_info.keys())
        
        # Layer distribution across GPUs
        self.layers = nn.ModuleDict()
        self.layer_devices: Dict[str, int] = {}
        
        self._distribute_layers()
        
        # Data parallel wrapper if enabled
        if enable_data_parallel and len(self.available_devices) > 1:
            self._setup_data_parallel()
    
    def _distribute_layers(self):
        """Distribute layers across available GPUs."""
        if not self.available_devices:
            # CPU fallback
            device = torch.device('cpu')
            for i, config in enumerate(self.layer_configs):
                layer_name = f"layer_{i}"
                if config['type'] == 'neurons':
                    layer = GPUAcceleratedNeurons(
                        config['num_neurons'], 
                        device_id=None,
                        precision_mode=PrecisionMode(config.get('precision', 'fp32'))
                    )
                elif config['type'] == 'synapses':
                    layer = GPUAcceleratedSynapses(
                        config['pre_neurons'],
                        config['post_neurons'],
                        device_id=None,
                        precision_mode=PrecisionMode(config.get('precision', 'fp32'))
                    )
                
                self.layers[layer_name] = layer
                self.layer_devices[layer_name] = -1  # CPU
            
            return
        
        # Distribute across available GPUs
        for i, config in enumerate(self.layer_configs):
            layer_name = f"layer_{i}"
            device_id = self.available_devices[i % len(self.available_devices)]
            
            if config['type'] == 'neurons':
                layer = GPUAcceleratedNeurons(
                    config['num_neurons'],
                    device_id=device_id,
                    precision_mode=PrecisionMode(config.get('precision', 'fp32'))
                )
            elif config['type'] == 'synapses':
                layer = GPUAcceleratedSynapses(
                    config['pre_neurons'],
                    config['post_neurons'],
                    device_id=device_id,
                    precision_mode=PrecisionMode(config.get('precision', 'fp32'))
                )
            
            self.layers[layer_name] = layer
            self.layer_devices[layer_name] = device_id
    
    def _setup_data_parallel(self):
        """Setup data parallel processing."""
        for layer_name, layer in self.layers.items():
            if hasattr(layer, 'device') and layer.device.type == 'cuda':
                # Wrap in DataParallel
                self.layers[layer_name] = nn.DataParallel(
                    layer, device_ids=self.available_devices
                )
    
    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Forward pass through multi-GPU network."""
        x = input_data
        
        for layer_name in sorted(self.layers.keys()):
            layer = self.layers[layer_name]
            device_id = self.layer_devices[layer_name]
            
            # Move data to correct device
            if device_id >= 0:
                target_device = torch.device(f'cuda:{device_id}')
                if x.device != target_device:
                    x = x.to(target_device)
            
            # Forward through layer
            x = layer(x)
        
        return x
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get comprehensive network statistics."""
        stats = {
            'num_layers': len(self.layers),
            'num_devices': len(self.available_devices),
            'data_parallel': self.enable_data_parallel,
            'model_parallel': self.enable_model_parallel,
            'layer_distribution': self.layer_devices
        }
        
        # Layer-specific stats
        layer_stats = {}
        for layer_name, layer in self.layers.items():
            if hasattr(layer, 'get_performance_stats'):
                layer_stats[layer_name] = layer.get_performance_stats()
        
        stats['layer_performance'] = layer_stats
        
        # GPU memory usage
        gpu_stats = {}
        for device_id in self.available_devices:
            gpu_stats[f'gpu_{device_id}'] = self.gpu_manager.get_memory_stats(device_id)
        
        stats['gpu_memory'] = gpu_stats
        
        return stats


class GPUFallbackManager:
    """Manages GPU computation with intelligent fallback strategies."""
    
    def __init__(self, fallback_threshold_ms: float = 1000.0,
                 memory_threshold_ratio: float = 0.9):
        self.fallback_threshold_ms = fallback_threshold_ms
        self.memory_threshold_ratio = memory_threshold_ratio
        
        self.gpu_manager = GPUResourceManager()
        self.performance_history: deque = deque(maxlen=100)
        self.fallback_history: deque = deque(maxlen=100)
        
        # Fallback strategies
        self.strategies = {
            'reduce_precision': self._reduce_precision_fallback,
            'reduce_batch_size': self._reduce_batch_size_fallback,
            'use_cpu': self._cpu_fallback,
            'use_different_gpu': self._different_gpu_fallback
        }
    
    def execute_with_fallback(self, gpu_fn: Callable, cpu_fn: Callable, 
                             *args, **kwargs) -> Any:
        """Execute function with automatic fallback."""
        start_time = time.perf_counter()
        
        try:
            # Check GPU availability
            if not self._is_gpu_available():
                logger.info("GPU not available, using CPU fallback")
                return cpu_fn(*args, **kwargs)
            
            # Try GPU computation
            result = gpu_fn(*args, **kwargs)
            
            # Record successful GPU execution
            execution_time = (time.perf_counter() - start_time) * 1000
            self.performance_history.append({
                'execution_time_ms': execution_time,
                'backend': 'gpu',
                'success': True
            })
            
            return result
            
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            logger.warning(f"GPU computation failed: {e}")
            
            # Try fallback strategies
            for strategy_name, strategy_fn in self.strategies.items():
                try:
                    result = strategy_fn(gpu_fn, cpu_fn, *args, **kwargs)
                    
                    execution_time = (time.perf_counter() - start_time) * 1000
                    self.fallback_history.append({
                        'strategy': strategy_name,
                        'execution_time_ms': execution_time,
                        'success': True
                    })
                    
                    return result
                    
                except Exception as fallback_error:
                    logger.warning(f"Fallback strategy {strategy_name} failed: {fallback_error}")
                    continue
            
            # All strategies failed, use CPU
            logger.error("All GPU fallback strategies failed, using CPU")
            result = cpu_fn(*args, **kwargs)
            
            execution_time = (time.perf_counter() - start_time) * 1000
            self.fallback_history.append({
                'strategy': 'cpu_final',
                'execution_time_ms': execution_time,
                'success': True
            })
            
            return result
    
    def _is_gpu_available(self) -> bool:
        """Check if GPU is available and has sufficient resources."""
        if not CUDA_AVAILABLE:
            return False
        
        best_device = self.gpu_manager.select_best_device()
        if best_device is None:
            return False
        
        # Check memory availability
        memory_stats = self.gpu_manager.get_memory_stats(best_device)
        memory_ratio = (memory_stats['allocated_gb'] / memory_stats['total_gb'])
        
        return memory_ratio < self.memory_threshold_ratio
    
    def _reduce_precision_fallback(self, gpu_fn: Callable, cpu_fn: Callable,
                                  *args, **kwargs) -> Any:
        """Try computation with reduced precision."""
        # This would modify the computation to use FP16 instead of FP32
        logger.info("Attempting reduced precision fallback")
        return gpu_fn(*args, **kwargs)
    
    def _reduce_batch_size_fallback(self, gpu_fn: Callable, cpu_fn: Callable,
                                   *args, **kwargs) -> Any:
        """Try computation with reduced batch size."""
        logger.info("Attempting reduced batch size fallback")
        
        # This would split the computation into smaller batches
        # For now, just try the original function
        return gpu_fn(*args, **kwargs)
    
    def _cpu_fallback(self, gpu_fn: Callable, cpu_fn: Callable,
                     *args, **kwargs) -> Any:
        """Use CPU computation."""
        logger.info("Using CPU fallback")
        return cpu_fn(*args, **kwargs)
    
    def _different_gpu_fallback(self, gpu_fn: Callable, cpu_fn: Callable,
                               *args, **kwargs) -> Any:
        """Try computation on different GPU."""
        logger.info("Attempting different GPU fallback")
        
        # This would try to move computation to a different GPU
        # For now, just try the original function
        return gpu_fn(*args, **kwargs)
    
    def get_fallback_stats(self) -> Dict[str, Any]:
        """Get fallback statistics."""
        total_executions = len(self.performance_history) + len(self.fallback_history)
        gpu_successes = len([h for h in self.performance_history if h['success']])
        fallback_count = len(self.fallback_history)
        
        fallback_strategies = {}
        for entry in self.fallback_history:
            strategy = entry['strategy']
            if strategy not in fallback_strategies:
                fallback_strategies[strategy] = 0
            fallback_strategies[strategy] += 1
        
        return {
            'total_executions': total_executions,
            'gpu_success_rate': gpu_successes / max(total_executions, 1),
            'fallback_rate': fallback_count / max(total_executions, 1),
            'fallback_strategies_used': fallback_strategies,
            'avg_gpu_execution_time_ms': np.mean([h['execution_time_ms'] for h in self.performance_history]),
            'avg_fallback_execution_time_ms': np.mean([h['execution_time_ms'] for h in self.fallback_history]) if self.fallback_history else 0
        }


# Global instances
_gpu_manager: Optional[GPUResourceManager] = None
_fallback_manager: Optional[GPUFallbackManager] = None


def get_gpu_manager() -> GPUResourceManager:
    """Get global GPU resource manager."""
    global _gpu_manager
    
    if _gpu_manager is None:
        _gpu_manager = GPUResourceManager()
    
    return _gpu_manager


def get_fallback_manager() -> GPUFallbackManager:
    """Get global GPU fallback manager."""
    global _fallback_manager
    
    if _fallback_manager is None:
        _fallback_manager = GPUFallbackManager()
    
    return _fallback_manager


# Utility functions
def create_gpu_accelerated_network(layer_configs: List[Dict[str, Any]]) -> MultiGPUNeuromorphicNetwork:
    """Create GPU-accelerated neuromorphic network."""
    return MultiGPUNeuromorphicNetwork(layer_configs)


def gpu_memory_summary() -> Dict[str, Any]:
    """Get comprehensive GPU memory summary."""
    manager = get_gpu_manager()
    
    summary = {
        'num_gpus': NUM_GPUS,
        'cuda_available': CUDA_AVAILABLE,
        'cupy_available': CUPY_AVAILABLE,
        'devices': {}
    }
    
    for device_id, device_info in manager.device_info.items():
        memory_stats = manager.get_memory_stats(device_id)
        
        summary['devices'][device_id] = {
            'name': device_info.name,
            'compute_capability': device_info.compute_capability,
            'memory_total_gb': device_info.memory_total_gb,
            'memory_stats': memory_stats
        }
    
    return summary