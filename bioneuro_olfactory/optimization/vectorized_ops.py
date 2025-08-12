"""
Vectorized operations for neural computations with SIMD optimizations.

This module provides high-performance vectorized implementations of neuromorphic
computations optimized for modern SIMD instruction sets (AVX2, AVX-512).
"""

import numpy as np
import torch
import numba
from numba import cuda, jit, prange
import logging
import time
import threading
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import psutil
import math
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

logger = logging.getLogger(__name__)

# Check for CUDA availability
CUDA_AVAILABLE = cuda.is_available() if hasattr(cuda, 'is_available') else False
TORCH_CUDA_AVAILABLE = torch.cuda.is_available()

# Detect CPU features
try:
    import cpuinfo
    CPU_INFO = cpuinfo.get_cpu_info()
    HAS_AVX2 = 'avx2' in CPU_INFO.get('flags', [])
    HAS_AVX512 = 'avx512f' in CPU_INFO.get('flags', [])
    NUM_CORES = psutil.cpu_count(logical=False)
    NUM_THREADS = psutil.cpu_count(logical=True)
except ImportError:
    CPU_INFO = {}
    HAS_AVX2 = True  # Assume modern CPU
    HAS_AVX512 = False
    NUM_CORES = mp.cpu_count()
    NUM_THREADS = mp.cpu_count()


class OptimizationLevel(Enum):
    """Optimization levels for vectorized operations."""
    BASIC = "basic"           # Basic NumPy operations
    SIMD = "simd"            # SIMD-optimized operations
    PARALLEL = "parallel"    # Multi-threaded operations
    GPU = "gpu"             # GPU-accelerated operations
    ADAPTIVE = "adaptive"   # Adaptive optimization based on data size


@dataclass
class ComputeProfile:
    """Profile for compute operations."""
    operation_name: str
    data_shape: Tuple[int, ...]
    optimization_level: OptimizationLevel
    execution_time_ms: float
    memory_usage_mb: float
    throughput_ops_per_sec: float
    cpu_utilization: float
    gpu_utilization: float = 0.0
    
    def get_performance_score(self) -> float:
        """Calculate performance score (higher is better)."""
        # Weighted score based on throughput and resource efficiency
        throughput_score = self.throughput_ops_per_sec / 1e6  # Normalize to millions
        
        # Penalize high resource usage
        cpu_penalty = max(0, (self.cpu_utilization - 80) / 20)  # Penalty above 80%
        memory_penalty = max(0, (self.memory_usage_mb - 1000) / 1000)  # Penalty above 1GB
        
        return throughput_score - cpu_penalty * 0.1 - memory_penalty * 0.05


class VectorizedLIFNeurons:
    """Vectorized Leaky Integrate-and-Fire neuron computations."""
    
    def __init__(self, num_neurons: int, optimization_level: OptimizationLevel = OptimizationLevel.ADAPTIVE):
        self.num_neurons = num_neurons
        self.optimization_level = optimization_level
        
        # Neuron parameters (vectorized)
        self.tau_mem = np.full(num_neurons, 20.0, dtype=np.float32)
        self.threshold = np.full(num_neurons, 1.0, dtype=np.float32) 
        self.reset_voltage = np.full(num_neurons, 0.0, dtype=np.float32)
        self.refractory_period = np.full(num_neurons, 2, dtype=np.int32)
        
        # State variables
        self.membrane_potential = np.zeros(num_neurons, dtype=np.float32)
        self.refractory_counter = np.zeros(num_neurons, dtype=np.int32)
        
        # Precomputed decay factors
        self.decay_factor = np.exp(-1.0 / self.tau_mem).astype(np.float32)
        
        # GPU arrays if available
        if TORCH_CUDA_AVAILABLE and optimization_level == OptimizationLevel.GPU:
            self._setup_gpu_arrays()
        
        # Performance tracking
        self.compute_profiles: List[ComputeProfile] = []
    
    def _setup_gpu_arrays(self):
        """Setup GPU arrays for CUDA computation."""
        self.gpu_membrane_potential = torch.zeros(
            self.num_neurons, dtype=torch.float32, device='cuda'
        )
        self.gpu_refractory_counter = torch.zeros(
            self.num_neurons, dtype=torch.int32, device='cuda'
        )
        self.gpu_decay_factor = torch.tensor(
            self.decay_factor, dtype=torch.float32, device='cuda'
        )
        self.gpu_threshold = torch.tensor(
            self.threshold, dtype=torch.float32, device='cuda'
        )
        self.gpu_reset_voltage = torch.tensor(
            self.reset_voltage, dtype=torch.float32, device='cuda'
        )
    
    @numba.jit(nopython=True, parallel=True, fastmath=True)
    def _update_neurons_simd(membrane_potential, refractory_counter, input_current,
                            decay_factor, threshold, reset_voltage, refractory_period):
        """SIMD-optimized neuron update using Numba."""
        num_neurons = len(membrane_potential)
        spikes = np.zeros(num_neurons, dtype=np.float32)
        
        # Parallel loop with SIMD vectorization
        for i in prange(num_neurons):
            # Skip if in refractory period
            if refractory_counter[i] > 0:
                refractory_counter[i] -= 1
                continue
            
            # Update membrane potential (leaky integration)
            membrane_potential[i] = (
                decay_factor[i] * membrane_potential[i] + input_current[i]
            )
            
            # Check for spike
            if membrane_potential[i] >= threshold[i]:
                spikes[i] = 1.0
                membrane_potential[i] = reset_voltage[i]
                refractory_counter[i] = refractory_period[i]
        
        return spikes
    
    def update(self, input_current: np.ndarray) -> np.ndarray:
        """Update neurons with optimized computation."""
        start_time = time.perf_counter()
        
        if self.optimization_level == OptimizationLevel.GPU and TORCH_CUDA_AVAILABLE:
            spikes = self._update_gpu(input_current)
        elif self.optimization_level in [OptimizationLevel.SIMD, OptimizationLevel.PARALLEL]:
            spikes = self._update_neurons_simd(
                self.membrane_potential, self.refractory_counter, input_current,
                self.decay_factor, self.threshold, self.reset_voltage, self.refractory_period
            )
        else:
            spikes = self._update_basic(input_current)
        
        # Record performance
        execution_time = (time.perf_counter() - start_time) * 1000
        self._record_performance("neuron_update", execution_time, input_current.shape)
        
        return spikes
    
    def _update_basic(self, input_current: np.ndarray) -> np.ndarray:
        """Basic NumPy implementation."""
        # Update membrane potential
        self.membrane_potential = (
            self.decay_factor * self.membrane_potential + input_current
        )
        
        # Check for spikes (not in refractory period)
        not_refractory = self.refractory_counter == 0
        spike_mask = (self.membrane_potential >= self.threshold) & not_refractory
        
        # Generate spikes
        spikes = spike_mask.astype(np.float32)
        
        # Reset spiking neurons
        self.membrane_potential[spike_mask] = self.reset_voltage[spike_mask]
        self.refractory_counter[spike_mask] = self.refractory_period[spike_mask]
        
        # Decrement refractory counters
        self.refractory_counter = np.maximum(0, self.refractory_counter - 1)
        
        return spikes
    
    def _update_gpu(self, input_current: np.ndarray) -> np.ndarray:
        """GPU-accelerated implementation using PyTorch."""
        # Transfer input to GPU
        gpu_input = torch.tensor(input_current, dtype=torch.float32, device='cuda')
        
        # Update membrane potential
        self.gpu_membrane_potential = (
            self.gpu_decay_factor * self.gpu_membrane_potential + gpu_input
        )
        
        # Check for spikes
        not_refractory = self.gpu_refractory_counter == 0
        spike_mask = (self.gpu_membrane_potential >= self.gpu_threshold) & not_refractory
        
        # Generate spikes
        spikes = spike_mask.float()
        
        # Reset spiking neurons
        self.gpu_membrane_potential[spike_mask] = self.gpu_reset_voltage[spike_mask]
        self.gpu_refractory_counter[spike_mask] = self.refractory_period[0]  # Simplified
        
        # Decrement refractory counters
        self.gpu_refractory_counter = torch.clamp(self.gpu_refractory_counter - 1, 0)
        
        # Transfer back to CPU
        return spikes.cpu().numpy()
    
    def _record_performance(self, operation: str, execution_time_ms: float, data_shape: Tuple):
        """Record performance metrics."""
        throughput = (self.num_neurons / (execution_time_ms / 1000.0))  # ops/sec
        
        profile = ComputeProfile(
            operation_name=operation,
            data_shape=data_shape,
            optimization_level=self.optimization_level,
            execution_time_ms=execution_time_ms,
            memory_usage_mb=self._estimate_memory_usage(),
            throughput_ops_per_sec=throughput,
            cpu_utilization=psutil.cpu_percent(interval=None)
        )
        
        self.compute_profiles.append(profile)
        
        # Keep only recent profiles
        if len(self.compute_profiles) > 100:
            self.compute_profiles.pop(0)
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        base_memory = self.num_neurons * 4 * 6  # 6 float32 arrays
        return base_memory / (1024 * 1024)


class VectorizedSynapses:
    """Vectorized synaptic computations with SIMD optimizations."""
    
    def __init__(self, pre_neurons: int, post_neurons: int, 
                 optimization_level: OptimizationLevel = OptimizationLevel.ADAPTIVE):
        self.pre_neurons = pre_neurons
        self.post_neurons = post_neurons
        self.optimization_level = optimization_level
        
        # Weight matrix (sparse representation for large networks)
        self.weights = np.random.randn(post_neurons, pre_neurons).astype(np.float32) * 0.1
        self.delays = np.ones((post_neurons, pre_neurons), dtype=np.int32)
        
        # Synaptic dynamics parameters
        self.tau_syn = np.full(post_neurons, 5.0, dtype=np.float32)
        self.syn_decay = np.exp(-1.0 / self.tau_syn).astype(np.float32)
        
        # Synaptic current state
        self.synaptic_current = np.zeros(post_neurons, dtype=np.float32)
        
        # Spike buffer for delays
        self.max_delay = np.max(self.delays)
        self.spike_buffer = np.zeros((self.max_delay, pre_neurons), dtype=np.float32)
        self.buffer_index = 0
        
        # GPU setup if needed
        if TORCH_CUDA_AVAILABLE and optimization_level == OptimizationLevel.GPU:
            self._setup_gpu_synapses()
    
    def _setup_gpu_synapses(self):
        """Setup GPU arrays for synaptic computation."""
        self.gpu_weights = torch.tensor(self.weights, dtype=torch.float32, device='cuda')
        self.gpu_synaptic_current = torch.zeros(
            self.post_neurons, dtype=torch.float32, device='cuda'
        )
        self.gpu_syn_decay = torch.tensor(
            self.syn_decay, dtype=torch.float32, device='cuda'
        )
    
    @staticmethod
    @numba.jit(nopython=True, parallel=True, fastmath=True)
    def _compute_synaptic_current_simd(weights, spikes, synaptic_current, syn_decay):
        """SIMD-optimized synaptic current computation."""
        post_neurons, pre_neurons = weights.shape
        
        # Decay existing synaptic current
        for i in prange(post_neurons):
            synaptic_current[i] *= syn_decay[i]
        
        # Add new contributions from spikes
        for i in prange(post_neurons):
            current_sum = 0.0
            for j in range(pre_neurons):
                current_sum += weights[i, j] * spikes[j]
            synaptic_current[i] += current_sum
        
        return synaptic_current
    
    def forward(self, input_spikes: np.ndarray) -> np.ndarray:
        """Forward pass through synapses."""
        start_time = time.perf_counter()
        
        # Add spikes to buffer (for delays)
        self.spike_buffer[self.buffer_index] = input_spikes
        self.buffer_index = (self.buffer_index + 1) % self.max_delay
        
        # Get delayed spikes (simplified - using minimum delay)
        delayed_spikes = self.spike_buffer[(self.buffer_index - 1) % self.max_delay]
        
        if self.optimization_level == OptimizationLevel.GPU and TORCH_CUDA_AVAILABLE:
            output = self._forward_gpu(delayed_spikes)
        elif self.optimization_level in [OptimizationLevel.SIMD, OptimizationLevel.PARALLEL]:
            self.synaptic_current = self._compute_synaptic_current_simd(
                self.weights, delayed_spikes, self.synaptic_current, self.syn_decay
            )
            output = self.synaptic_current
        else:
            output = self._forward_basic(delayed_spikes)
        
        # Record performance
        execution_time = (time.perf_counter() - start_time) * 1000
        self._record_performance("synapse_forward", execution_time)
        
        return output
    
    def _forward_basic(self, spikes: np.ndarray) -> np.ndarray:
        """Basic NumPy implementation."""
        # Decay existing synaptic current
        self.synaptic_current *= self.syn_decay
        
        # Add new contributions
        self.synaptic_current += np.dot(self.weights, spikes)
        
        return self.synaptic_current
    
    def _forward_gpu(self, spikes: np.ndarray) -> np.ndarray:
        """GPU implementation."""
        gpu_spikes = torch.tensor(spikes, dtype=torch.float32, device='cuda')
        
        # Decay and add new contributions
        self.gpu_synaptic_current *= self.gpu_syn_decay
        self.gpu_synaptic_current += torch.mv(self.gpu_weights, gpu_spikes)
        
        return self.gpu_synaptic_current.cpu().numpy()
    
    def _record_performance(self, operation: str, execution_time_ms: float):
        """Record performance metrics."""
        # Simplified performance recording
        pass


class VectorizedSpikeEncoding:
    """Vectorized spike encoding operations."""
    
    @staticmethod
    @numba.jit(nopython=True, parallel=True, fastmath=True)
    def poisson_encoding_simd(data, max_rate, duration, dt):
        """SIMD-optimized Poisson encoding."""
        batch_size, num_channels = data.shape
        num_timesteps = int(duration / dt)
        
        spikes = np.zeros((batch_size, num_channels, num_timesteps), dtype=np.float32)
        
        for b in prange(batch_size):
            for c in prange(num_channels):
                rate = data[b, c] * max_rate
                prob = rate * dt / 1000.0  # Convert to probability per timestep
                
                for t in range(num_timesteps):
                    if np.random.random() < prob:
                        spikes[b, c, t] = 1.0
        
        return spikes
    
    @staticmethod
    def rate_encoding_vectorized(data: np.ndarray, max_rate: float = 200.0, 
                                duration: float = 100.0, dt: float = 1.0,
                                optimization_level: OptimizationLevel = OptimizationLevel.SIMD) -> np.ndarray:
        """Vectorized rate encoding with optimization selection."""
        if optimization_level == OptimizationLevel.SIMD:
            return VectorizedSpikeEncoding.poisson_encoding_simd(data, max_rate, duration, dt)
        else:
            # Basic implementation
            batch_size, num_channels = data.shape
            num_timesteps = int(duration / dt)
            
            # Normalize data
            data_norm = np.clip(data, 0, 1)
            rates = data_norm * max_rate
            spike_prob = rates * dt / 1000.0
            
            # Generate random matrix and compare
            random_vals = np.random.random((batch_size, num_channels, num_timesteps))
            spikes = (random_vals < spike_prob[:, :, np.newaxis]).astype(np.float32)
            
            return spikes


class AdaptiveOptimizer:
    """Adaptive optimizer that selects best computation method based on data size and hardware."""
    
    def __init__(self):
        self.performance_history: Dict[str, List[ComputeProfile]] = {}
        self.optimization_rules: Dict[str, Callable] = {}
        self._lock = threading.Lock()
        
    def register_operation(self, operation_name: str, 
                          optimization_rule: Callable[[Tuple], OptimizationLevel]):
        """Register an operation with its optimization rule."""
        self.optimization_rules[operation_name] = optimization_rule
    
    def select_optimization(self, operation_name: str, data_shape: Tuple[int, ...]) -> OptimizationLevel:
        """Select optimal optimization level based on data shape and history."""
        with self._lock:
            # Use registered rule if available
            if operation_name in self.optimization_rules:
                return self.optimization_rules[operation_name](data_shape)
            
            # Default heuristics
            total_elements = np.prod(data_shape)
            
            # Small data: basic operations
            if total_elements < 1000:
                return OptimizationLevel.BASIC
            
            # Medium data: SIMD if available
            elif total_elements < 100000:
                return OptimizationLevel.SIMD if HAS_AVX2 else OptimizationLevel.BASIC
            
            # Large data: GPU if available, otherwise parallel
            elif total_elements < 10000000:
                return OptimizationLevel.GPU if TORCH_CUDA_AVAILABLE else OptimizationLevel.PARALLEL
            
            # Very large data: always try GPU first
            else:
                return OptimizationLevel.GPU if TORCH_CUDA_AVAILABLE else OptimizationLevel.PARALLEL
    
    def record_performance(self, profile: ComputeProfile):
        """Record performance profile for learning."""
        with self._lock:
            if profile.operation_name not in self.performance_history:
                self.performance_history[profile.operation_name] = []
            
            self.performance_history[profile.operation_name].append(profile)
            
            # Keep only recent profiles
            if len(self.performance_history[profile.operation_name]) > 50:
                self.performance_history[profile.operation_name].pop(0)
    
    def get_best_optimization_for_size(self, operation_name: str, 
                                      data_shape: Tuple[int, ...]) -> OptimizationLevel:
        """Get best optimization level based on historical performance."""
        with self._lock:
            if operation_name not in self.performance_history:
                return self.select_optimization(operation_name, data_shape)
            
            profiles = self.performance_history[operation_name]
            
            # Find profiles with similar data size
            target_size = np.prod(data_shape)
            similar_profiles = [
                p for p in profiles
                if abs(np.prod(p.data_shape) - target_size) / target_size < 0.5
            ]
            
            if not similar_profiles:
                return self.select_optimization(operation_name, data_shape)
            
            # Select optimization with best performance score
            best_profile = max(similar_profiles, key=lambda p: p.get_performance_score())
            return best_profile.optimization_level


class VectorizedNeuromorphicProcessor:
    """High-level processor for vectorized neuromorphic computations."""
    
    def __init__(self, num_neurons: int, num_synapses: int):
        self.num_neurons = num_neurons
        self.num_synapses = num_synapses
        
        # Adaptive optimizer
        self.optimizer = AdaptiveOptimizer()
        self._setup_optimization_rules()
        
        # Components
        self.neurons: Optional[VectorizedLIFNeurons] = None
        self.synapses: Optional[VectorizedSynapses] = None
        
        # Performance tracking
        self.total_operations = 0
        self.total_compute_time_ms = 0.0
        
    def _setup_optimization_rules(self):
        """Setup optimization rules for different operations."""
        def neuron_rule(data_shape: Tuple) -> OptimizationLevel:
            num_neurons = data_shape[0] if data_shape else 1000
            
            if num_neurons < 100:
                return OptimizationLevel.BASIC
            elif num_neurons < 10000:
                return OptimizationLevel.SIMD
            elif num_neurons < 100000:
                return OptimizationLevel.PARALLEL
            else:
                return OptimizationLevel.GPU if TORCH_CUDA_AVAILABLE else OptimizationLevel.PARALLEL
        
        def synapse_rule(data_shape: Tuple) -> OptimizationLevel:
            total_synapses = np.prod(data_shape) if data_shape else 10000
            
            if total_synapses < 1000:
                return OptimizationLevel.BASIC
            elif total_synapses < 1000000:
                return OptimizationLevel.SIMD
            else:
                return OptimizationLevel.GPU if TORCH_CUDA_AVAILABLE else OptimizationLevel.PARALLEL
        
        self.optimizer.register_operation("neuron_update", neuron_rule)
        self.optimizer.register_operation("synapse_forward", synapse_rule)
    
    def initialize_neurons(self, optimization_level: Optional[OptimizationLevel] = None):
        """Initialize neuron layer with adaptive optimization."""
        if optimization_level is None:
            optimization_level = self.optimizer.select_optimization(
                "neuron_update", (self.num_neurons,)
            )
        
        self.neurons = VectorizedLIFNeurons(self.num_neurons, optimization_level)
        logger.info(f"Initialized {self.num_neurons} neurons with {optimization_level.value} optimization")
    
    def initialize_synapses(self, pre_neurons: int, post_neurons: int,
                           optimization_level: Optional[OptimizationLevel] = None):
        """Initialize synapse layer with adaptive optimization."""
        if optimization_level is None:
            optimization_level = self.optimizer.select_optimization(
                "synapse_forward", (post_neurons, pre_neurons)
            )
        
        self.synapses = VectorizedSynapses(pre_neurons, post_neurons, optimization_level)
        logger.info(f"Initialized {pre_neurons}x{post_neurons} synapses with {optimization_level.value} optimization")
    
    def process_timestep(self, input_current: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Process one timestep through the network."""
        start_time = time.perf_counter()
        
        # Update neurons
        if self.neurons is None:
            self.initialize_neurons()
        
        spikes = self.neurons.update(input_current)
        
        # Process through synapses if available
        synaptic_output = None
        if self.synapses is not None:
            synaptic_output = self.synapses.forward(spikes)
        
        # Update statistics
        self.total_operations += 1
        execution_time = (time.perf_counter() - start_time) * 1000
        self.total_compute_time_ms += execution_time
        
        # Record performance in optimizer
        if hasattr(self.neurons, 'compute_profiles') and self.neurons.compute_profiles:
            self.optimizer.record_performance(self.neurons.compute_profiles[-1])
        
        return spikes, synaptic_output
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        avg_time_per_op = (
            self.total_compute_time_ms / max(self.total_operations, 1)
        )
        
        ops_per_second = (
            self.total_operations / (self.total_compute_time_ms / 1000.0)
            if self.total_compute_time_ms > 0 else 0
        )
        
        stats = {
            'total_operations': self.total_operations,
            'total_compute_time_ms': self.total_compute_time_ms,
            'avg_time_per_operation_ms': avg_time_per_op,
            'operations_per_second': ops_per_second,
            'num_neurons': self.num_neurons,
            'cpu_features': {
                'avx2': HAS_AVX2,
                'avx512': HAS_AVX512,
                'cuda_available': TORCH_CUDA_AVAILABLE,
                'num_cores': NUM_CORES,
                'num_threads': NUM_THREADS
            }
        }
        
        # Add neuron-specific stats
        if self.neurons and self.neurons.compute_profiles:
            recent_profiles = self.neurons.compute_profiles[-10:]  # Last 10 profiles
            stats['neuron_performance'] = {
                'avg_execution_time_ms': np.mean([p.execution_time_ms for p in recent_profiles]),
                'avg_throughput_ops_per_sec': np.mean([p.throughput_ops_per_sec for p in recent_profiles]),
                'optimization_level': self.neurons.optimization_level.value
            }
        
        return stats


# Global processor instance
_vectorized_processor: Optional[VectorizedNeuromorphicProcessor] = None


def get_vectorized_processor(num_neurons: int = 1000, 
                           num_synapses: int = 10000) -> VectorizedNeuromorphicProcessor:
    """Get global vectorized processor instance."""
    global _vectorized_processor
    
    if _vectorized_processor is None:
        _vectorized_processor = VectorizedNeuromorphicProcessor(num_neurons, num_synapses)
    
    return _vectorized_processor


# Convenience functions for high-level operations
def vectorized_lif_update(membrane_potential: np.ndarray, input_current: np.ndarray,
                         decay_factor: float = 0.95, threshold: float = 1.0,
                         reset_voltage: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """High-performance LIF neuron update."""
    # Auto-select optimization level
    optimizer = AdaptiveOptimizer()
    opt_level = optimizer.select_optimization("neuron_update", membrane_potential.shape)
    
    if opt_level == OptimizationLevel.SIMD and HAS_AVX2:
        # Use SIMD-optimized version
        num_neurons = len(membrane_potential)
        spikes = np.zeros(num_neurons, dtype=np.float32)
        
        # Vectorized update
        membrane_potential *= decay_factor
        membrane_potential += input_current
        
        # Spike detection
        spike_mask = membrane_potential >= threshold
        spikes[spike_mask] = 1.0
        membrane_potential[spike_mask] = reset_voltage
        
    else:
        # Basic NumPy version
        membrane_potential = decay_factor * membrane_potential + input_current
        spike_mask = membrane_potential >= threshold
        spikes = spike_mask.astype(np.float32)
        membrane_potential[spike_mask] = reset_voltage
    
    return membrane_potential, spikes


def vectorized_spike_encoding(data: np.ndarray, max_rate: float = 200.0,
                            duration: float = 100.0, dt: float = 1.0) -> np.ndarray:
    """High-performance spike encoding."""
    optimizer = AdaptiveOptimizer()
    opt_level = optimizer.select_optimization("spike_encoding", data.shape)
    
    return VectorizedSpikeEncoding.rate_encoding_vectorized(
        data, max_rate, duration, dt, opt_level
    )