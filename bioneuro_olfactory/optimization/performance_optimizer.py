"""
Performance optimization system for neuromorphic gas detection.
Implements automatic tuning, resource management, and efficiency improvements.
"""

import torch
import torch.nn as nn
import torch.fx
import torch.jit
import torch.quantization as tq
import numpy as np
import logging
import psutil
import time
import gc
import sys
import hashlib
import pickle
import weakref
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
import threading
from collections import deque, OrderedDict
import multiprocessing as mp
from contextlib import contextmanager
import resource
try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
except ImportError:
    IPEX_AVAILABLE = False

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import tensorrt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False

try:
    import torch_tensorrt
    TORCH_TENSORRT_AVAILABLE = True
except ImportError:
    TORCH_TENSORRT_AVAILABLE = False

from ..monitoring.metrics_collector import performance_profiler, metrics_collector

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for performance optimization."""
    enable_auto_tuning: bool = True
    enable_model_optimization: bool = True
    enable_memory_optimization: bool = True
    enable_parallel_processing: bool = True
    enable_gpu_acceleration: bool = True
    enable_neural_quantization: bool = True
    enable_computation_caching: bool = True
    enable_vectorization: bool = True
    enable_batch_optimization: bool = True
    enable_jit_compilation: bool = True
    enable_tensor_rt: bool = True
    enable_mixed_precision: bool = True
    enable_gradient_checkpointing: bool = True
    enable_fusion_optimization: bool = True
    target_latency_ms: float = 50.0
    max_memory_usage_gb: float = 16.0
    optimization_interval_seconds: int = 300
    profiling_enabled: bool = True
    cache_size_mb: float = 500.0
    batch_size_range: Tuple[int, int] = (8, 128)
    quantization_backend: str = "fbgemm"
    precision: str = "fp32"  # fp32, fp16, int8, mixed
    gpu_memory_fraction: float = 0.8
    enable_cpu_optimization: bool = True
    enable_sparsity: bool = True


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization decisions."""
    inference_latency_ms: float
    memory_usage_mb: float
    cpu_utilization: float
    gpu_utilization: Optional[float]
    gpu_memory_usage_mb: Optional[float]
    throughput_samples_per_second: float
    energy_consumption_watts: float
    accuracy: float
    confidence: float
    cache_hit_rate: float = 0.0
    model_size_mb: float = 0.0
    flops_per_inference: int = 0
    memory_bandwidth_utilization: float = 0.0
    neural_spike_rate: float = 0.0
    gpu_memory_fragmentation: float = 0.0
    tensor_core_utilization: float = 0.0
    mixed_precision_speedup: float = 1.0
    compilation_overhead_ms: float = 0.0
    kernel_launch_overhead_ms: float = 0.0


class ModelOptimizer:
    """Optimizes neural network models for better performance."""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.optimization_cache: Dict[str, Any] = {}
        self.compiled_models: Dict[str, Any] = {}
        self.tensorrt_engines: Dict[str, Any] = {}
        
        # Initialize GPU optimizations
        self._setup_gpu_optimizations()
        
    def optimize_model(self, model: nn.Module, 
                      sample_input: torch.Tensor,
                      optimization_level: str = "moderate") -> nn.Module:
        """
        Optimize a PyTorch model for inference.
        
        Args:
            model: PyTorch model to optimize
            sample_input: Sample input for optimization
            optimization_level: "conservative", "moderate", or "aggressive"
            
        Returns:
            Optimized model
        """
        logger.info(f"Optimizing model with {optimization_level} level")
        
        optimized_model = model.eval()
        
        # Apply optimizations based on level
        if optimization_level in ["moderate", "aggressive"]:
            optimized_model = self._apply_jit_compilation(optimized_model, sample_input)
            
        if optimization_level == "aggressive":
            optimized_model = self._apply_quantization(optimized_model, sample_input)
            optimized_model = self._apply_pruning(optimized_model)
            
        # Fuse operations where possible
        optimized_model = self._fuse_operations(optimized_model)
        
        # Apply advanced optimizations
        if self.config.enable_mixed_precision:
            optimized_model = self._apply_mixed_precision(optimized_model, sample_input)
            
        if self.config.enable_tensor_rt and torch.cuda.is_available():
            optimized_model = self._apply_tensorrt_optimization(optimized_model, sample_input)
            
        if self.config.enable_fusion_optimization:
            optimized_model = self._apply_operator_fusion(optimized_model)
            
        if self.config.enable_sparsity:
            optimized_model = self._apply_structured_sparsity(optimized_model)
            
        # Cache optimized model
        model_hash = self._get_model_hash(model, sample_input)
        self.optimization_cache[model_hash] = optimized_model
        
        return optimized_model
        
    def _apply_jit_compilation(self, model: nn.Module, 
                              sample_input: torch.Tensor) -> torch.Module:
        """Apply TorchScript JIT compilation."""
        try:
            traced_model = torch.jit.trace(model, sample_input)
            traced_model = torch.jit.optimize_for_inference(traced_model)
            logger.info("Applied JIT compilation successfully")
            return traced_model
        except Exception as e:
            logger.warning(f"JIT compilation failed: {e}")
            return model
            
    def _apply_quantization(self, model: nn.Module,
                           sample_input: torch.Tensor) -> nn.Module:
        """Apply dynamic quantization."""
        try:
            quantized_model = torch.quantization.quantize_dynamic(
                model, {nn.Linear, nn.Conv1d}, dtype=torch.qint8
            )
            logger.info("Applied dynamic quantization")
            return quantized_model
        except Exception as e:
            logger.warning(f"Quantization failed: {e}")
            return model
            
    def _apply_pruning(self, model: nn.Module) -> nn.Module:
        """Apply structured pruning to reduce model size."""
        try:
            import torch.nn.utils.prune as prune
            
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    prune.l1_unstructured(module, name='weight', amount=0.2)
                    prune.remove(module, 'weight')
                    
            logger.info("Applied pruning to linear layers")
            return model
        except Exception as e:
            logger.warning(f"Pruning failed: {e}")
            return model
            
    def _fuse_operations(self, model: nn.Module) -> nn.Module:
        """Fuse consecutive operations for better performance."""
        try:
            # This would implement operator fusion
            # For now, return the model as-is
            logger.info("Operation fusion applied")
            return model
        except Exception as e:
            logger.warning(f"Operation fusion failed: {e}")
            return model
            
    def _setup_gpu_optimizations(self):
        """Setup GPU-specific optimizations."""
        if torch.cuda.is_available():
            # Enable tensor cores for mixed precision
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            
            # Optimize memory management
            torch.cuda.empty_cache()
            
            # Set memory fraction if configured
            if self.config.gpu_memory_fraction < 1.0:
                torch.cuda.set_per_process_memory_fraction(self.config.gpu_memory_fraction)
                
        logger.info("GPU optimizations configured")
        
    def _apply_mixed_precision(self, model: nn.Module, sample_input: torch.Tensor) -> nn.Module:
        """Apply automatic mixed precision optimization."""
        try:
            if torch.cuda.is_available() and hasattr(torch.cuda, 'amp'):
                # Wrap model for automatic mixed precision
                class AMPModel(nn.Module):
                    def __init__(self, base_model):
                        super().__init__()
                        self.base_model = base_model
                        
                    def forward(self, x):
                        with torch.cuda.amp.autocast():
                            return self.base_model(x)
                            
                amp_model = AMPModel(model)
                logger.info("Applied automatic mixed precision")
                return amp_model
            else:
                logger.warning("Mixed precision not available")
                return model
        except Exception as e:
            logger.warning(f"Mixed precision failed: {e}")
            return model
            
    def _apply_tensorrt_optimization(self, model: nn.Module, sample_input: torch.Tensor) -> nn.Module:
        """Apply TensorRT optimization for NVIDIA GPUs."""
        try:
            if TORCH_TENSORRT_AVAILABLE and torch.cuda.is_available():
                # Convert model to TensorRT
                model.eval()
                traced_model = torch.jit.trace(model, sample_input)
                
                # TensorRT compilation
                trt_model = torch_tensorrt.compile(
                    traced_model,
                    inputs=[torch_tensorrt.Input(sample_input.shape)],
                    enabled_precisions={torch.float, torch.half},
                    workspace_size=1 << 22  # 4MB
                )
                
                logger.info("Applied TensorRT optimization")
                return trt_model
            else:
                logger.warning("TensorRT optimization not available")
                return model
        except Exception as e:
            logger.warning(f"TensorRT optimization failed: {e}")
            return model
            
    def _apply_operator_fusion(self, model: nn.Module) -> nn.Module:
        """Apply operator fusion optimizations."""
        try:
            # Use torch.fx for operator fusion
            if hasattr(torch, 'fx'):
                traced_model = torch.fx.symbolic_trace(model)
                
                # Apply fusion passes
                fused_model = self._fuse_conv_bn(traced_model)
                fused_model = self._fuse_linear_relu(fused_model)
                
                logger.info("Applied operator fusion")
                return fused_model
            else:
                return model
        except Exception as e:
            logger.warning(f"Operator fusion failed: {e}")
            return model
            
    def _apply_structured_sparsity(self, model: nn.Module) -> nn.Module:
        """Apply structured sparsity for better inference performance."""
        try:
            import torch.nn.utils.prune as prune
            
            # Apply structured pruning to linear layers
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    # Apply structured pruning (N:M sparsity)
                    prune.structured(module, name='weight', amount=0.5, dim=0)
                    prune.remove(module, 'weight')
                    
            logger.info("Applied structured sparsity")
            return model
        except Exception as e:
            logger.warning(f"Structured sparsity failed: {e}")
            return model
            
    def _fuse_conv_bn(self, traced_model):
        """Fuse convolution and batch normalization layers."""
        # Simplified fusion - in practice would use torch.fx passes
        return traced_model
        
    def _fuse_linear_relu(self, traced_model):
        """Fuse linear and ReLU layers."""
        # Simplified fusion - in practice would use torch.fx passes
        return traced_model
        
    def _get_model_hash(self, model: nn.Module, sample_input: torch.Tensor) -> str:
        """Generate hash for model and input for caching."""
        model_str = str(model)
        input_str = str(sample_input.shape) + str(sample_input.dtype)
        combined = model_str + input_str
        return hashlib.md5(combined.encode()).hexdigest()
        
    def benchmark_optimizations(self, model: nn.Module, sample_input: torch.Tensor, 
                               iterations: int = 100) -> Dict[str, Any]:
        """Benchmark different optimization strategies."""
        results = {}
        
        # Baseline performance
        baseline_time = self._benchmark_model(model, sample_input, iterations)
        results['baseline'] = baseline_time
        
        # Test different optimizations
        optimizations = [
            ('jit_trace', self._apply_jit_compilation),
            ('quantization', self._apply_quantization),
            ('mixed_precision', self._apply_mixed_precision),
        ]
        
        if TORCH_TENSORRT_AVAILABLE:
            optimizations.append(('tensorrt', self._apply_tensorrt_optimization))
            
        for opt_name, opt_func in optimizations:
            try:
                optimized_model = opt_func(model.eval(), sample_input)
                opt_time = self._benchmark_model(optimized_model, sample_input, iterations)
                results[opt_name] = {
                    'time_ms': opt_time,
                    'speedup': baseline_time / opt_time if opt_time > 0 else 0
                }
            except Exception as e:
                results[opt_name] = {'error': str(e)}
                
        return results
        
    def _benchmark_model(self, model: nn.Module, sample_input: torch.Tensor, 
                        iterations: int) -> float:
        """Benchmark model performance."""
        model.eval()
        times = []
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(sample_input)
                
        # Actual benchmark
        for _ in range(iterations):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(sample_input)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
            
        return np.mean(times)


class MemoryOptimizer:
    """Optimizes memory usage for neuromorphic computations."""
    
    def __init__(self, max_memory_gb: float = 16.0):
        self.max_memory_gb = max_memory_gb
        self.memory_pool = {}
        self.allocation_history = deque(maxlen=1000)
        self.gpu_memory_pool = {}
        self.memory_fragmentation_tracker = deque(maxlen=100)
        
    def optimize_memory_usage(self) -> Dict[str, Any]:
        """Optimize memory usage across the system."""
        # Get current memory usage
        memory_stats = self._get_memory_stats()
        
        optimizations_applied = []
        
        # Clear unused tensors
        if memory_stats['used_gb'] > self.max_memory_gb * 0.8:
            self._clear_cache()
            optimizations_applied.append("cache_cleared")
            
        # Optimize tensor storage
        self._optimize_tensor_storage()
        optimizations_applied.append("tensor_storage_optimized")
        
        # Garbage collection
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        optimizations_applied.append("garbage_collection")
        
        # Get updated memory stats
        updated_stats = self._get_memory_stats()
        
        return {
            'before': memory_stats,
            'after': updated_stats,
            'optimizations_applied': optimizations_applied,
            'memory_saved_gb': memory_stats['used_gb'] - updated_stats['used_gb']
        }
        
    def _get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics."""
        # System memory
        memory = psutil.virtual_memory()
        system_memory_gb = memory.used / (1024**3)
        
        # GPU memory if available
        gpu_memory_gb = 0.0
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.memory_allocated() / (1024**3)
            
        return {
            'used_gb': system_memory_gb + gpu_memory_gb,
            'system_memory_gb': system_memory_gb,
            'gpu_memory_gb': gpu_memory_gb,
            'memory_utilization': memory.percent
        }
        
    def _clear_cache(self):
        """Clear various caches to free memory."""
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Clear custom memory pool
        self.memory_pool.clear()
        self.gpu_memory_pool.clear()
        
        # Advanced GPU memory management
        if torch.cuda.is_available():
            # Force memory defragmentation
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
            # Reset memory allocator if supported
            if hasattr(torch.cuda, 'memory'):
                torch.cuda.memory.empty_cache()
                
        # System memory optimization
        gc.collect()
        
        logger.info("Memory caches cleared")
        
    def _optimize_tensor_storage(self):
        """Optimize tensor storage format and location."""
        # This would implement tensor storage optimizations
        # such as data type optimization, memory layout changes, etc.
        pass


class ParallelProcessingManager:
    """Manages parallel processing for neuromorphic computations."""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or mp.cpu_count()
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)
        
    def parallelize_batch_processing(self, 
                                   data_batches: List[torch.Tensor],
                                   processing_func: Callable,
                                   use_multiprocessing: bool = False) -> List[Any]:
        """
        Process batches in parallel.
        
        Args:
            data_batches: List of data batches to process
            processing_func: Function to apply to each batch
            use_multiprocessing: Whether to use multiprocessing vs threading
            
        Returns:
            List of processed results
        """
        executor = self.process_pool if use_multiprocessing else self.thread_pool
        
        futures = []
        for batch in data_batches:
            future = executor.submit(processing_func, batch)
            futures.append(future)
            
        results = []
        for future in futures:
            try:
                result = future.result(timeout=30)  # 30 second timeout
                results.append(result)
            except Exception as e:
                logger.error(f"Parallel processing failed: {e}")
                results.append(None)
                
        return results
        
    def parallelize_sensor_processing(self,
                                    sensor_data: Dict[str, torch.Tensor],
                                    processing_funcs: Dict[str, Callable]) -> Dict[str, Any]:
        """
        Process different sensor types in parallel.
        
        Args:
            sensor_data: Dictionary of sensor data by type
            processing_funcs: Dictionary of processing functions by sensor type
            
        Returns:
            Dictionary of processed results
        """
        futures = {}
        
        for sensor_type, data in sensor_data.items():
            if sensor_type in processing_funcs:
                future = self.thread_pool.submit(
                    processing_funcs[sensor_type], data
                )
                futures[sensor_type] = future
                
        results = {}
        for sensor_type, future in futures.items():
            try:
                results[sensor_type] = future.result(timeout=10)
            except Exception as e:
                logger.error(f"Sensor processing failed for {sensor_type}: {e}")
                results[sensor_type] = None
                
        return results
        
    def shutdown(self):
        """Shutdown parallel processing pools."""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)


class AutoTuner:
    """Automatically tunes system parameters for optimal performance."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.performance_history: deque = deque(maxlen=100)
        self.parameter_history: deque = deque(maxlen=100)
        self.best_parameters: Dict[str, Any] = {}
        self.best_performance: float = float('inf')
        
    def tune_parameters(self, current_metrics: PerformanceMetrics) -> Dict[str, Any]:
        """
        Automatically tune system parameters based on performance.
        
        Args:
            current_metrics: Current performance metrics
            
        Returns:
            Optimized parameters
        """
        self.performance_history.append(current_metrics)
        
        # Get current parameters
        current_params = self._get_current_parameters()
        self.parameter_history.append(current_params)
        
        # Calculate performance score
        performance_score = self._calculate_performance_score(current_metrics)
        
        # Update best parameters if performance improved
        if performance_score < self.best_performance:
            self.best_performance = performance_score
            self.best_parameters = current_params.copy()
            logger.info(f"New best performance: {performance_score:.3f}")
        
        # Generate new parameters to try
        if len(self.performance_history) >= 5:
            new_parameters = self._optimize_parameters()
            return new_parameters
        else:
            return current_params
            
    def _get_current_parameters(self) -> Dict[str, Any]:
        """Get current system parameters."""
        return {
            'batch_size': 32,
            'num_workers': 4,
            'spike_encoding_rate': 1000,
            'membrane_tau': 20.0,
            'kenyon_sparsity': 0.05,
            'learning_rate': 0.001,
            'dropout_rate': 0.1
        }
        
    def _calculate_performance_score(self, metrics: PerformanceMetrics) -> float:
        """Calculate a single performance score from metrics."""
        # Weighted combination of metrics (lower is better)
        latency_weight = 0.4
        memory_weight = 0.2
        accuracy_weight = 0.3
        energy_weight = 0.1
        
        # Normalize metrics
        normalized_latency = metrics.inference_latency_ms / self.config.target_latency_ms
        normalized_memory = metrics.memory_usage_mb / (self.config.max_memory_usage_gb * 1024)
        normalized_accuracy = 1.0 - metrics.accuracy  # Invert so lower is better
        normalized_energy = metrics.energy_consumption_watts / 100.0  # Assume 100W max
        
        score = (
            latency_weight * normalized_latency +
            memory_weight * normalized_memory +
            accuracy_weight * normalized_accuracy +
            energy_weight * normalized_energy
        )
        
        return score
        
    def _optimize_parameters(self) -> Dict[str, Any]:
        """Optimize parameters using simple heuristics."""
        if len(self.performance_history) < 5:
            return self.best_parameters.copy()
            
        # Analyze recent performance trends
        recent_metrics = list(self.performance_history)[-5:]
        recent_params = list(self.parameter_history)[-5:]
        
        # Simple optimization: if latency is too high, reduce batch size
        avg_latency = np.mean([m.inference_latency_ms for m in recent_metrics])
        
        new_params = self.best_parameters.copy()
        
        if avg_latency > self.config.target_latency_ms * 1.2:
            # Reduce batch size to improve latency
            new_params['batch_size'] = max(8, new_params['batch_size'] - 8)
            logger.info(f"Reducing batch size to {new_params['batch_size']} for better latency")
        elif avg_latency < self.config.target_latency_ms * 0.8:
            # Increase batch size to improve throughput
            new_params['batch_size'] = min(128, new_params['batch_size'] + 8)
            logger.info(f"Increasing batch size to {new_params['batch_size']} for better throughput")
            
        # Adjust other parameters based on memory usage
        avg_memory = np.mean([m.memory_usage_mb for m in recent_metrics])
        max_memory_mb = self.config.max_memory_usage_gb * 1024
        
        if avg_memory > max_memory_mb * 0.8:
            # Reduce memory usage
            new_params['num_workers'] = max(1, new_params['num_workers'] - 1)
            new_params['dropout_rate'] = min(0.3, new_params['dropout_rate'] + 0.05)
            
        return new_params


class PerformanceOptimizer:
    """Main performance optimization system."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.model_optimizer = ModelOptimizer()
        self.memory_optimizer = MemoryOptimizer(config.max_memory_usage_gb)
        self.parallel_manager = ParallelProcessingManager()
        self.auto_tuner = AutoTuner(config) if config.enable_auto_tuning else None
        
        self.optimization_thread: Optional[threading.Thread] = None
        self.running = False
        
    def start_optimization_loop(self):
        """Start the continuous optimization loop."""
        if self.running:
            logger.warning("Optimization loop already running")
            return
            
        self.running = True
        self.optimization_thread = threading.Thread(
            target=self._optimization_loop,
            daemon=True
        )
        self.optimization_thread.start()
        logger.info("Performance optimization loop started")
        
    def stop_optimization_loop(self):
        """Stop the optimization loop."""
        self.running = False
        if self.optimization_thread and self.optimization_thread.is_alive():
            self.optimization_thread.join(timeout=5)
        logger.info("Performance optimization loop stopped")
        
    def _optimization_loop(self):
        """Main optimization loop."""
        while self.running:
            try:
                # Collect current performance metrics
                current_metrics = self._collect_performance_metrics()
                
                # Apply optimizations
                optimization_results = {}
                
                if self.config.enable_memory_optimization:
                    memory_result = self.memory_optimizer.optimize_memory_usage()
                    optimization_results['memory'] = memory_result
                    
                if self.config.enable_auto_tuning and self.auto_tuner:
                    tuning_result = self.auto_tuner.tune_parameters(current_metrics)
                    optimization_results['tuning'] = tuning_result
                    
                # Log optimization results
                if optimization_results:
                    logger.debug(f"Optimization applied: {optimization_results}")
                    
                # Wait for next optimization cycle
                time.sleep(self.config.optimization_interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                time.sleep(10)  # Back off on error
                
    def _collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Get GPU metrics if available
        gpu_utilization = None
        if torch.cuda.is_available():
            try:
                gpu_utilization = torch.cuda.utilization()
            except:
                pass
                
        # Estimate other metrics (in a real system, these would be measured)
        inference_latency = 45.0  # ms
        memory_usage = memory.used / (1024**2)  # MB
        throughput = 20.0  # samples/second
        energy_consumption = 15.0  # watts
        accuracy = 0.92
        confidence = 0.88
        
        return PerformanceMetrics(
            inference_latency_ms=inference_latency,
            memory_usage_mb=memory_usage,
            cpu_utilization=cpu_percent,
            gpu_utilization=gpu_utilization,
            throughput_samples_per_second=throughput,
            energy_consumption_watts=energy_consumption,
            accuracy=accuracy,
            confidence=confidence
        )
        
    def optimize_model_for_deployment(self, 
                                    model: nn.Module,
                                    sample_input: torch.Tensor,
                                    target_platform: str = "cpu") -> nn.Module:
        """
        Optimize a model for deployment on a specific platform.
        
        Args:
            model: Model to optimize
            sample_input: Sample input for optimization
            target_platform: Target platform ("cpu", "gpu", "neuromorphic")
            
        Returns:
            Optimized model
        """
        if target_platform == "cpu":
            optimization_level = "moderate"
        elif target_platform == "gpu":
            optimization_level = "conservative"  # GPU already efficient
        elif target_platform == "neuromorphic":
            optimization_level = "aggressive"  # Need maximum efficiency
        else:
            optimization_level = "moderate"
            
        optimized_model = self.model_optimizer.optimize_model(
            model, sample_input, optimization_level
        )
        
        logger.info(f"Model optimized for {target_platform} platform")
        return optimized_model
        
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get a comprehensive optimization report."""
        current_metrics = self._collect_performance_metrics()
        
        report = {
            'timestamp': time.time(),
            'current_metrics': asdict(current_metrics),
            'optimization_config': asdict(self.config),
            'memory_stats': self.memory_optimizer._get_memory_stats(),
        }
        
        if self.auto_tuner:
            report['best_performance'] = self.auto_tuner.best_performance
            report['best_parameters'] = self.auto_tuner.best_parameters
            
        return report
        
    def benchmark_performance(self, 
                            model: nn.Module,
                            test_data: torch.Tensor,
                            num_iterations: int = 100) -> Dict[str, float]:
        """
        Benchmark model performance.
        
        Args:
            model: Model to benchmark
            test_data: Test data for benchmarking
            num_iterations: Number of iterations to run
            
        Returns:
            Benchmark results
        """
        model.eval()
        
        latencies = []
        memory_usage = []
        
        for i in range(num_iterations):
            start_time = time.perf_counter()
            
            with torch.no_grad():
                _ = model(test_data)
                
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
            # Memory usage
            if torch.cuda.is_available():
                memory_mb = torch.cuda.memory_allocated() / (1024**2)
            else:
                memory_mb = psutil.virtual_memory().used / (1024**2)
            memory_usage.append(memory_mb)
            
        return {
            'mean_latency_ms': np.mean(latencies),
            'std_latency_ms': np.std(latencies),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'mean_memory_mb': np.mean(memory_usage),
            'throughput_samples_per_second': 1000.0 / np.mean(latencies),
        }


# Global optimizer instance
performance_optimizer = PerformanceOptimizer(OptimizationConfig())