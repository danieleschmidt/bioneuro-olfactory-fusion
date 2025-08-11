"""
Neuromorphic hardware acceleration for gas detection system.
Optimizes computation for Intel Loihi, BrainScaleS, and SpiNNaker platforms.
"""

import numpy as np
import torch
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum
import time

logger = logging.getLogger(__name__)


class NeuromorphicPlatform(Enum):
    """Supported neuromorphic platforms."""
    LOIHI = "loihi"
    BRAINSCALES = "brainscales"
    SPINNAKER = "spinnaker"
    SOFTWARE_SIMULATION = "software"


@dataclass
class NeuromorphicConfig:
    """Configuration for neuromorphic acceleration."""
    platform: NeuromorphicPlatform
    num_neurons: int = 1024
    simulation_time_ms: float = 1000.0
    timestep_ms: float = 1.0
    enable_plasticity: bool = False
    enable_homeostasis: bool = True
    energy_optimization: bool = True
    spike_compression: bool = True


@dataclass
class SpikeData:
    """Container for spike train data."""
    spike_times: np.ndarray
    neuron_indices: np.ndarray
    num_neurons: int
    duration_ms: float
    
    def to_binary_matrix(self, timestep_ms: float = 1.0) -> np.ndarray:
        """Convert spike data to binary matrix representation."""
        num_timesteps = int(self.duration_ms / timestep_ms)
        spike_matrix = np.zeros((self.num_neurons, num_timesteps), dtype=np.uint8)
        
        time_indices = (self.spike_times / timestep_ms).astype(int)
        valid_mask = (time_indices >= 0) & (time_indices < num_timesteps)
        
        if np.any(valid_mask):
            spike_matrix[
                self.neuron_indices[valid_mask],
                time_indices[valid_mask]
            ] = 1
            
        return spike_matrix
        
    def compress_spikes(self, compression_ratio: float = 0.1) -> 'SpikeData':
        """Compress spike data using temporal downsampling."""
        if compression_ratio >= 1.0:
            return self
            
        # Downsample spike times
        step_size = int(1 / compression_ratio)
        compressed_indices = np.arange(0, len(self.spike_times), step_size)
        
        return SpikeData(
            spike_times=self.spike_times[compressed_indices] * compression_ratio,
            neuron_indices=self.neuron_indices[compressed_indices],
            num_neurons=self.num_neurons,
            duration_ms=self.duration_ms * compression_ratio
        )
        
    def get_spike_rate(self) -> float:
        """Calculate average spike rate in Hz."""
        if self.duration_ms <= 0:
            return 0.0
        return len(self.spike_times) / (self.duration_ms / 1000.0) / self.num_neurons
        
    def filter_neurons(self, active_threshold: int = 1) -> 'SpikeData':
        """Filter out inactive neurons."""
        # Count spikes per neuron
        spike_counts = np.bincount(self.neuron_indices, minlength=self.num_neurons)
        active_neurons = np.where(spike_counts >= active_threshold)[0]
        
        if len(active_neurons) == 0:
            return self
            
        # Filter spikes from active neurons only
        active_mask = np.isin(self.neuron_indices, active_neurons)
        
        # Remap neuron indices
        neuron_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(active_neurons)}
        new_neuron_indices = np.array([neuron_mapping[idx] for idx in self.neuron_indices[active_mask]])
        
        return SpikeData(
            spike_times=self.spike_times[active_mask],
            neuron_indices=new_neuron_indices,
            num_neurons=len(active_neurons),
            duration_ms=self.duration_ms
        )


class NeuromorphicAccelerator(ABC):
    """Enhanced abstract base class for neuromorphic accelerators."""
    
    def __init__(self, config: NeuromorphicConfig):
        self.config = config
        self.is_initialized = False
        
        # Performance monitoring
        self.execution_times: deque = deque(maxlen=100)
        self.energy_measurements: deque = deque(maxlen=100)
        self.temperature_readings: deque = deque(maxlen=100)
        self.throughput_history: deque = deque(maxlen=100)
        
        # Resource management
        self.active_networks: Dict[str, Any] = {}
        self.resource_usage = ResourceMonitor()
        self.thermal_manager = ThermalManager(config.thermal_threshold_celsius)
        self.power_manager = PowerManager(config.power_budget_watts)
        
        # Optimization features
        self.network_cache: Dict[str, Any] = {}
        self.adaptive_router = AdaptiveRouter() if config.enable_adaptive_routing else None
        self.batch_processor = BatchProcessor() if config.enable_batch_processing else None
        self.pipeline_optimizer = PipelineOptimizer() if config.enable_pipeline_optimization else None
        
        # Statistics
        self.total_executions = 0
        self.successful_executions = 0
        self.failed_executions = 0
        self.total_energy_consumed = 0.0
        
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the neuromorphic hardware."""
        pass
        
    @abstractmethod
    def compile_network(self, network_config: Dict[str, Any]) -> Any:
        """Compile neural network for the platform."""
        pass
        
    @abstractmethod
    def execute(self, input_spikes: SpikeData, 
               compiled_network: Any) -> SpikeData:
        """Execute computation on neuromorphic hardware."""
        pass
        
    @abstractmethod
    def get_energy_consumption(self) -> float:
        """Get energy consumption in watts."""
        pass
        
    def cleanup(self):
        """Cleanup resources."""
        # Clear caches
        self.network_cache.clear()
        self.active_networks.clear()
        
        # Stop monitoring components
        if hasattr(self, 'resource_usage'):
            self.resource_usage.stop_monitoring()
            
        self.is_initialized = False
        
    async def execute_batch(self, input_batch: List[SpikeData], 
                          compiled_network: Any) -> List[SpikeData]:
        """Execute batch of inputs for improved throughput."""
        if not self.batch_processor:
            # Fall back to sequential execution
            results = []
            for spike_data in input_batch:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self.execute, spike_data, compiled_network
                )
                results.append(result)
            return results
            
        return await self.batch_processor.process_batch(
            self, input_batch, compiled_network
        )
        
    def optimize_network(self, network_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply platform-specific optimizations to network configuration."""
        optimized_config = network_config.copy()
        
        # Apply energy optimizations
        if self.config.energy_optimization:
            optimized_config = self._apply_energy_optimizations(optimized_config)
            
        # Apply adaptive routing optimizations
        if self.adaptive_router:
            optimized_config = self.adaptive_router.optimize_routing(optimized_config)
            
        # Apply pipeline optimizations
        if self.pipeline_optimizer:
            optimized_config = self.pipeline_optimizer.optimize_pipeline(optimized_config)
            
        return optimized_config
        
    def _apply_energy_optimizations(self, network_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply energy-aware optimizations."""
        # Reduce precision where possible
        if 'precision' not in network_config:
            network_config['precision'] = 'int8'  # Lower precision for energy savings
            
        # Enable clock gating
        network_config['enable_clock_gating'] = True
        
        # Optimize spike frequency
        if 'max_spike_rate' not in network_config:
            network_config['max_spike_rate'] = 100.0  # Hz
            
        return network_config
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {
            'total_executions': self.total_executions,
            'successful_executions': self.successful_executions,
            'failed_executions': self.failed_executions,
            'success_rate': self.successful_executions / max(self.total_executions, 1),
            'total_energy_consumed': self.total_energy_consumed,
            'platform': self.config.platform.value
        }
        
        # Add timing statistics
        if self.execution_times:
            stats.update({
                'avg_execution_time_ms': np.mean(self.execution_times),
                'min_execution_time_ms': np.min(self.execution_times),
                'max_execution_time_ms': np.max(self.execution_times),
                'execution_time_std_ms': np.std(self.execution_times)
            })
            
        # Add energy statistics
        if self.energy_measurements:
            stats.update({
                'avg_power_watts': np.mean(self.energy_measurements),
                'min_power_watts': np.min(self.energy_measurements),
                'max_power_watts': np.max(self.energy_measurements),
                'energy_efficiency': np.mean(self.throughput_history) / np.mean(self.energy_measurements)
                if self.energy_measurements and self.throughput_history else 0
            })
            
        # Add thermal statistics
        if self.temperature_readings:
            stats.update({
                'avg_temperature_c': np.mean(self.temperature_readings),
                'max_temperature_c': np.max(self.temperature_readings),
                'thermal_violations': sum(1 for t in self.temperature_readings 
                                        if t > self.config.thermal_threshold_celsius)
            })
            
        return stats
        
    def record_execution(self, execution_time: float, energy_used: float, 
                        temperature: float, throughput: float, success: bool):
        """Record execution metrics."""
        self.total_executions += 1
        
        if success:
            self.successful_executions += 1
            self.execution_times.append(execution_time)
            self.energy_measurements.append(energy_used)
            self.temperature_readings.append(temperature)
            self.throughput_history.append(throughput)
            self.total_energy_consumed += energy_used * (execution_time / 1000.0)  # Wh
        else:
            self.failed_executions += 1


class LoihiAccelerator(NeuromorphicAccelerator):
    """Intel Loihi neuromorphic accelerator."""
    
    def __init__(self, config: NeuromorphicConfig):
        super().__init__(config)
        self.loihi_board = None
        self.network_graph = None
        self.chip_temperature = 25.0  # Celsius
        self.power_consumption = 0.5  # Watts
        
    def initialize(self) -> bool:
        """Initialize Loihi hardware."""
        try:
            # In a real implementation, this would initialize the Loihi hardware
            # For now, we simulate the initialization
            logger.info("Initializing Intel Loihi neuromorphic processor")
            
            # Simulate hardware detection and initialization
            time.sleep(0.1)  # Simulate initialization time
            
            self.is_initialized = True
            logger.info("Loihi initialization successful")
            return True
            
        except Exception as e:
            logger.error(f"Loihi initialization failed: {e}")
            return False
            
    def compile_network(self, network_config: Dict[str, Any]) -> Dict[str, Any]:
        """Compile network for Loihi execution."""
        if not self.is_initialized:
            raise RuntimeError("Loihi not initialized")
            
        logger.info("Compiling network for Loihi")
        
        # Extract network parameters
        input_neurons = network_config.get('input_neurons', 64)
        hidden_neurons = network_config.get('hidden_neurons', 512)
        output_neurons = network_config.get('output_neurons', 8)
        
        # Simulate network compilation
        compiled_network = {
            'platform': 'loihi',
            'input_layer': {
                'num_neurons': input_neurons,
                'neuron_type': 'current_based_lif',
                'threshold': 1.0,
                'refractory_period': 2
            },
            'hidden_layer': {
                'num_neurons': hidden_neurons,
                'neuron_type': 'current_based_lif',
                'threshold': 1.0,
                'refractory_period': 2,
                'plasticity_enabled': self.config.enable_plasticity
            },
            'output_layer': {
                'num_neurons': output_neurons,
                'neuron_type': 'current_based_lif',
                'threshold': 1.0,
                'refractory_period': 2
            },
            'synapses': {
                'input_to_hidden': {
                    'weight_precision': 8,
                    'delay_range': [1, 4],
                    'learning_rule': 'stdp' if self.config.enable_plasticity else None
                },
                'hidden_to_output': {
                    'weight_precision': 8,
                    'delay_range': [1, 2],
                    'learning_rule': None
                }
            }
        }
        
        logger.info(f"Network compiled with {input_neurons + hidden_neurons + output_neurons} neurons")
        return compiled_network
        
    def execute(self, input_spikes: SpikeData, 
               compiled_network: Dict[str, Any]) -> SpikeData:
        """Execute on Loihi hardware."""
        if not self.is_initialized:
            raise RuntimeError("Loihi not initialized")
            
        logger.debug(f"Executing on Loihi for {input_spikes.duration_ms}ms")
        
        # Simulate execution on Loihi
        start_time = time.perf_counter()
        
        # Convert input spikes to Loihi format
        spike_matrix = input_spikes.to_binary_matrix(self.config.timestep_ms)
        
        # Simulate neuromorphic computation
        output_spikes = self._simulate_loihi_computation(spike_matrix, compiled_network)
        
        execution_time = time.perf_counter() - start_time
        
        logger.debug(f"Loihi execution completed in {execution_time*1000:.2f}ms")
        
        return output_spikes
        
    def _simulate_loihi_computation(self, spike_matrix: np.ndarray,
                                  network: Dict[str, Any]) -> SpikeData:
        """Simulate computation on Loihi hardware."""
        num_timesteps = spike_matrix.shape[1]
        output_neurons = network['output_layer']['num_neurons']
        
        # Simulate spike propagation through network
        # This is a simplified simulation
        output_spike_times = []
        output_neuron_indices = []
        
        for t in range(num_timesteps):
            input_activity = np.sum(spike_matrix[:, t])
            
            # Simple model: output spikes based on input activity
            if input_activity > 2:  # Threshold
                # Generate output spikes with some probability
                spike_prob = min(0.3, input_activity / 20.0)
                num_output_spikes = np.random.poisson(spike_prob * output_neurons)
                
                if num_output_spikes > 0:
                    spike_neurons = np.random.choice(
                        output_neurons, 
                        min(num_output_spikes, output_neurons),
                        replace=False
                    )
                    spike_times = np.full(len(spike_neurons), t * self.config.timestep_ms)
                    
                    output_spike_times.extend(spike_times)
                    output_neuron_indices.extend(spike_neurons)
        
        return SpikeData(
            spike_times=np.array(output_spike_times),
            neuron_indices=np.array(output_neuron_indices, dtype=int),
            num_neurons=output_neurons,
            duration_ms=num_timesteps * self.config.timestep_ms
        )
        
    def get_energy_consumption(self) -> float:
        """Get Loihi energy consumption."""
        # Loihi is very energy efficient
        base_power = 0.1  # watts
        activity_power = 0.01 * (self.config.num_neurons / 1000)
        return base_power + activity_power


class BrainScaleSAccelerator(NeuromorphicAccelerator):
    """BrainScaleS neuromorphic accelerator."""
    
    def __init__(self, config: NeuromorphicConfig):
        super().__init__(config)
        self.wafer_module = None
        
    def initialize(self) -> bool:
        """Initialize BrainScaleS hardware."""
        try:
            logger.info("Initializing BrainScaleS neuromorphic system")
            
            # Simulate hardware initialization
            time.sleep(0.2)
            
            self.is_initialized = True
            logger.info("BrainScaleS initialization successful")
            return True
            
        except Exception as e:
            logger.error(f"BrainScaleS initialization failed: {e}")
            return False
            
    def compile_network(self, network_config: Dict[str, Any]) -> Dict[str, Any]:
        """Compile network for BrainScaleS."""
        if not self.is_initialized:
            raise RuntimeError("BrainScaleS not initialized")
            
        logger.info("Compiling network for BrainScaleS")
        
        # BrainScaleS uses analog neurons
        compiled_network = {
            'platform': 'brainscales',
            'analog_neurons': True,
            'acceleration_factor': 1000,  # BrainScaleS speedup
            'neuron_parameters': {
                'membrane_capacitance': 200e-12,  # pF
                'leak_conductance': 10e-9,        # nS
                'excitatory_reversal': 0,         # mV
                'inhibitory_reversal': -70,       # mV
                'threshold': -55,                 # mV
                'reset': -70                      # mV
            },
            'calibration_required': True
        }
        
        return compiled_network
        
    def execute(self, input_spikes: SpikeData,
               compiled_network: Dict[str, Any]) -> SpikeData:
        """Execute on BrainScaleS hardware."""
        if not self.is_initialized:
            raise RuntimeError("BrainScaleS not initialized")
            
        logger.debug("Executing on BrainScaleS")
        
        # BrainScaleS runs much faster than real-time
        acceleration_factor = compiled_network['acceleration_factor']
        actual_execution_time = input_spikes.duration_ms / acceleration_factor / 1000  # seconds
        
        start_time = time.perf_counter()
        
        # Simulate analog computation
        output_spikes = self._simulate_brainscales_computation(input_spikes, compiled_network)
        
        # Simulate actual hardware execution time
        time.sleep(max(0, actual_execution_time - (time.perf_counter() - start_time)))
        
        return output_spikes
        
    def _simulate_brainscales_computation(self, input_spikes: SpikeData,
                                        network: Dict[str, Any]) -> SpikeData:
        """Simulate BrainScaleS analog computation."""
        # Simplified analog neuron simulation
        output_spike_times = []
        output_neuron_indices = []
        
        # Simulate analog dynamics
        for spike_time, neuron_idx in zip(input_spikes.spike_times, input_spikes.neuron_indices):
            # Simple response model
            if np.random.random() < 0.4:  # Response probability
                response_delay = np.random.exponential(5.0)  # ms
                output_time = spike_time + response_delay
                
                if output_time < input_spikes.duration_ms:
                    output_spike_times.append(output_time)
                    output_neuron_indices.append(neuron_idx % 8)  # Map to output neurons
        
        return SpikeData(
            spike_times=np.array(output_spike_times),
            neuron_indices=np.array(output_neuron_indices, dtype=int),
            num_neurons=8,
            duration_ms=input_spikes.duration_ms
        )
        
    def get_energy_consumption(self) -> float:
        """Get BrainScaleS energy consumption."""
        # Analog circuits consume more power but are very fast
        return 5.0  # watts


class SpiNNakerAccelerator(NeuromorphicAccelerator):
    """SpiNNaker neuromorphic accelerator."""
    
    def __init__(self, config: NeuromorphicConfig):
        super().__init__(config)
        self.machine = None
        self.num_boards = 4
        
    def initialize(self) -> bool:
        """Initialize SpiNNaker machine."""
        try:
            logger.info("Initializing SpiNNaker neuromorphic machine")
            
            # Simulate machine initialization
            time.sleep(0.15)
            
            self.is_initialized = True
            logger.info(f"SpiNNaker initialization successful ({self.num_boards} boards)")
            return True
            
        except Exception as e:
            logger.error(f"SpiNNaker initialization failed: {e}")
            return False
            
    def compile_network(self, network_config: Dict[str, Any]) -> Dict[str, Any]:
        """Compile network for SpiNNaker."""
        if not self.is_initialized:
            raise RuntimeError("SpiNNaker not initialized")
            
        logger.info("Compiling network for SpiNNaker")
        
        compiled_network = {
            'platform': 'spinnaker',
            'distributed': True,
            'num_boards': self.num_boards,
            'neurons_per_core': 255,  # SpiNNaker limitation
            'routing_tables': True,
            'real_time_simulation': True,
            'neuron_model': 'IF_curr_exp',
            'synapse_model': 'current_based'
        }
        
        return compiled_network
        
    def execute(self, input_spikes: SpikeData,
               compiled_network: Dict[str, Any]) -> SpikeData:
        """Execute on SpiNNaker machine."""
        if not self.is_initialized:
            raise RuntimeError("SpiNNaker not initialized")
            
        logger.debug("Executing on SpiNNaker")
        
        # SpiNNaker runs in real-time
        execution_time_s = input_spikes.duration_ms / 1000.0
        
        start_time = time.perf_counter()
        
        # Simulate distributed computation
        output_spikes = self._simulate_spinnaker_computation(input_spikes, compiled_network)
        
        # Ensure real-time execution
        elapsed = time.perf_counter() - start_time
        if elapsed < execution_time_s:
            time.sleep(execution_time_s - elapsed)
            
        return output_spikes
        
    def _simulate_spinnaker_computation(self, input_spikes: SpikeData,
                                      network: Dict[str, Any]) -> SpikeData:
        """Simulate SpiNNaker distributed computation."""
        # Simulate integrate-and-fire dynamics
        output_spike_times = []
        output_neuron_indices = []
        
        # Group spikes by time windows
        timestep_ms = self.config.timestep_ms
        num_timesteps = int(input_spikes.duration_ms / timestep_ms)
        
        for t in range(num_timesteps):
            window_start = t * timestep_ms
            window_end = (t + 1) * timestep_ms
            
            # Find spikes in this window
            window_mask = (
                (input_spikes.spike_times >= window_start) &
                (input_spikes.spike_times < window_end)
            )
            
            if np.any(window_mask):
                window_activity = np.sum(window_mask)
                
                # Simulate network response
                if window_activity > 1:  # Threshold
                    num_output_spikes = min(window_activity // 2, 8)
                    
                    for _ in range(num_output_spikes):
                        spike_time = window_start + np.random.uniform(0, timestep_ms)
                        neuron_id = np.random.randint(0, 8)
                        
                        output_spike_times.append(spike_time)
                        output_neuron_indices.append(neuron_id)
        
        return SpikeData(
            spike_times=np.array(output_spike_times),
            neuron_indices=np.array(output_neuron_indices, dtype=int),
            num_neurons=8,
            duration_ms=input_spikes.duration_ms
        )
        
    def get_energy_consumption(self) -> float:
        """Get SpiNNaker energy consumption."""
        # ARM processors consume moderate power
        return 2.5 * self.num_boards  # watts per board


class SoftwareSimulator(NeuromorphicAccelerator):
    """Software-based neuromorphic simulator."""
    
    def __init__(self, config: NeuromorphicConfig):
        super().__init__(config)
        self.use_gpu = torch.cuda.is_available()
        
    def initialize(self) -> bool:
        """Initialize software simulator."""
        logger.info("Initializing software neuromorphic simulator")
        self.is_initialized = True
        return True
        
    def compile_network(self, network_config: Dict[str, Any]) -> Dict[str, Any]:
        """Compile network for software simulation."""
        compiled_network = {
            'platform': 'software',
            'use_gpu': self.use_gpu,
            'precision': 'float32',
            'optimization': 'vectorized',
            'parallel_cores': 8
        }
        return compiled_network
        
    def execute(self, input_spikes: SpikeData,
               compiled_network: Dict[str, Any]) -> SpikeData:
        """Execute software simulation."""
        logger.debug("Executing software simulation")
        
        # Fast software simulation
        output_spikes = self._simulate_software_computation(input_spikes)
        return output_spikes
        
    def _simulate_software_computation(self, input_spikes: SpikeData) -> SpikeData:
        """High-performance software simulation."""
        # Vectorized computation
        spike_matrix = input_spikes.to_binary_matrix(self.config.timestep_ms)
        
        # Simple feedforward processing
        weights = np.random.randn(8, input_spikes.num_neurons) * 0.1
        
        # Compute membrane potentials
        membrane_potentials = np.dot(weights, spike_matrix)
        
        # Apply threshold
        threshold = 0.5
        output_spikes_matrix = (membrane_potentials > threshold).astype(np.uint8)
        
        # Convert back to spike format
        output_times, output_neurons = np.where(output_spikes_matrix)
        output_spike_times = output_neurons * self.config.timestep_ms
        
        return SpikeData(
            spike_times=output_spike_times,
            neuron_indices=output_times,
            num_neurons=8,
            duration_ms=input_spikes.duration_ms
        )
        
    def get_energy_consumption(self) -> float:
        """Get software simulation energy consumption."""
        base_power = 50.0 if not self.use_gpu else 150.0  # watts
        return base_power


class NeuromorphicAcceleratorManager:
    """Manages multiple neuromorphic accelerators."""
    
    def __init__(self):
        self.accelerators: Dict[NeuromorphicPlatform, NeuromorphicAccelerator] = {}
        self.available_platforms: List[NeuromorphicPlatform] = []
        
    def detect_platforms(self) -> List[NeuromorphicPlatform]:
        """Detect available neuromorphic platforms."""
        available = []
        
        # Always available
        available.append(NeuromorphicPlatform.SOFTWARE_SIMULATION)
        
        # Check for hardware platforms (simulated for now)
        # In real implementation, this would check hardware availability
        if np.random.random() > 0.7:  # Simulate 30% chance of having Loihi
            available.append(NeuromorphicPlatform.LOIHI)
            
        if np.random.random() > 0.8:  # Simulate 20% chance of having BrainScaleS
            available.append(NeuromorphicPlatform.BRAINSCALES)
            
        if np.random.random() > 0.6:  # Simulate 40% chance of having SpiNNaker
            available.append(NeuromorphicPlatform.SPINNAKER)
            
        self.available_platforms = available
        logger.info(f"Detected platforms: {[p.value for p in available]}")
        return available
        
    def initialize_accelerator(self, platform: NeuromorphicPlatform,
                             config: NeuromorphicConfig) -> bool:
        """Initialize a specific accelerator."""
        if platform == NeuromorphicPlatform.LOIHI:
            accelerator = LoihiAccelerator(config)
        elif platform == NeuromorphicPlatform.BRAINSCALES:
            accelerator = BrainScaleSAccelerator(config)
        elif platform == NeuromorphicPlatform.SPINNAKER:
            accelerator = SpiNNakerAccelerator(config)
        else:
            accelerator = SoftwareSimulator(config)
            
        if accelerator.initialize():
            self.accelerators[platform] = accelerator
            logger.info(f"Initialized {platform.value} accelerator")
            return True
        else:
            logger.error(f"Failed to initialize {platform.value} accelerator")
            return False
            
    def select_optimal_platform(self, 
                               performance_requirements: Dict[str, float]) -> NeuromorphicPlatform:
        """Select optimal platform based on requirements."""
        target_latency = performance_requirements.get('latency_ms', 50.0)
        max_energy = performance_requirements.get('energy_watts', 10.0)
        accuracy_required = performance_requirements.get('accuracy', 0.9)
        
        # Score platforms based on requirements
        platform_scores = {}
        
        for platform in self.available_platforms:
            if platform == NeuromorphicPlatform.LOIHI:
                # Excellent energy efficiency, good latency
                energy_score = 1.0 if max_energy >= 0.5 else 0.0
                latency_score = 1.0 if target_latency >= 10.0 else 0.5
                accuracy_score = 0.9
            elif platform == NeuromorphicPlatform.BRAINSCALES:
                # Fast but power hungry
                energy_score = 1.0 if max_energy >= 5.0 else 0.2
                latency_score = 1.0  # Very fast
                accuracy_score = 0.85
            elif platform == NeuromorphicPlatform.SPINNAKER:
                # Real-time, moderate power
                energy_score = 1.0 if max_energy >= 10.0 else 0.5
                latency_score = 0.8 if target_latency >= 50.0 else 0.3
                accuracy_score = 0.88
            else:  # Software
                # Flexible but power hungry
                energy_score = 1.0 if max_energy >= 50.0 else 0.1
                latency_score = 0.7
                accuracy_score = 0.95
                
            total_score = (energy_score + latency_score + accuracy_score) / 3
            platform_scores[platform] = total_score
            
        # Select best platform
        best_platform = max(platform_scores, key=platform_scores.get)
        logger.info(f"Selected {best_platform.value} platform (score: {platform_scores[best_platform]:.2f})")
        
        return best_platform
        
    def execute_computation(self, 
                          platform: NeuromorphicPlatform,
                          input_spikes: SpikeData,
                          network_config: Dict[str, Any]) -> Tuple[SpikeData, Dict[str, float]]:
        """Execute computation on specified platform."""
        if platform not in self.accelerators:
            raise ValueError(f"Platform {platform.value} not initialized")
            
        accelerator = self.accelerators[platform]
        
        # Compile network
        start_time = time.perf_counter()
        compiled_network = accelerator.compile_network(network_config)
        compile_time = time.perf_counter() - start_time
        
        # Execute computation
        start_time = time.perf_counter()
        output_spikes = accelerator.execute(input_spikes, compiled_network)
        execution_time = time.perf_counter() - start_time
        
        # Get energy consumption
        energy_consumption = accelerator.get_energy_consumption()
        
        metrics = {
            'compile_time_ms': compile_time * 1000,
            'execution_time_ms': execution_time * 1000,
            'energy_consumption_watts': energy_consumption,
            'energy_per_spike_nj': (energy_consumption * execution_time * 1e9) / max(len(output_spikes.spike_times), 1),
            'throughput_spikes_per_second': len(output_spikes.spike_times) / execution_time if execution_time > 0 else 0
        }
        
        return output_spikes, metrics
        
    def cleanup_all(self):
        """Cleanup all accelerators."""
        for accelerator in self.accelerators.values():
            accelerator.cleanup()
        self.accelerators.clear()
        logger.info("All accelerators cleaned up")



class ResourceMonitor:
    """Monitors resource usage of neuromorphic hardware."""
    
    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        self.cpu_usage_history: deque = deque(maxlen=100)
        self.memory_usage_history: deque = deque(maxlen=100)
        
    def start_monitoring(self):
        """Start resource monitoring."""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
            
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # Get system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                self.cpu_usage_history.append(cpu_percent)
                self.memory_usage_history.append(memory.percent)
                
                time.sleep(5)  # Monitor every 5 seconds
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                
    def get_current_usage(self) -> Dict[str, float]:
        """Get current resource usage."""
        return {
            'cpu_percent': self.cpu_usage_history[-1] if self.cpu_usage_history else 0,
            'memory_percent': self.memory_usage_history[-1] if self.memory_usage_history else 0,
            'avg_cpu_percent': np.mean(self.cpu_usage_history) if self.cpu_usage_history else 0,
            'avg_memory_percent': np.mean(self.memory_usage_history) if self.memory_usage_history else 0
        }


class ThermalManager:
    """Manages thermal conditions for neuromorphic hardware."""
    
    def __init__(self, threshold_celsius: float = 85.0):
        self.threshold_celsius = threshold_celsius
        self.current_temperature = 25.0
        self.thermal_violations = 0
        
    def update_temperature(self, temperature_celsius: float):
        """Update current temperature reading."""
        self.current_temperature = temperature_celsius
        
        if temperature_celsius > self.threshold_celsius:
            self.thermal_violations += 1
            logger.warning(f"Thermal violation: {temperature_celsius:.1f}°C > {self.threshold_celsius:.1f}°C")
            
    def should_throttle(self) -> bool:
        """Check if thermal throttling should be applied."""
        return self.current_temperature > self.threshold_celsius * 0.9
        
    def get_throttle_factor(self) -> float:
        """Get throttling factor (0.0 to 1.0)."""
        if not self.should_throttle():
            return 1.0
            
        # Linear throttling based on temperature
        excess_temp = self.current_temperature - (self.threshold_celsius * 0.9)
        max_excess = self.threshold_celsius * 0.1
        throttle_factor = max(0.1, 1.0 - (excess_temp / max_excess))
        
        return throttle_factor


class PowerManager:
    """Manages power consumption and DVFS for neuromorphic hardware."""
    
    def __init__(self, power_budget_watts: float = 10.0):
        self.power_budget_watts = power_budget_watts
        self.current_power_watts = 0.0
        self.voltage_levels = [0.8, 0.9, 1.0, 1.1, 1.2]  # Available voltage levels
        self.frequency_levels = [100, 200, 400, 800, 1000]  # MHz
        self.current_voltage_idx = 2  # Default to middle level
        self.current_frequency_idx = 2
        
    def update_power_consumption(self, power_watts: float):
        """Update current power consumption."""
        self.current_power_watts = power_watts
        
        # Auto-adjust DVFS if over budget
        if power_watts > self.power_budget_watts:
            self._reduce_power()
        elif power_watts < self.power_budget_watts * 0.8:
            self._increase_power()
            
    def _reduce_power(self):
        """Reduce power consumption by lowering voltage/frequency."""
        if self.current_voltage_idx > 0:
            self.current_voltage_idx -= 1
            logger.info(f"Reduced voltage to {self.voltage_levels[self.current_voltage_idx]}V")
        elif self.current_frequency_idx > 0:
            self.current_frequency_idx -= 1
            logger.info(f"Reduced frequency to {self.frequency_levels[self.current_frequency_idx]}MHz")
            
    def _increase_power(self):
        """Increase power for better performance if budget allows."""
        if self.current_frequency_idx < len(self.frequency_levels) - 1:
            self.current_frequency_idx += 1
            logger.info(f"Increased frequency to {self.frequency_levels[self.current_frequency_idx]}MHz")
        elif self.current_voltage_idx < len(self.voltage_levels) - 1:
            self.current_voltage_idx += 1
            logger.info(f"Increased voltage to {self.voltage_levels[self.current_voltage_idx]}V")
            
    def get_current_settings(self) -> Dict[str, Any]:
        """Get current DVFS settings."""
        return {
            'voltage_v': self.voltage_levels[self.current_voltage_idx],
            'frequency_mhz': self.frequency_levels[self.current_frequency_idx],
            'power_watts': self.current_power_watts,
            'power_budget_watts': self.power_budget_watts,
            'utilization': self.current_power_watts / self.power_budget_watts
        }


class AdaptiveRouter:
    """Adaptive routing optimizer for neuromorphic networks."""
    
    def __init__(self):
        self.routing_history: Dict[str, List[float]] = {}
        self.optimal_routes: Dict[str, Dict[str, Any]] = {}
        
    def optimize_routing(self, network_config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize routing based on historical performance."""
        config_hash = self._hash_config(network_config)
        
        if config_hash in self.optimal_routes:
            # Use cached optimal routing
            network_config.update(self.optimal_routes[config_hash])
        else:
            # Apply heuristic routing optimizations
            network_config = self._apply_routing_heuristics(network_config)
            
        return network_config
        
    def _apply_routing_heuristics(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply routing optimization heuristics."""
        # Enable shortest path routing
        config['routing_algorithm'] = 'shortest_path'
        
        # Enable load balancing across cores
        config['enable_load_balancing'] = True
        
        # Minimize cross-chip communication
        config['minimize_cross_chip'] = True
        
        return config
        
    def record_performance(self, config_hash: str, latency_ms: float, 
                          energy_consumed: float):
        """Record routing performance for learning."""
        if config_hash not in self.routing_history:
            self.routing_history[config_hash] = []
            
        # Combined performance score (lower is better)
        score = latency_ms + energy_consumed * 10  # Weight energy 10x
        self.routing_history[config_hash].append(score)
        
    def _hash_config(self, config: Dict[str, Any]) -> str:
        """Generate hash for network configuration."""
        import hashlib
        config_str = str(sorted(config.items()))
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


class BatchProcessor:
    """Batch processing optimizer for neuromorphic execution."""
    
    def __init__(self, max_batch_size: int = 16):
        self.max_batch_size = max_batch_size
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def process_batch(self, accelerator: NeuromorphicAccelerator,
                          input_batch: List[SpikeData], 
                          compiled_network: Any) -> List[SpikeData]:
        """Process batch of spike data efficiently."""
        if len(input_batch) <= 1:
            # Single input, no batching needed
            return [accelerator.execute(input_batch[0], compiled_network)]
            
        # Split into sub-batches if needed
        sub_batches = [
            input_batch[i:i + self.max_batch_size]
            for i in range(0, len(input_batch), self.max_batch_size)
        ]
        
        results = []
        for sub_batch in sub_batches:
            # Process sub-batch
            if hasattr(accelerator, '_execute_batch_native'):
                # Use native batch execution if available
                batch_result = await asyncio.get_event_loop().run_in_executor(
                    None, accelerator._execute_batch_native, sub_batch, compiled_network
                )
                results.extend(batch_result)
            else:
                # Fall back to parallel execution
                futures = [
                    asyncio.get_event_loop().run_in_executor(
                        self.executor, accelerator.execute, spike_data, compiled_network
                    )
                    for spike_data in sub_batch
                ]
                batch_results = await asyncio.gather(*futures)
                results.extend(batch_results)
                
        return results


class PipelineOptimizer:
    """Pipeline optimization for neuromorphic processing."""
    
    def __init__(self):
        self.pipeline_stages = ['input', 'encode', 'process', 'decode', 'output']
        self.stage_latencies: Dict[str, deque] = {
            stage: deque(maxlen=100) for stage in self.pipeline_stages
        }
        
    def optimize_pipeline(self, network_config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize pipeline configuration."""
        # Enable pipeline parallelism
        network_config['enable_pipeline_parallelism'] = True
        
        # Set optimal pipeline depth based on latency analysis
        if self._has_sufficient_history():
            optimal_depth = self._calculate_optimal_pipeline_depth()
            network_config['pipeline_depth'] = optimal_depth
            
        # Enable prefetching
        network_config['enable_prefetch'] = True
        
        return network_config
        
    def _has_sufficient_history(self) -> bool:
        """Check if we have enough history for optimization."""
        return all(len(latencies) >= 10 for latencies in self.stage_latencies.values())
        
    def _calculate_optimal_pipeline_depth(self) -> int:
        """Calculate optimal pipeline depth based on stage latencies."""
        # Simple heuristic: balance pipeline depth with latency
        avg_latencies = {
            stage: np.mean(latencies) 
            for stage, latencies in self.stage_latencies.items()
        }
        
        max_latency_stage = max(avg_latencies, key=avg_latencies.get)
        max_latency = avg_latencies[max_latency_stage]
        
        # Deeper pipeline for stages with higher latency
        if max_latency > 10:  # ms
            return 8
        elif max_latency > 5:
            return 4
        else:
            return 2
            
    def record_stage_latency(self, stage: str, latency_ms: float):
        """Record latency for pipeline stage."""
        if stage in self.stage_latencies:
            self.stage_latencies[stage].append(latency_ms)


# Global manager instance
neuromorphic_manager = NeuromorphicAcceleratorManager()