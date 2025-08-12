"""Enhanced input validation and sanitization for neuromorphic systems.

This module extends the existing validation framework with specialized validators
for neuromorphic spiking neural networks, including spike train validation,
temporal pattern analysis, and neuromorphic-specific security checks.
"""

import numpy as np
import re
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import json
import hashlib

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from .error_handling_enhanced import ValidationError, SecurityError, ErrorSeverity
from .logging_enhanced import info, error, warning, security, audit
from ..security.input_validation import InputValidator, SensorDataValidator, NetworkInputValidator


class NeuromorphicValidationType(Enum):
    """Types of neuromorphic validation."""
    SPIKE_TRAIN = "spike_train"
    TEMPORAL_PATTERN = "temporal_pattern"
    NETWORK_TOPOLOGY = "network_topology"
    SYNAPTIC_WEIGHTS = "synaptic_weights"
    MEMBRANE_DYNAMICS = "membrane_dynamics"


@dataclass
class SpikeTrainValidationConfig:
    """Configuration for spike train validation."""
    max_spike_rate: float = 1000.0  # Hz
    min_spike_rate: float = 0.0     # Hz
    max_burst_length: int = 50      # timesteps
    max_isi_cv: float = 5.0         # Max coefficient of variation for ISI
    temporal_resolution: float = 1.0 # ms
    validate_causality: bool = True
    check_refractory_violations: bool = True
    refractory_period: float = 2.0  # ms


@dataclass
class NetworkTopologyValidationConfig:
    """Configuration for network topology validation."""
    max_neurons: int = 1000000
    max_synapses_per_neuron: int = 10000
    min_connection_probability: float = 0.001
    max_connection_probability: float = 0.5
    validate_small_world: bool = False
    validate_scale_free: bool = False
    check_connectivity: bool = True


class NeuromorphicValidator(InputValidator):
    """Enhanced validator for neuromorphic systems."""
    
    def __init__(self, validation_level=None):
        super().__init__(validation_level)
        self._register_neuromorphic_rules()
        
        # Neuromorphic-specific configurations
        self.spike_config = SpikeTrainValidationConfig()
        self.network_config = NetworkTopologyValidationConfig()
    
    def _register_neuromorphic_rules(self):
        """Register neuromorphic-specific validation rules."""
        
        # Spike train validations
        self.add_validation_rule(
            "spike_train",
            ValidationRule(
                name="spike_train",
                validator_func=self._validate_spike_train,
                error_message="Invalid spike train format or values"
            )
        )
        
        self.add_validation_rule(
            "spike_rate",
            ValidationRule(
                name="spike_rate", 
                validator_func=lambda x: self._validate_spike_rate(x),
                error_message="Spike rate outside valid range"
            )
        )
        
        self.add_validation_rule(
            "temporal_pattern",
            ValidationRule(
                name="temporal_pattern",
                validator_func=self._validate_temporal_pattern,
                error_message="Invalid temporal spike pattern"
            )
        )
        
        # Network topology validations
        self.add_validation_rule(
            "network_size",
            ValidationRule(
                name="network_size",
                validator_func=self._validate_network_size,
                error_message="Network size exceeds limits"
            )
        )
        
        self.add_validation_rule(
            "synaptic_weights",
            ValidationRule(
                name="synaptic_weights",
                validator_func=self._validate_synaptic_weights,
                error_message="Invalid synaptic weight values"
            )
        )
        
        self.add_validation_rule(
            "membrane_potential",
            ValidationRule(
                name="membrane_potential",
                validator_func=self._validate_membrane_potential,
                error_message="Membrane potential outside physiological range"
            )
        )
        
        # Connectivity validations
        self.add_validation_rule(
            "connectivity_matrix",
            ValidationRule(
                name="connectivity_matrix",
                validator_func=self._validate_connectivity_matrix,
                error_message="Invalid connectivity matrix"
            )
        )
    
    def validate_spike_trains(
        self, 
        spike_data: Union[np.ndarray, torch.Tensor, List],
        config: Optional[SpikeTrainValidationConfig] = None
    ) -> Union[np.ndarray, torch.Tensor, List]:
        """Comprehensive spike train validation.
        
        Args:
            spike_data: Spike train data [neurons, time] or [batch, neurons, time]
            config: Validation configuration
            
        Returns:
            Validated and potentially sanitized spike data
        """
        if config is None:
            config = self.spike_config
        
        info("Starting spike train validation")
        
        try:
            # Convert to numpy for validation
            if TORCH_AVAILABLE and isinstance(spike_data, torch.Tensor):
                np_data = spike_data.detach().cpu().numpy()
                original_tensor = spike_data
                is_torch = True
            elif isinstance(spike_data, list):
                np_data = np.array(spike_data)
                original_tensor = None
                is_torch = False
            else:
                np_data = spike_data
                original_tensor = None
                is_torch = False
            
            # Validate shape
            if np_data.ndim < 2 or np_data.ndim > 3:
                raise ValidationError(
                    f"Spike data must be 2D or 3D, got {np_data.ndim}D",
                    error_code="INVALID_SPIKE_DIMENSIONS"
                )
            
            # Handle batch dimension
            if np_data.ndim == 3:
                batch_size, num_neurons, time_steps = np_data.shape
                batch_mode = True
            else:
                num_neurons, time_steps = np_data.shape
                batch_size = 1
                batch_mode = False
                np_data = np_data.reshape(1, num_neurons, time_steps)
            
            validated_data = np.zeros_like(np_data)
            
            # Validate each batch
            for batch_idx in range(batch_size):
                batch_spikes = np_data[batch_idx]
                
                # Basic spike validation
                self._validate_basic_spikes(batch_spikes, config)
                
                # Temporal validation
                self._validate_spike_timing(batch_spikes, config)
                
                # Rate validation
                self._validate_spike_rates(batch_spikes, config)
                
                # Pattern validation
                self._validate_spike_patterns(batch_spikes, config)
                
                # Sanitize if needed
                validated_data[batch_idx] = self._sanitize_spike_train(batch_spikes, config)
            
            # Convert back to original format
            if not batch_mode:
                validated_data = validated_data.squeeze(0)
            
            if is_torch and TORCH_AVAILABLE:
                validated_data = torch.tensor(validated_data, 
                                            dtype=original_tensor.dtype,
                                            device=original_tensor.device)
            elif isinstance(spike_data, list):
                validated_data = validated_data.tolist()
            
            info("Spike train validation completed successfully")
            return validated_data
            
        except Exception as e:
            error(f"Spike train validation failed: {str(e)}")
            raise ValidationError(
                f"Spike train validation error: {str(e)}",
                error_code="SPIKE_VALIDATION_FAILED",
                severity=ErrorSeverity.HIGH
            )
    
    def validate_network_topology(
        self,
        connectivity_matrix: Union[np.ndarray, torch.Tensor],
        weights: Optional[Union[np.ndarray, torch.Tensor]] = None,
        config: Optional[NetworkTopologyValidationConfig] = None
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Optional[Union[np.ndarray, torch.Tensor]]]:
        """Validate network topology and connectivity.
        
        Args:
            connectivity_matrix: Boolean connectivity matrix
            weights: Optional synaptic weights matrix  
            config: Validation configuration
            
        Returns:
            Validated connectivity matrix and weights
        """
        if config is None:
            config = self.network_config
        
        try:
            # Convert to numpy for validation
            if TORCH_AVAILABLE and isinstance(connectivity_matrix, torch.Tensor):
                conn_np = connectivity_matrix.detach().cpu().numpy()
                is_torch = True
                original_device = connectivity_matrix.device
                original_dtype = connectivity_matrix.dtype
            else:
                conn_np = connectivity_matrix
                is_torch = False
                original_device = None
                original_dtype = None
            
            # Basic shape validation
            if conn_np.ndim != 2:
                raise ValidationError(
                    f"Connectivity matrix must be 2D, got {conn_np.ndim}D",
                    error_code="INVALID_CONNECTIVITY_SHAPE"
                )
            
            num_pre, num_post = conn_np.shape
            
            # Size limits
            if num_pre > config.max_neurons or num_post > config.max_neurons:
                raise ValidationError(
                    f"Network size ({num_pre}, {num_post}) exceeds limit {config.max_neurons}",
                    error_code="NETWORK_TOO_LARGE"
                )
            
            # Connectivity validation
            self._validate_connectivity_properties(conn_np, config)
            
            # Weights validation if provided
            validated_weights = None
            if weights is not None:
                validated_weights = self._validate_weight_matrix(weights, conn_np, config)
            
            # Convert back to original format
            if is_torch and TORCH_AVAILABLE:
                validated_conn = torch.tensor(conn_np, dtype=original_dtype, device=original_device)
                if validated_weights is not None:
                    validated_weights = torch.tensor(validated_weights, device=original_device)
            else:
                validated_conn = conn_np
            
            return validated_conn, validated_weights
            
        except Exception as e:
            error(f"Network topology validation failed: {str(e)}")
            raise ValidationError(
                f"Network topology validation error: {str(e)}",
                error_code="TOPOLOGY_VALIDATION_FAILED",
                severity=ErrorSeverity.HIGH
            )
    
    def validate_neuromorphic_config(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Validate neuromorphic model configuration."""
        validated_config = {}
        
        # Define neuromorphic parameter rules
        param_rules = {
            # Neuron parameters
            "tau_membrane": ["positive_number"],
            "threshold": ["positive_number"],
            "reset_voltage": [],  # Can be negative
            "refractory_period": ["positive_number"],
            
            # Network parameters
            "num_neurons": ["positive_number", "network_size"],
            "connection_probability": ["probability"],
            "synaptic_delay": ["positive_number"],
            
            # Learning parameters
            "learning_rate": ["positive_number", "learning_rate"],
            "stdp_tau_plus": ["positive_number"],
            "stdp_tau_minus": ["positive_number"],
            
            # Simulation parameters
            "dt": ["positive_number"],
            "simulation_time": ["positive_number"],
        }
        
        for param_name, rules in param_rules.items():
            if param_name in config_dict:
                try:
                    value = config_dict[param_name]
                    
                    # Type-specific sanitization
                    if isinstance(value, (int, float)):
                        sanitized = self.sanitize(value, "numeric")
                    else:
                        sanitized = value
                    
                    # Apply validation rules
                    validated = self.validate(sanitized, rules, {"parameter": param_name})
                    validated_config[param_name] = validated
                    
                except Exception as e:
                    raise ValidationError(
                        f"Invalid neuromorphic parameter {param_name}: {str(e)}",
                        error_code="INVALID_NEUROMORPHIC_PARAM",
                        context={"parameter": param_name, "value": config_dict[param_name]}
                    )
        
        # Copy non-validated parameters with warning
        for key, value in config_dict.items():
            if key not in validated_config:
                warning(f"Unvalidated parameter: {key}")
                validated_config[key] = value
        
        return validated_config
    
    # Private validation methods
    
    def _validate_spike_train(self, spike_data) -> bool:
        """Validate basic spike train properties."""
        try:
            if TORCH_AVAILABLE and isinstance(spike_data, torch.Tensor):
                spike_data = spike_data.detach().cpu().numpy()
            elif isinstance(spike_data, list):
                spike_data = np.array(spike_data)
            
            # Check for binary values (0 or 1)
            unique_values = np.unique(spike_data)
            if not all(val in [0, 1, 0.0, 1.0] for val in unique_values):
                return False
            
            # Check reasonable dimensions
            if spike_data.ndim < 1 or spike_data.ndim > 3:
                return False
            
            # Check for reasonable size
            if spike_data.size > 1e8:  # 100M elements
                return False
            
            return True
            
        except Exception:
            return False
    
    def _validate_spike_rate(self, spike_rate: float) -> bool:
        """Validate spike rate value."""
        return (isinstance(spike_rate, (int, float)) and 
                self.spike_config.min_spike_rate <= spike_rate <= self.spike_config.max_spike_rate)
    
    def _validate_temporal_pattern(self, pattern) -> bool:
        """Validate temporal spike pattern."""
        # Placeholder for complex temporal pattern validation
        return True
    
    def _validate_network_size(self, size: int) -> bool:
        """Validate network size."""
        return isinstance(size, int) and 1 <= size <= self.network_config.max_neurons
    
    def _validate_synaptic_weights(self, weights) -> bool:
        """Validate synaptic weights."""
        if TORCH_AVAILABLE and isinstance(weights, torch.Tensor):
            weights = weights.detach().cpu().numpy()
        elif isinstance(weights, list):
            weights = np.array(weights)
        
        # Check for reasonable weight values
        if np.any(np.abs(weights) > 100):  # Very large weights
            return False
        
        # Check for invalid values
        if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
            return False
        
        return True
    
    def _validate_membrane_potential(self, potential) -> bool:
        """Validate membrane potential values."""
        if isinstance(potential, (int, float)):
            # Typical range for biological neurons: -100mV to +50mV
            return -100 <= potential <= 50
        
        if TORCH_AVAILABLE and isinstance(potential, torch.Tensor):
            potential = potential.detach().cpu().numpy()
        elif isinstance(potential, list):
            potential = np.array(potential)
        
        return np.all((-100 <= potential) & (potential <= 50))
    
    def _validate_connectivity_matrix(self, matrix) -> bool:
        """Validate connectivity matrix."""
        if TORCH_AVAILABLE and isinstance(matrix, torch.Tensor):
            matrix = matrix.detach().cpu().numpy()
        elif isinstance(matrix, list):
            matrix = np.array(matrix)
        
        # Must be 2D
        if matrix.ndim != 2:
            return False
        
        # Must be binary
        unique_vals = np.unique(matrix)
        if not all(val in [0, 1, True, False] for val in unique_vals):
            return False
        
        return True
    
    def _validate_basic_spikes(self, spike_data: np.ndarray, config: SpikeTrainValidationConfig):
        """Validate basic spike properties."""
        # Check for binary values
        unique_values = np.unique(spike_data)
        if not all(val in [0, 1, 0.0, 1.0] for val in unique_values):
            raise ValidationError(
                "Spikes must be binary (0 or 1)",
                error_code="NON_BINARY_SPIKES"
            )
        
        # Check for NaN or infinite values
        if np.any(np.isnan(spike_data)) or np.any(np.isinf(spike_data)):
            raise ValidationError(
                "Spike data contains NaN or infinite values",
                error_code="INVALID_SPIKE_VALUES"
            )
    
    def _validate_spike_timing(self, spike_data: np.ndarray, config: SpikeTrainValidationConfig):
        """Validate spike timing properties."""
        if not config.validate_causality:
            return
        
        num_neurons, time_steps = spike_data.shape
        
        # Check refractory period violations
        if config.check_refractory_violations:
            for neuron_idx in range(num_neurons):
                spike_times = np.where(spike_data[neuron_idx] > 0)[0]
                if len(spike_times) > 1:
                    isi = np.diff(spike_times) * config.temporal_resolution
                    if np.any(isi < config.refractory_period):
                        warning(
                            f"Refractory period violation in neuron {neuron_idx}"
                        )
    
    def _validate_spike_rates(self, spike_data: np.ndarray, config: SpikeTrainValidationConfig):
        """Validate spike rates."""
        num_neurons, time_steps = spike_data.shape
        
        # Calculate firing rates
        firing_rates = np.sum(spike_data, axis=1) / (time_steps * config.temporal_resolution / 1000.0)
        
        # Check rate limits
        if np.any(firing_rates > config.max_spike_rate):
            high_rate_neurons = np.where(firing_rates > config.max_spike_rate)[0]
            raise ValidationError(
                f"Spike rates too high in neurons {high_rate_neurons}: max={np.max(firing_rates):.1f}Hz",
                error_code="SPIKE_RATE_TOO_HIGH"
            )
        
        if np.any(firing_rates < config.min_spike_rate):
            low_rate_neurons = np.where(firing_rates < config.min_spike_rate)[0]
            warning(
                f"Very low spike rates in neurons {low_rate_neurons}: min={np.min(firing_rates):.1f}Hz"
            )
    
    def _validate_spike_patterns(self, spike_data: np.ndarray, config: SpikeTrainValidationConfig):
        """Validate spike patterns for anomalies."""
        num_neurons, time_steps = spike_data.shape
        
        # Check for excessive bursting
        for neuron_idx in range(num_neurons):
            spikes = spike_data[neuron_idx]
            
            # Find burst periods
            burst_starts = []
            burst_lengths = []
            in_burst = False
            burst_start = 0
            
            for t in range(time_steps):
                if spikes[t] > 0:
                    if not in_burst:
                        in_burst = True
                        burst_start = t
                else:
                    if in_burst:
                        burst_length = t - burst_start
                        if burst_length > config.max_burst_length:
                            warning(
                                f"Long burst detected in neuron {neuron_idx}: {burst_length} timesteps"
                            )
                        in_burst = False
    
    def _sanitize_spike_train(self, spike_data: np.ndarray, config: SpikeTrainValidationConfig) -> np.ndarray:
        """Sanitize spike train data."""
        sanitized = spike_data.copy()
        
        # Ensure binary values
        sanitized = np.clip(sanitized, 0, 1)
        sanitized = np.round(sanitized).astype(np.float32)
        
        # Remove NaN and infinite values
        sanitized = np.nan_to_num(sanitized, nan=0.0, posinf=1.0, neginf=0.0)
        
        return sanitized
    
    def _validate_connectivity_properties(self, conn_matrix: np.ndarray, config: NetworkTopologyValidationConfig):
        """Validate connectivity matrix properties."""
        num_pre, num_post = conn_matrix.shape
        
        # Connection density check
        total_possible = num_pre * num_post
        total_connections = np.sum(conn_matrix)
        density = total_connections / total_possible
        
        if density < config.min_connection_probability:
            warning(f"Very sparse connectivity: {density:.4f}")
        elif density > config.max_connection_probability:
            raise ValidationError(
                f"Connectivity too dense: {density:.4f} > {config.max_connection_probability}",
                error_code="CONNECTIVITY_TOO_DENSE"
            )
        
        # Check for reasonable fan-in/fan-out
        fan_out = np.sum(conn_matrix, axis=1)  # Outgoing connections per neuron
        fan_in = np.sum(conn_matrix, axis=0)   # Incoming connections per neuron
        
        if np.any(fan_out > config.max_synapses_per_neuron):
            raise ValidationError(
                f"Too many outgoing connections: max={np.max(fan_out)}",
                error_code="EXCESSIVE_FAN_OUT"
            )
        
        if np.any(fan_in > config.max_synapses_per_neuron):
            raise ValidationError(
                f"Too many incoming connections: max={np.max(fan_in)}",
                error_code="EXCESSIVE_FAN_IN"
            )
    
    def _validate_weight_matrix(self, weights, conn_matrix: np.ndarray, config: NetworkTopologyValidationConfig):
        """Validate synaptic weight matrix."""
        if TORCH_AVAILABLE and isinstance(weights, torch.Tensor):
            weights_np = weights.detach().cpu().numpy()
        else:
            weights_np = np.array(weights)
        
        # Shape consistency
        if weights_np.shape != conn_matrix.shape:
            raise ValidationError(
                f"Weight matrix shape {weights_np.shape} != connectivity shape {conn_matrix.shape}",
                error_code="WEIGHT_SHAPE_MISMATCH"
            )
        
        # Weights should be zero where no connections exist
        zero_connections = (conn_matrix == 0)
        if np.any(weights_np[zero_connections] != 0):
            warning("Non-zero weights found for non-existent connections, setting to zero")
            weights_np[zero_connections] = 0
        
        # Check for invalid values
        if np.any(np.isnan(weights_np)) or np.any(np.isinf(weights_np)):
            raise ValidationError(
                "Weight matrix contains NaN or infinite values",
                error_code="INVALID_WEIGHT_VALUES"
            )
        
        # Check for extremely large weights
        if np.any(np.abs(weights_np) > 100):
            warning("Very large synaptic weights detected")
        
        return weights_np


class SecurityAwareNeuromorphicValidator(NeuromorphicValidator):
    """Security-enhanced neuromorphic validator."""
    
    def __init__(self, validation_level=None, enable_security_logging=True):
        super().__init__(validation_level)
        self.enable_security_logging = enable_security_logging
        self._register_security_rules()
    
    def _register_security_rules(self):
        """Register security-specific validation rules."""
        
        self.add_validation_rule(
            "resource_limits",
            ValidationRule(
                name="resource_limits",
                validator_func=self._validate_resource_limits,
                error_message="Resource limits exceeded",
                severity=ErrorSeverity.HIGH
            )
        )
        
        self.add_validation_rule(
            "data_integrity",
            ValidationRule(
                name="data_integrity",
                validator_func=self._validate_data_integrity,
                error_message="Data integrity check failed",
                severity=ErrorSeverity.HIGH
            )
        )
    
    def validate_with_security_check(self, data: Any, rule_types: List[str], 
                                   context: Optional[Dict[str, Any]] = None) -> Any:
        """Validate data with additional security checks."""
        
        # Standard validation
        validated_data = self.validate(data, rule_types, context)
        
        # Additional security checks
        self._perform_security_audit(validated_data, context)
        
        return validated_data
    
    def _validate_resource_limits(self, data: Any) -> bool:
        """Validate that data doesn't exceed resource limits."""
        if TORCH_AVAILABLE and isinstance(data, torch.Tensor):
            # Memory usage check
            memory_mb = data.numel() * data.element_size() / 1024 / 1024
            if memory_mb > 1000:  # 1GB limit
                return False
            
            # Dimension check
            if data.numel() > 1e8:  # 100M elements
                return False
        
        elif isinstance(data, np.ndarray):
            memory_mb = data.nbytes / 1024 / 1024
            if memory_mb > 1000:
                return False
            if data.size > 1e8:
                return False
        
        return True
    
    def _validate_data_integrity(self, data: Any) -> bool:
        """Validate data integrity."""
        # Check for suspicious patterns that might indicate malicious input
        if TORCH_AVAILABLE and isinstance(data, torch.Tensor):
            data_np = data.detach().cpu().numpy()
        elif isinstance(data, np.ndarray):
            data_np = data
        else:
            return True  # Skip for non-array data
        
        # Check for unusual patterns
        if data_np.size > 0:
            # Check for repeated patterns (possible attack)
            flat_data = data_np.flatten()
            if len(np.unique(flat_data)) < max(1, len(flat_data) // 1000):
                if self.enable_security_logging:
                    security("Highly repetitive data pattern detected")
                return False
        
        return True
    
    def _perform_security_audit(self, data: Any, context: Optional[Dict[str, Any]]):
        """Perform comprehensive security audit."""
        if not self.enable_security_logging:
            return
        
        audit_results = {
            "timestamp": time.time(),
            "data_type": type(data).__name__,
            "context": context or {},
            "checks_performed": []
        }
        
        # Resource usage audit
        if TORCH_AVAILABLE and isinstance(data, torch.Tensor):
            memory_mb = data.numel() * data.element_size() / 1024 / 1024
            audit_results["memory_usage_mb"] = memory_mb
            audit_results["tensor_shape"] = list(data.shape)
            audit_results["checks_performed"].append("tensor_resource_check")
        
        # Data pattern audit
        if isinstance(data, (np.ndarray, torch.Tensor)):
            if TORCH_AVAILABLE and isinstance(data, torch.Tensor):
                data_np = data.detach().cpu().numpy()
            else:
                data_np = data
            
            if data_np.size > 0:
                audit_results["data_range"] = {
                    "min": float(np.min(data_np)),
                    "max": float(np.max(data_np)),
                    "mean": float(np.mean(data_np))
                }
                audit_results["unique_values"] = len(np.unique(data_np))
                audit_results["checks_performed"].append("data_pattern_analysis")
        
        # Log audit results
        audit("Input validation security audit completed", structured_data=audit_results)


# Global validator instances
_neuromorphic_validator = None
_secure_neuromorphic_validator = None


def get_neuromorphic_validator() -> NeuromorphicValidator:
    """Get global neuromorphic validator."""
    global _neuromorphic_validator
    if _neuromorphic_validator is None:
        _neuromorphic_validator = NeuromorphicValidator()
    return _neuromorphic_validator


def get_secure_neuromorphic_validator() -> SecurityAwareNeuromorphicValidator:
    """Get global security-aware neuromorphic validator."""
    global _secure_neuromorphic_validator
    if _secure_neuromorphic_validator is None:
        _secure_neuromorphic_validator = SecurityAwareNeuromorphicValidator()
    return _secure_neuromorphic_validator


# Convenience functions
def validate_spike_trains(spike_data, config=None):
    """Validate spike trains using global validator."""
    return get_neuromorphic_validator().validate_spike_trains(spike_data, config)


def validate_network_topology(connectivity_matrix, weights=None, config=None):
    """Validate network topology using global validator."""
    return get_neuromorphic_validator().validate_network_topology(
        connectivity_matrix, weights, config
    )


def validate_neuromorphic_config(config_dict):
    """Validate neuromorphic configuration using global validator."""
    return get_neuromorphic_validator().validate_neuromorphic_config(config_dict)


def secure_validate(data, rule_types, context=None):
    """Perform security-aware validation."""
    return get_secure_neuromorphic_validator().validate_with_security_check(
        data, rule_types, context
    )