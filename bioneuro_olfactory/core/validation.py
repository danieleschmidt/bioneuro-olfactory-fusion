"""Input validation and error handling for neuromorphic components.

This module provides comprehensive validation and error handling
to ensure robust operation of the neuromorphic gas detection system.
"""

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class MockTorch:
        Tensor = list
        dtype = object
        float32 = float
        zeros = lambda *args, **kwargs: [0] * (args[0] if args else 1)  
        ones = lambda *args, **kwargs: [1] * (args[0] if args else 1)
        exp = lambda x: 0.9 if isinstance(x, (int, float)) else x
        clamp = lambda x, min=None, max=None: x
        randn = lambda *args, **kwargs: [0.1] * (args[0] if args else 1)  
        rand = lambda *args, **kwargs: [0.5] * (args[0] if args else 1)
        tensor = lambda x: x
        randint = lambda low, high, size: [low] * size[0] if hasattr(size, '__iter__') else [low]
        cat = lambda tensors, dim=0: sum(tensors, [])
        sum = lambda x, dim=None: x
        mean = lambda x, dim=None: x
        max = lambda x, dim=None: (x, [0])
        zeros_like = lambda x: []
        full_like = lambda x, fill_value: []
        where = lambda condition, x, y: []
        arange = lambda *args, **kwargs: []
        sin = lambda x: x
        linspace = lambda *args, **kwargs: []
        sigmoid = lambda x: x
        nn = type('nn', (), {
            'Module': object, 
            'Linear': object, 
            'Parameter': lambda x: x, 
            'init': type('init', (), {
                'xavier_uniform_': lambda x: x, 
                'zeros_': lambda x: x
            })()
        })()
        def is_tensor(x):
            return False
    torch = MockTorch()
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from enum import Enum
import warnings

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class ValidationLevel(Enum):
    """Validation strictness levels."""
    STRICT = "strict"        # Raise exceptions on any validation failure
    WARN = "warn"           # Issue warnings but continue
    SILENT = "silent"       # No warnings, best-effort correction


@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    message: str
    corrected_value: Optional[Any] = None
    warning_issued: bool = False


class InputValidator:
    """Comprehensive input validation for neuromorphic components."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STRICT):
        self.validation_level = validation_level
        self.validation_history: List[ValidationResult] = []
        
    def validate_tensor(
        self, 
        tensor: torch.Tensor,
        name: str,
        expected_shape: Optional[Tuple[int, ...]] = None,
        expected_dtype: Optional[torch.dtype] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        allow_nan: bool = False,
        allow_inf: bool = False
    ) -> ValidationResult:
        """Validate tensor properties.
        
        Args:
            tensor: Tensor to validate
            name: Name for error messages
            expected_shape: Expected tensor shape (None for any shape)
            expected_dtype: Expected data type
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            allow_nan: Whether NaN values are allowed
            allow_inf: Whether infinite values are allowed
            
        Returns:
            ValidationResult with validation outcome
        """
        try:
            # Check if tensor is actually a tensor
            if not isinstance(tensor, torch.Tensor):
                return self._handle_validation_error(
                    f"{name} must be a torch.Tensor, got {type(tensor)}"
                )
                
            # Check shape
            if expected_shape is not None and tensor.shape != expected_shape:
                return self._handle_validation_error(
                    f"{name} shape mismatch: expected {expected_shape}, got {tensor.shape}"
                )
                
            # Check data type
            if expected_dtype is not None and tensor.dtype != expected_dtype:
                if self.validation_level == ValidationLevel.STRICT:
                    return self._handle_validation_error(
                        f"{name} dtype mismatch: expected {expected_dtype}, got {tensor.dtype}"
                    )
                else:
                    # Try to convert
                    try:
                        corrected_tensor = tensor.to(expected_dtype)
                        return ValidationResult(
                            is_valid=True,
                            message=f"{name} dtype converted from {tensor.dtype} to {expected_dtype}",
                            corrected_value=corrected_tensor,
                            warning_issued=True
                        )
                    except Exception as e:
                        return self._handle_validation_error(
                            f"Could not convert {name} from {tensor.dtype} to {expected_dtype}: {e}"
                        )
                        
            # Check for NaN values
            if not allow_nan and torch.isnan(tensor).any():
                return self._handle_validation_error(f"{name} contains NaN values")
                
            # Check for infinite values
            if not allow_inf and torch.isinf(tensor).any():
                return self._handle_validation_error(f"{name} contains infinite values")
                
            # Check value range
            if min_value is not None and tensor.min().item() < min_value:
                if self.validation_level == ValidationLevel.STRICT:
                    return self._handle_validation_error(
                        f"{name} contains values below minimum {min_value}"
                    )
                else:
                    # Clamp values
                    corrected_tensor = torch.clamp(tensor, min=min_value)
                    return ValidationResult(
                        is_valid=True,
                        message=f"{name} values clamped to minimum {min_value}",
                        corrected_value=corrected_tensor,
                        warning_issued=True
                    )
                    
            if max_value is not None and tensor.max().item() > max_value:
                if self.validation_level == ValidationLevel.STRICT:
                    return self._handle_validation_error(
                        f"{name} contains values above maximum {max_value}"
                    )
                else:
                    # Clamp values
                    corrected_tensor = torch.clamp(tensor, max=max_value)
                    return ValidationResult(
                        is_valid=True,
                        message=f"{name} values clamped to maximum {max_value}",
                        corrected_value=corrected_tensor,
                        warning_issued=True
                    )
                    
            return ValidationResult(is_valid=True, message=f"{name} validation passed")
            
        except Exception as e:
            return self._handle_validation_error(f"Unexpected error validating {name}: {e}")
            
    def validate_spike_train(
        self,
        spike_train: torch.Tensor,
        name: str = "spike_train"
    ) -> ValidationResult:
        """Validate spike train tensor.
        
        Args:
            spike_train: Spike train tensor to validate
            name: Name for error messages
            
        Returns:
            ValidationResult with validation outcome
        """
        # Basic tensor validation
        result = self.validate_tensor(
            spike_train, name,
            expected_dtype=torch.float32,
            min_value=0.0,
            max_value=1.0
        )
        
        if not result.is_valid:
            return result
            
        # Use corrected tensor if available
        tensor_to_check = result.corrected_value if result.corrected_value is not None else spike_train
        
        # Check if values are binary (0 or 1)
        unique_values = torch.unique(tensor_to_check)
        non_binary_mask = (unique_values != 0) & (unique_values != 1)
        
        if non_binary_mask.any():
            if self.validation_level == ValidationLevel.STRICT:
                return self._handle_validation_error(
                    f"{name} must contain only binary values (0 or 1), found: {unique_values[non_binary_mask]}"
                )
            else:
                # Binarize the tensor
                corrected_tensor = (tensor_to_check > 0.5).float()
                return ValidationResult(
                    is_valid=True,
                    message=f"{name} binarized using threshold 0.5",
                    corrected_value=corrected_tensor,
                    warning_issued=True
                )
                
        return ValidationResult(is_valid=True, message=f"{name} spike train validation passed")
        
    def validate_concentration(
        self,
        concentration: Union[float, torch.Tensor],
        name: str = "concentration",
        max_concentration: float = 50000.0  # 5% by volume
    ) -> ValidationResult:
        """Validate gas concentration values.
        
        Args:
            concentration: Concentration value(s) in ppm
            name: Name for error messages
            max_concentration: Maximum allowed concentration in ppm
            
        Returns:
            ValidationResult with validation outcome
        """
        if isinstance(concentration, (int, float)):
            concentration = torch.tensor(concentration, dtype=torch.float32)
            
        return self.validate_tensor(
            concentration, name,
            expected_dtype=torch.float32,
            min_value=0.0,
            max_value=max_concentration
        )
        
    def validate_neuron_parameters(
        self,
        tau_membrane: float,
        threshold: float,
        dt: float,
        name_prefix: str = "neuron"
    ) -> List[ValidationResult]:
        """Validate neuron model parameters.
        
        Args:
            tau_membrane: Membrane time constant in ms
            threshold: Spike threshold
            dt: Time step in ms
            name_prefix: Prefix for parameter names
            
        Returns:
            List of ValidationResult objects
        """
        results = []
        
        # Validate tau_membrane
        if tau_membrane <= 0:
            results.append(self._handle_validation_error(
                f"{name_prefix}_tau_membrane must be positive, got {tau_membrane}"
            ))
        elif tau_membrane > 1000:
            results.append(ValidationResult(
                is_valid=True,
                message=f"{name_prefix}_tau_membrane is very large ({tau_membrane} ms), consider reducing",
                warning_issued=True
            ))
        else:
            results.append(ValidationResult(
                is_valid=True,
                message=f"{name_prefix}_tau_membrane validation passed"
            ))
            
        # Validate threshold
        if threshold <= 0:
            results.append(self._handle_validation_error(
                f"{name_prefix}_threshold must be positive, got {threshold}"
            ))
        else:
            results.append(ValidationResult(
                is_valid=True,
                message=f"{name_prefix}_threshold validation passed"
            ))
            
        # Validate dt
        if dt <= 0:
            results.append(self._handle_validation_error(
                f"{name_prefix}_dt must be positive, got {dt}"
            ))
        elif dt > tau_membrane:
            results.append(ValidationResult(
                is_valid=True,
                message=f"{name_prefix}_dt ({dt} ms) is larger than tau_membrane ({tau_membrane} ms), may cause instability",
                warning_issued=True
            ))
        else:
            results.append(ValidationResult(
                is_valid=True,
                message=f"{name_prefix}_dt validation passed"
            ))
            
        return results
        
    def validate_sensor_configuration(
        self,
        config: Dict[str, Any],
        name: str = "sensor_config"
    ) -> List[ValidationResult]:
        """Validate sensor configuration dictionary.
        
        Args:
            config: Sensor configuration dictionary
            name: Name for error messages
            
        Returns:
            List of ValidationResult objects
        """
        results = []
        required_fields = ['name', 'type', 'target_gases', 'range']
        
        # Check required fields
        for field in required_fields:
            if field not in config:
                results.append(self._handle_validation_error(
                    f"{name} missing required field: {field}"
                ))
                continue
                
            # Validate field types
            if field == 'name' and not isinstance(config[field], str):
                results.append(self._handle_validation_error(
                    f"{name}.{field} must be string, got {type(config[field])}"
                ))
            elif field == 'type' and not isinstance(config[field], str):
                results.append(self._handle_validation_error(
                    f"{name}.{field} must be string, got {type(config[field])}"
                ))
            elif field == 'target_gases' and not isinstance(config[field], list):
                results.append(self._handle_validation_error(
                    f"{name}.{field} must be list, got {type(config[field])}"
                ))
            elif field == 'range' and not isinstance(config[field], (list, tuple)):
                results.append(self._handle_validation_error(
                    f"{name}.{field} must be list or tuple, got {type(config[field])}"
                ))
            elif field == 'range' and len(config[field]) != 2:
                results.append(self._handle_validation_error(
                    f"{name}.{field} must have exactly 2 elements, got {len(config[field])}"
                ))
            elif field == 'range' and config[field][0] >= config[field][1]:
                results.append(self._handle_validation_error(
                    f"{name}.{field} min value must be less than max value"
                ))
            else:
                results.append(ValidationResult(
                    is_valid=True,
                    message=f"{name}.{field} validation passed"
                ))
                
        # Validate optional fields if present
        if 'response_time' in config and config['response_time'] <= 0:
            results.append(self._handle_validation_error(
                f"{name}.response_time must be positive, got {config['response_time']}"
            ))
            
        if 'sensitivity' in config and config['sensitivity'] <= 0:
            results.append(self._handle_validation_error(
                f"{name}.sensitivity must be positive, got {config['sensitivity']}"
            ))
            
        return results
        
    def validate_fusion_inputs(
        self,
        chemical_features: torch.Tensor,
        audio_features: torch.Tensor
    ) -> List[ValidationResult]:
        """Validate multi-modal fusion inputs.
        
        Args:
            chemical_features: Chemical sensor features
            audio_features: Audio features
            
        Returns:
            List of ValidationResult objects
        """
        results = []
        
        # Validate chemical features
        chem_result = self.validate_tensor(
            chemical_features,
            "chemical_features",
            expected_dtype=torch.float32,
            allow_nan=False,
            allow_inf=False
        )
        results.append(chem_result)
        
        # Validate audio features
        audio_result = self.validate_tensor(
            audio_features,
            "audio_features", 
            expected_dtype=torch.float32,
            allow_nan=False,
            allow_inf=False
        )
        results.append(audio_result)
        
        # Check batch size consistency
        if chem_result.is_valid and audio_result.is_valid:
            chem_tensor = chem_result.corrected_value or chemical_features
            audio_tensor = audio_result.corrected_value or audio_features
            
            if chem_tensor.shape[0] != audio_tensor.shape[0]:
                results.append(self._handle_validation_error(
                    f"Batch size mismatch: chemical_features has {chem_tensor.shape[0]} samples, "
                    f"audio_features has {audio_tensor.shape[0]} samples"
                ))
            else:
                results.append(ValidationResult(
                    is_valid=True,
                    message="Batch size consistency check passed"
                ))
                
        return results
        
    def _handle_validation_error(self, message: str) -> ValidationResult:
        """Handle validation error based on validation level.
        
        Args:
            message: Error message
            
        Returns:
            ValidationResult object
        """
        result = ValidationResult(is_valid=False, message=message)
        self.validation_history.append(result)
        
        if self.validation_level == ValidationLevel.STRICT:
            logger.error(message)
            raise ValidationError(message)
        elif self.validation_level == ValidationLevel.WARN:
            logger.warning(message)
            warnings.warn(message, UserWarning)
            result.warning_issued = True
        # SILENT level: no action taken
        
        return result
        
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation history.
        
        Returns:
            Dictionary with validation statistics
        """
        total_validations = len(self.validation_history)
        successful_validations = sum(1 for r in self.validation_history if r.is_valid)
        warnings_issued = sum(1 for r in self.validation_history if r.warning_issued)
        corrections_made = sum(1 for r in self.validation_history if r.corrected_value is not None)
        
        return {
            'total_validations': total_validations,
            'successful_validations': successful_validations,
            'failed_validations': total_validations - successful_validations,
            'warnings_issued': warnings_issued,
            'corrections_made': corrections_made,
            'success_rate': successful_validations / max(total_validations, 1)
        }
        
    def clear_history(self):
        """Clear validation history."""
        self.validation_history.clear()


class RobustnessChecker:
    """Check system robustness and stability."""
    
    def __init__(self):
        self.check_history: List[Dict[str, Any]] = []
        
    def check_numerical_stability(
        self,
        tensor: torch.Tensor,
        name: str = "tensor",
        max_gradient: float = 1000.0
    ) -> Dict[str, Any]:
        """Check for numerical stability issues.
        
        Args:
            tensor: Tensor to check
            name: Name for identification
            max_gradient: Maximum allowed gradient magnitude
            
        Returns:
            Dictionary with stability metrics
        """
        result = {
            'name': name,
            'timestamp': torch.tensor(torch.timestamp()),
            'is_stable': True,
            'issues': []
        }
        
        # Check for NaN or Inf
        if torch.isnan(tensor).any():
            result['is_stable'] = False
            result['issues'].append('NaN values detected')
            
        if torch.isinf(tensor).any():
            result['is_stable'] = False
            result['issues'].append('Infinite values detected')
            
        # Check dynamic range
        if tensor.numel() > 0:
            tensor_range = tensor.max().item() - tensor.min().item()
            if tensor_range > 1e6:
                result['issues'].append(f'Large dynamic range: {tensor_range:.2e}')
                
        # Check gradient magnitude (if tensor requires grad)
        if tensor.requires_grad and tensor.grad is not None:
            grad_norm = torch.norm(tensor.grad).item()
            if grad_norm > max_gradient:
                result['is_stable'] = False
                result['issues'].append(f'Large gradient magnitude: {grad_norm:.2e}')
                
        # Store statistics
        result['statistics'] = {
            'mean': tensor.mean().item() if tensor.numel() > 0 else 0.0,
            'std': tensor.std().item() if tensor.numel() > 0 else 0.0,
            'min': tensor.min().item() if tensor.numel() > 0 else 0.0,
            'max': tensor.max().item() if tensor.numel() > 0 else 0.0,
            'shape': list(tensor.shape)
        }
        
        self.check_history.append(result)
        return result
        
    def check_spike_train_validity(
        self,
        spike_train: torch.Tensor,
        expected_rate_range: Tuple[float, float] = (0.1, 500.0),
        name: str = "spike_train"
    ) -> Dict[str, Any]:
        """Check validity of spike train.
        
        Args:
            spike_train: Spike train tensor [batch, neurons, time]
            expected_rate_range: Expected firing rate range in Hz
            name: Name for identification
            
        Returns:
            Dictionary with validity metrics
        """
        result = {
            'name': name,
            'is_valid': True,
            'issues': [],
            'statistics': {}
        }
        
        # Basic shape check
        if spike_train.dim() != 3:
            result['is_valid'] = False
            result['issues'].append(f'Expected 3D tensor, got {spike_train.dim()}D')
            return result
            
        batch_size, num_neurons, duration = spike_train.shape
        
        # Calculate firing rates (assuming 1ms time steps)
        firing_rates = spike_train.sum(dim=-1).float() / (duration / 1000.0)  # Hz
        
        mean_rate = firing_rates.mean().item()
        min_rate = firing_rates.min().item()
        max_rate = firing_rates.max().item()
        
        # Check firing rate range
        if mean_rate < expected_rate_range[0]:
            result['issues'].append(f'Mean firing rate too low: {mean_rate:.2f} Hz')
        elif mean_rate > expected_rate_range[1]:
            result['issues'].append(f'Mean firing rate too high: {mean_rate:.2f} Hz')
            
        # Check for completely silent neurons
        silent_neurons = (firing_rates == 0).sum().item()
        if silent_neurons > num_neurons * 0.5:
            result['issues'].append(f'Too many silent neurons: {silent_neurons}/{num_neurons}')
            
        # Check for over-active neurons
        overactive_neurons = (firing_rates > expected_rate_range[1]).sum().item()
        if overactive_neurons > num_neurons * 0.1:
            result['issues'].append(f'Too many overactive neurons: {overactive_neurons}/{num_neurons}')
            
        # Store statistics
        result['statistics'] = {
            'mean_firing_rate': mean_rate,
            'min_firing_rate': min_rate,
            'max_firing_rate': max_rate,
            'silent_neurons': silent_neurons,
            'overactive_neurons': overactive_neurons,
            'sparsity': 1.0 - spike_train.mean().item()
        }
        
        if result['issues']:
            result['is_valid'] = False
            
        self.check_history.append(result)
        return result
        
    def get_robustness_summary(self) -> Dict[str, Any]:
        """Get summary of robustness checks.
        
        Returns:
            Dictionary with robustness statistics
        """
        if not self.check_history:
            return {'message': 'No robustness checks performed yet'}
            
        stable_checks = sum(1 for check in self.check_history if check.get('is_stable', check.get('is_valid', True)))
        total_checks = len(self.check_history)
        
        return {
            'total_checks': total_checks,
            'stable_checks': stable_checks,
            'unstable_checks': total_checks - stable_checks,
            'stability_rate': stable_checks / total_checks,
            'recent_issues': [
                check['issues'] for check in self.check_history[-5:] 
                if check.get('issues', [])
            ]
        }


# Global instances
default_validator = InputValidator(ValidationLevel.WARN)
robustness_checker = RobustnessChecker()


def validate_and_correct(
    tensor: torch.Tensor,
    name: str,
    **validation_kwargs
) -> torch.Tensor:
    """Convenience function for validation with automatic correction.
    
    Args:
        tensor: Tensor to validate
        name: Name for error messages
        **validation_kwargs: Additional validation parameters
        
    Returns:
        Validated (and potentially corrected) tensor
    """
    result = default_validator.validate_tensor(tensor, name, **validation_kwargs)
    
    if result.corrected_value is not None:
        return result.corrected_value
    elif result.is_valid:
        return tensor
    else:
        # This should only happen in SILENT mode
        logger.error(f"Validation failed for {name}: {result.message}")
        return tensor