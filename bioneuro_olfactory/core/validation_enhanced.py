"""Enhanced input validation and error handling for neuromorphic components.

This module provides comprehensive validation with schema validation, data quality checks,
and performance monitoring for the neuromorphic gas detection system.
"""

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class MockTorch:
        Tensor = list
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
        
        # Add dtype support
        class dtype:
            float32 = 'float32'
            float64 = 'float64' 
            int32 = 'int32'
            int64 = 'int64'
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
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Type
from dataclasses import dataclass, field
import logging
from enum import Enum
import warnings
import json
import jsonschema
from pathlib import Path
import hashlib
import statistics
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import re
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    def __init__(self, message: str, error_code: str = "VALIDATION_ERROR", 
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.timestamp = datetime.now()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for structured logging."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "context": self.context,
            "timestamp": self.timestamp.isoformat()
        }


class SchemaValidationError(ValidationError):
    """Schema validation specific errors."""
    pass


class DataQualityError(ValidationError):
    """Data quality specific errors."""
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
    confidence_score: float = 1.0
    validation_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "is_valid": self.is_valid,
            "message": self.message,
            "corrected_value": self.corrected_value,
            "warning_issued": self.warning_issued,
            "confidence_score": self.confidence_score,
            "validation_time_ms": self.validation_time_ms,
            "metadata": self.metadata
        }


class BaseValidator(ABC):
    """Abstract base class for validators."""
    
    @abstractmethod
    def validate(self, data: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate data and return result."""
        pass
        
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Get validation schema."""
        pass


class SchemaValidator(BaseValidator):
    """JSON Schema-based validator."""
    
    def __init__(self, schema: Dict[str, Any], name: str = "schema_validator"):
        self.schema = schema
        self.name = name
        self.validator = jsonschema.Draft7Validator(schema)
        
    def validate(self, data: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate data against JSON schema."""
        start_time = datetime.now()
        
        try:
            # Convert data to JSON-serializable format if needed
            if hasattr(data, 'to_dict'):
                json_data = data.to_dict()
            elif isinstance(data, (torch.Tensor, np.ndarray)):
                # Handle tensor/array data
                json_data = {
                    "shape": list(data.shape),
                    "dtype": str(data.dtype),
                    "min_value": float(data.min()) if data.numel() > 0 else 0.0,
                    "max_value": float(data.max()) if data.numel() > 0 else 0.0,
                    "mean_value": float(data.mean()) if data.numel() > 0 else 0.0,
                    "std_value": float(data.std()) if data.numel() > 0 else 0.0
                }
            else:
                json_data = data
                
            # Validate against schema
            errors = list(self.validator.iter_errors(json_data))
            
            validation_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if errors:
                error_messages = [f"{err.json_path}: {err.message}" for err in errors]
                return ValidationResult(
                    is_valid=False,
                    message=f"Schema validation failed: {'; '.join(error_messages)}",
                    validation_time_ms=validation_time,
                    metadata={"schema_errors": error_messages, "schema_name": self.name}
                )
            else:
                return ValidationResult(
                    is_valid=True,
                    message="Schema validation passed",
                    validation_time_ms=validation_time,
                    metadata={"schema_name": self.name}
                )
                
        except Exception as e:
            validation_time = (datetime.now() - start_time).total_seconds() * 1000
            return ValidationResult(
                is_valid=False,
                message=f"Schema validation error: {str(e)}",
                validation_time_ms=validation_time,
                metadata={"exception": str(e), "schema_name": self.name}
            )
            
    def get_schema(self) -> Dict[str, Any]:
        """Get validation schema."""
        return self.schema


class DataQualityValidator(BaseValidator):
    """Data quality assessment validator."""
    
    def __init__(self, quality_thresholds: Optional[Dict[str, float]] = None):
        self.quality_thresholds = quality_thresholds or {
            "completeness_threshold": 0.95,
            "consistency_threshold": 0.90,
            "accuracy_threshold": 0.95,
            "validity_threshold": 0.98,
            "uniqueness_threshold": 0.95
        }
        
    def validate(self, data: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Assess data quality metrics."""
        start_time = datetime.now()
        
        try:
            quality_metrics = self._assess_quality_metrics(data, context)
            overall_score = self._calculate_overall_quality_score(quality_metrics)
            
            validation_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Check if quality meets thresholds
            quality_issues = []
            for metric, score in quality_metrics.items():
                threshold_key = f"{metric}_threshold"
                if threshold_key in self.quality_thresholds:
                    threshold = self.quality_thresholds[threshold_key]
                    if score < threshold:
                        quality_issues.append(f"{metric}: {score:.2f} < {threshold:.2f}")
            
            is_valid = len(quality_issues) == 0
            message = "Data quality assessment passed" if is_valid else f"Quality issues: {'; '.join(quality_issues)}"
            
            return ValidationResult(
                is_valid=is_valid,
                message=message,
                confidence_score=overall_score,
                validation_time_ms=validation_time,
                metadata={
                    "quality_metrics": quality_metrics,
                    "overall_quality_score": overall_score,
                    "quality_issues": quality_issues
                }
            )
            
        except Exception as e:
            validation_time = (datetime.now() - start_time).total_seconds() * 1000
            return ValidationResult(
                is_valid=False,
                message=f"Data quality assessment error: {str(e)}",
                validation_time_ms=validation_time,
                metadata={"exception": str(e)}
            )
            
    def _assess_quality_metrics(self, data: Any, context: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Assess individual data quality metrics."""
        metrics = {}
        
        if isinstance(data, (torch.Tensor, np.ndarray)):
            # Handle tensor/array data
            metrics["completeness"] = self._assess_completeness_tensor(data)
            metrics["consistency"] = self._assess_consistency_tensor(data)
            metrics["validity"] = self._assess_validity_tensor(data)
            
        elif isinstance(data, (list, tuple)):
            # Handle list/tuple data
            metrics["completeness"] = self._assess_completeness_list(data)
            metrics["uniqueness"] = self._assess_uniqueness_list(data)
            
        elif isinstance(data, dict):
            # Handle dictionary data
            metrics["completeness"] = self._assess_completeness_dict(data)
            metrics["consistency"] = self._assess_consistency_dict(data)
            
        else:
            # Default metrics for other data types
            metrics["validity"] = 1.0 if data is not None else 0.0
            
        return metrics
        
    def _assess_completeness_tensor(self, tensor: Union[torch.Tensor, np.ndarray]) -> float:
        """Assess completeness of tensor data (non-NaN, non-zero ratio)."""
        if isinstance(tensor, torch.Tensor):
            total_elements = tensor.numel()
            if total_elements == 0:
                return 0.0
            non_nan_elements = (~torch.isnan(tensor)).sum().item()
            return non_nan_elements / total_elements
        else:
            total_elements = tensor.size
            if total_elements == 0:
                return 0.0
            non_nan_elements = np.sum(~np.isnan(tensor))
            return non_nan_elements / total_elements
            
    def _assess_consistency_tensor(self, tensor: Union[torch.Tensor, np.ndarray]) -> float:
        """Assess consistency of tensor data (variance analysis)."""
        try:
            if isinstance(tensor, torch.Tensor):
                if tensor.numel() == 0:
                    return 0.0
                std_dev = torch.std(tensor).item()
                mean_val = torch.mean(tensor).item()
            else:
                if tensor.size == 0:
                    return 0.0
                std_dev = np.std(tensor)
                mean_val = np.mean(tensor)
                
            # Coefficient of variation as consistency measure
            if mean_val == 0:
                return 1.0 if std_dev == 0 else 0.0
            cv = abs(std_dev / mean_val)
            # Convert to score (lower CV = higher consistency)
            return max(0.0, 1.0 - min(cv, 1.0))
        except Exception:
            return 0.5  # Neutral score on error
            
    def _assess_validity_tensor(self, tensor: Union[torch.Tensor, np.ndarray]) -> float:
        """Assess validity of tensor data (finite values, reasonable ranges)."""
        try:
            if isinstance(tensor, torch.Tensor):
                total_elements = tensor.numel()
                if total_elements == 0:
                    return 0.0
                finite_elements = torch.isfinite(tensor).sum().item()
            else:
                total_elements = tensor.size
                if total_elements == 0:
                    return 0.0
                finite_elements = np.sum(np.isfinite(tensor))
                
            return finite_elements / total_elements
        except Exception:
            return 0.5
            
    def _assess_completeness_list(self, data: List) -> float:
        """Assess completeness of list data."""
        if not data:
            return 0.0
        non_null_count = sum(1 for item in data if item is not None)
        return non_null_count / len(data)
        
    def _assess_uniqueness_list(self, data: List) -> float:
        """Assess uniqueness of list data."""
        if not data:
            return 1.0
        unique_count = len(set(str(item) for item in data))  # Convert to string for hashing
        return unique_count / len(data)
        
    def _assess_completeness_dict(self, data: Dict) -> float:
        """Assess completeness of dictionary data."""
        if not data:
            return 0.0
        non_null_count = sum(1 for value in data.values() if value is not None)
        return non_null_count / len(data)
        
    def _assess_consistency_dict(self, data: Dict) -> float:
        """Assess consistency of dictionary data."""
        if not data:
            return 1.0
        # Check type consistency
        value_types = [type(value).__name__ for value in data.values()]
        unique_types = len(set(value_types))
        return 1.0 if unique_types <= 2 else max(0.0, 1.0 - (unique_types - 2) * 0.2)
        
    def _calculate_overall_quality_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall quality score from individual metrics."""
        if not metrics:
            return 0.0
        return sum(metrics.values()) / len(metrics)
        
    def get_schema(self) -> Dict[str, Any]:
        """Get quality assessment schema."""
        return {
            "type": "object",
            "properties": {
                "quality_thresholds": {
                    "type": "object",
                    "properties": {key: {"type": "number"} for key in self.quality_thresholds}
                }
            }
        }


class ModelValidator(BaseValidator):
    """Validator for neural network models."""
    
    def __init__(self):
        self.model_schema = {
            "type": "object",
            "properties": {
                "parameters": {
                    "type": "object",
                    "properties": {
                        "total_params": {"type": "integer", "minimum": 1},
                        "trainable_params": {"type": "integer", "minimum": 0},
                        "param_size_mb": {"type": "number", "minimum": 0}
                    }
                },
                "architecture": {
                    "type": "object",
                    "properties": {
                        "layers": {"type": "integer", "minimum": 1},
                        "input_shape": {"type": "array"},
                        "output_shape": {"type": "array"}
                    }
                },
                "performance": {
                    "type": "object",
                    "properties": {
                        "memory_usage_mb": {"type": "number", "minimum": 0},
                        "forward_pass_time_ms": {"type": "number", "minimum": 0}
                    }
                }
            },
            "required": ["parameters"]
        }
        
    def validate(self, model: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate neural network model."""
        start_time = datetime.now()
        
        try:
            # Extract model information
            model_info = self._extract_model_info(model)
            
            # Validate against schema
            schema_validator = SchemaValidator(self.model_schema, "model_schema")
            schema_result = schema_validator.validate(model_info)
            
            # Additional model-specific checks
            specific_checks = self._perform_model_specific_checks(model)
            
            validation_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Combine results
            all_valid = schema_result.is_valid and all(check["is_valid"] for check in specific_checks)
            messages = [schema_result.message] + [check["message"] for check in specific_checks]
            
            return ValidationResult(
                is_valid=all_valid,
                message="; ".join(messages),
                validation_time_ms=validation_time,
                metadata={
                    "model_info": model_info,
                    "specific_checks": specific_checks,
                    "schema_validation": schema_result.to_dict()
                }
            )
            
        except Exception as e:
            validation_time = (datetime.now() - start_time).total_seconds() * 1000
            return ValidationResult(
                is_valid=False,
                message=f"Model validation error: {str(e)}",
                validation_time_ms=validation_time,
                metadata={"exception": str(e)}
            )
            
    def _extract_model_info(self, model: Any) -> Dict[str, Any]:
        """Extract model information for validation."""
        info = {"parameters": {}, "architecture": {}, "performance": {}}
        
        # Try to extract PyTorch model info
        if hasattr(model, 'parameters'):
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            param_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
            
            info["parameters"] = {
                "total_params": total_params,
                "trainable_params": trainable_params,
                "param_size_mb": param_size_mb
            }
            
        # Extract architecture info if available
        if hasattr(model, '__len__'):
            info["architecture"]["layers"] = len(model)
            
        return info
        
    def _perform_model_specific_checks(self, model: Any) -> List[Dict[str, Any]]:
        """Perform model-specific validation checks."""
        checks = []
        
        # Check for gradient flow
        if hasattr(model, 'parameters'):
            params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
            total_params = sum(1 for p in model.parameters())
            
            if total_params > 0:
                grad_ratio = params_with_grad / total_params
                checks.append({
                    "name": "gradient_flow",
                    "is_valid": grad_ratio > 0.1,  # At least 10% of parameters should have gradients
                    "message": f"Gradient flow check: {params_with_grad}/{total_params} parameters have gradients",
                    "gradient_ratio": grad_ratio
                })
                
        # Check for parameter initialization
        if hasattr(model, 'parameters'):
            zero_params = sum(1 for p in model.parameters() if torch.all(p == 0).item())
            total_params = sum(1 for p in model.parameters())
            
            if total_params > 0:
                zero_ratio = zero_params / total_params
                checks.append({
                    "name": "parameter_initialization",
                    "is_valid": zero_ratio < 0.5,  # Less than 50% zero parameters
                    "message": f"Parameter initialization check: {zero_params}/{total_params} parameters are zero",
                    "zero_ratio": zero_ratio
                })
                
        return checks
        
    def get_schema(self) -> Dict[str, Any]:
        """Get model validation schema."""
        return self.model_schema


class EnhancedInputValidator:
    """Enhanced comprehensive input validation for neuromorphic components."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STRICT):
        self.validation_level = validation_level
        self.validation_history: List[ValidationResult] = []
        self.validators: Dict[str, BaseValidator] = {}
        self.validation_cache: Dict[str, ValidationResult] = {}
        self.cache_lock = threading.Lock()
        self.performance_metrics = defaultdict(list)
        
        # Register default validators
        self._register_default_validators()
        
    def register_validator(self, name: str, validator: BaseValidator):
        """Register a custom validator."""
        self.validators[name] = validator
        logger.info(f"Registered validator: {name}")
        
    def _register_default_validators(self):
        """Register default validators."""
        # Tensor validation schema
        tensor_schema = {
            "type": "object",
            "properties": {
                "shape": {"type": "array", "items": {"type": "integer", "minimum": 1}},
                "dtype": {"type": "string"},
                "min_value": {"type": "number"},
                "max_value": {"type": "number"},
                "mean_value": {"type": "number"},
                "std_value": {"type": "number", "minimum": 0}
            },
            "required": ["shape", "dtype"]
        }
        
        self.register_validator("tensor_schema", SchemaValidator(tensor_schema, "tensor_schema"))
        self.register_validator("data_quality", DataQualityValidator())
        self.register_validator("model", ModelValidator())
        
    def _get_cache_key(self, data: Any, validation_params: Dict[str, Any]) -> str:
        """Generate cache key for validation result."""
        try:
            # Create a hash of the data and parameters
            data_str = str(type(data).__name__)
            if hasattr(data, 'shape'):
                data_str += f"_shape_{data.shape}"
            if hasattr(data, 'dtype'):
                data_str += f"_dtype_{data.dtype}"
                
            params_str = json.dumps(validation_params, sort_keys=True, default=str)
            combined_str = data_str + params_str
            return hashlib.md5(combined_str.encode()).hexdigest()
        except Exception:
            # Fallback: no caching for problematic data
            return f"no_cache_{id(data)}"
            
    def validate_with_cache(
        self,
        data: Any,
        validator_names: List[str],
        validation_params: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> List[ValidationResult]:
        """Validate data with caching support."""
        if validation_params is None:
            validation_params = {}
            
        cache_key = self._get_cache_key(data, validation_params) if use_cache else None
        
        # Check cache
        if use_cache and cache_key:
            with self.cache_lock:
                if cache_key in self.validation_cache:
                    cached_result = self.validation_cache[cache_key]
                    # Check if cache is still fresh (within 1 hour)
                    cache_age = datetime.now() - datetime.fromisoformat(cached_result.metadata.get("cache_timestamp", "1900-01-01"))
                    if cache_age < timedelta(hours=1):
                        logger.debug(f"Using cached validation result for {cache_key}")
                        return [cached_result]
        
        # Perform validation
        results = []
        for validator_name in validator_names:
            if validator_name in self.validators:
                validator = self.validators[validator_name]
                start_time = datetime.now()
                
                try:
                    result = validator.validate(data, validation_params)
                    validation_time = (datetime.now() - start_time).total_seconds() * 1000
                    
                    # Add performance metrics
                    self.performance_metrics[validator_name].append(validation_time)
                    
                    # Add cache timestamp to metadata
                    if cache_key:
                        result.metadata["cache_timestamp"] = datetime.now().isoformat()
                        result.metadata["cache_key"] = cache_key
                        
                    results.append(result)
                    self.validation_history.append(result)
                    
                except Exception as e:
                    error_result = ValidationResult(
                        is_valid=False,
                        message=f"Validation error in {validator_name}: {str(e)}",
                        validation_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                        metadata={"validator_name": validator_name, "exception": str(e)}
                    )
                    results.append(error_result)
                    self.validation_history.append(error_result)
                    
        # Update cache
        if use_cache and cache_key and results:
            with self.cache_lock:
                # Cache the combined result
                combined_result = self._combine_validation_results(results)
                self.validation_cache[cache_key] = combined_result
                
                # Limit cache size
                if len(self.validation_cache) > 1000:
                    # Remove oldest 20% of cache entries
                    keys_to_remove = list(self.validation_cache.keys())[:200]
                    for key in keys_to_remove:
                        del self.validation_cache[key]
                        
        return results
        
    def _combine_validation_results(self, results: List[ValidationResult]) -> ValidationResult:
        """Combine multiple validation results into one."""
        if not results:
            return ValidationResult(is_valid=False, message="No validation results")
            
        all_valid = all(r.is_valid for r in results)
        messages = [r.message for r in results]
        combined_metadata = {}
        total_time = sum(r.validation_time_ms for r in results)
        avg_confidence = sum(r.confidence_score for r in results) / len(results)
        
        # Combine metadata
        for result in results:
            for key, value in result.metadata.items():
                if key not in combined_metadata:
                    combined_metadata[key] = []
                combined_metadata[key].append(value)
                
        return ValidationResult(
            is_valid=all_valid,
            message="; ".join(messages),
            confidence_score=avg_confidence,
            validation_time_ms=total_time,
            metadata=combined_metadata
        )
    
    def validate_tensor(
        self, 
        tensor: torch.Tensor,
        name: str,
        expected_shape: Optional[Tuple[int, ...]] = None,
        expected_dtype: Optional[torch.dtype] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        allow_nan: bool = False,
        allow_inf: bool = False,
        use_cache: bool = True
    ) -> ValidationResult:
        """Enhanced tensor validation with schema and quality checks."""
        start_time = datetime.now()
        
        try:
            # Basic tensor validation
            if not isinstance(tensor, torch.Tensor):
                return self._handle_validation_error(
                    f"{name} must be a torch.Tensor, got {type(tensor)}",
                    validation_time=(datetime.now() - start_time).total_seconds() * 1000
                )
                
            # Prepare validation parameters
            validation_params = {
                "name": name,
                "expected_shape": expected_shape,
                "expected_dtype": str(expected_dtype) if expected_dtype else None,
                "min_value": min_value,
                "max_value": max_value,
                "allow_nan": allow_nan,
                "allow_inf": allow_inf
            }
            
            # Use enhanced validation with schema and quality checks
            validation_results = self.validate_with_cache(
                tensor, 
                ["tensor_schema", "data_quality"], 
                validation_params,
                use_cache
            )
            
            # Perform specific tensor validations
            specific_validations = self._perform_specific_tensor_validation(
                tensor, name, expected_shape, expected_dtype, 
                min_value, max_value, allow_nan, allow_inf
            )
            
            # Combine all validation results
            all_results = validation_results + [specific_validations]
            combined_result = self._combine_validation_results(all_results)
            
            # Add timing information
            validation_time = (datetime.now() - start_time).total_seconds() * 1000
            combined_result.validation_time_ms = validation_time
            
            return combined_result
            
        except Exception as e:
            validation_time = (datetime.now() - start_time).total_seconds() * 1000
            return self._handle_validation_error(
                f"Unexpected error validating {name}: {e}",
                validation_time=validation_time
            )
            
    def _perform_specific_tensor_validation(
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
        """Perform specific tensor validations."""
        issues = []
        corrected_tensor = None
        warnings_issued = False
        
        # Check shape
        if expected_shape is not None and tensor.shape != expected_shape:
            issues.append(f"Shape mismatch: expected {expected_shape}, got {tensor.shape}")
                
        # Check data type
        if expected_dtype is not None and tensor.dtype != expected_dtype:
            if self.validation_level == ValidationLevel.STRICT:
                issues.append(f"Dtype mismatch: expected {expected_dtype}, got {tensor.dtype}")
            else:
                # Try to convert
                try:
                    corrected_tensor = tensor.to(expected_dtype)
                    warnings_issued = True
                except Exception as e:
                    issues.append(f"Could not convert dtype: {e}")
                        
        # Check for NaN values
        if not allow_nan and tensor.numel() > 0:
            nan_count = torch.isnan(tensor).sum().item()
            if nan_count > 0:
                issues.append(f"Contains {nan_count} NaN values")
                
        # Check for infinite values
        if not allow_inf and tensor.numel() > 0:
            inf_count = torch.isinf(tensor).sum().item()
            if inf_count > 0:
                issues.append(f"Contains {inf_count} infinite values")
                
        # Check value range
        if tensor.numel() > 0:
            tensor_min = tensor.min().item()
            tensor_max = tensor.max().item()
            
            if min_value is not None and tensor_min < min_value:
                if self.validation_level == ValidationLevel.STRICT:
                    issues.append(f"Contains values below minimum {min_value}: {tensor_min}")
                else:
                    # Clamp values
                    if corrected_tensor is None:
                        corrected_tensor = tensor.clone()
                    corrected_tensor = torch.clamp(corrected_tensor, min=min_value)
                    warnings_issued = True
                    
            if max_value is not None and tensor_max > max_value:
                if self.validation_level == ValidationLevel.STRICT:
                    issues.append(f"Contains values above maximum {max_value}: {tensor_max}")
                else:
                    # Clamp values
                    if corrected_tensor is None:
                        corrected_tensor = tensor.clone()
                    corrected_tensor = torch.clamp(corrected_tensor, max=max_value)
                    warnings_issued = True
        
        # Calculate confidence score
        confidence = 1.0 - (len(issues) * 0.2)  # Reduce confidence for each issue
        confidence = max(0.0, confidence)
        
        if issues:
            return ValidationResult(
                is_valid=False,
                message=f"{name} validation failed: {'; '.join(issues)}",
                corrected_value=corrected_tensor,
                warning_issued=warnings_issued,
                confidence_score=confidence,
                metadata={"issues": issues, "tensor_name": name}
            )
        else:
            return ValidationResult(
                is_valid=True,
                message=f"{name} validation passed",
                corrected_value=corrected_tensor,
                warning_issued=warnings_issued,
                confidence_score=confidence,
                metadata={"tensor_name": name}
            )
            
    def validate_spike_train(
        self,
        spike_train: torch.Tensor,
        name: str = "spike_train",
        use_cache: bool = True
    ) -> ValidationResult:
        """Validate spike train tensor with enhanced checks."""
        # Basic tensor validation
        result = self.validate_tensor(
            spike_train, name,
            expected_dtype=torch.float32,
            min_value=0.0,
            max_value=1.0,
            use_cache=use_cache
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
        
    def validate_configuration_schema(self, config_data: Dict[str, Any], schema: Dict[str, Any]) -> ValidationResult:
        """Validate configuration data against a schema."""
        schema_validator = SchemaValidator(schema, "config_schema")
        return schema_validator.validate(config_data)
        
    def _handle_validation_error(self, message: str, validation_time: float = 0.0) -> ValidationResult:
        """Handle validation error based on validation level."""
        result = ValidationResult(
            is_valid=False, 
            message=message, 
            validation_time_ms=validation_time,
            metadata={"validation_level": self.validation_level.value}
        )
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
        
    def get_validation_summary(self, time_window_hours: Optional[int] = None) -> Dict[str, Any]:
        """Get comprehensive summary of validation history."""
        # Filter by time window if specified
        validations_to_analyze = self.validation_history
        if time_window_hours:
            cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
            validations_to_analyze = [
                r for r in self.validation_history
                if hasattr(r, 'metadata') and 'timestamp' in r.metadata
                and datetime.fromisoformat(r.metadata['timestamp']) > cutoff_time
            ]
        
        total_validations = len(validations_to_analyze)
        successful_validations = sum(1 for r in validations_to_analyze if r.is_valid)
        warnings_issued = sum(1 for r in validations_to_analyze if r.warning_issued)
        corrections_made = sum(1 for r in validations_to_analyze if r.corrected_value is not None)
        
        # Performance metrics
        validation_times = [r.validation_time_ms for r in validations_to_analyze if r.validation_time_ms > 0]
        avg_validation_time = statistics.mean(validation_times) if validation_times else 0.0
        max_validation_time = max(validation_times) if validation_times else 0.0
        
        # Confidence metrics
        confidence_scores = [r.confidence_score for r in validations_to_analyze]
        avg_confidence = statistics.mean(confidence_scores) if confidence_scores else 0.0
        
        # Validator performance
        validator_performance = {}
        for validator_name, times in self.performance_metrics.items():
            if times:
                validator_performance[validator_name] = {
                    "avg_time_ms": statistics.mean(times),
                    "max_time_ms": max(times),
                    "min_time_ms": min(times),
                    "total_calls": len(times)
                }
        
        return {
            'total_validations': total_validations,
            'successful_validations': successful_validations,
            'failed_validations': total_validations - successful_validations,
            'warnings_issued': warnings_issued,
            'corrections_made': corrections_made,
            'success_rate': successful_validations / max(total_validations, 1),
            'average_validation_time_ms': avg_validation_time,
            'max_validation_time_ms': max_validation_time,
            'average_confidence_score': avg_confidence,
            'cache_size': len(self.validation_cache),
            'registered_validators': list(self.validators.keys()),
            'validator_performance': validator_performance,
            'time_window_hours': time_window_hours
        }
        
    def clear_history(self):
        """Clear validation history and cache."""
        self.validation_history.clear()
        with self.cache_lock:
            self.validation_cache.clear()
        self.performance_metrics.clear()
        logger.info("Validation history and cache cleared")
        
    def export_validation_report(self, file_path: str, include_history: bool = False):
        """Export comprehensive validation report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": self.get_validation_summary(),
            "registered_validators": {name: validator.get_schema() for name, validator in self.validators.items()},
            "cache_statistics": {
                "cache_size": len(self.validation_cache),
                "cache_hit_ratio": self._calculate_cache_hit_ratio()
            }
        }
        
        if include_history:
            report["validation_history"] = [r.to_dict() for r in self.validation_history]
            
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        logger.info(f"Validation report exported to {file_path}")
        
    def _calculate_cache_hit_ratio(self) -> float:
        """Calculate cache hit ratio from performance metrics."""
        # This is a simplified calculation - in production you'd track actual hits/misses
        total_validations = len(self.validation_history)
        cache_size = len(self.validation_cache)
        
        if total_validations == 0:
            return 0.0
            
        # Estimate cache hits based on cache size vs total validations
        estimated_hits = min(cache_size, total_validations * 0.3)  # Assume 30% hit rate
        return estimated_hits / total_validations


class RobustnessChecker:
    """Enhanced system robustness and stability checker."""
    
    def __init__(self):
        self.check_history: List[Dict[str, Any]] = []
        
    def check_numerical_stability(
        self,
        tensor: torch.Tensor,
        name: str = "tensor",
        max_gradient: float = 1000.0
    ) -> Dict[str, Any]:
        """Check for numerical stability issues with enhanced analysis."""
        result = {
            'name': name,
            'timestamp': datetime.now().isoformat(),
            'is_stable': True,
            'issues': [],
            'stability_score': 1.0
        }
        
        issues_count = 0
        
        # Check for NaN or Inf
        if torch.isnan(tensor).any():
            result['is_stable'] = False
            result['issues'].append('NaN values detected')
            issues_count += 1
            
        if torch.isinf(tensor).any():
            result['is_stable'] = False
            result['issues'].append('Infinite values detected')
            issues_count += 1
            
        # Check dynamic range
        if tensor.numel() > 0:
            tensor_range = tensor.max().item() - tensor.min().item()
            if tensor_range > 1e6:
                result['issues'].append(f'Large dynamic range: {tensor_range:.2e}')
                issues_count += 1
                
        # Check gradient magnitude (if tensor requires grad)
        if tensor.requires_grad and tensor.grad is not None:
            grad_norm = torch.norm(tensor.grad).item()
            if grad_norm > max_gradient:
                result['is_stable'] = False
                result['issues'].append(f'Large gradient magnitude: {grad_norm:.2e}')
                issues_count += 1
                
        # Calculate stability score
        result['stability_score'] = max(0.0, 1.0 - issues_count * 0.25)
        
        # Store enhanced statistics
        result['statistics'] = {
            'mean': tensor.mean().item() if tensor.numel() > 0 else 0.0,
            'std': tensor.std().item() if tensor.numel() > 0 else 0.0,
            'min': tensor.min().item() if tensor.numel() > 0 else 0.0,
            'max': tensor.max().item() if tensor.numel() > 0 else 0.0,
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype),
            'device': str(tensor.device),
            'memory_usage_mb': tensor.element_size() * tensor.numel() / (1024 * 1024)
        }
        
        self.check_history.append(result)
        return result


# Enhanced global instances with thread safety
_default_validator = None
_default_validator_lock = threading.Lock()
robustness_checker = RobustnessChecker()


def get_default_validator() -> EnhancedInputValidator:
    """Get thread-safe default validator instance."""
    global _default_validator
    
    if _default_validator is None:
        with _default_validator_lock:
            if _default_validator is None:
                _default_validator = EnhancedInputValidator(ValidationLevel.WARN)
                
    return _default_validator


def configure_default_validator(
    validation_level: ValidationLevel = ValidationLevel.WARN,
    additional_validators: Optional[Dict[str, BaseValidator]] = None
) -> EnhancedInputValidator:
    """Configure the default validator with custom settings."""
    global _default_validator
    
    with _default_validator_lock:
        _default_validator = EnhancedInputValidator(validation_level)
        
        if additional_validators:
            for name, validator in additional_validators.items():
                _default_validator.register_validator(name, validator)
                
    return _default_validator


def validate_and_correct(
    tensor: torch.Tensor,
    name: str,
    **validation_kwargs
) -> torch.Tensor:
    """Enhanced convenience function for validation with automatic correction."""
    validator = get_default_validator()
    result = validator.validate_tensor(tensor, name, **validation_kwargs)
    
    if result.corrected_value is not None:
        logger.info(f"Applied corrections to {name}: {result.message}")
        return result.corrected_value
    elif result.is_valid:
        return tensor
    else:
        # Log detailed validation failure
        logger.error(
            f"Validation failed for {name}: {result.message}",
            extra={
                "structured_data": {
                    "event_type": "validation_failed",
                    "tensor_name": name,
                    "confidence_score": result.confidence_score,
                    "validation_time_ms": result.validation_time_ms,
                    "metadata": result.metadata
                }
            }
        )
        return tensor


def create_custom_validator(name: str, validation_func: Callable, schema: Optional[Dict[str, Any]] = None) -> BaseValidator:
    """Create a custom validator from a function."""
    
    class CustomValidator(BaseValidator):
        def __init__(self, func: Callable, validator_name: str, validator_schema: Optional[Dict[str, Any]] = None):
            self.func = func
            self.name = validator_name
            self.schema = validator_schema or {"type": "object"}
            
        def validate(self, data: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
            start_time = datetime.now()
            
            try:
                is_valid, message, corrections = self.func(data, context)
                validation_time = (datetime.now() - start_time).total_seconds() * 1000
                
                return ValidationResult(
                    is_valid=is_valid,
                    message=message,
                    corrected_value=corrections,
                    validation_time_ms=validation_time,
                    metadata={"custom_validator": self.name}
                )
            except Exception as e:
                validation_time = (datetime.now() - start_time).total_seconds() * 1000
                return ValidationResult(
                    is_valid=False,
                    message=f"Custom validation error: {str(e)}",
                    validation_time_ms=validation_time,
                    metadata={"custom_validator": self.name, "exception": str(e)}
                )
                
        def get_schema(self) -> Dict[str, Any]:
            return self.schema
            
    return CustomValidator(validation_func, name, schema)


# Initialize default validator on import
get_default_validator()

# Legacy compatibility - keep old name but use enhanced version  
default_validator = get_default_validator()