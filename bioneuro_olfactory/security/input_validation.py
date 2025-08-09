"""Advanced input validation and sanitization for security."""

import re
import hashlib
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import json

from ..core.error_handling import ValidationError, SecurityError, ErrorSeverity


class ValidationType(Enum):
    """Types of validation to perform."""
    BASIC = "basic"
    STRICT = "strict"
    PARANOID = "paranoid"


@dataclass
class ValidationRule:
    """Validation rule definition."""
    name: str
    validator_func: callable
    error_message: str
    severity: ErrorSeverity = ErrorSeverity.MEDIUM


class InputValidator:
    """Comprehensive input validation system."""
    
    def __init__(self, validation_level: ValidationType = ValidationType.STRICT):
        self.validation_level = validation_level
        self.validation_rules: Dict[str, List[ValidationRule]] = {}
        self.sanitization_rules: Dict[str, List[callable]] = {}
        
        # Register default validation rules
        self._register_default_rules()
        
    def _register_default_rules(self):
        """Register default validation rules."""
        # Numeric validations
        self.add_validation_rule(
            "positive_number",
            ValidationRule(
                name="positive_number",
                validator_func=lambda x: isinstance(x, (int, float)) and x > 0,
                error_message="Value must be a positive number"
            )
        )
        
        self.add_validation_rule(
            "non_negative_number", 
            ValidationRule(
                name="non_negative_number",
                validator_func=lambda x: isinstance(x, (int, float)) and x >= 0,
                error_message="Value must be non-negative"
            )
        )
        
        self.add_validation_rule(
            "probability",
            ValidationRule(
                name="probability",
                validator_func=lambda x: isinstance(x, (int, float)) and 0 <= x <= 1,
                error_message="Value must be between 0 and 1"
            )
        )
        
        # String validations
        self.add_validation_rule(
            "safe_string",
            ValidationRule(
                name="safe_string",
                validator_func=self._validate_safe_string,
                error_message="String contains potentially unsafe characters",
                severity=ErrorSeverity.HIGH
            )
        )
        
        self.add_validation_rule(
            "filename_safe",
            ValidationRule(
                name="filename_safe",
                validator_func=self._validate_filename,
                error_message="Invalid filename format",
                severity=ErrorSeverity.MEDIUM
            )
        )
        
        # Array validations
        self.add_validation_rule(
            "array_shape",
            ValidationRule(
                name="array_shape",
                validator_func=self._validate_array_shape,
                error_message="Invalid array shape"
            )
        )
        
        # Register sanitization rules
        self.add_sanitization_rule("string", self._sanitize_string)
        self.add_sanitization_rule("filename", self._sanitize_filename)
        self.add_sanitization_rule("numeric", self._sanitize_numeric)
        
    def add_validation_rule(self, rule_type: str, rule: ValidationRule):
        """Add a validation rule."""
        if rule_type not in self.validation_rules:
            self.validation_rules[rule_type] = []
        self.validation_rules[rule_type].append(rule)
        
    def add_sanitization_rule(self, data_type: str, sanitizer_func: callable):
        """Add a sanitization rule."""
        if data_type not in self.sanitization_rules:
            self.sanitization_rules[data_type] = []
        self.sanitization_rules[data_type].append(sanitizer_func)
        
    def validate(
        self, 
        value: Any, 
        rule_types: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Validate value against specified rule types."""
        context = context or {}
        
        for rule_type in rule_types:
            if rule_type in self.validation_rules:
                for rule in self.validation_rules[rule_type]:
                    try:
                        is_valid = rule.validator_func(value)
                        if not is_valid:
                            raise ValidationError(
                                f"Validation failed for {rule_type}: {rule.error_message}",
                                error_code=f"INVALID_{rule_type.upper()}",
                                severity=rule.severity,
                                context={**context, "rule_type": rule_type, "value": str(value)[:100]}
                            )
                    except Exception as e:
                        if isinstance(e, ValidationError):
                            raise
                        else:
                            raise ValidationError(
                                f"Validation error in {rule_type}: {str(e)}",
                                error_code="VALIDATION_ERROR",
                                severity=ErrorSeverity.HIGH,
                                context={**context, "rule_type": rule_type}
                            )
        
        return value
        
    def sanitize(self, value: Any, data_type: str) -> Any:
        """Sanitize value using registered sanitization rules."""
        if data_type in self.sanitization_rules:
            for sanitizer in self.sanitization_rules[data_type]:
                try:
                    value = sanitizer(value)
                except Exception as e:
                    raise SecurityError(
                        f"Sanitization failed for {data_type}: {str(e)}",
                        error_code="SANITIZATION_ERROR",
                        severity=ErrorSeverity.HIGH,
                        context={"data_type": data_type, "original_value": str(value)[:100]}
                    )
        
        return value
        
    def validate_and_sanitize(
        self, 
        value: Any,
        data_type: str,
        rule_types: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Validate and sanitize value in one operation."""
        # Sanitize first
        sanitized_value = self.sanitize(value, data_type)
        
        # Then validate
        validated_value = self.validate(sanitized_value, rule_types, context)
        
        return validated_value
        
    # Default validation implementations
    def _validate_safe_string(self, value: str) -> bool:
        """Validate string for potentially unsafe content."""
        if not isinstance(value, str):
            return False
            
        # Check for common injection patterns
        dangerous_patterns = [
            r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>',  # Script tags
            r'javascript:',  # JavaScript protocol
            r'vbscript:',   # VBScript protocol
            r'on\w+\s*=',   # Event handlers
            r'eval\s*\(',   # Eval calls
            r'exec\s*\(',   # Exec calls
            r'\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER)\b',  # SQL keywords
        ]
        
        if self.validation_level == ValidationType.PARANOID:
            # In paranoid mode, also check for suspicious characters
            dangerous_patterns.extend([
                r'[<>"\'\\\x00-\x1f]',  # Control characters and special chars
                r'%[0-9a-fA-F]{2}',     # URL encoding
                r'\\[nrtvfb\\"]',       # Escape sequences
            ])
            
        for pattern in dangerous_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return False
                
        return True
        
    def _validate_filename(self, value: str) -> bool:
        """Validate filename for safety."""
        if not isinstance(value, str):
            return False
            
        # Check for dangerous filename patterns
        dangerous_patterns = [
            r'\.\./',        # Directory traversal
            r'~',            # Home directory
            r'^/',           # Absolute path
            r'[<>:"|?*]',    # Invalid filename characters
            r'\x00',         # Null byte
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, value):
                return False
                
        # Check for reserved names (Windows)
        reserved_names = ['CON', 'PRN', 'AUX', 'NUL'] + \
                        [f'COM{i}' for i in range(1, 10)] + \
                        [f'LPT{i}' for i in range(1, 10)]
                        
        if value.upper().split('.')[0] in reserved_names:
            return False
            
        return True
        
    def _validate_array_shape(self, value: Any) -> bool:
        """Basic array shape validation."""
        # This is a placeholder - specific shape validation would be done
        # with additional parameters in a real implementation
        try:
            if hasattr(value, 'shape'):
                return len(value.shape) > 0  # Has at least one dimension
            elif hasattr(value, '__len__'):
                return len(value) > 0  # Has at least one element
            else:
                return False
        except:
            return False
            
    # Default sanitization implementations
    def _sanitize_string(self, value: str) -> str:
        """Sanitize string input."""
        if not isinstance(value, str):
            return str(value)
            
        # Remove/escape dangerous characters based on validation level
        if self.validation_level == ValidationType.BASIC:
            # Just remove null bytes
            value = value.replace('\x00', '')
            
        elif self.validation_level == ValidationType.STRICT:
            # Remove control characters and normalize whitespace
            value = re.sub(r'[\x00-\x1f\x7f]', '', value)
            value = re.sub(r'\s+', ' ', value).strip()
            
        elif self.validation_level == ValidationType.PARANOID:
            # Aggressive sanitization
            value = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', value)  # Remove control chars
            value = re.sub(r'[<>"\'&]', '', value)               # Remove HTML chars
            value = re.sub(r'\\', '/', value)                    # Convert backslashes
            value = re.sub(r'\s+', ' ', value).strip()           # Normalize whitespace
            
        return value
        
    def _sanitize_filename(self, value: str) -> str:
        """Sanitize filename."""
        if not isinstance(value, str):
            value = str(value)
            
        # Remove/replace dangerous characters
        value = re.sub(r'[<>:"|?*\\]', '_', value)  # Replace invalid chars
        value = re.sub(r'\.\.', '_', value)         # Replace directory traversal
        value = value.replace('\x00', '')           # Remove null bytes
        value = value.strip('. ')                   # Remove leading/trailing dots and spaces
        
        # Ensure reasonable length
        if len(value) > 255:
            # Keep extension if present
            name, ext = value.rsplit('.', 1) if '.' in value else (value, '')
            max_name_len = 250 - len(ext)
            value = name[:max_name_len] + ('.' + ext if ext else '')
            
        return value or 'sanitized_file'  # Ensure non-empty
        
    def _sanitize_numeric(self, value: Any) -> Union[int, float]:
        """Sanitize numeric input."""
        if isinstance(value, (int, float)):
            # Check for special float values
            if isinstance(value, float):
                if value != value:  # NaN check
                    raise SecurityError(
                        "NaN values not allowed",
                        error_code="INVALID_NAN",
                        severity=ErrorSeverity.MEDIUM
                    )
                if abs(value) == float('inf'):
                    raise SecurityError(
                        "Infinite values not allowed",
                        error_code="INVALID_INFINITY",
                        severity=ErrorSeverity.MEDIUM
                    )
            return value
            
        # Try to convert string to number
        if isinstance(value, str):
            value = value.strip()
            try:
                if '.' in value:
                    return float(value)
                else:
                    return int(value)
            except ValueError:
                raise ValidationError(
                    f"Cannot convert '{value}' to number",
                    error_code="INVALID_NUMBER_FORMAT",
                    severity=ErrorSeverity.MEDIUM
                )
                
        raise ValidationError(
            f"Invalid numeric type: {type(value).__name__}",
            error_code="INVALID_NUMBER_TYPE",
            severity=ErrorSeverity.MEDIUM
        )


class SensorDataValidator(InputValidator):
    """Specialized validator for sensor data."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._register_sensor_rules()
        
    def _register_sensor_rules(self):
        """Register sensor-specific validation rules."""
        self.add_validation_rule(
            "sensor_reading",
            ValidationRule(
                name="sensor_reading",
                validator_func=self._validate_sensor_reading,
                error_message="Invalid sensor reading value"
            )
        )
        
        self.add_validation_rule(
            "concentration_ppm",
            ValidationRule(
                name="concentration_ppm",
                validator_func=lambda x: isinstance(x, (int, float)) and 0 <= x <= 50000,
                error_message="Concentration must be between 0 and 50,000 ppm"
            )
        )
        
        self.add_validation_rule(
            "spike_rate",
            ValidationRule(
                name="spike_rate",
                validator_func=lambda x: isinstance(x, (int, float)) and 0 <= x <= 1000,
                error_message="Spike rate must be between 0 and 1000 Hz"
            )
        )
        
    def _validate_sensor_reading(self, value: Union[int, float]) -> bool:
        """Validate sensor reading value."""
        if not isinstance(value, (int, float)):
            return False
            
        # Check for reasonable sensor reading range (0-5V typically)
        if not (0 <= value <= 5.0):
            return False
            
        # Check for suspicious patterns (all same values, etc.)
        # This would be more sophisticated in production
        return True
        
    def validate_sensor_array(
        self, 
        readings: List[float],
        sensor_count: int,
        reading_type: str = "voltage"
    ) -> List[float]:
        """Validate entire sensor array."""
        # Validate array structure
        if not isinstance(readings, list):
            raise ValidationError(
                "Sensor readings must be a list",
                error_code="INVALID_ARRAY_TYPE"
            )
            
        if len(readings) != sensor_count:
            raise ValidationError(
                f"Expected {sensor_count} sensor readings, got {len(readings)}",
                error_code="SENSOR_COUNT_MISMATCH"
            )
            
        # Validate each reading
        validated_readings = []
        for i, reading in enumerate(readings):
            try:
                sanitized = self.sanitize(reading, "numeric")
                validated = self.validate(sanitized, ["sensor_reading"], {"sensor_index": i})
                validated_readings.append(validated)
            except Exception as e:
                raise ValidationError(
                    f"Invalid reading from sensor {i}: {str(e)}",
                    error_code="SENSOR_READING_INVALID",
                    context={"sensor_index": i, "reading": reading}
                )
                
        return validated_readings


class NetworkInputValidator(InputValidator):
    """Specialized validator for neural network inputs."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._register_network_rules()
        
    def _register_network_rules(self):
        """Register network-specific validation rules."""
        self.add_validation_rule(
            "network_dimensions",
            ValidationRule(
                name="network_dimensions",
                validator_func=self._validate_network_dimensions,
                error_message="Invalid network dimensions"
            )
        )
        
        self.add_validation_rule(
            "learning_rate",
            ValidationRule(
                name="learning_rate", 
                validator_func=lambda x: isinstance(x, float) and 0 < x < 1,
                error_message="Learning rate must be between 0 and 1"
            )
        )
        
    def _validate_network_dimensions(self, value: int) -> bool:
        """Validate network dimension parameters."""
        if not isinstance(value, int):
            return False
            
        # Reasonable bounds for network sizes
        return 1 <= value <= 1000000
        
    def validate_network_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate complete network configuration."""
        validated_config = {}
        
        # Define expected parameters with their validation rules
        param_rules = {
            "num_sensors": ["positive_number", "network_dimensions"],
            "num_projection_neurons": ["positive_number", "network_dimensions"],
            "num_kenyon_cells": ["positive_number", "network_dimensions"],
            "learning_rate": ["positive_number", "learning_rate"],
            "sparsity_level": ["probability"],
            "tau_membrane": ["positive_number"],
        }
        
        for param_name, rules in param_rules.items():
            if param_name in config:
                try:
                    value = config[param_name]
                    sanitized = self.sanitize(value, "numeric")
                    validated = self.validate(sanitized, rules, {"parameter": param_name})
                    validated_config[param_name] = validated
                except Exception as e:
                    raise ValidationError(
                        f"Invalid network parameter {param_name}: {str(e)}",
                        error_code="INVALID_NETWORK_PARAM",
                        context={"parameter": param_name, "value": config[param_name]}
                    )
                    
        return validated_config


# Global validator instances
_input_validator = None
_sensor_validator = None
_network_validator = None


def get_input_validator() -> InputValidator:
    """Get global input validator."""
    global _input_validator
    if _input_validator is None:
        _input_validator = InputValidator()
    return _input_validator


def get_sensor_validator() -> SensorDataValidator:
    """Get global sensor data validator."""
    global _sensor_validator
    if _sensor_validator is None:
        _sensor_validator = SensorDataValidator()
    return _sensor_validator


def get_network_validator() -> NetworkInputValidator:
    """Get global network input validator."""
    global _network_validator
    if _network_validator is None:
        _network_validator = NetworkInputValidator()
    return _network_validator


# Validation decorators
def validate_inputs(**validation_specs):
    """Decorator for automatic input validation."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            validator = get_input_validator()
            
            # Validate positional arguments
            if 'args' in validation_specs:
                for i, (arg, rule_types) in enumerate(zip(args, validation_specs['args'])):
                    args = list(args)
                    args[i] = validator.validate(arg, rule_types, {"argument_index": i})
                    args = tuple(args)
                    
            # Validate keyword arguments
            if 'kwargs' in validation_specs:
                for param_name, rule_types in validation_specs['kwargs'].items():
                    if param_name in kwargs:
                        kwargs[param_name] = validator.validate(
                            kwargs[param_name], 
                            rule_types,
                            {"parameter": param_name}
                        )
                        
            return func(*args, **kwargs)
        return wrapper
    return decorator