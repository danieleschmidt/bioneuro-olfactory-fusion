"""
Enhanced Validation System for Generation 2: MAKE IT ROBUST
Comprehensive validation for all neuromorphic components with advanced error detection
"""

import re
import json
import time
import math
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import hashlib


class ValidationLevel(Enum):
    """Validation strictness levels."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    PARANOID = "paranoid"


@dataclass
class ValidationRule:
    """A validation rule with condition and error message."""
    name: str
    condition: callable
    error_message: str
    level: ValidationLevel
    fix_suggestion: Optional[str] = None


class NeuroMorphicValidator:
    """Enhanced validator for neuromorphic components."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.rules = self._initialize_rules()
        self.validation_cache = {}
        self.cache_ttl = 300  # 5 minutes
    
    def _initialize_rules(self) -> Dict[str, List[ValidationRule]]:
        """Initialize validation rules for different components."""
        rules = {
            'sensor_data': self._get_sensor_data_rules(),
            'projection_neurons': self._get_projection_neuron_rules(),
            'kenyon_cells': self._get_kenyon_cell_rules(),
            'decision_layer': self._get_decision_layer_rules(),
            'fusion_layer': self._get_fusion_layer_rules(),
            'temporal_dynamics': self._get_temporal_dynamics_rules(),
            'security': self._get_security_rules()
        }
        return rules
    
    def _get_sensor_data_rules(self) -> List[ValidationRule]:
        """Get validation rules for sensor data."""
        return [
            ValidationRule(
                name="data_not_none",
                condition=lambda data: data is not None,
                error_message="Sensor data cannot be None",
                level=ValidationLevel.BASIC,
                fix_suggestion="Provide valid sensor data array"
            ),
            ValidationRule(
                name="data_not_empty",
                condition=lambda data: hasattr(data, '__len__') and len(data) > 0,
                error_message="Sensor data cannot be empty",
                level=ValidationLevel.BASIC,
                fix_suggestion="Ensure sensor array contains at least one element"
            ),
            ValidationRule(
                name="numeric_values",
                condition=lambda data: all(isinstance(x, (int, float)) for x in (data if hasattr(data, '__iter__') else [data])),
                error_message="All sensor values must be numeric",
                level=ValidationLevel.BASIC,
                fix_suggestion="Convert all values to float or int"
            ),
            ValidationRule(
                name="finite_values",
                condition=lambda data: all(math.isfinite(x) for x in (data if hasattr(data, '__iter__') else [data]) if isinstance(x, (int, float))),
                error_message="Sensor data contains NaN or infinite values",
                level=ValidationLevel.STANDARD,
                fix_suggestion="Remove or interpolate NaN/infinite values"
            ),
            ValidationRule(
                name="reasonable_range",
                condition=lambda data: all(-10000 <= x <= 10000 for x in (data if hasattr(data, '__iter__') else [data]) if isinstance(x, (int, float))),
                error_message="Sensor values outside reasonable range [-10000, 10000]",
                level=ValidationLevel.STANDARD,
                fix_suggestion="Check sensor calibration and scaling"
            ),
            ValidationRule(
                name="no_constant_values",
                condition=lambda data: len(set(data if hasattr(data, '__iter__') else [data])) > 1 if len(data if hasattr(data, '__iter__') else [data]) > 1 else True,
                error_message="All sensor values are identical (sensor may be stuck)",
                level=ValidationLevel.STRICT,
                fix_suggestion="Check sensor hardware and connections"
            ),
            ValidationRule(
                name="signal_to_noise_ratio",
                condition=lambda data: self._check_signal_noise_ratio(data),
                error_message="Poor signal-to-noise ratio detected",
                level=ValidationLevel.PARANOID,
                fix_suggestion="Improve sensor shielding or use filtering"
            )
        ]
    
    def _get_projection_neuron_rules(self) -> List[ValidationRule]:
        """Get validation rules for projection neurons."""
        return [
            ValidationRule(
                name="positive_neurons",
                condition=lambda config: config.get('num_projection_neurons', 0) > 0,
                error_message="Number of projection neurons must be positive",
                level=ValidationLevel.BASIC,
                fix_suggestion="Set num_projection_neurons to positive integer"
            ),
            ValidationRule(
                name="expansion_ratio",
                condition=lambda config: config.get('num_projection_neurons', 0) / max(config.get('num_receptors', 1), 1) >= 2,
                error_message="Insufficient expansion ratio in projection layer",
                level=ValidationLevel.STANDARD,
                fix_suggestion="Increase projection neurons to at least 2x input dimensions"
            ),
            ValidationRule(
                name="membrane_time_constant",
                condition=lambda config: 1.0 <= config.get('tau_membrane', 20.0) <= 200.0,
                error_message="Membrane time constant outside biological range [1.0, 200.0] ms",
                level=ValidationLevel.STANDARD,
                fix_suggestion="Set tau_membrane between 10-50 ms for typical neurons"
            ),
            ValidationRule(
                name="threshold_reasonable",
                condition=lambda config: 0.1 <= config.get('threshold', 1.0) <= 10.0,
                error_message="Neuron threshold outside reasonable range [0.1, 10.0]",
                level=ValidationLevel.STANDARD,
                fix_suggestion="Set threshold between 0.5-2.0 for typical LIF neurons"
            ),
            ValidationRule(
                name="adaptation_slower_than_membrane",
                condition=lambda config: config.get('tau_adaptation', 100.0) >= config.get('tau_membrane', 20.0),
                error_message="Adaptation time constant should be >= membrane time constant",
                level=ValidationLevel.STRICT,
                fix_suggestion="Set tau_adaptation to 2-10x tau_membrane"
            ),
            ValidationRule(
                name="computational_feasibility",
                condition=lambda config: config.get('num_projection_neurons', 0) * config.get('num_receptors', 0) < 1e8,
                error_message="Network size may be computationally infeasible",
                level=ValidationLevel.PARANOID,
                fix_suggestion="Reduce network size or use sparse connectivity"
            )
        ]
    
    def _get_kenyon_cell_rules(self) -> List[ValidationRule]:
        """Get validation rules for Kenyon cells."""
        return [
            ValidationRule(
                name="sparse_expansion",
                condition=lambda config: config.get('num_kenyon_cells', 0) >= config.get('num_projection_inputs', 0),
                error_message="Kenyon cells should exceed or equal projection inputs for sparse coding",
                level=ValidationLevel.BASIC,
                fix_suggestion="Increase num_kenyon_cells to at least match inputs"
            ),
            ValidationRule(
                name="sparsity_range",
                condition=lambda config: 0.001 <= config.get('sparsity_target', 0.05) <= 0.5,
                error_message="Sparsity target outside reasonable range [0.001, 0.5]",
                level=ValidationLevel.STANDARD,
                fix_suggestion="Set sparsity between 0.01-0.1 for biological realism"
            ),
            ValidationRule(
                name="sufficient_active_cells",
                condition=lambda config: config.get('num_kenyon_cells', 0) * config.get('sparsity_target', 0.05) >= 10,
                error_message="Too few active Kenyon cells for reliable coding",
                level=ValidationLevel.STANDARD,
                fix_suggestion="Increase num_kenyon_cells or sparsity_target"
            ),
            ValidationRule(
                name="connection_probability_range",
                condition=lambda config: 0.01 <= config.get('connection_probability', 0.1) <= 0.5,
                error_message="Connection probability outside reasonable range [0.01, 0.5]",
                level=ValidationLevel.STANDARD,
                fix_suggestion="Set connection_probability between 0.05-0.2"
            ),
            ValidationRule(
                name="inhibition_strength",
                condition=lambda config: config.get('inhibition_strength', 1.0) >= 0.1,
                error_message="Insufficient global inhibition for sparsity control",
                level=ValidationLevel.STRICT,
                fix_suggestion="Set inhibition_strength >= 0.5 for effective sparsity"
            ),
            ValidationRule(
                name="biological_scale",
                condition=lambda config: config.get('num_kenyon_cells', 0) <= 1e6,
                error_message="Kenyon cell count exceeds biological plausibility",
                level=ValidationLevel.PARANOID,
                fix_suggestion="Consider if such large networks are necessary"
            )
        ]
    
    def _get_decision_layer_rules(self) -> List[ValidationRule]:
        """Get validation rules for decision layer."""
        return [
            ValidationRule(
                name="positive_classes",
                condition=lambda config: config.get('num_gas_classes', 0) >= 2,
                error_message="Must have at least 2 gas classes for classification",
                level=ValidationLevel.BASIC,
                fix_suggestion="Define at least 2 gas types to detect"
            ),
            ValidationRule(
                name="reasonable_class_count",
                condition=lambda config: config.get('num_gas_classes', 0) <= 50,
                error_message="Too many gas classes may lead to confusion",
                level=ValidationLevel.STANDARD,
                fix_suggestion="Consider grouping similar gases or hierarchical classification"
            ),
            ValidationRule(
                name="integration_window",
                condition=lambda config: 10 <= config.get('integration_window', 100) <= 1000,
                error_message="Integration window outside reasonable range [10, 1000] ms",
                level=ValidationLevel.STANDARD,
                fix_suggestion="Set integration window to 50-200 ms for good temporal integration"
            ),
            ValidationRule(
                name="sufficient_inputs",
                condition=lambda config: config.get('num_kenyon_inputs', 0) >= config.get('num_gas_classes', 0) * 10,
                error_message="Insufficient Kenyon inputs for reliable classification",
                level=ValidationLevel.STRICT,
                fix_suggestion="Ensure at least 10x kenyon inputs vs gas classes"
            ),
            ValidationRule(
                name="decision_threshold",
                condition=lambda config: 0.1 <= config.get('confidence_threshold', 0.8) <= 0.99,
                error_message="Confidence threshold outside reasonable range [0.1, 0.99]",
                level=ValidationLevel.STRICT,
                fix_suggestion="Set confidence threshold between 0.7-0.9"
            )
        ]
    
    def _get_fusion_layer_rules(self) -> List[ValidationRule]:
        """Get validation rules for fusion layer."""
        return [
            ValidationRule(
                name="positive_dimensions",
                condition=lambda config: config.get('chemical_dim', 0) > 0 and config.get('audio_dim', 0) > 0,
                error_message="Both chemical and audio dimensions must be positive",
                level=ValidationLevel.BASIC,
                fix_suggestion="Set positive values for both input modalities"
            ),
            ValidationRule(
                name="dimensional_balance",
                condition=lambda config: 0.1 <= (config.get('audio_dim', 128) / max(config.get('chemical_dim', 6), 1)) <= 100,
                error_message="Severe dimensional imbalance between modalities",
                level=ValidationLevel.STANDARD,
                fix_suggestion="Balance dimensions or use feature selection/projection"
            ),
            ValidationRule(
                name="valid_fusion_strategy",
                condition=lambda config: config.get('fusion_strategy', 'hierarchical') in ['early', 'attention', 'hierarchical', 'spiking'],
                error_message="Unknown fusion strategy",
                level=ValidationLevel.BASIC,
                fix_suggestion="Use one of: early, attention, hierarchical, spiking"
            ),
            ValidationRule(
                name="hidden_dim_reasonable",
                condition=lambda config: 8 <= config.get('hidden_dim', 64) <= 1024,
                error_message="Hidden dimension outside reasonable range [8, 1024]",
                level=ValidationLevel.STANDARD,
                fix_suggestion="Set hidden_dim between 32-256 for most applications"
            ),
            ValidationRule(
                name="dropout_rate",
                condition=lambda config: 0.0 <= config.get('dropout_rate', 0.1) <= 0.8,
                error_message="Dropout rate outside valid range [0.0, 0.8]",
                level=ValidationLevel.STRICT,
                fix_suggestion="Set dropout between 0.1-0.5 for regularization"
            )
        ]
    
    def _get_temporal_dynamics_rules(self) -> List[ValidationRule]:
        """Get validation rules for temporal dynamics."""
        return [
            ValidationRule(
                name="positive_timestep",
                condition=lambda config: config.get('dt', 1.0) > 0,
                error_message="Time step must be positive",
                level=ValidationLevel.BASIC,
                fix_suggestion="Set dt to positive value (typically 0.1-1.0 ms)"
            ),
            ValidationRule(
                name="temporal_resolution",
                condition=lambda config: config.get('dt', 1.0) <= config.get('tau_membrane', 20.0) / 5,
                error_message="Time step too large for accurate temporal dynamics",
                level=ValidationLevel.STANDARD,
                fix_suggestion="Set dt <= tau_membrane/5 for numerical stability"
            ),
            ValidationRule(
                name="simulation_duration",
                condition=lambda config: config.get('simulation_duration', 100) >= config.get('tau_membrane', 20.0) * 3,
                error_message="Simulation too short for neural dynamics to develop",
                level=ValidationLevel.STANDARD,
                fix_suggestion="Set simulation duration to at least 3x tau_membrane"
            ),
            ValidationRule(
                name="reasonable_timesteps",
                condition=lambda config: 10 <= (config.get('simulation_duration', 100) / config.get('dt', 1.0)) <= 10000,
                error_message="Number of timesteps outside reasonable range [10, 10000]",
                level=ValidationLevel.STRICT,
                fix_suggestion="Adjust dt and simulation_duration for 50-1000 timesteps"
            ),
            ValidationRule(
                name="refractory_period",
                condition=lambda config: config.get('refractory_period', 2.0) >= config.get('dt', 1.0),
                error_message="Refractory period should be >= time step",
                level=ValidationLevel.STRICT,
                fix_suggestion="Set refractory period to 1-5 ms"
            )
        ]
    
    def _get_security_rules(self) -> List[ValidationRule]:
        """Get validation rules for security."""
        return [
            ValidationRule(
                name="safe_file_path",
                condition=lambda path: '../' not in str(path) and not str(path).startswith('/'),
                error_message="Unsafe file path detected (directory traversal attempt)",
                level=ValidationLevel.BASIC,
                fix_suggestion="Use relative paths without ../ patterns"
            ),
            ValidationRule(
                name="input_length_limit",
                condition=lambda data: len(str(data)) <= 10000,
                error_message="Input data exceeds maximum length limit",
                level=ValidationLevel.STANDARD,
                fix_suggestion="Limit input strings to reasonable lengths"
            ),
            ValidationRule(
                name="no_code_injection",
                condition=lambda data: not self._contains_code_patterns(str(data)),
                error_message="Potential code injection patterns detected",
                level=ValidationLevel.STRICT,
                fix_suggestion="Sanitize input to remove code-like patterns"
            ),
            ValidationRule(
                name="rate_limit_check",
                condition=lambda context: self._check_rate_limit(context),
                error_message="Rate limit exceeded",
                level=ValidationLevel.STANDARD,
                fix_suggestion="Reduce request frequency"
            )
        ]
    
    def _check_signal_noise_ratio(self, data) -> bool:
        """Check if signal has reasonable signal-to-noise ratio."""
        try:
            if not hasattr(data, '__iter__') or len(data) < 10:
                return True  # Skip check for small datasets
            
            values = [x for x in data if isinstance(x, (int, float))]
            if len(values) < 10:
                return True
            
            # Calculate simple SNR estimate
            mean_val = sum(values) / len(values)
            variance = sum((x - mean_val) ** 2 for x in values) / len(values)
            
            if variance == 0:
                return False  # All values identical
            
            snr = abs(mean_val) / math.sqrt(variance) if variance > 0 else float('inf')
            return snr > 0.1  # Minimum SNR threshold
            
        except Exception:
            return True  # Skip check if calculation fails
    
    def _contains_code_patterns(self, text: str) -> bool:
        """Check for potential code injection patterns."""
        dangerous_patterns = [
            r'<script',
            r'javascript:',
            r'eval\s*\(',
            r'exec\s*\(',
            r'import\s+os',
            r'__.*__',
            r'subprocess',
            r'system\s*\(',
            r'\.\./',
            r'[;&|`]'
        ]
        
        text_lower = text.lower()
        for pattern in dangerous_patterns:
            if re.search(pattern, text_lower):
                return True
        return False
    
    def _check_rate_limit(self, context: Dict) -> bool:
        """Check rate limiting (simplified implementation)."""
        # This would connect to actual rate limiting in production
        return True
    
    def validate_component(self, component_type: str, data: Any) -> Tuple[bool, List[str], List[str]]:
        """Validate a component with comprehensive rules."""
        if component_type not in self.rules:
            return False, [f"Unknown component type: {component_type}"], []
        
        # Check cache first
        cache_key = f"{component_type}_{hash(str(data))}"
        current_time = time.time()
        
        if cache_key in self.validation_cache:
            cached_result, timestamp = self.validation_cache[cache_key]
            if current_time - timestamp < self.cache_ttl:
                return cached_result
        
        errors = []
        warnings = []
        suggestions = []
        
        rules = self.rules[component_type]
        
        for rule in rules:
            # Skip rules above current validation level
            if rule.level.value > self.validation_level.value:
                continue
            
            try:
                if not rule.condition(data):
                    if rule.level in [ValidationLevel.BASIC, ValidationLevel.STANDARD]:
                        errors.append(f"{rule.name}: {rule.error_message}")
                    else:
                        warnings.append(f"{rule.name}: {rule.error_message}")
                    
                    if rule.fix_suggestion:
                        suggestions.append(f"{rule.name}: {rule.fix_suggestion}")
                        
            except Exception as e:
                errors.append(f"{rule.name}: Validation rule failed - {str(e)}")
        
        is_valid = len(errors) == 0
        result = (is_valid, errors, warnings, suggestions)
        
        # Cache result
        self.validation_cache[cache_key] = (result, current_time)
        
        return result
    
    def validate_complete_system(self, system_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the complete neuromorphic system."""
        results = {}
        overall_valid = True
        all_errors = []
        all_warnings = []
        all_suggestions = []
        
        # Component configurations to validate
        components = {
            'sensor_data': system_config.get('sensor_data'),
            'projection_neurons': system_config.get('projection_config'),
            'kenyon_cells': system_config.get('kenyon_config'),
            'decision_layer': system_config.get('decision_config'),
            'fusion_layer': system_config.get('fusion_config'),
            'temporal_dynamics': system_config.get('temporal_config')
        }
        
        for component_name, component_data in components.items():
            if component_data is not None:
                is_valid, errors, warnings, suggestions = self.validate_component(component_name, component_data)
                
                results[component_name] = {
                    'valid': is_valid,
                    'errors': errors,
                    'warnings': warnings,
                    'suggestions': suggestions
                }
                
                if not is_valid:
                    overall_valid = False
                    all_errors.extend(errors)
                
                all_warnings.extend(warnings)
                all_suggestions.extend(suggestions)
        
        # Cross-component validation
        cross_validation_errors = self._cross_validate_components(system_config)
        if cross_validation_errors:
            overall_valid = False
            all_errors.extend(cross_validation_errors)
        
        return {
            'overall_valid': overall_valid,
            'component_results': results,
            'summary': {
                'total_errors': len(all_errors),
                'total_warnings': len(all_warnings),
                'total_suggestions': len(all_suggestions),
                'validation_level': self.validation_level.value
            },
            'all_errors': all_errors,
            'all_warnings': all_warnings,
            'all_suggestions': all_suggestions
        }
    
    def _cross_validate_components(self, system_config: Dict[str, Any]) -> List[str]:
        """Validate interactions between components."""
        errors = []
        
        try:
            # Check dimension consistency
            projection_config = system_config.get('projection_config', {})
            kenyon_config = system_config.get('kenyon_config', {})
            
            pn_outputs = projection_config.get('num_projection_neurons', 0)
            kc_inputs = kenyon_config.get('num_projection_inputs', 0)
            
            if pn_outputs > 0 and kc_inputs > 0 and pn_outputs != kc_inputs:
                errors.append(f"Dimension mismatch: projection outputs ({pn_outputs}) != kenyon inputs ({kc_inputs})")
            
            # Check temporal consistency
            temporal_config = system_config.get('temporal_config', {})
            
            for component in ['projection_config', 'kenyon_config']:
                comp_config = system_config.get(component, {})
                comp_tau = comp_config.get('tau_membrane', 20.0)
                global_dt = temporal_config.get('dt', 1.0)
                
                if comp_tau / global_dt < 5:
                    errors.append(f"Temporal resolution warning: {component} tau_membrane/dt ratio too small")
            
            # Check computational feasibility
            total_neurons = (
                projection_config.get('num_projection_neurons', 0) +
                kenyon_config.get('num_kenyon_cells', 0) +
                system_config.get('decision_config', {}).get('num_gas_classes', 0)
            )
            
            if total_neurons > 1e6:
                errors.append(f"Total neuron count ({total_neurons}) may be computationally challenging")
                
        except Exception as e:
            errors.append(f"Cross-validation error: {str(e)}")
        
        return errors
    
    def get_validation_report(self, system_config: Dict[str, Any]) -> str:
        """Generate a human-readable validation report."""
        results = self.validate_complete_system(system_config)
        
        report = []
        report.append("=== NEUROMORPHIC SYSTEM VALIDATION REPORT ===")
        report.append(f"Validation Level: {self.validation_level.value.upper()}")
        report.append(f"Overall Status: {'PASSED' if results['overall_valid'] else 'FAILED'}")
        report.append("")
        
        # Summary
        summary = results['summary']
        report.append("SUMMARY:")
        report.append(f"  Errors: {summary['total_errors']}")
        report.append(f"  Warnings: {summary['total_warnings']}")
        report.append(f"  Suggestions: {summary['total_suggestions']}")
        report.append("")
        
        # Component details
        for component, result in results['component_results'].items():
            status = "✓" if result['valid'] else "✗"
            report.append(f"{status} {component.upper()}")
            
            if result['errors']:
                report.append("  ERRORS:")
                for error in result['errors']:
                    report.append(f"    - {error}")
            
            if result['warnings']:
                report.append("  WARNINGS:")
                for warning in result['warnings']:
                    report.append(f"    - {warning}")
            
            if result['suggestions']:
                report.append("  SUGGESTIONS:")
                for suggestion in result['suggestions']:
                    report.append(f"    - {suggestion}")
            
            report.append("")
        
        return "\n".join(report)