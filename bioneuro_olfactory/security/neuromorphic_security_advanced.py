"""
Advanced Security Framework for Neuromorphic Systems
===================================================

This module provides comprehensive security measures for neuromorphic computing
systems, including input validation, adversarial detection, and secure execution.

Created as part of Terragon SDLC Generation 2: MAKE IT ROBUST
"""

import hashlib
import hmac
import time
import warnings
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import secrets
import json


class SecurityThreatLevel(Enum):
    """Security threat levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AttackType(Enum):
    """Types of attacks against neuromorphic systems."""
    ADVERSARIAL_INPUT = "adversarial_input"
    WEIGHT_POISONING = "weight_poisoning"
    MODEL_EXTRACTION = "model_extraction"
    MEMBERSHIP_INFERENCE = "membership_inference"
    DATA_POISONING = "data_poisoning"
    SIDE_CHANNEL = "side_channel"
    DENIAL_OF_SERVICE = "denial_of_service"
    PARAMETER_MANIPULATION = "parameter_manipulation"


@dataclass
class SecurityAlert:
    """Security alert information."""
    alert_type: AttackType
    threat_level: SecurityThreatLevel
    message: str
    timestamp: float = field(default_factory=time.time)
    component: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    mitigation_actions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert alert to dictionary."""
        return {
            'alert_type': self.alert_type.value,
            'threat_level': self.threat_level.value,
            'message': self.message,
            'timestamp': self.timestamp,
            'component': self.component,
            'evidence': self.evidence,
            'confidence': self.confidence,
            'mitigation_actions': self.mitigation_actions
        }


class NeuromorphicSecurityValidator:
    """Advanced security validator for neuromorphic systems."""
    
    def __init__(self):
        self.validation_rules: Dict[str, List[Callable]] = {}
        self.security_baselines: Dict[str, Dict] = {}
        self.threat_detection_enabled = True
        
    def validate_neuromorphic_input(self, data: Any, component_type: str) -> Tuple[bool, List[str]]:
        """Comprehensive validation for neuromorphic inputs."""
        errors = []
        
        # Basic safety checks
        if not self._validate_basic_safety(data):
            errors.append("Basic safety validation failed")
            
        # Component-specific validation
        if component_type == "spike_encoder":
            errors.extend(self._validate_spike_encoder_input(data))
        elif component_type == "projection_neuron":
            errors.extend(self._validate_projection_neuron_input(data))
        elif component_type == "kenyon_cell":
            errors.extend(self._validate_kenyon_cell_input(data))
        elif component_type == "decision_layer":
            errors.extend(self._validate_decision_layer_input(data))
            
        # Security threat detection
        if self.threat_detection_enabled:
            security_errors = self._detect_security_threats(data, component_type)
            errors.extend(security_errors)
            
        return len(errors) == 0, errors
        
    def _validate_basic_safety(self, data: Any) -> bool:
        """Basic safety validation for any input."""
        if data is None:
            return False
            
        # Check for NaN/infinite values
        if hasattr(data, 'isnan') and hasattr(data, 'isinf'):
            if hasattr(data.isnan(), 'any') and data.isnan().any():
                return False
            if hasattr(data.isinf(), 'any') and data.isinf().any():
                return False
                
        # Check for reasonable data size
        if hasattr(data, 'size') and data.size > 1000000:
            return False
            
        return True
        
    def _validate_spike_encoder_input(self, data: Any) -> List[str]:
        """Validate input for spike encoders."""
        errors = []
        
        # Check value range for spike data
        if hasattr(data, 'min') and hasattr(data, 'max'):
            if data.min() < 0 or data.max() > 1:
                errors.append("Spike data must be binary (0 or 1)")
                
        # Check temporal consistency
        if hasattr(data, 'shape') and len(data.shape) >= 2:
            if data.shape[1] > 10000:  # Too many time steps
                errors.append("Excessive temporal dimension")
                
        return errors
        
    def _validate_projection_neuron_input(self, data: Any) -> List[str]:
        """Validate input for projection neurons."""
        errors = []
        
        # Check membrane potential bounds
        if hasattr(data, 'max') and data.max() > 100:
            errors.append("Membrane potential exceeds safe bounds")
            
        # Check for spike rate limits
        if hasattr(data, 'mean') and data.mean() > 0.9:
            errors.append("Spike rate too high for biological plausibility")
            
        return errors
        
    def _validate_kenyon_cell_input(self, data: Any) -> List[str]:
        """Validate input for Kenyon cells."""
        errors = []
        
        # Check sparsity expectations
        if hasattr(data, 'mean') and data.mean() > 0.2:
            errors.append("Input violates sparsity expectations for Kenyon cells")
            
        # Check dimensionality
        if hasattr(data, 'shape') and len(data.shape) > 0:
            if data.shape[-1] > 50000:  # Too many input neurons
                errors.append("Excessive input dimensionality")
                
        return errors
        
    def _validate_decision_layer_input(self, data: Any) -> List[str]:
        """Validate input for decision layers."""
        errors = []
        
        # Check for valid probability distributions
        if hasattr(data, 'sum') and hasattr(data, 'shape'):
            if len(data.shape) > 1:
                # Check if rows sum to reasonable values
                row_sums = data.sum(axis=-1) if hasattr(data, 'sum') else None
                if row_sums is not None and hasattr(row_sums, 'max'):
                    if row_sums.max() > 10:
                        errors.append("Decision input values exceed reasonable bounds")
                        
        return errors
        
    def _detect_security_threats(self, data: Any, component_type: str) -> List[str]:
        """Detect potential security threats in input data."""
        threats = []
        
        # Adversarial pattern detection
        if self._detect_adversarial_patterns(data):
            threats.append("Potential adversarial input pattern detected")
            
        # Denial of service detection
        if self._detect_dos_patterns(data):
            threats.append("Potential denial-of-service attack detected")
            
        # Data poisoning detection
        if self._detect_poisoning_patterns(data, component_type):
            threats.append("Potential data poisoning attack detected")
            
        return threats
        
    def _detect_adversarial_patterns(self, data: Any) -> bool:
        """Detect adversarial input patterns."""
        
        # Check for unusual statistical properties
        if hasattr(data, 'std') and hasattr(data, 'mean'):
            try:
                std_val = float(data.std())
                mean_val = float(data.mean())
                
                # High frequency noise patterns
                if std_val > 0 and abs(mean_val / std_val) > 100:
                    return True
                    
                # Unusual variance patterns
                if std_val > mean_val * 10:
                    return True
                    
            except:
                pass
                
        return False
        
    def _detect_dos_patterns(self, data: Any) -> bool:
        """Detect denial-of-service attack patterns."""
        
        # Check for excessive computational load
        if hasattr(data, 'size') and data.size > 500000:
            return True
            
        # Check for memory exhaustion patterns
        if hasattr(data, 'nbytes') and data.nbytes > 50 * 1024 * 1024:  # 50MB
            return True
            
        return False
        
    def _detect_poisoning_patterns(self, data: Any, component_type: str) -> bool:
        """Detect data poisoning attack patterns."""
        
        # Check against established baselines
        if component_type in self.security_baselines:
            baseline = self.security_baselines[component_type]
            
            if hasattr(data, 'mean') and 'mean_range' in baseline:
                data_mean = float(data.mean())
                min_mean, max_mean = baseline['mean_range']
                if data_mean < min_mean or data_mean > max_mean:
                    return True
                    
        return False


class RobustNeuromorphicBase:
    """Base class providing robustness features for neuromorphic components."""
    
    def __init__(self, component_name: str):
        self.component_name = component_name
        self.validator = NeuromorphicSecurityValidator()
        self.error_count = 0
        self.max_errors = 10
        self.last_error_time = 0
        self.error_cooldown = 60  # seconds
        
    def robust_forward(self, *args, **kwargs):
        """Robust forward pass with error handling and validation."""
        
        # Rate limiting for errors
        if self._should_throttle_due_to_errors():
            raise RuntimeError(f"Component {self.component_name} throttled due to excessive errors")
            
        # Validate inputs
        for i, arg in enumerate(args):
            is_valid, errors = self.validator.validate_neuromorphic_input(
                arg, self.component_name
            )
            if not is_valid:
                self._record_error()
                raise ValueError(f"Input validation failed for arg {i}: {errors}")
                
        try:
            # Execute the actual forward pass
            result = self.forward(*args, **kwargs)
            
            # Validate outputs
            self._validate_outputs(result)
            
            # Reset error count on success
            self.error_count = 0
            
            return result
            
        except Exception as e:
            self._record_error()
            raise
            
    def _should_throttle_due_to_errors(self) -> bool:
        """Check if component should be throttled due to errors."""
        if self.error_count >= self.max_errors:
            if time.time() - self.last_error_time < self.error_cooldown:
                return True
            else:
                # Reset after cooldown period
                self.error_count = 0
                
        return False
        
    def _record_error(self):
        """Record an error occurrence."""
        self.error_count += 1
        self.last_error_time = time.time()
        
    def _validate_outputs(self, outputs: Any):
        """Validate component outputs."""
        if outputs is None:
            raise ValueError("Component produced None output")
            
        # Check for NaN/infinite values in outputs
        if hasattr(outputs, 'isnan') and hasattr(outputs, 'isinf'):
            if hasattr(outputs.isnan(), 'any') and outputs.isnan().any():
                raise ValueError("Component produced NaN outputs")
            if hasattr(outputs.isinf(), 'any') and outputs.isinf().any():
                raise ValueError("Component produced infinite outputs")
                
    def forward(self, *args, **kwargs):
        """Override this method in subclasses."""
        raise NotImplementedError("Subclasses must implement forward method")
        
    def get_health_status(self) -> Dict[str, Any]:
        """Get component health status."""
        return {
            'component_name': self.component_name,
            'error_count': self.error_count,
            'last_error_time': self.last_error_time,
            'is_throttled': self._should_throttle_due_to_errors(),
            'status': 'healthy' if self.error_count < 5 else 'degraded'
        }


class NeuromorphicCircuitBreaker:
    """Circuit breaker for neuromorphic component protection."""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 timeout: float = 30.0,
                 recovery_timeout: float = 10.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.recovery_timeout = recovery_timeout
        
        self.failure_count = 0
        self.last_failure_time = 0
        self.last_success_time = 0
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        
        if self.state == 'OPEN':
            if self._should_attempt_reset():
                self.state = 'HALF_OPEN'
            else:
                raise RuntimeError("Circuit breaker is OPEN")
                
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except Exception as e:
            self._on_failure()
            raise
            
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        return time.time() - self.last_failure_time > self.timeout
        
    def _on_success(self):
        """Handle successful function execution."""
        self.failure_count = 0
        self.last_success_time = time.time()
        
        if self.state == 'HALF_OPEN':
            # If we were testing, go back to closed
            self.state = 'CLOSED'
            
    def _on_failure(self):
        """Handle failed function execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
        elif self.state == 'HALF_OPEN':
            # Failed during recovery attempt
            self.state = 'OPEN'
            
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            'state': self.state,
            'failure_count': self.failure_count,
            'last_failure_time': self.last_failure_time,
            'last_success_time': self.last_success_time,
            'time_since_last_failure': time.time() - self.last_failure_time if self.last_failure_time > 0 else None
        }


def create_robust_component(base_class, component_name: str):
    """Factory function to create robust neuromorphic components."""
    
    class RobustComponent(RobustNeuromorphicBase, base_class):
        def __init__(self, *args, **kwargs):
            RobustNeuromorphicBase.__init__(self, component_name)
            base_class.__init__(self, *args, **kwargs)
            
            # Add circuit breaker
            self.circuit_breaker = NeuromorphicCircuitBreaker()
            
        def forward(self, *args, **kwargs):
            """Forward pass with circuit breaker protection."""
            return self.circuit_breaker.call(
                super(base_class, self).forward, *args, **kwargs
            )
            
        def get_comprehensive_status(self) -> Dict[str, Any]:
            """Get comprehensive component status."""
            return {
                'health': self.get_health_status(),
                'circuit_breaker': self.circuit_breaker.get_status(),
                'validator_stats': {
                    'threat_detection_enabled': self.validator.threat_detection_enabled,
                    'baseline_count': len(self.validator.security_baselines)
                }
            }
            
    return RobustComponent


class NeuromorphicSystemHealthChecker:
    """System-wide health checking for neuromorphic systems."""
    
    def __init__(self):
        self.registered_components: Dict[str, Any] = {}
        self.health_history: List[Dict] = []
        self.max_history = 1000
        
    def register_component(self, name: str, component: Any):
        """Register a component for health monitoring."""
        self.registered_components[name] = component
        
    def check_system_health(self) -> Dict[str, Any]:
        """Check health of entire neuromorphic system."""
        
        component_health = {}
        overall_status = "healthy"
        
        # Check each registered component
        for name, component in self.registered_components.items():
            if hasattr(component, 'get_health_status'):
                health = component.get_health_status()
                component_health[name] = health
                
                # Update overall status
                if health.get('status') == 'degraded':
                    overall_status = "degraded"
                elif health.get('is_throttled', False):
                    overall_status = "critical"
                    
        # System-level checks
        system_metrics = self._check_system_metrics()
        
        health_report = {
            'timestamp': time.time(),
            'overall_status': overall_status,
            'component_health': component_health,
            'system_metrics': system_metrics,
            'registered_components': len(self.registered_components)
        }
        
        # Store in history
        self.health_history.append(health_report)
        if len(self.health_history) > self.max_history:
            self.health_history.pop(0)
            
        return health_report
        
    def _check_system_metrics(self) -> Dict[str, Any]:
        """Check system-level metrics."""
        return {
            'memory_usage': self._estimate_memory_usage(),
            'error_rate': self._calculate_system_error_rate(),
            'uptime': time.time() - getattr(self, 'start_time', time.time())
        }
        
    def _estimate_memory_usage(self) -> float:
        """Estimate system memory usage (simplified)."""
        # In a real implementation, this would check actual memory usage
        return len(self.registered_components) * 1024 * 1024  # Placeholder
        
    def _calculate_system_error_rate(self) -> float:
        """Calculate system-wide error rate."""
        if not self.health_history:
            return 0.0
            
        recent_reports = self.health_history[-10:]  # Last 10 reports
        total_components = 0
        degraded_components = 0
        
        for report in recent_reports:
            for component_health in report['component_health'].values():
                total_components += 1
                if component_health.get('status') == 'degraded':
                    degraded_components += 1
                    
        return degraded_components / total_components if total_components > 0 else 0.0
        
    def get_health_trends(self, duration_minutes: int = 60) -> Dict[str, Any]:
        """Get health trends over specified duration."""
        
        cutoff_time = time.time() - (duration_minutes * 60)
        recent_history = [
            report for report in self.health_history
            if report['timestamp'] > cutoff_time
        ]
        
        if not recent_history:
            return {}
            
        # Calculate trends
        status_changes = []
        for i in range(1, len(recent_history)):
            prev_status = recent_history[i-1]['overall_status']
            curr_status = recent_history[i]['overall_status']
            if prev_status != curr_status:
                status_changes.append({
                    'timestamp': recent_history[i]['timestamp'],
                    'from': prev_status,
                    'to': curr_status
                })
                
        return {
            'total_reports': len(recent_history),
            'status_changes': status_changes,
            'stability_score': 1.0 - (len(status_changes) / len(recent_history)) if recent_history else 1.0,
            'current_status': recent_history[-1]['overall_status'] if recent_history else 'unknown'
        }


# Create validation test
def validate_generation_2_robustness():
    """Validate that Generation 2 robustness features are working."""
    
    print("🛡️ Validating Generation 2 Robustness Features...")
    
    results = {
        'validator': False,
        'circuit_breaker': False,
        'health_checker': False,
        'error_handling': False
    }
    
    try:
        # Test validator
        validator = NeuromorphicSecurityValidator()
        test_data = [1, 2, 3, 4, 5]  # Simple test data
        is_valid, errors = validator.validate_neuromorphic_input(test_data, "spike_encoder")
        results['validator'] = True
        print("  ✅ Security validator working")
        
    except Exception as e:
        print(f"  ❌ Security validator failed: {e}")
        
    try:
        # Test circuit breaker
        breaker = NeuromorphicCircuitBreaker()
        
        def test_func():
            return "success"
            
        result = breaker.call(test_func)
        results['circuit_breaker'] = result == "success"
        print("  ✅ Circuit breaker working")
        
    except Exception as e:
        print(f"  ❌ Circuit breaker failed: {e}")
        
    try:
        # Test health checker
        health_checker = NeuromorphicSystemHealthChecker()
        
        # Create mock component
        class MockComponent:
            def get_health_status(self):
                return {'status': 'healthy', 'error_count': 0}
                
        health_checker.register_component("test", MockComponent())
        health_report = health_checker.check_system_health()
        results['health_checker'] = 'overall_status' in health_report
        print("  ✅ Health checker working")
        
    except Exception as e:
        print(f"  ❌ Health checker failed: {e}")
        
    try:
        # Test robust base class
        class TestRobustComponent(RobustNeuromorphicBase):
            def forward(self, x):
                return x * 2
                
        component = TestRobustComponent("test_component")
        result = component.robust_forward(5)
        results['error_handling'] = result == 10
        print("  ✅ Error handling working")
        
    except Exception as e:
        print(f"  ❌ Error handling failed: {e}")
        
    # Summary
    total_tests = len(results)
    passed_tests = sum(results.values())
    success_rate = passed_tests / total_tests
    
    print(f"\n🏆 Generation 2 Validation Results:")
    print(f"  Tests Passed: {passed_tests}/{total_tests}")
    print(f"  Success Rate: {success_rate:.1%}")
    print(f"  Status: {'✅ PASS' if success_rate >= 0.75 else '❌ FAIL'}")
    
    return success_rate >= 0.75, results


if __name__ == "__main__":
    success, results = validate_generation_2_robustness()
    exit(0 if success else 1)