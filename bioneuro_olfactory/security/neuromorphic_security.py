"""Advanced security measures for neuromorphic systems.

This module provides comprehensive security measures specifically designed
for neuromorphic spiking neural networks, including adversarial input detection,
model integrity verification, and secure processing pipelines.
"""

import hashlib
import hmac
import time
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import json
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

from ..core.error_handling_enhanced import SecurityError, ValidationError, ErrorSeverity
from ..core.logging_enhanced import security, error, audit
from .input_validation import get_input_validator
from .security_manager import SecurityManager


class ThreatLevel(Enum):
    """Security threat levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityPolicy:
    """Security policy configuration."""
    enable_adversarial_detection: bool = True
    enable_model_integrity_checks: bool = True
    enable_input_sanitization: bool = True
    enable_rate_limiting: bool = True
    enable_audit_logging: bool = True
    max_input_size_mb: float = 100.0
    max_processing_time_seconds: float = 60.0
    max_requests_per_minute: int = 1000
    adversarial_detection_threshold: float = 0.8
    model_checksum_verification: bool = True
    secure_random_seed: bool = True


class NeuromorphicSecurityManager:
    """Comprehensive security manager for neuromorphic systems."""
    
    def __init__(self, policy: Optional[SecurityPolicy] = None):
        self.policy = policy or SecurityPolicy()
        self.security_manager = SecurityManager()
        
        # Initialize security state
        self._model_checksums: Dict[str, str] = {}
        self._request_history: List[float] = []
        self._threat_cache: Dict[str, Tuple[ThreatLevel, float]] = {}
        self._security_violations: List[Dict[str, Any]] = []
        
        # Initialize adversarial detection
        if self.policy.enable_adversarial_detection:
            self._init_adversarial_detector()
    
    def _init_adversarial_detector(self):
        """Initialize adversarial input detection."""
        self.adversarial_patterns = {
            # Known adversarial patterns for neuromorphic systems
            'excessive_synchrony': {
                'threshold': 0.9,
                'description': 'Abnormally high spike synchronization'
            },
            'unnatural_periodicity': {
                'threshold': 0.8,
                'description': 'Artificial periodic spike patterns'
            },
            'impossible_spike_rates': {
                'min_rate': 0.0,
                'max_rate': 1000.0,
                'description': 'Physiologically impossible spike rates'
            },
            'gradient_attacks': {
                'threshold': 10.0,
                'description': 'Large input gradients indicating gradient-based attacks'
            }
        }
    
    def secure_process_input(
        self, 
        input_data: Union[torch.Tensor, np.ndarray],
        input_type: str = "sensor",
        source_id: Optional[str] = None
    ) -> Tuple[Union[torch.Tensor, np.ndarray], Dict[str, Any]]:
        """Securely process and validate input data.
        
        Args:
            input_data: Input data to process
            input_type: Type of input ("sensor", "spike_train", "network_config")
            source_id: Optional identifier for input source
            
        Returns:
            Tuple of (sanitized_input, security_report)
        """
        
        security_report = {
            "timestamp": time.time(),
            "input_type": input_type,
            "source_id": source_id,
            "threat_level": ThreatLevel.LOW,
            "security_checks": [],
            "violations": [],
            "sanitization_applied": False
        }
        
        try:
            # Rate limiting check
            if self.policy.enable_rate_limiting:
                self._check_rate_limits(security_report)
            
            # Input size validation
            self._validate_input_size(input_data, security_report)
            
            # Adversarial detection
            if self.policy.enable_adversarial_detection:
                self._detect_adversarial_input(input_data, input_type, security_report)
            
            # Input sanitization
            sanitized_input = input_data
            if self.policy.enable_input_sanitization:
                sanitized_input = self._sanitize_input(input_data, input_type, security_report)
            
            # Audit logging
            if self.policy.enable_audit_logging:
                self._log_security_audit(security_report)
            
            return sanitized_input, security_report
            
        except SecurityError as e:
            security_report["violations"].append({
                "type": "security_error",
                "message": str(e),
                "severity": e.severity.value if hasattr(e, 'severity') else "high"
            })
            security_report["threat_level"] = ThreatLevel.HIGH
            
            if self.policy.enable_audit_logging:
                security(f"Security error in input processing: {str(e)}")
            
            raise
        
        except Exception as e:
            security_report["violations"].append({
                "type": "processing_error",
                "message": str(e),
                "severity": "medium"
            })
            
            if self.policy.enable_audit_logging:
                error(f"Error in secure input processing: {str(e)}")
            
            raise SecurityError(
                f"Secure input processing failed: {str(e)}",
                error_code="SECURE_PROCESSING_FAILED",
                severity=ErrorSeverity.HIGH
            )
    
    def verify_model_integrity(
        self, 
        model: nn.Module,
        model_name: str,
        expected_checksum: Optional[str] = None
    ) -> Dict[str, Any]:
        """Verify model integrity and detect tampering.
        
        Args:
            model: Neural network model to verify
            model_name: Name/identifier for the model
            expected_checksum: Expected model checksum (if available)
            
        Returns:
            Integrity verification report
        """
        
        integrity_report = {
            "model_name": model_name,
            "timestamp": time.time(),
            "integrity_verified": False,
            "current_checksum": None,
            "expected_checksum": expected_checksum,
            "anomalies_detected": [],
            "threat_level": ThreatLevel.LOW
        }
        
        try:
            # Calculate current model checksum
            current_checksum = self._calculate_model_checksum(model)
            integrity_report["current_checksum"] = current_checksum
            
            # Compare with expected checksum
            if expected_checksum:
                if current_checksum == expected_checksum:
                    integrity_report["integrity_verified"] = True
                else:
                    integrity_report["anomalies_detected"].append({
                        "type": "checksum_mismatch",
                        "description": "Model checksum does not match expected value",
                        "severity": "high"
                    })
                    integrity_report["threat_level"] = ThreatLevel.HIGH
            else:
                # Store checksum for future verification
                self._model_checksums[model_name] = current_checksum
                integrity_report["integrity_verified"] = True
            
            # Check for suspicious model properties
            self._analyze_model_anomalies(model, integrity_report)
            
            # Log integrity check
            if self.policy.enable_audit_logging:
                audit(f"Model integrity check for {model_name}", structured_data={
                    "type": "model_integrity_check",
                    "model_name": model_name,
                    "integrity_verified": integrity_report["integrity_verified"],
                    "threat_level": integrity_report["threat_level"].value
                })
            
            return integrity_report
            
        except Exception as e:
            integrity_report["anomalies_detected"].append({
                "type": "verification_error",
                "description": f"Integrity verification failed: {str(e)}",
                "severity": "high"
            })
            integrity_report["threat_level"] = ThreatLevel.HIGH
            
            error(f"Model integrity verification failed: {str(e)}")
            return integrity_report
    
    def secure_model_execution(
        self,
        model: nn.Module,
        input_data: Union[torch.Tensor, Dict],
        model_name: str,
        execution_timeout: Optional[float] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Execute model in secure environment with monitoring.
        
        Args:
            model: Neural network model
            input_data: Model input data
            model_name: Model identifier
            execution_timeout: Maximum execution time in seconds
            
        Returns:
            Tuple of (model_output, security_report)
        """
        
        if execution_timeout is None:
            execution_timeout = self.policy.max_processing_time_seconds
        
        security_report = {
            "model_name": model_name,
            "execution_start": time.time(),
            "execution_end": None,
            "execution_time": None,
            "memory_usage": None,
            "security_violations": [],
            "execution_successful": False,
            "threat_level": ThreatLevel.LOW
        }
        
        try:
            # Pre-execution security checks
            if self.policy.enable_model_integrity_checks:
                integrity_report = self.verify_model_integrity(model, model_name)
                if not integrity_report["integrity_verified"]:
                    raise SecurityError(
                        "Model integrity verification failed",
                        error_code="MODEL_INTEGRITY_FAILED",
                        severity=ErrorSeverity.CRITICAL
                    )
            
            # Monitor resource usage during execution
            start_memory = self._get_memory_usage()
            start_time = time.time()
            
            # Execute with timeout protection
            try:
                with self._execution_timeout(execution_timeout):
                    model.eval()  # Ensure evaluation mode
                    with torch.no_grad():
                        if isinstance(input_data, dict):
                            output = model(**input_data)
                        else:
                            output = model(input_data)
            
            except TimeoutError:
                raise SecurityError(
                    f"Model execution timeout after {execution_timeout}s",
                    error_code="EXECUTION_TIMEOUT",
                    severity=ErrorSeverity.HIGH
                )
            
            # Post-execution monitoring
            end_time = time.time()
            end_memory = self._get_memory_usage()
            execution_time = end_time - start_time
            
            security_report.update({
                "execution_end": end_time,
                "execution_time": execution_time,
                "memory_usage": end_memory - start_memory,
                "execution_successful": True
            })
            
            # Check for suspicious execution patterns
            if execution_time > execution_timeout * 0.8:  # >80% of timeout
                security_report["security_violations"].append({
                    "type": "long_execution_time",
                    "description": f"Execution time {execution_time:.2f}s is suspiciously long",
                    "severity": "medium"
                })
                security_report["threat_level"] = ThreatLevel.MEDIUM
            
            # Validate output
            self._validate_model_output(output, security_report)
            
            # Log secure execution
            if self.policy.enable_audit_logging:
                audit(f"Secure model execution for {model_name}", structured_data={
                    "type": "secure_model_execution",
                    "model_name": model_name,
                    "execution_time": execution_time,
                    "memory_usage": security_report["memory_usage"],
                    "threat_level": security_report["threat_level"].value
                })
            
            return output, security_report
            
        except Exception as e:
            security_report["security_violations"].append({
                "type": "execution_error",
                "description": str(e),
                "severity": "high"
            })
            security_report["threat_level"] = ThreatLevel.HIGH
            security_report["execution_end"] = time.time()
            
            security(f"Secure model execution failed: {str(e)}")
            raise
    
    def detect_adversarial_patterns(
        self, 
        spike_data: Union[torch.Tensor, np.ndarray]
    ) -> Dict[str, Any]:
        """Detect adversarial patterns in spike train data.
        
        Args:
            spike_data: Spike train data to analyze
            
        Returns:
            Adversarial detection report
        """
        
        detection_report = {
            "timestamp": time.time(),
            "adversarial_detected": False,
            "threat_level": ThreatLevel.LOW,
            "detected_patterns": [],
            "confidence_score": 0.0,
            "analysis_details": {}
        }
        
        try:
            # Convert to numpy for analysis
            if TORCH_AVAILABLE and isinstance(spike_data, torch.Tensor):
                data_np = spike_data.detach().cpu().numpy()
            else:
                data_np = np.array(spike_data)
            
            # Ensure proper dimensions
            if data_np.ndim < 2:
                data_np = data_np.reshape(1, -1)
            elif data_np.ndim > 3:
                raise ValueError(f"Invalid spike data dimensions: {data_np.ndim}")
            
            # Handle batch dimension
            if data_np.ndim == 3:
                batch_size, num_neurons, time_steps = data_np.shape
                # Analyze each sample in batch
                for batch_idx in range(batch_size):
                    batch_report = self._analyze_adversarial_patterns(data_np[batch_idx])
                    if batch_report["adversarial_detected"]:
                        detection_report["adversarial_detected"] = True
                        detection_report["detected_patterns"].extend(batch_report["detected_patterns"])
                        detection_report["confidence_score"] = max(\n                            detection_report["confidence_score"],\n                            batch_report["confidence_score"]\n                        )
            else:
                pattern_report = self._analyze_adversarial_patterns(data_np)
                detection_report.update(pattern_report)
            
            # Set threat level based on detection
            if detection_report["adversarial_detected"]:
                if detection_report["confidence_score"] > 0.8:
                    detection_report["threat_level"] = ThreatLevel.CRITICAL
                elif detection_report["confidence_score"] > 0.6:
                    detection_report["threat_level"] = ThreatLevel.HIGH
                else:
                    detection_report["threat_level"] = ThreatLevel.MEDIUM
            
            # Log adversarial detection
            if detection_report["adversarial_detected"] and self.policy.enable_audit_logging:
                security(
                    f"Adversarial patterns detected: {detection_report['detected_patterns']}"
                )
            
            return detection_report
            
        except Exception as e:
            detection_report["analysis_details"]["error"] = str(e)
            error(f"Adversarial pattern detection failed: {str(e)}")
            return detection_report
    
    # Private security methods
    
    def _check_rate_limits(self, security_report: Dict[str, Any]):
        """Check rate limiting."""
        current_time = time.time()
        
        # Clean old requests (older than 1 minute)
        self._request_history = [t for t in self._request_history if current_time - t < 60]
        
        # Check rate limit
        if len(self._request_history) >= self.policy.max_requests_per_minute:
            security_report["violations"].append({
                "type": "rate_limit_exceeded",
                "description": f"Too many requests: {len(self._request_history)} in last minute",
                "severity": "high"
            })
            security_report["threat_level"] = ThreatLevel.HIGH
            
            raise SecurityError(
                "Rate limit exceeded",
                error_code="RATE_LIMIT_EXCEEDED",
                severity=ErrorSeverity.HIGH
            )
        
        # Add current request
        self._request_history.append(current_time)
        security_report["security_checks"].append("rate_limit_check")
    
    def _validate_input_size(self, input_data: Union[torch.Tensor, np.ndarray], 
                           security_report: Dict[str, Any]):
        """Validate input size limits."""
        
        if TORCH_AVAILABLE and isinstance(input_data, torch.Tensor):
            size_mb = input_data.numel() * input_data.element_size() / 1024 / 1024
        elif isinstance(input_data, np.ndarray):
            size_mb = input_data.nbytes / 1024 / 1024
        else:
            size_mb = 0
        
        if size_mb > self.policy.max_input_size_mb:
            security_report["violations"].append({
                "type": "input_size_exceeded",
                "description": f"Input size {size_mb:.1f}MB exceeds limit {self.policy.max_input_size_mb}MB",
                "severity": "high"
            })
            security_report["threat_level"] = ThreatLevel.HIGH
            
            raise SecurityError(
                f"Input size {size_mb:.1f}MB exceeds security limit",
                error_code="INPUT_SIZE_EXCEEDED",
                severity=ErrorSeverity.HIGH
            )
        
        security_report["security_checks"].append("input_size_validation")
    
    def _detect_adversarial_input(self, input_data: Union[torch.Tensor, np.ndarray], 
                                input_type: str, security_report: Dict[str, Any]):
        """Detect adversarial inputs."""
        
        if input_type == "spike_train":
            detection_report = self.detect_adversarial_patterns(input_data)
            
            if detection_report["adversarial_detected"]:
                security_report["violations"].append({
                    "type": "adversarial_input_detected",
                    "description": f"Adversarial patterns: {detection_report['detected_patterns']}",
                    "confidence": detection_report["confidence_score"],
                    "severity": "critical"
                })
                security_report["threat_level"] = ThreatLevel.CRITICAL
                
                raise SecurityError(
                    "Adversarial input patterns detected",
                    error_code="ADVERSARIAL_INPUT_DETECTED",
                    severity=ErrorSeverity.CRITICAL
                )
        
        security_report["security_checks"].append("adversarial_detection")
    
    def _sanitize_input(self, input_data: Union[torch.Tensor, np.ndarray], 
                       input_type: str, security_report: Dict[str, Any]) -> Union[torch.Tensor, np.ndarray]:
        """Sanitize input data."""
        
        sanitized = input_data
        sanitization_applied = False
        
        if TORCH_AVAILABLE and isinstance(input_data, torch.Tensor):
            # Remove NaN and infinite values
            if torch.isnan(input_data).any() or torch.isinf(input_data).any():
                sanitized = torch.nan_to_num(input_data, nan=0.0, posinf=1.0, neginf=0.0)
                sanitization_applied = True
            
            # Clamp extreme values
            if input_type == "spike_train":
                sanitized = torch.clamp(sanitized, 0.0, 1.0)
                sanitization_applied = True
            elif input_type == "sensor":
                sanitized = torch.clamp(sanitized, -10.0, 10.0)  # Reasonable sensor range
                sanitization_applied = True
        
        elif isinstance(input_data, np.ndarray):
            # Similar sanitization for numpy arrays
            if np.any(np.isnan(input_data)) or np.any(np.isinf(input_data)):
                sanitized = np.nan_to_num(input_data, nan=0.0, posinf=1.0, neginf=0.0)
                sanitization_applied = True
            
            if input_type == "spike_train":
                sanitized = np.clip(sanitized, 0.0, 1.0)
                sanitization_applied = True
            elif input_type == "sensor":
                sanitized = np.clip(sanitized, -10.0, 10.0)
                sanitization_applied = True
        
        if sanitization_applied:
            security_report["sanitization_applied"] = True
            security_report["security_checks"].append("input_sanitization")
        
        return sanitized
    
    def _analyze_adversarial_patterns(self, spike_data: np.ndarray) -> Dict[str, Any]:
        """Analyze spike data for adversarial patterns."""
        
        pattern_report = {
            "adversarial_detected": False,
            "detected_patterns": [],
            "confidence_score": 0.0,
            "pattern_details": {}
        }
        
        num_neurons, time_steps = spike_data.shape
        
        # Check for excessive synchrony
        synchrony_score = self._calculate_spike_synchrony(spike_data)
        if synchrony_score > self.adversarial_patterns['excessive_synchrony']['threshold']:
            pattern_report["detected_patterns"].append("excessive_synchrony")
            pattern_report["confidence_score"] = max(pattern_report["confidence_score"], synchrony_score)
            pattern_report["pattern_details"]["synchrony_score"] = synchrony_score
        
        # Check for unnatural periodicity
        periodicity_score = self._calculate_periodicity(spike_data)
        if periodicity_score > self.adversarial_patterns['unnatural_periodicity']['threshold']:
            pattern_report["detected_patterns"].append("unnatural_periodicity")
            pattern_report["confidence_score"] = max(pattern_report["confidence_score"], periodicity_score)
            pattern_report["pattern_details"]["periodicity_score"] = periodicity_score
        
        # Check spike rates
        spike_rates = np.sum(spike_data, axis=1) / (time_steps * 0.001)  # Assuming 1ms timesteps
        min_rate = self.adversarial_patterns['impossible_spike_rates']['min_rate']
        max_rate = self.adversarial_patterns['impossible_spike_rates']['max_rate']
        
        if np.any(spike_rates < min_rate) or np.any(spike_rates > max_rate):
            pattern_report["detected_patterns"].append("impossible_spike_rates")
            pattern_report["confidence_score"] = max(pattern_report["confidence_score"], 0.9)
            pattern_report["pattern_details"]["spike_rate_violations"] = {
                "min_rate": float(np.min(spike_rates)),
                "max_rate": float(np.max(spike_rates)),
                "violating_neurons": np.sum((spike_rates < min_rate) | (spike_rates > max_rate))
            }
        
        pattern_report["adversarial_detected"] = len(pattern_report["detected_patterns"]) > 0
        
        return pattern_report
    
    def _calculate_spike_synchrony(self, spike_data: np.ndarray) -> float:
        """Calculate spike synchrony measure."""
        # Simple synchrony measure: correlation between neurons
        if spike_data.shape[0] < 2:
            return 0.0
        
        correlations = []
        for i in range(spike_data.shape[0]):
            for j in range(i + 1, spike_data.shape[0]):
                corr = np.corrcoef(spike_data[i], spike_data[j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        
        return np.mean(correlations) if correlations else 0.0
    
    def _calculate_periodicity(self, spike_data: np.ndarray) -> float:
        """Calculate periodicity measure using FFT."""
        # Analyze frequency domain for artificial periodicities
        avg_spike_train = np.mean(spike_data, axis=0)
        
        if np.sum(avg_spike_train) == 0:
            return 0.0
        
        # Apply FFT
        fft_result = np.fft.fft(avg_spike_train)
        power_spectrum = np.abs(fft_result)
        
        # Look for dominant frequencies (indicating periodicity)
        max_power = np.max(power_spectrum[1:])  # Exclude DC component
        total_power = np.sum(power_spectrum[1:])
        
        if total_power == 0:
            return 0.0
        
        periodicity_score = max_power / total_power
        return min(periodicity_score, 1.0)
    
    def _calculate_model_checksum(self, model: nn.Module) -> str:
        """Calculate model checksum for integrity verification."""
        model_data = []
        
        # Collect all parameter data
        for param in model.parameters():
            model_data.append(param.data.detach().cpu().numpy().tobytes())
        
        # Create checksum
        combined_data = b''.join(model_data)
        checksum = hashlib.sha256(combined_data).hexdigest()
        
        return checksum
    
    def _analyze_model_anomalies(self, model: nn.Module, integrity_report: Dict[str, Any]):
        """Analyze model for suspicious properties."""
        
        # Check for extremely large parameters
        large_params = 0
        total_params = 0
        
        for param in model.parameters():
            param_data = param.data.detach().cpu().numpy()
            large_params += np.sum(np.abs(param_data) > 100)
            total_params += param_data.size
        
        if large_params > total_params * 0.01:  # >1% of parameters are very large
            integrity_report["anomalies_detected"].append({
                "type": "large_parameters",
                "description": f"{large_params} parameters with abs value > 100",
                "severity": "medium"
            })
        
        # Check for NaN or infinite parameters
        nan_inf_params = 0
        for param in model.parameters():
            param_data = param.data.detach().cpu()
            nan_inf_params += torch.isnan(param_data).sum().item()
            nan_inf_params += torch.isinf(param_data).sum().item()
        
        if nan_inf_params > 0:
            integrity_report["anomalies_detected"].append({
                "type": "invalid_parameters",
                "description": f"{nan_inf_params} NaN or infinite parameters",
                "severity": "high"
            })
            integrity_report["threat_level"] = ThreatLevel.HIGH
    
    def _validate_model_output(self, output: Any, security_report: Dict[str, Any]):
        """Validate model output for security issues."""
        
        if TORCH_AVAILABLE and isinstance(output, torch.Tensor):
            # Check for invalid values in output
            if torch.isnan(output).any() or torch.isinf(output).any():
                security_report["security_violations"].append({
                    "type": "invalid_output_values",
                    "description": "Model output contains NaN or infinite values",
                    "severity": "high"
                })
                security_report["threat_level"] = ThreatLevel.HIGH
        
        elif isinstance(output, dict):
            # Check dictionary outputs
            for key, value in output.items():
                if TORCH_AVAILABLE and isinstance(value, torch.Tensor):
                    if torch.isnan(value).any() or torch.isinf(value).any():
                        security_report["security_violations"].append({
                            "type": "invalid_output_values",
                            "description": f"Output key '{key}' contains invalid values",
                            "severity": "high"
                        })
                        security_report["threat_level"] = ThreatLevel.HIGH
    
    def _execution_timeout(self, timeout_seconds: float):
        """Context manager for execution timeout."""
        # Simple timeout implementation - in production, use more sophisticated approaches
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Execution timed out")
        
        # Set the signal handler and alarm
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout_seconds))
        
        try:
            yield
        finally:
            # Reset the alarm and handler
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def _log_security_audit(self, security_report: Dict[str, Any]):
        """Log security audit information."""
        audit_data = {
            "type": "security_audit",
            "timestamp": security_report["timestamp"],
            "input_type": security_report["input_type"],
            "threat_level": security_report["threat_level"].value,
            "checks_performed": security_report["security_checks"],
            "violations_count": len(security_report["violations"]),
            "sanitization_applied": security_report["sanitization_applied"]
        }
        
        audit("Security audit completed", structured_data=audit_data)


# Global security manager instance
_neuromorphic_security_manager = None


def get_neuromorphic_security_manager(policy: Optional[SecurityPolicy] = None) -> NeuromorphicSecurityManager:
    """Get global neuromorphic security manager."""
    global _neuromorphic_security_manager
    if _neuromorphic_security_manager is None:
        _neuromorphic_security_manager = NeuromorphicSecurityManager(policy)
    return _neuromorphic_security_manager


# Convenience functions
def secure_process_input(input_data, input_type="sensor", source_id=None):
    """Securely process input using global security manager."""
    return get_neuromorphic_security_manager().secure_process_input(input_data, input_type, source_id)


def verify_model_integrity(model, model_name, expected_checksum=None):
    """Verify model integrity using global security manager."""
    return get_neuromorphic_security_manager().verify_model_integrity(model, model_name, expected_checksum)


def secure_model_execution(model, input_data, model_name, execution_timeout=None):
    """Execute model securely using global security manager."""
    return get_neuromorphic_security_manager().secure_model_execution(model, input_data, model_name, execution_timeout)


def detect_adversarial_patterns(spike_data):
    """Detect adversarial patterns using global security manager."""
    return get_neuromorphic_security_manager().detect_adversarial_patterns(spike_data)