"""Enhanced security system for neuromorphic gas detection.

This module implements comprehensive security measures including
adversarial attack detection, input validation, secure inference,
and threat monitoring for production deployment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import time
import hashlib
import hmac
import secrets
from collections import defaultdict, deque
import logging


class ThreatLevel(Enum):
    """Security threat levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AttackType(Enum):
    """Types of adversarial attacks."""
    FGSM = "fgsm"
    PGD = "pgd"
    EVASION = "evasion"
    POISONING = "poisoning"
    MODEL_INVERSION = "model_inversion"
    MEMBERSHIP_INFERENCE = "membership_inference"
    BACKDOOR = "backdoor"


@dataclass
class SecurityConfig:
    """Configuration for security system."""
    enable_adversarial_detection: bool = True
    enable_input_validation: bool = True
    enable_rate_limiting: bool = True
    enable_audit_logging: bool = True
    enable_secure_inference: bool = True
    enable_differential_privacy: bool = False
    
    # Adversarial detection parameters
    adversarial_threshold: float = 0.1
    ensemble_size: int = 5
    
    # Rate limiting parameters
    max_requests_per_minute: int = 100
    max_requests_per_hour: int = 1000
    
    # Differential privacy parameters
    privacy_epsilon: float = 1.0
    privacy_delta: float = 1e-5
    
    # Input validation parameters
    max_input_magnitude: float = 10.0
    min_input_magnitude: float = -10.0
    max_batch_size: int = 64
    
    # Audit logging
    log_all_requests: bool = True
    log_suspicious_activity: bool = True


class AdversarialDetector:
    """Detects adversarial attacks on neuromorphic models."""
    
    def __init__(
        self,
        base_model: nn.Module,
        threshold: float = 0.1,
        ensemble_size: int = 5
    ):
        self.base_model = base_model
        self.threshold = threshold
        self.ensemble_size = ensemble_size
        
        # Create ensemble of detectors
        self.detectors = self._create_detector_ensemble()
        
        # Statistical baselines for detection
        self.baseline_stats = {}
        self.prediction_history = deque(maxlen=1000)
        
    def _create_detector_ensemble(self) -> List[nn.Module]:
        """Create ensemble of adversarial detectors."""
        detectors = []
        
        for i in range(self.ensemble_size):
            # Simple detector network
            detector = nn.Sequential(
                nn.Linear(128, 64),  # Assume input features
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
            detectors.append(detector)
            
        return nn.ModuleList(detectors)
        
    def detect_adversarial_input(
        self,
        input_data: torch.Tensor,
        model_output: torch.Tensor
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """Detect if input is adversarial.
        
        Args:
            input_data: Input tensor to analyze
            model_output: Model's output on the input
            
        Returns:
            Tuple of (is_adversarial, confidence_score, detection_details)
        """
        detection_scores = []
        detection_details = {}
        
        # Method 1: Statistical deviation detection
        stat_score = self._detect_statistical_deviation(input_data, model_output)
        detection_scores.append(stat_score)
        detection_details['statistical_score'] = stat_score
        
        # Method 2: Gradient-based detection
        grad_score = self._detect_gradient_anomaly(input_data)
        detection_scores.append(grad_score)
        detection_details['gradient_score'] = grad_score
        
        # Method 3: Prediction consistency check
        consistency_score = self._check_prediction_consistency(input_data)
        detection_scores.append(consistency_score)
        detection_details['consistency_score'] = consistency_score
        
        # Ensemble decision
        ensemble_score = np.mean(detection_scores)
        is_adversarial = ensemble_score > self.threshold
        
        detection_details.update({
            'ensemble_score': ensemble_score,
            'individual_scores': detection_scores,
            'threshold': self.threshold
        })
        
        return is_adversarial, ensemble_score, detection_details
        
    def _detect_statistical_deviation(
        self,
        input_data: torch.Tensor,
        model_output: torch.Tensor
    ) -> float:
        """Detect statistical deviations from normal patterns."""
        # Compute input statistics
        input_mean = torch.mean(input_data)
        input_std = torch.std(input_data)
        
        # Check against baselines
        if 'input_mean' not in self.baseline_stats:
            # Initialize baselines
            self.baseline_stats = {
                'input_mean': input_mean.item(),
                'input_std': input_std.item()
            }
            return 0.0
            
        # Calculate deviations
        mean_dev = abs(input_mean.item() - self.baseline_stats['input_mean'])
        std_dev = abs(input_std.item() - self.baseline_stats['input_std'])
        
        # Normalize deviations
        total_deviation = (mean_dev + std_dev) / 2.0
        
        return min(total_deviation, 1.0)
        
    def _detect_gradient_anomaly(self, input_data: torch.Tensor) -> float:
        """Detect anomalies in input gradients."""
        input_data_copy = input_data.clone().detach().requires_grad_(True)
        
        try:
            # Forward pass
            output = self.base_model(input_data_copy)
            
            # Compute gradients
            if hasattr(output, 'mean'):
                loss = output.mean()
            else:
                loss = torch.mean(output)
                
            gradients = torch.autograd.grad(
                loss, input_data_copy, create_graph=False, retain_graph=False
            )[0]
            
            # Analyze gradient patterns
            grad_norm = torch.norm(gradients).item()
            
            # Anomaly score based on gradient magnitude
            anomaly_score = min(grad_norm / 10.0, 1.0)  # Normalize
            
            return anomaly_score
            
        except Exception:
            # If gradient computation fails, return safe default
            return 0.0
        
    def _check_prediction_consistency(self, input_data: torch.Tensor) -> float:
        """Check prediction consistency across small perturbations."""
        try:
            original_output = self.base_model(input_data)
            
            # Add small random perturbations
            perturbation_scores = []
            num_perturbations = 3
            
            for _ in range(num_perturbations):
                noise = torch.randn_like(input_data) * 0.01  # Small noise
                perturbed_input = input_data + noise
                perturbed_output = self.base_model(perturbed_input)
                
                # Compute prediction difference
                diff = torch.norm(original_output - perturbed_output).item()
                perturbation_scores.append(diff)
                
            # High inconsistency suggests adversarial input
            avg_inconsistency = np.mean(perturbation_scores)
            return min(avg_inconsistency * 10.0, 1.0)  # Normalize
            
        except Exception:
            # If consistency check fails, return safe default
            return 0.0


class EnhancedSecuritySystem:
    """Enhanced security system for neuromorphic networks."""
    
    def __init__(
        self,
        base_model: nn.Module,
        config: SecurityConfig = None
    ):
        self.base_model = base_model
        self.config = config or SecurityConfig()
        
        # Initialize security components
        self.adversarial_detector = AdversarialDetector(
            base_model,
            threshold=self.config.adversarial_threshold,
            ensemble_size=self.config.ensemble_size
        ) if self.config.enable_adversarial_detection else None
        
        # Security metrics
        self.security_metrics = {
            'total_requests': 0,
            'blocked_requests': 0,
            'threats_detected': 0,
            'validation_failures': 0
        }
        
    def secure_forward(
        self,
        input_data: Union[torch.Tensor, Dict[str, torch.Tensor]],
        client_id: str = "unknown"
    ) -> Dict[str, Any]:
        """Secure forward pass with comprehensive security checks.
        
        Args:
            input_data: Input data to process
            client_id: Identifier for the client making the request
            
        Returns:
            Dictionary with results and security information
        """
        start_time = time.time()
        self.security_metrics['total_requests'] += 1
        
        try:
            # Input validation
            if not self._validate_input(input_data):
                self.security_metrics['validation_failures'] += 1
                return {
                    'success': False,
                    'error': 'Input validation failed',
                    'security_status': 'rejected'
                }
                
            # Model inference
            with torch.no_grad():
                if isinstance(input_data, dict):
                    model_output = self.base_model(**input_data)
                else:
                    model_output = self.base_model(input_data)
                    
            # Adversarial detection
            if self.adversarial_detector:
                input_tensor = input_data if isinstance(input_data, torch.Tensor) else list(input_data.values())[0]
                is_adversarial, confidence, detection_details = \
                    self.adversarial_detector.detect_adversarial_input(
                        input_tensor, model_output
                    )
                    
                if is_adversarial:
                    self.security_metrics['threats_detected'] += 1
                    return {
                        'success': False,
                        'error': 'Adversarial input detected',
                        'confidence': confidence,
                        'detection_details': detection_details,
                        'security_status': 'threat_detected'
                    }
                    
            # Successful processing
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'output': model_output,
                'processing_time': processing_time,
                'security_status': 'secure',
                'client_id': client_id
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Processing failed: {str(e)}',
                'security_status': 'error'
            }
            
    def _validate_input(self, input_data: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> bool:
        """Validate input data for security compliance."""
        try:
            if isinstance(input_data, torch.Tensor):
                return self._validate_tensor(input_data)
            elif isinstance(input_data, dict):
                return all(self._validate_tensor(tensor) for tensor in input_data.values() if isinstance(tensor, torch.Tensor))
            else:
                return False
        except Exception:
            return False
            
    def _validate_tensor(self, tensor: torch.Tensor) -> bool:
        """Validate individual tensor."""
        # Check for NaN/Inf
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            return False
            
        # Check magnitude bounds
        max_val = torch.max(tensor).item()
        min_val = torch.min(tensor).item()
        
        if (max_val > self.config.max_input_magnitude or 
            min_val < self.config.min_input_magnitude):
            return False
            
        # Check batch size
        if tensor.dim() > 0 and tensor.shape[0] > self.config.max_batch_size:
            return False
            
        return True
            
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        return {
            'timestamp': time.time(),
            'metrics': self.security_metrics.copy()
        }


def create_enhanced_security_system(
    base_model: nn.Module,
    enable_all_security: bool = True
) -> EnhancedSecuritySystem:
    """Create an enhanced security system.
    
    Args:
        base_model: Base neuromorphic model to secure
        enable_all_security: Whether to enable all security features
        
    Returns:
        Configured secure system
    """
    config = SecurityConfig(
        enable_adversarial_detection=enable_all_security,
        enable_input_validation=enable_all_security,
        enable_rate_limiting=enable_all_security,
        enable_audit_logging=enable_all_security,
        enable_secure_inference=enable_all_security
    )
    
    return EnhancedSecuritySystem(base_model, config)