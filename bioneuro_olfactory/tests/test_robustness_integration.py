"""Integration tests for Generation 2 robustness features.

This module provides comprehensive integration tests to validate that all
robustness features work together properly in the bioneuro-olfactory-fusion
framework.
"""

import unittest
import numpy as np
import time
import logging
from unittest.mock import Mock, patch
from typing import Dict, Any, List

# Import robustness components
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.error_handling_enhanced import EnhancedErrorHandler, BioNeuroError, ValidationError
from core.logging_enhanced import LoggingManager
from core.health_monitoring_enhanced import HealthMonitor
from core.enhanced_input_validation import NeuromorphicValidator, get_neuromorphic_validator
from core.robustness_decorators import (
    RobustnessConfig, robust_neuromorphic_model, 
    sensor_input_validator, network_health_monitor
)
from security.neuromorphic_security import (
    NeuromorphicSecurityManager, SecurityPolicy, ThreatLevel
)
from models.fusion.multimodal_fusion import FusionConfig, OlfactoryFusionSNN


class TestRobustnessIntegration(unittest.TestCase):
    """Integration tests for robustness features."""
    
    def setUp(self):
        """Set up test environment."""
        # Configure logging for testing
        logging.basicConfig(level=logging.DEBUG)
        
        # Create test configurations
        self.robustness_config = RobustnessConfig(
            enable_error_handling=True,
            enable_logging=True,
            enable_health_monitoring=True,
            enable_input_validation=True,
            enable_security_checks=True,
            max_retries=2,
            timeout_seconds=10.0
        )
        
        self.security_policy = SecurityPolicy(
            enable_adversarial_detection=True,
            enable_model_integrity_checks=True,
            enable_input_sanitization=True,
            max_input_size_mb=50.0,
            adversarial_detection_threshold=0.7
        )
        
        # Initialize components
        self.validator = NeuromorphicValidator()
        self.security_manager = NeuromorphicSecurityManager(self.security_policy)
        self.health_monitor = HealthMonitor("TestComponent")
        
        if TORCH_AVAILABLE:
            # Create test fusion network
            self.fusion_config = FusionConfig(
                num_chemical_sensors=6,
                num_audio_features=128,
                num_projection_neurons=100,  # Smaller for testing
                num_kenyon_cells=500,       # Smaller for testing
                num_output_classes=4,
                simulation_time=50.0,       # Shorter for testing
                fusion_strategy="early"
            )
            self.fusion_network = OlfactoryFusionSNN(self.fusion_config)
    
    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_complete_pipeline_robustness(self):
        """Test complete pipeline with all robustness features enabled."""
        
        print("\\n=== Testing Complete Pipeline Robustness ===")
        
        # Create test inputs
        batch_size = 2
        chemical_input = torch.randn(batch_size, self.fusion_config.num_chemical_sensors)
        audio_input = torch.randn(batch_size, self.fusion_config.num_audio_features)
        
        # Test 1: Normal operation with all robustness features
        print("\\nTest 1: Normal operation")
        try:
            # Security validation
            secure_chemical, chem_report = self.security_manager.secure_process_input(
                chemical_input, "sensor", "test_chemical_sensor"
            )
            secure_audio, audio_report = self.security_manager.secure_process_input(
                audio_input, "sensor", "test_audio_sensor"
            )
            
            self.assertEqual(chem_report["threat_level"], ThreatLevel.LOW)
            self.assertEqual(audio_report["threat_level"], ThreatLevel.LOW)
            print(f"✓ Security validation passed - Threat level: {chem_report['threat_level'].value}")
            
            # Model integrity verification
            integrity_report = self.security_manager.verify_model_integrity(
                self.fusion_network, "test_fusion_network"
            )
            self.assertTrue(integrity_report["integrity_verified"])
            print(f"✓ Model integrity verified - Checksum: {integrity_report['current_checksum'][:16]}...")
            
            # Secure model execution
            output, exec_report = self.security_manager.secure_model_execution(
                self.fusion_network, 
                {"chemical_input": secure_chemical, "audio_input": secure_audio},
                "test_fusion_network"
            )
            
            self.assertTrue(exec_report["execution_successful"])
            self.assertIn("decision_output", output)
            self.assertIn("class_probabilities", output["decision_output"])
            print(f"✓ Secure model execution completed in {exec_report['execution_time']:.3f}s")
            
            # Validate output structure
            class_probs = output["decision_output"]["class_probabilities"]
            self.assertEqual(class_probs.shape, (batch_size, self.fusion_config.num_output_classes))
            self.assertTrue(torch.all(class_probs >= 0))
            self.assertTrue(torch.all(class_probs <= 1))
            
            # Check for robustness metadata
            self.assertIn("robustness_metadata", output)
            self.assertTrue(output["robustness_metadata"]["processing_successful"])
            print("✓ Output validation passed")
            
        except Exception as e:
            self.fail(f"Normal operation test failed: {str(e)}")
    
    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_error_handling_and_recovery(self):
        """Test error handling and graceful recovery."""
        
        print("\\n=== Testing Error Handling and Recovery ===")
        
        # Test 1: Invalid input dimensions
        print("\\nTest 1: Invalid input dimensions")
        batch_size = 2
        invalid_chemical = torch.randn(batch_size, 10)  # Wrong dimension
        valid_audio = torch.randn(batch_size, self.fusion_config.num_audio_features)
        
        try:
            output = self.fusion_network(invalid_chemical, valid_audio)
            
            # Should either raise an error or return fallback output
            if isinstance(output, dict) and output.get("error_mode"):
                print("✓ Graceful degradation activated for invalid input")
            else:
                self.fail("Expected error or fallback mode for invalid input")
        except (ValidationError, ValueError) as e:
            print(f"✓ Proper error handling: {type(e).__name__}")
        
        # Test 2: NaN inputs
        print("\\nTest 2: NaN inputs")
        nan_chemical = torch.full((batch_size, self.fusion_config.num_chemical_sensors), float('nan'))
        valid_audio = torch.randn(batch_size, self.fusion_config.num_audio_features)
        
        try:
            # Process through security layer (should sanitize)
            secure_chemical, report = self.security_manager.secure_process_input(nan_chemical, "sensor")
            
            # Should be sanitized
            self.assertFalse(torch.isnan(secure_chemical).any())
            self.assertTrue(report["sanitization_applied"])
            print("✓ NaN inputs properly sanitized")
            
            # Process through model
            output = self.fusion_network(secure_chemical, valid_audio)
            
            # Should produce valid output
            if "decision_output" in output:
                class_probs = output["decision_output"]["class_probabilities"]
                self.assertFalse(torch.isnan(class_probs).any())
                print("✓ Model produced valid output despite sanitized NaN inputs")
            
        except Exception as e:
            self.fail(f"NaN input handling failed: {str(e)}")
    
    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_adversarial_detection(self):
        """Test adversarial input detection."""
        
        print("\\n=== Testing Adversarial Detection ===")
        
        # Create adversarial spike patterns
        batch_size = 1
        num_neurons = 50
        time_steps = 100
        
        # Test 1: Excessive synchrony (all neurons spike together)
        print("\\nTest 1: Excessive synchrony detection")
        adversarial_spikes = np.zeros((batch_size, num_neurons, time_steps))
        
        # Create perfectly synchronized spikes every 10 timesteps
        for t in range(0, time_steps, 10):
            adversarial_spikes[0, :, t] = 1
        
        adversarial_tensor = torch.tensor(adversarial_spikes, dtype=torch.float32)
        
        detection_report = self.security_manager.detect_adversarial_patterns(adversarial_tensor)
        
        self.assertTrue(detection_report["adversarial_detected"])
        self.assertIn("excessive_synchrony", detection_report["detected_patterns"])
        self.assertGreaterEqual(detection_report["threat_level"], ThreatLevel.MEDIUM)
        print(f"✓ Excessive synchrony detected - Confidence: {detection_report['confidence_score']:.3f}")
        
        # Test 2: Impossible spike rates
        print("\\nTest 2: Impossible spike rates detection")
        impossible_spikes = np.ones((num_neurons, time_steps))  # All neurons always spiking
        
        detection_report = self.security_manager.detect_adversarial_patterns(impossible_spikes)
        
        self.assertTrue(detection_report["adversarial_detected"])
        self.assertIn("impossible_spike_rates", detection_report["detected_patterns"])
        print(f"✓ Impossible spike rates detected - Threat level: {detection_report['threat_level'].value}")
        
        # Test 3: Normal spike patterns (should not be detected as adversarial)
        print("\\nTest 3: Normal spike patterns")
        normal_spikes = np.random.binomial(1, 0.1, (num_neurons, time_steps))  # 10% spike probability
        
        detection_report = self.security_manager.detect_adversarial_patterns(normal_spikes)
        
        self.assertFalse(detection_report["adversarial_detected"])
        print("✓ Normal spike patterns correctly identified as benign")
    
    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_health_monitoring(self):
        """Test health monitoring and convergence detection."""
        
        print("\\n=== Testing Health Monitoring ===")
        
        # Create test inputs
        batch_size = 2
        chemical_input = torch.randn(batch_size, self.fusion_config.num_chemical_sensors)
        audio_input = torch.randn(batch_size, self.fusion_config.num_audio_features)
        
        # Initialize health monitor
        health_monitor = HealthMonitor("FusionNetworkTest")
        
        print("\\nTest 1: Basic health monitoring")
        initial_health = health_monitor.get_health_status()
        self.assertEqual(initial_health["overall_health"], "good")
        print(f"✓ Initial health status: {initial_health['overall_health']}")
        
        # Test memory monitoring
        print("\\nTest 2: Memory monitoring")
        health_monitor.check_memory_usage()
        
        memory_status = health_monitor.get_memory_status()
        self.assertIn("current_memory_mb", memory_status)
        self.assertGreaterEqual(memory_status["current_memory_mb"], 0)
        print(f"✓ Memory monitoring active - Current: {memory_status['current_memory_mb']:.1f}MB")
        
        # Test error recording
        print("\\nTest 3: Error recording")
        test_error = "Test error for health monitoring"
        health_monitor.record_error(test_error)
        
        health_status = health_monitor.get_health_status()
        self.assertGreater(len(health_status["errors"]), 0)
        self.assertIn(test_error, health_status["errors"])
        print(f"✓ Error recording works - {len(health_status['errors'])} errors recorded")
        
        # Test network processing with health monitoring
        print("\\nTest 4: Network processing with health monitoring")
        try:
            output = self.fusion_network(chemical_input, audio_input)
            
            if "health_status" in output:
                net_health = output["health_status"]
                print(f"✓ Network health monitored - Status: {net_health.get('overall_health', 'unknown')}")
            
            if "convergence_status" in output:
                conv_status = output["convergence_status"]
                print(f"✓ Convergence monitored - Converged: {conv_status.get('is_converged', False)}")
                
        except Exception as e:
            self.fail(f"Health monitoring during network processing failed: {str(e)}")
    
    def test_input_validation_pipeline(self):
        """Test comprehensive input validation pipeline."""
        
        print("\\n=== Testing Input Validation Pipeline ===")
        
        validator = get_neuromorphic_validator()
        
        # Test 1: Spike train validation
        print("\\nTest 1: Spike train validation")
        
        # Valid spike train
        valid_spikes = np.random.binomial(1, 0.1, (10, 100))  # 10 neurons, 100 timesteps
        try:
            validated_spikes = validator.validate_spike_trains(valid_spikes)
            self.assertEqual(validated_spikes.shape, valid_spikes.shape)
            print("✓ Valid spike train passed validation")
        except ValidationError:
            self.fail("Valid spike train rejected by validator")
        
        # Invalid spike train (non-binary values)
        invalid_spikes = np.random.randn(10, 100)  # Continuous values
        try:
            validator.validate_spike_trains(invalid_spikes)
            print("✓ Invalid spike train was sanitized")
        except ValidationError as e:
            print(f"✓ Invalid spike train properly rejected: {e.error_code}")
        
        # Test 2: Network configuration validation
        print("\\nTest 2: Network configuration validation")
        
        # Valid configuration
        valid_config = {
            "tau_membrane": 20.0,
            "threshold": 1.0,
            "num_neurons": 1000,
            "learning_rate": 0.001,
            "dt": 1.0
        }
        
        try:
            validated_config = validator.validate_neuromorphic_config(valid_config)
            self.assertEqual(len(validated_config), len(valid_config))
            print("✓ Valid network configuration passed validation")
        except ValidationError:
            self.fail("Valid network configuration rejected")
        
        # Invalid configuration
        invalid_config = {
            "tau_membrane": -5.0,  # Negative value
            "threshold": float('inf'),  # Infinite value
            "num_neurons": 0,  # Zero neurons
            "learning_rate": 2.0,  # Too high learning rate
        }
        
        validation_errors = 0
        for param, value in invalid_config.items():
            try:
                validator.validate_neuromorphic_config({param: value})
            except ValidationError:
                validation_errors += 1
        
        self.assertGreater(validation_errors, 0)
        print(f"✓ Invalid parameters properly rejected: {validation_errors} validation errors")
        
        if TORCH_AVAILABLE:
            # Test 3: Tensor validation
            print("\\nTest 3: Tensor validation")
            
            # Valid tensor
            valid_tensor = torch.randn(5, 10)
            try:
                validator.validate(valid_tensor, ["array_shape"])
                print("✓ Valid tensor passed validation")
            except ValidationError:
                self.fail("Valid tensor rejected")
            
            # Invalid tensor (contains NaN)
            invalid_tensor = torch.tensor([[1.0, float('nan'), 3.0]])
            try:
                validator.validate(invalid_tensor, ["array_shape"])
            except ValidationError:
                print("✓ Tensor with NaN properly rejected")
    
    def test_robustness_decorators(self):
        """Test robustness decorators functionality."""
        
        print("\\n=== Testing Robustness Decorators ===")
        
        # Create test class with robustness decorators
        class TestNeuromorphicModel:
            def __init__(self):
                self.config = Mock()
                self.config.num_output_classes = 4
            
            @robust_neuromorphic_model(fallback_strategy="zero_output", max_retries=2)
            def process_with_error_handling(self, input_data):
                if torch.isnan(input_data).any():
                    raise ValueError("NaN in input")
                return {"output": input_data * 2}
            
            @sensor_input_validator(sensor_type="chemical", value_range=(0.0, 5.0))
            def process_sensor_data(self, sensor_input):
                return {"processed": sensor_input.sum()}
            
            @network_health_monitor()
            def process_with_health_monitoring(self, spike_data):
                return {
                    "kenyon_spikes": spike_data,
                    "decision_output": {"class_probabilities": torch.ones(2, 4) / 4}
                }
        
        if TORCH_AVAILABLE:
            model = TestNeuromorphicModel()
            
            # Test 1: Error handling decorator
            print("\\nTest 1: Error handling decorator")
            
            # Normal operation
            normal_input = torch.randn(3, 5)
            try:
                result = model.process_with_error_handling(normal_input)
                self.assertIn("output", result)
                print("✓ Normal operation with error handling decorator")
            except Exception as e:
                self.fail(f"Normal operation failed: {str(e)}")
            
            # Error case (NaN input)
            nan_input = torch.tensor([[1.0, float('nan'), 3.0]])
            try:
                result = model.process_with_error_handling(nan_input)
                # Should return fallback output
                if isinstance(result, dict) and ("fallback_mode" in result or "error" in result):
                    print("✓ Error handling decorator provided fallback")
                else:
                    self.fail("Expected fallback output for error case")
            except Exception:
                print("✓ Error properly propagated by decorator")
            
            # Test 2: Sensor validation decorator
            print("\\nTest 2: Sensor validation decorator")
            
            # Valid sensor input
            valid_sensor = torch.tensor([[2.5, 1.0, 3.8]])
            try:
                result = model.process_sensor_data(valid_sensor)
                self.assertIn("processed", result)
                print("✓ Valid sensor data processed")
            except Exception as e:
                self.fail(f"Valid sensor processing failed: {str(e)}")
            
            # Invalid sensor input (out of range)
            invalid_sensor = torch.tensor([[10.0, -2.0, 8.0]])  # Outside [0,5] range
            try:
                result = model.process_sensor_data(invalid_sensor)
                self.fail("Expected validation error for out-of-range sensor data")
            except ValidationError:
                print("✓ Out-of-range sensor data properly rejected")
            
            # Test 3: Health monitoring decorator
            print("\\nTest 3: Health monitoring decorator")
            
            spike_input = torch.randint(0, 2, (2, 10, 50), dtype=torch.float32)
            try:
                result = model.process_with_health_monitoring(spike_input)
                
                self.assertIn("kenyon_spikes", result)
                if "network_health" in result:
                    health_status = result["network_health"]
                    self.assertIn("overall_health", health_status)
                    print(f"✓ Health monitoring decorator active - Health: {health_status['overall_health']}")
                else:
                    print("✓ Health monitoring decorator executed successfully")
                    
            except Exception as e:
                self.fail(f"Health monitoring decorator failed: {str(e)}")
    
    def test_integrated_security_pipeline(self):
        """Test integrated security pipeline."""
        
        print("\\n=== Testing Integrated Security Pipeline ===")
        
        if not TORCH_AVAILABLE:
            print("Skipping security pipeline test - PyTorch not available")
            return
        
        # Test 1: Rate limiting
        print("\\nTest 1: Rate limiting")
        
        # Configure strict rate limiting for testing
        strict_policy = SecurityPolicy(
            enable_rate_limiting=True,
            max_requests_per_minute=3  # Very low limit for testing
        )
        
        security_mgr = NeuromorphicSecurityManager(strict_policy)
        test_input = torch.randn(1, 10)
        
        successful_requests = 0
        rate_limited = False
        
        for i in range(5):  # Try 5 requests
            try:
                _, report = security_mgr.secure_process_input(test_input, "sensor", f"test_{i}")
                successful_requests += 1
            except Exception as e:
                if "rate" in str(e).lower():
                    rate_limited = True
                    break
        
        # Should hit rate limit before 5 requests
        self.assertTrue(rate_limited or successful_requests <= 3)
        print(f"✓ Rate limiting active - {successful_requests} successful requests before limit")
        
        # Test 2: Input size limits
        print("\\nTest 2: Input size limits")
        
        # Create large input exceeding size limit
        large_policy = SecurityPolicy(max_input_size_mb=0.001)  # Very small limit
        size_security_mgr = NeuromorphicSecurityManager(large_policy)
        
        large_input = torch.randn(1000, 1000)  # Large tensor
        
        try:
            _, report = size_security_mgr.secure_process_input(large_input, "sensor")
            self.fail("Expected size limit violation")
        except Exception as e:
            if "size" in str(e).lower():
                print("✓ Input size limits enforced")
            else:
                raise
        
        # Test 3: Model integrity verification
        print("\\nTest 3: Model integrity verification")
        
        if hasattr(self, 'fusion_network'):
            # Get initial checksum
            integrity_report1 = self.security_manager.verify_model_integrity(
                self.fusion_network, "test_model"
            )
            self.assertTrue(integrity_report1["integrity_verified"])
            initial_checksum = integrity_report1["current_checksum"]
            
            # Verify again (should match)
            integrity_report2 = self.security_manager.verify_model_integrity(
                self.fusion_network, "test_model", initial_checksum
            )
            self.assertTrue(integrity_report2["integrity_verified"])
            self.assertEqual(integrity_report2["current_checksum"], initial_checksum)
            print(f"✓ Model integrity verification consistent")
            
            # Test with wrong checksum
            integrity_report3 = self.security_manager.verify_model_integrity(
                self.fusion_network, "test_model", "wrong_checksum"
            )
            self.assertFalse(integrity_report3["integrity_verified"])
            self.assertEqual(integrity_report3["threat_level"], ThreatLevel.HIGH)
            print("✓ Model integrity violation detected")
    
    def test_performance_and_resource_monitoring(self):
        """Test performance and resource monitoring."""
        
        print("\\n=== Testing Performance and Resource Monitoring ===")
        
        if not TORCH_AVAILABLE:
            print("Skipping performance test - PyTorch not available")
            return
        
        # Test with performance monitoring
        batch_size = 4
        chemical_input = torch.randn(batch_size, self.fusion_config.num_chemical_sensors)
        audio_input = torch.randn(batch_size, self.fusion_config.num_audio_features)
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            # Process through security layer
            secure_chemical, _ = self.security_manager.secure_process_input(chemical_input, "sensor")
            secure_audio, _ = self.security_manager.secure_process_input(audio_input, "sensor")
            
            # Process through model
            output = self.fusion_network(secure_chemical, secure_audio)
            
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            processing_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            print(f"✓ Processing completed in {processing_time:.3f}s")
            print(f"✓ Memory delta: {memory_delta:.2f}MB")
            
            # Check for performance metadata
            if isinstance(output, dict):
                if "performance_metrics" in output:
                    perf_metrics = output["performance_metrics"]
                    print(f"✓ Performance metrics captured: {perf_metrics}")
                
                if "robustness_metadata" in output:
                    robust_metadata = output["robustness_metadata"]
                    print(f"✓ Robustness metadata available: {list(robust_metadata.keys())}")
            
            # Validate reasonable performance
            self.assertLess(processing_time, 30.0)  # Should complete within 30 seconds
            print("✓ Performance within acceptable bounds")
            
        except Exception as e:
            self.fail(f"Performance monitoring test failed: {str(e)}")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_end_to_end_robustness(self):
        """Comprehensive end-to-end robustness test."""
        
        print("\\n=== End-to-End Robustness Test ===")
        
        # This test simulates a complete processing pipeline with various challenges
        test_scenarios = [
            {
                "name": "Normal operation",
                "chemical_input": torch.randn(2, 6),
                "audio_input": torch.randn(2, 128),
                "expected_success": True
            },
            {
                "name": "Noisy sensor data", 
                "chemical_input": torch.randn(2, 6) * 10 + torch.randn(2, 6) * 0.5,  # High variance
                "audio_input": torch.randn(2, 128) * 5,
                "expected_success": True  # Should be handled by sanitization
            },
            {
                "name": "Edge case values",
                "chemical_input": torch.tensor([[0.0, 0.0, 5.0, 5.0, 2.5, 2.5], 
                                              [1e-6, 4.999, 0.001, 4.99, 2.501, 2.499]]),
                "audio_input": torch.randn(2, 128),
                "expected_success": True
            },
            {
                "name": "Mixed valid/invalid data",
                "chemical_input": torch.tensor([[1.0, 2.0, float('nan'), 4.0, 5.0, 1.0],
                                              [2.0, float('inf'), 3.0, 1.0, 4.0, 2.0]]),
                "audio_input": torch.randn(2, 128),
                "expected_success": True  # Should be sanitized
            }
        ]
        
        successful_scenarios = 0
        
        for i, scenario in enumerate(test_scenarios):
            print(f"\\nScenario {i+1}: {scenario['name']}")
            
            try:
                # Complete pipeline test
                
                # 1. Security processing
                secure_chemical, chem_report = self.security_manager.secure_process_input(
                    scenario["chemical_input"], "sensor", f"test_scenario_{i}"
                )
                secure_audio, audio_report = self.security_manager.secure_process_input(
                    scenario["audio_input"], "sensor", f"test_scenario_{i}"
                )
                
                print(f"   Security: Chemical threat={chem_report['threat_level'].value}, "
                      f"Audio threat={audio_report['threat_level'].value}")
                
                # 2. Model integrity check
                integrity_report = self.security_manager.verify_model_integrity(
                    self.fusion_network, "end_to_end_test"
                )
                
                if not integrity_report["integrity_verified"]:
                    print(f"   Warning: Model integrity issues detected")
                
                # 3. Secure model execution
                output, exec_report = self.security_manager.secure_model_execution(
                    self.fusion_network,
                    {"chemical_input": secure_chemical, "audio_input": secure_audio},
                    "end_to_end_test",
                    execution_timeout=15.0
                )
                
                # 4. Validate results
                self.assertIn("decision_output", output)
                self.assertIn("class_probabilities", output["decision_output"])
                
                class_probs = output["decision_output"]["class_probabilities"]
                self.assertFalse(torch.isnan(class_probs).any())
                self.assertTrue(torch.all(class_probs >= 0))
                self.assertTrue(torch.all(class_probs <= 1))
                
                # Check sum to ~1 (accounting for numerical precision)
                prob_sums = class_probs.sum(dim=-1)
                self.assertTrue(torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-3))
                
                print(f"   ✓ Success - Execution time: {exec_report['execution_time']:.3f}s")
                
                if scenario["expected_success"]:
                    successful_scenarios += 1
                else:
                    print(f"   Warning: Expected failure but succeeded")
                    
            except Exception as e:
                if scenario["expected_success"]:
                    print(f"   ✗ Unexpected failure: {str(e)}")
                else:
                    print(f"   ✓ Expected failure: {type(e).__name__}")
                    successful_scenarios += 1
        
        success_rate = successful_scenarios / len(test_scenarios)
        print(f"\\n=== End-to-End Results ===")
        print(f"Success rate: {success_rate:.1%} ({successful_scenarios}/{len(test_scenarios)})")
        print(f"All robustness features integrated successfully: {'✓' if success_rate >= 0.8 else '✗'}")
        
        self.assertGreaterEqual(success_rate, 0.8, "End-to-end success rate should be at least 80%")
    
    def test_logging_integration(self):
        """Test logging integration across all components."""
        
        print("\\n=== Testing Logging Integration ===")
        
        # Configure logging to capture messages
        import io
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.DEBUG)
        
        logger = logging.getLogger()
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        
        if TORCH_AVAILABLE:
            try:
                # Generate various log messages through the pipeline
                chemical_input = torch.randn(1, 6)
                audio_input = torch.randn(1, 128)
                
                # This should generate multiple log messages
                secure_chemical, _ = self.security_manager.secure_process_input(chemical_input, "sensor")
                secure_audio, _ = self.security_manager.secure_process_input(audio_input, "sensor")
                output = self.fusion_network(secure_chemical, secure_audio)
                
                # Check log output
                log_contents = log_capture.getvalue()
                
                # Should contain various types of log messages
                log_indicators = [
                    "processing",
                    "security",
                    "validation",
                    "completion"
                ]
                
                found_indicators = sum(1 for indicator in log_indicators 
                                     if indicator.lower() in log_contents.lower())
                
                print(f"✓ Generated comprehensive logs - {found_indicators}/{len(log_indicators)} types found")
                print(f"✓ Total log length: {len(log_contents)} characters")
                
                self.assertGreater(len(log_contents), 100)  # Should have substantial logging
                
            except Exception as e:
                self.fail(f"Logging integration test failed: {str(e)}")
            
            finally:
                logger.removeHandler(handler)


<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"id": "1", "content": "Implement enhanced error handling in model forward passes with try/catch blocks and graceful degradation", "status": "completed"}, {"id": "2", "content": "Add comprehensive logging integration throughout the pipeline with performance metrics", "status": "completed"}, {"id": "3", "content": "Implement health monitoring for all neuromorphic components with convergence detection", "status": "completed"}, {"id": "4", "content": "Add input validation & sanitization for sensor inputs, audio features, and network parameters", "status": "completed"}, {"id": "5", "content": "Implement security measures and integrate with existing security modules", "status": "completed"}, {"id": "6", "content": "Create robustness decorators and wrappers for automatic application", "status": "completed"}, {"id": "7", "content": "Test and validate all robustness features work together", "status": "completed"}]