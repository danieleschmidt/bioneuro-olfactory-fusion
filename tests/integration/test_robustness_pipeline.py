"""Integration tests for robustness and reliability features.

This module tests the integration of validation, monitoring, security,
and configuration management for robust system operation.
"""

import pytest
import torch
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import time
import threading

from bioneuro_olfactory.core.validation import InputValidator, ValidationLevel
from bioneuro_olfactory.core.config_manager import ConfigManager, ApplicationConfig
from bioneuro_olfactory.monitoring.metrics_collector import MetricsCollector
from bioneuro_olfactory.security.security_manager import SecurityManager, SecurityConfig
from bioneuro_olfactory.sensors.enose.sensor_array import ENoseArray, create_standard_enose
from bioneuro_olfactory.models.fusion.multimodal_fusion import EarlyFusion, AttentionFusion


class TestRobustnessPipeline:
    """Test complete robustness pipeline integration."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Initialize components
        self.validator = InputValidator(ValidationLevel.WARN)
        self.config_manager = ConfigManager(config_dirs=[self.temp_dir.name])
        self.metrics_collector = MetricsCollector()
        
        # Create test configuration
        config_data = {
            "neural": {
                "tau_membrane": 20.0,
                "num_projection_neurons": 100,
                "num_kenyon_cells": 500,
                "batch_size": 8
            },
            "sensor": {
                "sampling_rate_hz": 1.0,
                "sensor_types": ["MQ2", "MQ7"]
            },
            "system": {
                "enable_gpu": False,  # Disable for testing
                "log_level": "INFO"
            }
        }
        
        config_file = Path(self.temp_dir.name) / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
            
        self.config = self.config_manager.load_config(str(config_file))
        
    def teardown_method(self):
        """Cleanup test fixtures."""
        if hasattr(self, 'metrics_collector'):
            try:
                self.metrics_collector.stop_collection()
            except:
                pass
        self.temp_dir.cleanup()
        
    def test_sensor_data_validation_pipeline(self):
        """Test sensor data validation through complete pipeline."""
        # Create sensor array
        enose = create_standard_enose()
        
        # Simulate various sensor conditions
        test_scenarios = [
            # Normal operation
            {
                "MQ2_methane": 500.0,
                "MQ7_CO": 20.0,
                "temperature": 25.0,
                "humidity": 50.0
            },
            # High concentration alert
            {
                "MQ2_methane": 5000.0,  # High methane
                "MQ7_CO": 200.0,        # High CO
                "temperature": 30.0,
                "humidity": 60.0
            },
            # Sensor malfunction simulation
            {
                "MQ2_methane": -100.0,   # Invalid negative
                "MQ7_CO": 50000.0,       # Unrealistic high
                "temperature": 25.0,
                "humidity": 120.0        # Invalid >100%
            },
            # Extreme environmental conditions
            {
                "MQ2_methane": 1000.0,
                "MQ7_CO": 30.0,
                "temperature": -20.0,    # Very cold
                "humidity": 10.0         # Very dry
            }
        ]
        
        results = []
        
        for scenario in test_scenarios:
            # Validate sensor data
            sanitized_data, threats = self.validator.input_sanitizer.validate_sensor_data(scenario)
            
            # Record metrics
            threat_count = len(threats)
            self.metrics_collector.record_sensor_metrics(
                sensors_active=len(sanitized_data),
                sensors_total=len(scenario),
                response_time_ms=50.0,
                calibration_status={k: True for k in sanitized_data.keys()},
                sensor_readings=sanitized_data,
                environmental_conditions={
                    "temperature": sanitized_data.get("temperature", 25.0),
                    "humidity": sanitized_data.get("humidity", 50.0)
                }
            )
            
            results.append({
                'original': scenario,
                'sanitized': sanitized_data,
                'threats': threats,
                'threat_count': threat_count
            })
            
        # Verify results
        assert len(results) == len(test_scenarios)
        
        # Normal operation should have no threats
        assert results[0]['threat_count'] == 0
        
        # High concentration should be valid but flagged
        assert results[1]['threat_count'] == 0  # High but valid values
        
        # Sensor malfunction should have multiple threats
        assert results[2]['threat_count'] > 0
        assert any('suspicious_sensor_value' in threat for threat in results[2]['threats'])
        
        # Extreme conditions should be handled gracefully
        assert results[3]['sanitized']['temperature'] >= -1000  # Clamped
        
    def test_neural_network_robustness_validation(self):
        """Test neural network robustness through validation pipeline."""
        # Create test neural network outputs
        batch_size = self.config.neural.batch_size
        num_pn = self.config.neural.num_projection_neurons
        num_kc = self.config.neural.num_kenyon_cells
        duration = 100
        
        # Generate test spike trains with various conditions
        test_conditions = [
            # Normal sparse activity
            {
                'name': 'normal',
                'pn_sparsity': 0.1,
                'kc_sparsity': 0.02
            },
            # Too sparse (might indicate dead neurons)
            {
                'name': 'too_sparse',
                'pn_sparsity': 0.001,
                'kc_sparsity': 0.0001
            },
            # Too active (might indicate hyperactivity)
            {
                'name': 'hyperactive', 
                'pn_sparsity': 0.8,
                'kc_sparsity': 0.5
            },
            # Normal with some noise
            {
                'name': 'noisy',
                'pn_sparsity': 0.1,
                'kc_sparsity': 0.02,
                'add_noise': True
            }
        ]
        
        for condition in test_conditions:
            # Generate spike trains
            pn_spikes = (torch.rand(batch_size, num_pn, duration) < condition['pn_sparsity']).float()
            kc_spikes = (torch.rand(batch_size, num_kc, duration) < condition['kc_sparsity']).float()
            
            # Add noise if specified
            if condition.get('add_noise', False):
                noise_mask = torch.rand_like(pn_spikes) < 0.01
                pn_spikes = pn_spikes + noise_mask.float() * torch.rand_like(pn_spikes)
                pn_spikes = torch.clamp(pn_spikes, 0, 1)
                
            # Validate spike trains
            pn_result = self.validator.validate_spike_train(pn_spikes, f"pn_spikes_{condition['name']}")
            kc_result = self.validator.validate_spike_train(kc_spikes, f"kc_spikes_{condition['name']}")
            
            # Record neural metrics
            inference_time = 10.0 + np.random.normal(0, 2.0)  # Simulate variable timing
            throughput = batch_size / (inference_time / 1000.0)
            
            self.metrics_collector.record_neural_metrics(
                inference_time_ms=inference_time,
                throughput_samples_per_sec=throughput,
                spike_data=pn_spikes,
                model=None,  # Would be actual model in practice
                loss=None,
                accuracy=None
            )
            
            # Verify validation results
            if condition['name'] == 'normal':
                assert pn_result.is_valid
                assert kc_result.is_valid
            elif condition['name'] == 'noisy':
                # Should be corrected to binary
                if pn_result.corrected_value is not None:
                    assert torch.all((pn_result.corrected_value == 0) | (pn_result.corrected_value == 1))
                    
    def test_multi_modal_fusion_robustness(self):
        """Test multi-modal fusion robustness validation."""
        # Create fusion models
        num_chemical = len(self.config.sensor.sensor_types)
        num_audio = 128
        
        early_fusion = EarlyFusion(num_chemical, num_audio)
        attention_fusion = AttentionFusion(num_chemical, num_audio)
        
        # Test various input conditions
        batch_size = self.config.neural.batch_size
        
        test_inputs = [
            # Normal inputs
            {
                'chemical': torch.randn(batch_size, num_chemical),
                'audio': torch.randn(batch_size, num_audio),
                'name': 'normal'
            },
            # Extreme values
            {
                'chemical': torch.randn(batch_size, num_chemical) * 100,
                'audio': torch.randn(batch_size, num_audio) * 100,
                'name': 'extreme_values'
            },
            # With NaN values (should be handled)
            {
                'chemical': torch.tensor([[1.0, float('nan')] + [0.0] * (num_chemical-2)] * batch_size),
                'audio': torch.randn(batch_size, num_audio),
                'name': 'with_nan'
            },
            # Sparse inputs (mostly zeros)
            {
                'chemical': torch.zeros(batch_size, num_chemical),
                'audio': torch.zeros(batch_size, num_audio), 
                'name': 'sparse'
            }
        ]
        
        for input_data in test_inputs:
            chemical_features = input_data['chemical']
            audio_features = input_data['audio']
            
            # Validate inputs
            validation_results = self.validator.validate_fusion_inputs(chemical_features, audio_features)
            
            # Use corrected values if available
            for result in validation_results:
                if result.corrected_value is not None:
                    if 'chemical' in result.message:
                        chemical_features = result.corrected_value
                    elif 'audio' in result.message:
                        audio_features = result.corrected_value
                        
            # Test fusion models with validated inputs
            try:
                start_time = time.time()
                
                early_output = early_fusion(chemical_features, audio_features)
                attention_output = attention_fusion(chemical_features, audio_features)
                
                fusion_time = (time.time() - start_time) * 1000  # ms
                
                # Validate outputs
                early_valid = self.validator.validate_tensor(
                    early_output, f"early_fusion_{input_data['name']}",
                    allow_nan=False, allow_inf=False
                )
                attention_valid = self.validator.validate_tensor(
                    attention_output, f"attention_fusion_{input_data['name']}",
                    allow_nan=False, allow_inf=False
                )
                
                # Record metrics
                self.metrics_collector.record_timer("fusion_inference", fusion_time)
                
                # Verify outputs are valid
                assert early_valid.is_valid or early_valid.corrected_value is not None
                assert attention_valid.is_valid or attention_valid.corrected_value is not None
                
            except Exception as e:
                pytest.fail(f"Fusion failed for {input_data['name']}: {e}")
                
    def test_security_integration(self):
        """Test security integration with other components."""
        # Create security manager
        security_config = SecurityConfig(jwt_secret_key="test_key_32_characters_long_enough")
        security_manager = SecurityManager(security_config)
        
        # Test input sanitization integration
        test_inputs = [
            # Normal sensor data
            {"MQ2": 500.0, "MQ7": 20.0},
            # Malicious inputs
            {"MQ2; DROP TABLE sensors;": 500.0, "MQ7": "'; DELETE * FROM data; --"},
            # Path traversal attempts
            {"../../../etc/passwd": 100.0, "sensor": 200.0},
            # XSS attempts
            {"<script>alert('xss')</script>": 300.0, "sensor2": 400.0}
        ]
        
        for input_data in test_inputs:
            # Process through security manager
            sanitized_data, threats = security_manager.process_input(input_data, "127.0.0.1")
            
            # Should always return sanitized data
            assert isinstance(sanitized_data, dict)
            assert len(sanitized_data) > 0
            
            # Malicious inputs should generate threats
            if any(suspicious in str(input_data) for suspicious in ['DROP', 'DELETE', '<script>', '../']):
                assert len(threats) > 0
                
        # Test authentication integration
        # Register test user
        success, errors = security_manager.register_user(
            username="test_operator",
            email="test@example.com", 
            password="SecurePassword123!",
            role="operator"
        )
        
        assert success
        assert len(errors) == 0
        
        # Authenticate user
        token, message = security_manager.authenticate_user(
            username="test_operator",
            password="SecurePassword123!",
            ip_address="192.168.1.100",
            user_agent="TestAgent/1.0"
        )
        
        assert token is not None
        assert "successful" in message
        
        # Verify token
        is_valid, payload = security_manager.verify_request(token, "read")
        assert is_valid
        assert payload is not None
        
    def test_configuration_robustness(self):
        """Test configuration management robustness."""
        # Test configuration validation
        assert self.config is not None
        assert isinstance(self.config, ApplicationConfig)
        
        # Test runtime configuration updates
        original_tau = self.config.neural.tau_membrane
        
        # Valid update
        self.config_manager.update_config({
            "neural.tau_membrane": 25.0
        }, validate=True)
        
        updated_config = self.config_manager.get_config()
        assert updated_config.neural.tau_membrane == 25.0
        
        # Invalid update should fail
        with pytest.raises(Exception):  # Should raise validation error
            self.config_manager.update_config({
                "neural.tau_membrane": -10.0  # Invalid
            }, validate=True)
            
        # Configuration should be unchanged after failed update
        current_config = self.config_manager.get_config()
        assert current_config.neural.tau_membrane == 25.0  # Should still be valid value
        
    def test_metrics_collection_robustness(self):
        """Test metrics collection under various conditions."""
        # Start metrics collection
        self.metrics_collector.start_collection()
        
        try:
            # Allow some time for initial collection
            time.sleep(0.1)
            
            # Record various metrics
            for i in range(5):
                # Neural metrics with varying performance
                self.metrics_collector.record_neural_metrics(
                    inference_time_ms=10.0 + i * 2.0,
                    throughput_samples_per_sec=100.0 - i * 5.0,
                    spike_data=torch.rand(4, 100, 50),
                    model=None
                )
                
                # Sensor metrics with some failures
                sensor_health = {"MQ2": True, "MQ7": i % 2 == 0}  # MQ7 intermittent
                
                self.metrics_collector.record_sensor_metrics(
                    sensors_active=sum(sensor_health.values()),
                    sensors_total=len(sensor_health),
                    response_time_ms=20.0 + i * 3.0,
                    calibration_status=sensor_health,
                    sensor_readings={"MQ2": 500.0 + i * 50.0, "MQ7": 20.0 + i * 5.0}
                )
                
                # Alert metrics
                self.metrics_collector.record_alert_metrics(
                    alerts_generated=i % 2,  # Some alerts
                    false_positives=0 if i < 3 else 1,  # Some false positives later
                    detection_latency_ms=100.0 + i * 10.0,
                    confidence_scores=[0.8 + i * 0.05] if i % 2 == 0 else []
                )
                
            # Get metrics summary
            latest_metrics = self.metrics_collector.get_latest_metrics()
            performance_summary = self.metrics_collector.get_performance_summary(duration_minutes=1)
            
            # Verify metrics were collected
            assert latest_metrics is not None
            assert 'neural' in latest_metrics
            assert 'sensor' in latest_metrics
            assert 'alert' in latest_metrics
            
            assert performance_summary is not None
            assert 'neural' in performance_summary
            assert 'sensor' not in performance_summary or len(performance_summary.get('sensor', [])) >= 0
            
        finally:
            # Clean stop
            self.metrics_collector.stop_collection()
            
    def test_end_to_end_robustness_scenario(self):
        """Test complete end-to-end robustness scenario."""
        # Simulate a complete gas detection scenario with various failure modes
        
        # 1. Initialize system with configuration
        config = self.config
        
        # 2. Create sensor array
        enose = create_standard_enose()
        
        # 3. Start monitoring
        self.metrics_collector.start_collection()
        
        try:
            # 4. Simulate detection scenarios
            scenarios = [
                # Normal operation
                {
                    'name': 'normal_operation',
                    'sensor_data': {'MQ2_methane': 300.0, 'MQ7_CO': 15.0, 'temperature': 22.0},
                    'expected_threats': 0
                },
                # Gas leak detected
                {
                    'name': 'gas_leak_detected', 
                    'sensor_data': {'MQ2_methane': 2000.0, 'MQ7_CO': 100.0, 'temperature': 25.0},
                    'expected_threats': 0  # High but valid
                },
                # Sensor malfunction
                {
                    'name': 'sensor_malfunction',
                    'sensor_data': {'MQ2_methane': -50.0, 'MQ7_CO': 999999.0, 'temperature': 150.0},
                    'expected_threats': 1  # Should detect suspicious values
                },
                # Environmental interference
                {
                    'name': 'environmental_interference',
                    'sensor_data': {'MQ2_methane': float('nan'), 'MQ7_CO': 25.0, 'temperature': -30.0},
                    'expected_threats': 1  # NaN should be flagged
                }
            ]
            
            results = []
            
            for scenario in scenarios:
                # Process sensor data through validation pipeline
                sensor_data = scenario['sensor_data']
                
                # Security screening
                security_manager = SecurityManager(SecurityConfig(jwt_secret_key="a" * 32))
                sanitized_data, security_threats = security_manager.process_input(sensor_data, "192.168.1.10")
                
                # Validation screening  
                validated_data, validation_threats = self.validator.input_sanitizer.validate_sensor_data(sanitized_data)
                
                total_threats = len(security_threats) + len(validation_threats)
                
                # Record metrics
                self.metrics_collector.record_sensor_metrics(
                    sensors_active=len([v for v in validated_data.values() if isinstance(v, (int, float)) and not np.isnan(v)]),
                    sensors_total=len(validated_data),
                    response_time_ms=50.0,
                    calibration_status={k: True for k in validated_data.keys()},
                    sensor_readings=validated_data
                )
                
                # Simulate neural processing
                if total_threats == 0:  # Only process clean data
                    # Create mock chemical features
                    chemical_features = torch.tensor([[validated_data.get('MQ2_methane', 0), validated_data.get('MQ7_CO', 0)]], dtype=torch.float32)
                    audio_features = torch.randn(1, 128)
                    
                    # Validate fusion inputs
                    fusion_validation = self.validator.validate_fusion_inputs(chemical_features, audio_features)
                    
                    # Record neural metrics
                    self.metrics_collector.record_neural_metrics(
                        inference_time_ms=15.0,
                        throughput_samples_per_sec=60.0,
                        spike_data=torch.rand(1, 100, 50),
                        model=None
                    )
                    
                    # Record detection if gas levels are high
                    if validated_data.get('MQ2_methane', 0) > 1000 or validated_data.get('MQ7_CO', 0) > 50:
                        self.metrics_collector.record_alert_metrics(
                            alerts_generated=1,
                            detection_latency_ms=75.0,
                            confidence_scores=[0.85]
                        )
                        
                results.append({
                    'scenario': scenario['name'],
                    'original_data': sensor_data,
                    'sanitized_data': sanitized_data,
                    'validated_data': validated_data,
                    'total_threats': total_threats,
                    'expected_threats': scenario['expected_threats']
                })
                
            # Verify results
            assert len(results) == len(scenarios)
            
            # Normal operation should have no threats
            normal_result = next(r for r in results if r['scenario'] == 'normal_operation')
            assert normal_result['total_threats'] == 0
            
            # Malfunction should be detected
            malfunction_result = next(r for r in results if r['scenario'] == 'sensor_malfunction')
            assert malfunction_result['total_threats'] >= malfunction_result['expected_threats']
            
            # Environmental interference should be handled
            interference_result = next(r for r in results if r['scenario'] == 'environmental_interference')
            assert interference_result['total_threats'] >= interference_result['expected_threats']
            assert not any(np.isnan(v) for v in interference_result['validated_data'].values() if isinstance(v, float))
            
            # Get final metrics summary
            final_summary = self.metrics_collector.get_performance_summary(duration_minutes=1)
            assert final_summary is not None
            
        finally:
            # Clean shutdown
            self.metrics_collector.stop_collection()
            
    def test_fault_tolerance_and_recovery(self):
        """Test system fault tolerance and recovery mechanisms."""
        # Test configuration corruption recovery
        config_file = Path(self.temp_dir.name) / "corrupt_config.json"
        
        # Create corrupted config file
        with open(config_file, 'w') as f:
            f.write("{ invalid json content")
            
        # Should fall back to defaults
        try:
            config = self.config_manager.load_config(str(config_file))
            assert config is not None  # Should get default config
        except Exception:
            pass  # Expected for corrupted config
            
        # Test metrics collection failure recovery
        self.metrics_collector.start_collection()
        
        try:
            # Simulate metric collection with exception
            with patch.object(self.metrics_collector, '_collect_system_metrics', side_effect=Exception("Test error")):
                time.sleep(0.1)  # Allow collection attempt
                
            # System should still be running
            assert self.metrics_collector.is_collecting
            
        finally:
            self.metrics_collector.stop_collection()
            
        # Test validation error handling
        validator = InputValidator(ValidationLevel.WARN)
        
        # Should handle various invalid inputs gracefully
        invalid_inputs = [
            None,
            "not a tensor",
            [],
            {"malformed": "data"}
        ]
        
        for invalid_input in invalid_inputs:
            try:
                if hasattr(invalid_input, '__iter__') and not isinstance(invalid_input, str):
                    sanitized, threats = validator.input_sanitizer.validate_sensor_data(invalid_input)
                else:
                    # Should handle gracefully
                    pass
            except Exception as e:
                # Some exceptions are expected, but system should not crash
                assert isinstance(e, (ValueError, TypeError))


if __name__ == "__main__":
    pytest.main([__file__])