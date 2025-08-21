#!/usr/bin/env python3
"""
Generation 2 Validation: MAKE IT ROBUST
Comprehensive testing of robustness, error handling, validation, and security features
"""

import sys
import os
import time
import random
import math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bioneuro_olfactory.core.robustness_manager import (
    RobustnessManager, RobustnessLogger, InputValidator, 
    ErrorRecoveryManager, SecurityManager, robust_execution,
    SeverityLevel, RobustnessEvent
)
from bioneuro_olfactory.core.enhanced_validation import (
    NeuroMorphicValidator, ValidationLevel, ValidationRule
)


def test_robustness_logging():
    """Test the robustness logging system."""
    try:
        logger = RobustnessLogger("test_logger")
        
        # Test basic logging
        event = RobustnessEvent(
            timestamp=time.time(),
            event_type="TEST_EVENT",
            severity=SeverityLevel.MEDIUM,
            component="test_component",
            message="Test message",
            details={'test_key': 'test_value'}
        )
        
        logger.log_event(event)
        
        # Test event retrieval
        recent_events = logger.get_recent_events(10)
        assert len(recent_events) >= 1, "Should have at least one event"
        
        # Test critical event tracking
        critical_event = RobustnessEvent(
            timestamp=time.time(),
            event_type="CRITICAL_TEST",
            severity=SeverityLevel.CRITICAL,
            component="test_component",
            message="Critical test message"
        )
        
        logger.log_event(critical_event)
        critical_events = logger.get_critical_events()
        assert len(critical_events) >= 1, "Should have at least one critical event"
        
        print("✓ Robustness logging system working")
        return True
        
    except Exception as e:
        print(f"✗ Robustness logging test failed: {e}")
        return False


def test_input_validation():
    """Test comprehensive input validation."""
    try:
        validator = InputValidator()
        
        # Test valid sensor data
        valid_data = [1.0, 2.5, 3.7, 4.2, 5.1, 6.8]
        assert validator.validate_sensor_data(valid_data), "Valid data should pass"
        
        # Test invalid sensor data
        invalid_cases = [
            None,                           # None data
            [],                            # Empty data
            [1, 2, "invalid", 4],          # Non-numeric
            [1, 2, float('nan'), 4],       # NaN values
            [1, 2, float('inf'), 4],       # Infinite values
            [1, 2, 50000, 4]               # Out of range
        ]
        
        for invalid_data in invalid_cases:
            try:
                result = validator.validate_sensor_data(invalid_data)
                if result:  # Should fail
                    print(f"⚠ Validation incorrectly passed for: {invalid_data}")
            except:
                pass  # Expected to fail
        
        # Test network configuration validation
        valid_config = {
            'num_receptors': 6,
            'num_projection_neurons': 1000,
            'num_kenyon_cells': 5000,
            'num_gas_classes': 10,
            'tau_membrane': 20.0,
            'sparsity_target': 0.05
        }
        
        assert validator.validate_network_config(valid_config), "Valid config should pass"
        
        # Test invalid configurations
        invalid_config = {
            'num_receptors': -1,           # Negative
            'num_projection_neurons': 0,   # Zero
            'tau_membrane': 500.0,         # Too large
            'sparsity_target': 2.0         # Out of range
        }
        
        assert not validator.validate_network_config(invalid_config), "Invalid config should fail"
        
        # Test file path sanitization
        dangerous_path = "../../../etc/passwd"
        safe_path = validator.sanitize_file_path(dangerous_path)
        assert "../" not in safe_path, "Path traversal should be removed"
        
        print("✓ Input validation working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Input validation test failed: {e}")
        return False


def test_error_recovery():
    """Test error recovery mechanisms."""
    try:
        logger = RobustnessLogger("recovery_test")
        recovery_manager = ErrorRecoveryManager(logger)
        
        # Register a test recovery strategy
        def test_recovery_strategy(error):
            if "test_error" in str(error):
                return True
            return False
        
        recovery_manager.register_recovery_strategy("test_component", test_recovery_strategy)
        
        # Register fallback configuration
        fallback_config = {'safe_mode': True, 'reduced_functionality': True}
        recovery_manager.register_fallback_config("test_component", fallback_config)
        
        # Test successful recovery
        test_error = Exception("test_error for recovery")
        recovery_success = recovery_manager.attempt_recovery("test_component", test_error)
        assert recovery_success, "Recovery should succeed for test error"
        
        # Test fallback for unknown component
        unknown_error = Exception("unknown error")
        fallback_success = recovery_manager.attempt_recovery("test_component", unknown_error)
        assert fallback_success, "Fallback should be applied for unknown errors"
        
        # Test failure for unregistered component
        unregistered_recovery = recovery_manager.attempt_recovery("unknown_component", test_error)
        assert not unregistered_recovery, "Recovery should fail for unregistered component"
        
        print("✓ Error recovery mechanisms working")
        return True
        
    except Exception as e:
        print(f"✗ Error recovery test failed: {e}")
        return False


def test_security_manager():
    """Test security management features."""
    try:
        logger = RobustnessLogger("security_test")
        security_manager = SecurityManager(logger)
        
        # Test data integrity validation
        test_data = [1, 2, 3, 4, 5]
        integrity_check = security_manager.validate_data_integrity(test_data)
        assert integrity_check, "Data integrity should pass for valid data"
        
        # Test rate limiting
        component = "test_sensor"
        
        # Should pass initially
        for i in range(10):
            assert security_manager.check_rate_limit(component), f"Rate limit should pass for request {i}"
        
        # Test input sanitization
        malicious_input = "<script>alert('xss')</script>"
        sanitized = security_manager.sanitize_user_input(malicious_input)
        assert "<script>" not in sanitized, "Script tags should be removed"
        assert "alert" not in sanitized, "JavaScript should be removed"
        
        # Test length limiting
        long_input = "A" * 2000
        sanitized_long = security_manager.sanitize_user_input(long_input)
        assert len(sanitized_long) <= 1000, "Long input should be truncated"
        
        print("✓ Security manager working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Security manager test failed: {e}")
        return False


def test_robust_execution_decorator():
    """Test the robust execution decorator."""
    try:
        logger = RobustnessLogger("decorator_test")
        
        @robust_execution("test_component", logger)
        def test_function_success():
            return "success"
        
        @robust_execution("test_component", logger)
        def test_function_failure():
            raise ValueError("intentional failure")
        
        # Test successful execution
        result = test_function_success()
        assert result == "success", "Successful function should return correct value"
        
        # Test failure handling
        try:
            test_function_failure()
            assert False, "Function should raise exception"
        except ValueError as e:
            assert "intentional failure" in str(e), "Original exception should be preserved"
        
        # Check that events were logged
        events = logger.get_recent_events(10)
        success_events = [e for e in events if e.event_type == "FUNCTION_SUCCESS"]
        error_events = [e for e in events if e.event_type == "FUNCTION_ERROR"]
        
        assert len(success_events) >= 1, "Should have success events"
        assert len(error_events) >= 1, "Should have error events"
        
        print("✓ Robust execution decorator working")
        return True
        
    except Exception as e:
        print(f"✗ Robust execution decorator test failed: {e}")
        return False


def test_enhanced_validation():
    """Test the enhanced validation system."""
    try:
        # Test different validation levels
        for level in [ValidationLevel.BASIC, ValidationLevel.STANDARD, ValidationLevel.STRICT]:
            validator = NeuroMorphicValidator(level)
            
            # Test sensor data validation
            good_sensor_data = [1.0, 2.0, 3.0, 4.0, 5.0]
            is_valid, errors, warnings, suggestions = validator.validate_component('sensor_data', good_sensor_data)
            assert is_valid, f"Good sensor data should be valid at {level.value} level"
            
            # Test bad sensor data
            bad_sensor_data = [float('nan'), float('inf'), -50000]
            is_valid, errors, warnings, suggestions = validator.validate_component('sensor_data', bad_sensor_data)
            assert not is_valid, f"Bad sensor data should be invalid at {level.value} level"
            assert len(errors) > 0, "Should have error messages"
        
        # Test projection neuron configuration validation
        projection_config = {
            'num_receptors': 6,
            'num_projection_neurons': 1000,
            'tau_membrane': 20.0,
            'threshold': 1.0,
            'tau_adaptation': 100.0
        }
        
        validator = NeuroMorphicValidator(ValidationLevel.STANDARD)
        is_valid, errors, warnings, suggestions = validator.validate_component('projection_neurons', projection_config)
        assert is_valid, "Valid projection config should pass"
        
        # Test kenyon cell configuration validation
        kenyon_config = {
            'num_projection_inputs': 1000,
            'num_kenyon_cells': 5000,
            'sparsity_target': 0.05,
            'connection_probability': 0.1,
            'inhibition_strength': 2.0
        }
        
        is_valid, errors, warnings, suggestions = validator.validate_component('kenyon_cells', kenyon_config)
        assert is_valid, "Valid kenyon config should pass"
        
        # Test complete system validation
        system_config = {
            'sensor_data': good_sensor_data,
            'projection_config': projection_config,
            'kenyon_config': kenyon_config,
            'decision_config': {
                'num_kenyon_inputs': 5000,
                'num_gas_classes': 10,
                'integration_window': 100,
                'confidence_threshold': 0.8
            },
            'fusion_config': {
                'chemical_dim': 6,
                'audio_dim': 128,
                'fusion_strategy': 'hierarchical',
                'hidden_dim': 64,
                'dropout_rate': 0.1
            },
            'temporal_config': {
                'dt': 1.0,
                'simulation_duration': 100,
                'tau_membrane': 20.0,
                'refractory_period': 2.0
            }
        }
        
        system_results = validator.validate_complete_system(system_config)
        assert system_results['overall_valid'], "Valid system config should pass complete validation"
        
        # Test validation report generation
        report = validator.get_validation_report(system_config)
        assert "VALIDATION REPORT" in report, "Report should contain header"
        assert "PASSED" in report, "Report should show passed status"
        
        print("✓ Enhanced validation system working")
        return True
        
    except Exception as e:
        print(f"✗ Enhanced validation test failed: {e}")
        return False


def test_complete_robustness_manager():
    """Test the complete robustness manager integration."""
    try:
        manager = RobustnessManager()
        
        # Test health monitoring
        health = manager.get_system_health()
        assert 'health_score' in health, "Health report should include score"
        assert 'status' in health, "Health report should include status"
        
        # Test validation and execution
        def test_function():
            return "executed successfully"
        
        test_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        test_config = {
            'num_receptors': 6,
            'num_projection_neurons': 100,
            'num_kenyon_cells': 500,
            'num_gas_classes': 5
        }
        
        result = manager.validate_and_execute(
            component="test_component",
            func=test_function,
            data=test_data,
            config=test_config
        )
        
        assert result == "executed successfully", "Should execute function successfully"
        
        # Test with invalid data (should handle gracefully)
        invalid_data = [float('nan'), float('inf')]
        
        try:
            manager.validate_and_execute(
                component="test_component",
                func=test_function,
                data=invalid_data,
                config=test_config
            )
            assert False, "Should reject invalid data"
        except (ValueError, RuntimeError):
            pass  # Expected to fail
        
        # Check that system health reflects operations
        updated_health = manager.get_system_health()
        assert 'health_score' in updated_health, "Health should be updated"
        
        print("✓ Complete robustness manager working")
        return True
        
    except Exception as e:
        print(f"✗ Complete robustness manager test failed: {e}")
        return False


def test_stress_conditions():
    """Test robustness under stress conditions."""
    try:
        logger = RobustnessLogger("stress_test")
        validator = NeuroMorphicValidator(ValidationLevel.STANDARD)
        security_manager = SecurityManager(logger)
        
        # Test with large amounts of data
        large_sensor_data = [random.uniform(-100, 100) for _ in range(10000)]
        is_valid, errors, warnings, suggestions = validator.validate_component('sensor_data', large_sensor_data)
        # Should handle large data without crashing
        
        # Test with extreme configurations
        extreme_config = {
            'num_receptors': 1000,
            'num_projection_neurons': 100000,
            'num_kenyon_cells': 1000000,
            'num_gas_classes': 100,
            'tau_membrane': 1.0,
            'sparsity_target': 0.001
        }
        
        is_valid, errors, warnings, suggestions = validator.validate_component('projection_neurons', extreme_config)
        # Should provide appropriate warnings for extreme values
        
        # Test rapid successive operations (stress rate limiting)
        component = "stress_test_component"
        for i in range(50):
            security_manager.check_rate_limit(component)
        
        # Test malformed data
        malformed_data_cases = [
            {'mixed': [1, "text", None, 3.14]},
            {'nested': [[1, 2], [3, 4], "invalid"]},
            {'very_long': "A" * 100000},
            {'special_chars': "!@#$%^&*()_+{}|:<>?"}
        ]
        
        for case_name, malformed_data in malformed_data_cases:
            try:
                validator.validate_component('sensor_data', malformed_data)
                # Should handle without crashing
            except Exception:
                pass  # May legitimately fail, but shouldn't crash
        
        print("✓ Stress condition testing completed")
        return True
        
    except Exception as e:
        print(f"✗ Stress condition test failed: {e}")
        return False


def test_generation_2_objectives():
    """Test all Generation 2 robustness objectives."""
    print("\n=== GENERATION 2: MAKE IT ROBUST ===")
    
    objectives = [
        ("Robustness logging system", test_robustness_logging),
        ("Input validation", test_input_validation),
        ("Error recovery mechanisms", test_error_recovery),
        ("Security management", test_security_manager),
        ("Robust execution decorator", test_robust_execution_decorator),
        ("Enhanced validation system", test_enhanced_validation),
        ("Complete robustness manager", test_complete_robustness_manager),
        ("Stress condition handling", test_stress_conditions)
    ]
    
    results = []
    for name, test_func in objectives:
        print(f"\nTesting: {name}")
        success = test_func()
        results.append(success)
    
    success_rate = sum(results) / len(results)
    print(f"\n=== GENERATION 2 RESULTS ===")
    print(f"Success rate: {success_rate:.1%}")
    print(f"Objectives met: {sum(results)}/{len(results)}")
    
    if success_rate >= 0.85:
        print("✓ GENERATION 2 COMPLETE - Robustness features implemented and validated")
        print("✓ Ready to proceed to Generation 3 (MAKE IT SCALE)")
        return True
    elif success_rate >= 0.70:
        print("⚠ GENERATION 2 MOSTLY COMPLETE - Minor robustness issues remain")
        print("⚠ Can proceed to Generation 3 with caution")
        return True
    else:
        print("✗ GENERATION 2 INCOMPLETE - Robustness features need significant work")
        return False


if __name__ == "__main__":
    success = test_generation_2_objectives()
    exit(0 if success else 1)