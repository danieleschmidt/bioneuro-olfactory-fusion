"""Tests for comprehensive input validation and error handling."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from bioneuro_olfactory.core.validation import (
    InputValidator, ValidationError, ValidationLevel, ValidationResult,
    RobustnessChecker, validate_and_correct, default_validator
)


class TestInputValidator:
    """Test input validation functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.validator = InputValidator(ValidationLevel.STRICT)
        self.warn_validator = InputValidator(ValidationLevel.WARN)
        self.silent_validator = InputValidator(ValidationLevel.SILENT)
        
    def test_validate_tensor_valid(self):
        """Test validation of valid tensor."""
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        result = self.validator.validate_tensor(
            tensor, "test_tensor",
            expected_shape=(2, 2),
            expected_dtype=torch.float32,
            min_value=0.0,
            max_value=5.0
        )
        
        assert result.is_valid
        assert "validation passed" in result.message
        assert result.corrected_value is None
        
    def test_validate_tensor_wrong_shape(self):
        """Test validation of tensor with wrong shape."""
        tensor = torch.tensor([1.0, 2.0], dtype=torch.float32)
        
        with pytest.raises(ValidationError):
            self.validator.validate_tensor(
                tensor, "test_tensor",
                expected_shape=(2, 2)
            )
            
    def test_validate_tensor_wrong_dtype_strict(self):
        """Test validation of tensor with wrong dtype in strict mode."""
        tensor = torch.tensor([1, 2], dtype=torch.int32)
        
        with pytest.raises(ValidationError):
            self.validator.validate_tensor(
                tensor, "test_tensor",
                expected_dtype=torch.float32
            )
            
    def test_validate_tensor_wrong_dtype_warn(self):
        """Test validation of tensor with wrong dtype in warn mode."""
        tensor = torch.tensor([1, 2], dtype=torch.int32)
        
        result = self.warn_validator.validate_tensor(
            tensor, "test_tensor",
            expected_dtype=torch.float32
        )
        
        assert result.is_valid
        assert result.corrected_value is not None
        assert result.corrected_value.dtype == torch.float32
        
    def test_validate_tensor_nan_values(self):
        """Test validation of tensor with NaN values."""
        tensor = torch.tensor([1.0, float('nan'), 3.0])
        
        with pytest.raises(ValidationError):
            self.validator.validate_tensor(tensor, "test_tensor", allow_nan=False)
            
    def test_validate_tensor_inf_values(self):
        """Test validation of tensor with infinite values."""
        tensor = torch.tensor([1.0, float('inf'), 3.0])
        
        with pytest.raises(ValidationError):
            self.validator.validate_tensor(tensor, "test_tensor", allow_inf=False)
            
    def test_validate_tensor_value_range_strict(self):
        """Test validation of tensor values outside range in strict mode."""
        tensor = torch.tensor([1.0, 2.0, 10.0])  # 10.0 > max_value
        
        with pytest.raises(ValidationError):
            self.validator.validate_tensor(
                tensor, "test_tensor",
                min_value=0.0,
                max_value=5.0
            )
            
    def test_validate_tensor_value_range_warn(self):
        """Test validation of tensor values outside range in warn mode."""
        tensor = torch.tensor([1.0, 2.0, 10.0])  # 10.0 > max_value
        
        result = self.warn_validator.validate_tensor(
            tensor, "test_tensor",
            min_value=0.0,
            max_value=5.0
        )
        
        assert result.is_valid
        assert result.corrected_value is not None
        assert result.corrected_value.max().item() <= 5.0
        
    def test_validate_spike_train_valid(self):
        """Test validation of valid spike train."""
        spike_train = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]], dtype=torch.float32)
        
        result = self.validator.validate_spike_train(spike_train)
        
        assert result.is_valid
        assert "validation passed" in result.message
        
    def test_validate_spike_train_non_binary(self):
        """Test validation of non-binary spike train."""
        spike_train = torch.tensor([[0.0, 0.5, 1.0], [0.3, 0.7, 1.0]], dtype=torch.float32)
        
        # Strict mode should fail
        with pytest.raises(ValidationError):
            self.validator.validate_spike_train(spike_train)
            
        # Warn mode should correct
        result = self.warn_validator.validate_spike_train(spike_train)
        assert result.is_valid
        assert result.corrected_value is not None
        
    def test_validate_concentration_valid(self):
        """Test validation of valid concentration values."""
        concentration = torch.tensor([100.0, 500.0, 1000.0])
        
        result = self.validator.validate_concentration(concentration)
        
        assert result.is_valid
        
    def test_validate_concentration_negative(self):
        """Test validation of negative concentration values."""
        concentration = torch.tensor([-10.0, 100.0])
        
        with pytest.raises(ValidationError):
            self.validator.validate_concentration(concentration)
            
    def test_validate_neuron_parameters_valid(self):
        """Test validation of valid neuron parameters."""
        results = self.validator.validate_neuron_parameters(
            tau_membrane=20.0,
            threshold=1.0,
            dt=1.0
        )
        
        assert all(r.is_valid for r in results)
        
    def test_validate_neuron_parameters_invalid(self):
        """Test validation of invalid neuron parameters."""
        # tau_membrane <= 0 should fail
        with pytest.raises(ValidationError):
            self.validator.validate_neuron_parameters(
                tau_membrane=-5.0,
                threshold=1.0,
                dt=1.0
            )
            
        # dt > tau_membrane should warn
        results = self.warn_validator.validate_neuron_parameters(
            tau_membrane=10.0,
            threshold=1.0,
            dt=20.0  # dt > tau_membrane
        )
        
        # Should have warnings
        assert any(r.warning_issued for r in results)
        
    def test_validate_sensor_configuration_valid(self):
        """Test validation of valid sensor configuration."""
        config = {
            'name': 'MQ2_sensor',
            'type': 'MQ2',
            'target_gases': ['methane', 'propane'],
            'range': [200, 10000],
            'response_time': 30.0,
            'sensitivity': 1.0
        }
        
        results = self.validator.validate_sensor_configuration(config)
        
        # Should have validation results for each field
        assert len(results) > 0
        # Most should be valid
        valid_results = [r for r in results if r.is_valid]
        assert len(valid_results) > 0
        
    def test_validate_sensor_configuration_missing_fields(self):
        """Test validation of sensor config with missing required fields."""
        config = {
            'name': 'MQ2_sensor'
            # Missing required fields
        }
        
        with pytest.raises(ValidationError):
            self.validator.validate_sensor_configuration(config)
            
    def test_validate_fusion_inputs_valid(self):
        """Test validation of valid fusion inputs."""
        chemical_features = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        audio_features = torch.tensor([[0.5, 1.5], [2.5, 3.5]], dtype=torch.float32)
        
        results = self.validator.validate_fusion_inputs(chemical_features, audio_features)
        
        # Should validate both inputs plus batch size consistency
        assert len(results) >= 3
        
    def test_validate_fusion_inputs_batch_mismatch(self):
        """Test validation of fusion inputs with batch size mismatch."""
        chemical_features = torch.tensor([[1.0, 2.0]], dtype=torch.float32)  # Batch size 1
        audio_features = torch.tensor([[0.5, 1.5], [2.5, 3.5]], dtype=torch.float32)  # Batch size 2
        
        with pytest.raises(ValidationError):
            self.validator.validate_fusion_inputs(chemical_features, audio_features)
            
    def test_validation_history(self):
        """Test validation history tracking."""
        tensor = torch.tensor([1.0, 2.0], dtype=torch.float32)
        
        # Clear history
        self.validator.clear_history()
        
        # Perform some validations
        self.validator.validate_tensor(tensor, "test1")
        
        try:
            self.validator.validate_tensor(tensor, "test2", min_value=5.0)  # Should fail
        except ValidationError:
            pass
            
        # Check history
        summary = self.validator.get_validation_summary()
        assert summary['total_validations'] >= 2
        assert summary['failed_validations'] >= 1
        

class TestRobustnessChecker:
    """Test robustness checking functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.checker = RobustnessChecker()
        
    def test_check_numerical_stability_stable(self):
        """Test numerical stability check on stable tensor."""
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        
        result = self.checker.check_numerical_stability(tensor, "stable_tensor")
        
        assert result['is_stable']
        assert len(result['issues']) == 0
        assert 'statistics' in result
        
    def test_check_numerical_stability_nan(self):
        """Test numerical stability check with NaN values."""
        tensor = torch.tensor([[1.0, float('nan')], [3.0, 4.0]], dtype=torch.float32)
        
        result = self.checker.check_numerical_stability(tensor, "nan_tensor")
        
        assert not result['is_stable']
        assert any('NaN' in issue for issue in result['issues'])
        
    def test_check_numerical_stability_inf(self):
        """Test numerical stability check with infinite values."""
        tensor = torch.tensor([[1.0, float('inf')], [3.0, 4.0]], dtype=torch.float32)
        
        result = self.checker.check_numerical_stability(tensor, "inf_tensor")
        
        assert not result['is_stable']
        assert any('Infinite' in issue for issue in result['issues'])
        
    def test_check_spike_train_validity_valid(self):
        """Test spike train validity check on valid train."""
        # Valid spike train: 3D tensor with reasonable firing rates
        spike_train = torch.zeros(2, 100, 1000)  # batch, neurons, time
        # Add some spikes (5% sparsity)
        spike_indices = torch.randperm(spike_train.numel())[:int(0.05 * spike_train.numel())]
        spike_train.view(-1)[spike_indices] = 1.0
        
        result = self.checker.check_spike_train_validity(spike_train, name="valid_spikes")
        
        assert result['is_valid']
        assert 'statistics' in result
        
    def test_check_spike_train_validity_wrong_shape(self):
        """Test spike train validity check on wrong shape."""
        spike_train = torch.zeros(100, 1000)  # 2D instead of 3D
        
        result = self.checker.check_spike_train_validity(spike_train, name="wrong_shape")
        
        assert not result['is_valid']
        assert any('3D tensor' in issue for issue in result['issues'])
        
    def test_check_spike_train_validity_silent_neurons(self):
        """Test spike train validity check with too many silent neurons."""
        spike_train = torch.zeros(1, 100, 1000)  # All silent
        
        result = self.checker.check_spike_train_validity(spike_train, name="silent_neurons")
        
        assert not result['is_valid']
        assert any('silent neurons' in issue for issue in result['issues'])
        
    def test_robustness_summary(self):
        """Test robustness summary generation."""
        # Perform some checks
        stable_tensor = torch.tensor([1.0, 2.0, 3.0])
        unstable_tensor = torch.tensor([1.0, float('nan'), 3.0])
        
        self.checker.check_numerical_stability(stable_tensor, "stable")
        self.checker.check_numerical_stability(unstable_tensor, "unstable")
        
        summary = self.checker.get_robustness_summary()
        
        assert 'total_checks' in summary
        assert 'stable_checks' in summary
        assert 'stability_rate' in summary
        assert summary['total_checks'] >= 2
        

class TestValidateAndCorrect:
    """Test the validate_and_correct convenience function."""
    
    def test_validate_and_correct_valid(self):
        """Test validate_and_correct with valid tensor."""
        tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        
        result = validate_and_correct(tensor, "test_tensor")
        
        assert torch.equal(result, tensor)
        
    def test_validate_and_correct_with_correction(self):
        """Test validate_and_correct with tensor needing correction."""
        tensor = torch.tensor([1, 2, 3], dtype=torch.int32)  # Wrong dtype
        
        # Should return corrected tensor
        result = validate_and_correct(
            tensor, "test_tensor",
            expected_dtype=torch.float32
        )
        
        assert result.dtype == torch.float32
        
    def test_validate_and_correct_extreme_values(self):
        """Test validate_and_correct with extreme values."""
        tensor = torch.tensor([1.0, 100.0, 3.0], dtype=torch.float32)
        
        result = validate_and_correct(
            tensor, "test_tensor",
            max_value=10.0
        )
        
        # Should be clamped
        assert result.max().item() <= 10.0
        

class TestValidationLevels:
    """Test different validation levels."""
    
    def test_strict_level_raises_exceptions(self):
        """Test that STRICT level raises exceptions on errors."""
        validator = InputValidator(ValidationLevel.STRICT)
        invalid_tensor = torch.tensor([float('nan')])
        
        with pytest.raises(ValidationError):
            validator.validate_tensor(invalid_tensor, "test", allow_nan=False)
            
    def test_warn_level_issues_warnings(self):
        """Test that WARN level issues warnings but continues."""
        validator = InputValidator(ValidationLevel.WARN)
        invalid_tensor = torch.tensor([float('nan')])
        
        with patch('logging.Logger.warning') as mock_warning:
            result = validator.validate_tensor(invalid_tensor, "test", allow_nan=False)
            
            # Should not raise exception but should warn
            assert not result.is_valid
            mock_warning.assert_called()
            
    def test_silent_level_no_warnings(self):
        """Test that SILENT level doesn't issue warnings."""
        validator = InputValidator(ValidationLevel.SILENT)
        invalid_tensor = torch.tensor([float('nan')])
        
        with patch('logging.Logger.warning') as mock_warning:
            result = validator.validate_tensor(invalid_tensor, "test", allow_nan=False)
            
            # Should not raise exception or warn
            assert not result.is_valid
            mock_warning.assert_not_called()
            

class TestValidationIntegration:
    """Integration tests for validation system."""
    
    def test_end_to_end_sensor_data_validation(self):
        """Test end-to-end validation of sensor data."""
        validator = InputValidator(ValidationLevel.WARN)
        
        # Simulate sensor data with some issues
        sensor_data = {
            'MQ2_methane': 5000.0,      # Valid
            'MQ7_CO': -10.0,            # Negative (invalid)
            'temperature': 25.5,         # Valid
            'humidity': 120.0,          # Too high (>100%)
            'invalid_key': 'some_value'  # Unknown sensor
        }
        
        sanitized_data, threats = validator.input_sanitizer.validate_sensor_data(sensor_data)
        
        # Should detect threats
        assert len(threats) > 0
        assert any('suspicious_sensor_value' in threat for threat in threats)
        
        # Should sanitize data
        assert sanitized_data['MQ7_CO'] >= -1000  # Clamped to valid range
        
    def test_multi_modal_fusion_validation(self):
        """Test validation of multi-modal fusion inputs."""
        validator = InputValidator(ValidationLevel.STRICT)
        
        # Valid inputs
        chemical_features = torch.randn(32, 6, dtype=torch.float32)  # Batch size 32
        audio_features = torch.randn(32, 128, dtype=torch.float32)   # Batch size 32
        
        results = validator.validate_fusion_inputs(chemical_features, audio_features)
        
        # Should pass all validations
        assert all(r.is_valid for r in results)
        
    def test_robustness_checking_pipeline(self):
        """Test complete robustness checking pipeline."""
        checker = RobustnessChecker()
        
        # Simulate neural network outputs
        projection_spikes = torch.rand(16, 1000, 100) > 0.95  # Sparse spikes
        kenyon_spikes = torch.rand(16, 5000, 100) > 0.98     # Very sparse
        
        # Check spike trains
        pn_result = checker.check_spike_train_validity(
            projection_spikes.float(), 
            name="projection_neurons"
        )
        kc_result = checker.check_spike_train_validity(
            kenyon_spikes.float(),
            name="kenyon_cells"
        )
        
        # Both should be valid (sparse but reasonable)
        assert pn_result['is_valid']
        assert kc_result['is_valid']
        
        # Check sparsity levels
        pn_sparsity = pn_result['statistics']['sparsity']
        kc_sparsity = kc_result['statistics']['sparsity']
        
        # Kenyon cells should be sparser than projection neurons
        assert kc_sparsity > pn_sparsity
        
        # Get summary
        summary = checker.get_robustness_summary()
        assert summary['stability_rate'] == 1.0  # All checks should pass


if __name__ == "__main__":
    pytest.main([__file__])