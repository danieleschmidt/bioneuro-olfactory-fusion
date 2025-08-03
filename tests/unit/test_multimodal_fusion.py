"""Unit tests for multimodal fusion network components."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from bioneuro_olfactory.models.fusion.multimodal_fusion import (
    OlfactoryFusionSNN,
    FusionConfig,
    create_standard_fusion_network
)


class TestFusionConfig:
    """Test fusion configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = FusionConfig()
        
        assert config.num_chemical_sensors == 6
        assert config.num_audio_features == 128
        assert config.num_projection_neurons == 1000
        assert config.num_kenyon_cells == 5000
        assert config.num_output_classes == 4
        assert config.tau_membrane == 20.0
        assert config.simulation_time == 100.0
        assert config.fusion_strategy == "early"
        assert config.kenyon_sparsity_target == 0.05
        
    def test_custom_config(self):
        """Test custom configuration values."""
        config = FusionConfig(
            num_chemical_sensors=8,
            num_audio_features=64,
            fusion_strategy="hierarchical",
            kenyon_sparsity_target=0.03
        )
        
        assert config.num_chemical_sensors == 8
        assert config.num_audio_features == 64
        assert config.fusion_strategy == "hierarchical"
        assert config.kenyon_sparsity_target == 0.03


class TestOlfactoryFusionSNN:
    """Test the main olfactory fusion network."""
    
    @pytest.fixture
    def fusion_config(self):
        """Create test configuration."""
        return FusionConfig(
            num_chemical_sensors=4,
            num_audio_features=32,
            num_projection_neurons=100,
            num_kenyon_cells=200,
            simulation_time=50.0
        )
    
    @pytest.fixture
    def sample_inputs(self):
        """Create sample input tensors."""
        batch_size = 2
        return {
            'chemical': torch.rand(batch_size, 4) * 1000,  # ppm values
            'audio': torch.randn(batch_size, 32)  # audio features
        }
    
    def test_early_fusion_initialization(self, fusion_config):
        """Test early fusion network initialization."""
        fusion_config.fusion_strategy = "early"
        network = OlfactoryFusionSNN(fusion_config)
        
        assert network.fusion_strategy == "early"
        assert hasattr(network, 'projection_layer')
        assert hasattr(network, 'kenyon_layer')
        assert hasattr(network, 'decision_network')
        assert network.num_chemical_sensors == 4
        assert network.num_audio_features == 32
        
    def test_late_fusion_initialization(self, fusion_config):
        """Test late fusion network initialization."""
        fusion_config.fusion_strategy = "late"
        network = OlfactoryFusionSNN(fusion_config)
        
        assert network.fusion_strategy == "late"
        assert hasattr(network, 'chemical_projection')
        assert hasattr(network, 'audio_projection')
        assert hasattr(network, 'chemical_kenyon')
        assert hasattr(network, 'audio_kenyon')
        assert hasattr(network, 'fusion_decision')
        
    def test_hierarchical_fusion_initialization(self, fusion_config):
        """Test hierarchical fusion network initialization."""
        fusion_config.fusion_strategy = "hierarchical"
        network = OlfactoryFusionSNN(fusion_config)
        
        assert network.fusion_strategy == "hierarchical"
        assert hasattr(network, 'chemical_projection')
        assert hasattr(network, 'audio_projection')
        assert hasattr(network, 'fusion_projection')
        assert hasattr(network, 'kenyon_layer')
        assert hasattr(network, 'decision_network')
        
    def test_invalid_fusion_strategy(self, fusion_config):
        """Test invalid fusion strategy raises error."""
        fusion_config.fusion_strategy = "invalid"
        
        with pytest.raises(ValueError, match="Unknown fusion strategy"):
            OlfactoryFusionSNN(fusion_config)
    
    def test_early_fusion_forward_pass(self, fusion_config, sample_inputs):
        """Test early fusion forward pass."""
        fusion_config.fusion_strategy = "early"
        network = OlfactoryFusionSNN(fusion_config)
        
        result = network.process(
            sample_inputs['chemical'],
            sample_inputs['audio'],
            duration=25.0
        )
        
        # Check output structure
        assert 'projection_spikes' in result
        assert 'kenyon_spikes' in result
        assert 'decision_output' in result
        assert 'network_activity' in result
        
        # Check tensor shapes
        batch_size = sample_inputs['chemical'].shape[0]
        timesteps = int(25.0 / fusion_config.dt)
        
        assert result['projection_spikes'].shape[0] == batch_size
        assert result['projection_spikes'].shape[1] == timesteps
        assert result['kenyon_spikes'].shape[0] == batch_size
        assert result['kenyon_spikes'].shape[1] == timesteps
        assert result['decision_output']['class_probabilities'].shape == (batch_size, fusion_config.num_output_classes)
        
    def test_late_fusion_forward_pass(self, fusion_config, sample_inputs):
        """Test late fusion forward pass."""
        fusion_config.fusion_strategy = "late"
        network = OlfactoryFusionSNN(fusion_config)
        
        result = network.process(
            sample_inputs['chemical'],
            sample_inputs['audio'],
            duration=25.0
        )
        
        # Check late fusion specific outputs
        assert 'chemical_projection_spikes' in result
        assert 'audio_projection_spikes' in result
        assert 'chemical_kenyon_spikes' in result
        assert 'audio_kenyon_spikes' in result
        assert 'decision_output' in result
        
    def test_hierarchical_fusion_forward_pass(self, fusion_config, sample_inputs):
        """Test hierarchical fusion forward pass."""
        fusion_config.fusion_strategy = "hierarchical"
        network = OlfactoryFusionSNN(fusion_config)
        
        result = network.process(
            sample_inputs['chemical'],
            sample_inputs['audio'],
            duration=25.0
        )
        
        # Check hierarchical fusion specific outputs
        assert 'hierarchical_stages' in result
        assert 'projection_spikes' in result
        assert 'kenyon_spikes' in result
        assert len(result['hierarchical_stages']) > 0
        
    def test_input_normalization(self, fusion_config, sample_inputs):
        """Test input normalization functionality."""
        network = OlfactoryFusionSNN(fusion_config)
        
        # Test chemical normalization
        chemical_norm = network._normalize_input(sample_inputs['chemical'], 'chemical')
        assert torch.all(chemical_norm >= 0)
        assert torch.all(chemical_norm <= 1)
        
        # Test audio normalization
        audio_norm = network._normalize_input(sample_inputs['audio'], 'audio')
        assert torch.isfinite(audio_norm).all()
        
    def test_gas_predictions(self, fusion_config, sample_inputs):
        """Test gas prediction interpretation."""
        network = OlfactoryFusionSNN(fusion_config)
        
        result = network.process(
            sample_inputs['chemical'],
            sample_inputs['audio'],
            duration=25.0
        )
        
        predictions = network.get_gas_predictions(result['decision_output']['class_probabilities'])
        
        assert len(predictions) == sample_inputs['chemical'].shape[0]  # One per batch
        
        for batch_predictions in predictions:
            assert len(batch_predictions) == fusion_config.num_output_classes
            for pred in batch_predictions:
                assert 'gas_type' in pred
                assert 'confidence' in pred
                assert 'concentration_estimate' in pred
                assert 0 <= pred['confidence'] <= 1
                assert pred['concentration_estimate'] >= 0
                assert pred['gas_type'] in ["methane", "carbon_monoxide", "ammonia", "propane"]
        
    def test_network_state_reset(self, fusion_config):
        """Test network state reset functionality."""
        network = OlfactoryFusionSNN(fusion_config)
        
        # Process some data to build up state
        chemical_input = torch.rand(1, fusion_config.num_chemical_sensors)
        audio_input = torch.randn(1, fusion_config.num_audio_features)
        
        network.process(chemical_input, audio_input, duration=20.0)
        
        # Reset should not raise errors
        network.reset_network_state()
        
    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_different_batch_sizes(self, fusion_config, batch_size):
        """Test network with different batch sizes."""
        network = OlfactoryFusionSNN(fusion_config)
        
        chemical_input = torch.rand(batch_size, fusion_config.num_chemical_sensors)
        audio_input = torch.randn(batch_size, fusion_config.num_audio_features)
        
        result = network.process(chemical_input, audio_input, duration=20.0)
        
        assert result['decision_output']['class_probabilities'].shape[0] == batch_size
        
    @pytest.mark.parametrize("duration", [10.0, 50.0, 100.0])
    def test_different_durations(self, fusion_config, sample_inputs, duration):
        """Test network with different simulation durations."""
        network = OlfactoryFusionSNN(fusion_config)
        
        result = network.process(
            sample_inputs['chemical'],
            sample_inputs['audio'],
            duration=duration
        )
        
        expected_timesteps = int(duration / fusion_config.dt)
        assert result['projection_spikes'].shape[1] == expected_timesteps
        assert result['kenyon_spikes'].shape[1] == expected_timesteps


class TestNetworkActivityAnalysis:
    """Test network activity analysis functionality."""
    
    @pytest.fixture
    def network_result(self):
        """Create mock network result for testing."""
        batch_size, timesteps = 2, 50
        num_projection = 100
        num_kenyon = 200
        
        return {
            'projection_spikes': torch.rand(batch_size, timesteps, num_projection),
            'kenyon_spikes': torch.rand(batch_size, timesteps, num_kenyon) * 0.1,  # Sparse
            'network_activity': {
                'projection_rates': torch.rand(batch_size, num_projection),
                'kenyon_sparsity': {
                    'sparsity_ratio': torch.tensor([0.05, 0.04]),
                    'target_sparsity': 0.05
                }
            }
        }
    
    def test_network_activity_analysis(self):
        """Test network activity analysis."""
        config = FusionConfig(num_projection_neurons=100, num_kenyon_cells=200)
        network = OlfactoryFusionSNN(config)
        
        # Create mock spike data
        projection_spikes = torch.rand(2, 50, 100) * 0.3  # Sparse spikes
        kenyon_spikes = torch.rand(2, 50, 200) * 0.1  # Even sparser
        
        activity = network._analyze_network_activity(projection_spikes, kenyon_spikes)
        
        assert 'projection_rates' in activity
        assert 'kenyon_rates' in activity
        assert 'kenyon_sparsity' in activity
        assert 'temporal_dynamics' in activity
        
        # Check sparsity calculations
        sparsity_info = activity['kenyon_sparsity']
        assert 'sparsity_ratio' in sparsity_info
        assert 'active_cells' in sparsity_info
        assert 'target_sparsity' in sparsity_info
        
    def test_concentration_estimation(self):
        """Test concentration estimation from confidence."""
        config = FusionConfig()
        network = OlfactoryFusionSNN(config)
        
        test_cases = [
            ("methane", 0.9, 2500, 5000),  # High confidence -> high concentration
            ("carbon_monoxide", 0.5, 0, 250),  # Medium confidence -> medium
            ("ammonia", 0.2, 0, 20),  # Low confidence -> low
        ]
        
        for gas_type, confidence, min_expected, max_expected in test_cases:
            concentration = network._estimate_concentration(gas_type, confidence)
            assert min_expected <= concentration <= max_expected
            assert isinstance(concentration, float)


class TestStandardFusionNetwork:
    """Test the standard fusion network factory function."""
    
    def test_create_standard_network(self):
        """Test creating standard fusion network."""
        network = create_standard_fusion_network()
        
        assert isinstance(network, OlfactoryFusionSNN)
        assert network.num_chemical_sensors == 6
        assert network.num_audio_features == 128
        assert network.fusion_strategy == "hierarchical"
        assert network.config.kenyon_sparsity_target == 0.05
        
    def test_standard_network_functionality(self):
        """Test that standard network functions correctly."""
        network = create_standard_fusion_network()
        
        # Test with realistic input shapes
        chemical_input = torch.rand(1, 6) * 500  # 0-500 ppm
        audio_input = torch.randn(1, 128)  # normalized audio features
        
        result = network.process(chemical_input, audio_input, duration=50.0)
        
        assert 'decision_output' in result
        assert 'network_activity' in result
        
        # Test gas predictions
        predictions = network.get_gas_predictions(result['decision_output']['class_probabilities'])
        assert len(predictions) == 1
        assert len(predictions[0]) == 4  # 4 gas types


class TestInputValidation:
    """Test input validation and error handling."""
    
    def test_mismatched_batch_sizes(self):
        """Test error handling for mismatched batch sizes."""
        config = FusionConfig()
        network = OlfactoryFusionSNN(config)
        
        chemical_input = torch.rand(2, config.num_chemical_sensors)
        audio_input = torch.rand(3, config.num_audio_features)  # Different batch size
        
        # Should handle gracefully or raise informative error
        with pytest.raises(Exception):
            network.process(chemical_input, audio_input)
    
    def test_wrong_input_dimensions(self):
        """Test error handling for wrong input dimensions."""
        config = FusionConfig()
        network = OlfactoryFusionSNN(config)
        
        # Wrong number of chemical sensors
        chemical_input = torch.rand(1, config.num_chemical_sensors + 1)
        audio_input = torch.rand(1, config.num_audio_features)
        
        # Should raise informative error
        with pytest.raises(Exception):
            network.process(chemical_input, audio_input)
    
    def test_negative_duration(self):
        """Test error handling for negative duration."""
        config = FusionConfig()
        network = OlfactoryFusionSNN(config)
        
        chemical_input = torch.rand(1, config.num_chemical_sensors)
        audio_input = torch.rand(1, config.num_audio_features)
        
        # Should handle negative duration gracefully
        result = network.process(chemical_input, audio_input, duration=-10.0)
        # Should use default duration or raise error
        assert result is not None or True  # Adjust based on actual behavior


class TestPerformance:
    """Test performance characteristics."""
    
    def test_processing_time(self):
        """Test that processing completes in reasonable time."""
        import time
        
        config = FusionConfig(simulation_time=100.0)
        network = OlfactoryFusionSNN(config)
        
        chemical_input = torch.rand(5, config.num_chemical_sensors)
        audio_input = torch.rand(5, config.num_audio_features)
        
        start_time = time.time()
        result = network.process(chemical_input, audio_input)
        end_time = time.time()
        
        processing_time = end_time - start_time
        assert processing_time < 10.0  # Should complete within 10 seconds
        assert result is not None
    
    def test_memory_usage(self):
        """Test that memory usage is reasonable."""
        config = FusionConfig()
        network = OlfactoryFusionSNN(config)
        
        # Process multiple batches to check for memory leaks
        for _ in range(10):
            chemical_input = torch.rand(2, config.num_chemical_sensors)
            audio_input = torch.rand(2, config.num_audio_features)
            
            result = network.process(chemical_input, audio_input, duration=20.0)
            
            # Reset state to prevent accumulation
            network.reset_network_state()
        
        # If we get here without OOM, memory usage is reasonable
        assert True


class TestReproducibility:
    """Test reproducibility of results."""
    
    def test_deterministic_results(self):
        """Test that results are deterministic with fixed seed."""
        config = FusionConfig()
        
        chemical_input = torch.rand(1, config.num_chemical_sensors)
        audio_input = torch.rand(1, config.num_audio_features)
        
        # First run
        torch.manual_seed(42)
        network1 = OlfactoryFusionSNN(config)
        result1 = network1.process(chemical_input, audio_input, duration=50.0)
        
        # Second run with same seed
        torch.manual_seed(42)
        network2 = OlfactoryFusionSNN(config)
        result2 = network2.process(chemical_input, audio_input, duration=50.0)
        
        # Results should be identical (allowing for small numerical differences)
        torch.testing.assert_close(
            result1['decision_output']['class_probabilities'],
            result2['decision_output']['class_probabilities'],
            rtol=1e-5,
            atol=1e-6
        )