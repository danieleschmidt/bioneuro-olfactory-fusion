"""Unit tests for spike encoding implementations."""

import pytest
import torch
import numpy as np

from bioneuro_olfactory.core.encoding.spike_encoding import (
    RateEncoder,
    TemporalEncoder,
    PhaseEncoder,
    BurstEncoder,
    PopulationEncoder,
    AdaptiveEncoder
)


class TestRateEncoder:
    """Test rate-based spike encoding."""
    
    def test_initialization(self):
        """Test encoder initialization."""
        encoder = RateEncoder(max_rate=100.0, dt=2.0)
        
        assert encoder.max_rate == 100.0
        assert encoder.dt == 2.0
        
    def test_basic_encoding(self):
        """Test basic rate encoding functionality."""
        encoder = RateEncoder(max_rate=200.0, dt=1.0)
        
        # Test data: batch_size=2, num_channels=3
        data = torch.tensor([[0.0, 0.5, 1.0], [0.2, 0.8, 0.4]])
        duration = 100
        
        spikes = encoder.encode(data, duration)
        
        assert spikes.shape == (2, 3, duration)
        assert torch.all((spikes >= 0) & (spikes <= 1))  # Binary spikes
        
    def test_rate_proportionality(self):
        """Test that spike rate is proportional to input intensity."""
        encoder = RateEncoder(max_rate=100.0, dt=1.0)
        
        # Different input intensities
        low_input = torch.tensor([[0.1]])
        high_input = torch.tensor([[0.9]])
        duration = 1000
        
        low_spikes = encoder.encode(low_input, duration)
        high_spikes = encoder.encode(high_input, duration)
        
        low_rate = low_spikes.sum().item() / duration * 1000
        high_rate = high_spikes.sum().item() / duration * 1000
        
        # Higher input should produce higher rate
        assert high_rate > low_rate
        
    def test_zero_input(self):
        """Test encoding with zero input."""
        encoder = RateEncoder()
        
        zero_data = torch.zeros(1, 3)
        spikes = encoder.encode(zero_data, 100)
        
        # Zero input should produce very few or no spikes
        assert spikes.sum().item() < 5  # Allow for some noise
        
    def test_saturated_input(self):
        """Test encoding with saturated input."""
        encoder = RateEncoder(max_rate=100.0, dt=1.0)
        
        # Input above 1.0 (should be clamped)
        saturated_data = torch.tensor([[1.5, 2.0]])
        duration = 100
        
        spikes = encoder.encode(saturated_data, duration)
        
        # Should handle gracefully
        assert spikes.shape == (1, 2, duration)
        
    def test_negative_input(self):
        """Test encoding with negative input."""
        encoder = RateEncoder()
        
        negative_data = torch.tensor([[-0.5, -1.0]])
        spikes = encoder.encode(negative_data, 100)
        
        # Negative inputs should be clamped to 0
        assert spikes.sum().item() < 5


class TestTemporalEncoder:
    """Test temporal spike encoding."""
    
    def test_time_to_first_spike(self):
        """Test that spike timing reflects input intensity."""
        encoder = TemporalEncoder(precision=1.0, max_delay=50)
        
        # High value should spike early, low value should spike late
        data = torch.tensor([[0.9, 0.1]])
        duration = 60
        
        spikes = encoder.encode(data, duration)
        
        # Find spike times
        spike_times = []
        for channel in range(2):
            spike_indices = torch.where(spikes[0, channel, :] > 0)[0]
            if len(spike_indices) > 0:
                spike_times.append(spike_indices[0].item())
            else:
                spike_times.append(duration)  # No spike
                
        if len(spike_times) == 2:
            assert spike_times[0] < spike_times[1]  # High value spikes first
            
    def test_precision_parameter(self):
        """Test that precision parameter affects encoding."""
        encoder_precise = TemporalEncoder(precision=0.5, max_delay=50)
        encoder_coarse = TemporalEncoder(precision=2.0, max_delay=50)
        
        data = torch.tensor([[0.5]])
        duration = 60
        
        spikes_precise = encoder_precise.encode(data, duration)
        spikes_coarse = encoder_coarse.encode(data, duration)
        
        # Both should produce spikes but possibly at different times
        assert spikes_precise.sum() > 0
        assert spikes_coarse.sum() > 0
        
    def test_no_spike_for_zero(self):
        """Test that zero input produces no spikes."""
        encoder = TemporalEncoder()
        
        zero_data = torch.zeros(1, 3)
        spikes = encoder.encode(zero_data, 100)
        
        assert spikes.sum().item() == 0


class TestPhaseEncoder:
    """Test phase-based spike encoding."""
    
    def test_carrier_frequency(self):
        """Test encoding with different carrier frequencies."""
        encoder = PhaseEncoder(carrier_freq=10.0, dt=1.0)
        
        data = torch.tensor([[0.5]])
        duration = 100  # 100ms at 1ms resolution
        
        spikes = encoder.encode(data, duration)
        
        assert spikes.shape == (1, 1, duration)
        
        # Should have periodic structure related to carrier frequency
        spike_rate = spikes.sum().item() / duration * 1000  # Hz
        assert 5 < spike_rate < 15  # Roughly around carrier frequency
        
    def test_phase_modulation(self):
        """Test that different inputs produce different phase relationships."""
        encoder = PhaseEncoder(carrier_freq=20.0)
        
        data1 = torch.tensor([[0.0]])
        data2 = torch.tensor([[1.0]])
        duration = 100
        
        spikes1 = encoder.encode(data1, duration)
        spikes2 = encoder.encode(data2, duration)
        
        # Different phases should produce different spike patterns
        assert not torch.equal(spikes1, spikes2)


class TestBurstEncoder:
    """Test burst-based spike encoding."""
    
    def test_burst_structure(self):
        """Test that encoding produces burst patterns."""
        encoder = BurstEncoder(max_burst_size=5, inter_burst_interval=10)
        
        data = torch.tensor([[0.8]])  # Should produce long bursts
        duration = 100
        
        spikes = encoder.encode(data, duration)
        spike_train = spikes[0, 0, :].numpy()
        
        # Find burst structure
        burst_starts = []
        in_burst = False
        
        for i, spike in enumerate(spike_train):
            if spike > 0 and not in_burst:
                burst_starts.append(i)
                in_burst = True
            elif spike == 0 and in_burst:
                in_burst = False
                
        # Should have multiple bursts
        assert len(burst_starts) >= 2
        
    def test_burst_size_proportional(self):
        """Test that burst size is proportional to input."""
        encoder = BurstEncoder(max_burst_size=10, inter_burst_interval=20)
        
        low_data = torch.tensor([[0.2]])
        high_data = torch.tensor([[0.8]])
        duration = 100
        
        low_spikes = encoder.encode(low_data, duration)
        high_spikes = encoder.encode(high_data, duration)
        
        # Higher input should produce more spikes overall
        assert high_spikes.sum() >= low_spikes.sum()


class TestPopulationEncoder:
    """Test population vector encoding."""
    
    def test_population_structure(self):
        """Test that encoding distributes across population."""
        encoder = PopulationEncoder(num_neurons=10, sigma=0.2)
        
        data = torch.tensor([[0.5]])
        duration = 50
        
        spikes = encoder.encode(data, duration)
        
        # Shape should include population dimension
        assert spikes.shape == (1, 1, 10, duration)
        
    def test_tuning_curves(self):
        """Test that different inputs activate different neurons."""
        encoder = PopulationEncoder(num_neurons=10, sigma=0.3)
        
        data1 = torch.tensor([[0.2]])
        data2 = torch.tensor([[0.8]])
        duration = 50
        
        spikes1 = encoder.encode(data1, duration)
        spikes2 = encoder.encode(data2, duration)
        
        # Population activity should be different
        pop_activity1 = spikes1.sum(dim=-1)  # Sum over time
        pop_activity2 = spikes2.sum(dim=-1)
        
        assert not torch.equal(pop_activity1, pop_activity2)
        
    def test_preferred_values(self):
        """Test that neurons have preferred input values."""
        encoder = PopulationEncoder(num_neurons=5, sigma=0.2)
        
        # Test with values that should activate different neurons
        test_values = torch.linspace(0, 1, 5).unsqueeze(0)  # [1, 5]
        duration = 100
        
        all_responses = []
        for i in range(5):
            data = test_values[:, i:i+1]  # Single value
            spikes = encoder.encode(data, duration)
            response = spikes.sum(dim=-1)  # [1, 1, 5]
            all_responses.append(response[0, 0, :])  # [5]
            
        # Each input should maximally activate a different neuron
        all_responses = torch.stack(all_responses)  # [5, 5]
        
        # Check that diagonal elements are among the largest
        for i in range(5):
            neuron_responses = all_responses[:, i]
            max_response_idx = torch.argmax(neuron_responses).item()
            # Allow some tolerance for overlapping tuning curves
            assert abs(max_response_idx - i) <= 1


class TestAdaptiveEncoder:
    """Test adaptive spike encoder."""
    
    def test_adaptation_statistics(self):
        """Test that encoder adapts to input statistics."""
        base_encoder = RateEncoder(max_rate=100.0)
        encoder = AdaptiveEncoder(
            base_encoder=base_encoder,
            adaptation_rate=0.1
        )
        
        # First batch - high values
        high_data = torch.ones(1, 3) * 0.8
        duration = 50
        
        spikes1 = encoder.encode(high_data, duration)
        stats1 = (encoder.running_mean, encoder.running_std)
        
        # Second batch - low values
        low_data = torch.ones(1, 3) * 0.2
        spikes2 = encoder.encode(low_data, duration)
        stats2 = (encoder.running_mean, encoder.running_std)
        
        # Statistics should have changed
        assert stats1[0] != stats2[0]  # Mean changed
        
    def test_normalization_effect(self):
        """Test that adaptation normalizes different input ranges."""
        base_encoder = RateEncoder(max_rate=100.0)
        encoder = AdaptiveEncoder(base_encoder, adaptation_rate=0.05)
        
        # Train with high values
        for _ in range(10):
            high_data = torch.rand(1, 3) * 10.0  # High range
            encoder.encode(high_data, 50)
            
        # Test with new high value
        test_data = torch.tensor([[8.0, 9.0, 10.0]])
        adapted_spikes = encoder.encode(test_data, 100)
        
        # Reset and test without adaptation
        encoder.reset_adaptation()
        non_adapted_spikes = encoder.encode(test_data, 100)
        
        # Adaptation should change the encoding
        assert not torch.equal(adapted_spikes, non_adapted_spikes)
        
    def test_reset_adaptation(self):
        """Test adaptation reset functionality."""
        base_encoder = RateEncoder()
        encoder = AdaptiveEncoder(base_encoder)
        
        # Build up adaptation
        data = torch.rand(5, 3) * 2.0
        for i in range(5):
            encoder.encode(data[i:i+1], 50)
            
        # Check adaptation exists
        assert encoder.running_mean != 0.0
        assert encoder.update_count > 0
        
        # Reset
        encoder.reset_adaptation()
        
        assert encoder.running_mean == 0.0
        assert encoder.running_std == 1.0
        assert encoder.update_count == 0


class TestEncoderIntegration:
    """Integration tests across encoders."""
    
    @pytest.mark.parametrize("encoder_class,kwargs", [
        (RateEncoder, {"max_rate": 50.0}),
        (TemporalEncoder, {"max_delay": 30}),
        (PhaseEncoder, {"carrier_freq": 15.0}),
        (BurstEncoder, {"max_burst_size": 3}),
        (PopulationEncoder, {"num_neurons": 5}),
    ])
    def test_encoder_consistency(self, encoder_class, kwargs):
        """Test that all encoders work with standard interface."""
        encoder = encoder_class(**kwargs)
        
        # Standard test data
        data = torch.tensor([[0.0, 0.25, 0.5, 0.75, 1.0]])
        duration = 100
        
        spikes = encoder.encode(data, duration)
        
        # Check output format
        if encoder_class == PopulationEncoder:
            assert len(spikes.shape) == 4  # Includes population dimension
        else:
            assert len(spikes.shape) == 3
            assert spikes.shape[:2] == (1, 5)  # batch, channels
            assert spikes.shape[2] == duration
            
        # Check binary spikes
        assert torch.all((spikes >= 0) & (spikes <= 1))
        
    def test_encoder_comparison(self):
        """Compare different encoders on same input."""
        data = torch.tensor([[0.3, 0.7]])
        duration = 100
        
        encoders = {
            'rate': RateEncoder(max_rate=50.0),
            'temporal': TemporalEncoder(max_delay=50),
            'phase': PhaseEncoder(carrier_freq=10.0),
            'burst': BurstEncoder(max_burst_size=5)
        }
        
        results = {}
        for name, encoder in encoders.items():
            spikes = encoder.encode(data, duration)
            if len(spikes.shape) == 3:  # Standard encoders
                results[name] = spikes.sum(dim=-1)  # Total spikes per channel
                
        # Each encoder should produce different patterns
        rate_pattern = results['rate']
        for name, pattern in results.items():
            if name != 'rate':
                assert not torch.equal(rate_pattern, pattern)
                
    def test_batch_processing(self):
        """Test encoders with batch processing."""
        batch_size = 5
        num_channels = 3
        duration = 50
        
        data = torch.rand(batch_size, num_channels)
        
        encoders = [
            RateEncoder(),
            TemporalEncoder(),
            PhaseEncoder(),
            BurstEncoder()
        ]
        
        for encoder in encoders:
            spikes = encoder.encode(data, duration)
            
            if isinstance(encoder, PopulationEncoder):
                assert spikes.shape[0] == batch_size
            else:
                assert spikes.shape == (batch_size, num_channels, duration)
                
    def test_deterministic_encoding(self):
        """Test that encoders are deterministic (except for random components)."""
        torch.manual_seed(42)
        
        data = torch.tensor([[0.4, 0.6]])
        duration = 100
        
        # Test deterministic encoders
        deterministic_encoders = [
            TemporalEncoder(),
            PhaseEncoder(),
            BurstEncoder()
        ]
        
        for encoder in deterministic_encoders:
            spikes1 = encoder.encode(data, duration)
            spikes2 = encoder.encode(data, duration)
            
            # Should be identical for deterministic encoders
            if not isinstance(encoder, RateEncoder):  # Rate encoder is stochastic
                assert torch.equal(spikes1, spikes2)


class TestEncoderEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_input(self):
        """Test handling of empty input."""
        encoder = RateEncoder()
        
        # Empty data should be handled gracefully
        empty_data = torch.empty(0, 0)
        
        with pytest.raises((RuntimeError, ValueError)):
            encoder.encode(empty_data, 100)
            
    def test_single_timestep(self):
        """Test encoding with single timestep."""
        encoder = RateEncoder()
        
        data = torch.tensor([[0.5]])
        spikes = encoder.encode(data, 1)
        
        assert spikes.shape == (1, 1, 1)
        
    def test_zero_duration(self):
        """Test encoding with zero duration."""
        encoder = RateEncoder()
        
        data = torch.tensor([[0.5]])
        
        with pytest.raises((ValueError, RuntimeError)):
            encoder.encode(data, 0)
            
    def test_extreme_values(self):
        """Test encoders with extreme input values."""
        encoders = [
            RateEncoder(),
            TemporalEncoder(),
            PhaseEncoder(),
            BurstEncoder()
        ]
        
        extreme_data = torch.tensor([[-1000.0, 0.0, 1000.0]])
        duration = 50
        
        for encoder in encoders:
            # Should handle extreme values gracefully
            spikes = encoder.encode(extreme_data, duration)
            assert torch.all(torch.isfinite(spikes))
            assert torch.all((spikes >= 0) & (spikes <= 1))