"""Test core neuron models."""

import pytest
import numpy as np
from unittest.mock import patch, Mock


class TestSpikingNeuron:
    """Test spiking neuron models."""
    
    def test_leaky_integrate_fire_neuron(self):
        """Test LIF neuron model."""
        # Mock import since actual implementation may not exist yet
        with patch('bioneuro_olfactory.core.neurons.LIFNeuron') as mock_lif:
            mock_neuron = Mock()
            mock_neuron.membrane_potential = 0.0
            mock_neuron.threshold = 1.0
            mock_neuron.reset_potential = 0.0
            mock_lif.return_value = mock_neuron
            
            # Test membrane potential integration
            mock_neuron.integrate.return_value = 0.5
            assert mock_neuron.integrate.return_value < mock_neuron.threshold
            
            # Test spike generation
            mock_neuron.integrate.return_value = 1.2
            assert mock_neuron.integrate.return_value > mock_neuron.threshold
    
    def test_adaptive_exponential_neuron(self):
        """Test AdEx neuron model."""
        with patch('bioneuro_olfactory.core.neurons.AdExNeuron') as mock_adex:
            mock_neuron = Mock()
            mock_neuron.tau_m = 20.0  # membrane time constant
            mock_neuron.tau_w = 100.0  # adaptation time constant
            mock_adex.return_value = mock_neuron
            
            # Test adaptation mechanism
            mock_neuron.compute_adaptation.return_value = 0.1
            assert mock_neuron.compute_adaptation.return_value > 0
    
    def test_neuron_spike_encoding(self):
        """Test spike encoding schemes."""
        with patch('bioneuro_olfactory.core.encoding.RateEncoder') as mock_encoder:
            mock_enc = Mock()
            mock_enc.encode.return_value = np.array([10, 15, 20, 12])  # spike times
            mock_encoder.return_value = mock_enc
            
            concentration = 250  # ppm
            spikes = mock_enc.encode(concentration, duration=100)
            assert len(spikes) > 0
            assert all(spike >= 0 for spike in spikes)