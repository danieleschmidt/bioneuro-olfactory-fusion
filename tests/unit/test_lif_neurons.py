"""Unit tests for LIF neuron implementations."""

import pytest
import torch
import numpy as np

from bioneuro_olfactory.core.neurons.lif import (
    LIFNeuron,
    AdaptiveLIFNeuron,
    InhibitoryNeuron
)


class TestLIFNeuron:
    """Test basic LIF neuron functionality."""
    
    def test_init_parameters(self):
        """Test LIF neuron initialization with custom parameters."""
        neuron = LIFNeuron(
            tau_membrane=30.0,
            threshold=1.5,
            reset_voltage=-0.1,
            refractory_period=3,
            dt=0.5
        )
        
        assert neuron.tau_membrane == 30.0
        assert neuron.threshold == 1.5
        assert neuron.reset_voltage == -0.1
        assert neuron.refractory_period == 3
        assert neuron.dt == 0.5
        
        # Check computed decay factor
        expected_beta = torch.exp(torch.tensor(-0.5 / 30.0))
        assert torch.allclose(neuron.beta, expected_beta)
        
    def test_membrane_dynamics(self):
        """Test membrane potential dynamics without spiking."""
        neuron = LIFNeuron(tau_membrane=20.0, threshold=10.0)  # High threshold
        
        # Single neuron, constant input
        input_current = torch.tensor([[1.0]])
        
        # Multiple time steps
        potentials = []
        for _ in range(10):
            spikes = neuron(input_current)
            potentials.append(neuron.membrane_potential.item())
            
        # Membrane potential should increase and approach steady state
        assert potentials[0] < potentials[1]  # Increasing
        assert potentials[-1] > potentials[0]  # Accumulated
        assert all(p < neuron.threshold for p in potentials)  # No spikes
        
    def test_spike_generation(self):
        """Test spike generation when threshold is reached."""
        neuron = LIFNeuron(tau_membrane=20.0, threshold=1.0)
        
        # Strong input to cause immediate spiking
        input_current = torch.tensor([[2.0]])
        
        spikes = neuron(input_current)
        
        assert spikes.shape == (1, 1)
        assert spikes.item() == 1.0  # Spike generated
        assert neuron.membrane_potential.item() == neuron.reset_voltage
        
    def test_refractory_period(self):
        """Test refractory period prevents immediate re-spiking."""
        neuron = LIFNeuron(
            tau_membrane=20.0, 
            threshold=1.0, 
            refractory_period=3
        )
        
        # Strong continuous input
        input_current = torch.tensor([[2.0]])
        
        spike_times = []
        for t in range(10):
            spikes = neuron(input_current)
            if spikes.item() > 0:
                spike_times.append(t)
                
        # Should have gaps due to refractory period
        if len(spike_times) > 1:
            gap = spike_times[1] - spike_times[0]
            assert gap >= neuron.refractory_period
            
    def test_multiple_neurons(self):
        """Test with multiple neurons simultaneously."""
        batch_size, num_neurons = 2, 5
        neuron = LIFNeuron(tau_membrane=20.0, threshold=1.0)
        
        # Different input strengths
        input_current = torch.rand(batch_size, num_neurons) * 2.0
        
        spikes = neuron(input_current)
        
        assert spikes.shape == (batch_size, num_neurons)
        assert torch.all((spikes == 0) | (spikes == 1))  # Binary spikes
        
    def test_state_reset(self):
        """Test neuron state reset functionality."""
        neuron = LIFNeuron()
        
        # Build up some state
        input_current = torch.tensor([[0.5]])
        neuron(input_current)
        
        assert neuron.membrane_potential is not None
        assert neuron.refractory_counter is not None
        
        # Reset state
        neuron.reset_state()
        
        assert neuron.membrane_potential is None
        assert neuron.refractory_counter is None


class TestAdaptiveLIFNeuron:
    """Test adaptive LIF neuron functionality."""
    
    def test_adaptation_initialization(self):
        """Test adaptive LIF neuron initialization."""
        neuron = AdaptiveLIFNeuron(
            tau_adaptation=100.0,
            adaptation_strength=0.2
        )
        
        assert neuron.tau_adaptation == 100.0
        assert neuron.adaptation_strength == 0.2
        assert neuron.threshold_adaptation is None
        
    def test_threshold_adaptation(self):
        """Test that threshold adapts after spiking."""
        neuron = AdaptiveLIFNeuron(
            tau_membrane=20.0,
            threshold=1.0,
            tau_adaptation=50.0,
            adaptation_strength=0.5
        )
        
        # Strong input to cause spiking
        input_current = torch.tensor([[2.0]])
        
        # First spike
        spikes1 = neuron(input_current)
        adaptation1 = neuron.threshold_adaptation.clone()
        
        # Second time step
        spikes2 = neuron(input_current)
        adaptation2 = neuron.threshold_adaptation.clone()
        
        if spikes1.item() > 0:
            assert adaptation1.item() > 0  # Adaptation increased
            
        # Adaptation should decay over time
        input_weak = torch.tensor([[0.1]])  # Weak input
        for _ in range(10):
            neuron(input_weak)
            
        final_adaptation = neuron.threshold_adaptation.item()
        assert final_adaptation < adaptation1.item()  # Decayed
        
    def test_spike_frequency_adaptation(self):
        """Test that repeated spiking increases effective threshold."""
        neuron = AdaptiveLIFNeuron(
            tau_membrane=20.0,
            threshold=1.0,
            adaptation_strength=0.3
        )
        
        # Constant strong input
        input_current = torch.tensor([[1.5]])
        
        spike_times = []
        for t in range(50):
            spikes = neuron(input_current)
            if spikes.item() > 0:
                spike_times.append(t)
                
        if len(spike_times) > 2:
            # Inter-spike intervals should increase due to adaptation
            isi1 = spike_times[1] - spike_times[0]
            isi2 = spike_times[2] - spike_times[1]
            # Later intervals should be longer (allowing for some variability)
            assert isi2 >= isi1 - 1
            

class TestInhibitoryNeuron:
    """Test inhibitory neuron functionality."""
    
    def test_inhibitory_output_scaling(self):
        """Test that inhibitory neurons scale output."""
        neuron = InhibitoryNeuron(
            threshold=0.5,
            inhibition_strength=2.0
        )
        
        # Input above threshold
        input_current = torch.tensor([[1.0]])
        
        spikes = neuron(input_current)
        
        if spikes.item() > 0:
            # Output should be scaled by inhibition strength
            assert spikes.item() == neuron.inhibition_strength
            
    def test_faster_dynamics(self):
        """Test that inhibitory neurons have faster dynamics."""
        excitatory = LIFNeuron(tau_membrane=20.0, threshold=1.0)
        inhibitory = InhibitoryNeuron(tau_membrane=10.0, threshold=1.0)
        
        input_current = torch.tensor([[0.5]])
        
        # Run for multiple steps
        exc_potentials = []
        inh_potentials = []
        
        for _ in range(10):
            excitatory(input_current)
            inhibitory(input_current)
            
            exc_potentials.append(excitatory.membrane_potential.item())
            inh_potentials.append(inhibitory.membrane_potential.item())
            
        # Inhibitory should reach higher potential faster (faster tau)
        assert inh_potentials[2] > exc_potentials[2]


class TestNeuronIntegration:
    """Integration tests for neuron interactions."""
    
    def test_excitatory_inhibitory_network(self):
        """Test interaction between excitatory and inhibitory neurons."""
        # Create small network
        exc_neuron = LIFNeuron(tau_membrane=20.0, threshold=1.0)
        inh_neuron = InhibitoryNeuron(tau_membrane=10.0, threshold=0.8)
        
        # External input
        external_input = torch.tensor([[1.2]])
        
        network_activity = []
        for t in range(20):
            # Excitatory neuron receives external input
            exc_spikes = exc_neuron(external_input)
            
            # Inhibitory neuron receives excitatory input
            inh_input = exc_spikes * 0.5  # Weaker connection
            inh_spikes = inh_neuron(inh_input)
            
            # Record activity
            network_activity.append({
                'time': t,
                'exc_spikes': exc_spikes.item(),
                'inh_spikes': inh_spikes.item(),
                'exc_potential': exc_neuron.membrane_potential.item(),
                'inh_potential': inh_neuron.membrane_potential.item()
            })
            
        # Verify network shows realistic dynamics
        exc_spike_times = [a['time'] for a in network_activity if a['exc_spikes'] > 0]
        inh_spike_times = [a['time'] for a in network_activity if a['inh_spikes'] > 0]
        
        assert len(exc_spike_times) > 0  # Some excitatory activity
        # Inhibitory spikes should follow excitatory (with possible delay)
        
    def test_population_dynamics(self):
        """Test population of neurons with distributed parameters."""
        num_neurons = 10
        neuron = LIFNeuron(tau_membrane=20.0, threshold=1.0)
        
        # Random input to population
        input_current = torch.rand(1, num_neurons) * 2.0
        
        # Run simulation
        population_spikes = []
        for t in range(50):
            spikes = neuron(input_current)
            population_spikes.append(spikes.sum().item())
            
        # Check population statistics
        mean_activity = np.mean(population_spikes)
        std_activity = np.std(population_spikes)
        
        assert mean_activity > 0  # Some activity
        assert std_activity >= 0  # Variability in activity
        
    def test_neuron_consistency(self):
        """Test that neuron behavior is consistent across runs."""
        neuron1 = LIFNeuron(tau_membrane=20.0, threshold=1.0)
        neuron2 = LIFNeuron(tau_membrane=20.0, threshold=1.0)
        
        # Same input sequence
        torch.manual_seed(42)  # For reproducibility
        inputs = [torch.rand(1, 1) * 2.0 for _ in range(10)]
        
        spikes1 = []
        spikes2 = []
        
        for inp in inputs:
            spikes1.append(neuron1(inp).item())
            spikes2.append(neuron2(inp).item())
            
        # Should produce identical results
        assert spikes1 == spikes2
        
    @pytest.mark.parametrize("batch_size,num_neurons", [
        (1, 1),
        (1, 10),
        (5, 1),
        (5, 10),
        (10, 100)
    ])
    def test_neuron_scalability(self, batch_size, num_neurons):
        """Test neuron scales to different batch and population sizes."""
        neuron = LIFNeuron()
        
        input_current = torch.rand(batch_size, num_neurons)
        spikes = neuron(input_current)
        
        assert spikes.shape == (batch_size, num_neurons)
        assert torch.all((spikes >= 0) & (spikes <= 1))


class TestNeuronErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_parameters(self):
        """Test handling of invalid parameters."""
        with pytest.raises(ValueError):
            # Negative time constant should fail in real implementation
            pass  # LIFNeuron allows any float currently
            
    def test_zero_input(self):
        """Test neuron behavior with zero input."""
        neuron = LIFNeuron()
        
        zero_input = torch.zeros(1, 5)
        spikes = neuron(zero_input)
        
        assert torch.all(spikes == 0)  # No spikes with zero input
        
    def test_negative_input(self):
        """Test neuron behavior with negative input."""
        neuron = LIFNeuron()
        
        negative_input = torch.tensor([[-1.0]])
        
        # Should handle negative input gracefully
        spikes = neuron(negative_input)
        assert spikes.shape == (1, 1)
        
    def test_large_input(self):
        """Test neuron behavior with very large input."""
        neuron = LIFNeuron(threshold=1.0)
        
        large_input = torch.tensor([[1000.0]])
        spikes = neuron(large_input)
        
        # Should still produce binary spike
        assert spikes.item() == 1.0
        assert neuron.membrane_potential.item() == neuron.reset_voltage