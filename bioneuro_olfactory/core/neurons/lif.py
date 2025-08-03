"""Leaky Integrate-and-Fire (LIF) neuron implementations for neuromorphic computing."""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire neuron model.
    
    Implements the classic LIF dynamics with configurable parameters
    for membrane time constant, threshold, and reset behavior.
    """
    
    def __init__(
        self,
        tau_membrane: float = 20.0,
        threshold: float = 1.0,
        reset_voltage: float = 0.0,
        refractory_period: int = 2,
        dt: float = 1.0
    ):
        super().__init__()
        self.tau_membrane = tau_membrane
        self.threshold = threshold
        self.reset_voltage = reset_voltage
        self.refractory_period = refractory_period
        self.dt = dt
        
        # Decay factor for membrane potential
        self.beta = torch.exp(-dt / tau_membrane)
        
        # Internal state
        self.membrane_potential = None
        self.refractory_counter = None
        
    def forward(self, input_current: torch.Tensor) -> torch.Tensor:
        """Forward pass through LIF neurons.
        
        Args:
            input_current: Input current [batch_size, num_neurons]
            
        Returns:
            spike_output: Binary spike tensor [batch_size, num_neurons]
        """
        batch_size, num_neurons = input_current.shape
        
        # Initialize state if needed
        if self.membrane_potential is None:
            self.membrane_potential = torch.zeros_like(input_current)
            self.refractory_counter = torch.zeros_like(input_current)
            
        # Update membrane potential (leaky integration)
        self.membrane_potential = (
            self.beta * self.membrane_potential + input_current
        )
        
        # Check for spikes
        spikes = (self.membrane_potential >= self.threshold).float()
        
        # Apply refractory period
        refractory_mask = self.refractory_counter > 0
        spikes = spikes * (~refractory_mask).float()
        
        # Reset membrane potential after spike
        self.membrane_potential = torch.where(
            spikes.bool(),
            torch.full_like(self.membrane_potential, self.reset_voltage),
            self.membrane_potential
        )
        
        # Update refractory counter
        self.refractory_counter = torch.where(
            spikes.bool(),
            torch.full_like(self.refractory_counter, self.refractory_period),
            torch.clamp(self.refractory_counter - 1, min=0)
        )
        
        return spikes
        
    def reset_state(self):
        """Reset neuron state to initial conditions."""
        self.membrane_potential = None
        self.refractory_counter = None


class AdaptiveLIFNeuron(LIFNeuron):
    """Adaptive LIF neuron with spike-frequency adaptation.
    
    Extends basic LIF with an adaptive threshold that increases
    after each spike, providing gain control.
    """
    
    def __init__(
        self,
        tau_membrane: float = 20.0,
        threshold: float = 1.0,
        reset_voltage: float = 0.0,
        refractory_period: int = 2,
        dt: float = 1.0,
        tau_adaptation: float = 100.0,
        adaptation_strength: float = 0.1
    ):
        super().__init__(tau_membrane, threshold, reset_voltage, refractory_period, dt)
        
        self.tau_adaptation = tau_adaptation
        self.adaptation_strength = adaptation_strength
        self.beta_adaptation = torch.exp(-dt / tau_adaptation)
        
        # Adaptive threshold component
        self.threshold_adaptation = None
        
    def forward(self, input_current: torch.Tensor) -> torch.Tensor:
        """Forward pass with adaptive threshold."""
        batch_size, num_neurons = input_current.shape
        
        # Initialize adaptive threshold if needed
        if self.threshold_adaptation is None:
            self.threshold_adaptation = torch.zeros_like(input_current)
            
        # Update membrane potential
        if self.membrane_potential is None:
            self.membrane_potential = torch.zeros_like(input_current)
            self.refractory_counter = torch.zeros_like(input_current)
            
        self.membrane_potential = (
            self.beta * self.membrane_potential + input_current
        )
        
        # Adaptive threshold
        effective_threshold = self.threshold + self.threshold_adaptation
        
        # Check for spikes
        spikes = (self.membrane_potential >= effective_threshold).float()
        
        # Apply refractory period
        refractory_mask = self.refractory_counter > 0
        spikes = spikes * (~refractory_mask).float()
        
        # Reset membrane potential after spike
        self.membrane_potential = torch.where(
            spikes.bool(),
            torch.full_like(self.membrane_potential, self.reset_voltage),
            self.membrane_potential
        )
        
        # Update adaptive threshold
        self.threshold_adaptation = (
            self.beta_adaptation * self.threshold_adaptation +
            self.adaptation_strength * spikes
        )
        
        # Update refractory counter
        self.refractory_counter = torch.where(
            spikes.bool(),
            torch.full_like(self.refractory_counter, self.refractory_period),
            torch.clamp(self.refractory_counter - 1, min=0)
        )
        
        return spikes
        
    def reset_state(self):
        """Reset neuron state including adaptation."""
        super().reset_state()
        self.threshold_adaptation = None


class InhibitoryNeuron(LIFNeuron):
    """Inhibitory LIF neuron for lateral inhibition.
    
    Specialized neuron for implementing winner-take-all
    and sparse coding mechanisms.
    """
    
    def __init__(
        self,
        tau_membrane: float = 10.0,  # Faster dynamics
        threshold: float = 0.5,      # Lower threshold
        reset_voltage: float = 0.0,
        refractory_period: int = 1,   # Shorter refractory
        dt: float = 1.0,
        inhibition_strength: float = 2.0
    ):
        super().__init__(tau_membrane, threshold, reset_voltage, refractory_period, dt)
        self.inhibition_strength = inhibition_strength
        
    def forward(self, input_current: torch.Tensor) -> torch.Tensor:
        """Forward pass with inhibitory output."""
        spikes = super().forward(input_current)
        # Scale inhibitory output
        return spikes * self.inhibition_strength
