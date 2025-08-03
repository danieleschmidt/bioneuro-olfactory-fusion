"""Spike encoding schemes for converting analog sensor data to spike trains."""

import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple


class SpikeEncoder(ABC):
    """Abstract base class for spike encoding schemes."""
    
    @abstractmethod
    def encode(self, data: torch.Tensor, duration: int) -> torch.Tensor:
        """Encode analog data into spike trains.
        
        Args:
            data: Input analog data [batch_size, num_channels]
            duration: Duration in time steps
            
        Returns:
            spikes: Binary spike tensor [batch_size, num_channels, duration]
        """
        pass


class RateEncoder(SpikeEncoder):
    """Rate-based spike encoding.
    
    Converts analog values to Poisson spike trains where
    higher values produce higher firing rates.
    """
    
    def __init__(self, max_rate: float = 200.0, dt: float = 1.0):
        self.max_rate = max_rate
        self.dt = dt
        
    def encode(self, data: torch.Tensor, duration: int) -> torch.Tensor:
        """Encode using Poisson process."""
        batch_size, num_channels = data.shape
        
        # Normalize data to [0, 1] range
        data_norm = torch.clamp(data, 0, 1)
        
        # Convert to firing rates (Hz)
        rates = data_norm * self.max_rate
        
        # Convert to spike probability per time step
        spike_prob = rates * self.dt / 1000.0  # Convert ms to s
        
        # Generate Poisson spikes
        spikes = torch.rand(batch_size, num_channels, duration) < spike_prob.unsqueeze(-1)
        
        return spikes.float()


class TemporalEncoder(SpikeEncoder):
    """Temporal spike encoding based on time-to-first-spike.
    
    Higher analog values produce earlier spike times,
    providing precise temporal information.
    """
    
    def __init__(self, precision: float = 1.0, max_delay: int = 100):
        self.precision = precision
        self.max_delay = max_delay
        
    def encode(self, data: torch.Tensor, duration: int) -> torch.Tensor:
        """Encode using time-to-first-spike."""
        batch_size, num_channels = data.shape
        
        # Initialize output tensor
        spikes = torch.zeros(batch_size, num_channels, duration)
        
        # Normalize data to [0, 1] range
        data_norm = torch.clamp(data, 0, 1)
        
        # Calculate spike times (higher values = earlier spikes)
        spike_times = (1 - data_norm) * self.max_delay
        spike_times = torch.clamp(spike_times, 0, duration - 1).long()
        
        # Set spikes at calculated times
        for b in range(batch_size):
            for c in range(num_channels):
                if data_norm[b, c] > 0:  # Only spike if input > 0
                    t = spike_times[b, c].item()
                    spikes[b, c, t] = 1.0
                    
        return spikes


class PhaseEncoder(SpikeEncoder):
    """Phase-based spike encoding using oscillatory patterns.
    
    Encodes analog values as phase relationships relative
    to a carrier oscillation.
    """
    
    def __init__(self, carrier_freq: float = 40.0, dt: float = 1.0):
        self.carrier_freq = carrier_freq  # Hz
        self.dt = dt
        
    def encode(self, data: torch.Tensor, duration: int) -> torch.Tensor:
        """Encode using phase modulation."""
        batch_size, num_channels = data.shape
        
        # Create time vector
        t = torch.arange(duration) * self.dt / 1000.0  # Convert to seconds
        
        # Normalize data to [0, 2π] phase range
        data_norm = torch.clamp(data, 0, 1)
        phases = data_norm * 2 * np.pi
        
        # Generate carrier wave with phase modulation
        carrier = torch.sin(2 * np.pi * self.carrier_freq * t)
        
        # Apply phase modulation
        spikes = torch.zeros(batch_size, num_channels, duration)
        for b in range(batch_size):
            for c in range(num_channels):
                modulated = torch.sin(
                    2 * np.pi * self.carrier_freq * t + phases[b, c]
                )
                # Generate spikes on positive peaks
                spikes[b, c] = (modulated > 0.8).float()
                
        return spikes


class BurstEncoder(SpikeEncoder):
    """Burst-based spike encoding.
    
    Encodes analog values as burst patterns where
    higher values produce longer bursts.
    """
    
    def __init__(self, max_burst_size: int = 10, inter_burst_interval: int = 20):
        self.max_burst_size = max_burst_size
        self.inter_burst_interval = inter_burst_interval
        
    def encode(self, data: torch.Tensor, duration: int) -> torch.Tensor:
        """Encode using burst patterns."""
        batch_size, num_channels = data.shape
        
        # Initialize output tensor
        spikes = torch.zeros(batch_size, num_channels, duration)
        
        # Normalize data to [0, 1] range
        data_norm = torch.clamp(data, 0, 1)
        
        # Calculate burst sizes
        burst_sizes = (data_norm * self.max_burst_size).long()
        
        # Generate burst patterns
        for b in range(batch_size):
            for c in range(num_channels):
                burst_size = burst_sizes[b, c].item()
                if burst_size > 0:
                    # Generate bursts throughout duration
                    total_period = burst_size + self.inter_burst_interval
                    num_bursts = duration // total_period
                    
                    for burst_idx in range(num_bursts):
                        start_time = burst_idx * total_period
                        end_time = start_time + burst_size
                        if end_time <= duration:
                            spikes[b, c, start_time:end_time] = 1.0
                            
        return spikes


class PopulationEncoder(SpikeEncoder):
    """Population vector encoding.
    
    Distributes analog values across multiple neurons
    with overlapping tuning curves.
    """
    
    def __init__(self, num_neurons: int = 10, sigma: float = 0.3):
        self.num_neurons = num_neurons
        self.sigma = sigma
        
        # Create preferred values for each neuron
        self.preferred_values = torch.linspace(0, 1, num_neurons)
        
    def encode(self, data: torch.Tensor, duration: int) -> torch.Tensor:
        """Encode using population vector."""
        batch_size, num_channels = data.shape
        
        # Normalize data to [0, 1] range
        data_norm = torch.clamp(data, 0, 1)
        
        # Calculate responses for each neuron
        responses = torch.zeros(batch_size, num_channels, self.num_neurons)
        
        for i, pref_val in enumerate(self.preferred_values):
            # Gaussian tuning curve
            response = torch.exp(-0.5 * ((data_norm - pref_val) / self.sigma) ** 2)
            responses[:, :, i] = response
            
        # Convert responses to spike trains using rate encoding
        rate_encoder = RateEncoder(max_rate=100.0)
        
        # Reshape for encoding
        responses_flat = responses.view(batch_size, num_channels * self.num_neurons)
        spikes_flat = rate_encoder.encode(responses_flat, duration)
        
        # Reshape back
        spikes = spikes_flat.view(batch_size, num_channels, self.num_neurons, duration)
        
        return spikes


class AdaptiveEncoder(SpikeEncoder):
    """Adaptive spike encoder with gain control.
    
    Automatically adjusts encoding parameters based on
    input statistics for optimal dynamic range usage.
    """
    
    def __init__(
        self,
        base_encoder: SpikeEncoder,
        adaptation_rate: float = 0.01,
        target_rate: float = 20.0
    ):
        self.base_encoder = base_encoder
        self.adaptation_rate = adaptation_rate
        self.target_rate = target_rate
        
        # Running statistics
        self.running_mean = 0.0
        self.running_std = 1.0
        self.update_count = 0
        
    def encode(self, data: torch.Tensor, duration: int) -> torch.Tensor:
        """Encode with adaptive normalization."""
        # Update running statistics
        batch_mean = data.mean().item()
        batch_std = data.std().item() + 1e-8
        
        if self.update_count == 0:
            self.running_mean = batch_mean
            self.running_std = batch_std
        else:
            self.running_mean += self.adaptation_rate * (batch_mean - self.running_mean)
            self.running_std += self.adaptation_rate * (batch_std - self.running_std)
            
        self.update_count += 1
        
        # Normalize data using running statistics
        data_normalized = (data - self.running_mean) / self.running_std
        data_normalized = torch.sigmoid(data_normalized)  # Map to [0, 1]
        
        # Encode using base encoder
        return self.base_encoder.encode(data_normalized, duration)
        
    def reset_adaptation(self):
        """Reset adaptation statistics."""
        self.running_mean = 0.0
        self.running_std = 1.0
        self.update_count = 0
