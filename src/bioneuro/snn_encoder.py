"""
SNNTemporalEncoder: Leaky Integrate-and-Fire (LIF) neuron model for encoding
sensor readings as spike trains.
"""

import numpy as np


class SNNTemporalEncoder:
    """
    Spiking Neural Network encoder using LIF neurons.
    Encodes sensor readings as temporal spike patterns.
    """

    def __init__(
        self,
        n_input: int = 8,
        n_neurons: int = 32,
        tau_mem: float = 10.0,
        threshold: float = 1.0,
        dt: float = 0.1,
    ):
        self.n_input = n_input
        self.n_neurons = n_neurons
        self.tau_mem = tau_mem
        self.threshold = threshold
        self.dt = dt

        # Random input weights: (n_input -> n_neurons)
        rng = np.random.default_rng(seed=0)
        self.W = rng.uniform(0.0, 0.5, size=(n_neurons, n_input))

        # Membrane potentials
        self.membrane_potential = np.zeros(n_neurons)

    def reset(self):
        """Reset membrane potentials to zero."""
        self.membrane_potential = np.zeros(self.n_neurons)

    def step(self, input_current: np.ndarray) -> np.ndarray:
        """
        Perform one LIF timestep.

        Args:
            input_current: array of shape (n_input,) — raw sensor inputs

        Returns:
            Binary spike vector of shape (n_neurons,)
        """
        # Project input to neuron space
        projected = self.W @ input_current  # shape: (n_neurons,)

        # Update membrane potential: leaky decay + input
        self.membrane_potential = (
            (1.0 - self.dt / self.tau_mem) * self.membrane_potential + projected
        )

        # Fire where membrane potential >= threshold
        spikes = (self.membrane_potential >= self.threshold).astype(float)

        # Reset fired neurons
        self.membrane_potential = np.where(spikes > 0, 0.0, self.membrane_potential)

        return spikes

    def encode_sequence(
        self, sensor_readings: np.ndarray, n_timesteps: int = 50
    ) -> np.ndarray:
        """
        Encode a sensor reading into a spike train over n_timesteps.

        Args:
            sensor_readings: array of shape (n_input,)
            n_timesteps: number of simulation steps

        Returns:
            Spike train array of shape (n_timesteps, n_neurons)
        """
        self.reset()
        spike_train = np.zeros((n_timesteps, self.n_neurons))
        for t in range(n_timesteps):
            spike_train[t] = self.step(sensor_readings)
        return spike_train

    def get_spike_rate(self, spike_train: np.ndarray) -> np.ndarray:
        """
        Compute mean firing rate per neuron.

        Args:
            spike_train: array of shape (n_timesteps, n_neurons)

        Returns:
            Mean firing rate per neuron, shape (n_neurons,)
        """
        return np.mean(spike_train, axis=0)

    def get_temporal_features(self, spike_train: np.ndarray) -> np.ndarray:
        """
        Extract temporal features from a spike train.

        Args:
            spike_train: array of shape (n_timesteps, n_neurons)

        Returns:
            Concatenation of [mean, std, first_spike_time] per neuron,
            shape (3 * n_neurons,)
        """
        mean_rate = np.mean(spike_train, axis=0)
        std_rate = np.std(spike_train, axis=0)

        # First spike time: index of first 1 per neuron, or n_timesteps if no spike
        n_timesteps = spike_train.shape[0]
        first_spike_time = np.full(self.n_neurons, float(n_timesteps))
        for n in range(self.n_neurons):
            spikes_at = np.where(spike_train[:, n] > 0)[0]
            if len(spikes_at) > 0:
                first_spike_time[n] = float(spikes_at[0])

        return np.concatenate([mean_rate, std_rate, first_spike_time])
