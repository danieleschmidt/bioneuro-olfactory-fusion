"""
ChemicalSensorArray: 8-sensor array with gas-specific selectivity profiles.
Each sensor has a different sensitivity to each gas type.
"""

import numpy as np

# Gas response profiles for each sensor (8 sensors, 6 gases)
# Each vector represents relative sensitivity of the 8 sensors to that gas
GAS_PROFILES = {
    "ethanol":  np.array([0.9, 0.7, 0.2, 0.1, 0.3, 0.5, 0.4, 0.6]),
    "acetone":  np.array([0.3, 0.8, 0.6, 0.2, 0.1, 0.7, 0.5, 0.4]),
    "ammonia":  np.array([0.1, 0.2, 0.9, 0.8, 0.4, 0.3, 0.6, 0.5]),
    "benzene":  np.array([0.4, 0.3, 0.1, 0.9, 0.7, 0.2, 0.8, 0.6]),
    "methane":  np.array([0.6, 0.1, 0.3, 0.4, 0.9, 0.8, 0.2, 0.7]),
    "CO2":      np.array([0.2, 0.5, 0.4, 0.6, 0.8, 0.9, 0.7, 0.3]),
}

SENSOR_NOISE_STD = 0.05


class ChemicalSensorArray:
    """
    Simulates an 8-sensor chemical sensor array with gas-specific selectivity.
    """

    def __init__(self, n_sensors: int = 8):
        self.n_sensors = n_sensors
        self.gas_profiles = GAS_PROFILES
        self.noise_std = SENSOR_NOISE_STD
        self.rng = np.random.default_rng(seed=42)

    def measure(self, gas_concentration: dict) -> np.ndarray:
        """
        Measure the sensor response for a given gas concentration mix.

        Args:
            gas_concentration: dict mapping gas name -> concentration (0.0 to 1.0)

        Returns:
            8-dimensional response vector
        """
        response = np.zeros(self.n_sensors)
        for gas, conc in gas_concentration.items():
            if gas in self.gas_profiles:
                response += conc * self.gas_profiles[gas]
        # Add Gaussian noise
        noise = self.rng.normal(0, self.noise_std, size=self.n_sensors)
        return response + noise

    def measure_batch(self, gas_list: list) -> np.ndarray:
        """
        Measure a batch of gas concentration dicts.

        Args:
            gas_list: list of dicts, each mapping gas name -> concentration

        Returns:
            (batch_size x n_sensors) array
        """
        return np.array([self.measure(g) for g in gas_list])

    def normalize(self, readings: np.ndarray) -> np.ndarray:
        """
        Z-score normalize sensor readings.

        Args:
            readings: array of shape (n_sensors,) or (batch, n_sensors)

        Returns:
            Normalized array of same shape
        """
        mean = np.mean(readings, axis=-1, keepdims=True)
        std = np.std(readings, axis=-1, keepdims=True)
        # Avoid division by zero
        std = np.where(std == 0, 1.0, std)
        return (readings - mean) / std
