"""
CNNCrossSection: 1D convolution over sensor channels for pattern recognition.
Pure numpy implementation — no torch dependency.
"""

import numpy as np


class CNNCrossSection:
    """
    1D convolutional network over sensor channel patterns.
    """

    def __init__(
        self,
        n_sensors: int = 8,
        n_filters: int = 16,
        kernel_size: int = 3,
    ):
        self.n_sensors = n_sensors
        self.n_filters = n_filters
        self.kernel_size = kernel_size

        rng = np.random.default_rng(seed=1)
        # Conv kernel: (n_filters, kernel_size)
        self.kernel = rng.uniform(
            -0.5, 0.5, size=(n_filters, kernel_size)
        )
        self.bias = rng.uniform(-0.1, 0.1, size=(n_filters,))

    def conv1d(self, x: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
        """
        Manual 1D convolution (valid padding).

        Args:
            x: input signal of shape (length,)
            kernel: filters of shape (n_filters, kernel_size)
            bias: shape (n_filters,)

        Returns:
            Output of shape (n_filters, length - kernel_size + 1)
        """
        n_filters, kernel_size = kernel.shape
        length = len(x)
        out_len = length - kernel_size + 1
        output = np.zeros((n_filters, out_len))
        for f in range(n_filters):
            for i in range(out_len):
                output[f, i] = np.dot(x[i: i + kernel_size], kernel[f]) + bias[f]
        return output

    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation."""
        return np.maximum(0.0, x)

    def max_pool(self, x: np.ndarray, pool_size: int = 2) -> np.ndarray:
        """
        1D max pooling along the last axis.

        Args:
            x: array of shape (n_filters, length)
            pool_size: pooling window size

        Returns:
            Pooled array of shape (n_filters, length // pool_size)
        """
        n_filters, length = x.shape
        out_len = length // pool_size
        output = np.zeros((n_filters, out_len))
        for i in range(out_len):
            output[:, i] = np.max(x[:, i * pool_size: (i + 1) * pool_size], axis=1)
        return output

    def forward(self, sensor_readings: np.ndarray) -> np.ndarray:
        """
        Forward pass through conv1d -> relu -> max_pool -> flatten.

        Args:
            sensor_readings: array of shape (n_sensors,)

        Returns:
            Flattened feature vector
        """
        out = self.conv1d(sensor_readings, self.kernel, self.bias)
        out = self.relu(out)
        out = self.max_pool(out)
        return out.flatten()

    @property
    def output_size(self) -> int:
        """Size of the flattened output feature vector."""
        conv_out_len = self.n_sensors - self.kernel_size + 1
        pool_out_len = conv_out_len // 2
        return self.n_filters * pool_out_len
