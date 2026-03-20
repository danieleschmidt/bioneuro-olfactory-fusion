"""
FusionClassifier: Concatenates SNN and CNN features, applies a 2-layer MLP,
and produces gas class probabilities.
"""

import numpy as np

GAS_CLASSES = [
    "ethanol",
    "acetone",
    "ammonia",
    "benzene",
    "methane",
    "CO2",
    "mixture",
    "clean_air",
]


class FusionClassifier:
    """
    Multi-layer perceptron that fuses SNN and CNN features for gas classification.
    """

    def __init__(
        self,
        snn_features_dim: int,
        cnn_features_dim: int,
        hidden_dim: int = 64,
        n_classes: int = 8,
    ):
        self.snn_features_dim = snn_features_dim
        self.cnn_features_dim = cnn_features_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.gas_classes = GAS_CLASSES

        input_dim = snn_features_dim + cnn_features_dim

        rng = np.random.default_rng(seed=2)
        # Layer 1: input_dim -> hidden_dim
        scale1 = np.sqrt(2.0 / input_dim)
        self.W1 = rng.normal(0, scale1, size=(hidden_dim, input_dim))
        self.b1 = np.zeros(hidden_dim)

        # Layer 2: hidden_dim -> n_classes
        scale2 = np.sqrt(2.0 / hidden_dim)
        self.W2 = rng.normal(0, scale2, size=(n_classes, hidden_dim))
        self.b2 = np.zeros(n_classes)

    def mlp_layer(self, x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Linear layer followed by ReLU activation.

        Args:
            x: input vector
            W: weight matrix (out_dim, in_dim)
            b: bias vector (out_dim,)

        Returns:
            Activated output of shape (out_dim,)
        """
        return np.maximum(0.0, W @ x + b)

    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def forward(self, snn_features: np.ndarray, cnn_features: np.ndarray) -> np.ndarray:
        """
        Compute class probabilities from SNN and CNN features.

        Args:
            snn_features: feature vector from SNN encoder
            cnn_features: feature vector from CNN

        Returns:
            Class probability vector of shape (n_classes,)
        """
        x = np.concatenate([snn_features, cnn_features])
        h = self.mlp_layer(x, self.W1, self.b1)
        logits = self.W2 @ h + self.b2
        return self.softmax(logits)

    def predict(self, snn_features: np.ndarray, cnn_features: np.ndarray) -> str:
        """
        Predict the most likely gas class.

        Args:
            snn_features: feature vector from SNN encoder
            cnn_features: feature vector from CNN

        Returns:
            Gas class name as string
        """
        probs = self.forward(snn_features, cnn_features)
        idx = int(np.argmax(probs))
        return self.gas_classes[idx]
