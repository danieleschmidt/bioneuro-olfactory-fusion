"""
BioNeuroOlfactoryPipeline: End-to-end gas detection pipeline combining
ChemicalSensorArray, SNNTemporalEncoder, CNNCrossSection, and FusionClassifier.
"""

import numpy as np

from .sensor_array import ChemicalSensorArray
from .snn_encoder import SNNTemporalEncoder
from .cnn_cross_section import CNNCrossSection
from .fusion_classifier import FusionClassifier


class BioNeuroOlfactoryPipeline:
    """
    Full pipeline for bio-inspired multi-modal gas detection.
    """

    def __init__(self):
        self.sensor = ChemicalSensorArray(n_sensors=8)
        self.snn = SNNTemporalEncoder(n_input=8, n_neurons=32)
        self.cnn = CNNCrossSection(n_sensors=8, n_filters=16, kernel_size=3)

        # SNN temporal features: 3 * n_neurons
        snn_features_dim = 3 * self.snn.n_neurons  # 96
        cnn_features_dim = self.cnn.output_size

        self.classifier = FusionClassifier(
            snn_features_dim=snn_features_dim,
            cnn_features_dim=cnn_features_dim,
        )

    def detect(self, gas_concentration: dict, n_timesteps: int = 50) -> dict:
        """
        Detect gas from a concentration dictionary.

        Args:
            gas_concentration: dict mapping gas name -> concentration
            n_timesteps: number of SNN simulation steps

        Returns:
            dict with keys: gas, confidence, snn_spikes, cnn_features
        """
        # 1. Measure sensor response
        reading = self.sensor.measure(gas_concentration)

        # 2. SNN encoding
        spike_train = self.snn.encode_sequence(reading, n_timesteps=n_timesteps)
        snn_features = self.snn.get_temporal_features(spike_train)

        # 3. CNN features
        cnn_features = self.cnn.forward(reading)

        # 4. Fusion classification
        probs = self.classifier.forward(snn_features, cnn_features)
        gas = self.classifier.gas_classes[int(np.argmax(probs))]
        confidence = float(np.max(probs))

        return {
            "gas": gas,
            "confidence": confidence,
            "snn_spikes": spike_train,
            "cnn_features": cnn_features,
        }

    def detect_batch(self, gas_list: list) -> list:
        """
        Detect gas for a batch of concentration dicts.

        Args:
            gas_list: list of dicts

        Returns:
            List of result dicts
        """
        return [self.detect(g) for g in gas_list]

    def demo(self):
        """Run detection on 6 example gas mixtures and print results."""
        examples = [
            {"ethanol": 1.0},
            {"acetone": 1.0},
            {"ammonia": 0.8},
            {"benzene": 0.6, "ethanol": 0.4},
            {"methane": 0.9, "CO2": 0.1},
            {},  # clean air
        ]

        labels = [
            "Pure ethanol",
            "Pure acetone",
            "Ammonia (0.8)",
            "Benzene+Ethanol mix",
            "Methane+CO2 mix",
            "Clean air",
        ]

        print("BioNeuro Olfactory Fusion — Demo\n" + "=" * 40)
        for label, gas_conc in zip(labels, examples):
            result = self.detect(gas_conc)
            print(
                f"{label:25s} → {result['gas']:12s} (confidence: {result['confidence']:.3f})"
            )
