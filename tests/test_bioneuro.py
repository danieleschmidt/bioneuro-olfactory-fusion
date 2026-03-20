"""
Comprehensive tests for bioneuro-olfactory-fusion.
"""

import numpy as np
import pytest
import sys
import os

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bioneuro.sensor_array import ChemicalSensorArray, GAS_PROFILES, SENSOR_NOISE_STD
from bioneuro.snn_encoder import SNNTemporalEncoder
from bioneuro.cnn_cross_section import CNNCrossSection
from bioneuro.fusion_classifier import FusionClassifier, GAS_CLASSES
from bioneuro.pipeline import BioNeuroOlfactoryPipeline


# ── Sensor Array ──────────────────────────────────────────────────────────────

def test_sensor_array_response_shape():
    sensor = ChemicalSensorArray()
    reading = sensor.measure({"ethanol": 1.0})
    assert reading.shape == (8,), f"Expected (8,), got {reading.shape}"


def test_sensor_array_noise():
    """Two measurements of same gas should differ due to noise."""
    sensor = ChemicalSensorArray()
    r1 = sensor.measure({"ethanol": 1.0})
    r2 = sensor.measure({"ethanol": 1.0})
    assert not np.allclose(r1, r2), "Measurements should differ due to noise"


def test_sensor_array_normalize():
    sensor = ChemicalSensorArray()
    reading = sensor.measure({"ethanol": 1.0})
    normed = sensor.normalize(reading)
    assert normed.shape == reading.shape
    assert abs(np.mean(normed)) < 1e-10 or normed.shape == (1,), \
        "Z-score normalized mean should be ~0"
    # std should be ~1 (within float tolerance) for non-constant input
    assert abs(np.std(normed) - 1.0) < 1e-6


def test_sensor_array_known_gas():
    """Measure a pure gas — response should correlate with its profile."""
    sensor = ChemicalSensorArray(n_sensors=8)
    # Use a fixed seed to get a deterministic measurement
    sensor.rng = np.random.default_rng(seed=999)
    reading = sensor.measure({"ethanol": 1.0})
    profile = GAS_PROFILES["ethanol"]
    # The reading should be close to the profile (within noise tolerance)
    diff = np.abs(reading - profile)
    assert np.all(diff < 5 * SENSOR_NOISE_STD), \
        f"Reading too far from gas profile: max diff = {diff.max():.4f}"


# ── SNN Encoder ───────────────────────────────────────────────────────────────

def test_snn_encoder_step():
    snn = SNNTemporalEncoder(n_input=8, n_neurons=32)
    input_current = np.ones(8) * 0.5
    spikes = snn.step(input_current)
    assert spikes.shape == (32,)
    assert set(np.unique(spikes)).issubset({0.0, 1.0}), "Spikes must be binary (0 or 1)"


def test_snn_encoder_membrane_dynamics():
    """Membrane potential should accumulate over time."""
    snn = SNNTemporalEncoder(n_input=8, n_neurons=32, threshold=999.0)  # Never fire
    input_current = np.ones(8)
    v_before = snn.membrane_potential.copy()
    snn.step(input_current)
    v_after = snn.membrane_potential.copy()
    # At least some potentials should have increased
    assert np.any(v_after != v_before), "Membrane potential should change after step"


def test_snn_encoder_spike_threshold():
    """Neurons should fire when membrane potential exceeds threshold."""
    snn = SNNTemporalEncoder(n_input=8, n_neurons=32, threshold=0.001)  # Very low threshold
    input_current = np.ones(8)
    spikes = snn.step(input_current)
    # At least some neurons should fire with very low threshold and large input
    assert np.sum(spikes) > 0, "At least some neurons should fire"


def test_snn_encode_sequence_shape():
    snn = SNNTemporalEncoder(n_input=8, n_neurons=32)
    reading = np.ones(8) * 0.5
    spike_train = snn.encode_sequence(reading, n_timesteps=50)
    assert spike_train.shape == (50, 32), f"Expected (50, 32), got {spike_train.shape}"


def test_snn_temporal_features():
    snn = SNNTemporalEncoder(n_input=8, n_neurons=32)
    reading = np.ones(8) * 0.3
    spike_train = snn.encode_sequence(reading, n_timesteps=50)
    features = snn.get_temporal_features(spike_train)
    # Should be 3 * n_neurons = 96
    assert features.shape == (96,), f"Expected (96,), got {features.shape}"


# ── CNN Cross Section ─────────────────────────────────────────────────────────

def test_cnn_conv1d_basic():
    cnn = CNNCrossSection(n_sensors=8, n_filters=4, kernel_size=3)
    x = np.ones(8)
    out = cnn.conv1d(x, cnn.kernel[:4], cnn.bias[:4])
    # Valid convolution: 8 - 3 + 1 = 6 positions, 4 filters
    assert out.shape == (4, 6), f"Expected (4, 6), got {out.shape}"


def test_cnn_forward_shape():
    cnn = CNNCrossSection(n_sensors=8, n_filters=16, kernel_size=3)
    reading = np.random.rand(8)
    features = cnn.forward(reading)
    expected_size = cnn.output_size
    assert features.shape == (expected_size,), \
        f"Expected ({expected_size},), got {features.shape}"


# ── Fusion Classifier ─────────────────────────────────────────────────────────

def test_fusion_classifier_output_shape():
    clf = FusionClassifier(snn_features_dim=96, cnn_features_dim=48, n_classes=8)
    snn_feat = np.random.rand(96)
    cnn_feat = np.random.rand(48)
    probs = clf.forward(snn_feat, cnn_feat)
    assert probs.shape == (8,), f"Expected (8,), got {probs.shape}"


def test_fusion_classifier_probabilities_sum():
    clf = FusionClassifier(snn_features_dim=96, cnn_features_dim=48, n_classes=8)
    snn_feat = np.random.rand(96)
    cnn_feat = np.random.rand(48)
    probs = clf.forward(snn_feat, cnn_feat)
    assert abs(probs.sum() - 1.0) < 1e-6, f"Probabilities should sum to 1, got {probs.sum()}"
    assert np.all(probs >= 0), "All probabilities should be non-negative"


# ── Pipeline ──────────────────────────────────────────────────────────────────

def test_pipeline_detect():
    pipeline = BioNeuroOlfactoryPipeline()
    result = pipeline.detect({"ethanol": 1.0})
    assert "gas" in result
    assert "confidence" in result
    assert "snn_spikes" in result
    assert "cnn_features" in result
    assert isinstance(result["gas"], str)
    assert result["gas"] in GAS_CLASSES
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["snn_spikes"].shape == (50, 32)


def test_pipeline_batch():
    pipeline = BioNeuroOlfactoryPipeline()
    gas_list = [
        {"ethanol": 1.0},
        {"acetone": 1.0},
        {"ammonia": 0.8},
    ]
    results = pipeline.detect_batch(gas_list)
    assert len(results) == 3
    for r in results:
        assert "gas" in r
        assert "confidence" in r
        assert r["gas"] in GAS_CLASSES
