"""Pytest configuration and shared fixtures."""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from pathlib import Path


@pytest.fixture
def mock_sensor_data():
    """Mock sensor array data."""
    return {
        "chemical": np.random.rand(6, 100),  # 6 sensors, 100 samples
        "timestamps": np.linspace(0, 1, 100),
        "temperature": 23.5,
        "humidity": 65.2
    }


@pytest.fixture
def mock_audio_features():
    """Mock audio feature data."""
    return {
        "mfcc": np.random.rand(13, 100),
        "mel_spectrogram": np.random.rand(128, 100),
        "spectral_contrast": np.random.rand(7, 100),
        "zero_crossing_rate": np.random.rand(100)
    }


@pytest.fixture
def mock_spike_train():
    """Mock spike train data."""
    return {
        "spike_times": np.sort(np.random.rand(50) * 1000),  # 50 spikes in 1000ms
        "neuron_ids": np.random.randint(0, 1000, 50),
        "weights": np.random.rand(50)
    }


@pytest.fixture
def test_data_dir():
    """Test data directory path."""
    return Path(__file__).parent / "fixtures" / "data"


@pytest.fixture
def mock_neuromorphic_hardware():
    """Mock neuromorphic hardware interface."""
    mock_hw = Mock()
    mock_hw.is_available.return_value = False
    mock_hw.compile.return_value = MagicMock()
    mock_hw.run.return_value = {"spikes": [], "energy_mj": 0.1}
    return mock_hw


@pytest.fixture(scope="session")
def sample_gas_dataset():
    """Sample gas detection dataset."""
    return {
        "gases": ["methane", "carbon_monoxide", "ammonia"],
        "concentrations": np.logspace(1, 4, 100),  # 10-10000 ppm
        "labels": np.random.randint(0, 3, 100),
        "sensor_responses": np.random.rand(100, 6)
    }