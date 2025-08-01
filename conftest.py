"""
Global pytest configuration and fixtures.
"""

import pytest
import tempfile
import shutil
from pathlib import Path


@pytest.fixture(scope="session")
def temp_workspace():
    """Create a temporary workspace for tests."""
    temp_dir = tempfile.mkdtemp(prefix="bioneuro_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_sensor_data():
    """Mock sensor data for testing."""
    return {
        "chemical": [0.1, 0.2, 0.3, 0.4, 0.5],
        "audio": [0.01, 0.02, 0.03, 0.04, 0.05],
        "timestamp": "2025-01-15T10:30:00Z"
    }


@pytest.fixture
def performance_threshold():
    """Performance thresholds for benchmark tests."""
    return {
        "inference_time_ms": 50,
        "memory_usage_mb": 100,
        "spike_rate": 1000
    }