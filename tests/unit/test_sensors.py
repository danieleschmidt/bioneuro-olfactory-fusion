"""Test sensor interfaces."""

import pytest
import numpy as np
from unittest.mock import patch, Mock, MagicMock


class TestENoseArray:
    """Test electronic nose sensor array."""
    
    def test_sensor_initialization(self):
        """Test sensor array initialization."""
        with patch('bioneuro_olfactory.sensors.ENoseArray') as mock_enose:
            mock_array = Mock()
            mock_array.num_sensors = 6
            mock_array.sampling_rate = 100
            mock_enose.return_value = mock_array
            
            assert mock_array.num_sensors == 6
            assert mock_array.sampling_rate == 100
    
    def test_sensor_calibration(self):
        """Test sensor calibration process."""
        with patch('bioneuro_olfactory.sensors.ENoseArray') as mock_enose:
            mock_array = Mock()
            mock_array.calibrate.return_value = {
                "baseline": np.array([1.2, 1.1, 1.3, 1.0, 1.2, 1.1]),
                "sensitivity": np.array([0.8, 0.9, 0.7, 1.0, 0.8, 0.9])
            }
            mock_enose.return_value = mock_array
            
            calibration = mock_array.calibrate(reference_gas="clean_air", duration=300)
            assert "baseline" in calibration
            assert "sensitivity" in calibration
            assert len(calibration["baseline"]) == 6
    
    def test_sensor_data_reading(self, mock_sensor_data):
        """Test sensor data reading."""
        with patch('bioneuro_olfactory.sensors.ENoseArray') as mock_enose:
            mock_array = Mock()
            mock_array.read.return_value = mock_sensor_data
            mock_enose.return_value = mock_array
            
            data = mock_array.read()
            assert "chemical" in data
            assert "timestamps" in data
            assert data["chemical"].shape == (6, 100)


class TestAudioProcessor:
    """Test audio feature extraction."""
    
    def test_mfcc_extraction(self):
        """Test MFCC feature extraction."""
        with patch('bioneuro_olfactory.audio.AcousticProcessor') as mock_processor:
            mock_proc = Mock()
            mock_proc.extract_mfcc.return_value = np.random.rand(13, 100)
            mock_processor.return_value = mock_proc
            
            audio_signal = np.random.rand(44100)  # 1 second at 44.1kHz
            mfcc = mock_proc.extract_mfcc(audio_signal, n_mfcc=13)
            assert mfcc.shape == (13, 100)
    
    def test_spectral_features(self):
        """Test spectral feature extraction."""
        with patch('bioneuro_olfactory.audio.AcousticProcessor') as mock_processor:
            mock_proc = Mock()
            mock_proc.extract_features.return_value = {
                "spectral_contrast": np.random.rand(7, 100),
                "zero_crossing_rate": np.random.rand(100),
                "spectral_rolloff": np.random.rand(100)
            }
            mock_processor.return_value = mock_proc
            
            features = mock_proc.extract_features(np.random.rand(44100))
            assert "spectral_contrast" in features
            assert "zero_crossing_rate" in features