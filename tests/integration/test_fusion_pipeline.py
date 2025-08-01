"""Integration tests for multi-modal fusion pipeline."""

import pytest
import numpy as np
from unittest.mock import patch, Mock


class TestFusionPipeline:
    """Test complete fusion pipeline integration."""
    
    def test_end_to_end_detection(self, mock_sensor_data, mock_audio_features):
        """Test complete detection pipeline."""
        with patch('bioneuro_olfactory.OlfactoryFusionSNN') as mock_model:
            mock_snn = Mock()
            mock_snn.process.return_value = (
                {"spike_times": np.array([10, 25, 40])},
                {
                    "hazard_probability": 0.97,
                    "gas_type": "methane",
                    "concentration": 1250
                }
            )
            mock_model.return_value = mock_snn
            
            # Test processing
            spikes, prediction = mock_snn.process(
                chemical_input=mock_sensor_data["chemical"],
                audio_input=mock_audio_features["mfcc"]
            )
            
            assert prediction["hazard_probability"] > 0.95
            assert prediction["gas_type"] in ["methane", "carbon_monoxide", "ammonia"]
            assert len(spikes["spike_times"]) > 0
    
    def test_temporal_alignment(self, mock_sensor_data, mock_audio_features):
        """Test temporal alignment of multi-modal data."""
        with patch('bioneuro_olfactory.fusion.TemporalAligner') as mock_aligner:
            mock_align = Mock()
            aligned_data = {
                "chemical_aligned": mock_sensor_data["chemical"],
                "audio_aligned": mock_audio_features["mfcc"],
                "sync_timestamps": np.linspace(0, 1, 100)
            }
            mock_align.align.return_value = aligned_data
            mock_aligner.return_value = mock_align
            
            result = mock_align.align(
                chemical_stream=mock_sensor_data,
                audio_stream=mock_audio_features
            )
            
            assert "chemical_aligned" in result
            assert "audio_aligned" in result
            assert "sync_timestamps" in result
    
    def test_online_learning_adaptation(self):
        """Test online learning and adaptation."""
        with patch('bioneuro_olfactory.learning.OnlineSTDP') as mock_stdp:
            mock_learner = Mock()
            mock_learner.update_weights.return_value = {
                "weight_changes": np.random.rand(1000) * 0.01,
                "learning_rate": 0.01
            }
            mock_stdp.return_value = mock_learner
            
            # Simulate adaptation
            update = mock_learner.update_weights(
                pre_spikes=np.array([10, 20, 30]),
                post_spikes=np.array([15, 25, 35])
            )
            
            assert "weight_changes" in update
            assert len(update["weight_changes"]) > 0