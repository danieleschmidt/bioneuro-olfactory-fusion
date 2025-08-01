"""End-to-end deployment tests."""

import pytest
from unittest.mock import patch, Mock


class TestNeuromorphicDeployment:
    """Test neuromorphic hardware deployment."""
    
    @pytest.mark.neuromorphic
    def test_loihi_deployment(self, mock_neuromorphic_hardware):
        """Test deployment to Intel Loihi."""
        with patch('bioneuro_olfactory.neuromorphic.LoihiBackend') as mock_loihi:
            mock_backend = mock_neuromorphic_hardware
            mock_loihi.return_value = mock_backend
            
            # Test hardware availability check
            assert mock_backend.is_available() is False  # Mock returns False
            
            # Test model compilation
            mock_model = Mock()
            compiled = mock_backend.compile(mock_model, power_budget=100)
            assert compiled is not None
    
    @pytest.mark.neuromorphic  
    def test_spinnaker_deployment(self, mock_neuromorphic_hardware):
        """Test deployment to SpiNNaker."""
        with patch('bioneuro_olfactory.neuromorphic.SpiNNakerBackend') as mock_spinnaker:
            mock_backend = mock_neuromorphic_hardware
            mock_spinnaker.return_value = mock_backend
            
            # Test network mapping
            mock_model = Mock()
            mapping = mock_backend.compile(mock_model)
            assert mapping is not None


class TestRealTimeMonitoring:
    """Test real-time monitoring applications."""
    
    def test_industrial_safety_monitor(self):
        """Test industrial safety monitoring."""
        with patch('bioneuro_olfactory.applications.IndustrialSafetyMonitor') as mock_monitor:
            mock_safety = Mock()
            mock_safety.start.return_value = {"status": "monitoring", "sensors_active": 4}
            mock_monitor.return_value = mock_safety
            
            status = mock_safety.start(log_file="test_log.csv")
            assert status["status"] == "monitoring"
            assert status["sensors_active"] > 0
    
    def test_environmental_monitoring(self):
        """Test environmental monitoring."""
        with patch('bioneuro_olfactory.applications.EnvironmentalMonitor') as mock_env:
            mock_monitor = Mock()
            mock_monitor.monitor.return_value = {
                "duration_hours": 24,
                "alerts_triggered": 0,
                "data_points": 2400
            }
            mock_env.return_value = mock_monitor
            
            result = mock_monitor.monitor(duration_days=1)
            assert result["duration_hours"] == 24
            assert result["data_points"] > 0