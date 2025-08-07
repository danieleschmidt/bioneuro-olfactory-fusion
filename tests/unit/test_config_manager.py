"""Tests for comprehensive configuration management."""

import pytest
import tempfile
import json
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from bioneuro_olfactory.core.config_manager import (
    ConfigManager, ApplicationConfig, NeuralConfig, SensorConfig,
    SecurityConfig, SystemConfig, ConfigValidationError, ConfigFormat
)


class TestNeuralConfig:
    """Test neural configuration validation."""
    
    def test_neural_config_defaults(self):
        """Test default neural configuration values."""
        config = NeuralConfig()
        
        assert config.tau_membrane == 20.0
        assert config.threshold == 1.0
        assert config.reset_voltage == 0.0
        assert config.refractory_period == 2
        assert config.dt == 1.0
        assert config.num_projection_neurons == 1000
        assert config.num_kenyon_cells == 5000
        assert config.sparsity_level == 0.05
        
    def test_neural_config_validation_valid(self):
        """Test neural configuration validation with valid values."""
        config = NeuralConfig()
        
        errors = config.validate()
        
        assert len(errors) == 0
        
    def test_neural_config_validation_invalid(self):
        """Test neural configuration validation with invalid values."""
        config = NeuralConfig(
            tau_membrane=-10.0,      # Invalid: negative
            threshold=-1.0,          # Invalid: negative
            dt=-1.0,                # Invalid: negative
            sparsity_level=1.5,     # Invalid: > 1
            learning_rate=-0.01,    # Invalid: negative
            batch_size=0,           # Invalid: zero
            epochs=-5               # Invalid: negative
        )
        
        errors = config.validate()
        
        assert len(errors) > 0
        assert any('tau_membrane must be positive' in err for err in errors)
        assert any('threshold must be positive' in err for err in errors)
        assert any('dt must be positive' in err for err in errors)
        assert any('sparsity_level must be between 0 and 1' in err for err in errors)
        assert any('learning_rate must be positive' in err for err in errors)
        assert any('batch_size must be positive' in err for err in errors)
        assert any('epochs must be positive' in err for err in errors)
        
    def test_neural_config_numerical_stability_warning(self):
        """Test neural configuration warning for numerical stability."""
        config = NeuralConfig(
            tau_membrane=5.0,
            dt=10.0  # dt > tau_membrane
        )
        
        errors = config.validate()
        
        assert any('numerical stability' in err for err in errors)


class TestSensorConfig:
    """Test sensor configuration validation."""
    
    def test_sensor_config_defaults(self):
        """Test default sensor configuration values."""
        config = SensorConfig()
        
        assert config.sensor_types == ["MQ2", "MQ7", "MQ135"]
        assert config.sampling_rate_hz == 1.0
        assert config.calibration_interval_hours == 24
        assert config.temperature_compensation is True
        assert config.humidity_compensation is True
        assert config.serial_port == "/dev/ttyUSB0"
        assert config.baud_rate == 9600
        
    def test_sensor_config_validation_valid(self):
        """Test sensor configuration validation with valid values."""
        config = SensorConfig()
        
        errors = config.validate()
        
        assert len(errors) == 0
        
    def test_sensor_config_validation_invalid(self):
        """Test sensor configuration validation with invalid values."""
        config = SensorConfig(
            sampling_rate_hz=-1.0,              # Invalid: negative
            calibration_interval_hours=-5,      # Invalid: negative  
            baud_rate=12345,                    # Invalid: non-standard
            timeout_seconds=-1.0,               # Invalid: negative
            sensor_ranges={
                "MQ2": (10000, 200),           # Invalid: min > max
                "MQ7": (10,),                  # Invalid: not 2 values
            }
        )
        
        errors = config.validate()
        
        assert len(errors) > 0
        assert any('sampling_rate_hz must be positive' in err for err in errors)
        assert any('calibration_interval_hours must be positive' in err for err in errors)
        assert any('baud_rate must be a standard value' in err for err in errors)
        assert any('timeout_seconds must be positive' in err for err in errors)
        assert any('min must be less than max' in err for err in errors)
        assert any('must have exactly 2 elements' in err for err in errors)


class TestSecurityConfig:
    """Test security configuration validation."""
    
    def test_security_config_defaults(self):
        """Test default security configuration values."""
        config = SecurityConfig()
        
        assert config.jwt_secret_key == "change_me_in_production"
        assert config.jwt_expiration_hours == 24
        assert config.password_min_length == 12
        assert config.max_login_attempts == 5
        assert config.enable_encryption is True
        assert config.session_timeout_minutes == 60
        
    def test_security_config_validation_valid(self):
        """Test security configuration validation with valid values."""
        config = SecurityConfig(jwt_secret_key="a" * 32)  # Long enough
        
        errors = config.validate()
        
        assert len(errors) == 0
        
    def test_security_config_validation_invalid(self):
        """Test security configuration validation with invalid values."""
        config = SecurityConfig(
            jwt_secret_key="short",              # Invalid: too short
            jwt_expiration_hours=-1,            # Invalid: negative
            password_min_length=4,              # Invalid: too short
            max_login_attempts=0,               # Invalid: zero
            session_timeout_minutes=-10,       # Invalid: negative
            rate_limit_requests_per_minute=-5, # Invalid: negative
            audit_log_retention_days=0         # Invalid: zero
        )
        
        errors = config.validate()
        
        assert len(errors) > 0
        assert any('jwt_secret_key should be at least 32 characters' in err for err in errors)
        assert any('jwt_expiration_hours must be positive' in err for err in errors)
        assert any('password_min_length should be at least 8' in err for err in errors)
        assert any('max_login_attempts must be positive' in err for err in errors)
        assert any('session_timeout_minutes must be positive' in err for err in errors)
        assert any('rate_limit_requests_per_minute must be positive' in err for err in errors)
        assert any('audit_log_retention_days must be positive' in err for err in errors)


class TestSystemConfig:
    """Test system configuration validation."""
    
    def test_system_config_defaults(self):
        """Test default system configuration values."""
        config = SystemConfig()
        
        assert config.log_level == "INFO"
        assert config.log_file == "/var/log/bioneuro/system.log"
        assert config.num_workers == 4
        assert config.enable_gpu is True
        assert config.bind_address == "0.0.0.0"
        assert config.port == 8080
        assert config.enable_ssl is False
        
    def test_system_config_validation_valid(self):
        """Test system configuration validation with valid values."""
        config = SystemConfig()
        
        errors = config.validate()
        
        assert len(errors) == 0
        
    def test_system_config_validation_invalid(self):
        """Test system configuration validation with invalid values."""
        config = SystemConfig(
            log_level="INVALID",          # Invalid: not a valid level
            log_rotation_size_mb=-1,      # Invalid: negative
            num_workers=0,                # Invalid: zero
            max_memory_usage_mb=-100,     # Invalid: negative
            port=100000,                  # Invalid: > 65535
            enable_ssl=True,              # Invalid: SSL enabled but no certs
            ssl_cert_file=None,
            ssl_key_file=None
        )
        
        errors = config.validate()
        
        assert len(errors) > 0
        assert any('log_level must be a valid logging level' in err for err in errors)
        assert any('log_rotation_size_mb must be positive' in err for err in errors)
        assert any('num_workers must be positive' in err for err in errors)
        assert any('max_memory_usage_mb must be positive' in err for err in errors)
        assert any('port must be between 1 and 65535' in err for err in errors)
        assert any('SSL enabled but certificate/key files not specified' in err for err in errors)


class TestApplicationConfig:
    """Test application configuration validation."""
    
    def test_application_config_defaults(self):
        """Test default application configuration."""
        config = ApplicationConfig()
        
        assert isinstance(config.neural, NeuralConfig)
        assert isinstance(config.sensor, SensorConfig)
        assert isinstance(config.security, SecurityConfig)
        assert isinstance(config.system, SystemConfig)
        assert config.version == "0.1.0"
        assert config.environment == "development"
        assert config.debug is False
        
    def test_application_config_validation_valid(self):
        """Test application configuration validation with valid configs."""
        config = ApplicationConfig()
        
        errors = config.validate()
        
        # Should have some errors due to default values (like short JWT key)
        security_errors = [err for err in errors if 'security.' in err]
        assert len(security_errors) > 0  # jwt_secret_key too short
        
    def test_application_config_cross_component_validation(self):
        """Test cross-component validation."""
        # Mock PyTorch not available
        with patch('builtins.__import__', side_effect=ImportError):
            config = ApplicationConfig()
            config.system.enable_gpu = True
            
            errors = config.validate()
            
            assert any('GPU enabled but PyTorch not available' in err for err in errors)


class TestConfigManager:
    """Test configuration manager functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_manager = ConfigManager(config_dirs=[self.temp_dir.name])
        
    def teardown_method(self):
        """Cleanup test fixtures."""
        self.temp_dir.cleanup()
        
    def test_load_config_no_file(self):
        """Test loading configuration when no file exists."""
        config = self.config_manager.load_config()
        
        assert isinstance(config, ApplicationConfig)
        assert config.version == "0.1.0"  # Default value
        
    def test_load_config_json_file(self):
        """Test loading configuration from JSON file."""
        config_data = {
            "neural": {
                "tau_membrane": 25.0,
                "threshold": 1.5
            },
            "sensor": {
                "sampling_rate_hz": 2.0
            },
            "version": "1.2.3",
            "environment": "testing"
        }
        
        config_file = Path(self.temp_dir.name) / "bioneuro.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
            
        config = self.config_manager.load_config()
        
        assert config.neural.tau_membrane == 25.0
        assert config.neural.threshold == 1.5
        assert config.sensor.sampling_rate_hz == 2.0
        assert config.version == "1.2.3"
        assert config.environment == "testing"
        
    def test_load_config_yaml_file(self):
        """Test loading configuration from YAML file."""
        config_data = {
            "neural": {
                "tau_membrane": 30.0,
                "num_projection_neurons": 2000
            },
            "system": {
                "log_level": "DEBUG",
                "port": 9090
            }
        }
        
        config_file = Path(self.temp_dir.name) / "bioneuro.yaml"
        with open(config_file, 'w') as f:
            yaml.safe_dump(config_data, f)
            
        config = self.config_manager.load_config()
        
        assert config.neural.tau_membrane == 30.0
        assert config.neural.num_projection_neurons == 2000
        assert config.system.log_level == "DEBUG"
        assert config.system.port == 9090
        
    def test_load_config_environment_overrides(self):
        """Test configuration overrides from environment variables."""
        with patch.dict('os.environ', {
            'BIONEURO_NEURAL_TAU_MEMBRANE': '15.0',
            'BIONEURO_SENSOR_SAMPLING_RATE_HZ': '5.0',
            'BIONEURO_SYSTEM_PORT': '7777',
            'BIONEURO_DEBUG': 'true'
        }):
            config = self.config_manager.load_config()
            
            assert config.neural.tau_membrane == 15.0
            assert config.sensor.sampling_rate_hz == 5.0
            assert config.system.port == 7777
            assert config.debug is True
            
    def test_load_config_validation_error(self):
        """Test loading configuration that fails validation."""
        config_data = {
            "neural": {
                "tau_membrane": -10.0  # Invalid
            }
        }
        
        config_file = Path(self.temp_dir.name) / "bioneuro.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
            
        with pytest.raises(ConfigValidationError):
            self.config_manager.load_config()
            
    def test_update_config(self):
        """Test updating configuration at runtime."""
        config = self.config_manager.load_config(validate=False)  # Skip validation for test
        
        updates = {
            "neural.tau_membrane": 35.0,
            "sensor.sampling_rate_hz": 0.5,
            "version": "2.0.0"
        }
        
        self.config_manager.update_config(updates, validate=False)
        
        updated_config = self.config_manager.get_config()
        
        assert updated_config.neural.tau_membrane == 35.0
        assert updated_config.sensor.sampling_rate_hz == 0.5
        assert updated_config.version == "2.0.0"
        
    def test_update_config_validation_error(self):
        """Test updating configuration with invalid values."""
        config = self.config_manager.load_config(validate=False)
        
        updates = {
            "neural.tau_membrane": -5.0  # Invalid
        }
        
        with pytest.raises(ConfigValidationError):
            self.config_manager.update_config(updates)
            
    def test_save_config_json(self):
        """Test saving configuration to JSON file."""
        config = self.config_manager.load_config(validate=False)
        
        # Modify config
        config.neural.tau_membrane = 40.0
        config.version = "3.0.0"
        
        save_path = Path(self.temp_dir.name) / "saved_config.json"
        self.config_manager.save_config(save_path)
        
        # Verify file was created and has correct content
        assert save_path.exists()
        
        with open(save_path, 'r') as f:
            saved_data = json.load(f)
            
        assert saved_data["neural"]["tau_membrane"] == 40.0
        assert saved_data["version"] == "3.0.0"
        
    def test_save_config_yaml(self):
        """Test saving configuration to YAML file."""
        config = self.config_manager.load_config(validate=False)
        
        config.neural.threshold = 2.5
        config.system.port = 8888
        
        save_path = Path(self.temp_dir.name) / "saved_config.yaml"
        self.config_manager.save_config(save_path)
        
        assert save_path.exists()
        
        with open(save_path, 'r') as f:
            saved_data = yaml.safe_load(f)
            
        assert saved_data["neural"]["threshold"] == 2.5
        assert saved_data["system"]["port"] == 8888
        
    def test_config_change_watcher(self):
        """Test configuration change notification."""
        config = self.config_manager.load_config(validate=False)
        
        # Register watcher
        callback_called = False
        old_config_received = None
        new_config_received = None
        
        def config_change_callback(old_config, new_config):
            nonlocal callback_called, old_config_received, new_config_received
            callback_called = True
            old_config_received = old_config
            new_config_received = new_config
            
        self.config_manager.watch_config_changes(config_change_callback)
        
        # Update config
        updates = {"neural.tau_membrane": 50.0}
        self.config_manager.update_config(updates, validate=False)
        
        # Verify callback was called
        assert callback_called
        assert old_config_received is not None
        assert new_config_received is not None
        assert new_config_received.neural.tau_membrane == 50.0
        
    def test_config_file_change_detection(self):
        """Test detection of configuration file changes."""
        # Create initial config file
        config_data = {"version": "1.0.0"}
        config_file = Path(self.temp_dir.name) / "bioneuro.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
            
        # Load config
        config = self.config_manager.load_config()
        assert config.version == "1.0.0"
        
        # Initially no changes
        assert not self.config_manager.check_config_file_changes()
        
        # Modify file
        config_data["version"] = "2.0.0"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
            
        # Should detect change
        assert self.config_manager.check_config_file_changes()
        
    def test_reload_config_if_changed(self):
        """Test automatic reloading when config file changes."""
        # Create initial config file
        config_data = {"version": "1.0.0"}
        config_file = Path(self.temp_dir.name) / "bioneuro.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
            
        # Load config
        config = self.config_manager.load_config(validate=False)
        assert config.version == "1.0.0"
        
        # Modify file
        config_data["version"] = "2.0.0"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
            
        # Reload if changed
        reloaded = self.config_manager.reload_config_if_changed()
        assert reloaded
        
        # Verify new config
        updated_config = self.config_manager.get_config()
        assert updated_config.version == "2.0.0"
        
    def test_parse_environment_values(self):
        """Test parsing of environment variable values."""
        test_cases = [
            ("true", True),
            ("false", False),
            ("True", True),
            ("False", False),
            ("42", 42),
            ("3.14", 3.14),
            ('{"key": "value"}', {"key": "value"}),
            ('["a", "b", "c"]', ["a", "b", "c"]),
            ("simple_string", "simple_string")
        ]
        
        for env_value, expected in test_cases:
            parsed = self.config_manager._parse_env_value(env_value)
            assert parsed == expected, f"Failed to parse {env_value}"
            
    def test_merge_configs(self):
        """Test configuration dictionary merging."""
        base_config = {
            "neural": {
                "tau_membrane": 20.0,
                "threshold": 1.0
            },
            "system": {
                "port": 8080
            }
        }
        
        override_config = {
            "neural": {
                "threshold": 1.5,  # Override existing
                "dt": 0.5          # Add new
            },
            "sensor": {             # Add new section
                "sampling_rate_hz": 2.0
            }
        }
        
        merged = self.config_manager._merge_configs(base_config, override_config)
        
        # Check merged values
        assert merged["neural"]["tau_membrane"] == 20.0  # Preserved
        assert merged["neural"]["threshold"] == 1.5      # Overridden
        assert merged["neural"]["dt"] == 0.5             # Added
        assert merged["system"]["port"] == 8080          # Preserved
        assert merged["sensor"]["sampling_rate_hz"] == 2.0  # Added
        
    def test_invalid_config_file_format(self):
        """Test handling of invalid configuration file format."""
        # Create invalid JSON file
        config_file = Path(self.temp_dir.name) / "bioneuro.json"
        with open(config_file, 'w') as f:
            f.write("invalid json content {")
            
        with pytest.raises(ConfigValidationError):
            self.config_manager.load_config()


if __name__ == "__main__":
    pytest.main([__file__])