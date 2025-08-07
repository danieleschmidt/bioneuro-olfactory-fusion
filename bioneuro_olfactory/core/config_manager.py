"""Robust configuration management for neuromorphic gas detection system.

This module provides comprehensive configuration management with validation,
environment-specific settings, and runtime configuration updates.
"""

import os
import json
import yaml
import toml
from typing import Dict, Any, Optional, List, Union, Type
from dataclasses import dataclass, field, fields
from pathlib import Path
from enum import Enum
import logging
from copy import deepcopy
import threading
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


class ConfigFormat(Enum):
    """Supported configuration file formats."""
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    ENV = "env"


class ConfigValidationError(Exception):
    """Configuration validation error."""
    pass


@dataclass
class NeuralConfig:
    """Neural network configuration parameters."""
    # LIF Neuron parameters
    tau_membrane: float = 20.0
    threshold: float = 1.0
    reset_voltage: float = 0.0
    refractory_period: int = 2
    dt: float = 1.0
    
    # Adaptive parameters
    tau_adaptation: float = 100.0
    adaptation_strength: float = 0.1
    
    # Network architecture
    num_projection_neurons: int = 1000
    num_kenyon_cells: int = 5000
    sparsity_level: float = 0.05
    
    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    
    def validate(self) -> List[str]:
        """Validate neural configuration parameters."""
        errors = []
        
        if self.tau_membrane <= 0:
            errors.append("tau_membrane must be positive")
        if self.threshold <= 0:
            errors.append("threshold must be positive")
        if self.dt <= 0:
            errors.append("dt must be positive")
        if self.dt > self.tau_membrane:
            errors.append("dt should not exceed tau_membrane for numerical stability")
        if not 0 < self.sparsity_level < 1:
            errors.append("sparsity_level must be between 0 and 1")
        if self.learning_rate <= 0:
            errors.append("learning_rate must be positive")
        if self.batch_size <= 0:
            errors.append("batch_size must be positive")
        if self.epochs <= 0:
            errors.append("epochs must be positive")
            
        return errors


@dataclass
class SensorConfig:
    """Sensor array configuration parameters."""
    # Sensor types and configurations
    sensor_types: List[str] = field(default_factory=lambda: ["MQ2", "MQ7", "MQ135"])
    sampling_rate_hz: float = 1.0
    calibration_interval_hours: int = 24
    
    # Environmental compensation
    temperature_compensation: bool = True
    humidity_compensation: bool = True
    pressure_compensation: bool = False
    
    # Sensor thresholds (ppm)
    detection_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "methane": 1000.0,
        "carbon_monoxide": 50.0,
        "ammonia": 25.0
    })
    
    # Sensor ranges (ppm)
    sensor_ranges: Dict[str, tuple] = field(default_factory=lambda: {
        "MQ2": (200, 10000),
        "MQ7": (10, 1000), 
        "MQ135": (10, 300)
    })
    
    # Communication settings
    serial_port: Optional[str] = "/dev/ttyUSB0"
    baud_rate: int = 9600
    timeout_seconds: float = 1.0
    
    def validate(self) -> List[str]:
        """Validate sensor configuration parameters."""
        errors = []
        
        if self.sampling_rate_hz <= 0:
            errors.append("sampling_rate_hz must be positive")
        if self.calibration_interval_hours <= 0:
            errors.append("calibration_interval_hours must be positive")
        if self.baud_rate not in [9600, 19200, 38400, 57600, 115200]:
            errors.append("baud_rate must be a standard value")
        if self.timeout_seconds <= 0:
            errors.append("timeout_seconds must be positive")
            
        # Validate sensor ranges
        for sensor_type, range_tuple in self.sensor_ranges.items():
            if len(range_tuple) != 2:
                errors.append(f"sensor_ranges[{sensor_type}] must have exactly 2 values")
            elif range_tuple[0] >= range_tuple[1]:
                errors.append(f"sensor_ranges[{sensor_type}] min must be less than max")
                
        return errors


@dataclass
class SecurityConfig:
    """Security configuration parameters."""
    # Authentication
    jwt_secret_key: str = "change_me_in_production"
    jwt_expiration_hours: int = 24
    password_min_length: int = 12
    max_login_attempts: int = 5
    
    # Encryption
    enable_encryption: bool = True
    encryption_algorithm: str = "AES-256-GCM"
    key_rotation_days: int = 90
    
    # Access control
    enable_rbac: bool = True
    session_timeout_minutes: int = 60
    
    # Rate limiting
    rate_limit_requests_per_minute: int = 100
    rate_limit_window_minutes: int = 1
    
    # Audit logging
    audit_log_enabled: bool = True
    audit_log_retention_days: int = 365
    
    def validate(self) -> List[str]:
        """Validate security configuration parameters."""
        errors = []
        
        if len(self.jwt_secret_key) < 32:
            errors.append("jwt_secret_key should be at least 32 characters")
        if self.jwt_expiration_hours <= 0:
            errors.append("jwt_expiration_hours must be positive")
        if self.password_min_length < 8:
            errors.append("password_min_length should be at least 8")
        if self.max_login_attempts <= 0:
            errors.append("max_login_attempts must be positive")
        if self.session_timeout_minutes <= 0:
            errors.append("session_timeout_minutes must be positive")
        if self.rate_limit_requests_per_minute <= 0:
            errors.append("rate_limit_requests_per_minute must be positive")
        if self.audit_log_retention_days <= 0:
            errors.append("audit_log_retention_days must be positive")
            
        return errors


@dataclass
class SystemConfig:
    """System-level configuration parameters."""
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = "/var/log/bioneuro/system.log"
    log_rotation_size_mb: int = 100
    log_retention_days: int = 30
    
    # Performance
    num_workers: int = 4
    max_memory_usage_mb: int = 1024
    enable_gpu: bool = True
    gpu_device_id: int = 0
    
    # Monitoring
    metrics_collection_enabled: bool = True
    metrics_collection_interval_seconds: int = 10
    health_check_interval_seconds: int = 30
    
    # Storage
    data_directory: str = "/var/lib/bioneuro"
    backup_enabled: bool = True
    backup_interval_hours: int = 6
    backup_retention_days: int = 7
    
    # Network
    bind_address: str = "0.0.0.0"
    port: int = 8080
    enable_ssl: bool = False
    ssl_cert_file: Optional[str] = None
    ssl_key_file: Optional[str] = None
    
    def validate(self) -> List[str]:
        """Validate system configuration parameters."""
        errors = []
        
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            errors.append("log_level must be a valid logging level")
        if self.log_rotation_size_mb <= 0:
            errors.append("log_rotation_size_mb must be positive")
        if self.num_workers <= 0:
            errors.append("num_workers must be positive")
        if self.max_memory_usage_mb <= 0:
            errors.append("max_memory_usage_mb must be positive")
        if not 1 <= self.port <= 65535:
            errors.append("port must be between 1 and 65535")
        if self.enable_ssl and (not self.ssl_cert_file or not self.ssl_key_file):
            errors.append("SSL enabled but certificate/key files not specified")
            
        return errors


@dataclass
class ApplicationConfig:
    """Main application configuration."""
    # Component configurations
    neural: NeuralConfig = field(default_factory=NeuralConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig) 
    security: SecurityConfig = field(default_factory=SecurityConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    # Application metadata
    version: str = "0.1.0"
    environment: str = "development"
    debug: bool = False
    
    # Custom settings
    custom: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> List[str]:
        """Validate entire application configuration."""
        all_errors = []
        
        # Validate each component
        all_errors.extend([f"neural.{err}" for err in self.neural.validate()])
        all_errors.extend([f"sensor.{err}" for err in self.sensor.validate()])
        all_errors.extend([f"security.{err}" for err in self.security.validate()])
        all_errors.extend([f"system.{err}" for err in self.system.validate()])
        
        # Cross-component validation
        if self.system.enable_gpu and not hasattr(self, '_gpu_available'):
            try:
                import torch
                if not torch.cuda.is_available():
                    all_errors.append("GPU enabled but CUDA not available")
            except ImportError:
                all_errors.append("GPU enabled but PyTorch not available")
                
        return all_errors


class ConfigManager:
    """Comprehensive configuration management system."""
    
    def __init__(self, config_dirs: Optional[List[str]] = None):
        self.config_dirs = config_dirs or [
            "/etc/bioneuro",
            "~/.config/bioneuro", 
            "./config",
            "."
        ]
        
        self._config: Optional[ApplicationConfig] = None
        self._config_lock = threading.RLock()
        self._config_watchers: List[callable] = []
        self._config_file_hash: Optional[str] = None
        self._config_file_path: Optional[Path] = None
        
    def load_config(
        self,
        config_file: Optional[str] = None,
        format: Optional[ConfigFormat] = None,
        validate: bool = True
    ) -> ApplicationConfig:
        """Load configuration from file or environment variables.
        
        Args:
            config_file: Specific config file path
            format: Configuration format (auto-detected if not specified)
            validate: Whether to validate configuration
            
        Returns:
            Loaded and validated configuration
        """
        with self._config_lock:
            if config_file:
                config_path = Path(config_file)
            else:
                config_path = self._find_config_file()
                
            if config_path and config_path.exists():
                logger.info(f"Loading configuration from: {config_path}")
                config_data = self._load_config_file(config_path, format)
                self._config_file_path = config_path
                self._config_file_hash = self._calculate_file_hash(config_path)
            else:
                logger.info("No configuration file found, using defaults")
                config_data = {}
                
            # Override with environment variables
            env_overrides = self._load_from_environment()
            config_data = self._merge_configs(config_data, env_overrides)
            
            # Create configuration object
            self._config = self._create_config_object(config_data)
            
            # Validate if requested
            if validate:
                self._validate_config(self._config)
                
            logger.info("Configuration loaded successfully")
            return self._config
            
    def get_config(self) -> ApplicationConfig:
        """Get current configuration."""
        with self._config_lock:
            if self._config is None:
                return self.load_config()
            return self._config
            
    def update_config(
        self,
        updates: Dict[str, Any],
        validate: bool = True,
        persist: bool = False
    ):
        """Update configuration at runtime.
        
        Args:
            updates: Configuration updates as nested dictionary
            validate: Whether to validate updated configuration
            persist: Whether to save changes to file
        """
        with self._config_lock:
            if self._config is None:
                raise ValueError("No configuration loaded")
                
            # Apply updates
            updated_config = deepcopy(self._config)
            self._apply_updates(updated_config, updates)
            
            # Validate if requested
            if validate:
                self._validate_config(updated_config)
                
            # Update active configuration
            old_config = self._config
            self._config = updated_config
            
            # Notify watchers
            self._notify_config_change(old_config, updated_config)
            
            # Persist if requested
            if persist and self._config_file_path:
                self.save_config(self._config_file_path)
                
            logger.info("Configuration updated successfully")
            
    def save_config(
        self,
        file_path: Union[str, Path],
        format: Optional[ConfigFormat] = None
    ):
        """Save current configuration to file.
        
        Args:
            file_path: Path to save configuration
            format: File format (auto-detected if not specified)
        """
        with self._config_lock:
            if self._config is None:
                raise ValueError("No configuration to save")
                
            file_path = Path(file_path)
            
            # Auto-detect format from extension
            if format is None:
                format = self._detect_format(file_path)
                
            # Convert config to dictionary
            config_dict = self._config_to_dict(self._config)
            
            # Save to file
            self._save_config_file(config_dict, file_path, format)
            
            # Update hash
            self._config_file_hash = self._calculate_file_hash(file_path)
            
            logger.info(f"Configuration saved to: {file_path}")
            
    def watch_config_changes(self, callback: callable):
        """Register callback for configuration changes.
        
        Args:
            callback: Function called with (old_config, new_config) when config changes
        """
        self._config_watchers.append(callback)
        
    def check_config_file_changes(self) -> bool:
        """Check if configuration file has changed on disk.
        
        Returns:
            True if file has changed, False otherwise
        """
        if not self._config_file_path or not self._config_file_path.exists():
            return False
            
        current_hash = self._calculate_file_hash(self._config_file_path)
        return current_hash != self._config_file_hash
        
    def reload_config_if_changed(self) -> bool:
        """Reload configuration if file has changed.
        
        Returns:
            True if configuration was reloaded, False otherwise
        """
        if self.check_config_file_changes():
            logger.info("Configuration file changed, reloading...")
            self.load_config()
            return True
        return False
        
    def _find_config_file(self) -> Optional[Path]:
        """Find configuration file in search directories."""
        config_names = [
            "bioneuro.toml",
            "bioneuro.yaml", 
            "bioneuro.yml",
            "bioneuro.json",
            "config.toml",
            "config.yaml",
            "config.yml", 
            "config.json"
        ]
        
        for config_dir in self.config_dirs:
            config_dir = Path(config_dir).expanduser()
            if not config_dir.exists():
                continue
                
            for config_name in config_names:
                config_path = config_dir / config_name
                if config_path.exists():
                    return config_path
                    
        return None
        
    def _load_config_file(
        self,
        file_path: Path,
        format: Optional[ConfigFormat] = None
    ) -> Dict[str, Any]:
        """Load configuration from file."""
        if format is None:
            format = self._detect_format(file_path)
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if format == ConfigFormat.JSON:
                return json.loads(content)
            elif format == ConfigFormat.YAML:
                return yaml.safe_load(content) or {}
            elif format == ConfigFormat.TOML:
                return toml.loads(content)
            else:
                raise ConfigValidationError(f"Unsupported config format: {format}")
                
        except Exception as e:
            raise ConfigValidationError(f"Failed to load config file {file_path}: {e}")
            
    def _save_config_file(
        self,
        config_data: Dict[str, Any],
        file_path: Path,
        format: ConfigFormat
    ):
        """Save configuration to file."""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                if format == ConfigFormat.JSON:
                    json.dump(config_data, f, indent=2, default=str)
                elif format == ConfigFormat.YAML:
                    yaml.safe_dump(config_data, f, default_flow_style=False)
                elif format == ConfigFormat.TOML:
                    toml.dump(config_data, f)
                else:
                    raise ConfigValidationError(f"Unsupported config format: {format}")
                    
        except Exception as e:
            raise ConfigValidationError(f"Failed to save config file {file_path}: {e}")
            
    def _detect_format(self, file_path: Path) -> ConfigFormat:
        """Detect configuration format from file extension."""
        suffix = file_path.suffix.lower()
        
        if suffix == '.json':
            return ConfigFormat.JSON
        elif suffix in ['.yaml', '.yml']:
            return ConfigFormat.YAML
        elif suffix == '.toml':
            return ConfigFormat.TOML
        else:
            # Default to YAML
            return ConfigFormat.YAML
            
    def _load_from_environment(self) -> Dict[str, Any]:
        """Load configuration overrides from environment variables."""
        env_config = {}
        prefix = "BIONEURO_"
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Convert environment variable to nested config key
                config_key = key[len(prefix):].lower()
                key_parts = config_key.split('_')
                
                # Parse value
                parsed_value = self._parse_env_value(value)
                
                # Set nested value
                current = env_config
                for part in key_parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[key_parts[-1]] = parsed_value
                
        return env_config
        
    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type."""
        # Try boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
            
        # Try integer
        try:
            return int(value)
        except ValueError:
            pass
            
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
            
        # Try JSON
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            pass
            
        # Return as string
        return value
        
    def _merge_configs(
        self,
        base_config: Dict[str, Any],
        override_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge configuration dictionaries recursively."""
        result = deepcopy(base_config)
        
        for key, value in override_config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
                
        return result
        
    def _create_config_object(self, config_data: Dict[str, Any]) -> ApplicationConfig:
        """Create ApplicationConfig object from dictionary."""
        try:
            # Extract component configurations
            neural_data = config_data.get('neural', {})
            sensor_data = config_data.get('sensor', {})
            security_data = config_data.get('security', {})
            system_data = config_data.get('system', {})
            
            # Create component config objects
            neural_config = NeuralConfig(**neural_data)
            sensor_config = SensorConfig(**sensor_data)
            security_config = SecurityConfig(**security_data)
            system_config = SystemConfig(**system_data)
            
            # Create application config
            app_config = ApplicationConfig(
                neural=neural_config,
                sensor=sensor_config,
                security=security_config,
                system=system_config,
                version=config_data.get('version', '0.1.0'),
                environment=config_data.get('environment', 'development'),
                debug=config_data.get('debug', False),
                custom=config_data.get('custom', {})
            )
            
            return app_config
            
        except Exception as e:
            raise ConfigValidationError(f"Failed to create config object: {e}")
            
    def _config_to_dict(self, config: ApplicationConfig) -> Dict[str, Any]:
        """Convert ApplicationConfig object to dictionary."""
        def dataclass_to_dict(obj):
            if hasattr(obj, '__dataclass_fields__'):
                result = {}
                for field in fields(obj):
                    value = getattr(obj, field.name)
                    if hasattr(value, '__dataclass_fields__'):
                        result[field.name] = dataclass_to_dict(value)
                    elif isinstance(value, dict):
                        result[field.name] = {k: dataclass_to_dict(v) if hasattr(v, '__dataclass_fields__') else v 
                                             for k, v in value.items()}
                    elif isinstance(value, list):
                        result[field.name] = [dataclass_to_dict(item) if hasattr(item, '__dataclass_fields__') else item 
                                             for item in value]
                    else:
                        result[field.name] = value
                return result
            else:
                return obj
                
        return dataclass_to_dict(config)
        
    def _validate_config(self, config: ApplicationConfig):
        """Validate configuration and raise exception if invalid."""
        errors = config.validate()
        if errors:
            raise ConfigValidationError(f"Configuration validation failed: {'; '.join(errors)}")
            
    def _apply_updates(self, config: ApplicationConfig, updates: Dict[str, Any]):
        """Apply configuration updates to config object."""
        for key, value in updates.items():
            if '.' in key:
                # Handle nested keys like "neural.tau_membrane"
                keys = key.split('.')
                current = config
                for k in keys[:-1]:
                    current = getattr(current, k)
                setattr(current, keys[-1], value)
            else:
                setattr(config, key, value)
                
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate hash of configuration file."""
        if not file_path.exists():
            return ""
            
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
            
    def _notify_config_change(
        self,
        old_config: ApplicationConfig,
        new_config: ApplicationConfig
    ):
        """Notify registered watchers of configuration changes."""
        for callback in self._config_watchers:
            try:
                callback(old_config, new_config)
            except Exception as e:
                logger.error(f"Config change callback failed: {e}")


# Global configuration manager instance
default_config_manager = ConfigManager()


def get_config() -> ApplicationConfig:
    """Get current application configuration."""
    return default_config_manager.get_config()


def load_config(config_file: Optional[str] = None) -> ApplicationConfig:
    """Load configuration from file."""
    return default_config_manager.load_config(config_file)


def update_config(updates: Dict[str, Any], validate: bool = True):
    """Update configuration at runtime."""
    default_config_manager.update_config(updates, validate)


def watch_config_changes(callback: callable):
    """Register callback for configuration changes."""
    default_config_manager.watch_config_changes(callback)