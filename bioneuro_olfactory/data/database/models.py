"""Data models for the BioNeuro-Olfactory-Fusion system."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
import numpy as np


@dataclass
class ExperimentModel:
    """Experiment data model."""
    name: str
    description: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    status: str = "created"  # created, running, completed, failed
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'config': self.config,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentModel':
        """Create from dictionary representation."""
        return cls(
            id=data.get('id'),
            name=data['name'],
            description=data.get('description', ''),
            config=data.get('config', {}),
            status=data.get('status', 'created'),
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else None,
            updated_at=datetime.fromisoformat(data['updated_at']) if data.get('updated_at') else None
        )


@dataclass
class SensorDataModel:
    """Sensor reading data model."""
    experiment_id: int
    sensor_type: str
    sensor_id: str
    raw_value: float
    calibrated_value: float
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[int] = None
    timestamp: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'id': self.id,
            'experiment_id': self.experiment_id,
            'sensor_type': self.sensor_type,
            'sensor_id': self.sensor_id,
            'raw_value': self.raw_value,
            'calibrated_value': self.calibrated_value,
            'temperature': self.temperature,
            'humidity': self.humidity,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SensorDataModel':
        """Create from dictionary representation."""
        return cls(
            id=data.get('id'),
            experiment_id=data['experiment_id'],
            sensor_type=data['sensor_type'],
            sensor_id=data['sensor_id'],
            raw_value=data['raw_value'],
            calibrated_value=data['calibrated_value'],
            temperature=data.get('temperature'),
            humidity=data.get('humidity'),
            metadata=data.get('metadata', {}),
            timestamp=datetime.fromisoformat(data['timestamp']) if data.get('timestamp') else None
        )


@dataclass
class NetworkStateModel:
    """Network state snapshot data model."""
    experiment_id: int
    network_type: str
    layer_name: str
    state_data: np.ndarray
    sparsity_level: Optional[float] = None
    firing_rate: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[int] = None
    timestamp: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'id': self.id,
            'experiment_id': self.experiment_id,
            'network_type': self.network_type,
            'layer_name': self.layer_name,
            'sparsity_level': self.sparsity_level,
            'firing_rate': self.firing_rate,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'state_shape': self.state_data.shape if self.state_data is not None else None,
            'state_dtype': str(self.state_data.dtype) if self.state_data is not None else None
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any], state_data: np.ndarray = None) -> 'NetworkStateModel':
        """Create from dictionary representation."""
        return cls(
            id=data.get('id'),
            experiment_id=data['experiment_id'],
            network_type=data['network_type'],
            layer_name=data['layer_name'],
            state_data=state_data,
            sparsity_level=data.get('sparsity_level'),
            firing_rate=data.get('firing_rate'),
            metadata=data.get('metadata', {}),
            timestamp=datetime.fromisoformat(data['timestamp']) if data.get('timestamp') else None
        )


@dataclass
class GasDetectionEventModel:
    """Gas detection event data model."""
    experiment_id: int
    gas_type: str
    concentration: float
    confidence: float
    alert_level: str = "info"  # info, warning, critical
    response_time: Optional[float] = None
    sensor_fusion_method: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[int] = None
    timestamp: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'id': self.id,
            'experiment_id': self.experiment_id,
            'gas_type': self.gas_type,
            'concentration': self.concentration,
            'confidence': self.confidence,
            'alert_level': self.alert_level,
            'response_time': self.response_time,
            'sensor_fusion_method': self.sensor_fusion_method,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GasDetectionEventModel':
        """Create from dictionary representation."""
        return cls(
            id=data.get('id'),
            experiment_id=data['experiment_id'],
            gas_type=data['gas_type'],
            concentration=data['concentration'],
            confidence=data['confidence'],
            alert_level=data.get('alert_level', 'info'),
            response_time=data.get('response_time'),
            sensor_fusion_method=data.get('sensor_fusion_method'),
            metadata=data.get('metadata', {}),
            timestamp=datetime.fromisoformat(data['timestamp']) if data.get('timestamp') else None
        )
        
    def is_critical(self) -> bool:
        """Check if this is a critical detection event."""
        return self.alert_level == "critical" or self.confidence > 0.95
        
    def get_alert_message(self) -> str:
        """Generate alert message for this detection."""
        return (
            f"Gas Detection Alert: {self.gas_type.upper()} detected at "
            f"{self.concentration:.1f} ppm (confidence: {self.confidence:.2%})"
        )


@dataclass
class ModelCheckpointModel:
    """Model checkpoint data model."""
    experiment_id: int
    checkpoint_name: str
    model_data: bytes
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'id': self.id,
            'experiment_id': self.experiment_id,
            'checkpoint_name': self.checkpoint_name,
            'performance_metrics': self.performance_metrics,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'model_size_bytes': len(self.model_data) if self.model_data else 0
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any], model_data: bytes = None) -> 'ModelCheckpointModel':
        """Create from dictionary representation."""
        return cls(
            id=data.get('id'),
            experiment_id=data['experiment_id'],
            checkpoint_name=data['checkpoint_name'],
            model_data=model_data or b'',
            performance_metrics=data.get('performance_metrics', {}),
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else None
        )


@dataclass
class DatasetMetadata:
    """Dataset metadata model."""
    name: str
    description: str
    version: str = "1.0.0"
    source: str = "unknown"
    gases: List[str] = field(default_factory=list)
    sensors: List[str] = field(default_factory=list)
    sample_count: int = 0
    duration_seconds: float = 0.0
    sampling_rate_hz: float = 1.0
    features: Dict[str, Any] = field(default_factory=dict)
    preprocessing_steps: List[str] = field(default_factory=list)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'source': self.source,
            'gases': self.gases,
            'sensors': self.sensors,
            'sample_count': self.sample_count,
            'duration_seconds': self.duration_seconds,
            'sampling_rate_hz': self.sampling_rate_hz,
            'features': self.features,
            'preprocessing_steps': self.preprocessing_steps,
            'quality_metrics': self.quality_metrics,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetMetadata':
        """Create from dictionary representation."""
        return cls(
            name=data['name'],
            description=data.get('description', ''),
            version=data.get('version', '1.0.0'),
            source=data.get('source', 'unknown'),
            gases=data.get('gases', []),
            sensors=data.get('sensors', []),
            sample_count=data.get('sample_count', 0),
            duration_seconds=data.get('duration_seconds', 0.0),
            sampling_rate_hz=data.get('sampling_rate_hz', 1.0),
            features=data.get('features', {}),
            preprocessing_steps=data.get('preprocessing_steps', []),
            quality_metrics=data.get('quality_metrics', {}),
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else None
        )


@dataclass
class CalibrationRecord:
    """Sensor calibration record model."""
    sensor_type: str
    sensor_id: str
    reference_gas: str
    reference_concentrations: List[float]
    sensor_readings: List[float]
    calibration_coefficients: List[float]
    r_squared: float
    temperature: float
    humidity: float
    calibration_date: datetime
    expiry_date: Optional[datetime] = None
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'sensor_type': self.sensor_type,
            'sensor_id': self.sensor_id,
            'reference_gas': self.reference_gas,
            'reference_concentrations': self.reference_concentrations,
            'sensor_readings': self.sensor_readings,
            'calibration_coefficients': self.calibration_coefficients,
            'r_squared': self.r_squared,
            'temperature': self.temperature,
            'humidity': self.humidity,
            'calibration_date': self.calibration_date.isoformat(),
            'expiry_date': self.expiry_date.isoformat() if self.expiry_date else None,
            'notes': self.notes
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CalibrationRecord':
        """Create from dictionary representation."""
        return cls(
            sensor_type=data['sensor_type'],
            sensor_id=data['sensor_id'],
            reference_gas=data['reference_gas'],
            reference_concentrations=data['reference_concentrations'],
            sensor_readings=data['sensor_readings'],
            calibration_coefficients=data['calibration_coefficients'],
            r_squared=data['r_squared'],
            temperature=data['temperature'],
            humidity=data['humidity'],
            calibration_date=datetime.fromisoformat(data['calibration_date']),
            expiry_date=datetime.fromisoformat(data['expiry_date']) if data.get('expiry_date') else None,
            notes=data.get('notes', '')
        )
        
    def is_expired(self) -> bool:
        """Check if calibration is expired."""
        if not self.expiry_date:
            return False
        return datetime.now() > self.expiry_date
        
    def days_until_expiry(self) -> int:
        """Get days until calibration expires."""
        if not self.expiry_date:
            return -1
        delta = self.expiry_date - datetime.now()
        return max(0, delta.days)