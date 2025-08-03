"""Electronic nose sensor array interface and management."""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class SensorSpec:
    """Specification for a single gas sensor."""
    sensor_type: str
    target_gases: List[str]
    concentration_range: Tuple[float, float]  # ppm
    response_time: float  # seconds
    sensitivity: float
    cross_sensitivity: Dict[str, float]
    temperature_coefficient: float = 0.02  # %/°C
    humidity_coefficient: float = 0.01  # %/RH%


class GasSensor(ABC):
    """Abstract base class for gas sensors."""
    
    def __init__(self, spec: SensorSpec):
        self.spec = spec
        self.is_calibrated = False
        self.baseline_reading = 0.0
        self.calibration_coefficients = [1.0, 0.0]  # linear: y = a*x + b
        
    @abstractmethod
    def read_raw(self) -> float:
        """Read raw sensor value."""
        pass
        
    def read_compensated(
        self, 
        temperature: float = 25.0, 
        humidity: float = 50.0
    ) -> float:
        """Read sensor value with environmental compensation.
        
        Args:
            temperature: Temperature in Celsius
            humidity: Relative humidity in %
            
        Returns:
            Compensated sensor reading
        """
        raw_value = self.read_raw()
        
        # Apply calibration
        calibrated = self.calibration_coefficients[0] * raw_value + self.calibration_coefficients[1]
        
        # Temperature compensation
        temp_factor = 1 + self.spec.temperature_coefficient * (temperature - 25.0) / 100.0
        
        # Humidity compensation
        humidity_factor = 1 + self.spec.humidity_coefficient * (humidity - 50.0) / 100.0
        
        compensated = calibrated / (temp_factor * humidity_factor)
        
        return max(0.0, compensated - self.baseline_reading)
        
    def calibrate(self, reference_concentrations: List[float], readings: List[float]):
        """Calibrate sensor using reference gas concentrations.
        
        Args:
            reference_concentrations: Known gas concentrations in ppm
            readings: Corresponding sensor readings
        """
        if len(reference_concentrations) != len(readings):
            raise ValueError("Concentration and reading arrays must have same length")
            
        # Linear least squares fit
        A = np.vstack([readings, np.ones(len(readings))]).T
        coeffs, _, _, _ = np.linalg.lstsq(A, reference_concentrations, rcond=None)
        
        self.calibration_coefficients = coeffs.tolist()
        self.is_calibrated = True
        
        logger.info(f"Sensor {self.spec.sensor_type} calibrated with coefficients: {coeffs}")
        
    def get_concentration(self, reading: float) -> float:
        """Convert sensor reading to gas concentration.
        
        Args:
            reading: Compensated sensor reading
            
        Returns:
            Gas concentration in ppm
        """
        if not self.is_calibrated:
            logger.warning(f"Sensor {self.spec.sensor_type} not calibrated")
            return reading  # Return raw reading if not calibrated
            
        concentration = self.calibration_coefficients[0] * reading + self.calibration_coefficients[1]
        return max(0.0, concentration)


class MOSSensor(GasSensor):
    """Metal Oxide Semiconductor sensor implementation."""
    
    def __init__(self, spec: SensorSpec, baseline_resistance: float = 10000.0):
        super().__init__(spec)
        self.baseline_resistance = baseline_resistance
        self.current_resistance = baseline_resistance
        
    def read_raw(self) -> float:
        """Simulate MOS sensor reading based on resistance change."""
        # In real implementation, this would interface with ADC
        # For simulation, add some noise to current resistance
        noise = np.random.normal(0, 0.05 * self.current_resistance)
        return self.current_resistance + noise
        
    def set_gas_concentration(self, concentration: float, gas_type: str = 'methane'):
        """Simulate sensor response to gas concentration.
        
        Args:
            concentration: Gas concentration in ppm
            gas_type: Type of gas being detected
        """
        if gas_type in self.spec.target_gases:
            # Typical MOS response: Rs/R0 = a * concentration^(-b)
            response_factor = 1.0 / (1 + 0.001 * concentration ** 0.5)
            self.current_resistance = self.baseline_resistance * response_factor
        else:
            # Cross-sensitivity response
            if gas_type in self.spec.cross_sensitivity:
                cross_response = self.spec.cross_sensitivity[gas_type]
                response_factor = 1.0 / (1 + 0.001 * concentration * cross_response ** 0.5)
                self.current_resistance = self.baseline_resistance * response_factor


class ElectrochemicalSensor(GasSensor):
    """Electrochemical sensor implementation."""
    
    def __init__(self, spec: SensorSpec, sensitivity_na_ppm: float = 1.0):
        super().__init__(spec)
        self.sensitivity_na_ppm = sensitivity_na_ppm  # nA/ppm
        self.current_output = 0.0  # nA
        
    def read_raw(self) -> float:
        """Read current output in nanoamperes."""
        # Add sensor noise
        noise = np.random.normal(0, 0.1)  # 0.1 nA noise
        return self.current_output + noise
        
    def set_gas_concentration(self, concentration: float, gas_type: str = 'CO'):
        """Simulate electrochemical sensor response.
        
        Args:
            concentration: Gas concentration in ppm
            gas_type: Type of gas being detected
        """
        if gas_type in self.spec.target_gases:
            self.current_output = concentration * self.sensitivity_na_ppm
        else:
            # Cross-sensitivity
            if gas_type in self.spec.cross_sensitivity:
                cross_response = self.spec.cross_sensitivity[gas_type]
                self.current_output = concentration * self.sensitivity_na_ppm * cross_response


class PIDSensor(GasSensor):
    """Photoionization Detector (PID) sensor implementation."""
    
    def __init__(self, spec: SensorSpec, lamp_energy: float = 10.6):
        super().__init__(spec)
        self.lamp_energy = lamp_energy  # eV
        self.current_output = 0.0  # pA
        
    def read_raw(self) -> float:
        """Read photoionization current in picoamperes."""
        # Add noise
        noise = np.random.normal(0, 0.05 * abs(self.current_output) + 0.1)
        return self.current_output + noise
        
    def set_gas_concentration(self, concentration: float, gas_type: str = 'benzene'):
        """Simulate PID sensor response.
        
        Args:
            concentration: Gas concentration in ppm
            gas_type: Type of gas being detected
        """
        # PID response depends on ionization potential of gas
        ionization_potentials = {
            'benzene': 9.2, 'toluene': 8.8, 'xylene': 8.6,
            'methane': 12.6, 'propane': 11.1, 'butane': 10.5
        }
        
        if gas_type in ionization_potentials:
            if ionization_potentials[gas_type] < self.lamp_energy:
                # Response factor based on cross-section
                response_factor = 1.0 / (ionization_potentials[gas_type] - 8.0 + 1.0)
                self.current_output = concentration * response_factor * 10.0  # pA
            else:
                self.current_output = 0.0  # Cannot ionize


class ENoseArray:
    """Electronic nose sensor array manager."""
    
    def __init__(self, sensor_configs: List[Dict]):
        self.sensors: List[GasSensor] = []
        self.sensor_names: List[str] = []
        self.sampling_rate = 1.0  # Hz
        self.is_monitoring = False
        
        # Initialize sensors
        for config in sensor_configs:
            sensor = self._create_sensor(config)
            self.sensors.append(sensor)
            self.sensor_names.append(config['name'])
            
        logger.info(f"Initialized e-nose array with {len(self.sensors)} sensors")
        
    def _create_sensor(self, config: Dict) -> GasSensor:
        """Create sensor instance from configuration."""
        spec = SensorSpec(
            sensor_type=config['type'],
            target_gases=config['target_gases'],
            concentration_range=tuple(config['range']),
            response_time=config.get('response_time', 30.0),
            sensitivity=config.get('sensitivity', 1.0),
            cross_sensitivity=config.get('cross_sensitivity', {})
        )
        
        if config['type'].startswith('MQ'):
            return MOSSensor(spec)
        elif config['type'] in ['CO_EC', 'NH3_EC', 'NO2_EC']:
            return ElectrochemicalSensor(spec)
        elif config['type'] == 'PID':
            return PIDSensor(spec)
        else:
            raise ValueError(f"Unknown sensor type: {config['type']}")
            
    def read_all_sensors(
        self, 
        temperature: float = 25.0, 
        humidity: float = 50.0
    ) -> Dict[str, float]:
        """Read all sensors with environmental compensation.
        
        Args:
            temperature: Environmental temperature in Celsius
            humidity: Relative humidity in %
            
        Returns:
            Dictionary mapping sensor names to readings
        """
        readings = {}
        
        for sensor, name in zip(self.sensors, self.sensor_names):
            try:
                reading = sensor.read_compensated(temperature, humidity)
                readings[name] = reading
            except Exception as e:
                logger.error(f"Error reading sensor {name}: {e}")
                readings[name] = 0.0
                
        return readings
        
    def read_as_tensor(
        self, 
        temperature: float = 25.0, 
        humidity: float = 50.0
    ) -> torch.Tensor:
        """Read all sensors and return as tensor.
        
        Args:
            temperature: Environmental temperature
            humidity: Relative humidity
            
        Returns:
            Tensor of sensor readings [num_sensors]
        """
        readings = self.read_all_sensors(temperature, humidity)
        values = [readings[name] for name in self.sensor_names]
        return torch.tensor(values, dtype=torch.float32)
        
    def calibrate_all(
        self, 
        reference_gas: str, 
        concentrations: List[float],
        duration_per_step: float = 300.0
    ):
        """Calibrate all sensors using reference gas.
        
        Args:
            reference_gas: Reference gas type
            concentrations: List of concentrations to use for calibration
            duration_per_step: Time to stabilize at each concentration
        """
        logger.info(f"Starting calibration with {reference_gas}")
        
        for i, concentration in enumerate(concentrations):
            logger.info(f"Calibration step {i+1}/{len(concentrations)}: {concentration} ppm")
            
            # Set gas concentration for simulation
            for sensor in self.sensors:
                if hasattr(sensor, 'set_gas_concentration'):
                    sensor.set_gas_concentration(concentration, reference_gas)
                    
            # Wait for stabilization (simulated)
            import time
            time.sleep(min(duration_per_step / 100, 3.0))  # Reduced for simulation
            
            # Collect readings
            readings = []
            for _ in range(10):  # Average multiple readings
                reading_dict = self.read_all_sensors()
                readings.append([reading_dict[name] for name in self.sensor_names])
                time.sleep(0.1)
                
            # Average readings for this concentration
            avg_readings = np.mean(readings, axis=0)
            
            # Store calibration data
            if not hasattr(self, 'calibration_data'):
                self.calibration_data = {'concentrations': [], 'readings': []}
                
            self.calibration_data['concentrations'].append(concentration)
            self.calibration_data['readings'].append(avg_readings.tolist())
            
        # Apply calibration to each sensor
        for i, sensor in enumerate(self.sensors):
            sensor_readings = [reading[i] for reading in self.calibration_data['readings']]
            sensor.calibrate(self.calibration_data['concentrations'], sensor_readings)
            
        logger.info("Calibration completed for all sensors")
        
    def get_concentration_estimates(self) -> Dict[str, float]:
        """Get gas concentration estimates from all sensors.
        
        Returns:
            Dictionary mapping sensor names to concentration estimates
        """
        raw_readings = self.read_all_sensors()
        concentrations = {}
        
        for sensor, name in zip(self.sensors, self.sensor_names):
            raw_value = raw_readings[name]
            concentration = sensor.get_concentration(raw_value)
            concentrations[name] = concentration
            
        return concentrations
        
    def start_monitoring(self, callback=None, sampling_rate: float = 1.0):
        """Start continuous monitoring.
        
        Args:
            callback: Function to call with new readings
            sampling_rate: Sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
        self.is_monitoring = True
        
        logger.info(f"Started monitoring at {sampling_rate} Hz")
        
        if callback:
            import threading
            import time
            
            def monitoring_loop():
                while self.is_monitoring:
                    readings = self.read_all_sensors()
                    callback(readings)
                    time.sleep(1.0 / self.sampling_rate)
                    
            self.monitor_thread = threading.Thread(target=monitoring_loop)
            self.monitor_thread.start()
            
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.is_monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
        logger.info("Stopped monitoring")
        
    def get_sensor_status(self) -> Dict[str, Dict]:
        """Get status information for all sensors.
        
        Returns:
            Dictionary with sensor status information
        """
        status = {}
        
        for sensor, name in zip(self.sensors, self.sensor_names):
            status[name] = {
                'type': sensor.spec.sensor_type,
                'target_gases': sensor.spec.target_gases,
                'is_calibrated': sensor.is_calibrated,
                'concentration_range': sensor.spec.concentration_range,
                'response_time': sensor.spec.response_time
            }
            
        return status
        
    def simulate_gas_exposure(
        self, 
        gas_type: str, 
        concentration: float, 
        duration: float = 60.0
    ):
        """Simulate exposure to a specific gas.
        
        Args:
            gas_type: Type of gas to simulate
            concentration: Gas concentration in ppm
            duration: Exposure duration in seconds
        """
        logger.info(f"Simulating exposure to {concentration} ppm {gas_type} for {duration}s")
        
        # Set concentration for all sensors
        for sensor in self.sensors:
            if hasattr(sensor, 'set_gas_concentration'):
                sensor.set_gas_concentration(concentration, gas_type)
                
        # Return readings during exposure
        readings_over_time = []
        num_samples = int(duration * self.sampling_rate)
        
        for _ in range(num_samples):
            readings = self.read_all_sensors()
            readings_over_time.append(readings)
            
        return readings_over_time


def create_standard_enose() -> ENoseArray:
    """Create a standard e-nose configuration for gas detection.
    
    Returns:
        Configured ENoseArray with typical sensor setup
    """
    sensor_configs = [
        {
            'name': 'MQ2_methane',
            'type': 'MQ2',
            'target_gases': ['methane', 'propane', 'butane'],
            'range': [200, 10000],
            'response_time': 30.0,
            'cross_sensitivity': {'hydrogen': 0.3, 'alcohol': 0.2}
        },
        {
            'name': 'MQ7_CO',
            'type': 'MQ7',
            'target_gases': ['carbon_monoxide'],
            'range': [10, 1000],
            'response_time': 60.0,
            'cross_sensitivity': {'hydrogen': 0.1, 'methane': 0.05}
        },
        {
            'name': 'MQ135_NH3',
            'type': 'MQ135',
            'target_gases': ['ammonia', 'benzene', 'alcohol'],
            'range': [10, 300],
            'response_time': 45.0,
            'cross_sensitivity': {'methane': 0.1, 'hydrogen': 0.2}
        },
        {
            'name': 'CO_electrochemical',
            'type': 'CO_EC',
            'target_gases': ['carbon_monoxide'],
            'range': [0, 500],
            'response_time': 15.0,
            'cross_sensitivity': {'hydrogen': 0.05}
        },
        {
            'name': 'NH3_electrochemical',
            'type': 'NH3_EC',
            'target_gases': ['ammonia'],
            'range': [0, 100],
            'response_time': 20.0,
            'cross_sensitivity': {'hydrogen_sulfide': 0.1}
        },
        {
            'name': 'PID_VOC',
            'type': 'PID',
            'target_gases': ['benzene', 'toluene', 'xylene'],
            'range': [0, 1000],
            'response_time': 5.0,
            'cross_sensitivity': {}
        }
    ]
    
    return ENoseArray(sensor_configs)
