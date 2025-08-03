"""
Adaptive threshold algorithm for dynamic sensitivity adjustment in neuromorphic gas detection.
Implements intelligent threshold adaptation based on environmental conditions and detection history.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import torch
from scipy.stats import norm
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class ThresholdParameters:
    """Parameters for adaptive threshold algorithm."""
    base_threshold: float = 0.7
    min_threshold: float = 0.4
    max_threshold: float = 0.95
    adaptation_rate: float = 0.1
    noise_sensitivity: float = 0.05
    confidence_weight: float = 0.3
    false_positive_penalty: float = 0.2
    environmental_factor: float = 0.15


@dataclass
class EnvironmentalConditions:
    """Environmental conditions affecting detection sensitivity."""
    temperature: float
    humidity: float
    air_pressure: float
    wind_speed: float
    background_noise_level: float


@dataclass
class DetectionEvent:
    """Individual detection event for threshold adaptation."""
    timestamp: datetime
    gas_type: str
    concentration: float
    confidence: float
    threshold_used: float
    confirmed: bool  # Whether detection was confirmed by human/sensor
    false_positive: bool = False


class AdaptiveThresholdManager:
    """
    Manages adaptive thresholds for neuromorphic gas detection system.
    
    The algorithm adapts detection thresholds based on:
    - Recent detection accuracy and false positive rates
    - Environmental conditions (temperature, humidity, etc.)
    - Sensor drift and calibration status
    - Historical performance patterns
    """
    
    def __init__(self, params: Optional[ThresholdParameters] = None):
        self.params = params or ThresholdParameters()
        self.detection_history: deque = deque(maxlen=1000)
        self.threshold_history: Dict[str, deque] = {}
        self.performance_metrics: Dict[str, float] = {}
        self.last_calibration: Optional[datetime] = None
        
        # Initialize thresholds for different gas types
        self.gas_thresholds: Dict[str, float] = {
            'methane': self.params.base_threshold,
            'carbon_monoxide': self.params.base_threshold * 0.9,  # More sensitive
            'hydrogen_sulfide': self.params.base_threshold * 0.8,  # Most sensitive
            'ammonia': self.params.base_threshold * 1.1,
            'volatile_organic_compounds': self.params.base_threshold,
        }
        
        # Initialize threshold history for each gas type
        for gas_type in self.gas_thresholds:
            self.threshold_history[gas_type] = deque(maxlen=100)
            
    def adapt_threshold(self, gas_type: str, 
                       environmental_conditions: EnvironmentalConditions,
                       recent_detections: List[DetectionEvent]) -> float:
        """
        Adapt threshold for specific gas type based on conditions and history.
        
        Args:
            gas_type: Type of gas being detected
            environmental_conditions: Current environmental conditions
            recent_detections: Recent detection events for learning
            
        Returns:
            Adapted threshold value
        """
        current_threshold = self.gas_thresholds.get(gas_type, self.params.base_threshold)
        
        # Calculate adaptation factors
        performance_factor = self._calculate_performance_factor(gas_type, recent_detections)
        environmental_factor = self._calculate_environmental_factor(environmental_conditions)
        drift_factor = self._calculate_drift_factor(gas_type)
        
        # Combine factors with weights
        total_adaptation = (
            performance_factor * 0.5 +
            environmental_factor * self.params.environmental_factor +
            drift_factor * 0.2
        )
        
        # Apply adaptation with rate limiting
        threshold_change = total_adaptation * self.params.adaptation_rate
        new_threshold = current_threshold + threshold_change
        
        # Clamp to valid range
        new_threshold = np.clip(
            new_threshold,
            self.params.min_threshold,
            self.params.max_threshold
        )
        
        # Update threshold and history
        self.gas_thresholds[gas_type] = new_threshold
        self.threshold_history[gas_type].append({
            'timestamp': datetime.utcnow(),
            'threshold': new_threshold,
            'adaptation': threshold_change,
            'factors': {
                'performance': performance_factor,
                'environmental': environmental_factor,
                'drift': drift_factor
            }
        })
        
        logger.debug(f"Adapted threshold for {gas_type}: {current_threshold:.3f} -> {new_threshold:.3f} "
                    f"(change: {threshold_change:+.3f})")
        
        return new_threshold
        
    def _calculate_performance_factor(self, gas_type: str, 
                                    recent_detections: List[DetectionEvent]) -> float:
        """Calculate adaptation factor based on recent detection performance."""
        if not recent_detections:
            return 0.0
            
        # Filter detections for this gas type
        gas_detections = [d for d in recent_detections if d.gas_type == gas_type]
        
        if not gas_detections:
            return 0.0
            
        # Calculate false positive rate
        false_positives = sum(1 for d in gas_detections if d.false_positive)
        fp_rate = false_positives / len(gas_detections)
        
        # Calculate missed detection rate (if we have confirmation data)
        confirmed_detections = [d for d in gas_detections if d.confirmed is not None]
        if confirmed_detections:
            missed_detections = sum(1 for d in confirmed_detections if not d.confirmed)
            miss_rate = missed_detections / len(confirmed_detections)
        else:
            miss_rate = 0.0
            
        # Calculate average confidence
        avg_confidence = np.mean([d.confidence for d in gas_detections])
        
        # Performance factor calculation
        # High false positive rate -> increase threshold
        # High miss rate -> decrease threshold
        # Low confidence -> adjust based on other factors
        
        fp_penalty = fp_rate * self.params.false_positive_penalty
        miss_penalty = -miss_rate * 0.3  # Negative to decrease threshold
        confidence_bonus = (avg_confidence - 0.5) * self.params.confidence_weight
        
        performance_factor = fp_penalty + miss_penalty + confidence_bonus
        
        return np.clip(performance_factor, -0.3, 0.3)
        
    def _calculate_environmental_factor(self, conditions: EnvironmentalConditions) -> float:
        """Calculate adaptation factor based on environmental conditions."""
        # Temperature effect (higher temp may increase volatility)
        temp_factor = 0.0
        if conditions.temperature > 30:  # Above 30°C
            temp_factor = (conditions.temperature - 30) * 0.01
        elif conditions.temperature < 10:  # Below 10°C
            temp_factor = (10 - conditions.temperature) * 0.005
            
        # Humidity effect (high humidity may affect sensor sensitivity)
        humidity_factor = 0.0
        if conditions.humidity > 70:  # High humidity
            humidity_factor = (conditions.humidity - 70) * 0.002
            
        # Air pressure effect
        pressure_factor = 0.0
        if conditions.air_pressure < 1000:  # Low pressure
            pressure_factor = (1000 - conditions.air_pressure) * 0.0001
            
        # Wind effect (high wind may dilute concentrations)
        wind_factor = 0.0
        if conditions.wind_speed > 5:  # m/s
            wind_factor = conditions.wind_speed * 0.01
            
        # Background noise effect
        noise_factor = conditions.background_noise_level * self.params.noise_sensitivity
        
        total_factor = temp_factor + humidity_factor + pressure_factor + wind_factor + noise_factor
        
        return np.clip(total_factor, -0.2, 0.2)
        
    def _calculate_drift_factor(self, gas_type: str) -> float:
        """Calculate adaptation factor based on sensor drift."""
        if not self.last_calibration:
            return 0.0
            
        # Time since last calibration
        time_since_cal = datetime.utcnow() - self.last_calibration
        days_since_cal = time_since_cal.total_seconds() / 86400
        
        # Assume gradual drift over time
        # Different gases may have different drift characteristics
        drift_rates = {
            'methane': 0.001,  # per day
            'carbon_monoxide': 0.002,
            'hydrogen_sulfide': 0.0015,
            'ammonia': 0.001,
            'volatile_organic_compounds': 0.0012,
        }
        
        drift_rate = drift_rates.get(gas_type, 0.001)
        estimated_drift = days_since_cal * drift_rate
        
        # If significant drift expected, adjust threshold
        if estimated_drift > 0.05:
            return estimated_drift * 0.5  # Partially compensate
            
        return 0.0
        
    def record_detection(self, detection: DetectionEvent):
        """Record a detection event for threshold adaptation."""
        self.detection_history.append(detection)
        
    def get_threshold(self, gas_type: str) -> float:
        """Get current threshold for gas type."""
        return self.gas_thresholds.get(gas_type, self.params.base_threshold)
        
    def update_calibration(self):
        """Update last calibration timestamp."""
        self.last_calibration = datetime.utcnow()
        logger.info("Sensor calibration timestamp updated")
        
    def get_adaptation_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics about threshold adaptations."""
        stats = {}
        
        for gas_type, history in self.threshold_history.items():
            if not history:
                continue
                
            thresholds = [h['threshold'] for h in history]
            adaptations = [h['adaptation'] for h in history]
            
            stats[gas_type] = {
                'current_threshold': self.gas_thresholds[gas_type],
                'mean_threshold': np.mean(thresholds),
                'std_threshold': np.std(thresholds),
                'min_threshold': np.min(thresholds),
                'max_threshold': np.max(thresholds),
                'mean_adaptation': np.mean(adaptations),
                'total_adaptations': len(history),
                'adaptation_range': np.max(thresholds) - np.min(thresholds),
            }
            
        return stats
        
    def reset_thresholds(self):
        """Reset all thresholds to base values."""
        for gas_type in self.gas_thresholds:
            self.gas_thresholds[gas_type] = self.params.base_threshold
            self.threshold_history[gas_type].clear()
            
        logger.info("All thresholds reset to base values")


class ThresholdOptimizer:
    """Optimize threshold parameters using historical data."""
    
    def __init__(self):
        self.optimization_history: List[Dict] = []
        
    def optimize_parameters(self, detection_history: List[DetectionEvent],
                          target_fp_rate: float = 0.05,
                          target_recall: float = 0.95) -> ThresholdParameters:
        """
        Optimize threshold parameters based on historical performance.
        
        Args:
            detection_history: Historical detection events
            target_fp_rate: Target false positive rate
            target_recall: Target recall (1 - miss rate)
            
        Returns:
            Optimized threshold parameters
        """
        if len(detection_history) < 50:
            logger.warning("Insufficient data for parameter optimization")
            return ThresholdParameters()
            
        # Analyze current performance
        current_metrics = self._analyze_performance(detection_history)
        
        # Use grid search for parameter optimization
        best_params = self._grid_search_optimization(
            detection_history, target_fp_rate, target_recall
        )
        
        # Store optimization result
        self.optimization_history.append({
            'timestamp': datetime.utcnow(),
            'data_points': len(detection_history),
            'current_metrics': current_metrics,
            'optimized_params': best_params,
            'improvement': self._calculate_improvement(current_metrics, best_params)
        })
        
        return best_params
        
    def _analyze_performance(self, detection_history: List[DetectionEvent]) -> Dict[str, float]:
        """Analyze current detection performance."""
        confirmed_detections = [d for d in detection_history if d.confirmed is not None]
        
        if not confirmed_detections:
            return {'fp_rate': 0.0, 'recall': 0.0, 'precision': 0.0}
            
        true_positives = sum(1 for d in confirmed_detections if d.confirmed and not d.false_positive)
        false_positives = sum(1 for d in confirmed_detections if d.false_positive)
        false_negatives = sum(1 for d in confirmed_detections if not d.confirmed)
        
        fp_rate = false_positives / len(confirmed_detections) if confirmed_detections else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        
        return {
            'fp_rate': fp_rate,
            'recall': recall,
            'precision': precision,
            'f1_score': 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        }
        
    def _grid_search_optimization(self, detection_history: List[DetectionEvent],
                                target_fp_rate: float, target_recall: float) -> ThresholdParameters:
        """Perform grid search to optimize parameters."""
        # Define parameter ranges
        base_threshold_range = np.arange(0.5, 0.9, 0.05)
        adaptation_rate_range = np.arange(0.05, 0.3, 0.05)
        fp_penalty_range = np.arange(0.1, 0.4, 0.05)
        
        best_score = -float('inf')
        best_params = ThresholdParameters()
        
        for base_thresh in base_threshold_range:
            for adapt_rate in adaptation_rate_range:
                for fp_penalty in fp_penalty_range:
                    params = ThresholdParameters(
                        base_threshold=base_thresh,
                        adaptation_rate=adapt_rate,
                        false_positive_penalty=fp_penalty
                    )
                    
                    # Simulate performance with these parameters
                    score = self._simulate_performance(detection_history, params, target_fp_rate, target_recall)
                    
                    if score > best_score:
                        best_score = score
                        best_params = params
                        
        return best_params
        
    def _simulate_performance(self, detection_history: List[DetectionEvent],
                            params: ThresholdParameters, target_fp_rate: float,
                            target_recall: float) -> float:
        """Simulate performance with given parameters."""
        # This would simulate the adaptive threshold algorithm
        # For now, return a simple score based on parameter values
        
        # Penalize extreme values
        threshold_penalty = abs(params.base_threshold - 0.7) * 2
        adaptation_penalty = abs(params.adaptation_rate - 0.15) * 1
        fp_penalty_score = abs(params.false_positive_penalty - 0.2) * 1
        
        score = 1.0 - threshold_penalty - adaptation_penalty - fp_penalty_score
        
        return score
        
    def _calculate_improvement(self, current_metrics: Dict[str, float],
                             optimized_params: ThresholdParameters) -> Dict[str, float]:
        """Calculate expected improvement from optimization."""
        # This would calculate expected improvement
        # For now, return placeholder values
        return {
            'fp_rate_improvement': 0.01,
            'recall_improvement': 0.02,
            'f1_improvement': 0.015
        }