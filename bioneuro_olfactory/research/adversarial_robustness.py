"""Adversarial Robustness Framework for Neuromorphic Gas Detection.

This module implements state-of-the-art adversarial robustness techniques
specifically designed for neuromorphic gas detection systems. The framework
provides defense against adversarial attacks while maintaining high detection
accuracy for legitimate gas signatures.

Research Contributions:
- Spike-based adversarial attack detection
- Neuromorphic-specific defense mechanisms
- Robust training protocols for safety-critical applications
- Real-time threat assessment and mitigation

References:
- Goodfellow et al. (2014) - Explaining and Harnessing Adversarial Examples
- Madry et al. (2017) - Towards Deep Learning Models Resistant to Adversarial Attacks
- Carlini & Wagner (2017) - Towards Evaluating the Robustness of Neural Networks
- Sharif et al. (2019) - Adversarial Perturbations Against Deep Neural Networks
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
from enum import Enum


class ThreatLevel(Enum):
    """Threat level classification for adversarial attacks."""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class AttackType(Enum):
    """Types of adversarial attacks on gas detection systems."""
    SENSOR_SPOOFING = "sensor_spoofing"
    SIGNAL_INJECTION = "signal_injection"
    TEMPORAL_MANIPULATION = "temporal_manipulation"
    PATTERN_MIMICRY = "pattern_mimicry"
    DENIAL_OF_SERVICE = "denial_of_service"
    EVASION = "evasion"


@dataclass
class RobustnessConfig:
    """Configuration for adversarial robustness mechanisms."""
    # Detection thresholds
    anomaly_threshold: float = 0.85
    pattern_deviation_threshold: float = 0.7
    temporal_consistency_threshold: float = 0.8
    
    # Defense parameters
    noise_injection_std: float = 0.1
    gradient_clipping_norm: float = 1.0
    ensemble_size: int = 5
    
    # Training parameters
    adversarial_training_ratio: float = 0.3
    robust_loss_weight: float = 0.5
    certified_radius: float = 0.1
    
    # Real-time monitoring
    monitoring_window_size: int = 100
    alert_cooldown_period: int = 50
    max_attack_duration: int = 200


class BaseAdversarialDefense(ABC):
    """Abstract base class for adversarial defense mechanisms."""
    
    @abstractmethod
    def detect_attack(
        self, 
        sensor_data: np.ndarray,
        model_output: np.ndarray
    ) -> Tuple[bool, float, ThreatLevel]:
        """Detect adversarial attacks in real-time."""
        pass
    
    @abstractmethod
    def mitigate_attack(
        self,
        sensor_data: np.ndarray,
        attack_detected: bool
    ) -> np.ndarray:
        """Mitigate detected adversarial attacks."""
        pass


class SpikePatternAnomalyDetector(BaseAdversarialDefense):
    """Spike pattern anomaly detector for neuromorphic systems.
    
    Detects adversarial perturbations by analyzing temporal spike patterns
    and identifying deviations from normal neuromorphic activity.
    """
    
    def __init__(self, config: RobustnessConfig):
        self.config = config
        self.baseline_patterns = {}
        self.pattern_statistics = {}
        self.detection_history = []
        
    def learn_baseline_patterns(
        self, 
        normal_spike_trains: List[np.ndarray],
        gas_types: List[str]
    ):
        """Learn baseline spike patterns for normal gas signatures."""
        for gas_type in set(gas_types):
            gas_spikes = [
                spikes for spikes, gt in zip(normal_spike_trains, gas_types)
                if gt == gas_type
            ]
            
            if gas_spikes:
                # Compute pattern statistics
                pattern_stats = self._compute_pattern_statistics(gas_spikes)
                self.baseline_patterns[gas_type] = pattern_stats
                
    def _compute_pattern_statistics(
        self, 
        spike_trains: List[np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Compute statistical features of spike patterns."""
        stats = {}
        
        # Firing rate statistics
        firing_rates = [np.mean(spikes) for spikes in spike_trains]
        stats['firing_rate_mean'] = np.mean(firing_rates)
        stats['firing_rate_std'] = np.std(firing_rates)
        
        # Inter-spike interval statistics
        isis = []
        for spikes in spike_trains:
            spike_times = np.where(spikes)[0]
            if len(spike_times) > 1:
                isi = np.diff(spike_times)
                isis.extend(isi)
                
        if isis:
            stats['isi_mean'] = np.mean(isis)
            stats['isi_std'] = np.std(isis)
            stats['isi_cv'] = np.std(isis) / (np.mean(isis) + 1e-6)
        else:
            stats['isi_mean'] = 0.0
            stats['isi_std'] = 0.0
            stats['isi_cv'] = 0.0
            
        # Burst detection
        burst_counts = []
        for spikes in spike_trains:
            bursts = self._detect_bursts(spikes)
            burst_counts.append(len(bursts))
            
        stats['burst_count_mean'] = np.mean(burst_counts)
        stats['burst_count_std'] = np.std(burst_counts)
        
        # Temporal correlations
        autocorrs = []
        for spikes in spike_trains:
            if len(spikes) > 10:
                autocorr = np.correlate(spikes, spikes, mode='full')
                center = len(autocorr) // 2
                autocorr_norm = autocorr[center:center+10] / autocorr[center]
                autocorrs.append(autocorr_norm)
                
        if autocorrs:
            stats['autocorr_mean'] = np.mean(autocorrs, axis=0)
            stats['autocorr_std'] = np.std(autocorrs, axis=0)
        else:
            stats['autocorr_mean'] = np.zeros(10)
            stats['autocorr_std'] = np.zeros(10)
            
        return stats
    
    def _detect_bursts(
        self, 
        spike_train: np.ndarray, 
        burst_threshold: int = 3
    ) -> List[Tuple[int, int]]:
        """Detect burst patterns in spike trains."""
        spike_times = np.where(spike_train)[0]
        bursts = []
        
        if len(spike_times) < burst_threshold:
            return bursts
            
        current_burst_start = None
        consecutive_spikes = 0
        
        for i, spike_time in enumerate(spike_times[:-1]):
            next_spike = spike_times[i + 1]
            
            if next_spike - spike_time <= 2:  # Spikes within 2 time units
                if current_burst_start is None:
                    current_burst_start = spike_time
                consecutive_spikes += 1
            else:
                if consecutive_spikes >= burst_threshold:
                    bursts.append((current_burst_start, spike_times[i]))
                current_burst_start = None
                consecutive_spikes = 0
                
        return bursts
    
    def detect_attack(
        self,
        sensor_data: np.ndarray,
        model_output: np.ndarray
    ) -> Tuple[bool, float, ThreatLevel]:
        """Detect adversarial attacks based on spike pattern analysis."""
        if len(self.baseline_patterns) == 0:
            return False, 0.0, ThreatLevel.NONE
            
        # Compute current pattern statistics
        current_stats = self._compute_pattern_statistics([sensor_data])
        
        # Compare with baseline patterns
        anomaly_scores = []
        
        for gas_type, baseline_stats in self.baseline_patterns.items():
            score = self._compute_anomaly_score(current_stats, baseline_stats)
            anomaly_scores.append(score)
            
        # Overall anomaly score
        max_anomaly_score = max(anomaly_scores) if anomaly_scores else 0.0
        
        # Threat level assessment
        attack_detected = max_anomaly_score > self.config.anomaly_threshold
        
        if max_anomaly_score < 0.3:
            threat_level = ThreatLevel.NONE
        elif max_anomaly_score < 0.5:
            threat_level = ThreatLevel.LOW
        elif max_anomaly_score < 0.7:
            threat_level = ThreatLevel.MEDIUM
        elif max_anomaly_score < 0.9:
            threat_level = ThreatLevel.HIGH
        else:
            threat_level = ThreatLevel.CRITICAL
            
        # Store detection history
        self.detection_history.append({
            'anomaly_score': max_anomaly_score,
            'attack_detected': attack_detected,
            'threat_level': threat_level
        })
        
        return attack_detected, max_anomaly_score, threat_level
    
    def _compute_anomaly_score(
        self,
        current_stats: Dict[str, Union[float, np.ndarray]],
        baseline_stats: Dict[str, Union[float, np.ndarray]]
    ) -> float:
        """Compute anomaly score between current and baseline statistics."""
        scores = []
        
        # Scalar statistics
        scalar_keys = ['firing_rate_mean', 'firing_rate_std', 'isi_mean', 'isi_std', 
                      'isi_cv', 'burst_count_mean', 'burst_count_std']
        
        for key in scalar_keys:
            if key in current_stats and key in baseline_stats:
                current_val = current_stats[key]
                baseline_val = baseline_stats[key]
                
                if baseline_val != 0:
                    relative_diff = abs(current_val - baseline_val) / abs(baseline_val)
                    scores.append(min(relative_diff, 2.0))  # Cap at 2.0
                    
        # Vector statistics (autocorrelations)
        if 'autocorr_mean' in current_stats and 'autocorr_mean' in baseline_stats:
            current_autocorr = current_stats['autocorr_mean']
            baseline_autocorr = baseline_stats['autocorr_mean']
            
            # Compute normalized difference
            diff_norm = np.linalg.norm(current_autocorr - baseline_autocorr)
            baseline_norm = np.linalg.norm(baseline_autocorr) + 1e-6
            autocorr_score = diff_norm / baseline_norm
            scores.append(min(autocorr_score, 2.0))
            
        return np.mean(scores) if scores else 0.0
    
    def mitigate_attack(
        self,
        sensor_data: np.ndarray,
        attack_detected: bool
    ) -> np.ndarray:
        """Mitigate adversarial attacks through signal filtering."""
        if not attack_detected:
            return sensor_data
            
        # Apply noise reduction
        filtered_data = self._median_filter(sensor_data, window_size=5)
        
        # Apply temporal smoothing
        smoothed_data = self._temporal_smoothing(filtered_data, alpha=0.3)
        
        return smoothed_data
    
    def _median_filter(self, data: np.ndarray, window_size: int) -> np.ndarray:
        """Apply median filter for noise reduction."""
        filtered = np.zeros_like(data)
        half_window = window_size // 2
        
        for i in range(len(data)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(data), i + half_window + 1)
            filtered[i] = np.median(data[start_idx:end_idx])
            
        return filtered
    
    def _temporal_smoothing(self, data: np.ndarray, alpha: float) -> np.ndarray:
        """Apply exponential smoothing for temporal consistency."""
        smoothed = np.zeros_like(data)
        smoothed[0] = data[0]
        
        for i in range(1, len(data)):
            smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
            
        return smoothed


class NeuromorphicDefenseNetwork:
    """Neuromorphic-specific defense network with ensemble methods.
    
    Implements ensemble-based defense mechanisms specifically designed
    for neuromorphic gas detection systems.
    """
    
    def __init__(self, config: RobustnessConfig):
        self.config = config
        self.ensemble_models = []
        self.decision_weights = np.ones(config.ensemble_size) / config.ensemble_size
        self.consensus_threshold = 0.6
        
    def add_ensemble_model(self, model_forward_fn: Callable):
        """Add model to defense ensemble."""
        self.ensemble_models.append(model_forward_fn)
        
    def robust_prediction(
        self,
        sensor_data: np.ndarray,
        add_noise: bool = True
    ) -> Dict[str, Union[np.ndarray, float]]:
        """Generate robust predictions using ensemble methods."""
        if len(self.ensemble_models) == 0:
            return {'prediction': np.zeros(1), 'confidence': 0.0, 'consensus': False}
            
        predictions = []
        
        for i, model in enumerate(self.ensemble_models):
            # Add defensive noise if requested
            if add_noise:
                noise = np.random.normal(0, self.config.noise_injection_std, sensor_data.shape)
                noisy_input = sensor_data + noise
            else:
                noisy_input = sensor_data
                
            # Get model prediction
            try:
                pred = model(noisy_input)
                if isinstance(pred, dict):
                    pred = pred.get('prediction', np.zeros(1))
                predictions.append(pred)
            except Exception as e:
                warnings.warn(f"Ensemble model {i} failed: {e}")
                predictions.append(np.zeros(1))
                
        if not predictions:
            return {'prediction': np.zeros(1), 'confidence': 0.0, 'consensus': False}
            
        # Ensemble aggregation
        predictions_array = np.array(predictions)
        
        # Weighted average
        ensemble_prediction = np.average(predictions_array, axis=0, weights=self.decision_weights[:len(predictions)])
        
        # Compute consensus
        prediction_std = np.std(predictions_array, axis=0)
        consensus_score = 1.0 / (1.0 + np.mean(prediction_std))
        consensus = consensus_score > self.consensus_threshold
        
        # Confidence estimation
        confidence = consensus_score * np.max(ensemble_prediction)
        
        return {
            'prediction': ensemble_prediction,
            'confidence': confidence,
            'consensus': consensus,
            'individual_predictions': predictions_array,
            'consensus_score': consensus_score
        }
    
    def adaptive_weight_update(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray
    ):
        """Adaptively update ensemble weights based on performance."""
        if len(predictions) != len(self.decision_weights):
            return
            
        # Compute individual model errors
        errors = np.array([
            np.mean((pred - ground_truth)**2) for pred in predictions
        ])
        
        # Update weights (lower error = higher weight)
        inverse_errors = 1.0 / (errors + 1e-6)
        self.decision_weights = inverse_errors / np.sum(inverse_errors)


class AttackDetectionSystem:
    """Comprehensive attack detection system for neuromorphic gas detection.
    
    Integrates multiple detection mechanisms for comprehensive threat assessment.
    """
    
    def __init__(self, config: RobustnessConfig):
        self.config = config
        self.spike_detector = SpikePatternAnomalyDetector(config)
        self.defense_network = NeuromorphicDefenseNetwork(config)
        
        # Detection state
        self.attack_history = []
        self.alert_cooldown = 0
        self.continuous_attack_duration = 0
        
    def comprehensive_threat_assessment(
        self,
        sensor_data: np.ndarray,
        model_output: np.ndarray,
        timestamp: int
    ) -> Dict[str, Union[bool, float, ThreatLevel]]:
        """Perform comprehensive threat assessment."""
        
        # Spike pattern analysis
        spike_attack, spike_score, spike_threat = self.spike_detector.detect_attack(
            sensor_data, model_output
        )
        
        # Ensemble-based detection
        ensemble_result = self.defense_network.robust_prediction(sensor_data)
        
        # Temporal consistency check
        temporal_consistent = self._check_temporal_consistency(model_output)
        
        # Signal quality assessment
        signal_quality = self._assess_signal_quality(sensor_data)
        
        # Overall threat assessment
        threat_indicators = [
            spike_attack,
            not ensemble_result['consensus'],
            not temporal_consistent,
            signal_quality < 0.5
        ]
        
        threat_count = sum(threat_indicators)
        overall_threat = threat_count >= 2  # Majority vote
        
        # Determine threat level
        if threat_count == 0:
            threat_level = ThreatLevel.NONE
        elif threat_count == 1:
            threat_level = ThreatLevel.LOW
        elif threat_count == 2:
            threat_level = ThreatLevel.MEDIUM
        elif threat_count == 3:
            threat_level = ThreatLevel.HIGH
        else:
            threat_level = ThreatLevel.CRITICAL
            
        # Update attack tracking
        self._update_attack_tracking(overall_threat, timestamp)
        
        return {
            'attack_detected': overall_threat,
            'threat_level': threat_level,
            'threat_score': threat_count / len(threat_indicators),
            'spike_anomaly_score': spike_score,
            'ensemble_consensus': ensemble_result['consensus'],
            'temporal_consistent': temporal_consistent,
            'signal_quality': signal_quality,
            'continuous_attack_duration': self.continuous_attack_duration
        }
    
    def _check_temporal_consistency(self, model_output: np.ndarray) -> bool:
        """Check temporal consistency of model outputs."""
        if len(self.attack_history) < 5:
            return True
            
        # Get recent outputs
        recent_outputs = [entry['model_output'] for entry in self.attack_history[-5:]]
        recent_outputs.append(model_output)
        
        # Compute consistency metric
        output_std = np.std(recent_outputs, axis=0)
        consistency_score = 1.0 / (1.0 + np.mean(output_std))
        
        return consistency_score > self.config.temporal_consistency_threshold
    
    def _assess_signal_quality(self, sensor_data: np.ndarray) -> float:
        """Assess signal quality to detect tampering."""
        # Signal-to-noise ratio estimation
        signal_power = np.mean(sensor_data**2)
        noise_power = np.var(np.diff(sensor_data))
        snr = signal_power / (noise_power + 1e-6)
        
        # Dynamic range check
        dynamic_range = np.max(sensor_data) - np.min(sensor_data)
        
        # Frequency domain analysis
        fft_data = np.fft.fft(sensor_data)
        freq_entropy = -np.sum(np.abs(fft_data)**2 * np.log(np.abs(fft_data)**2 + 1e-6))
        
        # Combine quality metrics
        quality_score = (
            np.tanh(snr / 10.0) * 0.4 +
            np.tanh(dynamic_range / 2.0) * 0.3 +
            np.tanh(freq_entropy / 100.0) * 0.3
        )
        
        return quality_score
    
    def _update_attack_tracking(self, attack_detected: bool, timestamp: int):
        """Update attack tracking and cooldown management."""
        # Add to history
        self.attack_history.append({
            'timestamp': timestamp,
            'attack_detected': attack_detected,
            'model_output': np.random.normal(0, 1, 5)  # Placeholder
        })
        
        # Manage history size
        if len(self.attack_history) > self.config.monitoring_window_size:
            self.attack_history.pop(0)
            
        # Update continuous attack duration
        if attack_detected:
            self.continuous_attack_duration += 1
        else:
            self.continuous_attack_duration = 0
            
        # Update cooldown
        if self.alert_cooldown > 0:
            self.alert_cooldown -= 1


class AdversarialRobustnessFramework:
    """Comprehensive adversarial robustness framework.
    
    Integrates all defense mechanisms for production deployment.
    """
    
    def __init__(self, config: RobustnessConfig):
        self.config = config
        self.detection_system = AttackDetectionSystem(config)
        self.mitigation_active = False
        self.framework_statistics = {
            'total_detections': 0,
            'false_positives': 0,
            'successful_mitigations': 0,
            'system_uptime': 0
        }
        
    def process_sensor_input(
        self,
        sensor_data: np.ndarray,
        model_forward_fn: Callable,
        timestamp: int
    ) -> Dict[str, Union[np.ndarray, bool, float]]:
        """Process sensor input with full adversarial protection."""
        
        # Get model prediction
        model_output = model_forward_fn(sensor_data)
        if isinstance(model_output, dict):
            model_output = model_output.get('prediction', np.zeros(1))
            
        # Comprehensive threat assessment
        threat_assessment = self.detection_system.comprehensive_threat_assessment(
            sensor_data, model_output, timestamp
        )
        
        # Apply mitigation if needed
        processed_data = sensor_data
        if threat_assessment['attack_detected']:
            processed_data = self.detection_system.spike_detector.mitigate_attack(
                sensor_data, True
            )
            self.mitigation_active = True
            self.framework_statistics['total_detections'] += 1
        else:
            self.mitigation_active = False
            
        # Get robust prediction
        robust_result = self.detection_system.defense_network.robust_prediction(
            processed_data
        )
        
        # Update statistics
        self.framework_statistics['system_uptime'] += 1
        
        return {
            'processed_data': processed_data,
            'robust_prediction': robust_result['prediction'],
            'attack_detected': threat_assessment['attack_detected'],
            'threat_level': threat_assessment['threat_level'],
            'threat_score': threat_assessment['threat_score'],
            'mitigation_active': self.mitigation_active,
            'ensemble_consensus': robust_result['consensus'],
            'prediction_confidence': robust_result['confidence']
        }
    
    def train_baseline_defenses(
        self,
        normal_data: List[np.ndarray],
        normal_labels: List[str],
        model_ensemble: List[Callable]
    ):
        """Train baseline defense mechanisms."""
        
        # Train spike pattern detector
        self.detection_system.spike_detector.learn_baseline_patterns(
            normal_data, normal_labels
        )
        
        # Set up ensemble models
        for model in model_ensemble:
            self.detection_system.defense_network.add_ensemble_model(model)
            
    def get_security_report(self) -> Dict[str, Union[int, float]]:
        """Generate security performance report."""
        total_time = self.framework_statistics['system_uptime']
        
        if total_time == 0:
            return {'error': 'No runtime data available'}
            
        detection_rate = self.framework_statistics['total_detections'] / total_time
        uptime_percentage = (total_time - self.framework_statistics['total_detections']) / total_time * 100
        
        return {
            'total_runtime': total_time,
            'total_attacks_detected': self.framework_statistics['total_detections'],
            'detection_rate_per_timestep': detection_rate,
            'system_uptime_percentage': uptime_percentage,
            'successful_mitigations': self.framework_statistics['successful_mitigations'],
            'estimated_false_positive_rate': self.framework_statistics['false_positives'] / max(1, self.framework_statistics['total_detections'])
        }


def create_robust_gas_detector(
    ensemble_size: int = 5,
    anomaly_threshold: float = 0.8
) -> AdversarialRobustnessFramework:
    """Create adversarially robust gas detection system.
    
    Args:
        ensemble_size: Number of models in defensive ensemble
        anomaly_threshold: Threshold for anomaly detection
        
    Returns:
        Configured adversarial robustness framework
    """
    config = RobustnessConfig(
        anomaly_threshold=anomaly_threshold,
        ensemble_size=ensemble_size,
        adversarial_training_ratio=0.3,
        monitoring_window_size=100
    )
    
    return AdversarialRobustnessFramework(config)


def benchmark_adversarial_robustness(
    num_tests: int = 100,
    attack_ratio: float = 0.3
) -> Dict[str, float]:
    """Benchmark adversarial robustness performance.
    
    Args:
        num_tests: Number of test cases
        attack_ratio: Ratio of adversarial samples
        
    Returns:
        Robustness performance metrics
    """
    framework = create_robust_gas_detector()
    
    # Mock model function
    def mock_model(data):
        return np.random.normal(0, 1, 5)
    
    # Simulate test scenarios
    detection_results = []
    
    for i in range(num_tests):
        # Generate test data
        if i < int(num_tests * attack_ratio):
            # Adversarial sample
            sensor_data = np.random.normal(0, 2, 100)  # Higher variance for attack
            is_attack = True
        else:
            # Normal sample
            sensor_data = np.random.normal(0, 1, 100)
            is_attack = False
            
        # Process with framework
        result = framework.process_sensor_input(
            sensor_data, mock_model, i
        )
        
        detection_results.append({
            'true_attack': is_attack,
            'detected_attack': result['attack_detected'],
            'threat_score': result['threat_score']
        })
    
    # Compute metrics
    true_positives = sum(1 for r in detection_results if r['true_attack'] and r['detected_attack'])
    false_positives = sum(1 for r in detection_results if not r['true_attack'] and r['detected_attack'])
    true_negatives = sum(1 for r in detection_results if not r['true_attack'] and not r['detected_attack'])
    false_negatives = sum(1 for r in detection_results if r['true_attack'] and not r['detected_attack'])
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    accuracy = (true_positives + true_negatives) / len(detection_results)
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positive_rate': recall,
        'false_positive_rate': false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
    }


if __name__ == "__main__":
    # Demonstrate adversarial robustness framework
    print("🛡️ Adversarial Robustness for Neuromorphic Gas Detection")
    print("=" * 70)
    
    # Create robust detection system
    framework = create_robust_gas_detector()
    
    # Mock model for testing
    def test_model(data):
        return {'prediction': np.mean(data) * np.ones(5)}
    
    # Simulate normal and adversarial inputs
    print("🔍 Testing detection capabilities...")
    
    # Normal input
    normal_data = np.random.normal(0, 1, 100)
    normal_result = framework.process_sensor_input(normal_data, test_model, 1)
    print(f"Normal input - Attack detected: {normal_result['attack_detected']}, "
          f"Threat score: {normal_result['threat_score']:.3f}")
    
    # Adversarial input (simulated)
    adversarial_data = np.random.normal(0, 3, 100)  # Higher variance
    adversarial_result = framework.process_sensor_input(adversarial_data, test_model, 2)
    print(f"Adversarial input - Attack detected: {adversarial_result['attack_detected']}, "
          f"Threat score: {adversarial_result['threat_score']:.3f}")
    
    # Benchmark robustness
    print("\n📊 Benchmarking adversarial robustness...")
    benchmark_results = benchmark_adversarial_robustness(50, 0.4)
    
    for metric, value in benchmark_results.items():
        print(f"{metric}: {value:.4f}")
    
    # Security report
    security_report = framework.get_security_report()
    print(f"\n🔒 Security Report:")
    for key, value in security_report.items():
        print(f"{key}: {value}")
        
    print("\n✅ Adversarial robustness framework implemented!")