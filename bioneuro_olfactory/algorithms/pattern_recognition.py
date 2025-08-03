"""
Advanced pattern recognition algorithms for neuromorphic gas detection.
Implements sophisticated pattern matching, anomaly detection, and temporal analysis.
"""

import numpy as np
import torch
import torch.nn as nn
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy import signal
from scipy.stats import entropy
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class SpikePattern:
    """Represents a detected spike pattern in neural activity."""
    pattern_id: str
    spike_times: np.ndarray
    neuron_indices: np.ndarray
    duration_ms: float
    frequency_hz: float
    spatial_distribution: np.ndarray
    confidence: float
    gas_association: Optional[str] = None


@dataclass
class TemporalSignature:
    """Temporal signature of gas detection events."""
    gas_type: str
    onset_pattern: np.ndarray
    steady_state_pattern: np.ndarray
    decay_pattern: np.ndarray
    characteristic_frequency: float
    burst_intervals: List[float]
    pattern_entropy: float


class SpikeTrainAnalyzer:
    """Analyzes spike trains for characteristic patterns."""
    
    def __init__(self, sampling_rate: float = 1000.0):
        self.sampling_rate = sampling_rate
        self.pattern_library: Dict[str, List[SpikePattern]] = {}
        self.temporal_signatures: Dict[str, TemporalSignature] = {}
        
    def analyze_spike_train(self, spike_times: np.ndarray, 
                           neuron_indices: np.ndarray,
                           duration: float) -> List[SpikePattern]:
        """
        Analyze spike train for characteristic patterns.
        
        Args:
            spike_times: Array of spike timestamps
            neuron_indices: Array of neuron indices corresponding to spikes
            duration: Total duration of recording in seconds
            
        Returns:
            List of detected spike patterns
        """
        patterns = []
        
        # 1. Detect burst patterns
        burst_patterns = self._detect_burst_patterns(spike_times, neuron_indices)
        patterns.extend(burst_patterns)
        
        # 2. Detect synchronization patterns
        sync_patterns = self._detect_synchronization_patterns(spike_times, neuron_indices)
        patterns.extend(sync_patterns)
        
        # 3. Detect oscillatory patterns
        osc_patterns = self._detect_oscillatory_patterns(spike_times, neuron_indices, duration)
        patterns.extend(osc_patterns)
        
        # 4. Detect spatial patterns
        spatial_patterns = self._detect_spatial_patterns(spike_times, neuron_indices)
        patterns.extend(spatial_patterns)
        
        return patterns
        
    def _detect_burst_patterns(self, spike_times: np.ndarray, 
                              neuron_indices: np.ndarray) -> List[SpikePattern]:
        """Detect burst patterns in spike trains."""
        patterns = []
        
        # Group spikes by neuron
        unique_neurons = np.unique(neuron_indices)
        
        for neuron_id in unique_neurons:
            neuron_spikes = spike_times[neuron_indices == neuron_id]
            
            if len(neuron_spikes) < 3:
                continue
                
            # Calculate inter-spike intervals
            isi = np.diff(neuron_spikes)
            
            # Detect bursts using ISI threshold
            burst_threshold = np.percentile(isi, 25)  # 25th percentile as threshold
            burst_mask = isi < burst_threshold
            
            # Find burst boundaries
            burst_starts = np.where(np.diff(np.concatenate(([False], burst_mask))) == 1)[0]
            burst_ends = np.where(np.diff(np.concatenate((burst_mask, [False]))) == -1)[0]
            
            for start_idx, end_idx in zip(burst_starts, burst_ends):
                if end_idx - start_idx >= 2:  # At least 3 spikes in burst
                    burst_spikes = neuron_spikes[start_idx:end_idx+2]
                    duration_ms = (burst_spikes[-1] - burst_spikes[0]) * 1000
                    frequency = len(burst_spikes) / (duration_ms / 1000)
                    
                    # Create spatial distribution (single neuron)
                    spatial_dist = np.zeros(len(unique_neurons))
                    spatial_dist[np.where(unique_neurons == neuron_id)[0]] = 1.0
                    
                    pattern = SpikePattern(
                        pattern_id=f"burst_{neuron_id}_{start_idx}",
                        spike_times=burst_spikes,
                        neuron_indices=np.full(len(burst_spikes), neuron_id),
                        duration_ms=duration_ms,
                        frequency_hz=frequency,
                        spatial_distribution=spatial_dist,
                        confidence=self._calculate_burst_confidence(burst_spikes, isi[start_idx:end_idx+1])
                    )
                    patterns.append(pattern)
                    
        return patterns
        
    def _detect_synchronization_patterns(self, spike_times: np.ndarray,
                                       neuron_indices: np.ndarray) -> List[SpikePattern]:
        """Detect synchronized firing patterns across neurons."""
        patterns = []
        
        # Time binning for synchrony detection
        bin_size = 0.01  # 10ms bins
        max_time = np.max(spike_times) if len(spike_times) > 0 else 1.0
        bins = np.arange(0, max_time + bin_size, bin_size)
        
        # Create binned spike matrix
        unique_neurons = np.unique(neuron_indices)
        spike_matrix = np.zeros((len(unique_neurons), len(bins) - 1))
        
        for i, neuron_id in enumerate(unique_neurons):
            neuron_spikes = spike_times[neuron_indices == neuron_id]
            counts, _ = np.histogram(neuron_spikes, bins)
            spike_matrix[i, :] = counts
            
        # Detect synchronous events (multiple neurons firing in same bin)
        sync_threshold = max(2, len(unique_neurons) * 0.3)  # At least 30% of neurons
        sync_bins = np.where(np.sum(spike_matrix > 0, axis=0) >= sync_threshold)[0]
        
        # Group consecutive synchronous bins
        if len(sync_bins) > 0:
            sync_groups = []
            current_group = [sync_bins[0]]
            
            for i in range(1, len(sync_bins)):
                if sync_bins[i] - sync_bins[i-1] <= 2:  # Allow 1 bin gap
                    current_group.append(sync_bins[i])
                else:
                    if len(current_group) >= 2:
                        sync_groups.append(current_group)
                    current_group = [sync_bins[i]]
                    
            if len(current_group) >= 2:
                sync_groups.append(current_group)
                
            # Create patterns for each synchronous group
            for group_idx, group in enumerate(sync_groups):
                start_time = bins[group[0]]
                end_time = bins[group[-1] + 1]
                
                # Find spikes in this time window
                window_mask = (spike_times >= start_time) & (spike_times <= end_time)
                window_spikes = spike_times[window_mask]
                window_neurons = neuron_indices[window_mask]
                
                if len(window_spikes) >= sync_threshold:
                    duration_ms = (end_time - start_time) * 1000
                    frequency = len(window_spikes) / (duration_ms / 1000)
                    
                    # Spatial distribution
                    spatial_dist = np.zeros(len(unique_neurons))
                    for neuron_id in np.unique(window_neurons):
                        neuron_idx = np.where(unique_neurons == neuron_id)[0]
                        spatial_dist[neuron_idx] = 1.0
                        
                    pattern = SpikePattern(
                        pattern_id=f"sync_{group_idx}",
                        spike_times=window_spikes,
                        neuron_indices=window_neurons,
                        duration_ms=duration_ms,
                        frequency_hz=frequency,
                        spatial_distribution=spatial_dist,
                        confidence=len(np.unique(window_neurons)) / len(unique_neurons)
                    )
                    patterns.append(pattern)
                    
        return patterns
        
    def _detect_oscillatory_patterns(self, spike_times: np.ndarray,
                                   neuron_indices: np.ndarray,
                                   duration: float) -> List[SpikePattern]:
        """Detect oscillatory patterns in population activity."""
        patterns = []
        
        if len(spike_times) < 10:
            return patterns
            
        # Create population spike rate over time
        bin_size = 0.005  # 5ms bins for good frequency resolution
        bins = np.arange(0, duration + bin_size, bin_size)
        spike_rate, _ = np.histogram(spike_times, bins)
        
        # Apply smoothing
        from scipy.ndimage import gaussian_filter1d
        smooth_rate = gaussian_filter1d(spike_rate.astype(float), sigma=2.0)
        
        # Compute power spectral density
        freqs, psd = signal.welch(smooth_rate, fs=1/bin_size, nperseg=min(256, len(smooth_rate)//4))
        
        # Find dominant frequencies
        peak_indices, properties = signal.find_peaks(psd, height=np.max(psd) * 0.1, distance=5)
        dominant_freqs = freqs[peak_indices]
        peak_powers = psd[peak_indices]
        
        # Create patterns for significant oscillations
        for freq, power in zip(dominant_freqs, peak_powers):
            if 1 <= freq <= 100:  # Biologically relevant range
                # Find time periods with strong oscillation at this frequency
                # Use bandpass filter around the frequency
                from scipy.signal import butter, filtfilt
                
                nyquist = 0.5 / bin_size
                low = max(freq - 2, 0.5) / nyquist
                high = min(freq + 2, nyquist - 0.1)
                
                if low < high:
                    b, a = butter(4, [low, high], btype='band')
                    filtered_signal = filtfilt(b, a, smooth_rate)
                    
                    # Find periods of high oscillatory power
                    envelope = np.abs(signal.hilbert(filtered_signal))
                    osc_threshold = np.percentile(envelope, 75)
                    osc_periods = envelope > osc_threshold
                    
                    # Group consecutive oscillatory periods
                    osc_starts = np.where(np.diff(np.concatenate(([False], osc_periods))) == 1)[0]
                    osc_ends = np.where(np.diff(np.concatenate((osc_periods, [False]))) == -1)[0]
                    
                    for start_idx, end_idx in zip(osc_starts, osc_ends):
                        if end_idx - start_idx >= 10:  # At least 10 bins (50ms)
                            start_time = bins[start_idx]
                            end_time = bins[end_idx]
                            
                            # Find spikes in this oscillatory period
                            period_mask = (spike_times >= start_time) & (spike_times <= end_time)
                            period_spikes = spike_times[period_mask]
                            period_neurons = neuron_indices[period_mask]
                            
                            if len(period_spikes) >= 5:
                                duration_ms = (end_time - start_time) * 1000
                                
                                # Spatial distribution
                                unique_neurons = np.unique(neuron_indices)
                                spatial_dist = np.zeros(len(unique_neurons))
                                for neuron_id in np.unique(period_neurons):
                                    neuron_idx = np.where(unique_neurons == neuron_id)[0]
                                    spatial_dist[neuron_idx] = 1.0
                                    
                                pattern = SpikePattern(
                                    pattern_id=f"osc_{freq:.1f}Hz_{start_idx}",
                                    spike_times=period_spikes,
                                    neuron_indices=period_neurons,
                                    duration_ms=duration_ms,
                                    frequency_hz=freq,
                                    spatial_distribution=spatial_dist,
                                    confidence=power / np.max(psd)
                                )
                                patterns.append(pattern)
                                
        return patterns
        
    def _detect_spatial_patterns(self, spike_times: np.ndarray,
                               neuron_indices: np.ndarray) -> List[SpikePattern]:
        """Detect spatial patterns in neural activity."""
        patterns = []
        
        unique_neurons = np.unique(neuron_indices)
        if len(unique_neurons) < 3:
            return patterns
            
        # Calculate pairwise correlations
        bin_size = 0.02  # 20ms bins
        max_time = np.max(spike_times) if len(spike_times) > 0 else 1.0
        bins = np.arange(0, max_time + bin_size, bin_size)
        
        # Create spike count matrix
        spike_matrix = np.zeros((len(unique_neurons), len(bins) - 1))
        for i, neuron_id in enumerate(unique_neurons):
            neuron_spikes = spike_times[neuron_indices == neuron_id]
            counts, _ = np.histogram(neuron_spikes, bins)
            spike_matrix[i, :] = counts
            
        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(spike_matrix)
        correlation_matrix = np.nan_to_num(correlation_matrix)
        
        # Find highly correlated neuron groups using clustering
        # Convert correlation to distance
        distance_matrix = 1 - np.abs(correlation_matrix)
        
        # Use DBSCAN clustering
        clustering = DBSCAN(eps=0.3, min_samples=2, metric='precomputed')
        cluster_labels = clustering.fit_predict(distance_matrix)
        
        # Create patterns for each cluster
        for cluster_id in np.unique(cluster_labels):
            if cluster_id == -1:  # Noise cluster
                continue
                
            cluster_neurons = unique_neurons[cluster_labels == cluster_id]
            if len(cluster_neurons) < 2:
                continue
                
            # Find spikes from neurons in this cluster
            cluster_mask = np.isin(neuron_indices, cluster_neurons)
            cluster_spikes = spike_times[cluster_mask]
            cluster_neuron_indices = neuron_indices[cluster_mask]
            
            if len(cluster_spikes) >= 10:
                duration_ms = (np.max(cluster_spikes) - np.min(cluster_spikes)) * 1000
                frequency = len(cluster_spikes) / (duration_ms / 1000) if duration_ms > 0 else 0
                
                # Spatial distribution
                spatial_dist = np.zeros(len(unique_neurons))
                for neuron_id in cluster_neurons:
                    neuron_idx = np.where(unique_neurons == neuron_id)[0]
                    spatial_dist[neuron_idx] = 1.0
                    
                # Calculate cluster coherence as confidence
                cluster_correlations = correlation_matrix[cluster_labels == cluster_id][:, cluster_labels == cluster_id]
                coherence = np.mean(cluster_correlations[cluster_correlations != 1.0])
                
                pattern = SpikePattern(
                    pattern_id=f"spatial_cluster_{cluster_id}",
                    spike_times=cluster_spikes,
                    neuron_indices=cluster_neuron_indices,
                    duration_ms=duration_ms,
                    frequency_hz=frequency,
                    spatial_distribution=spatial_dist,
                    confidence=max(0, coherence)
                )
                patterns.append(pattern)
                
        return patterns
        
    def _calculate_burst_confidence(self, burst_spikes: np.ndarray, 
                                  burst_isi: np.ndarray) -> float:
        """Calculate confidence score for burst pattern."""
        if len(burst_spikes) < 3:
            return 0.0
            
        # Burst quality metrics
        isi_cv = np.std(burst_isi) / np.mean(burst_isi) if np.mean(burst_isi) > 0 else 1.0
        burst_intensity = len(burst_spikes) / (burst_spikes[-1] - burst_spikes[0])
        
        # Lower CV (more regular) and higher intensity = higher confidence
        confidence = (1.0 - min(isi_cv, 1.0)) * min(burst_intensity / 50, 1.0)
        
        return confidence
        
    def add_pattern_to_library(self, pattern: SpikePattern, gas_type: str):
        """Add a pattern to the pattern library for future recognition."""
        pattern.gas_association = gas_type
        
        if gas_type not in self.pattern_library:
            self.pattern_library[gas_type] = []
            
        self.pattern_library[gas_type].append(pattern)
        
    def match_pattern(self, observed_pattern: SpikePattern) -> Dict[str, float]:
        """Match observed pattern against library patterns."""
        similarities = {}
        
        for gas_type, patterns in self.pattern_library.items():
            max_similarity = 0.0
            
            for library_pattern in patterns:
                similarity = self._calculate_pattern_similarity(observed_pattern, library_pattern)
                max_similarity = max(max_similarity, similarity)
                
            similarities[gas_type] = max_similarity
            
        return similarities
        
    def _calculate_pattern_similarity(self, pattern1: SpikePattern, 
                                    pattern2: SpikePattern) -> float:
        """Calculate similarity between two spike patterns."""
        # Frequency similarity
        freq_diff = abs(pattern1.frequency_hz - pattern2.frequency_hz)
        freq_similarity = np.exp(-freq_diff / 10.0)  # Exponential decay
        
        # Duration similarity
        dur_diff = abs(pattern1.duration_ms - pattern2.duration_ms)
        dur_similarity = np.exp(-dur_diff / 100.0)
        
        # Spatial similarity (cosine similarity)
        spatial_sim = np.dot(pattern1.spatial_distribution, pattern2.spatial_distribution)
        spatial_norm = (np.linalg.norm(pattern1.spatial_distribution) * 
                       np.linalg.norm(pattern2.spatial_distribution))
        if spatial_norm > 0:
            spatial_similarity = spatial_sim / spatial_norm
        else:
            spatial_similarity = 0.0
            
        # Combined similarity
        similarity = (freq_similarity * 0.3 + dur_similarity * 0.3 + spatial_similarity * 0.4)
        
        return similarity


class AnomalyDetector:
    """Detects anomalous patterns in neuromorphic gas detection data."""
    
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.baseline_patterns: deque = deque(maxlen=1000)
        self.anomaly_threshold = 2.0  # Standard deviations
        
    def fit_baseline(self, normal_patterns: List[SpikePattern]):
        """Fit the anomaly detector on normal patterns."""
        self.baseline_patterns.extend(normal_patterns)
        
    def detect_anomaly(self, pattern: SpikePattern) -> Tuple[bool, float]:
        """
        Detect if a pattern is anomalous.
        
        Returns:
            Tuple of (is_anomaly, anomaly_score)
        """
        if len(self.baseline_patterns) < 10:
            return False, 0.0
            
        # Extract features from patterns
        baseline_features = self._extract_features(list(self.baseline_patterns))
        pattern_features = self._extract_features([pattern])
        
        # Calculate anomaly score using statistical approach
        anomaly_score = self._calculate_anomaly_score(baseline_features, pattern_features[0])
        
        is_anomaly = anomaly_score > self.anomaly_threshold
        
        return is_anomaly, anomaly_score
        
    def _extract_features(self, patterns: List[SpikePattern]) -> np.ndarray:
        """Extract feature vectors from spike patterns."""
        features = []
        
        for pattern in patterns:
            feature_vector = [
                pattern.frequency_hz,
                pattern.duration_ms,
                pattern.confidence,
                np.sum(pattern.spatial_distribution),
                np.std(pattern.spike_times) if len(pattern.spike_times) > 1 else 0,
                len(pattern.spike_times),
                entropy(pattern.spatial_distribution + 1e-10),  # Add small value to avoid log(0)
            ]
            features.append(feature_vector)
            
        return np.array(features)
        
    def _calculate_anomaly_score(self, baseline_features: np.ndarray, 
                               pattern_features: np.ndarray) -> float:
        """Calculate anomaly score using Mahalanobis distance."""
        if baseline_features.shape[0] < 2:
            return 0.0
            
        # Calculate mean and covariance of baseline
        baseline_mean = np.mean(baseline_features, axis=0)
        baseline_cov = np.cov(baseline_features.T)
        
        # Add regularization to covariance matrix
        baseline_cov += np.eye(baseline_cov.shape[0]) * 1e-6
        
        try:
            # Calculate Mahalanobis distance
            diff = pattern_features - baseline_mean
            inv_cov = np.linalg.inv(baseline_cov)
            mahal_dist = np.sqrt(diff.T @ inv_cov @ diff)
            
            return mahal_dist
        except np.linalg.LinAlgError:
            # Fallback to Euclidean distance if covariance is singular
            euclidean_dist = np.linalg.norm(pattern_features - baseline_mean)
            return euclidean_dist