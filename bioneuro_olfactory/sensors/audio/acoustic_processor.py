"""Acoustic feature extraction for gas release detection.

This module processes audio signals to extract features that can
indicate gas releases (hissing, bubbling, equipment sounds) for
multi-modal gas detection systems.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

try:
    import librosa
    import librosa.display
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    librosa = None

try:
    import scipy.signal
    import scipy.fftpack
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    scipy = None

logger = logging.getLogger(__name__)


@dataclass
class AudioConfig:
    """Configuration for audio processing."""
    sample_rate: int = 44100
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    n_mfcc: int = 13
    n_chroma: int = 12
    window_duration: float = 2.0  # seconds
    overlap_ratio: float = 0.5
    preemphasis_coeff: float = 0.97
    noise_floor_db: float = -60.0


class AcousticProcessor:
    """Audio feature extractor for gas detection."""
    
    def __init__(self, config: AudioConfig = None):
        self.config = config or AudioConfig()
        
        if not LIBROSA_AVAILABLE:
            logger.warning("librosa not available, using simplified audio processing")
        if not SCIPY_AVAILABLE:
            logger.warning("scipy not available, some features may be limited")
            
        # Gas-specific frequency ranges (Hz) based on acoustic signatures
        self.gas_signature_ranges = {
            'leak_detection': (1000, 8000),    # High-frequency hissing
            'bubbling': (200, 1000),           # Low-frequency bubbling
            'valve_operation': (500, 2000),    # Mechanical sounds
            'equipment_hum': (50, 200),        # Low-frequency equipment noise
            'ultrasonic': (20000, 22000)       # Ultrasonic leak detection
        }
        
        # Initialize feature history for temporal analysis
        self.feature_history = []
        self.max_history_length = 100
        
    def extract_features(
        self, 
        audio_signal: np.ndarray, 
        include_spectral_contrast: bool = True,
        include_zero_crossing_rate: bool = True,
        include_spectral_rolloff: bool = True,
        include_temporal_features: bool = True
    ) -> Dict[str, np.ndarray]:
        """Extract comprehensive acoustic features from audio signal.
        
        Args:
            audio_signal: Audio time series
            include_spectral_contrast: Include spectral contrast features
            include_zero_crossing_rate: Include ZCR features
            include_spectral_rolloff: Include spectral rolloff
            include_temporal_features: Include temporal evolution features
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Preprocessing
        audio_processed = self._preprocess_audio(audio_signal)
        
        if LIBROSA_AVAILABLE:
            features.update(self._extract_librosa_features(
                audio_processed, include_spectral_contrast,
                include_zero_crossing_rate, include_spectral_rolloff
            ))
        else:
            features.update(self._extract_basic_features(audio_processed))
            
        # Gas-specific spectral features
        features.update(self._extract_gas_specific_features(audio_processed))
        
        # Statistical features
        features.update(self._extract_statistical_features(audio_processed))
        
        # Temporal features
        if include_temporal_features:
            features.update(self._extract_temporal_features(features))
            
        # Store in history
        self._update_feature_history(features)
        
        return features
        
    def _preprocess_audio(self, audio_signal: np.ndarray) -> np.ndarray:
        """Preprocess audio signal."""
        # Apply preemphasis filter
        if len(audio_signal) > 1:
            preemphasized = np.append(
                audio_signal[0], 
                audio_signal[1:] - self.config.preemphasis_coeff * audio_signal[:-1]
            )
        else:
            preemphasized = audio_signal
            
        # Normalize amplitude
        if np.max(np.abs(preemphasized)) > 0:
            normalized = preemphasized / np.max(np.abs(preemphasized))
        else:
            normalized = preemphasized
            
        return normalized
        
    def _extract_librosa_features(
        self, 
        audio: np.ndarray,
        include_spectral_contrast: bool,
        include_zero_crossing_rate: bool,
        include_spectral_rolloff: bool
    ) -> Dict[str, np.ndarray]:
        """Extract features using librosa library."""
        features = {}
        
        try:
            # MFCC features
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=self.config.sample_rate,
                n_mfcc=self.config.n_mfcc,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length
            )
            features['mfcc'] = np.mean(mfcc, axis=1)
            features['mfcc_std'] = np.std(mfcc, axis=1)
            
            # Mel-scale spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.config.sample_rate,
                n_mels=self.config.n_mels,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length
            )
            mel_db = librosa.power_to_db(mel_spec, ref=np.max)
            features['mel_spectrogram'] = np.mean(mel_db, axis=1)
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(
                y=audio,
                sr=self.config.sample_rate,
                n_chroma=self.config.n_chroma,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length
            )
            features['chroma'] = np.mean(chroma, axis=1)
            
            # Spectral centroid
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio,
                sr=self.config.sample_rate,
                hop_length=self.config.hop_length
            )
            features['spectral_centroid'] = np.mean(spectral_centroids)
            features['spectral_centroid_std'] = np.std(spectral_centroids)
            
            # Spectral bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=audio,
                sr=self.config.sample_rate,
                hop_length=self.config.hop_length
            )
            features['spectral_bandwidth'] = np.mean(spectral_bandwidth)
            
            # Zero crossing rate
            if include_zero_crossing_rate:
                zcr = librosa.feature.zero_crossing_rate(
                    audio,
                    hop_length=self.config.hop_length
                )
                features['zero_crossing_rate'] = np.mean(zcr)
                features['zero_crossing_rate_std'] = np.std(zcr)
                
            # Spectral rolloff
            if include_spectral_rolloff:
                rolloff = librosa.feature.spectral_rolloff(
                    y=audio,
                    sr=self.config.sample_rate,
                    hop_length=self.config.hop_length
                )
                features['spectral_rolloff'] = np.mean(rolloff)
                
            # Spectral contrast
            if include_spectral_contrast:
                contrast = librosa.feature.spectral_contrast(
                    y=audio,
                    sr=self.config.sample_rate,
                    hop_length=self.config.hop_length
                )
                features['spectral_contrast'] = np.mean(contrast, axis=1)
                
            # Tempo and rhythm
            tempo, beats = librosa.beat.beat_track(
                y=audio,
                sr=self.config.sample_rate,
                hop_length=self.config.hop_length
            )
            features['tempo'] = tempo
            features['beat_strength'] = np.mean(librosa.util.normalize(beats))
            
        except Exception as e:
            logger.error(f"Error extracting librosa features: {e}")
            
        return features
        
    def _extract_basic_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract basic features without librosa."""
        features = {}
        
        if SCIPY_AVAILABLE:
            # Basic FFT-based features
            fft = scipy.fftpack.fft(audio)
            freqs = scipy.fftpack.fftfreq(len(audio), 1/self.config.sample_rate)
            magnitude = np.abs(fft)
            
            # Spectral centroid (basic version)
            spectral_centroid = np.sum(freqs[:len(freqs)//2] * magnitude[:len(magnitude)//2]) / np.sum(magnitude[:len(magnitude)//2])
            features['spectral_centroid'] = spectral_centroid
            
            # Spectral energy in different bands
            low_energy = np.sum(magnitude[np.logical_and(freqs >= 0, freqs < 1000)])
            mid_energy = np.sum(magnitude[np.logical_and(freqs >= 1000, freqs < 4000)])
            high_energy = np.sum(magnitude[np.logical_and(freqs >= 4000, freqs < 8000)])
            
            total_energy = low_energy + mid_energy + high_energy
            if total_energy > 0:
                features['low_band_ratio'] = low_energy / total_energy
                features['mid_band_ratio'] = mid_energy / total_energy
                features['high_band_ratio'] = high_energy / total_energy
            else:
                features['low_band_ratio'] = 0.0
                features['mid_band_ratio'] = 0.0
                features['high_band_ratio'] = 0.0
                
        # Basic time-domain features
        features['rms_energy'] = np.sqrt(np.mean(audio**2))
        features['peak_amplitude'] = np.max(np.abs(audio))
        
        # Simple zero crossing rate
        zero_crossings = np.sum(np.diff(np.sign(audio)) != 0)
        features['zero_crossing_rate'] = zero_crossings / len(audio)
        
        return features
        
    def _extract_gas_specific_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract features specific to gas detection applications."""
        features = {}
        
        if SCIPY_AVAILABLE:
            # Compute power spectral density
            freqs, psd = scipy.signal.welch(
                audio,
                fs=self.config.sample_rate,
                nperseg=min(len(audio), 1024)
            )
            
            # Energy in gas-specific frequency ranges
            for gas_type, (low_freq, high_freq) in self.gas_signature_ranges.items():
                freq_mask = np.logical_and(freqs >= low_freq, freqs <= high_freq)
                if np.any(freq_mask):
                    energy = np.sum(psd[freq_mask])
                    features[f'{gas_type}_energy'] = energy
                else:
                    features[f'{gas_type}_energy'] = 0.0
                    
            # Leak detection specific features
            # High-frequency content (indicative of leaks)
            high_freq_mask = freqs > 2000
            if np.any(high_freq_mask):
                high_freq_energy = np.sum(psd[high_freq_mask])
                total_energy = np.sum(psd)
                features['high_freq_ratio'] = high_freq_energy / (total_energy + 1e-8)
            else:
                features['high_freq_ratio'] = 0.0
                
            # Spectral flatness (measure of noise-like quality)
            geometric_mean = scipy.stats.gmean(psd[psd > 0]) if len(psd[psd > 0]) > 0 else 1e-8
            arithmetic_mean = np.mean(psd)
            features['spectral_flatness'] = geometric_mean / (arithmetic_mean + 1e-8)
            
        else:
            # Simplified versions without scipy
            for gas_type in self.gas_signature_ranges:
                features[f'{gas_type}_energy'] = 0.0
            features['high_freq_ratio'] = 0.0
            features['spectral_flatness'] = 0.0
            
        return features
        
    def _extract_statistical_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract statistical features from audio signal."""
        features = {}
        
        # Basic statistics
        features['mean'] = np.mean(audio)
        features['std'] = np.std(audio)
        features['variance'] = np.var(audio)
        features['skewness'] = self._calculate_skewness(audio)
        features['kurtosis'] = self._calculate_kurtosis(audio)
        
        # Dynamic range
        features['dynamic_range'] = np.max(audio) - np.min(audio)
        
        # Crest factor (peak to RMS ratio)
        rms = np.sqrt(np.mean(audio**2))
        peak = np.max(np.abs(audio))
        features['crest_factor'] = peak / (rms + 1e-8)
        
        return features
        
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
        
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
        
    def _extract_temporal_features(self, current_features: Dict) -> Dict[str, float]:
        """Extract temporal evolution features."""
        temporal_features = {}
        
        if len(self.feature_history) > 1:
            # Feature change rate
            for key in ['spectral_centroid', 'rms_energy', 'zero_crossing_rate']:
                if key in current_features and key in self.feature_history[-1]:
                    current_val = current_features[key]
                    prev_val = self.feature_history[-1][key]
                    change_rate = abs(current_val - prev_val) / (abs(prev_val) + 1e-8)
                    temporal_features[f'{key}_change_rate'] = change_rate
                    
            # Feature trend (over last few frames)
            if len(self.feature_history) >= 5:
                for key in ['spectral_centroid', 'rms_energy']:
                    if key in current_features:
                        recent_values = [frame.get(key, 0) for frame in self.feature_history[-5:]]
                        if len(recent_values) > 1:
                            # Simple linear trend
                            x = np.arange(len(recent_values))
                            trend = np.polyfit(x, recent_values, 1)[0]
                            temporal_features[f'{key}_trend'] = trend
                            
        return temporal_features
        
    def _update_feature_history(self, features: Dict):
        """Update feature history for temporal analysis."""
        self.feature_history.append(features.copy())
        
        # Limit history length
        if len(self.feature_history) > self.max_history_length:
            self.feature_history.pop(0)
            
    def detect_gas_events(
        self, 
        audio_signal: np.ndarray,
        threshold_multiplier: float = 2.0
    ) -> Dict[str, Any]:
        """Detect potential gas-related acoustic events.
        
        Args:
            audio_signal: Audio time series
            threshold_multiplier: Multiplier for detection thresholds
            
        Returns:
            Dictionary with detection results
        """
        features = self.extract_features(audio_signal)
        
        detections = {
            'leak_detected': False,
            'bubble_detected': False,
            'equipment_anomaly': False,
            'confidence_scores': {},
            'feature_values': features
        }
        
        # Leak detection based on high-frequency content
        if 'high_freq_ratio' in features:
            leak_threshold = 0.3 * threshold_multiplier
            leak_confidence = min(features['high_freq_ratio'] / leak_threshold, 1.0)
            detections['leak_detected'] = features['high_freq_ratio'] > leak_threshold
            detections['confidence_scores']['leak'] = leak_confidence
            
        # Bubble detection based on low-frequency energy
        if 'bubbling_energy' in features:
            bubble_threshold = np.mean([f.get('bubbling_energy', 0) for f in self.feature_history[-10:]]) * threshold_multiplier
            bubble_confidence = min(features['bubbling_energy'] / (bubble_threshold + 1e-8), 1.0)
            detections['bubble_detected'] = features['bubbling_energy'] > bubble_threshold
            detections['confidence_scores']['bubble'] = bubble_confidence
            
        # Equipment anomaly detection
        if 'spectral_centroid_change_rate' in features:
            anomaly_threshold = 0.5 * threshold_multiplier
            anomaly_confidence = min(features['spectral_centroid_change_rate'] / anomaly_threshold, 1.0)
            detections['equipment_anomaly'] = features['spectral_centroid_change_rate'] > anomaly_threshold
            detections['confidence_scores']['equipment'] = anomaly_confidence
            
        return detections
        
    def create_spectrogram(self, audio_signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create spectrogram for visualization.
        
        Args:
            audio_signal: Audio time series
            
        Returns:
            Tuple of (frequencies, times, spectrogram_magnitude)
        """
        if SCIPY_AVAILABLE:
            frequencies, times, spectrogram = scipy.signal.spectrogram(
                audio_signal,
                fs=self.config.sample_rate,
                nperseg=self.config.n_fft,
                noverlap=self.config.n_fft - self.config.hop_length
            )
            return frequencies, times, np.log10(spectrogram + 1e-8)
        else:
            # Simple fallback
            logger.warning("scipy not available, returning empty spectrogram")
            return np.array([]), np.array([]), np.array([[]])
            
    def process_streaming_audio(
        self, 
        audio_chunk: np.ndarray,
        detection_callback=None
    ) -> Dict[str, Any]:
        """Process streaming audio data.
        
        Args:
            audio_chunk: New audio data chunk
            detection_callback: Optional callback for detections
            
        Returns:
            Processing results
        """
        # Extract features from chunk
        features = self.extract_features(audio_chunk)
        
        # Detect events
        detections = self.detect_gas_events(audio_chunk)
        
        # Call detection callback if provided
        if detection_callback and any([
            detections['leak_detected'],
            detections['bubble_detected'], 
            detections['equipment_anomaly']
        ]):
            detection_callback(detections)
            
        return {
            'features': features,
            'detections': detections,
            'timestamp': len(self.feature_history)
        }
        
    def get_feature_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics of extracted features over history.
        
        Returns:
            Dictionary with feature statistics
        """
        if not self.feature_history:
            return {}
            
        summary = {}
        
        # Get all feature keys
        all_keys = set()
        for frame in self.feature_history:
            all_keys.update(frame.keys())
            
        # Calculate statistics for each feature
        for key in all_keys:
            values = [frame.get(key, 0) for frame in self.feature_history if key in frame]
            if values:
                summary[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'current': values[-1] if values else 0
                }
                
        return summary


# Factory functions
def create_standard_audio_processor() -> AcousticProcessor:
    """Create standard audio processor for gas detection."""
    config = AudioConfig(
        sample_rate=44100,
        n_mfcc=13,
        n_mels=64,
        window_duration=1.0,
        overlap_ratio=0.5
    )
    return AcousticProcessor(config)


def create_realtime_audio_processor() -> AcousticProcessor:
    """Create optimized audio processor for real-time applications."""
    config = AudioConfig(
        sample_rate=22050,  # Reduced for faster processing
        n_mfcc=8,           # Fewer coefficients
        n_mels=32,          # Fewer mel bands
        n_fft=1024,         # Smaller FFT
        hop_length=256,     # Smaller hop
        window_duration=0.5 # Shorter windows
    )
    return AcousticProcessor(config)