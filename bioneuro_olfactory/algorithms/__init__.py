"""
Advanced algorithms for neuromorphic gas detection.
"""

from .adaptive_threshold import (
    AdaptiveThresholdManager,
    ThresholdOptimizer,
    ThresholdParameters,
    EnvironmentalConditions,
    DetectionEvent
)
from .pattern_recognition import (
    SpikeTrainAnalyzer,
    AnomalyDetector,
    SpikePattern,
    TemporalSignature
)

__all__ = [
    'AdaptiveThresholdManager',
    'ThresholdOptimizer',
    'ThresholdParameters',
    'EnvironmentalConditions', 
    'DetectionEvent',
    'SpikeTrainAnalyzer',
    'AnomalyDetector',
    'SpikePattern',
    'TemporalSignature'
]