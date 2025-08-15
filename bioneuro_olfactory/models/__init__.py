"""
Model components for neuromorphic gas detection.

This module provides access to the core model components with
graceful handling of missing dependencies.
"""

# Import basic models that don't require torch
try:
    from .projection import (
        ProjectionNeuronConfig
    )
except ImportError:
    # Create minimal placeholder classes
    class ProjectionNeuronConfig:
        def __init__(self, *args, **kwargs):
            pass

try:
    from .kenyon import (
        KenyonCellConfig
    )
except ImportError:
    class KenyonCellConfig:
        def __init__(self, *args, **kwargs):
            pass

try:
    from .mushroom_body import (
        DecisionConfig,
        GasType,
        DetectionResult
    )
except ImportError:
    from enum import Enum
    
    class GasType(Enum):
        CLEAN_AIR = 0
        METHANE = 1
        CARBON_MONOXIDE = 2
        AMMONIA = 3
        PROPANE = 4
        
    class DetectionResult:
        def __init__(self, gas_type, concentration, confidence, hazard_probability, response_time, network_activity):
            self.gas_type = gas_type
            self.concentration = concentration
            self.confidence = confidence
            self.hazard_probability = hazard_probability
            self.response_time = response_time
            self.network_activity = network_activity
            
    class DecisionConfig:
        def __init__(self, *args, **kwargs):
            pass

try:
    from .fusion import (
        FusionConfig,
        EarlyFusion,
        AttentionFusion,
        HierarchicalFusion,
        SpikingFusion,
        TemporalAligner
    )
except ImportError:
    # Create placeholder fusion classes
    class FusionConfig:
        def __init__(self, *args, **kwargs):
            pass
            
    class EarlyFusion:
        def __init__(self, *args, **kwargs):
            pass
            
    class AttentionFusion:
        def __init__(self, *args, **kwargs):
            pass
            
    class HierarchicalFusion:
        def __init__(self, *args, **kwargs):
            pass
            
    class SpikingFusion:
        def __init__(self, *args, **kwargs):
            pass
            
    class TemporalAligner:
        def __init__(self, *args, **kwargs):
            pass

__all__ = [
    'ProjectionNeuronConfig',
    'KenyonCellConfig', 
    'DecisionConfig',
    'GasType',
    'DetectionResult',
    'FusionConfig',
    'EarlyFusion',
    'AttentionFusion',
    'HierarchicalFusion',
    'SpikingFusion',
    'TemporalAligner'
]