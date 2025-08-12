"""Neuromorphic models for bioneuro-olfactory fusion."""

# Projection neurons
from .projection import (
    ProjectionNeuronLayer,
    ProjectionNeuronNetwork,
    ProjectionNeuronConfig,
    create_standard_projection_network
)

# Kenyon cells  
from .kenyon import (
    KenyonCellLayer,
    AdaptiveKenyonCells,
    KenyonCellConfig,
    create_standard_kenyon_layer,
    create_adaptive_kenyon_network
)

# Multi-modal fusion
from .fusion import (
    EarlyFusion,
    AttentionFusion,
    HierarchicalFusion,
    SpikingFusion,
    TemporalAligner,
    OlfactoryFusionSNN,
    FusionConfig,
    create_standard_fusion_network
)

# Mushroom body decision layer
from .mushroom_body import (
    DecisionLayer,
    AdaptiveDecisionLayer,
    MushroomBodyOutputNeuron,
    DecisionLayerConfig,
    create_standard_decision_layer
)

__all__ = [
    # Projection neurons
    'ProjectionNeuronLayer',
    'ProjectionNeuronNetwork', 
    'ProjectionNeuronConfig',
    'create_standard_projection_network',
    
    # Kenyon cells
    'KenyonCellLayer',
    'AdaptiveKenyonCells',
    'KenyonCellConfig', 
    'create_standard_kenyon_layer',
    'create_adaptive_kenyon_network',
    
    # Multi-modal fusion
    'EarlyFusion',
    'AttentionFusion',
    'HierarchicalFusion', 
    'SpikingFusion',
    'TemporalAligner',
    'OlfactoryFusionSNN',
    'FusionConfig',
    'create_standard_fusion_network',
    
    # Mushroom body decision layer
    'DecisionLayer',
    'AdaptiveDecisionLayer',
    'MushroomBodyOutputNeuron',
    'DecisionLayerConfig',
    'create_standard_decision_layer'
]