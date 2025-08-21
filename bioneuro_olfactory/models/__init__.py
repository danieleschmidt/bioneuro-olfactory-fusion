"""BioNeuro-Olfactory models package.

This package contains all the neural network models for the bioneuro-olfactory system
including fusion models, projection neurons, Kenyon cells, and decision layers.
"""

# Import fusion models
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

# Import projection neuron models
from .projection import (
    ProjectionNeuronLayer,
    ProjectionNeuronNetwork,
    ProjectionNeuronConfig,
    AdaptiveProjectionLayer,
    create_moth_inspired_projection_network,
    create_standard_projection_network
)

# Import Kenyon cell models
from .kenyon import (
    KenyonCellLayer,
    AdaptiveKenyonCells,
    CompetitiveKenyonLayer,
    KenyonCellConfig,
    create_moth_kenyon_layer,
    create_efficient_kenyon_layer
)

# Import mushroom body models
from .mushroom_body import (
    DecisionLayer,
    AdaptiveDecisionLayer,
    EnsembleDecisionLayer,
    MushroomBodyOutputNeuron,
    DecisionLayerConfig,
    create_standard_decision_layer,
    create_adaptive_decision_layer
)

__all__ = [
    # Fusion models
    'EarlyFusion',
    'AttentionFusion',
    'HierarchicalFusion',
    'SpikingFusion',
    'TemporalAligner',
    'OlfactoryFusionSNN',
    'FusionConfig',
    'create_standard_fusion_network',
    
    # Projection models
    'ProjectionNeuronLayer',
    'ProjectionNeuronNetwork',
    'ProjectionNeuronConfig',
    'AdaptiveProjectionLayer',
    'create_moth_inspired_projection_network',
    'create_standard_projection_network',
    
    # Kenyon models
    'KenyonCellLayer',
    'AdaptiveKenyonCells',
    'CompetitiveKenyonLayer',
    'KenyonCellConfig',
    'create_moth_kenyon_layer',
    'create_efficient_kenyon_layer',
    
    # Mushroom body models
    'DecisionLayer',
    'AdaptiveDecisionLayer',
    'EnsembleDecisionLayer',
    'MushroomBodyOutputNeuron',
    'DecisionLayerConfig',
    'create_standard_decision_layer',
    'create_adaptive_decision_layer'
]