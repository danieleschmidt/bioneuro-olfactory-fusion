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
    AdaptiveProjectionNeurons,
    create_moth_projection_network,
    create_efficient_projection_network
)

# Import Kenyon cell models
from .kenyon import (
    KenyonCellLayer,
    AdaptiveKenyonCells,
    KenyonCellConfig,
    CompetitiveKenyonNetwork,
    HierarchicalKenyonCells,
    create_sparse_kenyon_network,
    create_efficient_kenyon_network
)

# Import mushroom body models
from .mushroom_body import (
    DecisionLayer,
    AdaptiveDecisionLayer,
    MushroomBodyOutputNeuron,
    DecisionLayerConfig,
    EnsembleDecisionLayer,
    create_moth_decision_layer,
    create_efficient_decision_layer
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
    'AdaptiveProjectionNeurons',
    'create_moth_projection_network',
    'create_efficient_projection_network',
    
    # Kenyon models
    'KenyonCellLayer',
    'AdaptiveKenyonCells',
    'KenyonCellConfig',
    'CompetitiveKenyonNetwork',
    'HierarchicalKenyonCells',
    'create_sparse_kenyon_network',
    'create_efficient_kenyon_network',
    
    # Mushroom body models
    'DecisionLayer',
    'AdaptiveDecisionLayer',
    'MushroomBodyOutputNeuron',
    'DecisionLayerConfig',
    'EnsembleDecisionLayer',
    'create_moth_decision_layer',
    'create_efficient_decision_layer'
]