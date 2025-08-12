"""Multi-modal fusion models for combining chemical and audio features."""

from .multimodal_fusion import (
    EarlyFusion,
    AttentionFusion,
    HierarchicalFusion,
    SpikingFusion,
    TemporalAligner,
    OlfactoryFusionSNN,
    FusionConfig,
    create_standard_fusion_network
)

__all__ = [
    'EarlyFusion',
    'AttentionFusion', 
    'HierarchicalFusion',
    'SpikingFusion',
    'TemporalAligner',
    'OlfactoryFusionSNN',
    'FusionConfig',
    'create_standard_fusion_network'
]