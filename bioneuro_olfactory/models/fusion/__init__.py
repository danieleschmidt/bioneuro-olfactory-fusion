"""Multi-modal fusion models for combining chemical and audio features."""

from .multimodal_fusion import (
    EarlyFusion,
    AttentionFusion,
    HierarchicalFusion,
    SpikingFusion,
    TemporalAligner
)

__all__ = [
    'EarlyFusion',
    'AttentionFusion', 
    'HierarchicalFusion',
    'SpikingFusion',
    'TemporalAligner'
]