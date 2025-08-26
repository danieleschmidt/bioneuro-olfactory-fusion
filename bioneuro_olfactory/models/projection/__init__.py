"""Projection neuron models for olfactory processing."""

from .projection_neurons import (
    ProjectionNeuron,
    ProjectionNeuronLayer,
    AdaptiveProjectionLayer,
    BiologicallyAccurateProjectionLayer,
    ProjectionNeuronNetwork,
    ProjectionNeuronConfig,
    create_moth_projection_system,
    create_moth_projection_network,
    create_efficient_projection_layer,
    create_efficient_projection_network,
    MothProjectionNeurons,
    BioinspiredProjectionLayer
)

# Additional aliases for compatibility
AdaptiveProjectionNeurons = AdaptiveProjectionLayer

__all__ = [
    'ProjectionNeuron',
    'ProjectionNeuronLayer',
    'AdaptiveProjectionLayer', 
    'BiologicallyAccurateProjectionLayer',
    'ProjectionNeuronNetwork',
    'ProjectionNeuronConfig',
    'create_moth_projection_system',
    'create_moth_projection_network',
    'create_efficient_projection_layer',
    'create_efficient_projection_network',
    'MothProjectionNeurons',
    'BioinspiredProjectionLayer',
    'AdaptiveProjectionNeurons'
]