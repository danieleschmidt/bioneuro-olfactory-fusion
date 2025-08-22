"""Projection neuron models for olfactory processing."""

from .projection_neurons import (
    ProjectionNeuronLayer,
    ProjectionNeuronNetwork,
    ProjectionNeuronConfig,
    AdaptiveProjectionNeurons,
    GainControlNetwork,
    create_moth_projection_network,
    create_efficient_projection_network
)

__all__ = [
    'ProjectionNeuronLayer',
    'ProjectionNeuronNetwork', 
    'ProjectionNeuronConfig',
    'AdaptiveProjectionNeurons',
    'GainControlNetwork',
    'create_moth_projection_network',
    'create_efficient_projection_network'
]