"""Projection neuron models for olfactory processing."""

from .projection_neurons import (
    ProjectionNeuronLayer,
    ProjectionNeuronNetwork,
    ProjectionNeuronConfig,
    AdaptiveProjectionLayer,
    create_moth_inspired_projection_network
)

__all__ = [
    'ProjectionNeuronLayer',
    'ProjectionNeuronNetwork', 
    'ProjectionNeuronConfig',
    'AdaptiveProjectionLayer',
    'create_moth_inspired_projection_network'
]

from .projection_neurons import (
    ProjectionNeuronLayer,
    ProjectionNeuronNetwork,
    ProjectionNeuronConfig,
    create_standard_projection_network
)

__all__ = [
    'ProjectionNeuronLayer',
    'ProjectionNeuronNetwork',
    'ProjectionNeuronConfig',
    'create_standard_projection_network'
]