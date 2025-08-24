"""Projection neuron models for olfactory processing."""

from .projection_neurons import (
    ProjectionNeuron,
    ProjectionNeuronLayer,
    ProjectionNeuronNetwork,
    ProjectionNeuronConfig,
    create_moth_projection_network,
    create_efficient_projection_layer
)

__all__ = [
    'ProjectionNeuron',
    'ProjectionNeuronLayer',
    'ProjectionNeuronNetwork', 
    'ProjectionNeuronConfig',
    'create_moth_projection_network',
    'create_efficient_projection_layer'
]