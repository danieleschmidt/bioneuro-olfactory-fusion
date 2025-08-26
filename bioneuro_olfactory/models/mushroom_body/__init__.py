"""Mushroom body decision layer models."""

from .decision_layer import (
    DecisionLayer,
    ReinforcementDecisionLayer,
    HierarchicalDecisionSystem,
    EnsembleDecisionSystem,
    DecisionConfig,
    create_decision_system,
    MushroomBodyDecision,
    RLDecisionLayer,
    HierarchicalDecision
)

# Legacy compatibility
AdaptiveDecisionLayer = ReinforcementDecisionLayer
MushroomBodyOutputNeuron = DecisionLayer
DecisionLayerConfig = DecisionConfig
EnsembleDecisionLayer = EnsembleDecisionSystem
create_moth_decision_layer = create_decision_system
create_efficient_decision_layer = create_decision_system

__all__ = [
    'DecisionLayer',
    'ReinforcementDecisionLayer',
    'HierarchicalDecisionSystem',
    'EnsembleDecisionSystem',
    'DecisionConfig',
    'create_decision_system',
    'MushroomBodyDecision',
    'RLDecisionLayer',
    'HierarchicalDecision',
    # Legacy exports
    'AdaptiveDecisionLayer',
    'MushroomBodyOutputNeuron',
    'DecisionLayerConfig',
    'EnsembleDecisionLayer',
    'create_moth_decision_layer',
    'create_efficient_decision_layer'
]