"""Next-Generation Autonomous Enhancements for BioNeuro-Olfactory-Fusion.

This module implements cutting-edge autonomous capabilities including
self-improving algorithms, autonomous optimization, adaptive learning,
and predictive maintenance for neuromorphic gas detection systems.

Autonomous Features:
- Self-improving neural architectures
- Autonomous hyperparameter optimization
- Predictive maintenance and self-healing
- Adaptive threat response
- Continuous learning and evolution
- Meta-learning for rapid adaptation
"""

from .self_improving_networks import (
    SelfImprovingNeuromorphicNetwork,
    ArchitectureEvolution,
    PerformanceBasedOptimization,
    AdaptiveNeuralGrowth
)

from .autonomous_optimization import (
    AutonomousHyperparameterOptimizer,
    BayesianOptimization,
    EvolutionaryOptimizer,
    MultiObjectiveOptimizer
)

from .predictive_maintenance import (
    PredictiveMaintenanceSystem,
    AnomalyPredictor,
    SelfHealingManager,
    ComponentHealthMonitor
)

from .adaptive_learning import (
    ContinuousLearningSystem,
    MetaLearningFramework,
    OnlineAdaptationEngine,
    EnvironmentalAdaptation
)

__all__ = [
    # Self-improving networks
    'SelfImprovingNeuromorphicNetwork',
    'ArchitectureEvolution',
    'PerformanceBasedOptimization', 
    'AdaptiveNeuralGrowth',
    
    # Autonomous optimization
    'AutonomousHyperparameterOptimizer',
    'BayesianOptimization',
    'EvolutionaryOptimizer',
    'MultiObjectiveOptimizer',
    
    # Predictive maintenance
    'PredictiveMaintenanceSystem',
    'AnomalyPredictor',
    'SelfHealingManager',
    'ComponentHealthMonitor',
    
    # Adaptive learning
    'ContinuousLearningSystem',
    'MetaLearningFramework',
    'OnlineAdaptationEngine',
    'EnvironmentalAdaptation'
]