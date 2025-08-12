"""BioNeuro-Olfactory-Fusion: Neuromorphic gas detection framework.

A bio-inspired spiking neural network framework for multi-modal hazardous
gas detection using electronic nose sensors and acoustic features.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.com"

# Core components
from .models.fusion.multimodal_fusion import (
    EarlyFusion,
    AttentionFusion,
    HierarchicalFusion,
    SpikingFusion,
    TemporalAligner
)

from .models.projection.projection_neurons import (
    ProjectionNeuronLayer,
    ProjectionNeuronNetwork,
    ProjectionNeuronConfig
)

from .models.kenyon.kenyon_cells import (
    KenyonCellLayer,
    AdaptiveKenyonCells,
    KenyonCellConfig
)

from .models.mushroom_body.decision_layer import (
    DecisionLayer,
    AdaptiveDecisionLayer,
    MushroomBodyOutputNeuron
)

from .core.neurons.lif import (
    LIFNeuron,
    AdaptiveLIFNeuron,
    InhibitoryNeuron
)

from .core.encoding.spike_encoding import (
    RateEncoder,
    TemporalEncoder,
    PhaseEncoder,
    BurstEncoder,
    PopulationEncoder,
    AdaptiveEncoder
)

from .sensors.enose.sensor_array import (
    ENoseArray,
    GasSensor,
    MOSSensor,
    ElectrochemicalSensor,
    PIDSensor,
    SensorSpec,
    create_standard_enose
)

# High-level API
class OlfactoryFusionSNN:
    """Main interface for the olfactory fusion spiking neural network.
    
    This class provides a high-level API for gas detection using bio-inspired
    neuromorphic computing with multi-modal sensor fusion.
    """
    
    def __init__(
        self,
        num_chemical_sensors: int = 6,
        num_audio_features: int = 128,
        num_projection_neurons: int = 1000,
        num_kenyon_cells: int = 5000,
        tau_membrane: float = 20.0,
        fusion_strategy: str = 'hierarchical'
    ):
        """Initialize the olfactory fusion network.
        
        Args:
            num_chemical_sensors: Number of chemical sensors in e-nose array
            num_audio_features: Number of audio feature dimensions
            num_projection_neurons: Number of projection neurons (first layer)
            num_kenyon_cells: Number of Kenyon cells (sparse coding layer)
            tau_membrane: Membrane time constant in milliseconds
            fusion_strategy: Multi-modal fusion strategy ('early', 'attention', 'hierarchical', 'spiking')
        """
        self.num_chemical_sensors = num_chemical_sensors
        self.num_audio_features = num_audio_features
        self.num_projection_neurons = num_projection_neurons
        self.num_kenyon_cells = num_kenyon_cells
        self.tau_membrane = tau_membrane
        self.fusion_strategy = fusion_strategy
        
        # Initialize network components
        self._build_network()
        
    def _build_network(self):
        """Build the complete network architecture."""
        # Projection neuron layer (olfactory receptor neurons)
        pn_config = ProjectionNeuronConfig(
            num_receptors=self.num_chemical_sensors,
            num_projection_neurons=self.num_projection_neurons,
            tau_membrane=self.tau_membrane
        )
        self.projection_layer = ProjectionNeuronLayer(pn_config)
        
        # Kenyon cell layer (sparse coding)
        kc_config = KenyonCellConfig(
            num_projection_inputs=self.num_projection_neurons,
            num_kenyon_cells=self.num_kenyon_cells,
            sparsity_target=0.05,
            tau_membrane=self.tau_membrane * 1.5  # Slower dynamics
        )
        self.kenyon_layer = KenyonCellLayer(kc_config)
        
        # Multi-modal fusion layer
        if self.fusion_strategy == 'early':
            self.fusion_layer = EarlyFusion(
                chemical_dim=self.num_chemical_sensors,
                audio_dim=self.num_audio_features,
                output_dim=self.num_chemical_sensors + self.num_audio_features
            )
        elif self.fusion_strategy == 'attention':
            self.fusion_layer = AttentionFusion(
                chemical_dim=self.num_chemical_sensors,
                audio_dim=self.num_audio_features,
                embed_dim=128
            )
        elif self.fusion_strategy == 'hierarchical':
            self.fusion_layer = HierarchicalFusion(
                chemical_dim=self.num_chemical_sensors,
                audio_dim=self.num_audio_features,
                hidden_dims=[64, 32]
            )
        elif self.fusion_strategy == 'spiking':
            self.fusion_layer = SpikingFusion(
                chemical_dim=self.num_chemical_sensors,
                audio_dim=self.num_audio_features,
                tau_membrane=self.tau_membrane
            )
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")
            
    def process(
        self, 
        chemical_input, 
        audio_input, 
        duration: int = 100
    ):
        """Process multi-modal input through the network.
        
        Args:
            chemical_input: Chemical sensor data
            audio_input: Audio feature data
            duration: Simulation duration in time steps
            
        Returns:
            Network output with spike patterns and predictions
        """
        # Process chemical input through projection neurons
        pn_spikes, pn_potentials = self.projection_layer(chemical_input, duration)
        
        # Process through Kenyon cells for sparse coding
        kc_spikes, kc_potentials = self.kenyon_layer(pn_spikes)
        
        # Multi-modal fusion
        if self.fusion_strategy == 'spiking':
            # For spiking fusion, we need spike trains
            encoder = RateEncoder()
            audio_spikes = encoder.encode(audio_input, duration)
            fused_output, _ = self.fusion_layer(pn_spikes, audio_spikes)
        else:
            # For other fusion strategies, use raw features
            fused_output = self.fusion_layer(chemical_input, audio_input)
            
        return {
            'projection_spikes': pn_spikes,
            'kenyon_spikes': kc_spikes,
            'fused_output': fused_output,
            'network_activity': {
                'projection_rates': self.projection_layer.get_firing_rates(pn_spikes),
                'kenyon_sparsity': self.kenyon_layer.get_sparsity_statistics(kc_spikes)
            }
        }
        
    def train(self, dataset, epochs: int = 100, learning_rate: float = 0.001):
        """Train the network on gas detection dataset."""
        raise NotImplementedError("Training implementation coming in next checkpoint")
        
    def evaluate(self, test_dataset):
        """Evaluate network performance on test dataset."""
        raise NotImplementedError("Evaluation implementation coming in next checkpoint")


# Convenience functions
def create_moth_inspired_network(
    num_sensors: int = 6,
    sparsity_level: float = 0.05
) -> OlfactoryFusionSNN:
    """Create a moth-inspired olfactory network configuration."""
    return OlfactoryFusionSNN(
        num_chemical_sensors=num_sensors,
        num_audio_features=128,
        num_projection_neurons=1000,  # ~1000 PNs in moth AL
        num_kenyon_cells=50000,       # ~50000 KCs in moth MB
        tau_membrane=20.0,
        fusion_strategy='hierarchical'
    )


def create_efficient_network(
    num_sensors: int = 6
) -> OlfactoryFusionSNN:
    """Create an efficient network for edge deployment."""
    return OlfactoryFusionSNN(
        num_chemical_sensors=num_sensors,
        num_audio_features=64,
        num_projection_neurons=200,
        num_kenyon_cells=1000,
        tau_membrane=30.0,
        fusion_strategy='early'
    )


# Package exports
__all__ = [
    # Main API
    'OlfactoryFusionSNN',
    'create_moth_inspired_network',
    'create_efficient_network',
    
    # Core components
    'LIFNeuron',
    'AdaptiveLIFNeuron',
    'InhibitoryNeuron',
    'ProjectionNeuronLayer',
    'ProjectionNeuronNetwork',
    'KenyonCellLayer',
    'AdaptiveKenyonCells',
    'DecisionLayer',
    'AdaptiveDecisionLayer',
    'MushroomBodyOutputNeuron',
    
    # Encoding
    'RateEncoder',
    'TemporalEncoder',
    'PhaseEncoder',
    'BurstEncoder',
    'PopulationEncoder',
    'AdaptiveEncoder',
    
    # Fusion
    'EarlyFusion',
    'AttentionFusion',
    'HierarchicalFusion',
    'SpikingFusion',
    'TemporalAligner',
    
    # Sensors
    'ENoseArray',
    'GasSensor',
    'MOSSensor',
    'ElectrochemicalSensor',
    'PIDSensor',
    'SensorSpec',
    'create_standard_enose'
]