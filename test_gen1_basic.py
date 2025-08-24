#!/usr/bin/env python3
"""Basic test for Generation 1 functionality without external dependencies."""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, '/root/repo')

# Mock numpy before importing our modules
import sys
from bioneuro_olfactory.core.mock_numpy import *
sys.modules['numpy'] = sys.modules['bioneuro_olfactory.core.mock_numpy']

def test_basic_fusion():
    """Test basic fusion functionality."""
    print("Testing basic fusion functionality...")
    
    try:
        from bioneuro_olfactory.models.fusion.multimodal_fusion import (
            EarlyFusion, AttentionFusion, HierarchicalFusion, create_standard_fusion_network
        )
        print("✓ Successfully imported fusion modules")
        
        # Test early fusion
        early_fusion = EarlyFusion(chemical_dim=6, audio_dim=8)
        chemical_data = [0.5, 0.3, 0.8, 0.2, 0.6, 0.4]
        audio_data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        
        result = early_fusion(chemical_data, audio_data)
        print(f"✓ Early fusion output length: {len(result.data)}")
        
        # Test attention fusion
        attention_fusion = AttentionFusion(chemical_dim=6, audio_dim=8)
        result = attention_fusion(chemical_data, audio_data)
        print(f"✓ Attention fusion output length: {len(result.data)}")
        
        # Test hierarchical fusion
        hierarchical_fusion = HierarchicalFusion(chemical_dim=6, audio_dim=8)
        result = hierarchical_fusion(chemical_data, audio_data)
        print(f"✓ Hierarchical fusion output length: {len(result.data)}")
        
        print("✓ All basic fusion tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Fusion test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_projection_neurons():
    """Test projection neuron functionality."""
    print("\nTesting projection neurons...")
    
    try:
        from bioneuro_olfactory.models.projection.projection_neurons import (
            ProjectionNeuron, ProjectionNeuronLayer, ProjectionNeuronConfig
        )
        print("✓ Successfully imported projection neuron modules")
        
        # Test single neuron
        neuron = ProjectionNeuron(tau_membrane=20.0, threshold=1.0)
        spike = neuron.update(input_current=1.5)
        print(f"✓ Single neuron spike: {spike}")
        
        # Test layer
        config = ProjectionNeuronConfig(num_receptors=6, num_projection_neurons=20)
        layer = ProjectionNeuronLayer(config)
        
        chemical_input = [0.5, 0.3, 0.8, 0.2, 0.6, 0.4]
        spikes, potentials = layer(chemical_input, duration=50)
        print(f"✓ Layer spikes shape: {spikes.shape}")
        print(f"✓ Layer potentials shape: {potentials.shape}")
        
        print("✓ All projection neuron tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Projection neuron test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_kenyon_cells():
    """Test Kenyon cell functionality."""
    print("\nTesting Kenyon cells...")
    
    try:
        from bioneuro_olfactory.models.kenyon.kenyon_cells import (
            KenyonCell, KenyonCellLayer, KenyonCellConfig
        )
        print("✓ Successfully imported Kenyon cell modules")
        
        # Test single cell
        cell = KenyonCell(num_inputs=20, connection_probability=0.2)
        input_spikes = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0] * 2
        spike = cell.update(input_spikes, global_inhibition=0.1)
        print(f"✓ Single Kenyon cell spike: {spike}")
        
        # Test layer
        config = KenyonCellConfig(num_projection_inputs=20, num_kenyon_cells=50, sparsity_target=0.1)
        layer = KenyonCellLayer(config)
        
        # Create mock projection spikes
        projection_spikes = [[_random.random() < 0.1 for _ in range(10)] for _ in range(20)]
        kenyon_spikes, potentials = layer(projection_spikes)
        print(f"✓ Kenyon spikes shape: {kenyon_spikes.shape}")
        
        stats = layer.get_sparsity_statistics(kenyon_spikes)
        print(f"✓ Sparsity stats: {stats}")
        
        print("✓ All Kenyon cell tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Kenyon cell test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_decision_layer():
    """Test decision layer functionality."""
    print("\nTesting decision layer...")
    
    try:
        from bioneuro_olfactory.models.mushroom_body.decision_layer import (
            MushroomBodyOutputNeuron, DecisionLayer, DecisionLayerConfig, GasType
        )
        print("✓ Successfully imported decision layer modules")
        
        # Test single output neuron
        neuron = MushroomBodyOutputNeuron(num_kenyon_inputs=50, gas_type=GasType.METHANE)
        kenyon_activity = [_random.random() * 0.1 for _ in range(50)]
        activation = neuron.update(kenyon_activity)
        print(f"✓ Single output neuron activation: {activation:.3f}")
        
        # Test decision layer
        config = DecisionLayerConfig(num_kenyon_inputs=50, num_output_neurons=4)
        layer = DecisionLayer(config)
        
        result = layer(kenyon_activity)
        print(f"✓ Detection result - Gas: {result.gas_type}, Confidence: {result.confidence:.3f}")
        
        print("✓ All decision layer tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Decision layer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Generation 1 tests."""
    print("=== GENERATION 1 BASIC FUNCTIONALITY TESTS ===")
    
    tests = [
        test_basic_fusion,
        test_projection_neurons,
        test_kenyon_cells,
        test_decision_layer
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n=== RESULTS ===")
    print(f"Passed: {passed}/{total} tests")
    
    if passed == total:
        print("🎉 GENERATION 1 SUCCESS: All basic functionality working!")
        return True
    else:
        print("❌ GENERATION 1 INCOMPLETE: Some tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)