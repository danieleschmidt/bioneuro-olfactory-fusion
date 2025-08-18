#!/usr/bin/env python3
"""
Advanced Generation 1 Validation: Next-Generation Neuromorphic Features
========================================================================

This script validates the newly implemented advanced neuromorphic components:
- Multi-modal fusion strategies (Early, Attention, Hierarchical, Spiking)
- Bio-inspired projection neurons with competitive dynamics
- Adaptive Kenyon cells with advanced plasticity mechanisms
- Decision layers with uncertainty estimation and meta-learning

Created as part of Terragon SDLC autonomous execution.
"""

import torch
import numpy as np
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# Import our new implementations
from bioneuro_olfactory.models.fusion.multimodal_fusion import (
    OlfactoryFusionSNN,
    EarlyFusion,
    AttentionFusion, 
    HierarchicalFusion,
    SpikingFusion,
    TemporalAligner,
    FusionConfig,
    interpret_gas_detection
)

from bioneuro_olfactory.models.projection.projection_neurons import (
    ProjectionNeuronLayer,
    ProjectionNeuronNetwork,
    CompetitiveProjectionLayer,
    create_moth_inspired_projection_network,
    analyze_projection_dynamics
)

from bioneuro_olfactory.models.kenyon.kenyon_cells import (
    KenyonCellLayer,
    AdaptiveKenyonCells,
    analyze_sparse_coding_quality,
    optimize_kenyon_sparsity
)

from bioneuro_olfactory.models.mushroom_body.decision_layer import (
    DecisionLayer,
    AdaptiveDecisionLayer,
    analyze_decision_dynamics,
    optimize_decision_parameters
)


class AdvancedNeuromorphicValidator:
    """Comprehensive validator for advanced neuromorphic components."""
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🔬 Running validation on: {self.device}")
        
        self.results = {
            'fusion_strategies': {},
            'projection_neurons': {},
            'kenyon_cells': {},
            'decision_layers': {},
            'integration_tests': {},
            'performance_metrics': {}
        }
        
    def validate_fusion_strategies(self) -> Dict:
        """Test all fusion strategies with synthetic data."""
        print("\n🧬 Validating Multi-Modal Fusion Strategies...")
        
        # Create synthetic sensor data
        batch_size, num_sensors, audio_features = 5, 6, 128
        chemical_data = torch.randn(batch_size, num_sensors).to(self.device)
        audio_data = torch.randn(batch_size, audio_features).to(self.device)
        
        fusion_results = {}
        
        # Test each fusion strategy
        strategies = ['early', 'attention', 'hierarchical', 'spiking']
        
        for strategy in strategies:
            print(f"  Testing {strategy} fusion...")
            
            try:
                start_time = time.time()
                
                config = FusionConfig(
                    chemical_dim=num_sensors,
                    audio_dim=audio_features,
                    device=self.device
                )
                
                if strategy == 'early':
                    fusion_layer = EarlyFusion(config)
                elif strategy == 'attention':
                    fusion_layer = AttentionFusion(config)
                elif strategy == 'hierarchical':
                    fusion_layer = HierarchicalFusion(config)
                elif strategy == 'spiking':
                    fusion_layer = SpikingFusion(config)
                    
                fusion_layer = fusion_layer.to(self.device)
                
                # Test forward pass
                if strategy == 'spiking':
                    output, debug_info = fusion_layer(chemical_data, audio_data)
                    fusion_results[strategy] = {
                        'output_shape': output.shape,
                        'output_mean': output.mean().item(),
                        'output_std': output.std().item(),
                        'spike_rates': debug_info['spike_rates'],
                        'processing_time': time.time() - start_time,
                        'status': 'SUCCESS'
                    }
                else:
                    output = fusion_layer(chemical_data, audio_data)
                    fusion_results[strategy] = {
                        'output_shape': output.shape,
                        'output_mean': output.mean().item(),
                        'output_std': output.std().item(),
                        'processing_time': time.time() - start_time,
                        'status': 'SUCCESS'
                    }
                    
                print(f"    ✅ {strategy} fusion: {output.shape} in {time.time() - start_time:.3f}s")
                
            except Exception as e:
                fusion_results[strategy] = {
                    'status': 'FAILED',
                    'error': str(e)
                }
                print(f"    ❌ {strategy} fusion failed: {e}")
                
        # Test complete OlfactoryFusionSNN
        print("  Testing complete OlfactoryFusionSNN...")
        try:
            network = OlfactoryFusionSNN(
                num_chemical_sensors=num_sensors,
                num_audio_features=audio_features,
                fusion_strategy='hierarchical',
                device=self.device
            )
            
            results = network(chemical_data, audio_data)
            interpretation = interpret_gas_detection(results)
            
            fusion_results['complete_network'] = {
                'detected_gas': interpretation['detected_gas'],
                'concentration_ppm': interpretation['concentration_ppm'],
                'hazard_probability': interpretation['hazard_probability'],
                'alert_level': interpretation['alert_level'],
                'status': 'SUCCESS'
            }
            
            print(f"    ✅ Complete network: Gas={interpretation['detected_gas']}, "
                  f"Concentration={interpretation['concentration_ppm']:.1f}ppm")
            
        except Exception as e:
            fusion_results['complete_network'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            print(f"    ❌ Complete network failed: {e}")
            
        self.results['fusion_strategies'] = fusion_results
        return fusion_results
        
    def validate_projection_neurons(self) -> Dict:
        """Test projection neuron implementations."""
        print("\n🦋 Validating Projection Neurons...")
        
        projection_results = {}
        
        # Test basic projection layer
        print("  Testing ProjectionNeuronLayer...")
        try:
            from bioneuro_olfactory.models.projection.projection_neurons import ProjectionNeuronConfig
            
            config = ProjectionNeuronConfig(
                num_receptors=6,
                num_projection_neurons=100,
                device=self.device
            )
            
            projection_layer = ProjectionNeuronLayer(config).to(self.device)
            
            # Test with synthetic receptor input
            receptor_input = torch.randn(3, 6).to(self.device)
            
            start_time = time.time()
            pn_spikes, pn_potentials = projection_layer(receptor_input, duration=50)
            process_time = time.time() - start_time
            
            # Analyze dynamics
            dynamics = analyze_projection_dynamics(pn_spikes, pn_potentials)
            
            projection_results['basic_layer'] = {
                'spike_shape': pn_spikes.shape,
                'potential_shape': pn_potentials.shape,
                'mean_firing_rate': dynamics['temporal_dynamics']['mean_rate'],
                'sparsity': dynamics['spatial_dynamics']['sparsity'],
                'processing_time': process_time,
                'status': 'SUCCESS'
            }
            
            print(f"    ✅ Basic layer: {pn_spikes.shape}, "
                  f"firing_rate={dynamics['temporal_dynamics']['mean_rate']:.3f}")
            
        except Exception as e:
            projection_results['basic_layer'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            print(f"    ❌ Basic layer failed: {e}")
            
        # Test competitive projection layer
        print("  Testing CompetitiveProjectionLayer...")
        try:
            config = ProjectionNeuronConfig(
                num_receptors=6,
                num_projection_neurons=100,
                device=self.device
            )
            
            competitive_layer = CompetitiveProjectionLayer(config).to(self.device)
            receptor_input = torch.randn(3, 6).to(self.device)
            
            start_time = time.time()
            comp_spikes, comp_stats = competitive_layer(receptor_input, duration=50)
            process_time = time.time() - start_time
            
            projection_results['competitive_layer'] = {
                'spike_shape': comp_spikes.shape,
                'competition_level': comp_stats['competition_level'],
                'winner_percentage': comp_stats['winner_percentage'],
                'processing_time': process_time,
                'status': 'SUCCESS'
            }
            
            print(f"    ✅ Competitive layer: competition={comp_stats['competition_level']:.3f}, "
                  f"winners={comp_stats['winner_percentage']:.3f}")
            
        except Exception as e:
            projection_results['competitive_layer'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            print(f"    ❌ Competitive layer failed: {e}")
            
        # Test moth-inspired network
        print("  Testing moth-inspired network...")
        try:
            moth_network = create_moth_inspired_projection_network(
                num_sensors=6, device=self.device
            ).to(self.device)
            
            receptor_input = torch.randn(2, 6).to(self.device)
            
            start_time = time.time()
            moth_outputs = moth_network(receptor_input, duration=30)
            process_time = time.time() - start_time
            
            # Get final layer statistics
            final_layer = f'layer_{moth_network.num_layers - 1}'
            final_stats = moth_outputs[final_layer]
            
            projection_results['moth_inspired'] = {
                'num_layers': moth_network.num_layers,
                'final_firing_rate': final_stats['firing_rates'].mean().item(),
                'population_vector_norm': torch.norm(final_stats['population_vector']).item(),
                'processing_time': process_time,
                'status': 'SUCCESS'
            }
            
            print(f"    ✅ Moth-inspired: {moth_network.num_layers} layers, "
                  f"rate={final_stats['firing_rates'].mean().item():.3f}")
            
        except Exception as e:
            projection_results['moth_inspired'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            print(f"    ❌ Moth-inspired failed: {e}")
            
        self.results['projection_neurons'] = projection_results
        return projection_results
        
    def validate_kenyon_cells(self) -> Dict:
        """Test Kenyon cell implementations."""
        print("\n🧠 Validating Kenyon Cells...")
        
        kenyon_results = {}
        
        # Test basic Kenyon cell layer
        print("  Testing KenyonCellLayer...")
        try:
            from bioneuro_olfactory.models.kenyon.kenyon_cells import (
                KenyonCellConfig, create_standard_kenyon_network
            )
            
            # Create smaller network for testing
            kc_layer = create_standard_kenyon_network(
                num_projection_inputs=100,
                num_kenyon_cells=500,
                sparsity_target=0.05,
                device=self.device
            ).to(self.device)
            
            # Synthetic PN input
            pn_spikes = torch.rand(2, 30, 100).to(self.device) > 0.8  # Sparse spikes
            
            start_time = time.time()
            kc_spikes, kc_potentials = kc_layer(pn_spikes.float())
            process_time = time.time() - start_time
            
            # Analyze sparsity
            sparsity_stats = kc_layer.get_sparsity_statistics(kc_spikes)
            
            kenyon_results['basic_layer'] = {
                'spike_shape': kc_spikes.shape,
                'achieved_sparsity': sparsity_stats['population_sparsity'],
                'target_sparsity': sparsity_stats['target_sparsity'],
                'pattern_separation': sparsity_stats['pattern_separation'],
                'processing_time': process_time,
                'status': 'SUCCESS'
            }
            
            print(f"    ✅ Basic KC layer: sparsity={sparsity_stats['population_sparsity']:.3f}, "
                  f"separation={sparsity_stats['pattern_separation']:.3f}")
            
        except Exception as e:
            kenyon_results['basic_layer'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            print(f"    ❌ Basic KC layer failed: {e}")
            
        # Test adaptive Kenyon cells
        print("  Testing AdaptiveKenyonCells...")
        try:
            from bioneuro_olfactory.models.kenyon.kenyon_cells import create_adaptive_kenyon_network
            
            adaptive_kc = create_adaptive_kenyon_network(
                num_projection_inputs=100,
                num_kenyon_cells=500,
                device=self.device
            ).to(self.device)
            
            pn_spikes = torch.rand(2, 30, 100).to(self.device) > 0.8
            
            start_time = time.time()
            adaptive_spikes, adaptation_stats = adaptive_kc(pn_spikes.float())
            process_time = time.time() - start_time
            
            kenyon_results['adaptive_layer'] = {
                'spike_shape': adaptive_spikes.shape,
                'homeostatic_adjustment': adaptation_stats['homeostatic_adjustment'],
                'competition_dynamics': adaptation_stats['competition_dynamics'],
                'plasticity_state': adaptation_stats['plasticity_state'],
                'processing_time': process_time,
                'status': 'SUCCESS'
            }
            
            print(f"    ✅ Adaptive KC: homeostatic_adj={adaptation_stats['homeostatic_adjustment']['mean_activity']:.3f}")
            
        except Exception as e:
            kenyon_results['adaptive_layer'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            print(f"    ❌ Adaptive KC failed: {e}")
            
        # Test sparse coding quality analysis
        print("  Testing sparse coding analysis...")
        try:
            # Use the basic layer results for analysis
            if 'basic_layer' in kenyon_results and kenyon_results['basic_layer']['status'] == 'SUCCESS':
                # Recreate layer for analysis
                kc_layer = create_standard_kenyon_network(
                    num_projection_inputs=100,
                    num_kenyon_cells=500,
                    device=self.device
                ).to(self.device)
                
                pn_input = torch.rand(3, 30, 100).to(self.device) > 0.8
                kc_output, _ = kc_layer(pn_input.float())
                
                from bioneuro_olfactory.models.kenyon.kenyon_cells import KenyonCellConfig
                config = KenyonCellConfig(
                    num_projection_inputs=100,
                    num_kenyon_cells=500,
                    sparsity_target=0.05
                )
                
                quality_analysis = analyze_sparse_coding_quality(
                    pn_input.float(), kc_output, config
                )
                
                kenyon_results['quality_analysis'] = {
                    'sparsity_quality': quality_analysis['sparsity_metrics']['sparsity_quality'],
                    'separation_enhancement': quality_analysis['pattern_separation']['separation_enhancement'],
                    'information_preservation': quality_analysis['information_processing']['information_preservation'],
                    'status': 'SUCCESS'
                }
                
                print(f"    ✅ Quality analysis: sparsity_quality={quality_analysis['sparsity_metrics']['sparsity_quality']:.3f}")
                
        except Exception as e:
            kenyon_results['quality_analysis'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            print(f"    ❌ Quality analysis failed: {e}")
            
        self.results['kenyon_cells'] = kenyon_results
        return kenyon_results
        
    def validate_decision_layers(self) -> Dict:
        """Test decision layer implementations."""
        print("\n🎯 Validating Decision Layers...")
        
        decision_results = {}
        
        # Test basic decision layer
        print("  Testing DecisionLayer...")
        try:
            from bioneuro_olfactory.models.mushroom_body.decision_layer import (
                create_standard_decision_layer
            )
            
            decision_layer = create_standard_decision_layer(
                num_kenyon_inputs=500,
                num_output_neurons=10,
                device=self.device
            ).to(self.device)
            
            # Synthetic KC input (sparse)
            kc_input = torch.rand(3, 500).to(self.device) * 0.1  # Sparse activity
            
            start_time = time.time()
            decision_output = decision_layer(kc_input)
            process_time = time.time() - start_time
            
            decision_results['basic_layer'] = {
                'decision_shape': decision_output['decisions'].shape,
                'confidence_mean': decision_output['confidence'].mean().item(),
                'max_activation_mean': decision_output['max_activation'].mean().item(),
                'processing_time': process_time,
                'status': 'SUCCESS'
            }
            
            print(f"    ✅ Basic decision: confidence={decision_output['confidence'].mean().item():.3f}")
            
        except Exception as e:
            decision_results['basic_layer'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            print(f"    ❌ Basic decision failed: {e}")
            
        # Test adaptive decision layer
        print("  Testing AdaptiveDecisionLayer...")
        try:
            from bioneuro_olfactory.models.mushroom_body.decision_layer import (
                create_adaptive_decision_layer
            )
            
            adaptive_decision = create_adaptive_decision_layer(
                num_kenyon_inputs=500,
                num_output_neurons=10,
                device=self.device
            ).to(self.device)
            
            kc_input = torch.rand(3, 500).to(self.device) * 0.1
            context = torch.randn(3, 128).to(self.device)  # Context vector
            
            start_time = time.time()
            adaptive_output = adaptive_decision(kc_input, context=context)
            process_time = time.time() - start_time
            
            decision_results['adaptive_layer'] = {
                'final_decisions_shape': adaptive_output['final_decisions'].shape,
                'uncertainty_mean': adaptive_output['uncertainty']['total'].mean().item(),
                'meta_adjustment_norm': torch.norm(adaptive_output['meta_adjustments']).item(),
                'processing_time': process_time,
                'status': 'SUCCESS'
            }
            
            print(f"    ✅ Adaptive decision: uncertainty={adaptive_output['uncertainty']['total'].mean().item():.3f}")
            
        except Exception as e:
            decision_results['adaptive_layer'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            print(f"    ❌ Adaptive decision failed: {e}")
            
        # Test decision dynamics analysis
        print("  Testing decision analysis...")
        try:
            if 'basic_layer' in decision_results and decision_results['basic_layer']['status'] == 'SUCCESS':
                # Recreate for analysis
                decision_layer = create_standard_decision_layer(
                    num_kenyon_inputs=500,
                    num_output_neurons=10,
                    device=self.device
                ).to(self.device)
                
                kc_input = torch.rand(5, 500).to(self.device) * 0.1
                decision_output = decision_layer(kc_input)
                
                # Create synthetic ground truth
                ground_truth = torch.randint(0, 10, (5,)).to(self.device)
                
                dynamics_analysis = analyze_decision_dynamics(
                    kc_input, decision_output, ground_truth
                )
                
                decision_results['dynamics_analysis'] = {
                    'decision_entropy': dynamics_analysis['decision_statistics']['decision_entropy'],
                    'competition_strength': dynamics_analysis['competition_analysis']['competition_strength'],
                    'accuracy': dynamics_analysis['performance']['accuracy'],
                    'status': 'SUCCESS'
                }
                
                print(f"    ✅ Decision analysis: entropy={dynamics_analysis['decision_statistics']['decision_entropy']:.3f}")
                
        except Exception as e:
            decision_results['dynamics_analysis'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            print(f"    ❌ Decision analysis failed: {e}")
            
        self.results['decision_layers'] = decision_results
        return decision_results
        
    def validate_integration(self) -> Dict:
        """Test complete end-to-end integration."""
        print("\n🔗 Validating End-to-End Integration...")
        
        integration_results = {}
        
        try:
            # Create complete pipeline
            print("  Building complete neuromorphic pipeline...")
            
            # 1. Fusion layer
            fusion_network = OlfactoryFusionSNN(
                num_chemical_sensors=6,
                num_audio_features=128,
                fusion_strategy='hierarchical',
                device=self.device
            ).to(self.device)
            
            # 2. Projection neurons  
            from bioneuro_olfactory.models.projection.projection_neurons import (
                create_moth_inspired_projection_network
            )
            projection_net = create_moth_inspired_projection_network(
                num_sensors=6, device=self.device
            ).to(self.device)
            
            # 3. Kenyon cells
            from bioneuro_olfactory.models.kenyon.kenyon_cells import create_standard_kenyon_network
            kenyon_layer = create_standard_kenyon_network(
                num_projection_inputs=1000,
                num_kenyon_cells=2000,
                device=self.device
            ).to(self.device)
            
            # 4. Decision layer
            from bioneuro_olfactory.models.mushroom_body.decision_layer import (
                create_adaptive_decision_layer
            )
            decision_layer = create_adaptive_decision_layer(
                num_kenyon_inputs=2000,
                num_output_neurons=10,
                device=self.device
            ).to(self.device)
            
            print("  Testing complete pipeline...")
            
            # Test data
            chemical_input = torch.randn(2, 6).to(self.device)
            audio_input = torch.randn(2, 128).to(self.device)
            
            start_time = time.time()
            
            # Step 1: Multi-modal fusion
            fusion_output = fusion_network(chemical_input, audio_input)
            
            # Step 2: Convert to receptor-like input for projection neurons
            receptor_input = fusion_output['fused_features'][:, :6]  # Take first 6 features
            
            # Step 3: Process through projection neurons
            projection_outputs = projection_net(receptor_input, duration=30)
            final_pn_layer = f'layer_{projection_net.num_layers - 1}'
            pn_spikes = projection_outputs[final_pn_layer]['spikes']
            
            # Reshape to match KC input expectations
            pn_rates = torch.mean(pn_spikes, dim=1)  # [batch, neurons]
            
            # Create synthetic PN spike trains for KC layer
            duration = 50
            pn_spike_trains = torch.zeros(2, duration, 1000).to(self.device)
            for t in range(duration):
                spikes = torch.rand(2, 1000).to(self.device) < (pn_rates[:, :1000] * 0.1)
                pn_spike_trains[:, t, :] = spikes.float()
            
            # Step 4: Process through Kenyon cells
            kc_spikes, _ = kenyon_layer(pn_spike_trains)
            kc_rates = torch.mean(kc_spikes, dim=1)  # [batch, kc_neurons]
            
            # Step 5: Final decision
            decision_output = decision_layer(kc_rates)
            
            total_time = time.time() - start_time
            
            # Interpret final results
            final_results = []
            for i in range(2):
                gas_detection = {
                    'gas_probabilities': fusion_output['gas_probabilities'][i],
                    'concentration': fusion_output['concentration'][i],
                    'gas_type_index': decision_output['final_decisions'][i],
                    'hazard_probability': decision_output['confidence'][i]
                }
                interpretation = interpret_gas_detection(gas_detection)
                final_results.append(interpretation)
            
            integration_results['complete_pipeline'] = {
                'fusion_shape': fusion_output['fused_features'].shape,
                'projection_layers': projection_net.num_layers,
                'kc_sparsity': torch.mean((kc_spikes > 0.01).float()).item(),
                'decision_confidence': decision_output['confidence'].mean().item(),
                'final_interpretations': final_results,
                'total_processing_time': total_time,
                'status': 'SUCCESS'
            }
            
            print(f"    ✅ Complete pipeline: {total_time:.3f}s, "
                  f"confidence={decision_output['confidence'].mean().item():.3f}")
            print(f"    Gas detections: {[r['detected_gas'] for r in final_results]}")
            
        except Exception as e:
            integration_results['complete_pipeline'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            print(f"    ❌ Complete pipeline failed: {e}")
            
        self.results['integration_tests'] = integration_results
        return integration_results
        
    def run_performance_benchmarks(self) -> Dict:
        """Run performance benchmarks."""
        print("\n⚡ Running Performance Benchmarks...")
        
        performance_results = {}
        
        # Throughput test
        print("  Testing throughput...")
        try:
            network = OlfactoryFusionSNN(device=self.device).to(self.device)
            
            batch_sizes = [1, 10, 50, 100]
            throughput_results = {}
            
            for batch_size in batch_sizes:
                chemical_input = torch.randn(batch_size, 6).to(self.device)
                audio_input = torch.randn(batch_size, 128).to(self.device)
                
                # Warmup
                for _ in range(5):
                    _ = network(chemical_input, audio_input)
                
                # Benchmark
                start_time = time.time()
                num_runs = 20
                for _ in range(num_runs):
                    _ = network(chemical_input, audio_input)
                    
                total_time = time.time() - start_time
                samples_per_second = (batch_size * num_runs) / total_time
                
                throughput_results[f'batch_{batch_size}'] = {
                    'samples_per_second': samples_per_second,
                    'latency_per_sample': total_time / (batch_size * num_runs)
                }
                
                print(f"    Batch {batch_size}: {samples_per_second:.1f} samples/sec")
                
            performance_results['throughput'] = throughput_results
            
        except Exception as e:
            performance_results['throughput'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            print(f"    ❌ Throughput test failed: {e}")
            
        # Memory usage test
        print("  Testing memory usage...")
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                initial_memory = torch.cuda.memory_allocated()
                
                # Create large network
                large_network = OlfactoryFusionSNN(
                    num_projection_neurons=2000,
                    num_kenyon_cells=10000,
                    device=self.device
                ).to(self.device)
                
                large_input_chemical = torch.randn(100, 6).to(self.device)
                large_input_audio = torch.randn(100, 128).to(self.device)
                
                _ = large_network(large_input_chemical, large_input_audio)
                
                peak_memory = torch.cuda.memory_allocated()
                memory_usage = (peak_memory - initial_memory) / 1024**2  # MB
                
                performance_results['memory_usage'] = {
                    'peak_memory_mb': memory_usage,
                    'memory_per_sample_kb': (memory_usage * 1024) / 100,
                    'status': 'SUCCESS'
                }
                
                print(f"    Memory usage: {memory_usage:.1f} MB ({memory_usage*1024/100:.1f} KB/sample)")
                
            else:
                performance_results['memory_usage'] = {
                    'status': 'SKIPPED',
                    'reason': 'CUDA not available'
                }
                print("    Memory test skipped (CPU only)")
                
        except Exception as e:
            performance_results['memory_usage'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            print(f"    ❌ Memory test failed: {e}")
            
        self.results['performance_metrics'] = performance_results
        return performance_results
        
    def generate_report(self) -> Dict:
        """Generate comprehensive validation report."""
        print("\n📊 Generating Validation Report...")
        
        # Count successes and failures
        total_tests = 0
        successful_tests = 0
        
        for category, tests in self.results.items():
            if isinstance(tests, dict):
                for test_name, result in tests.items():
                    total_tests += 1
                    if isinstance(result, dict) and result.get('status') == 'SUCCESS':
                        successful_tests += 1
                        
        success_rate = successful_tests / total_tests if total_tests > 0 else 0
        
        # Generate summary
        summary = {
            'validation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'device_used': self.device,
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate': success_rate,
            'overall_status': 'PASS' if success_rate >= 0.8 else 'FAIL'
        }
        
        # Performance highlights
        if 'performance_metrics' in self.results:
            perf = self.results['performance_metrics']
            if 'throughput' in perf and 'batch_1' in perf['throughput']:
                summary['peak_throughput'] = max(
                    result['samples_per_second'] 
                    for result in perf['throughput'].values()
                    if isinstance(result, dict) and 'samples_per_second' in result
                )
                
        # Component highlights
        highlights = []
        
        if 'fusion_strategies' in self.results:
            fusion_success = sum(
                1 for result in self.results['fusion_strategies'].values()
                if isinstance(result, dict) and result.get('status') == 'SUCCESS'
            )
            highlights.append(f"✅ {fusion_success}/5 fusion strategies working")
            
        if 'integration_tests' in self.results:
            if self.results['integration_tests'].get('complete_pipeline', {}).get('status') == 'SUCCESS':
                highlights.append("✅ End-to-end pipeline operational")
                
        summary['highlights'] = highlights
        
        report = {
            'summary': summary,
            'detailed_results': self.results
        }
        
        print(f"📈 Validation Summary:")
        print(f"  Total Tests: {total_tests}")
        print(f"  Successful: {successful_tests}")
        print(f"  Success Rate: {success_rate:.1%}")
        print(f"  Overall Status: {summary['overall_status']}")
        
        for highlight in highlights:
            print(f"  {highlight}")
            
        return report
        
    def save_results(self, filepath: str = "advanced_validation_results.json"):
        """Save validation results to file."""
        report = self.generate_report()
        
        # Convert tensors to lists for JSON serialization
        def convert_tensors(obj):
            if isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            elif isinstance(obj, dict):
                return {k: convert_tensors(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_tensors(item) for item in obj]
            else:
                return obj
                
        serializable_report = convert_tensors(report)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_report, f, indent=2)
            
        print(f"📄 Results saved to: {filepath}")
        return filepath


def main():
    """Run the advanced validation suite."""
    print("🚀 Advanced Neuromorphic Validation Suite")
    print("=" * 50)
    
    validator = AdvancedNeuromorphicValidator()
    
    # Run all validation tests
    validator.validate_fusion_strategies()
    validator.validate_projection_neurons()
    validator.validate_kenyon_cells()
    validator.validate_decision_layers()
    validator.validate_integration()
    validator.run_performance_benchmarks()
    
    # Generate and save report
    report = validator.generate_report()
    validator.save_results()
    
    print("\n🎉 Advanced validation complete!")
    
    # Return final status
    return report['summary']['overall_status'] == 'PASS'


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)