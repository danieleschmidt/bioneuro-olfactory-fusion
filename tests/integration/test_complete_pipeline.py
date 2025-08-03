"""Integration tests for complete neuromorphic gas detection pipeline."""

import pytest
import torch
import numpy as np
from datetime import datetime

from bioneuro_olfactory import (
    OlfactoryFusionSNN,
    create_moth_inspired_network,
    create_efficient_network,
    create_standard_enose
)
from bioneuro_olfactory.data.database.connection import DatabaseManager
from bioneuro_olfactory.data.repositories.experiment_repository import ExperimentRepository
from bioneuro_olfactory.data.database.models import ExperimentModel


class TestCompletePipeline:
    """Test complete gas detection pipeline integration."""
    
    @pytest.fixture
    def temp_db(self, tmp_path):
        """Create temporary database for testing."""
        db_path = tmp_path / "test.db"
        from bioneuro_olfactory.data.database.connection import DatabaseConfig
        
        config = DatabaseConfig(
            db_type="sqlite",
            sqlite_path=str(db_path)
        )
        
        db_manager = DatabaseManager(config)
        yield db_manager
        db_manager.close()
        
    @pytest.fixture
    def sample_experiment(self, temp_db):
        """Create sample experiment for testing."""
        repo = ExperimentRepository(temp_db)
        
        experiment = ExperimentModel(
            name="Test Gas Detection",
            description="Integration test experiment",
            config={
                "num_sensors": 6,
                "target_gases": ["methane", "carbon_monoxide"],
                "fusion_strategy": "hierarchical"
            }
        )
        
        experiment_id = repo.create(experiment)
        experiment.id = experiment_id
        return experiment
        
    def test_end_to_end_detection(self, sample_experiment):
        """Test complete end-to-end gas detection pipeline."""
        # Create network
        network = OlfactoryFusionSNN(
            num_chemical_sensors=6,
            num_audio_features=64,
            num_projection_neurons=100,
            num_kenyon_cells=500,
            fusion_strategy='hierarchical'
        )
        
        # Simulate sensor data
        chemical_data = torch.rand(1, 6) * 2.0  # Chemical sensor readings
        audio_data = torch.rand(1, 64) * 1.0    # Audio features
        
        # Process through network
        result = network.process(
            chemical_input=chemical_data,
            audio_input=audio_data,
            duration=100
        )
        
        # Verify outputs
        assert 'projection_spikes' in result
        assert 'kenyon_spikes' in result
        assert 'fused_output' in result
        assert 'network_activity' in result
        
        # Check spike train shapes
        pn_spikes = result['projection_spikes']
        kc_spikes = result['kenyon_spikes']
        
        assert pn_spikes.shape == (1, 100, 100)  # batch, neurons, time
        assert kc_spikes.shape == (1, 500, 100)  # batch, neurons, time
        
        # Check network activity statistics
        activity = result['network_activity']
        assert 'projection_rates' in activity
        assert 'kenyon_sparsity' in activity
        
        sparsity_stats = activity['kenyon_sparsity']
        assert 'sparsity_ratio' in sparsity_stats
        assert 0 <= sparsity_stats['sparsity_ratio'] <= 1
        
    def test_sensor_integration(self):
        """Test integration with electronic nose sensors."""
        # Create sensor array
        enose = create_standard_enose()
        
        # Simulate gas exposure
        readings_over_time = enose.simulate_gas_exposure(
            gas_type="methane",
            concentration=1500.0,  # ppm
            duration=10.0
        )
        
        assert len(readings_over_time) > 0
        
        # Check sensor responses
        first_reading = readings_over_time[0]
        assert len(first_reading) == 6  # Standard e-nose has 6 sensors
        
        # Get sensor status
        status = enose.get_sensor_status()
        assert len(status) == 6
        
        for sensor_name, sensor_info in status.items():
            assert 'type' in sensor_info
            assert 'target_gases' in sensor_info
            assert 'is_calibrated' in sensor_info
            
    def test_data_persistence_integration(self, temp_db, sample_experiment):
        """Test integration with data persistence layer."""
        experiment_id = sample_experiment.id
        
        # Store sensor readings
        sensor_readings = []
        for i in range(10):
            reading_id = temp_db.store_sensor_reading(
                experiment_id=experiment_id,
                sensor_type="MQ2",
                sensor_id=f"sensor_{i % 3}",
                raw_value=100.0 + i * 10,
                calibrated_value=50.0 + i * 5,
                temperature=25.0,
                humidity=50.0,
                metadata={"simulation": True}
            )
            sensor_readings.append(reading_id)
            
        # Store network states
        network_states = []
        for i in range(5):
            state_data = np.random.rand(100, 50)  # Random network state
            state_id = temp_db.store_network_state(
                experiment_id=experiment_id,
                network_type="spiking_neural_network",
                layer_name="projection_neurons",
                state_data=state_data,
                sparsity_level=0.05,
                firing_rate=15.0 + i,
                metadata={"timestep": i}
            )
            network_states.append(state_id)
            
        # Store gas detection events
        detection_events = []
        gas_types = ["methane", "carbon_monoxide"]
        for i in range(3):
            event_id = temp_db.store_gas_detection_event(
                experiment_id=experiment_id,
                gas_type=gas_types[i % 2],
                concentration=500.0 + i * 100,
                confidence=0.85 + i * 0.05,
                alert_level="warning" if i < 2 else "critical",
                response_time=50.0 + i * 10,
                sensor_fusion_method="hierarchical",
                metadata={"detection_round": i}
            )
            detection_events.append(event_id)
            
        # Retrieve complete experiment data
        experiment_data = temp_db.get_experiment_data(experiment_id)
        
        assert len(experiment_data['sensor_data']) == 10
        assert len(experiment_data['network_states']) == 5
        assert len(experiment_data['detection_events']) == 3
        
        # Verify data integrity
        sensor_data = experiment_data['sensor_data'][0]
        assert sensor_data['sensor_type'] == "MQ2"
        assert sensor_data['experiment_id'] == experiment_id
        
        network_state = experiment_data['network_states'][0]
        assert network_state['layer_name'] == "projection_neurons"
        assert network_state['sparsity_level'] == 0.05
        
        detection_event = experiment_data['detection_events'][0]
        assert detection_event['gas_type'] in gas_types
        assert detection_event['confidence'] >= 0.85
        
    def test_multimodal_fusion_strategies(self):
        """Test different fusion strategies."""
        fusion_strategies = ['early', 'attention', 'hierarchical']
        
        chemical_data = torch.rand(2, 6) * 1.5
        audio_data = torch.rand(2, 64) * 1.0
        
        results = {}
        for strategy in fusion_strategies:
            network = OlfactoryFusionSNN(
                num_chemical_sensors=6,
                num_audio_features=64,
                num_projection_neurons=50,
                num_kenyon_cells=200,
                fusion_strategy=strategy
            )
            
            result = network.process(
                chemical_input=chemical_data,
                audio_input=audio_data,
                duration=50
            )
            
            results[strategy] = result
            
        # Each strategy should produce different fusion outputs
        for i, strategy1 in enumerate(fusion_strategies):
            for strategy2 in fusion_strategies[i+1:]:
                fused1 = results[strategy1]['fused_output']
                fused2 = results[strategy2]['fused_output']
                
                # Different strategies should produce different outputs
                assert not torch.equal(fused1, fused2)
                
    def test_scalability(self):
        """Test system scalability with different network sizes."""
        configs = [
            # Small network
            {
                'num_chemical_sensors': 3,
                'num_projection_neurons': 50,
                'num_kenyon_cells': 200
            },
            # Medium network
            {
                'num_chemical_sensors': 6,
                'num_projection_neurons': 200,
                'num_kenyon_cells': 1000
            },
            # Large network
            {
                'num_chemical_sensors': 10,
                'num_projection_neurons': 500,
                'num_kenyon_cells': 2500
            }
        ]
        
        for config in configs:
            network = OlfactoryFusionSNN(**config)
            
            # Test with appropriate input sizes
            chemical_data = torch.rand(1, config['num_chemical_sensors'])
            audio_data = torch.rand(1, 64)
            
            result = network.process(
                chemical_input=chemical_data,
                audio_input=audio_data,
                duration=50
            )
            
            # Verify correct output shapes
            pn_spikes = result['projection_spikes']
            kc_spikes = result['kenyon_spikes']
            
            assert pn_spikes.shape[1] == config['num_projection_neurons']
            assert kc_spikes.shape[1] == config['num_kenyon_cells']
            
    def test_convenience_functions(self):
        """Test convenience network creation functions."""
        # Test moth-inspired network
        moth_network = create_moth_inspired_network(num_sensors=6)
        
        assert moth_network.num_chemical_sensors == 6
        assert moth_network.num_projection_neurons == 1000
        assert moth_network.num_kenyon_cells == 50000
        assert moth_network.fusion_strategy == 'hierarchical'
        
        # Test efficient network
        efficient_network = create_efficient_network(num_sensors=4)
        
        assert efficient_network.num_chemical_sensors == 4
        assert efficient_network.num_projection_neurons == 200
        assert efficient_network.num_kenyon_cells == 1000
        assert efficient_network.fusion_strategy == 'early'
        
        # Both should be functional
        test_data = torch.rand(1, 6)
        audio_data = torch.rand(1, 128)
        
        # Test moth network
        result_moth = moth_network.process(test_data, audio_data, duration=50)
        assert 'projection_spikes' in result_moth
        
        # Test efficient network  
        test_data_small = torch.rand(1, 4)
        audio_data_small = torch.rand(1, 64)
        result_efficient = efficient_network.process(test_data_small, audio_data_small, duration=50)
        assert 'projection_spikes' in result_efficient


class TestPerformanceBenchmarks:
    """Performance benchmarks for the complete system."""
    
    def test_processing_latency(self):
        """Test processing latency requirements."""
        import time
        
        network = create_efficient_network(num_sensors=6)
        
        chemical_data = torch.rand(1, 6)
        audio_data = torch.rand(1, 64)
        
        # Measure processing time
        start_time = time.time()
        
        for _ in range(10):  # Multiple runs for average
            result = network.process(
                chemical_input=chemical_data,
                audio_input=audio_data,
                duration=100
            )
            
        end_time = time.time()
        avg_time = (end_time - start_time) / 10
        
        # Should process in reasonable time (< 1 second for small network)
        assert avg_time < 1.0
        
    def test_memory_efficiency(self):
        """Test memory usage with different batch sizes."""
        network = create_efficient_network(num_sensors=6)
        
        batch_sizes = [1, 5, 10, 20]
        
        for batch_size in batch_sizes:
            chemical_data = torch.rand(batch_size, 6)
            audio_data = torch.rand(batch_size, 64)
            
            # Should handle different batch sizes
            result = network.process(
                chemical_input=chemical_data,
                audio_input=audio_data,
                duration=50
            )
            
            # Verify correct batch dimensions
            pn_spikes = result['projection_spikes']
            assert pn_spikes.shape[0] == batch_size
            
    def test_concurrent_processing(self):
        """Test concurrent processing capabilities."""
        import threading
        import queue
        
        network = create_efficient_network(num_sensors=6)
        results_queue = queue.Queue()
        
        def process_data(data_id):
            chemical_data = torch.rand(1, 6) * (data_id + 1)  # Different for each thread
            audio_data = torch.rand(1, 64)
            
            result = network.process(
                chemical_input=chemical_data,
                audio_input=audio_data,
                duration=30
            )
            
            results_queue.put((data_id, result))
            
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=process_data, args=(i,))
            threads.append(thread)
            thread.start()
            
        # Wait for completion
        for thread in threads:
            thread.join()
            
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
            
        assert len(results) == 5
        
        # Each result should be valid
        for data_id, result in results:
            assert 'projection_spikes' in result
            assert 'kenyon_spikes' in result


class TestRobustness:
    """Test system robustness and error handling."""
    
    def test_invalid_input_handling(self):
        """Test handling of invalid inputs."""
        network = OlfactoryFusionSNN(
            num_chemical_sensors=6,
            num_audio_features=64
        )
        
        # Test with wrong input dimensions
        with pytest.raises((RuntimeError, ValueError)):
            wrong_chemical = torch.rand(1, 5)  # Wrong number of sensors
            audio_data = torch.rand(1, 64)
            network.process(wrong_chemical, audio_data, duration=50)
            
        with pytest.raises((RuntimeError, ValueError)):
            chemical_data = torch.rand(1, 6)
            wrong_audio = torch.rand(1, 32)  # Wrong number of features
            network.process(chemical_data, wrong_audio, duration=50)
            
    def test_extreme_values(self):
        """Test handling of extreme input values."""
        network = create_efficient_network(num_sensors=6)
        
        # Test with extreme values
        extreme_cases = [
            (torch.zeros(1, 6), torch.zeros(1, 64)),  # All zeros
            (torch.ones(1, 6) * 1000, torch.ones(1, 64) * 1000),  # Very large
            (torch.ones(1, 6) * -100, torch.ones(1, 64) * -100),  # Negative
        ]
        
        for chemical_data, audio_data in extreme_cases:
            # Should handle gracefully without crashing
            result = network.process(
                chemical_input=chemical_data,
                audio_input=audio_data,
                duration=50
            )
            
            # Results should be finite
            assert torch.all(torch.isfinite(result['projection_spikes']))
            assert torch.all(torch.isfinite(result['kenyon_spikes']))
            
    def test_state_consistency(self):
        """Test that network state remains consistent."""
        network = create_efficient_network(num_sensors=6)
        
        chemical_data = torch.rand(1, 6)
        audio_data = torch.rand(1, 64)
        
        # Process multiple times
        results = []
        for i in range(3):
            result = network.process(
                chemical_input=chemical_data,
                audio_input=audio_data,
                duration=50
            )
            results.append(result)
            
        # Network activity should remain in reasonable ranges
        for result in results:
            activity = result['network_activity']
            sparsity = activity['kenyon_sparsity']['sparsity_ratio']
            
            assert 0 <= sparsity <= 1
            assert sparsity < 0.5  # Should maintain sparsity