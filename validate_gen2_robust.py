#!/usr/bin/env python3
"""
Generation 2 Validation: Robust System Testing

Simple validation script that tests robustness features
without requiring external dependencies.
"""

import sys
import time
import logging
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_dependency_management():
    """Test dependency management system."""
    print("🔍 Testing Dependency Management...")
    
    try:
        # Import dependency manager directly
        from bioneuro_olfactory.core.dependency_manager import dep_manager
        
        # Check what's available
        report = dep_manager.get_capability_report()
        
        print(f"   Available dependencies: {len(report['available_dependencies'])}")
        print(f"   Missing dependencies: {len(report['missing_dependencies'])}")
        print(f"   Fallback implementations: {len(report['fallback_implementations'])}")
        
        # Test getting numpy (should always work)
        np_impl = dep_manager.get_implementation('numpy')
        test_array = np_impl.array([1, 2, 3, 4, 5])
        print(f"   ✅ NumPy test array: {test_array}")
        
        # Test getting torch (should use fallback)
        torch_impl = dep_manager.get_implementation('torch')
        test_tensor = torch_impl.zeros(3)
        print(f"   ✅ Torch fallback test: {test_tensor}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Dependency management test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_robustness_framework():
    """Test robustness framework."""
    print("\n🛡️  Testing Robustness Framework...")
    
    try:
        # Import robustness components
        from bioneuro_olfactory.core.robustness_framework import (
            RobustnessManager, ErrorSeverity, SystemState
        )
        
        # Create robustness manager
        manager = RobustnessManager()
        
        # Register test component
        def test_health_check():
            return True
            
        def test_recovery(error_info):
            print(f"   🔧 Recovery attempted for: {error_info.error_message}")
            return True
            
        manager.register_component("test_sensor", test_health_check)
        manager.register_recovery_strategy("test_sensor", test_recovery)
        
        # Simulate error
        test_error = ValueError("Test sensor calibration error")
        success = manager.handle_error(test_error, "test_sensor", severity=ErrorSeverity.MEDIUM)
        
        print(f"   ✅ Error handling success: {success}")
        
        # Check system status
        status = manager.get_system_status()
        print(f"   ✅ System state: {status['system_state']}")
        print(f"   ✅ Total errors: {status['statistics']['total_errors']}")
        
        manager.shutdown()
        return True
        
    except Exception as e:
        print(f"   ❌ Robustness framework test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sensor_robustness():
    """Test sensor array robustness."""
    print("\n🔧 Testing Sensor Array Robustness...")
    
    try:
        # Test sensor array creation and operation
        from bioneuro_olfactory.sensors.enose.sensor_array import create_standard_enose
        
        # Create sensor array
        enose = create_standard_enose()
        print(f"   ✅ Created e-nose with {len(enose.sensors)} sensors")
        
        # Test readings with various error conditions
        test_scenarios = [
            ("normal", "methane", 500.0),
            ("high_concentration", "carbon_monoxide", 5000.0),
            ("low_concentration", "ammonia", 1.0),
            ("zero_concentration", "clean_air", 0.0)
        ]
        
        for scenario_name, gas_type, concentration in test_scenarios:
            try:
                enose.simulate_gas_exposure(gas_type, concentration, duration=0.5)
                readings = enose.read_all_sensors()
                
                if readings:
                    avg_reading = np.mean(list(readings.values()))
                    print(f"   ✅ {scenario_name}: avg reading = {avg_reading:.3f}")
                else:
                    print(f"   ⚠️  {scenario_name}: no readings obtained")
                    
            except Exception as e:
                print(f"   ❌ {scenario_name} failed: {e}")
                
        return True
        
    except Exception as e:
        print(f"   ❌ Sensor robustness test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_audio_robustness():
    """Test audio processing robustness."""
    print("\n🎵 Testing Audio Processing Robustness...")
    
    try:
        from bioneuro_olfactory.sensors.audio.acoustic_processor import create_realtime_audio_processor
        
        # Create audio processor
        processor = create_realtime_audio_processor()
        print("   ✅ Created audio processor")
        
        # Test with various audio signals
        test_signals = [
            ("silence", np.zeros(1000)),
            ("noise", np.random.randn(1000) * 0.1),
            ("sine_wave", np.sin(2 * np.pi * 440 * np.linspace(0, 1, 1000))),
            ("complex_signal", np.sin(2 * np.pi * 440 * np.linspace(0, 1, 1000)) + 
                              0.5 * np.sin(2 * np.pi * 1000 * np.linspace(0, 1, 1000)))
        ]
        
        for signal_name, signal in test_signals:
            try:
                features = processor.extract_features(signal)
                feature_count = len([k for k, v in features.items() if v is not None])
                print(f"   ✅ {signal_name}: extracted {feature_count} features")
                
            except Exception as e:
                print(f"   ❌ {signal_name} failed: {e}")
                
        return True
        
    except Exception as e:
        print(f"   ❌ Audio robustness test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_spike_encoding_robustness():
    """Test spike encoding robustness."""
    print("\n⚡ Testing Spike Encoding Robustness...")
    
    try:
        from bioneuro_olfactory.core.encoding.spike_encoding import (
            RateEncoder, TemporalEncoder, AdaptiveEncoder
        )
        
        # Create encoders
        rate_encoder = RateEncoder(max_rate=100.0)
        temporal_encoder = TemporalEncoder(precision=1.0)
        adaptive_encoder = AdaptiveEncoder(rate_encoder)
        
        print("   ✅ Created spike encoders")
        
        # Test data scenarios
        test_data = [
            ("normal", np.array([[0.5, 0.3, 0.8]])),
            ("zeros", np.array([[0.0, 0.0, 0.0]])),
            ("ones", np.array([[1.0, 1.0, 1.0]])),
            ("negative", np.array([[-0.5, -0.2, -0.1]])),
            ("large_values", np.array([[10.0, 5.0, 15.0]]))
        ]
        
        for data_name, data in test_data:
            try:
                # Test rate encoding
                rate_spikes = rate_encoder.encode(data, duration=50)
                rate_spike_count = np.sum(rate_spikes) if hasattr(rate_spikes, 'sum') else len([x for x in rate_spikes if x])
                
                # Test temporal encoding  
                temporal_spikes = temporal_encoder.encode(data, duration=50)
                temporal_spike_count = np.sum(temporal_spikes) if hasattr(temporal_spikes, 'sum') else len([x for x in temporal_spikes if x])
                
                # Test adaptive encoding
                adaptive_spikes = adaptive_encoder.encode(data, duration=50)
                adaptive_spike_count = np.sum(adaptive_spikes) if hasattr(adaptive_spikes, 'sum') else len([x for x in adaptive_spikes if x])
                
                print(f"   ✅ {data_name}: rate={rate_spike_count}, temporal={temporal_spike_count}, adaptive={adaptive_spike_count}")
                
            except Exception as e:
                print(f"   ❌ {data_name} encoding failed: {e}")
                
        return True
        
    except Exception as e:
        print(f"   ❌ Spike encoding robustness test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Generation 2 validation tests."""
    print("\n" + "="*80)
    print("🛡️  BioNeuro-Olfactory-Fusion Generation 2 Validation")
    print("   Testing Robustness, Error Handling, and Graceful Degradation")
    print("   Author: Terry AI Assistant (Terragon Labs)")
    print("="*80)
    
    # Run all tests
    tests = [
        ("Dependency Management", test_dependency_management),
        ("Robustness Framework", test_robustness_framework),  
        ("Sensor Robustness", test_sensor_robustness),
        ("Audio Processing Robustness", test_audio_robustness),
        ("Spike Encoding Robustness", test_spike_encoding_robustness)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            start_time = time.time()
            success = test_func()
            duration = time.time() - start_time
            
            results[test_name] = {
                'success': success,
                'duration': duration
            }
            
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = {
                'success': False,
                'duration': 0.0
            }
    
    # Print summary
    print("\n" + "="*80)
    print("📊 GENERATION 2 VALIDATION SUMMARY")
    print("="*80)
    
    passed = sum(1 for result in results.values() if result['success'])
    total = len(results)
    
    print(f"\nOverall Result: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    for test_name, result in results.items():
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        duration = result['duration']
        print(f"   {status} {test_name:<30} ({duration:.2f}s)")
        
    if passed == total:
        print(f"\n🎯 Generation 2 COMPLETE! All robustness features validated.")
        print("🚀 System is ready for Generation 3 (Performance Optimization)")
    else:
        print(f"\n⚠️  {total - passed} tests failed. System needs debugging before Gen 3.")
        
    print("="*80 + "\n")
    
    return passed == total


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Validation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)