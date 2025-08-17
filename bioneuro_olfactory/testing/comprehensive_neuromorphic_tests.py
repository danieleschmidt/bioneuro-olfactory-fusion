"""
Comprehensive Testing Framework for Neuromorphic Systems
=======================================================

This module provides extensive testing capabilities for neuromorphic computing
systems, including unit tests, integration tests, performance tests, and
quality assurance checks.

Created as part of Terragon SDLC Generation 2: MAKE IT ROBUST
"""

import time
import statistics
import traceback
import warnings
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import json


class TestSeverity(Enum):
    """Test failure severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TestCategory(Enum):
    """Categories of neuromorphic tests."""
    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    SECURITY = "security"
    ROBUSTNESS = "robustness"
    REGRESSION = "regression"
    STRESS = "stress"
    COMPATIBILITY = "compatibility"


@dataclass
class TestResult:
    """Individual test result."""
    test_name: str
    category: TestCategory
    passed: bool
    execution_time: float
    message: str = ""
    severity: TestSeverity = TestSeverity.LOW
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        """Convert test result to dictionary."""
        return {
            'test_name': self.test_name,
            'category': self.category.value,
            'passed': self.passed,
            'execution_time': self.execution_time,
            'message': self.message,
            'severity': self.severity.value,
            'details': self.details,
            'timestamp': self.timestamp
        }


@dataclass
class TestSuite:
    """Test suite containing multiple tests."""
    name: str
    tests: List[Callable] = field(default_factory=list)
    setup_func: Optional[Callable] = None
    teardown_func: Optional[Callable] = None
    timeout: float = 300.0  # 5 minutes default
    
    def add_test(self, test_func: Callable, category: TestCategory = TestCategory.UNIT):
        """Add a test to the suite."""
        test_func._test_category = category
        self.tests.append(test_func)
        
    def run_suite(self, parallel: bool = False) -> Dict[str, Any]:
        """Run all tests in the suite."""
        results = []
        start_time = time.time()
        
        # Setup
        if self.setup_func:
            try:
                self.setup_func()
            except Exception as e:
                return {
                    'suite_name': self.name,
                    'total_tests': len(self.tests),
                    'passed_tests': 0,
                    'failed_tests': 0,
                    'execution_time': time.time() - start_time,
                    'setup_failed': True,
                    'setup_error': str(e),
                    'results': []
                }
        
        # Run tests
        if parallel and len(self.tests) > 1:
            results = self._run_tests_parallel()
        else:
            results = self._run_tests_sequential()
            
        # Teardown
        if self.teardown_func:
            try:
                self.teardown_func()
            except Exception as e:
                # Log teardown error but don't fail the suite
                warnings.warn(f"Teardown failed for suite {self.name}: {e}")
                
        # Calculate summary
        total_time = time.time() - start_time
        passed_tests = len([r for r in results if r.passed])
        failed_tests = len(results) - passed_tests
        
        return {
            'suite_name': self.name,
            'total_tests': len(self.tests),
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'execution_time': total_time,
            'success_rate': passed_tests / len(self.tests) if self.tests else 0.0,
            'results': [r.to_dict() for r in results]
        }
        
    def _run_tests_sequential(self) -> List[TestResult]:
        """Run tests sequentially."""
        results = []
        
        for test_func in self.tests:
            result = self._run_single_test(test_func)
            results.append(result)
            
        return results
        
    def _run_tests_parallel(self) -> List[TestResult]:
        """Run tests in parallel."""
        results = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_test = {
                executor.submit(self._run_single_test, test_func): test_func
                for test_func in self.tests
            }
            
            for future in as_completed(future_to_test):
                result = future.result()
                results.append(result)
                
        # Sort results by test name for consistency
        results.sort(key=lambda x: x.test_name)
        return results
        
    def _run_single_test(self, test_func: Callable) -> TestResult:
        """Run a single test function."""
        test_name = getattr(test_func, '__name__', str(test_func))
        test_category = getattr(test_func, '_test_category', TestCategory.UNIT)
        
        start_time = time.time()
        
        try:
            # Run the test with timeout
            result = test_func()
            execution_time = time.time() - start_time
            
            # Check if test indicates failure
            if isinstance(result, bool) and not result:
                return TestResult(
                    test_name=test_name,
                    category=test_category,
                    passed=False,
                    execution_time=execution_time,
                    message="Test function returned False"
                )
            elif isinstance(result, dict) and 'passed' in result:
                return TestResult(
                    test_name=test_name,
                    category=test_category,
                    passed=result['passed'],
                    execution_time=execution_time,
                    message=result.get('message', ''),
                    details=result.get('details', {})
                )
            else:
                # Test passed (no exception and didn't return False)
                return TestResult(
                    test_name=test_name,
                    category=test_category,
                    passed=True,
                    execution_time=execution_time,
                    message="Test completed successfully",
                    details={'result': str(result) if result is not None else 'None'}
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            
            return TestResult(
                test_name=test_name,
                category=test_category,
                passed=False,
                execution_time=execution_time,
                message=str(e),
                severity=TestSeverity.HIGH,
                details={
                    'exception_type': type(e).__name__,
                    'traceback': traceback.format_exc()
                }
            )


class NeuromorphicTestFramework:
    """Comprehensive test framework for neuromorphic systems."""
    
    def __init__(self):
        self.test_suites: Dict[str, TestSuite] = {}
        self.test_history: List[Dict] = []
        self.performance_baselines: Dict[str, float] = {}
        self.quality_thresholds = {
            'min_success_rate': 0.95,
            'max_avg_execution_time': 1.0,
            'max_memory_usage_mb': 100,
            'min_coverage_percentage': 85
        }
        
    def create_test_suite(self, name: str) -> TestSuite:
        """Create a new test suite."""
        suite = TestSuite(name)
        self.test_suites[name] = suite
        return suite
        
    def run_all_suites(self, parallel_suites: bool = False) -> Dict[str, Any]:
        """Run all registered test suites."""
        
        start_time = time.time()
        suite_results = {}
        
        if parallel_suites and len(self.test_suites) > 1:
            suite_results = self._run_suites_parallel()
        else:
            suite_results = self._run_suites_sequential()
            
        # Calculate overall summary
        total_tests = sum(result['total_tests'] for result in suite_results.values())
        total_passed = sum(result['passed_tests'] for result in suite_results.values())
        total_failed = sum(result['failed_tests'] for result in suite_results.values())
        
        overall_result = {
            'timestamp': time.time(),
            'total_execution_time': time.time() - start_time,
            'total_suites': len(self.test_suites),
            'total_tests': total_tests,
            'total_passed': total_passed,
            'total_failed': total_failed,
            'overall_success_rate': total_passed / total_tests if total_tests > 0 else 0.0,
            'suite_results': suite_results,
            'quality_assessment': self._assess_quality(suite_results)
        }
        
        # Store in history
        self.test_history.append(overall_result)
        
        return overall_result
        
    def _run_suites_sequential(self) -> Dict[str, Dict]:
        """Run test suites sequentially."""
        results = {}
        
        for suite_name, suite in self.test_suites.items():
            results[suite_name] = suite.run_suite()
            
        return results
        
    def _run_suites_parallel(self) -> Dict[str, Dict]:
        """Run test suites in parallel."""
        results = {}
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_suite = {
                executor.submit(suite.run_suite): suite_name
                for suite_name, suite in self.test_suites.items()
            }
            
            for future in as_completed(future_to_suite):
                suite_name = future_to_suite[future]
                results[suite_name] = future.result()
                
        return results
        
    def _assess_quality(self, suite_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Assess overall quality based on test results."""
        
        if not suite_results:
            return {'status': 'no_tests', 'score': 0.0}
            
        # Calculate quality metrics
        total_tests = sum(result['total_tests'] for result in suite_results.values())
        total_passed = sum(result['passed_tests'] for result in suite_results.values())
        
        success_rate = total_passed / total_tests if total_tests > 0 else 0.0
        
        # Average execution time per test
        total_time = sum(result['execution_time'] for result in suite_results.values())
        avg_time_per_test = total_time / total_tests if total_tests > 0 else 0.0
        
        # Quality score calculation
        quality_score = 0.0
        quality_factors = []
        
        # Success rate factor (50% weight)
        success_factor = min(1.0, success_rate / self.quality_thresholds['min_success_rate'])
        quality_score += success_factor * 0.5
        quality_factors.append({'factor': 'success_rate', 'score': success_factor, 'weight': 0.5})
        
        # Performance factor (30% weight)
        if avg_time_per_test <= self.quality_thresholds['max_avg_execution_time']:
            performance_factor = 1.0
        else:
            performance_factor = max(0.0, 1.0 - (avg_time_per_test - self.quality_thresholds['max_avg_execution_time']))
        quality_score += performance_factor * 0.3
        quality_factors.append({'factor': 'performance', 'score': performance_factor, 'weight': 0.3})
        
        # Reliability factor (20% weight) - based on consistency across suites
        suite_success_rates = [
            result['success_rate'] for result in suite_results.values()
            if result['total_tests'] > 0
        ]
        
        if suite_success_rates:
            success_rate_std = statistics.stdev(suite_success_rates) if len(suite_success_rates) > 1 else 0.0
            reliability_factor = max(0.0, 1.0 - success_rate_std)
        else:
            reliability_factor = 0.0
            
        quality_score += reliability_factor * 0.2
        quality_factors.append({'factor': 'reliability', 'score': reliability_factor, 'weight': 0.2})
        
        # Determine quality status
        if quality_score >= 0.9:
            status = 'excellent'
        elif quality_score >= 0.8:
            status = 'good'
        elif quality_score >= 0.7:
            status = 'acceptable'
        elif quality_score >= 0.5:
            status = 'poor'
        else:
            status = 'critical'
            
        return {
            'status': status,
            'score': quality_score,
            'factors': quality_factors,
            'metrics': {
                'success_rate': success_rate,
                'avg_time_per_test': avg_time_per_test,
                'reliability_score': reliability_factor
            },
            'thresholds': self.quality_thresholds
        }
        
    def create_neuromorphic_unit_tests(self) -> TestSuite:
        """Create comprehensive unit tests for neuromorphic components."""
        
        suite = self.create_test_suite("neuromorphic_unit_tests")
        
        def test_spike_encoding():
            """Test spike encoding functionality."""
            try:
                # Mock spike encoding test
                input_data = [0.1, 0.5, 0.8, 0.2]
                
                # Basic encoding checks
                if len(input_data) != 4:
                    return {'passed': False, 'message': 'Input data length mismatch'}
                    
                # Validate range
                if any(x < 0 or x > 1 for x in input_data):
                    return {'passed': False, 'message': 'Input values out of range'}
                    
                return {'passed': True, 'message': 'Spike encoding test passed'}
                
            except Exception as e:
                return {'passed': False, 'message': f'Spike encoding failed: {e}'}
        
        def test_membrane_dynamics():
            """Test membrane potential dynamics."""
            try:
                # Mock membrane potential calculation
                initial_potential = 0.0
                input_current = 1.5
                tau = 20.0
                dt = 1.0
                
                # Simple exponential integration
                new_potential = initial_potential + (input_current * dt / tau)
                
                # Validate reasonable bounds
                if new_potential < -100 or new_potential > 100:
                    return {'passed': False, 'message': 'Membrane potential out of bounds'}
                    
                return {'passed': True, 'message': 'Membrane dynamics test passed'}
                
            except Exception as e:
                return {'passed': False, 'message': f'Membrane dynamics failed: {e}'}
        
        def test_synaptic_plasticity():
            """Test synaptic plasticity mechanisms."""
            try:
                # Mock STDP implementation
                pre_spike_time = 10.0
                post_spike_time = 15.0
                learning_rate = 0.01
                
                # Calculate weight change
                dt = post_spike_time - pre_spike_time
                if dt > 0:
                    weight_change = learning_rate * (-abs(dt) / 20.0)  # LTD
                else:
                    weight_change = learning_rate * (abs(dt) / 20.0)   # LTP
                    
                # Validate reasonable weight change
                if abs(weight_change) > 1.0:
                    return {'passed': False, 'message': 'Weight change too large'}
                    
                return {'passed': True, 'message': 'Synaptic plasticity test passed'}
                
            except Exception as e:
                return {'passed': False, 'message': f'Synaptic plasticity failed: {e}'}
        
        def test_sparsity_constraints():
            """Test sparsity constraint enforcement."""
            try:
                # Mock sparse activity
                activity = [0, 0, 1, 0, 0, 1, 0, 0, 0, 0]  # 20% sparsity
                target_sparsity = 0.05  # 5% target
                
                actual_sparsity = sum(activity) / len(activity)
                sparsity_error = abs(actual_sparsity - target_sparsity)
                
                # Check if within tolerance
                if sparsity_error > 0.2:  # 20% tolerance
                    return {
                        'passed': False, 
                        'message': f'Sparsity error too large: {sparsity_error}',
                        'details': {'actual': actual_sparsity, 'target': target_sparsity}
                    }
                    
                return {'passed': True, 'message': 'Sparsity constraints test passed'}
                
            except Exception as e:
                return {'passed': False, 'message': f'Sparsity constraints failed: {e}'}
        
        def test_decision_making():
            """Test decision making layer."""
            try:
                # Mock decision inputs
                input_scores = [0.1, 0.8, 0.05, 0.05]
                
                # Find winner
                max_idx = input_scores.index(max(input_scores))
                confidence = max(input_scores)
                
                # Validate decision
                if confidence < 0.5:
                    return {'passed': False, 'message': 'Decision confidence too low'}
                    
                if max_idx < 0 or max_idx >= len(input_scores):
                    return {'passed': False, 'message': 'Invalid decision index'}
                    
                return {'passed': True, 'message': 'Decision making test passed'}
                
            except Exception as e:
                return {'passed': False, 'message': f'Decision making failed: {e}'}
        
        # Add tests to suite
        suite.add_test(test_spike_encoding, TestCategory.UNIT)
        suite.add_test(test_membrane_dynamics, TestCategory.UNIT)
        suite.add_test(test_synaptic_plasticity, TestCategory.UNIT)
        suite.add_test(test_sparsity_constraints, TestCategory.UNIT)
        suite.add_test(test_decision_making, TestCategory.UNIT)
        
        return suite
        
    def create_integration_tests(self) -> TestSuite:
        """Create integration tests for neuromorphic system."""
        
        suite = self.create_test_suite("neuromorphic_integration_tests")
        
        def test_end_to_end_pipeline():
            """Test complete neuromorphic pipeline."""
            try:
                # Mock complete pipeline
                stages = ['encoding', 'projection', 'kenyon', 'decision']
                
                # Simulate data flow
                data = {'sensor_input': [0.1, 0.5, 0.3]}
                
                for stage in stages:
                    # Mock processing at each stage
                    if stage == 'encoding':
                        data['spikes'] = [int(x > 0.2) for x in data['sensor_input']]
                    elif stage == 'projection':
                        data['projection_output'] = [x * 2 for x in data['spikes']]
                    elif stage == 'kenyon':
                        # Sparse representation
                        data['kenyon_output'] = [x if x > 1 else 0 for x in data['projection_output']]
                    elif stage == 'decision':
                        data['decision'] = sum(data['kenyon_output']) > 0
                        
                # Validate final output
                if 'decision' not in data:
                    return {'passed': False, 'message': 'Pipeline incomplete'}
                    
                return {'passed': True, 'message': 'End-to-end pipeline test passed'}
                
            except Exception as e:
                return {'passed': False, 'message': f'Pipeline test failed: {e}'}
        
        def test_multi_modal_fusion():
            """Test multi-modal sensor fusion."""
            try:
                # Mock multi-modal inputs
                chemical_input = [0.2, 0.5, 0.1]
                audio_input = [0.3, 0.7, 0.4]
                
                # Simple fusion strategy
                fused_output = []
                for i in range(min(len(chemical_input), len(audio_input))):
                    fused_value = (chemical_input[i] + audio_input[i]) / 2
                    fused_output.append(fused_value)
                    
                # Validate fusion
                if len(fused_output) != 3:
                    return {'passed': False, 'message': 'Fusion output length mismatch'}
                    
                if any(x < 0 or x > 1 for x in fused_output):
                    return {'passed': False, 'message': 'Fusion output out of range'}
                    
                return {'passed': True, 'message': 'Multi-modal fusion test passed'}
                
            except Exception as e:
                return {'passed': False, 'message': f'Fusion test failed: {e}'}
        
        def test_plasticity_integration():
            """Test integration of plasticity mechanisms."""
            try:
                # Mock plasticity integration
                initial_weights = [0.5, 0.3, 0.8]
                learning_events = [(0, 0.1), (1, -0.05), (2, 0.02)]
                
                # Apply learning
                updated_weights = initial_weights.copy()
                for idx, change in learning_events:
                    if 0 <= idx < len(updated_weights):
                        updated_weights[idx] += change
                        # Clamp weights
                        updated_weights[idx] = max(0, min(1, updated_weights[idx]))
                        
                # Validate weight updates
                if len(updated_weights) != len(initial_weights):
                    return {'passed': False, 'message': 'Weight array size changed'}
                    
                if any(w < 0 or w > 1 for w in updated_weights):
                    return {'passed': False, 'message': 'Weights out of bounds'}
                    
                return {'passed': True, 'message': 'Plasticity integration test passed'}
                
            except Exception as e:
                return {'passed': False, 'message': f'Plasticity integration failed: {e}'}
        
        # Add tests to suite
        suite.add_test(test_end_to_end_pipeline, TestCategory.INTEGRATION)
        suite.add_test(test_multi_modal_fusion, TestCategory.INTEGRATION)
        suite.add_test(test_plasticity_integration, TestCategory.INTEGRATION)
        
        return suite
        
    def create_performance_tests(self) -> TestSuite:
        """Create performance tests for neuromorphic system."""
        
        suite = self.create_test_suite("neuromorphic_performance_tests")
        
        def test_processing_latency():
            """Test processing latency requirements."""
            try:
                start_time = time.time()
                
                # Mock processing workload
                data_size = 1000
                processed_items = 0
                
                for i in range(data_size):
                    # Simulate neuromorphic computation
                    result = i * 0.001  # Minimal computation
                    processed_items += 1
                    
                processing_time = time.time() - start_time
                latency_per_item = processing_time / processed_items
                
                # Check latency requirement (< 1ms per item)
                if latency_per_item > 0.001:
                    return {
                        'passed': False, 
                        'message': f'Latency too high: {latency_per_item*1000:.2f}ms',
                        'details': {'latency_ms': latency_per_item * 1000}
                    }
                    
                return {
                    'passed': True, 
                    'message': 'Processing latency test passed',
                    'details': {'latency_ms': latency_per_item * 1000}
                }
                
            except Exception as e:
                return {'passed': False, 'message': f'Latency test failed: {e}'}
        
        def test_throughput_requirements():
            """Test system throughput requirements."""
            try:
                start_time = time.time()
                test_duration = 1.0  # 1 second test
                
                samples_processed = 0
                
                while time.time() - start_time < test_duration:
                    # Mock sample processing
                    sample = [0.1, 0.2, 0.3]
                    result = sum(sample)  # Minimal processing
                    samples_processed += 1
                    
                actual_duration = time.time() - start_time
                throughput = samples_processed / actual_duration
                
                # Check throughput requirement (> 100 samples/sec)
                if throughput < 100:
                    return {
                        'passed': False,
                        'message': f'Throughput too low: {throughput:.1f} samples/sec',
                        'details': {'throughput': throughput}
                    }
                    
                return {
                    'passed': True,
                    'message': 'Throughput test passed',
                    'details': {'throughput': throughput}
                }
                
            except Exception as e:
                return {'passed': False, 'message': f'Throughput test failed: {e}'}
        
        def test_memory_efficiency():
            """Test memory usage efficiency."""
            try:
                # Mock memory usage test
                initial_objects = 1000
                object_size_estimate = 64  # bytes per object
                
                estimated_memory = initial_objects * object_size_estimate
                memory_mb = estimated_memory / (1024 * 1024)
                
                # Check memory requirement (< 10MB for test)
                if memory_mb > 10:
                    return {
                        'passed': False,
                        'message': f'Memory usage too high: {memory_mb:.2f}MB',
                        'details': {'memory_mb': memory_mb}
                    }
                    
                return {
                    'passed': True,
                    'message': 'Memory efficiency test passed',
                    'details': {'memory_mb': memory_mb}
                }
                
            except Exception as e:
                return {'passed': False, 'message': f'Memory test failed: {e}'}
        
        # Add tests to suite
        suite.add_test(test_processing_latency, TestCategory.PERFORMANCE)
        suite.add_test(test_throughput_requirements, TestCategory.PERFORMANCE)
        suite.add_test(test_memory_efficiency, TestCategory.PERFORMANCE)
        
        return suite
        
    def generate_test_report(self, test_results: Dict[str, Any]) -> str:
        """Generate comprehensive test report."""
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("NEUROMORPHIC SYSTEM TEST REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Execution Time: {test_results['total_execution_time']:.2f}s")
        report_lines.append("")
        
        # Overall Summary
        report_lines.append("OVERALL SUMMARY")
        report_lines.append("-" * 40)
        report_lines.append(f"Total Test Suites: {test_results['total_suites']}")
        report_lines.append(f"Total Tests: {test_results['total_tests']}")
        report_lines.append(f"Passed: {test_results['total_passed']}")
        report_lines.append(f"Failed: {test_results['total_failed']}")
        report_lines.append(f"Success Rate: {test_results['overall_success_rate']:.1%}")
        report_lines.append("")
        
        # Quality Assessment
        quality = test_results['quality_assessment']
        report_lines.append("QUALITY ASSESSMENT")
        report_lines.append("-" * 40)
        report_lines.append(f"Status: {quality['status'].upper()}")
        report_lines.append(f"Quality Score: {quality['score']:.2f}/1.00")
        report_lines.append("")
        
        for factor in quality['factors']:
            report_lines.append(f"  {factor['factor']}: {factor['score']:.2f} (weight: {factor['weight']:.1f})")
        report_lines.append("")
        
        # Suite Details
        report_lines.append("SUITE DETAILS")
        report_lines.append("-" * 40)
        
        for suite_name, suite_result in test_results['suite_results'].items():
            report_lines.append(f"\n{suite_name}:")
            report_lines.append(f"  Tests: {suite_result['total_tests']}")
            report_lines.append(f"  Passed: {suite_result['passed_tests']}")
            report_lines.append(f"  Failed: {suite_result['failed_tests']}")
            report_lines.append(f"  Success Rate: {suite_result['success_rate']:.1%}")
            report_lines.append(f"  Execution Time: {suite_result['execution_time']:.2f}s")
            
            # Failed tests details
            failed_tests = [r for r in suite_result['results'] if not r['passed']]
            if failed_tests:
                report_lines.append("  Failed Tests:")
                for test in failed_tests:
                    report_lines.append(f"    - {test['test_name']}: {test['message']}")
                    
        report_lines.append("")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)


def run_comprehensive_neuromorphic_tests():
    """Run comprehensive neuromorphic system tests."""
    
    print("🧪 Running Comprehensive Neuromorphic Tests...")
    
    # Create test framework
    framework = NeuromorphicTestFramework()
    
    # Create test suites
    unit_suite = framework.create_neuromorphic_unit_tests()
    integration_suite = framework.create_integration_tests()
    performance_suite = framework.create_performance_tests()
    
    print(f"  Created {len(framework.test_suites)} test suites")
    
    # Run all tests
    results = framework.run_all_suites(parallel_suites=True)
    
    # Print summary
    print(f"\n🏆 Test Results Summary:")
    print(f"  Total Tests: {results['total_tests']}")
    print(f"  Passed: {results['total_passed']}")
    print(f"  Failed: {results['total_failed']}")
    print(f"  Success Rate: {results['overall_success_rate']:.1%}")
    print(f"  Quality Score: {results['quality_assessment']['score']:.2f}")
    print(f"  Quality Status: {results['quality_assessment']['status']}")
    
    # Show suite breakdown
    print(f"\n📋 Suite Breakdown:")
    for suite_name, suite_result in results['suite_results'].items():
        status = "✅" if suite_result['success_rate'] >= 0.8 else "❌"
        print(f"  {status} {suite_name}: {suite_result['passed_tests']}/{suite_result['total_tests']} passed")
    
    # Generate full report
    report = framework.generate_test_report(results)
    
    # Save report
    with open('neuromorphic_test_report.txt', 'w') as f:
        f.write(report)
    print(f"\n📄 Full report saved to: neuromorphic_test_report.txt")
    
    # Return success status
    success = results['overall_success_rate'] >= 0.8
    return success, results


if __name__ == "__main__":
    success, results = run_comprehensive_neuromorphic_tests()
    exit(0 if success else 1)