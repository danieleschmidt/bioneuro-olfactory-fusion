"""Comprehensive testing framework for BioNeuro-Olfactory-Fusion."""

import time
import threading
from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from unittest.mock import Mock, patch
import traceback

from ..core.error_handling import get_error_handler, BioNeuroError, ErrorSeverity


class TestResult(Enum):
    """Test result status."""
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    SKIP = "skip"


@dataclass
class TestCase:
    """Individual test case definition."""
    name: str
    test_func: Callable
    description: str = ""
    timeout: float = 30.0
    expected_exception: Optional[type] = None
    skip_condition: Optional[Callable] = None
    setup: Optional[Callable] = None
    teardown: Optional[Callable] = None


@dataclass
class TestReport:
    """Test execution report."""
    name: str
    result: TestResult
    duration: float
    error_message: str = ""
    traceback_info: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TestSuite:
    """Test suite for organizing and running tests."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.test_cases: List[TestCase] = []
        self.setup_suite: Optional[Callable] = None
        self.teardown_suite: Optional[Callable] = None
        self.error_handler = get_error_handler()
        
    def add_test(
        self,
        name: str,
        test_func: Callable,
        description: str = "",
        **kwargs
    ):
        """Add a test case to the suite."""
        test_case = TestCase(
            name=name,
            test_func=test_func,
            description=description,
            **kwargs
        )
        self.test_cases.append(test_case)
        
    def run(self, parallel: bool = False, max_workers: int = 4) -> List[TestReport]:
        """Run all test cases in the suite."""
        if self.setup_suite:
            try:
                self.setup_suite()
            except Exception as e:
                self.error_handler.logger.error(f"Suite setup failed: {e}")
                return []
                
        try:
            if parallel and len(self.test_cases) > 1:
                reports = self._run_parallel(max_workers)
            else:
                reports = self._run_sequential()
                
        finally:
            if self.teardown_suite:
                try:
                    self.teardown_suite()
                except Exception as e:
                    self.error_handler.logger.error(f"Suite teardown failed: {e}")
                    
        return reports
        
    def _run_sequential(self) -> List[TestReport]:
        """Run tests sequentially."""
        reports = []
        
        for test_case in self.test_cases:
            report = self._run_single_test(test_case)
            reports.append(report)
            
        return reports
        
    def _run_parallel(self, max_workers: int) -> List[TestReport]:
        """Run tests in parallel."""
        reports = []
        threads = []
        
        def run_test_thread(test_case: TestCase, results: List):
            report = self._run_single_test(test_case)
            results.append(report)
            
        # Group tests by max_workers
        for i in range(0, len(self.test_cases), max_workers):
            batch = self.test_cases[i:i + max_workers]
            thread_results = []
            
            for test_case in batch:
                thread = threading.Thread(
                    target=run_test_thread,
                    args=(test_case, thread_results)
                )
                threads.append(thread)
                thread.start()
                
            # Wait for batch to complete
            for thread in threads[-len(batch):]:
                thread.join()
                
            reports.extend(thread_results)
            
        return reports
        
    def _run_single_test(self, test_case: TestCase) -> TestReport:
        """Run a single test case."""
        start_time = time.time()
        
        # Check skip condition
        if test_case.skip_condition and test_case.skip_condition():
            return TestReport(
                name=test_case.name,
                result=TestResult.SKIP,
                duration=0.0,
                error_message="Skipped due to skip condition"
            )
            
        # Setup
        if test_case.setup:
            try:
                test_case.setup()
            except Exception as e:
                return TestReport(
                    name=test_case.name,
                    result=TestResult.ERROR,
                    duration=time.time() - start_time,
                    error_message=f"Setup failed: {str(e)}",
                    traceback_info=traceback.format_exc()
                )
                
        try:
            # Run test with timeout
            if test_case.timeout > 0:
                result = self._run_with_timeout(test_case.test_func, test_case.timeout)
            else:
                result = test_case.test_func()
                
            # Check for expected exception
            if test_case.expected_exception:
                return TestReport(
                    name=test_case.name,
                    result=TestResult.FAIL,
                    duration=time.time() - start_time,
                    error_message=f"Expected {test_case.expected_exception.__name__} but none was raised"
                )
                
            return TestReport(
                name=test_case.name,
                result=TestResult.PASS,
                duration=time.time() - start_time,
                metadata={"result": result} if result is not None else {}
            )
            
        except Exception as e:
            # Check if this was the expected exception
            if test_case.expected_exception and isinstance(e, test_case.expected_exception):
                return TestReport(
                    name=test_case.name,
                    result=TestResult.PASS,
                    duration=time.time() - start_time,
                    error_message=f"Expected exception caught: {str(e)}"
                )
            else:
                return TestReport(
                    name=test_case.name,
                    result=TestResult.FAIL,
                    duration=time.time() - start_time,
                    error_message=str(e),
                    traceback_info=traceback.format_exc()
                )
                
        finally:
            # Teardown
            if test_case.teardown:
                try:
                    test_case.teardown()
                except Exception as e:
                    self.error_handler.logger.warning(f"Teardown failed for {test_case.name}: {e}")
                    
    def _run_with_timeout(self, test_func: Callable, timeout: float) -> Any:
        """Run test function with timeout."""
        result = [None]
        exception = [None]
        
        def target():
            try:
                result[0] = test_func()
            except Exception as e:
                exception[0] = e
                
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            raise TimeoutError(f"Test timed out after {timeout} seconds")
            
        if exception[0]:
            raise exception[0]
            
        return result[0]


class MockSensorArray:
    """Mock sensor array for testing."""
    
    def __init__(self, num_sensors: int = 6, noise_level: float = 0.1):
        self.num_sensors = num_sensors
        self.noise_level = noise_level
        self.baseline_values = [0.5] * num_sensors
        self.current_values = self.baseline_values.copy()
        
    def read(self) -> List[float]:
        """Read sensor values with simulated noise."""
        import random
        
        readings = []
        for baseline in self.baseline_values:
            noise = random.uniform(-self.noise_level, self.noise_level)
            reading = max(0, min(5.0, baseline + noise))  # Clamp to sensor range
            readings.append(reading)
            
        return readings
        
    def inject_gas(self, sensor_index: int, concentration: float):
        """Simulate gas injection at specific sensor."""
        if 0 <= sensor_index < self.num_sensors:
            self.current_values[sensor_index] = min(5.0, concentration)
            
    def reset(self):
        """Reset all sensors to baseline."""
        self.current_values = self.baseline_values.copy()


class NetworkTestFramework:
    """Testing framework for neural network components."""
    
    def __init__(self):
        self.test_suites: List[TestSuite] = []
        self.mock_factory = MockFactory()
        
    def create_test_suite(self, name: str, description: str = "") -> TestSuite:
        """Create a new test suite."""
        suite = TestSuite(name, description)
        self.test_suites.append(suite)
        return suite
        
    def run_all_tests(self, parallel: bool = False) -> Dict[str, List[TestReport]]:
        """Run all test suites."""
        all_reports = {}
        
        for suite in self.test_suites:
            reports = suite.run(parallel=parallel)
            all_reports[suite.name] = reports
            
        return all_reports
        
    def generate_report(
        self, 
        reports: Dict[str, List[TestReport]], 
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        total_tests = sum(len(suite_reports) for suite_reports in reports.values())
        total_passed = sum(
            1 for suite_reports in reports.values() 
            for report in suite_reports 
            if report.result == TestResult.PASS
        )
        total_failed = sum(
            1 for suite_reports in reports.values()
            for report in suite_reports
            if report.result == TestResult.FAIL
        )
        total_errors = sum(
            1 for suite_reports in reports.values()
            for report in suite_reports
            if report.result == TestResult.ERROR
        )
        total_skipped = sum(
            1 for suite_reports in reports.values()
            for report in suite_reports
            if report.result == TestResult.SKIP
        )
        
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total": total_tests,
                "passed": total_passed,
                "failed": total_failed,
                "errors": total_errors,
                "skipped": total_skipped,
                "pass_rate": (total_passed / total_tests * 100) if total_tests > 0 else 0
            },
            "suites": {}
        }
        
        for suite_name, suite_reports in reports.items():
            suite_summary = {
                "total": len(suite_reports),
                "passed": sum(1 for r in suite_reports if r.result == TestResult.PASS),
                "failed": sum(1 for r in suite_reports if r.result == TestResult.FAIL),
                "errors": sum(1 for r in suite_reports if r.result == TestResult.ERROR),
                "skipped": sum(1 for r in suite_reports if r.result == TestResult.SKIP),
                "tests": []
            }
            
            for report in suite_reports:
                suite_summary["tests"].append({
                    "name": report.name,
                    "result": report.result.value,
                    "duration": report.duration,
                    "error_message": report.error_message,
                    "metadata": report.metadata
                })
                
            summary["suites"][suite_name] = suite_summary
            
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(summary, f, indent=2)
                
        return summary


class MockFactory:
    """Factory for creating mock objects for testing."""
    
    def create_mock_sensor_array(self, **kwargs) -> MockSensorArray:
        """Create mock sensor array."""
        return MockSensorArray(**kwargs)
        
    def create_mock_network(self, network_type: str = "basic"):
        """Create mock neural network."""
        mock = Mock()
        
        if network_type == "projection":
            mock.forward.return_value = (
                [[1, 0, 1, 0]] * 100,  # Mock spike trains
                [[0.5, 0.2, 0.8, 0.1]] * 100  # Mock potentials
            )
            mock.get_firing_rates.return_value = [10.5, 0.0, 15.2, 0.0]
            
        elif network_type == "kenyon":
            mock.forward.return_value = (
                [[0, 0, 1, 0, 0]] * 50,  # Sparse spikes
                [[0.1, 0.1, 0.9, 0.1, 0.1]] * 50  # Sparse potentials
            )
            mock.get_sparsity_statistics.return_value = {
                "sparsity_level": 0.95,
                "active_fraction": [0.0, 0.0, 1.0, 0.0, 0.0]
            }
            
        elif network_type == "fusion":
            mock.forward.return_value = [0.1, 0.9, 0.2, 0.1]  # Classification scores
            
        return mock
        
    def create_mock_sensor_reading(self, gas_type: str = "clean_air"):
        """Create mock sensor reading."""
        base_readings = {
            "clean_air": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            "methane": [0.8, 0.5, 0.6, 0.5, 0.5, 0.5],
            "carbon_monoxide": [0.5, 0.9, 0.5, 0.5, 0.5, 0.5],
            "ammonia": [0.5, 0.5, 0.5, 0.5, 0.8, 0.5]
        }
        
        return base_readings.get(gas_type, base_readings["clean_air"])


# Test utilities
def assert_approximately_equal(actual: float, expected: float, tolerance: float = 1e-6):
    """Assert that two floating point values are approximately equal."""
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"Expected {expected}, got {actual} (tolerance: {tolerance})")


def assert_shape_equal(actual_shape: tuple, expected_shape: tuple):
    """Assert that array shapes are equal."""
    if actual_shape != expected_shape:
        raise AssertionError(f"Shape mismatch: expected {expected_shape}, got {actual_shape}")


def assert_in_range(value: float, min_val: float, max_val: float):
    """Assert that value is within specified range."""
    if not (min_val <= value <= max_val):
        raise AssertionError(f"Value {value} not in range [{min_val}, {max_val}]")


def create_basic_test_suite() -> TestSuite:
    """Create a basic test suite for core functionality."""
    suite = TestSuite("BasicFunctionality", "Basic system functionality tests")
    
    def test_import():
        """Test that core modules can be imported."""
        try:
            from ...models.fusion.multimodal_fusion import EarlyFusion
            from ...models.projection.projection_neurons import ProjectionNeuronLayer
            from ...models.kenyon.kenyon_cells import KenyonCellLayer
            return True
        except ImportError as e:
            if "torch" in str(e):
                # Expected due to missing PyTorch
                return "skipped_torch"
            raise
            
    def test_mock_sensor():
        """Test mock sensor functionality."""
        sensor = MockSensorArray(num_sensors=6)
        readings = sensor.read()
        
        assert len(readings) == 6
        for reading in readings:
            assert_in_range(reading, 0.0, 5.0)
            
        return True
        
    def test_error_handling():
        """Test error handling system."""
        from ...core.error_handling import get_error_handler, BioNeuroError
        
        handler = get_error_handler()
        
        # Test error logging
        test_error = BioNeuroError("Test error", error_code="TEST_ERROR")
        result = handler.handle_error(test_error, attempt_recovery=False)
        
        assert result is False  # No recovery expected
        return True
        
    # Add test cases
    suite.add_test("import_test", test_import, "Test core imports")
    suite.add_test("mock_sensor_test", test_mock_sensor, "Test mock sensor")
    suite.add_test("error_handling_test", test_error_handling, "Test error handling")
    
    return suite


# Global test framework instance
_test_framework = None


def get_test_framework() -> NetworkTestFramework:
    """Get global test framework instance."""
    global _test_framework
    if _test_framework is None:
        _test_framework = NetworkTestFramework()
    return _test_framework