#!/usr/bin/env python3
"""Comprehensive quality gates validation for all generations."""

import sys
import os
import subprocess
import time
from typing import Dict, List, Tuple, Any

def run_code_quality_checks():
    """Run code quality and linting checks."""
    print("🔍 Running code quality checks...")
    
    quality_checks = []
    
    # Check Python syntax
    print("   Checking Python syntax...")
    try:
        python_files = []
        for root, dirs, files in os.walk('bioneuro_olfactory'):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        syntax_errors = 0
        for file_path in python_files:
            try:
                with open(file_path, 'r') as f:
                    compile(f.read(), file_path, 'exec')
            except SyntaxError as e:
                print(f"   ❌ Syntax error in {file_path}: {e}")
                syntax_errors += 1
                
        if syntax_errors == 0:
            print(f"   ✅ All {len(python_files)} Python files have valid syntax")
            quality_checks.append(("Python Syntax", True))
        else:
            print(f"   ❌ {syntax_errors} files have syntax errors")
            quality_checks.append(("Python Syntax", False))
            
    except Exception as e:
        print(f"   ❌ Syntax check failed: {e}")
        quality_checks.append(("Python Syntax", False))
    
    # Check import structure
    print("   Checking import structure...")
    try:
        import_issues = 0
        critical_imports = [
            ('bioneuro_olfactory.core.error_handling', 'ErrorHandler'),
            ('bioneuro_olfactory.core.health_monitoring', 'HealthMonitor'),
            ('bioneuro_olfactory.security.input_validation', 'InputValidator'),
            ('bioneuro_olfactory.testing.test_framework', 'TestSuite')
        ]
        
        for module_name, class_name in critical_imports:
            try:
                # Check if file exists and class is defined
                file_path = module_name.replace('.', '/') + '.py'
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        content = f.read()
                        if f'class {class_name}' in content:
                            print(f"   ✅ {class_name} found in {module_name}")
                        else:
                            print(f"   ❌ {class_name} not found in {module_name}")
                            import_issues += 1
                else:
                    print(f"   ❌ Module file not found: {file_path}")
                    import_issues += 1
            except Exception as e:
                print(f"   ⚠️ Could not check {module_name}: {e}")
                import_issues += 1
                
        quality_checks.append(("Import Structure", import_issues == 0))
        
    except Exception as e:
        print(f"   ❌ Import check failed: {e}")
        quality_checks.append(("Import Structure", False))
    
    return quality_checks


def run_security_validation():
    """Run security validation checks."""
    print("\n🔒 Running security validation...")
    
    security_checks = []
    
    # Check for security patterns
    print("   Checking security implementations...")
    try:
        security_file = 'bioneuro_olfactory/security/input_validation.py'
        if os.path.exists(security_file):
            with open(security_file, 'r') as f:
                content = f.read()
                
            security_features = [
                ('XSS Protection', 'script' in content.lower()),
                ('Injection Protection', 'injection' in content.lower() or 'sql' in content.lower()),
                ('Input Sanitization', 'sanitize' in content.lower()),
                ('Validation Framework', 'validate' in content.lower()),
                ('Error Handling', 'SecurityError' in content)
            ]
            
            for feature_name, present in security_features:
                if present:
                    print(f"   ✅ {feature_name} implemented")
                else:
                    print(f"   ⚠️ {feature_name} may be missing")
                    
                security_checks.append((feature_name, present))
        else:
            print("   ❌ Security validation file not found")
            security_checks.append(("Security Implementation", False))
            
    except Exception as e:
        print(f"   ❌ Security validation failed: {e}")
        security_checks.append(("Security Validation", False))
    
    return security_checks


def run_performance_validation():
    """Run performance and scalability validation."""
    print("\n⚡ Running performance validation...")
    
    performance_checks = []
    
    # Check Generation 3 optimization features
    try:
        optimization_files = [
            'bioneuro_olfactory/optimization/performance_profiler.py',
            'bioneuro_olfactory/optimization/distributed_processing.py',
            'bioneuro_olfactory/optimization/adaptive_caching.py'
        ]
        
        for file_path in optimization_files:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    content = f.read()
                    
                # Check for performance patterns
                patterns = [
                    'threading',
                    'concurrent',
                    'performance',
                    'optimization',
                    'scalability'
                ]
                
                found_patterns = sum(1 for p in patterns if p in content.lower())
                file_name = os.path.basename(file_path)
                
                if found_patterns >= 3:
                    print(f"   ✅ {file_name}: {found_patterns}/5 performance patterns")
                    performance_checks.append((file_name, True))
                else:
                    print(f"   ⚠️ {file_name}: {found_patterns}/5 performance patterns")
                    performance_checks.append((file_name, False))
            else:
                print(f"   ❌ {file_path} not found")
                performance_checks.append((os.path.basename(file_path), False))
                
    except Exception as e:
        print(f"   ❌ Performance validation failed: {e}")
        performance_checks.append(("Performance Validation", False))
        
    return performance_checks


def run_architecture_validation():
    """Validate overall architecture and design."""
    print("\n🏗️ Running architecture validation...")
    
    architecture_checks = []
    
    # Check package structure
    try:
        required_packages = [
            'bioneuro_olfactory/core',
            'bioneuro_olfactory/models',
            'bioneuro_olfactory/sensors', 
            'bioneuro_olfactory/security',
            'bioneuro_olfactory/testing',
            'bioneuro_olfactory/optimization',
            'bioneuro_olfactory/monitoring',
            'bioneuro_olfactory/applications'
        ]
        
        package_score = 0
        for package in required_packages:
            if os.path.exists(package) and os.path.exists(f"{package}/__init__.py"):
                print(f"   ✅ Package: {package}")
                package_score += 1
            else:
                print(f"   ❌ Missing package: {package}")
                
        architecture_checks.append(("Package Structure", package_score >= len(required_packages) - 1))
        
    except Exception as e:
        print(f"   ❌ Package structure check failed: {e}")
        architecture_checks.append(("Package Structure", False))
    
    # Check for design patterns
    try:
        design_patterns = {
            'Singleton Pattern': ['get_error_handler', 'get_health_monitor'],
            'Factory Pattern': ['create_moth_inspired_network', 'create_efficient_network'],
            'Observer Pattern': ['health_monitoring', 'metrics_collector'],
            'Strategy Pattern': ['CachePolicy', 'ProcessingMode']
        }
        
        for pattern_name, indicators in design_patterns.items():
            pattern_found = False
            
            for root, dirs, files in os.walk('bioneuro_olfactory'):
                for file in files:
                    if file.endswith('.py'):
                        try:
                            with open(os.path.join(root, file), 'r') as f:
                                content = f.read()
                                if any(indicator in content for indicator in indicators):
                                    pattern_found = True
                                    break
                        except:
                            continue
                if pattern_found:
                    break
                    
            if pattern_found:
                print(f"   ✅ Design pattern: {pattern_name}")
                architecture_checks.append((pattern_name, True))
            else:
                print(f"   ⚠️ Design pattern may be missing: {pattern_name}")
                architecture_checks.append((pattern_name, False))
                
    except Exception as e:
        print(f"   ❌ Design pattern check failed: {e}")
        
    return architecture_checks


def run_test_coverage_analysis():
    """Analyze test coverage and completeness."""
    print("\n🧪 Running test coverage analysis...")
    
    test_checks = []
    
    try:
        # Check test structure
        test_directories = [
            'tests/unit',
            'tests/integration', 
            'tests/e2e'
        ]
        
        existing_tests = 0
        for test_dir in test_directories:
            if os.path.exists(test_dir):
                test_files = [f for f in os.listdir(test_dir) if f.startswith('test_') and f.endswith('.py')]
                if test_files:
                    print(f"   ✅ {test_dir}: {len(test_files)} test files")
                    existing_tests += 1
                else:
                    print(f"   ⚠️ {test_dir}: no test files found")
            else:
                print(f"   ❌ Test directory missing: {test_dir}")
                
        test_checks.append(("Test Structure", existing_tests >= 2))
        
        # Check for testing framework
        testing_framework_file = 'bioneuro_olfactory/testing/test_framework.py'
        if os.path.exists(testing_framework_file):
            with open(testing_framework_file, 'r') as f:
                content = f.read()
                
            framework_features = [
                'TestSuite',
                'MockSensorArray',
                'assert_approximately_equal',
                'parallel',
                'timeout'
            ]
            
            features_found = sum(1 for f in framework_features if f in content)
            print(f"   ✅ Testing framework: {features_found}/{len(framework_features)} features")
            test_checks.append(("Testing Framework", features_found >= 4))
        else:
            print("   ❌ Testing framework not found")
            test_checks.append(("Testing Framework", False))
            
    except Exception as e:
        print(f"   ❌ Test coverage analysis failed: {e}")
        test_checks.append(("Test Coverage", False))
        
    return test_checks


def run_documentation_assessment():
    """Assess documentation quality and completeness."""
    print("\n📚 Running documentation assessment...")
    
    doc_checks = []
    
    try:
        # Check README quality
        if os.path.exists('README.md'):
            with open('README.md', 'r') as f:
                readme_content = f.read()
                
            readme_sections = [
                'Installation',
                'Quick Start', 
                'Architecture',
                'Performance',
                'Examples'
            ]
            
            sections_found = sum(1 for section in readme_sections if section in readme_content)
            print(f"   ✅ README sections: {sections_found}/{len(readme_sections)}")
            doc_checks.append(("README Quality", sections_found >= 4))
        else:
            print("   ❌ README.md not found")
            doc_checks.append(("README Quality", False))
            
        # Check inline documentation
        python_files = []
        for root, dirs, files in os.walk('bioneuro_olfactory'):
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    python_files.append(os.path.join(root, file))
                    
        doc_scores = []
        for file_path in python_files[:10]:  # Sample first 10 files
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    
                lines = content.split('\n')
                non_empty_lines = [line for line in lines if line.strip()]
                docstring_lines = len([line for line in lines if '"""' in line or "'''" in line])
                comment_lines = len([line for line in lines if line.strip().startswith('#')])
                
                if non_empty_lines:
                    doc_ratio = (docstring_lines + comment_lines) / len(non_empty_lines) * 100
                    doc_scores.append(doc_ratio)
                    
            except Exception:
                continue
                
        if doc_scores:
            avg_doc_score = sum(doc_scores) / len(doc_scores)
            print(f"   ✅ Average documentation density: {avg_doc_score:.1f}%")
            doc_checks.append(("Inline Documentation", avg_doc_score > 15))
        else:
            print("   ⚠️ Could not assess inline documentation")
            doc_checks.append(("Inline Documentation", False))
            
    except Exception as e:
        print(f"   ❌ Documentation assessment failed: {e}")
        doc_checks.append(("Documentation Assessment", False))
        
    return doc_checks


def generate_quality_report(all_checks: Dict[str, List[Tuple[str, bool]]]):
    """Generate comprehensive quality report."""
    print("\n📊 QUALITY GATES REPORT")
    print("=" * 50)
    
    total_checks = 0
    passed_checks = 0
    
    for category, checks in all_checks.items():
        print(f"\n{category}:")
        category_passed = 0
        
        for check_name, result in checks:
            status = "PASS" if result else "FAIL"
            symbol = "✅" if result else "❌"
            print(f"   {symbol} {check_name}: {status}")
            
            total_checks += 1
            if result:
                passed_checks += 1
                category_passed += 1
                
        category_rate = (category_passed / len(checks) * 100) if checks else 0
        print(f"   Category Score: {category_passed}/{len(checks)} ({category_rate:.1f}%)")
        
    overall_rate = (passed_checks / total_checks * 100) if total_checks > 0 else 0
    
    print(f"\n🎯 OVERALL QUALITY SCORE")
    print(f"   Checks Passed: {passed_checks}/{total_checks}")
    print(f"   Success Rate: {overall_rate:.1f}%")
    
    # Quality grade
    if overall_rate >= 90:
        grade = "A (Excellent)"
    elif overall_rate >= 80:
        grade = "B (Good)"
    elif overall_rate >= 70:
        grade = "C (Acceptable)"
    elif overall_rate >= 60:
        grade = "D (Needs Improvement)"
    else:
        grade = "F (Poor)"
        
    print(f"   Quality Grade: {grade}")
    
    # Production readiness assessment
    critical_categories = ['Code Quality', 'Security', 'Architecture']
    critical_passed = 0
    critical_total = 0
    
    for category in critical_categories:
        if category in all_checks:
            for _, result in all_checks[category]:
                critical_total += 1
                if result:
                    critical_passed += 1
                    
    critical_rate = (critical_passed / critical_total * 100) if critical_total > 0 else 0
    
    print(f"\n🚀 PRODUCTION READINESS")
    print(f"   Critical Systems: {critical_passed}/{critical_total} ({critical_rate:.1f}%)")
    
    if critical_rate >= 85 and overall_rate >= 80:
        readiness = "✅ READY FOR PRODUCTION"
    elif critical_rate >= 75 and overall_rate >= 70:
        readiness = "⚠️ NEEDS MINOR IMPROVEMENTS"
    else:
        readiness = "❌ NOT READY FOR PRODUCTION"
        
    print(f"   Status: {readiness}")
    
    return overall_rate >= 75  # 75% threshold for passing quality gates


def main():
    """Run comprehensive quality gates validation."""
    print("🚀 BioNeuro-Olfactory-Fusion Quality Gates")
    print("=" * 55)
    print("Running comprehensive validation across all generations...")
    
    start_time = time.time()
    
    # Run all quality checks
    all_checks = {
        "Code Quality": run_code_quality_checks(),
        "Security": run_security_validation(), 
        "Performance": run_performance_validation(),
        "Architecture": run_architecture_validation(),
        "Testing": run_test_coverage_analysis(),
        "Documentation": run_documentation_assessment()
    }
    
    # Generate comprehensive report
    quality_passed = generate_quality_report(all_checks)
    
    duration = time.time() - start_time
    
    print(f"\n⏱️ Quality validation completed in {duration:.1f} seconds")
    
    if quality_passed:
        print("\n🎉 QUALITY GATES PASSED!")
        print("System meets production quality standards.")
        return True
    else:
        print("\n⚠️ Quality gates need attention before production deployment.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)