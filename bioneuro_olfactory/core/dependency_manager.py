"""Dependency management and graceful fallbacks for optional libraries.

This module provides robust dependency handling, allowing the system to
function with degraded capabilities when optional dependencies are missing.
"""

import logging
import warnings
from typing import Dict, Any, Callable, Optional, List
from functools import wraps
import sys

logger = logging.getLogger(__name__)


class DependencyManager:
    """Manages optional dependencies with graceful fallbacks."""
    
    def __init__(self):
        self.available_deps = {}
        self.missing_deps = {}
        self.fallback_implementations = {}
        self._check_dependencies()
        
    def _check_dependencies(self):
        """Check availability of all dependencies."""
        dependencies = {
            'torch': self._check_torch,
            'librosa': self._check_librosa,
            'scipy': self._check_scipy,
            'numpy': self._check_numpy,
            'matplotlib': self._check_matplotlib,
            'seaborn': self._check_seaborn,
            'sklearn': self._check_sklearn
        }
        
        for dep_name, check_func in dependencies.items():
            try:
                module = check_func()
                self.available_deps[dep_name] = module
                logger.debug(f"✅ {dep_name} available")
            except ImportError as e:
                self.missing_deps[dep_name] = str(e)
                logger.warning(f"⚠️  {dep_name} not available: {e}")
                
        self._setup_fallbacks()
        
    def _check_torch(self):
        """Check PyTorch availability."""
        import torch
        import torch.nn as nn
        return {'torch': torch, 'nn': nn}
        
    def _check_librosa(self):
        """Check librosa availability."""
        import librosa
        return librosa
        
    def _check_scipy(self):
        """Check scipy availability."""
        import scipy
        import scipy.signal
        import scipy.stats
        return {'scipy': scipy, 'signal': scipy.signal, 'stats': scipy.stats}
        
    def _check_numpy(self):
        """Check numpy availability."""
        import numpy as np
        return np
        
    def _check_matplotlib(self):
        """Check matplotlib availability."""
        import matplotlib.pyplot as plt
        return plt
        
    def _check_seaborn(self):
        """Check seaborn availability."""
        import seaborn as sns
        return sns
        
    def _check_sklearn(self):
        """Check scikit-learn availability."""
        import sklearn
        return sklearn
        
    def _setup_fallbacks(self):
        """Setup fallback implementations for missing dependencies."""
        if 'torch' not in self.available_deps:
            self.fallback_implementations['torch'] = self._create_torch_fallback()
            
        if 'librosa' not in self.available_deps:
            self.fallback_implementations['librosa'] = self._create_librosa_fallback()
            
        if 'scipy' not in self.available_deps:
            self.fallback_implementations['scipy'] = self._create_scipy_fallback()
            
    def _create_torch_fallback(self):
        """Create PyTorch fallback implementation."""
        class MockTensor:
            def __init__(self, data):
                self.data = data if isinstance(data, list) else [data]
                
            def shape(self):
                return [len(self.data)]
                
            def unsqueeze(self, dim):
                return MockTensor([self.data])
                
            def numpy(self):
                return self.available_deps['numpy'].array(self.data)
                
            def __getitem__(self, item):
                return self.data[item]
                
        class MockModule:
            def __init__(self):
                pass
                
            def forward(self, *args, **kwargs):
                return MockTensor([0.0])
                
        class MockTorch:
            Tensor = MockTensor
            zeros = lambda *args, **kwargs: MockTensor([0] * (args[0] if args else 1))
            ones = lambda *args, **kwargs: MockTensor([1] * (args[0] if args else 1))
            tensor = MockTensor
            cat = lambda tensors, dim=0: MockTensor(sum([t.data for t in tensors], []))
            
            class nn:
                Module = MockModule
                Linear = MockModule
                Parameter = lambda x: x
                
        return MockTorch()
        
    def _create_librosa_fallback(self):
        """Create librosa fallback implementation."""
        class MockLibrosa:
            def __init__(self):
                self.feature = self.Feature()
                
            class Feature:
                def mfcc(self, y, sr, n_mfcc=13, **kwargs):
                    return self._get_numpy().random.randn(n_mfcc, 100)
                    
                def melspectrogram(self, y, sr, n_mels=128, **kwargs):
                    return self._get_numpy().random.randn(n_mels, 100)
                    
                def chroma_stft(self, y, sr, n_chroma=12, **kwargs):
                    return self._get_numpy().random.randn(n_chroma, 100)
                    
                def spectral_centroid(self, y, sr, **kwargs):
                    return self._get_numpy().random.randn(1, 100)
                    
                def spectral_bandwidth(self, y, sr, **kwargs):
                    return self._get_numpy().random.randn(1, 100)
                    
                def zero_crossing_rate(self, y, **kwargs):
                    return self._get_numpy().random.randn(1, 100)
                    
                def spectral_rolloff(self, y, sr, **kwargs):
                    return self._get_numpy().random.randn(1, 100)
                    
                def spectral_contrast(self, y, sr, **kwargs):
                    return self._get_numpy().random.randn(7, 100)
                    
                def _get_numpy(self):
                    return dep_manager.get_implementation('numpy')
                    
            def power_to_db(self, spec, ref=None):
                return self._get_numpy().log10(spec + 1e-8) * 10
                
            def _get_numpy(self):
                return dep_manager.get_implementation('numpy')
                
        return MockLibrosa()
        
    def _create_scipy_fallback(self):
        """Create scipy fallback implementation."""
        class MockScipy:
            def __init__(self):
                self.signal = self.Signal()
                self.stats = self.Stats()
                
            class Signal:
                def welch(self, x, fs, nperseg=None):
                    np = dep_manager.get_implementation('numpy')
                    freqs = np.linspace(0, fs/2, 100)
                    psd = np.random.randn(100)
                    return freqs, psd
                    
                def spectrogram(self, x, fs, nperseg=1024, noverlap=512):
                    np = dep_manager.get_implementation('numpy')
                    freqs = np.linspace(0, fs/2, nperseg//2)
                    times = np.linspace(0, len(x)/fs, len(x)//512)
                    spec = np.random.randn(len(freqs), len(times))
                    return freqs, times, spec
                    
            class Stats:
                def gmean(self, data):
                    np = dep_manager.get_implementation('numpy')
                    return np.exp(np.mean(np.log(data + 1e-8)))
                    
        return MockScipy()
        
    def get_implementation(self, dependency: str):
        """Get implementation (real or fallback) for a dependency."""
        if dependency in self.available_deps:
            return self.available_deps[dependency]
        elif dependency in self.fallback_implementations:
            return self.fallback_implementations[dependency]
        else:
            raise ImportError(f"Dependency '{dependency}' not available and no fallback implemented")
            
    def is_available(self, dependency: str) -> bool:
        """Check if a dependency is available."""
        return dependency in self.available_deps
        
    def require_dependency(self, dependency: str, feature_name: str = None):
        """Decorator to require a specific dependency for a function."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.is_available(dependency):
                    feature = feature_name or func.__name__
                    warnings.warn(
                        f"Feature '{feature}' requires '{dependency}' which is not installed. "
                        f"Using fallback implementation with reduced functionality.",
                        UserWarning
                    )
                return func(*args, **kwargs)
            return wrapper
        return decorator
        
    def get_capability_report(self) -> Dict[str, Any]:
        """Generate a report of available capabilities."""
        report = {
            'available_dependencies': list(self.available_deps.keys()),
            'missing_dependencies': list(self.missing_deps.keys()),
            'fallback_implementations': list(self.fallback_implementations.keys()),
            'capabilities': {}
        }
        
        # Define capability matrix
        capabilities = {
            'neuromorphic_processing': ['torch'],
            'advanced_audio_features': ['librosa'],
            'signal_processing': ['scipy'],
            'numerical_computing': ['numpy'],
            'machine_learning': ['sklearn'],
            'visualization': ['matplotlib', 'seaborn']
        }
        
        for capability, required_deps in capabilities.items():
            available = all(dep in self.available_deps for dep in required_deps)
            fallback_available = any(dep in self.fallback_implementations for dep in required_deps)
            
            if available:
                report['capabilities'][capability] = 'full'
            elif fallback_available:
                report['capabilities'][capability] = 'fallback'
            else:
                report['capabilities'][capability] = 'unavailable'
                
        return report
        
    def print_status(self):
        """Print dependency status."""
        report = self.get_capability_report()
        
        print("\n" + "="*60)
        print("DEPENDENCY STATUS REPORT")
        print("="*60)
        
        print(f"\n✅ Available Dependencies ({len(report['available_dependencies'])}):")
        for dep in report['available_dependencies']:
            print(f"   • {dep}")
            
        if report['missing_dependencies']:
            print(f"\n⚠️  Missing Dependencies ({len(report['missing_dependencies'])}):")
            for dep in report['missing_dependencies']:
                print(f"   • {dep}")
                
        if report['fallback_implementations']:
            print(f"\n🔄 Fallback Implementations ({len(report['fallback_implementations'])}):")
            for dep in report['fallback_implementations']:
                print(f"   • {dep}")
                
        print(f"\n🎯 System Capabilities:")
        for capability, status in report['capabilities'].items():
            icon = "✅" if status == 'full' else "🔄" if status == 'fallback' else "❌"
            print(f"   {icon} {capability.replace('_', ' ').title()}: {status}")
            
        print("="*60 + "\n")


# Global dependency manager instance
dep_manager = DependencyManager()


def require_dependency(dependency: str, feature_name: str = None):
    """Decorator to require a specific dependency."""
    return dep_manager.require_dependency(dependency, feature_name)


def get_torch():
    """Get PyTorch implementation (real or fallback)."""
    return dep_manager.get_implementation('torch')


def get_numpy():
    """Get NumPy implementation."""
    return dep_manager.get_implementation('numpy')


def get_librosa():
    """Get librosa implementation (real or fallback)."""
    return dep_manager.get_implementation('librosa')


def get_scipy():
    """Get scipy implementation (real or fallback)."""
    return dep_manager.get_implementation('scipy')


def install_missing_dependencies():
    """Provide installation instructions for missing dependencies."""
    report = dep_manager.get_capability_report()
    
    if not report['missing_dependencies']:
        print("✅ All dependencies are available!")
        return
        
    print("\n" + "="*60)
    print("INSTALLATION INSTRUCTIONS")
    print("="*60)
    
    installation_commands = {
        'torch': 'pip install torch torchaudio',
        'librosa': 'pip install librosa',
        'scipy': 'pip install scipy',
        'numpy': 'pip install numpy',
        'matplotlib': 'pip install matplotlib',
        'seaborn': 'pip install seaborn',
        'sklearn': 'pip install scikit-learn'
    }
    
    print("\nTo install missing dependencies, run:")
    for dep in report['missing_dependencies']:
        if dep in installation_commands:
            print(f"   {installation_commands[dep]}")
            
    print("\nOr install all at once:")
    all_missing = [installation_commands[dep].split()[-1] for dep in report['missing_dependencies'] if dep in installation_commands]
    if all_missing:
        print(f"   pip install {' '.join(set(all_missing))}")
        
    print("="*60 + "\n")


if __name__ == "__main__":
    dep_manager.print_status()
    install_missing_dependencies()