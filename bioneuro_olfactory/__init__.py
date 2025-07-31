"""
BioNeuro-Olfactory-Fusion: Neuromorphic Multi-Modal Gas Detection

A bio-inspired framework combining spiking neural networks with multi-modal
sensor fusion for real-time hazardous gas detection.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.com"

from .core import *
from .models import *
from .sensors import *

__all__ = [
    "__version__",
    "__author__",
    "__email__",
]