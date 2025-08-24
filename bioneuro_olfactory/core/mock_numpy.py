"""Mock numpy module for testing without dependencies."""

import random as _random
import math
from typing import List, Union, Any


class MockArray:
    """Mock numpy array."""
    
    def __init__(self, data):
        if isinstance(data, (list, tuple)):
            if isinstance(data[0], (list, tuple)):  # 2D array
                self.data = [list(row) for row in data]
                self._shape = (len(data), len(data[0]) if data else 0)
            else:  # 1D array
                self.data = list(data)
                self._shape = (len(self.data),)
        else:
            self.data = [data]
            self._shape = (1,)
    
    @property 
    def shape(self):
        return self._shape
    
    @property
    def ndim(self):
        return len(self._shape)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __setitem__(self, idx, value):
        self.data[idx] = value
    
    def __add__(self, other):
        if isinstance(other, MockArray):
            return MockArray([a + b for a, b in zip(self.data, other.data)])
        else:
            return MockArray([a + other for a in self.data])
    
    def __mul__(self, other):
        if isinstance(other, MockArray):
            return MockArray([a * b for a, b in zip(self.data, other.data)])
        else:
            return MockArray([a * other for a in self.data])
    
    def __matmul__(self, other):
        """Matrix multiplication."""
        if isinstance(other, MockArray):
            if self.ndim == 2 and other.ndim == 1:
                # Matrix @ vector
                result = []
                for row in self.data:
                    result.append(sum(a * b for a, b in zip(row, other.data)))
                return MockArray(result)
            elif self.ndim == 1 and other.ndim == 1:
                # Vector @ vector (dot product)
                return sum(a * b for a, b in zip(self.data, other.data))
            else:
                # Fallback to element-wise
                return MockArray([a * b for a, b in zip(self.data, other.data)])
        else:
            return MockArray([a * other for a in self.data])
    
    def astype(self, dtype):
        if dtype == float:
            if self.ndim == 1:
                return MockArray([float(x) for x in self.data])
            else:
                return MockArray([[float(x) for x in row] for row in self.data])
        elif dtype == bool:
            if self.ndim == 1:
                return MockArray([bool(x) for x in self.data])
            else:
                return MockArray([[bool(x) for x in row] for row in self.data])
        return MockArray(self.data)
    
    def reshape(self, *shape):
        # Simple reshape - just return same data
        return MockArray(self.data)


def array(data):
    """Create mock array."""
    return MockArray(data)


def zeros(shape, dtype=float):
    """Create zeros array."""
    if isinstance(shape, tuple):
        if len(shape) == 2:
            return MockArray([[0.0] * shape[1] for _ in range(shape[0])])
        else:
            size = shape[0] if len(shape) > 0 else 1
            return MockArray([0.0] * size)
    else:
        return MockArray([0.0] * shape)


def ones(shape, dtype=float):
    """Create ones array.""" 
    if isinstance(shape, tuple):
        if len(shape) == 2:
            return MockArray([[1.0] * shape[1] for _ in range(shape[0])])
        else:
            size = shape[0] if len(shape) > 0 else 1
            return MockArray([1.0] * size)
    else:
        return MockArray([1.0] * shape)


def concatenate(arrays):
    """Concatenate arrays."""
    result = []
    for arr in arrays:
        if isinstance(arr, MockArray):
            result.extend(arr.data)
        elif isinstance(arr, (list, tuple)):
            result.extend(arr)
        else:
            result.append(arr)
    return MockArray(result)


def mean(arr, axis=None):
    """Calculate mean."""
    if isinstance(arr, MockArray):
        if arr.ndim == 1:
            return sum(arr.data) / len(arr.data) if arr.data else 0
        else:
            # 2D array - flatten and compute mean
            flat = [x for row in arr.data for x in row]
            return sum(flat) / len(flat) if flat else 0
    else:
        return arr


def std(arr, axis=None):
    """Calculate standard deviation."""
    if isinstance(arr, MockArray):
        mean_val = mean(arr)
        if arr.ndim == 1:
            variance = sum((x - mean_val) ** 2 for x in arr.data) / len(arr.data) if arr.data else 0
        else:
            flat = [x for row in arr.data for x in row]
            variance = sum((x - mean_val) ** 2 for x in flat) / len(flat) if flat else 0
        return math.sqrt(variance)
    else:
        return 0.0


def sum(arr, axis=None):
    """Calculate sum."""
    if isinstance(arr, MockArray):
        if arr.ndim == 1:
            return sum(arr.data)
        else:
            return sum(sum(row) for row in arr.data)
    else:
        return arr


def maximum(arr1, arr2):
    """Element-wise maximum."""
    if isinstance(arr1, MockArray) and isinstance(arr2, MockArray):
        return MockArray([max(a, b) for a, b in zip(arr1.data, arr2.data)])
    elif isinstance(arr1, MockArray):
        return MockArray([max(a, arr2) for a in arr1.data])
    elif isinstance(arr2, MockArray):
        return MockArray([max(arr1, b) for b in arr2.data])
    else:
        return max(arr1, arr2)


def tanh(arr):
    """Hyperbolic tangent."""
    if isinstance(arr, MockArray):
        if arr.ndim == 1:
            return MockArray([math.tanh(x) for x in arr.data])
        else:
            return MockArray([[math.tanh(x) for x in row] for row in arr.data])
    else:
        return math.tanh(arr)


def exp(arr):
    """Exponential function."""
    if isinstance(arr, MockArray):
        if arr.ndim == 1:
            return MockArray([math.exp(min(x, 700)) for x in arr.data])  # Prevent overflow
        else:
            return MockArray([[math.exp(min(x, 700)) for x in row] for row in arr.data])
    else:
        return math.exp(min(arr, 700))


def clip(arr, min_val, max_val):
    """Clip values."""
    if isinstance(arr, MockArray):
        if arr.ndim == 1:
            return MockArray([max(min_val, min(max_val, x)) for x in arr.data])
        else:
            return MockArray([[max(min_val, min(max_val, x)) for x in row] for row in arr.data])
    else:
        return max(min_val, min(max_val, arr))


def fill_diagonal(arr, val):
    """Fill diagonal of 2D array."""
    if isinstance(arr, MockArray) and arr.ndim == 2:
        for i in range(min(len(arr.data), len(arr.data[0]) if arr.data else 0)):
            if i < len(arr.data) and i < len(arr.data[i]):
                arr.data[i][i] = val


class random:
    @staticmethod
    def randn(*shape):
        """Generate random normal numbers."""
        if len(shape) == 1:
            return MockArray([_random.gauss(0, 1) for _ in range(shape[0])])
        elif len(shape) == 2:
            return MockArray([[_random.gauss(0, 1) for _ in range(shape[1])] for _ in range(shape[0])])
        else:
            return MockArray([_random.gauss(0, 1)])
    
    @staticmethod  
    def random(shape=None):
        """Generate random numbers."""
        if shape is None:
            return _random.random()
        elif isinstance(shape, tuple):
            if len(shape) == 1:
                return MockArray([_random.random() for _ in range(shape[0])])
            elif len(shape) == 2:
                return MockArray([[_random.random() for _ in range(shape[1])] for _ in range(shape[0])])
        else:
            return MockArray([_random.random() for _ in range(shape)])


class linalg:
    @staticmethod
    def norm(arr):
        """Calculate vector norm.""" 
        if isinstance(arr, MockArray):
            if arr.ndim == 1:
                return math.sqrt(sum(x**2 for x in arr.data))
            else:
                flat = [x for row in arr.data for x in row]
                return math.sqrt(sum(x**2 for x in flat))
        else:
            return abs(arr)


# Module-level variables
ndim = 1