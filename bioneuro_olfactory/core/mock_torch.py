"""Mock PyTorch interface for environments without torch installed.

This provides a lightweight interface for development and testing
when PyTorch is not available in the system environment.
"""

import math
import random
from typing import Tuple, List, Any, Dict, Optional, Union


class MockTensor:
    """Mock tensor class that mimics basic PyTorch tensor operations."""
    
    def __init__(self, data: Any, shape: Tuple[int, ...] = None):
        if isinstance(data, (list, tuple)):
            self.data = self._flatten_nested_list(data)
            if shape is None:
                self.shape = self._infer_shape(data)
            else:
                self.shape = shape
        elif isinstance(data, (int, float)):
            self.data = [data]
            self.shape = (1,) if shape is None else shape
        else:
            self.data = data if isinstance(data, list) else [data]
            self.shape = shape or (len(self.data),)
            
        self.requires_grad = False
        
    def _flatten_nested_list(self, nested_list: Any) -> List[float]:
        """Flatten nested list structure."""
        result = []
        if isinstance(nested_list, (list, tuple)):
            for item in nested_list:
                if isinstance(item, (list, tuple)):
                    result.extend(self._flatten_nested_list(item))
                else:
                    result.append(float(item))
        else:
            result.append(float(nested_list))
        return result
    
    def _infer_shape(self, nested_list: Any) -> Tuple[int, ...]:
        """Infer shape from nested list."""
        if not isinstance(nested_list, (list, tuple)):
            return (1,)
        
        shape = [len(nested_list)]
        if len(nested_list) > 0 and isinstance(nested_list[0], (list, tuple)):
            inner_shape = self._infer_shape(nested_list[0])
            shape.extend(inner_shape)
        return tuple(shape)
    
    def size(self, dim: int = None) -> Union[Tuple[int, ...], int]:
        """Get tensor size."""
        if dim is None:
            return self.shape
        return self.shape[dim] if dim < len(self.shape) else 1
    
    def dim(self) -> int:
        """Get number of dimensions."""
        return len(self.shape)
    
    def numel(self) -> int:
        """Get total number of elements."""
        return len(self.data)
    
    def view(self, *shape) -> 'MockTensor':
        """Reshape tensor."""
        new_tensor = MockTensor(self.data.copy())
        new_tensor.shape = shape
        return new_tensor
    
    def reshape(self, *shape) -> 'MockTensor':
        """Reshape tensor."""
        return self.view(*shape)
    
    def unsqueeze(self, dim: int) -> 'MockTensor':
        """Add dimension."""
        new_shape = list(self.shape)
        new_shape.insert(dim, 1)
        return self.view(*new_shape)
    
    def squeeze(self, dim: int = None) -> 'MockTensor':
        """Remove dimensions of size 1."""
        new_shape = list(self.shape)
        if dim is None:
            new_shape = [s for s in new_shape if s != 1]
        else:
            if new_shape[dim] == 1:
                new_shape.pop(dim)
        return self.view(*new_shape)
    
    def mean(self, dim: int = None, keepdim: bool = False) -> 'MockTensor':
        """Calculate mean."""
        if dim is None:
            result = sum(self.data) / len(self.data)
            return MockTensor([result])
        # Simplified - just return same tensor
        return MockTensor(self.data.copy(), self.shape)
    
    def sum(self, dim: int = None, keepdim: bool = False) -> 'MockTensor':
        """Calculate sum."""
        if dim is None:
            result = sum(self.data)
            return MockTensor([result])
        # Simplified - just return same tensor
        return MockTensor(self.data.copy(), self.shape)
    
    def max(self, dim: int = None) -> Union['MockTensor', Tuple['MockTensor', 'MockTensor']]:
        """Calculate maximum."""
        if dim is None:
            result = max(self.data)
            return MockTensor([result])
        
        # For dim specified, return values and indices
        values = MockTensor(self.data.copy(), self.shape)
        indices = MockTensor([0] * len(self.data), self.shape)
        return values, indices
    
    def argmax(self, dim: int = None) -> 'MockTensor':
        """Calculate argmax."""
        if dim is None:
            idx = self.data.index(max(self.data))
            return MockTensor([idx])
        # Simplified
        return MockTensor([0] * len(self.data), self.shape)
    
    def clone(self) -> 'MockTensor':
        """Clone tensor."""
        return MockTensor(self.data.copy(), self.shape)
    
    def detach(self) -> 'MockTensor':
        """Detach from computation graph."""
        return self.clone()
    
    def numpy(self) -> List[float]:
        """Convert to numpy-like list."""
        return self.data.copy()
    
    def float(self) -> 'MockTensor':
        """Convert to float tensor."""
        return MockTensor([float(x) for x in self.data], self.shape)
    
    def cuda(self, device: str = None) -> 'MockTensor':
        """Mock GPU transfer."""
        return self.clone()
    
    def cpu(self) -> 'MockTensor':
        """Mock CPU transfer."""
        return self.clone()
    
    def __add__(self, other) -> 'MockTensor':
        if isinstance(other, MockTensor):
            result_data = [a + b for a, b in zip(self.data, other.data)]
        else:
            result_data = [a + other for a in self.data]
        return MockTensor(result_data, self.shape)
    
    def __sub__(self, other) -> 'MockTensor':
        if isinstance(other, MockTensor):
            result_data = [a - b for a, b in zip(self.data, other.data)]
        else:
            result_data = [a - other for a in self.data]
        return MockTensor(result_data, self.shape)
    
    def __mul__(self, other) -> 'MockTensor':
        if isinstance(other, MockTensor):
            result_data = [a * b for a, b in zip(self.data, other.data)]
        else:
            result_data = [a * other for a in self.data]
        return MockTensor(result_data, self.shape)
    
    def __truediv__(self, other) -> 'MockTensor':
        if isinstance(other, MockTensor):
            result_data = [a / b for a, b in zip(self.data, other.data)]
        else:
            result_data = [a / other for a in self.data]
        return MockTensor(result_data, self.shape)
    
    def __getitem__(self, key):
        """Tensor indexing."""
        if isinstance(key, int):
            return MockTensor([self.data[key]])
        # Simplified indexing
        return self.clone()
    
    def __setitem__(self, key, value):
        """Tensor assignment."""
        if isinstance(key, int) and key < len(self.data):
            self.data[key] = float(value)
    
    def __repr__(self) -> str:
        return f"MockTensor(shape={self.shape}, data={self.data[:5]}{'...' if len(self.data) > 5 else ''})"


class MockParameter(MockTensor):
    """Mock parameter class."""
    
    def __init__(self, data: Any, requires_grad: bool = True):
        super().__init__(data)
        self.requires_grad = requires_grad
        
    @property 
    def data(self):
        return MockTensor(self._data, self.shape)
    
    @data.setter
    def data(self, value):
        self._data = value if isinstance(value, list) else [value]


class MockModule:
    """Mock neural network module."""
    
    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self.training = True
        
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement forward")
    
    def parameters(self):
        """Return parameters."""
        for param in self._parameters.values():
            yield param
        for module in self._modules.values():
            if hasattr(module, 'parameters'):
                for param in module.parameters():
                    yield param
    
    def named_parameters(self):
        """Return named parameters."""
        for name, param in self._parameters.items():
            yield name, param
        for name, module in self._modules.items():
            if hasattr(module, 'named_parameters'):
                for param_name, param in module.named_parameters():
                    yield f"{name}.{param_name}", param
    
    def train(self, mode: bool = True):
        """Set training mode."""
        self.training = mode
        return self
    
    def eval(self):
        """Set evaluation mode."""
        return self.train(False)
    
    def cuda(self, device: str = None):
        """Mock GPU transfer."""
        return self
    
    def cpu(self):
        """Mock CPU transfer."""
        return self


class MockLinear(MockModule):
    """Mock linear layer."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize random weights
        weight_data = [[random.gauss(0, 0.1) for _ in range(in_features)] 
                       for _ in range(out_features)]
        self.weight = MockParameter(weight_data)
        self._parameters['weight'] = self.weight
        
        if bias:
            bias_data = [random.gauss(0, 0.1) for _ in range(out_features)]
            self.bias = MockParameter(bias_data)
            self._parameters['bias'] = self.bias
        else:
            self.bias = None
    
    def forward(self, input_tensor: MockTensor) -> MockTensor:
        """Forward pass."""
        # Simplified matrix multiplication
        batch_size = input_tensor.size(0) if input_tensor.dim() > 1 else 1
        output_data = [random.gauss(0, 1) for _ in range(batch_size * self.out_features)]
        output_shape = (batch_size, self.out_features) if batch_size > 1 else (self.out_features,)
        return MockTensor(output_data, output_shape)


class MockSequential(MockModule):
    """Mock sequential container."""
    
    def __init__(self, *layers):
        super().__init__()
        for i, layer in enumerate(layers):
            self._modules[str(i)] = layer
    
    def forward(self, x: MockTensor) -> MockTensor:
        """Forward pass through layers."""
        for layer in self._modules.values():
            if hasattr(layer, '__call__'):
                x = layer(x)
        return x


class MockModuleList(list):
    """Mock module list."""
    
    def __init__(self, modules=None):
        super().__init__()
        if modules:
            self.extend(modules)


class MockBatchNorm1d(MockModule):
    """Mock 1D batch normalization."""
    
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features
        
    def forward(self, x: MockTensor) -> MockTensor:
        """Forward pass - just return input."""
        return x


class MockDropout(MockModule):
    """Mock dropout layer."""
    
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x: MockTensor) -> MockTensor:
        """Forward pass - just return input."""
        return x


class MockMultiheadAttention(MockModule):
    """Mock multihead attention."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
    def forward(self, query: MockTensor, key: MockTensor, value: MockTensor) -> Tuple[MockTensor, MockTensor]:
        """Forward pass."""
        # Return query as output and random attention weights
        attention_weights = MockTensor([[0.5, 0.5]], (1, 2))
        return query, attention_weights


# Mock PyTorch functions
def zeros(*shape) -> MockTensor:
    """Create zero tensor."""
    total_elements = 1
    for s in shape:
        total_elements *= s
    return MockTensor([0.0] * total_elements, shape)


def ones(*shape) -> MockTensor:
    """Create ones tensor."""
    total_elements = 1
    for s in shape:
        total_elements *= s
    return MockTensor([1.0] * total_elements, shape)


def randn(*shape) -> MockTensor:
    """Create random normal tensor."""
    total_elements = 1
    for s in shape:
        total_elements *= s
    data = [random.gauss(0, 1) for _ in range(total_elements)]
    return MockTensor(data, shape)


def rand(*shape) -> MockTensor:
    """Create random uniform tensor."""
    total_elements = 1
    for s in shape:
        total_elements *= s
    data = [random.random() for _ in range(total_elements)]
    return MockTensor(data, shape)


def eye(n: int) -> MockTensor:
    """Create identity matrix."""
    data = []
    for i in range(n):
        for j in range(n):
            data.append(1.0 if i == j else 0.0)
    return MockTensor(data, (n, n))


def tensor(data, dtype=None) -> MockTensor:
    """Create tensor from data."""
    return MockTensor(data)


def from_numpy(array) -> MockTensor:
    """Create tensor from numpy array."""
    return MockTensor(array)


def stack(tensors: List[MockTensor], dim: int = 0) -> MockTensor:
    """Stack tensors."""
    # Simplified - just combine data
    all_data = []
    for t in tensors:
        all_data.extend(t.data)
    
    # Calculate new shape
    if tensors:
        base_shape = list(tensors[0].shape)
        base_shape.insert(dim, len(tensors))
        return MockTensor(all_data, tuple(base_shape))
    return MockTensor([])


def cat(tensors: List[MockTensor], dim: int = 0) -> MockTensor:
    """Concatenate tensors."""
    # Simplified - just combine data
    all_data = []
    for t in tensors:
        all_data.extend(t.data)
    
    if tensors:
        base_shape = list(tensors[0].shape)
        base_shape[dim] = sum(t.shape[dim] for t in tensors)
        return MockTensor(all_data, tuple(base_shape))
    return MockTensor([])


def matmul(a: MockTensor, b: MockTensor) -> MockTensor:
    """Matrix multiplication."""
    # Simplified - return random result with appropriate shape
    if a.dim() == 1 and b.dim() == 1:
        # Dot product
        result = sum(x * y for x, y in zip(a.data, b.data))
        return MockTensor([result])
    elif a.dim() == 2 and b.dim() == 2:
        # Matrix multiplication
        m, k = a.shape
        k2, n = b.shape
        result_data = [random.gauss(0, 1) for _ in range(m * n)]
        return MockTensor(result_data, (m, n))
    else:
        # General case - return reasonable shape
        result_shape = a.shape[:-1] + b.shape[-1:]
        result_size = 1
        for s in result_shape:
            result_size *= s
        result_data = [random.gauss(0, 1) for _ in range(result_size)]
        return MockTensor(result_data, result_shape)


def dot(a: MockTensor, b: MockTensor) -> MockTensor:
    """Dot product."""
    result = sum(x * y for x, y in zip(a.data, b.data))
    return MockTensor([result])


def clamp(input_tensor: MockTensor, min_val: float = None, max_val: float = None) -> MockTensor:
    """Clamp tensor values."""
    data = input_tensor.data.copy()
    if min_val is not None:
        data = [max(x, min_val) for x in data]
    if max_val is not None:
        data = [min(x, max_val) for x in data]
    return MockTensor(data, input_tensor.shape)


def exp(input_tensor: MockTensor) -> MockTensor:
    """Element-wise exponential."""
    data = [math.exp(x) for x in input_tensor.data]
    return MockTensor(data, input_tensor.shape)


def sigmoid(input_tensor: MockTensor) -> MockTensor:
    """Sigmoid activation."""
    data = [1.0 / (1.0 + math.exp(-x)) for x in input_tensor.data]
    return MockTensor(data, input_tensor.shape)


def softmax(input_tensor: MockTensor, dim: int = -1) -> MockTensor:
    """Softmax activation."""
    # Simplified - normalize to sum to 1
    total = sum(math.exp(x) for x in input_tensor.data)
    if total == 0:
        total = 1
    data = [math.exp(x) / total for x in input_tensor.data]
    return MockTensor(data, input_tensor.shape)


# Mock torch.nn module
class nn:
    Module = MockModule
    Linear = MockLinear
    Sequential = MockSequential
    ModuleList = MockModuleList
    Parameter = MockParameter
    BatchNorm1d = MockBatchNorm1d
    Dropout = MockDropout
    MultiheadAttention = MockMultiheadAttention
    
    class functional:
        @staticmethod
        def relu(x: MockTensor) -> MockTensor:
            """ReLU activation."""
            data = [max(0, val) for val in x.data]
            return MockTensor(data, x.shape)
        
        @staticmethod 
        def sigmoid(x: MockTensor) -> MockTensor:
            """Sigmoid activation."""
            return sigmoid(x)
        
        @staticmethod
        def softmax(x: MockTensor, dim: int = -1) -> MockTensor:
            """Softmax activation."""
            return softmax(x, dim)
    
    F = functional


# Export mock torch interface
def get_mock_torch():
    """Get mock torch module."""
    class MockTorch:
        Tensor = MockTensor
        tensor = tensor
        zeros = zeros
        ones = ones
        randn = randn
        rand = rand
        eye = eye
        from_numpy = from_numpy
        stack = stack
        cat = cat
        matmul = matmul
        dot = dot
        clamp = clamp
        exp = exp
        sigmoid = sigmoid
        softmax = softmax
        nn = nn
        
        @property
        def __version__(self):
            return "mock-1.0.0"
    
    return MockTorch()


# Mock numpy as well
class MockNumpy:
    """Mock numpy interface."""
    
    # Add ndarray as alias for list
    ndarray = list
    
    @staticmethod
    def array(data) -> List:
        """Create array."""
        if isinstance(data, (list, tuple)):
            return list(data)
        return [data]
    
    @staticmethod
    def zeros(shape) -> List:
        """Create zeros array."""
        if isinstance(shape, int):
            return [0.0] * shape
        total = 1
        for s in shape:
            total *= s
        return [0.0] * total
    
    @staticmethod
    def ones(shape) -> List:
        """Create ones array."""
        if isinstance(shape, int):
            return [1.0] * shape
        total = 1
        for s in shape:
            total *= s
        return [1.0] * total
    
    @staticmethod
    def random(shape=None) -> Union[float, List]:
        """Create random array."""
        if shape is None:
            return random.random()
        if isinstance(shape, int):
            return [random.random() for _ in range(shape)]
        total = 1
        for s in shape:
            total *= s
        return [random.random() for _ in range(total)]
    
    @staticmethod
    def linspace(start: float, stop: float, num: int = 50) -> List[float]:
        """Create linearly spaced array."""
        if num <= 1:
            return [start]
        step = (stop - start) / (num - 1)
        return [start + i * step for i in range(num)]
    
    @staticmethod
    def interp(x: List[float], xp: List[float], fp: List[float]) -> List[float]:
        """Linear interpolation."""
        # Simplified interpolation
        result = []
        for xi in x:
            # Find closest points
            if xi <= xp[0]:
                result.append(fp[0])
            elif xi >= xp[-1]:
                result.append(fp[-1])
            else:
                # Linear interpolation between closest points
                for i in range(len(xp) - 1):
                    if xp[i] <= xi <= xp[i + 1]:
                        t = (xi - xp[i]) / (xp[i + 1] - xp[i])
                        result.append(fp[i] + t * (fp[i + 1] - fp[i]))
                        break
        return result
    
    @property
    def __version__(self):
        return "mock-1.0.0"


def get_mock_numpy():
    """Get mock numpy module."""
    return MockNumpy()