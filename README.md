# Tensor Container

*Modern tensor containers for PyTorch with PyTree compatibility and torch.compile optimization*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

> **⚠️ Academic Research Project**: This project exists solely for academic purposes to explore and learn PyTorch internals. For production use, please use the official, well-maintained [**torch/tensordict**](https://github.com/pytorch/tensordict) library.

Tensor Container provides efficient, type-safe tensor container implementations designed for modern PyTorch workflows. Built from the ground up with PyTree integration and torch.compile optimization, it enables seamless batched tensor operations with minimal overhead and maximum performance.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Features](#features)
- [API Overview](#api-overview)
- [torch.compile Compatibility](#torchcompile-compatibility)
- [Contributing](#contributing)
- [License](#license)
- [Authors](#authors)
- [Contact and Support](#contact-and-support)

## Installation

### From Source (Development)

```bash
# Clone the repository
git clone https://github.com/mctigger/tensor-container.git
cd tensor-container

# Install in development mode
pip install -e .

# Install with development dependencies
pip install -e .[dev]
```

### Requirements

- Python 3.8+
- PyTorch 2.0+

## Quick Start

### TensorDict: Dictionary-Style Containers

```python
import torch
from tensorcontainer import TensorDict

# Create a TensorDict with batch semantics
data = TensorDict({
    'observations': torch.randn(32, 128),
    'actions': torch.randn(32, 4),
    'rewards': torch.randn(32, 1)
}, shape=(32,), device='cpu')

# Dictionary-like access
obs = data['observations']
data['new_field'] = torch.zeros(32, 10)

# Batch operations work seamlessly
stacked_data = torch.stack([data, data])  # Shape: (2, 32)
```

### TensorDataClass: Type-Safe Containers

```python
import torch
from tensorcontainer import TensorDataClass

class RLData(TensorDataClass):
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor

# Create with full type safety and IDE support
data = RLData(
    observations=torch.randn(32, 128),
    actions=torch.randn(32, 4),
    rewards=torch.randn(32, 1),
    shape=(32,),
    device='cpu'
)

# Type-safe field access with autocomplete
obs = data.observations
data.actions = torch.randn(32, 8)  # Type-checked assignment
```

### PyTree Operations

```python
# All containers work seamlessly with PyTree operations
import torch.utils._pytree as pytree

# Transform all tensors in the container
doubled_data = pytree.tree_map(lambda x: x * 2, data)

# Combine multiple containers
combined = pytree.tree_map(lambda x, y: x + y, data1, data2)
```

## Features

- **🔥 torch.compile Optimized**: Built for maximum performance with PyTorch's JIT compiler
- **🌳 Native PyTree Support**: Seamless integration with `torch.utils._pytree` for tree operations
- **⚡ Zero-Copy Operations**: Efficient tensor sharing and manipulation without unnecessary copies
- **🎯 Type Safety**: Full static typing support with IDE autocomplete and type checking
- **📊 Batch Semantics**: Consistent batch/event dimension handling across all operations
- **🔍 Shape Validation**: Automatic validation of tensor shapes and device consistency
- **🏗️ Flexible Architecture**: Multiple container types for different use cases
- **🧪 Comprehensive Testing**: Extensive test suite with compile compatibility verification

## API Overview

### Core Components

- **`TensorContainer`**: Base class providing core tensor manipulation operations
- **`TensorDict`**: Dictionary-like container for dynamic tensor collections
- **`TensorDataClass`**: DataClass-based container for static, typed tensor structures
- **`TensorDistribution`**: Distribution wrapper for probabilistic tensor operations

### Key Concepts

- **Batch Dimensions**: Leading dimensions defined by the `shape` parameter, consistent across all tensors
- **Event Dimensions**: Trailing dimensions beyond batch shape, can vary per tensor
- **PyTree Integration**: All containers are registered PyTree nodes for seamless tree operations
- **Device Consistency**: Automatic validation ensures all tensors reside on compatible devices

## torch.compile Compatibility

Tensor Container is designed from the ground up for `torch.compile` compatibility:

```python
@torch.compile
def process_batch(data: TensorDict) -> TensorDict:
    return data.apply(lambda x: torch.relu(x))

# Compiles efficiently with minimal graph breaks
compiled_result = process_batch(tensor_dict)
```

Our testing framework includes comprehensive compile compatibility verification to ensure all operations work efficiently under JIT compilation.

## Contributing

We welcome contributions! Tensor Container is a learning project for exploring PyTorch internals and tensor container implementations.

### Development Setup

```bash
# Clone and install in development mode
git clone https://github.com/mctigger/tensor-container.git
cd tensor-container
pip install -e .[dev]
```

### Running Tests

```bash
# Run all tests with coverage
pytest --strict-markers --cov=src

# Run specific test modules
pytest tests/tensor_dict/test_compile.py
pytest tests/tensor_dataclass/
```

### Development Guidelines

- All new features must maintain `torch.compile` compatibility
- Comprehensive tests required, including compile compatibility verification
- Follow existing code patterns and typing conventions
- See `CLAUDE.md` for detailed architecture documentation

### Contribution Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with appropriate tests
4. Ensure all tests pass and maintain coverage
5. Submit a pull request with a clear description

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- **Tim Joseph** - *Creator and Lead Developer* - [mctigger](https://github.com/mctigger)

## Contact and Support

- **Issues**: Report bugs and request features on [GitHub Issues](https://github.com/mctigger/tensor-container/issues)
- **Discussions**: Join conversations on [GitHub Discussions](https://github.com/mctigger/tensor-container/discussions)
- **Email**: For direct inquiries, contact [tim@mctigger.com](mailto:tim@mctigger.com)

---

*Tensor Container is an academic research project for learning PyTorch internals and tensor container patterns. For production applications, we strongly recommend using the official [torch/tensordict](https://github.com/pytorch/tensordict) library, which is actively maintained by the PyTorch team.*