# Tensor Container

*Tensor containers for PyTorch with PyTree compatibility and torch.compile optimization*

[![Docs](https://img.shields.io/static/v1?logo=github&style=flat&color=pink&label=docs&message=tensorcontainer)](tree/main/docs)
[![Documentation](https://img.shields.io/badge/docs-local-blue)](./docs/user_guide/README.md)
[![Python 3.9, 3.10, 3.11, 3.12](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-blue.svg)](https://pytorch.org/)
<a href="https://pypi.org/project/tensorcontainer"><img src="https://img.shields.io/pypi/v/tensorcontainer" alt="pypi version"></a>


Tensor Container provides efficient, type-safe tensor container implementations for PyTorch workflows. It includes PyTree integration and torch.compile optimization for batched tensor operations.

The library includes tensor containers (dict, dataclass) and distributions (torch.distributions equivalent).

## What is TensorContainer?

TensorContainer transforms how you work with structured tensor data in PyTorch by providing **tensor-like operations for entire data structures**. Instead of manually managing individual tensors, TensorContainer lets you treat complex data as unified entities that behave just like regular tensors.

### **Core Benefits**

- **Unified Operations**: Apply tensor operations like `view()`, `permute()`, `detach()`, and device transfers to entire data structures
- **Drop-in Compatibility**: Seamless integration with existing PyTorch workflows and `torch.compile`
- **Zero Boilerplate**: Eliminate manual parameter handling and type-specific operations
- **Type Safety**: Full IDE support with static typing and autocomplete

```python
data = TensorDict(
   {"a": torch.rand(24), "b": torch.rand(24)}, 
   shape=(24,), 
   device="cpu"
)

# Single operation transforms entire structure
data = data.view(2, 3, 4).permute(1, 0, 2).to('cuda').detach()
```

### **Key Features**

- **⚡ JIT Compilation**: Designed for `torch.compile` with `fullgraph=True`, minimizing graph breaks and maximizing performance
- **📐 Batch/Event Semantics**: Clear distinction between batch dimensions (consistent across tensors) and event dimensions (tensor-specific)
- **🔄 Device Management**: Move entire structures between CPU/GPU with single operations and flexible device compatibility
- **🔒 Type Safety**: Full IDE support with static typing and autocomplete
- **🏗️ Multiple Container Types**: Three specialized containers for different use cases:
  - `TensorDict` for dynamic, dictionary-style data collections
  - `TensorDataClass` for type-safe, dataclass-based structures
  - `TensorDistribution` for probabilistic modeling with 40+ probability distributions
- **🔧 Advanced Operations**: Full PyTorch tensor operations support including `view()`, `permute()`, `stack()`, `cat()`, and more
- **🎯 Advanced Indexing**: Complete PyTorch indexing semantics with boolean masks, tensor indices, and ellipsis support
- **📊 Shape Validation**: Automatic verification of tensor compatibility with detailed error messages
- **🌳 Nested Structure Support**: Create nested structure with different TensorContainers


## Table of Contents

- [What is TensorContainer?](#what-is-tensorcontainer)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Features](#features)
- [API Overview](#api-overview)
- [torch.compile Compatibility](#torchcompile-compatibility)
- [Examples](#examples)
- [Contributing](#contributing)
- [Documentation](#documentation)
- [License](#license)
- [Authors](#authors)
- [Contact and Support](#contact-and-support)

## Installation

### Using pip

```bash
pip install tensorcontainer
```

### Requirements

- Python 3.9+
- PyTorch 2.6+

## Quick Start

TensorContainer transforms how you work with structured tensor data. Instead of managing individual tensors, you can treat entire data structures as unified entities that behave like regular tensors.

```python
# Single operation transforms entire structure
data = data.view(2, 3, 4).permute(1, 0, 2).to('cuda').detach()
```

### 1. TensorDict: Dynamic Data Collections

Perfect for reinforcement learning data and dynamic collections:

```python
import torch
from tensorcontainer import TensorDict

# Create a container for RL training data
data = TensorDict({
    'observations': torch.randn(32, 128),
    'actions': torch.randn(32, 4),
    'rewards': torch.randn(32, 1)
}, shape=(32,))

# Dictionary-like access with tensor operations
obs = data['observations']
data['advantages'] = torch.randn(32, 1)  # Add new fields dynamically

# Batch operations work seamlessly
batch = torch.stack([data, data])  # Shape: (2, 32)
```

### 2. TensorDataClass: Type-Safe Structures

Ideal for model inputs and structured data with compile-time safety:

```python
import torch
from tensorcontainer import TensorDataClass

class ModelInput(TensorDataClass):
    features: torch.Tensor
    labels: torch.Tensor

# Create with full type safety and IDE support
batch = ModelInput(
    features=torch.randn(32, 64, 784),
    labels=torch.randint(0, 10, (32, 64)),
    shape=(32, 64)
)

# Unified operations on entire structure - reshape all tensors at once
batch = batch.view(2048)

# Type-safe access with autocomplete works on reshaped data too
loss = torch.nn.functional.cross_entropy(batch.features, batch.labels)
```

### 3. TensorDistribution: Probabilistic Modeling

Streamline probabilistic computations in reinforcement learning and generative models:

```python
import torch
from tensorcontainer.tensor_distribution import TensorNormal

normal = TensorNormal(
    loc=torch.zeros(100, 4),
    scale=torch.ones(100, 4)  
)

# With torch.distributions we need to extract the parameters, detach them
# and create a new Normal distribution. With TensorDistribution we just call
# .detach() on the distribution. We can also apply other tensor operations,
# such as .view()!
detached_normal = normal.detach()
```

## Documentation

The project includes comprehensive documentation:

- **[`docs/user_guide/overview.md`](docs/user_guide/overview.md)**: Complete user guide with examples and best practices
- **[`docs/developer_guide/compatibility.md`](docs/developer_guide/compatibility.md)**: Python version compatibility guide and best practices
- **[`docs/developer_guide/testing.md`](docs/developer_guide/testing.md)**: Testing philosophy, standards, and guidelines
- **Source Code Documentation**: Extensive docstrings and type annotations throughout the codebase
- **Test Coverage**: 643+ tests covering all major functionality with 86% code coverage

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- **Tim Joseph** - [mctigger](https://github.com/mctigger)

## Contact and Support

- **Issues**: Report bugs and request features on [GitHub Issues](https://github.com/mctigger/tensor-container/issues)
- **Discussions**: Join conversations on [GitHub Discussions](https://github.com/mctigger/tensor-container/discussions)
- **Email**: For direct inquiries, contact [tim@mctigger.com](mailto:tim@mctigger.com)

---

*Tensor Container is an academic research project for learning PyTorch internals and tensor container patterns. For production applications, we strongly recommend using the official [torch/tensordict](https://github.com/pytorch/tensordict) library, which is actively maintained by the PyTorch team.*