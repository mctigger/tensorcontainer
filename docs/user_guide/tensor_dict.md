# TensorDict Deep Dive

TensorDict is a dictionary-style TensorContainer that combines Python's dictionary interface with tensor operations, providing dynamic, key-based access to collections of tensors with shared batch dimensions. Use it when your schema changes during development or when you need flexible key-based access.

## Quick Start

Get started with `TensorDict` by creating a dictionary-like container for your tensors. The container automatically manages shape consistency and applies tensor operations to all values at once.

```python
import torch
from tensorcontainer import TensorDict

# 1. Create a TensorDict with related tensors
data = TensorDict({
    'observations': torch.randn(4, 10),
    'actions': torch.randn(4, 3),
    'rewards': torch.randn(4),
}, shape=(4,), device='cpu')

# 2. Dictionary-style access
obs = data['observations']
data['values'] = torch.randn(4)

# 3. Apply unified operations to all tensors
data_cuda = data.to('cuda')
data_reshaped = data.reshape(2, 2)
data_detached = data.detach()

# 4. Dictionary-like interface
assert len(data) == 3
assert 'observations' in data
```
*Example available at: [examples/tensor_dict/01_basic.py](../../examples/tensor_dict/01_basic.py)*


## Construction and Validation

To create a `TensorDict`, you provide a dictionary of tensors along with the mandatory `shape` and `device` arguments. The class validates two principal constraints on initialization:

1. **Shape Consistency**: Every tensor's leading dimensions must match the `shape` argument.
2. **Device Consistency**: Every tensor must reside on the specified `device` (unless `device=None`).

```python
# Invalid shape: raises an error because the shape argument must be a prefix of every tensor's shape
try:
    TensorDict({
        'x': torch.randn(4, 10),
        'y': torch.randn(3, 5),  # Wrong batch size (3 vs 4)
    }, shape=(4,), device='cpu')
except Exception as e:
    # An error is expected here due to the shape mismatch
    print(e)
```
*Example available at: [examples/tensor_dict/01_basic.py](../../examples/tensor_dict/01_basic.py)*


## Indexing and Slicing

Indexing operates exclusively on the batch dimensions, leaving event dimensions untouched. It mirrors `torch.Tensor` indexing and returns a new `TensorDict` instance that is a **view** of the original data.

```python
import torch
from tensorcontainer import TensorDict

# Create batch of tensors
data = TensorDict({
    'x': torch.randn(6, 4),
    'y': torch.randn(6, 2),
}, shape=(6,), device='cpu')

# Test indexing returns correct shapes
single_item = data[0]
assert single_item.shape == ()
assert single_item['x'].shape == (4,)

# Test slicing returns correct shapes
slice_data = data[2:5]
assert slice_data.shape == (3,)

# You can assign to an indexed TensorDict using another instance with matching batch shape
replacement = TensorDict({
    'x': torch.ones(3, 4),
    'y': torch.zeros(3, 2)
}, shape=(3,), device='cpu')
data[2:5] = replacement
```
*Example available at: [examples/tensor_dict/02_indexing.py](../../examples/tensor_dict/02_indexing.py)*


## Shape Operations

Shape operations like `reshape()`, `view()`, `squeeze()`, and `unsqueeze()` apply only to the **batch dimensions**. Event dimensions are automatically preserved.

```python
import torch
from tensorcontainer import TensorDict

data = TensorDict({
    'observations': torch.randn(2, 3, 128),
    'actions': torch.randn(2, 3, 6),
    'rewards': torch.randn(2, 3),
}, shape=(2, 3), device='cpu')

# Reshape only affects batch dimensions
reshaped = data.reshape(6)
assert reshaped.shape == (6,)
assert reshaped['observations'].shape == (6, 128)  # Event dimension preserved

# Unsqueeze adds batch dimension
unsqueezed = data.unsqueeze(0)
assert unsqueezed.shape == (1, 2, 3)
```
*Example available at: [examples/tensor_dict/03_shape_ops.py](../../examples/tensor_dict/03_shape_ops.py)*


## Stacking and Concatenation

You can combine multiple `TensorDict` instances using `torch.stack` and `torch.cat`.

- `torch.stack`: Creates a new batch dimension.
- `torch.cat`: Concatenates along an existing batch dimension.

```python
import torch
from tensorcontainer import TensorDict

batch1 = TensorDict({'x': torch.randn(3, 4), 'y': torch.randn(3, 2)}, shape=(3,), device='cpu')
batch2 = TensorDict({'x': torch.randn(3, 4), 'y': torch.randn(3, 2)}, shape=(3,), device='cpu')

# Stack creates new batch dimension
stacked = torch.stack([batch1, batch2], dim=0)
assert stacked.shape == (2, 3)
assert stacked['x'].shape == (2, 3, 4)

# Concatenate along existing dimension
concatenated = torch.cat([batch1, batch2], dim=0)
assert concatenated.shape == (6,)
```
*Example available at: [examples/tensor_dict/04_stack.py](../../examples/tensor_dict/04_stack.py)*


## Device Operations

The `.to()` method provides a convenient way to move all tensor values in a `TensorDict` instance to a different device (CPU, GPU, etc.) in a single operation.

```python
import torch
from tensorcontainer import TensorDict

# Create instance on CPU
data_cpu = TensorDict({'x': torch.rand(2, 2), 'y': torch.rand(2, 5)}, shape=(2,), device='cpu')

# Move to CUDA device
data_cuda = data_cpu.to(device='cuda')

# Verify all tensors moved to CUDA
assert data_cuda.device == torch.device('cuda:0')
assert data_cuda['x'].device == torch.device('cuda:0')
assert data_cuda['y'].device == torch.device('cuda:0')
```
*Example available at: [examples/tensor_dict/05_device.py](../../examples/tensor_dict/05_device.py)*


## Copy and Clone Operations

- **`copy.copy()`**: Creates a shallow copy of the instance (new instance, shared tensors).
- **`.clone()`**: Creates a deep copy of the instance with new, independent tensor storage.

```python
import torch
import copy
from tensorcontainer import TensorDict

original = TensorDict({'x': torch.randn(3, 4), 'y': torch.randn(3, 2)}, shape=(3,), device='cpu')

# Shallow copy shares tensor memory
shallow_copy = copy.copy(original)
assert shallow_copy['x'] is original['x']  # Same tensor objects

# Clone creates independent tensors
cloned = original.clone()
assert cloned['x'] is not original['x']  # Different tensor objects
```
*Example available at: [examples/tensor_dict/07_copy_clone.py](../../examples/tensor_dict/07_copy_clone.py)*


## Gradient Management

The `detach()` method is essential for gradient management in training scenarios. It creates a new `TensorDict` instance with the same data but stops gradient flow.

```python
import torch
from tensorcontainer import TensorDict

# Create batch with gradient tracking
batch = TensorDict({
    'observations': torch.randn(4, 10, requires_grad=True),
    'actions': torch.randn(4, 3, requires_grad=True),
}, shape=(4,), device='cpu')

# Verify gradients are initially tracked
assert batch['observations'].requires_grad
assert batch['actions'].requires_grad

# Detach to stop gradient flow
detached_batch = batch.detach()

# Verify gradients are no longer tracked
assert not detached_batch['observations'].requires_grad
assert not detached_batch['actions'].requires_grad
```
*Example available at: [examples/tensor_dict/08_detach_gradients.py](../../examples/tensor_dict/08_detach_gradients.py)*


## Nested Structures

You can nest `TensorDict` instances within each other to create complex, hierarchical data structures. All operations will propagate through the nested structure correctly.

```python
import torch
from tensorcontainer import TensorDict

# Create with appropriate tensors and shapes
agent_state = TensorDict({
    'actor_params': torch.randn(4, 20),
    'critic_params': torch.randn(4, 15),
}, shape=(4,), device='cpu')

full = TensorDict({
    'env_state': torch.randn(4, 10),
    'agent_state': agent_state,  # Nested TensorDict
}, shape=(4,), device='cpu')

# Operations on 'full' will propagate to 'agent_state'
# For example, moving to a different device
full_cuda = full.to('cuda')
assert full_cuda.device == torch.device('cuda:0')
assert full_cuda['agent_state'].device == torch.device('cuda:0')
assert full_cuda['agent_state']['actor_params'].device == torch.device('cuda:0')
```
*Example available at: [examples/tensor_dict/09_nested.py](../../examples/tensor_dict/09_nested.py)*


## TensorDict-Specific Features

TensorDict provides unique capabilities not available in other TensorContainer types, making it ideal for dynamic scenarios and exploratory development.

### Dynamic Key Management

Unlike `TensorDataClass`, `TensorDict` allows you to add and remove keys dynamically at runtime, providing flexibility for evolving schemas.

```python
import torch
from tensorcontainer import TensorDict

data = TensorDict({'observations': torch.randn(4, 10)}, shape=(4,), device='cpu')

# Add keys dynamically
data['actions'] = torch.randn(4, 3)
data['rewards'] = torch.randn(4)

# Remove keys dynamically
del data['rewards']
assert 'rewards' not in data
```
*Example available at: [examples/tensor_dict/10_dynamic_keys.py](../../examples/tensor_dict/10_dynamic_keys.py)*


### Dictionary Interface

TensorDict implements the full `MutableMapping` interface, providing familiar dictionary-like methods and iteration patterns.

```python
import torch
from tensorcontainer import TensorDict

data = TensorDict({'x': torch.randn(3, 4), 'y': torch.randn(3, 2)}, shape=(3,), device='cpu')

# Dictionary-like access methods
assert 'x' in data.keys()
assert 'y' in data.keys()
assert len(data) == 2

# Dictionary-like iteration
for key in data:
    assert key in ['x', 'y']

for key, value in data.items():
    assert isinstance(value, torch.Tensor)

# Update from another TensorDict or dictionary
other = TensorDict({'z': torch.randn(3, 1)}, shape=(3,), device='cpu')
data.update(other)
assert 'z' in data
```
*Example available at: [examples/tensor_dict/11_mapping_interface.py](../../examples/tensor_dict/11_mapping_interface.py)*


### Automatic Dict Conversion

TensorDict automatically converts nested Python dictionaries into nested TensorDict instances, simplifying the construction of hierarchical data structures.

```python
import torch
from tensorcontainer import TensorDict

# Nested dictionaries are automatically converted to TensorDict instances
data = TensorDict({
    'agent': {
        'position': torch.randn(3, 2),
        'velocity': torch.randn(3, 2),
    },
    'tensor': torch.randn(3, 5),
}, shape=(3,), device='cpu')

# Verify nested structure
assert isinstance(data['agent'], TensorDict)
assert data['agent']['position'].shape == (3, 2)
```
*Example available at: [examples/tensor_dict/12_nested_dict_handling.py](../../examples/tensor_dict/12_nested_dict_handling.py)*


### Key Flattening

TensorDict provides powerful key flattening capabilities to convert nested structures into flat namespaces with dot-separated keys while sharing tensor memory.

```python
import torch
from tensorcontainer import TensorDict

nested = TensorDict({
    'env': {
        'obs': torch.randn(3, 10),
        'info': {'step': torch.tensor([1, 2, 3])},
    },
    'agent': {'policy': torch.randn(3, 6)},
}, shape=(3,), device='cpu')

# Flatten with default dot separator
flattened = nested.flatten_keys()
assert 'env.obs' in flattened
assert 'env.info.step' in flattened
assert 'agent.policy' in flattened

# Memory is shared between original and flattened
assert nested['env']['obs'] is flattened['env.obs']
assert torch.equal(nested['env']['info']['step'], flattened['env.info.step'])
```
*Example available at: [examples/tensor_dict/13_flatten_keys.py](../../examples/tensor_dict/13_flatten_keys.py)*

---

## Comparison and Decision Guide

### TensorDict vs. TensorDataClass

| Feature          | TensorDict                          | TensorDataClass                   |
|------------------|-------------------------------------|-----------------------------------|
| **Access**       | `obj["key"]` (dictionary)           | `obj.field` (attribute)           |
| **Schema**       | Dynamic, keys added at runtime      | Static, defined at class level    |
| **Type Safety**  | Runtime                             | Compile-time (static analysis)    |
| **IDE Support**  | Limited to string keys              | Full autocomplete & refactoring   |
| **Flexibility**  | High - modify structure at runtime  | Low - fixed schema               |
| **Use Case**     | Exploratory, dynamic structures     | Stable, production-ready schemas  |

Choose `TensorDict` for dynamic scenarios where the contents might change during execution, prototyping, or when you need dictionary-like flexibility. Choose `TensorDataClass` when you have a well-defined, stable data structure and can benefit from static type checking and IDE support.