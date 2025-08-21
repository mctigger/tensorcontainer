# TensorDataClass Deep Dive

TensorDataClass is a dataclass-based TensorContainer that combines Python's dataclass system with tensor operations, providing type-safe, attribute-based access to collections of tensors with shared batch dimensions. Use it when you need static schemas, IDE support, and compile-time type checking.

## Quick Start

Get started with `TensorDataClass` by defining a class with type-annotated tensor fields. The container automatically generates a constructor and applies tensor operations to all fields at once.

```python
import torch
from tensorcontainer import TensorDataClass

# 1. Define a schema using class annotations
class SimpleData(TensorDataClass):
    observations: torch.Tensor
    actions: torch.Tensor

# 2. Instantiate with tensors and a shared batch shape
obs_tensor = torch.rand(2, 3)
act_tensor = torch.rand(2)
data = SimpleData(
    observations=obs_tensor,
    actions=act_tensor,
    shape=(2,),  # Batch dimensions shared by all fields
    device="cpu",
)

# 3. Access fields with type-safe attribute access
assert data.observations.shape == (2, 3)
assert data.actions.shape == (2,)

# 4. Apply unified operations to all tensors
data_reshaped = data.reshape(2)
assert data.shape == (2,)
assert data_reshaped.shape == (2,)
```
*Example available at: [examples/tensor_dataclass/01_basic.py](../../examples/tensor_dataclass/01_basic.py)*


## Defining Schemas

The schema of a `TensorDataClass` is defined using Python type annotations. This provides a clear, static structure that can be verified at compile time.

### Basic and Optional Fields

You can define required tensor fields, optional fields that can be `None`, and non-tensor metadata.

```python
from dataclasses import field
from typing import Optional, List
import torch
from tensorcontainer import TensorDataClass

class FlexibleData(TensorDataClass):
    # Required tensor
    features: torch.Tensor
    
    # Optional tensor, defaults to None
    labels: Optional[torch.Tensor] = None
    
    # Non-tensor metadata
    episode_ids: List[int] = field(default_factory=list)
```

### Inheritance

`TensorDataClass` supports standard Python inheritance, allowing you to build complex data schemas by extending base classes. Fields from all parent classes are automatically merged.

```python
import torch
from tensorcontainer import TensorDataClass

class Base(TensorDataClass):
    x: torch.Tensor

class Child(Base):
    y: torch.Tensor

# Create tensors with shared batch dimension
x = torch.rand(2, 3)
y = torch.rand(2, 5)

# Child inherits all fields from Base
data = Child(x=x, y=y, shape=(2,), device="cpu")

assert data.shape == (2,)
assert data.x.shape == (2, 3)
assert data.y.shape == (2, 5)
```
*Example available at: [examples/tensor_dataclass/06_inheritance.py](../../examples/tensor_dataclass/06_inheritance.py)*


## Construction and Validation

To create an instance, you must provide values for all annotated tensor fields, along with the mandatory `shape` and `device` arguments. The class validates two principal constraints on initialization:

1.  **Shape Consistency**: Every tensor's leading dimensions must match the `shape` argument.
2.  **Device Consistency**: Every tensor must reside on the specified `device`.

```python
# Invalid shape: raises an error because the shape argument must be a prefix of every field's shape
try:
    SimpleData(
        observations=torch.rand(2, 3),
        actions=torch.rand(2),
        shape=(3,),  # This shape (3,) is not a prefix of (2, 3) or (2,)
        device="cpu",
    )
except Exception as e:
    # An error is expected here due to the shape mismatch.
    print(e)
```
*Example available at: [examples/tensor_dataclass/01_basic.py](../../examples/tensor_dataclass/01_basic.py)*


## Indexing and Slicing

Indexing operates exclusively on the batch dimensions, leaving event dimensions untouched. It mirrors `torch.Tensor` indexing and returns a new `TensorDataClass` instance that is a **view** of the original data.

```python
import torch
from tensorcontainer import TensorDataClass

class DataPair(TensorDataClass):
    x: torch.Tensor
    y: torch.Tensor

# Create batch of tensors
x = torch.rand(3, 4)
y = torch.rand(3)
data = DataPair(x=x, y=y, shape=(3,), device="cpu")

# Test indexing returns correct shapes
single_item = data[0]
assert single_item.shape == ()
assert single_item.x.shape == (4,)

# Test slicing returns correct shapes
slice_data = data[1:3]
assert slice_data.shape == (2,)
assert slice_data.x.shape == (2, 4)

# You can assign to an indexed TensorDataClass using another instance with matching batch shape
replacement = DataPair(
    x=torch.rand(2, 4),
    y=torch.rand(2),
    shape=(2,),
    device="cpu",
)
data[1:3] = replacement
```
*Example available at: [examples/tensor_dataclass/02_indexing.py](../../examples/tensor_dataclass/02_indexing.py)*


## Shape Operations

Shape operations like `reshape()`, `view()`, `squeeze()`, and `unsqueeze()` apply only to the **batch dimensions**. Event dimensions are automatically preserved.

```python
import torch
from tensorcontainer import TensorDataClass

class Data(TensorDataClass):
    x: torch.Tensor
    y: torch.Tensor

x_tensor = torch.rand(2, 3, 4)
y_tensor = torch.rand(2, 3, 5)

data = Data(
    x=x_tensor,
    y=y_tensor,
    shape=(2, 3),
    device="cpu",
)

# Reshape batch dimensions while preserving event dimensions
reshaped_data = data.reshape(6)

# Verify reshape preserves total elements and event dimensions
assert reshaped_data.shape == (6,)
assert reshaped_data.x.shape == (6, 4)  # Event dimension (4,) preserved
assert reshaped_data.y.shape == (6, 5)  # Event dimension (5,) preserved
```
*Example available at: [examples/tensor_dataclass/03_shape_ops.py](../../examples/tensor_dataclass/03_shape_ops.py)*


## Device Operations

The `.to()` method provides a convenient way to move all tensor fields in a `TensorDataClass` instance to a different device (CPU, GPU, etc.) in a single operation.

```python
import torch
from tensorcontainer import TensorDataClass

class MyData(TensorDataClass):
    x: torch.Tensor
    y: torch.Tensor

# Create instance on CPU
data_cpu = MyData(x=torch.rand(2, 2), y=torch.rand(2, 5), shape=(2,), device="cpu")

# Move to CUDA device
data_cuda = data_cpu.to(device="cuda")

# Verify all tensors moved to CUDA
assert data_cuda.device == torch.device("cuda:0")
assert data_cuda.x.device == torch.device("cuda:0")
```
*Example available at: [examples/tensor_dataclass/05_device.py](../../examples/tensor_dataclass/05_device.py)*


## Stacking and Concatenation

You can combine multiple `TensorDataClass` instances using `torch.stack` and `torch.cat`.

- `torch.stack`: Creates a new batch dimension.
- `torch.cat`: Concatenates along an existing batch dimension.

```python
import torch
from tensorcontainer import TensorDataClass

class DataPoint(TensorDataClass):
    x: torch.Tensor
    y: torch.Tensor

point1 = DataPoint(
    x=torch.rand(3, 4),
    y=torch.rand(3, 5),
    shape=(3,),
    device="cpu",
)

point2 = DataPoint(
    x=torch.rand(3, 4),
    y=torch.rand(3, 5),
    shape=(3,),
    device="cpu",
)

# Stack instances along new leading dimension
stacked = torch.stack([point1, point2], dim=0)

# Verify stacking creates new batch dimension
assert stacked.shape == (2, 3)
assert stacked.x.shape == (2, 3, 4)
```
*Example available at: [examples/tensor_dataclass/04_stack.py](../../examples/tensor_dataclass/04_stack.py)*


## Copy and Clone Operations

- **`.copy()`**: Creates a shallow copy of the instance (new instance, shared tensors).
- **`.clone()`**: Creates a deep copy of the instance with new, independent tensor storage.

```python
import torch
import copy
from tensorcontainer import TensorDataClass

class DataPair(TensorDataClass):
    x: torch.Tensor
    y: torch.Tensor

# Create tensors
original = DataPair(x=torch.rand(3, 3), y=torch.rand(3, 5), shape=(3,), device="cpu")

# Shallow copy shares tensor storage
copied = copy.copy(original)
assert copied.x is original.x  # Same tensor objects

# Deep clone creates independent tensor storage
cloned = original.clone()
assert cloned.x is not original.x  # Different tensor objects
```
*Example available at: [examples/tensor_dataclass/07_copy_clone.py](../../examples/tensor_dataclass/07_copy_clone.py)*


## Gradient Management

The `detach()` method is essential for gradient management in training scenarios. It creates a new `TensorDataClass` instance with the same data but stops gradient flow.

```python
import torch
from tensorcontainer import TensorDataClass

class TrainingBatch(TensorDataClass):
    observations: torch.Tensor
    actions: torch.Tensor


# Create batch with gradient tracking
batch = TrainingBatch(
    observations=torch.randn(4, 10, requires_grad=True),
    actions=torch.randn(4, 3, requires_grad=True),
    shape=(4,),
    device="cpu",
)

# Detach to stop gradient flow
detached_batch = batch.detach()

# Verify gradients are no longer tracked
assert not detached_batch.observations.requires_grad
assert not detached_batch.actions.requires_grad
```
*Example available at: [examples/tensor_dataclass/08_detach_gradients.py](../../examples/tensor_dataclass/08_detach_gradients.py)*


## Nested Structures

You can nest `TensorDataClass` instances or other `TensorContainer` types (like `TensorDict`) within each other. All operations will propagate through the nested structure correctly.

```python
import torch
from tensorcontainer import TensorDataClass

class AgentState(TensorDataClass):
    actor_params: torch.Tensor
    critic_params: torch.Tensor

class FullState(TensorDataClass):
    env_state: torch.Tensor
    agent_state: AgentState # Nested TensorDataClass

# Create with appropriate tensors and shapes
full = FullState(
    env_state=torch.randn(4, 10),
    agent_state=AgentState(
        actor_params=torch.randn(4, 20),
        critic_params=torch.randn(4, 15),
        shape=(4,),
        device="cpu"
    ),
    shape=(4,),
    device="cpu"
)
# Operations on 'full' will propagate to 'agent_state'
full_gpu = full.to("cuda")
```
*Example available at: [examples/tensor_dataclass/09_nested.py](../../examples/tensor_dataclass/09_nested.py)*

---

## Comparison and Decision Guide

### TensorDataClass vs. TensorDict

| Feature          | TensorDataClass                     | TensorDict                        |
|------------------|-------------------------------------|-----------------------------------|
| **Access**       | `obj.field` (attribute)             | `obj["key"]` (dictionary)         |
| **Schema**       | Static, defined at class level      | Dynamic, keys added at runtime    |
| **Type Safety**  | Compile-time (static analysis)      | Runtime                           |
| **IDE Support**  | Full autocomplete & refactoring     | Limited to string keys            |
| **Inheritance**  | Natural OOP inheritance           | Composition-based nesting         |

Choose `TensorDataClass` when you have a well-defined, stable data structure and can benefit from static type checking and IDE support. Choose `TensorDict` for more dynamic scenarios where the contents might change during execution.