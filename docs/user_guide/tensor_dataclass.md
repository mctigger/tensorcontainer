# TensorDataClass Deep Dive

TensorDataClass is a dataclass-based TensorContainer that combines Python's dataclass system with tensor operations, providing type-safe, attribute-based access to collections of tensors with shared batch dimensions. Use it when you need static schemas, IDE support, and compile-time type checking.

## Quick Start

Get started with `TensorDataClass` by defining a class with type-annotated tensor fields. The container automatically generates a constructor and applies tensor operations to all fields at once.

```python
import torch
from tensorcontainer import TensorDataClass

# 1. Define a schema using class annotations
class RLBatch(TensorDataClass):
    observations: torch.Tensor
    actions: torch.Tensor

# 2. Instantiate with tensors and a shared batch shape
batch = RLBatch(
    observations=torch.randn(32, 64),  # (B, F)
    actions=torch.randint(0, 4, (32,)), # (B,)
    shape=(32,),                       # Shared batch dimension
    device="cpu",
)

# 3. Access fields with type-safe attribute access
obs = batch.observations
assert obs.shape == (32, 64)

# 4. Apply unified operations to all tensors
batch_gpu = batch.to("cuda")      # -> [TensorContainer.to()](src/tensorcontainer/tensor_container.py:599)
batch_reshaped = batch.reshape(4, 8) # -> [TensorContainer.reshape()](src/tensorcontainer/tensor_container.py:573)
batch_detached = batch.detach()    # -> [TensorContainer.detach()](src/tensorcontainer/tensor_container.py:617)

print(f"Batch shape: {batch.shape}")
print(f"Reshaped shape: {batch_reshaped.shape}")
print(f"Device: {batch_gpu.device}")
```

---

## TensorDataClass-Specific Features

### Automatic Dataclass Generation
When you inherit from `TensorDataClass`, it is automatically transformed into a Python `dataclass` with optimized settings for tensor operations. This behavior is enabled by the [`TensorDataClass.__init_subclass__()`](../../src/tensorcontainer/tensor_dataclass.py:191) method, which provides:
- A generated `__init__` based on your type annotations.
- IDE support for autocomplete and type-checking, thanks to `@dataclass_transform`.
- Memory efficiency via `slots=True` by default.
- Disabled `eq=True` to avoid ambiguity with tensor equality.

```python
# This class definition...
class MyData(TensorDataClass):
    features: torch.Tensor
    labels: torch.Tensor

# ...is automatically transformed to a dataclass, as if you wrote:
#
# @dataclass(eq=False, slots=True)
# class MyData:
#     features: torch.Tensor
#     labels: torch.Tensor
#     shape: torch.Size
#     device: torch.device
```

### Field Definition Patterns

#### Basic Tensor Fields
The most common use case is defining required tensor fields. Annotate each field with `torch.Tensor`.
```python
class BasicData(TensorDataClass):
    # observations and actions are required tensor fields
    observations: torch.Tensor
    actions: torch.Tensor
```

#### Optional Fields and Defaults
You can define optional fields and non-tensor metadata using `Optional` and `dataclasses.field`.
- **Optional Tensors**: Use `Optional[torch.Tensor] = None`.
- **Non-Tensor Metadata**: Use `field(default_factory=...)` to provide default values for mutable types like `list` or `dict`.
- **Default Tensors**: Use `field(default_factory=...)` to generate a default tensor value.

```python
from dataclasses import field
from typing import Optional, List, Dict, Any

class FlexibleData(TensorDataClass):
    # Required tensor
    features: torch.Tensor
    
    # Optional tensor, defaults to None
    labels: Optional[torch.Tensor] = None
    
    # Non-tensor metadata
    episode_ids: List[int] = field(default_factory=list)
    
    # Default tensor value
    rewards: torch.Tensor = field(default_factory=lambda: torch.zeros(1))
```

### Inheritance and Composition
`TensorDataClass` supports standard Python inheritance, allowing you to build complex data schemas by extending base classes. Fields from all parent classes are automatically merged.

```python
class Base(TensorDataClass):
    observations: torch.Tensor

# Child inherits 'observations' and adds 'actions'
class Child(Base):
    actions: torch.Tensor

# GrandChild inherits 'observations', 'actions', and adds 'rewards'
class GrandChild(Child):
    rewards: torch.Tensor

# Instance of GrandChild has all three fields
data = GrandChild(
    observations=torch.randn(4, 10),
    actions=torch.randn(4, 1),
    rewards=torch.randn(4, 1),
    shape=(4,),
    device="cpu",
)
```
Example available at: [`examples/tensor_dataclass/06_nested_inheritance.py`](../../examples/tensor_dataclass/06_nested_inheritance.py)

---

## Construction and Initialization

### Constructor and Validation
To create an instance, you must provide values for all annotated tensor fields, along with the mandatory `shape` and `device` arguments. The [`__post_init__()`](../../src/tensorcontainer/tensor_dataclass.py:241) method validates two principal constraints:
1. **Shape Consistency**: Every tensor's leading dimensions must match the `shape` argument.
2. **Device Consistency**: Every tensor must reside on the specified `device`.

```python
# Valid construction
data = RLBatch(
    observations=torch.randn(32, 64, device="cpu"),
    actions=torch.randn(32, 4, device="cpu"),
    shape=(32,),
    device="cpu",
)

# Invalid shape: raises an error because actions.shape[0] is not 32
try:
    RLBatch(
        observations=torch.randn(32, 64),
        actions=torch.randn(16, 4), # Mismatched batch dim
        shape=(32,),
        device="cpu",
    )
except RuntimeError as e:
    print(e)
```

---

## Core Operations

### Attribute Access and Assignment
Access tensor fields using standard attribute dot-notation (`obj.field`). This provides a type-safe and IDE-friendly way to interact with your data. Assignment is also supported and will be validated at runtime.

```python
# Access
obs = batch.observations

# Assignment
new_actions = torch.randn(32, 8)
batch.actions = new_actions
```

### Indexing and Slicing
Indexing operates exclusively on the batch dimensions, leaving event dimensions untouched. It mirrors `torch.Tensor` indexing and returns a new `TensorDataClass` instance that is a **view** of the original data.
- **Integer Indexing**: `batch[0]` - reduces rank.
- **Slice Indexing**: `batch[:16]` - creates a sub-batch.
- **Boolean Masking**: `batch[mask]` - filters the batch.

```python
# Create data with batch shape (32,)
data = RLBatch(
    observations=torch.randn(32, 64),
    actions=torch.randn(32, 4),
    shape=(32,),
    device="cpu"
)

# Integer index: returns a new instance with no batch dims
sample = data[0] 
print(f"Sample shape: {sample.shape}") # ()

# Slice index: returns a new instance with shape (16,)
sub_batch = data[:16]
print(f"Sub-batch shape: {sub_batch.shape}") # (16,)

# Changes to the view are reflected in the original
sub_batch.observations[0] = 0.0
assert data.observations[0].sum() == 0.0
```
Example available at: [`examples/tensor_dataclass/02_indexing.py`](../../examples/tensor_dataclass/02_indexing.py)

### Shape Operations
Shape operations like `reshape()`, `view()`, `squeeze()`, and `unsqueeze()` apply only to the **batch dimensions**. Event dimensions are automatically preserved.

```python
data = RLBatch(
    observations=torch.randn(32, 64), # event shape is (64,)
    actions=torch.randn(32, 4),    # event shape is (4,)
    shape=(32,),
    device="cpu"
)

# Reshape batch from (32,) to (4, 8)
reshaped = data.reshape(4, 8)
print(f"Reshaped batch shape: {reshaped.shape}") # (4, 8)

# Event dimensions are preserved
print(f"Reshaped observations shape: {reshaped.observations.shape}") # (4, 8, 64)
print(f"Reshaped actions shape: {reshaped.actions.shape}")       # (4, 8, 4)
```
Example available at: [`examples/tensor_dataclass/03_shape_ops.py`](../../examples/tensor_dataclass/03_shape_ops.py)

### Device and Memory Operations
- **`.to(device)`**: Moves all contained tensors to the specified device. See [`TensorContainer.to()`](../../src/tensorcontainer/tensor_container.py:599).
- **`.detach()`**: Creates a new instance with all tensors detached from the computation graph.
- **`.clone()`**: Creates a deep copy of the instance with new, independent tensor storage.
- **`copy.copy()`**: [`__copy__()`](../../src/tensorcontainer/tensor_dataclass.py:258) creates a shallow copy (new instance, shared tensors).

```python
# Device transfer
data_gpu = data.to("cuda")

# Create a clone for independent modification
cloned_data = data_gpu.clone()
cloned_data.observations[0] = 42.0 # original data_gpu is not modified

# Create a shallow copy
shallow_copy = copy.copy(data) 
shallow_copy.observations[0] = 99.0 # original data is modified
```
Device example: [`examples/tensor_dataclass/05_device.py`](../../examples/tensor_dataclass/05_device.py)
Copy/Clone example: [`examples/tensor_dataclass/07_copy_clone.py`](../../examples/tensor_dataclass/07_copy_clone.py)


---

## Advanced Features

### PyTree Integration
`TensorDataClass` is a registered PyTree, which means it works seamlessly with `torch.stack`, `torch.cat`, and other functions that operate on nested structures.
- **Tensor Fields** are treated as leaves of the tree.
- **Non-Tensor Metadata** is preserved.
This integration is handled by the [`_pytree_flatten()`](../../src/tensorcontainer/tensor_annotated.py:82) and [`_pytree_unflatten()`](../../src/tensorcontainer/tensor_annotated.py:104) methods.

### Stacking and Concatenation
You can combine multiple `TensorDataClass` instances using `torch.stack` and `torch.cat`.
- `torch.stack`: Creates a new batch dimension.
- `torch.cat`: Concatenates along an existing batch dimension.

```python
instances = [
    RLBatch(obs, act, shape=(32,), device="cpu") for obs, act in ...
]

# Stack 10 instances to create a new batch dim: shape becomes (10, 32)
stacked_batch = torch.stack(instances, dim=0)

# Concatenate 10 instances along the existing batch dim: shape remains (320,)
catted_batch = torch.cat(instances, dim=0)
```
Example available at: [`examples/tensor_dataclass/04_stack_cat.py`](../../examples/tensor_dataclass/04_stack_cat.py)

### Nested Structures
You can nest `TensorDataClass` instances or other `TensorContainer` types (like `TensorDict`) within each other. All operations will propagate through the nested structure correctly.

```python
class AgentState(TensorDataClass):
    actor_params: torch.Tensor
    critic_params: torch.Tensor

class FullState(TensorDataClass):
    env_state: torch.Tensor
    agent_state: AgentState # Nested TensorDataClass

# Operations on 'full' will propagate to 'agent_state'
full = FullState(...)
full_gpu = full.to("cuda") 
```

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
| **Use Case**     | Stable, production-ready schemas    | Exploratory, dynamic structures   |

Choose `TensorDataClass` when you have a well-defined, stable data structure and can benefit from static type checking and IDE support. Choose `TensorDict` for more dynamic scenarios where the contents might change during execution.