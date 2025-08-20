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
print(f"Observations shape: {data.observations.shape}")
print(f"Actions shape: {data.actions.shape}")

# 4. Apply unified operations to all tensors
data_gpu = data.to("cuda")
data_reshaped = data.reshape(1, 2)
data_detached = data.detach()

print(f"Original shape: {data.shape}")
print(f"Reshaped shape: {data_reshaped.shape}")
print(f"Device: {data_gpu.device}")
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
class SimpleData(TensorDataClass):
    observations: torch.Tensor
    actions: torch.Tensor

# ...is automatically transformed to a dataclass, as if you wrote:
#
# @dataclass(eq=False, slots=True)
# class SimpleData:
#     observations: torch.Tensor
#     actions: torch.Tensor
#     shape: torch.Size
#     device: torch.device
```

### Field Definition Patterns

#### Basic Tensor Fields
The most common use case is defining required tensor fields. Annotate each field with `torch.Tensor`.
```python
class SimpleData(TensorDataClass):
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
    x: torch.Tensor

class Child(Base):
    y: torch.Tensor

# Create tensors with shared batch dimension
x = torch.rand(2, 3)
y = torch.rand(2, 5)

# Child inherits all fields from Base
data = Child(x=x, y=y, shape=(2,), device="cpu")

# Verify inheritance works correctly
print(f"Shape: {data.shape}")
print(f"X shape: {data.x.shape}")
print(f"Y shape: {data.y.shape}")
```
Example available at: [`examples/tensor_dataclass/06_inheritance.py`](../../examples/tensor_dataclass/06_inheritance.py)

---

## Construction and Initialization

### Constructor and Validation
To create an instance, you must provide values for all annotated tensor fields, along with the mandatory `shape` and `device` arguments. The [`__post_init__()`](../../src/tensorcontainer/tensor_dataclass.py:241) method validates two principal constraints:
1. **Shape Consistency**: Every tensor's leading dimensions must match the `shape` argument.
2. **Device Consistency**: Every tensor must reside on the specified `device`.

```python
# Valid construction
obs_tensor = torch.rand(2, 3)
act_tensor = torch.rand(2)
data = SimpleData(
    observations=obs_tensor,
    actions=act_tensor,
    shape=(2,),  # Batch dimensions shared by all fields
    device="cpu",
)

# Invalid shape: raises an error because the shape argument must be a prefix of every field's shape
try:
    SimpleData(
        observations=obs_tensor,
        actions=act_tensor,
        shape=(3,),  # This shape (3,) is not a prefix of (2, 3) or (2,)
        device="cpu",
    )
except Exception as e:
    print(e)
```

---

## Core Operations

### Attribute Access and Assignment
Access tensor fields using standard attribute dot-notation (`obj.field`). This provides a type-safe and IDE-friendly way to interact with your data. Assignment is also supported and will be validated at runtime.

```python
# Access
obs = data.observations
actions = data.actions

# Assignment
new_actions = torch.randn(2, 8)
data.actions = new_actions
```

### Indexing and Slicing
Indexing operates exclusively on the batch dimensions, leaving event dimensions untouched. It mirrors `torch.Tensor` indexing and returns a new `TensorDataClass` instance that is a **view** of the original data.
- **Integer Indexing**: `data[0]` - reduces rank.
- **Slice Indexing**: `data[1:3]` - creates a sub-batch.
- **Assignment**: You can assign a `TensorDataClass` instance to a slice of another.

```python
# Create batch of tensors
x = torch.rand(3, 4)
y = torch.rand(3)
data = DataPair(x=x, y=y, shape=(3,), device="cpu")

# Test indexing returns correct shapes
single_item = data[0]
print(f"Single item shape: {single_item.shape}") # ()
print(f"Single item x shape: {single_item.x.shape}") # (4,)
print(f"Single item y shape: {single_item.y.shape}") # ()

# Test slicing returns correct shapes
slice_data = data[1:3]
print(f"Slice shape: {slice_data.shape}") # (2,)
print(f"Slice x shape: {slice_data.x.shape}") # (2, 4)
print(f"Slice y shape: {slice_data.y.shape}") # (2,)

# You can assign to an indexed TensorDataClass using another instance with matching batch shape
replacement = DataPair(
    x=torch.rand(2, 4),
    y=torch.rand(2),
    shape=(2,),
    device="cpu",
)
data[1:3] = replacement
```
Example available at: [`examples/tensor_dataclass/02_indexing.py`](../../examples/tensor_dataclass/02_indexing.py)

### Shape Operations
Shape operations like `reshape()`, `view()`, `squeeze()`, and `unsqueeze()` apply only to the **batch dimensions**. Event dimensions are automatically preserved.

```python
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
print(f"Reshaped batch shape: {reshaped_data.shape}") # (6,)
print(f"Reshaped x shape: {reshaped_data.x.shape}") # (6, 4) - Event dimension (4,) preserved
print(f"Reshaped y shape: {reshaped_data.y.shape}") # (6, 5) - Event dimension (5,) preserved
```
Example available at: [`examples/tensor_dataclass/03_shape_ops.py`](../../examples/tensor_dataclass/03_shape_ops.py)

### Device and Memory Operations
- **`.to(device)`**: Moves all contained tensors to the specified device. See [`TensorContainer.to()`](../../src/tensorcontainer/tensor_container.py:599).
- **`.detach()`**: Creates a new instance with all tensors detached from the computation graph.
- **`.clone()`**: Creates a deep copy of the instance with new, independent tensor storage.
- **`copy.copy()`**: [`__copy__()`](../../src/tensorcontainer/tensor_dataclass.py:258) creates a shallow copy (new instance, shared tensors).

```python
# Device transfer
x = torch.rand(2, 2)
y = torch.rand(2, 5)
data_cpu = MyData(x=x, y=y, shape=(2,), device="cpu")
data_cuda = data_cpu.to(device="cuda")

# Verify all tensors moved to CUDA
print(f"Device: {data_cuda.device}")
print(f"X device: {data_cuda.x.device}")
print(f"Y device: {data_cuda.y.device}")

# Create a clone for independent modification
x = torch.rand(3, 3)
y = torch.rand(3, 5)
original = DataPair(x=x, y=y, shape=(3,), device="cpu")
cloned = original.clone()

# Deep clone creates independent tensor storage
print(f"X is same object: {cloned.x is not original.x}")
print(f"Y is same object: {cloned.y is not original.y}")

# Create a shallow copy
copied = original.copy()
# Shallow copy shares tensor storage
print(f"X is same object: {copied.x is original.x}")
print(f"Y is same object: {copied.y is original.y}")

# Detach gradients
batch = TrainingBatch(
    observations=torch.randn(4, 10, requires_grad=True),
    actions=torch.randn(4, 3, requires_grad=True),
    shape=(4,),
    device="cpu",
)
detached_batch = batch.detach()

# Verify gradients are no longer tracked
print(f"Original requires grad: {batch.observations.requires_grad}")
print(f"Detached requires grad: {detached_batch.observations.requires_grad}")
```
Device example: [`examples/tensor_dataclass/05_device.py`](../../examples/tensor_dataclass/05_device.py)
Copy/Clone example: [`examples/tensor_dataclass/07_copy_clone.py`](../../examples/tensor_dataclass/07_copy_clone.py)
Detach example: [`examples/tensor_dataclass/08_detach_gradients.py`](../../examples/tensor_dataclass/08_detach_gradients.py)


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
print(f"Stacked shape: {stacked.shape}") # (2, 3)
print(f"Stacked x shape: {stacked.x.shape}") # (2, 3, 4)
print(f"Stacked y shape: {stacked.y.shape}") # (2, 3, 5)
```
Example available at: [`examples/tensor_dataclass/04_stack.py`](../../examples/tensor_dataclass/04_stack.py)

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