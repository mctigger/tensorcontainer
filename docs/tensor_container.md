# TensorContainer Design Decisions

## Overview

[`TensorContainer`](src/tensorcontainer/tensor_container.py) is the foundational base class for organizing tensors in PyTorch-based machine learning workflows. This document explains the key design decisions that shaped its architecture and the rationale behind each choice.

## Design Decision 1: Batch vs Event Dimension Separation

### The Problem
Machine learning workflows often need to process collections of heterogeneous tensors together while maintaining consistent batching semantics. Traditional approaches either force all tensors to have identical shapes or provide no guarantees about batch consistency.

### The Solution
TensorContainer enforces a strict separation between **batch dimensions** (leading, shared across all tensors) and **event dimensions** (trailing, variable per tensor).

```python
# Container with batch shape (4, 3) - 4 samples, 3 time steps
container = MyContainer(shape=(4, 3))

# All tensors must share batch dimensions but can vary in event dimensions
observations = torch.randn(4, 3, 128)    # Event dims: (128,)
actions = torch.randn(4, 3, 6)           # Event dims: (6,)
rewards = torch.randn(4, 3)              # Event dims: ()
```

### Why This Works
- **Batched Processing**: Enables consistent batched operations across heterogeneous tensor collections without shape conflicts
- **Shape Safety**: Prevents accidental operations that would break batch consistency, catching errors at construction time rather than during computation
- **ML Workflow Alignment**: Matches common machine learning patterns where batch dimensions represent samples/time while event dimensions represent features/actions

This design decision is enforced through [`_is_shape_compatible()`](src/tensorcontainer/tensor_container.py), which validates that tensors have at least `len(shape)` dimensions and that their leading dimensions exactly match the container's batch shape.

## Design Decision 2: PyTree Integration with `__torch_function__`

### The Problem
PyTorch's ecosystem includes many functions that operate on tensors (like `torch.stack`, `torch.cat`) but don't natively understand custom container types. Without integration, these functions would either fail or extract tensors in unpredictable ways.

### The Solution
TensorContainer uses PyTorch's [`__torch_function__`](src/tensorcontainer/tensor_container.py) protocol combined with the [`@implements`](src/tensorcontainer/tensor_container.py) decorator to intercept and customize torch operations.

```python
@implements(torch.stack)
def stack(tensors, dim=0, *, out=None):
    # Custom implementation that understands TensorContainer structure
    return _stack_impl(tensors, dim, out)
```

### Why This Works
- **Seamless PyTorch Integration**: Functions like `torch.stack(containers)` work naturally without requiring users to learn new APIs
- **Compile Compatibility**: Enables `torch.compile` with `fullgraph=True` by ensuring all operations go through PyTorch's dispatch system
- **Automatic Registration**: The `PytreeRegistered` mixin automates registration with PyTorch's PyTree system, ensuring containers work with `torch.utils._pytree` functions

The [`__torch_function__`](src/tensorcontainer/tensor_container.py) method intercepts torch operations and delegates to registered implementations, maintaining the container abstraction while leveraging PyTorch's optimized kernels.

## Design Decision 3: Performance Optimization with `unsafe_construction`

### The Problem
In performance-critical scenarios like training loops, the overhead of validating tensor shapes and devices during container construction can become significant. However, validation is crucial for catching errors during development.

### The Solution
TensorContainer provides an [`unsafe_construction()`](src/tensorcontainer/tensor_container.py) context manager that uses [`threading.local()`](src/tensorcontainer/tensor_container.py) storage to temporarily disable validation.

```python
# Performance-critical path
with TensorContainer.unsafe_construction():
    containers = []
    for batch_data in data_loader:
        # Validation skipped for performance
        container = MyContainer(batch_data, shape=(32, 10), device="cuda")
        containers.append(container)
```

### Why This Works
- **Performance Critical Paths**: Allows skipping validation overhead in tight loops where shapes and devices are guaranteed to be correct
- **Thread Safety**: [`threading.local()`](src/tensorcontainer/tensor_container.py) ensures the validation state is isolated per thread, preventing race conditions in multi-threaded environments
- **Context Isolation**: The context manager ensures validation is only disabled within a specific scope, automatically restoring normal validation behavior

The implementation stores the unsafe state in thread-local storage, which is checked by the [`_validate()`](src/tensorcontainer/tensor_container.py) method before performing expensive compatibility checks.

## Design Decision 4: Batch-Only Operations

### The Problem
Allowing arbitrary shape transformations on containers could break the fundamental batch/event dimension contract. Users might accidentally reshape event dimensions, corrupting the relationships between tensors.

### The Solution
Methods like [`view()`](src/tensorcontainer/tensor_container.py), [`reshape()`](src/tensorcontainer/tensor_container.py), [`permute()`](src/tensorcontainer/tensor_container.py), and indexing operations only affect batch dimensions, leaving event dimensions untouched.

```python
container = MyContainer(shape=(4, 3))  # Batch: (4, 3)
# Contains tensors with shapes (4, 3, 128) and (4, 3, 6)

reshaped = container.reshape(12)       # Batch becomes (12,)
# Tensors now have shapes (12, 128) and (12, 6) - events preserved

indexed = container[0]                 # Batch becomes (3,)
# Tensors now have shapes (3, 128) and (3, 6) - events preserved
```

### Why This Works
- **Consistency Guarantee**: Ensures all tensors in a container maintain the same batch structure after operations
- **Predictable Behavior**: Users can rely on event dimensions being preserved, making code more predictable and less error-prone
- **Error Prevention**: Prevents accidental corruption of tensor relationships that could lead to subtle bugs in downstream computations

The [`__getitem__()`](src/tensorcontainer/tensor_container.py) method uses [`transform_ellipsis_index()`](src/tensorcontainer/tensor_container.py) to ensure indexing operations only affect batch dimensions.

## Indexing and Assignment

This section documents the base container's indexing and in-place assignment semantics and aligns with the implementations of [`__getitem__`](src/tensorcontainer/tensor_container.py) and [`__setitem__`](src/tensorcontainer/tensor_container.py).

- Same-subclass, exact structure: Slice assignment requires the right-hand side (RHS) to be an instance of the same subclass with exactly matching structure (e.g., identical keys/fields). Assignment is performed leafwise.
- Broadcasting scope: For each corresponding leaf, the RHS must be broadcastable to the addressed batch slice per PyTorch rules. Event dimensions are preserved; they are not consumed or reshaped by assignment.
- No scalar/tensor RHS in base: Scalar values or raw torch.Tensors on the RHS are not supported by the base class. Such semantics are subclass-specific (e.g., TensorDict) and out of scope here.
- Ellipsis handling: Ellipses in indices are normalized via [`transform_ellipsis_index()`](src/tensorcontainer/tensor_container.py), so assignment supports the same ellipsis behavior as PyTorch.

Example:

```python
# Batch shape (4, 3)
container = MyContainer(shape=(4, 3))
# Conceptual leaves in `container` (not shown):
#   obs:    torch.Size([4, 3, 128])
#   action: torch.Size([4, 3, 6])

# Assign to a batch slice with a container of matching structure
rhs = MyContainer(shape=(4,))
# Conceptual leaves in `rhs`:
#   obs:    torch.Size([4, 128])   # broadcastable to container[:, 0].obs
#   action: torch.Size([4, 6])     # broadcastable to container[:, 0].action

container[:, 0] = rhs  # Leafwise assignment; event dims preserved

# Boolean mask assignment
mask = torch.tensor([True, False, True, False])
rhs_mask = MyContainer(shape=(2,))
# Conceptual leaves in `rhs_mask`:
#   obs:    torch.Size([2, 128])
#   action: torch.Size([2, 6])
container[mask] = rhs_mask
```

## Design Decision 5: Path-Based Error Reporting

### The Problem
When validation fails in nested container structures, generic error messages like "tensor shape mismatch" provide insufficient information to locate the problematic tensor. Debugging becomes time-consuming and frustrating.

### The Solution
TensorContainer uses [`_format_path()`](src/tensorcontainer/tensor_container.py) with PyTree's KeyPath system to provide detailed error reporting that pinpoints the exact location of validation failures.

```python
# Error message example:
# "Shape mismatch at path 'data.observations': expected (4, 3, ...), got (4, 2, 128)"
```

### Why This Works
- **Debugging Efficiency**: In nested structures, developers can immediately identify which specific tensor failed validation without manual inspection
- **Leverages PyTree**: Uses PyTree's built-in path tracking system, which already understands the container's structure
- **Contextual Information**: Error messages include both the expected and actual values along with the precise location

The [`_tree_map()`](src/tensorcontainer/tensor_container.py) method captures KeyPath information during tree traversal and passes it to [`_format_path()`](src/tensorcontainer/tensor_container.py) when validation errors occur.

## Design Decision 6: `torch.compile` and Static Shape Design

### The Problem
Supporting fully dynamic shapes in `torch.compile` with `fullgraph=True` is challenging because operations like `torch.Size()` are not traceable by Dynamo. This leads to graph breaks and recompilations, negating the performance benefits of ahead-of-time compilation.

### The Solution
The design of `TensorContainer` prioritizes `fullgraph=True` compatibility by treating the batch shape as static within a compiled graph.

### Why This Works
- **`fullgraph=True`**: When a function using `TensorContainer` is compiled with `fullgraph=True`, the graph is specialized for the specific batch shape it was first compiled with. Any change to this shape will trigger a tracing error, enforcing static shape behavior. This is a deliberate trade-off for maximum performance in scenarios where input shapes are consistent.
- **`fullgraph=False`**: For use cases with a small, finite number of different shapes, `fullgraph=False` can be used. In this mode, `torch.compile` generates a new specialized kernel for each unique shape it encounters. While this introduces some overhead from recompilation, it remains efficient if the number of shape variations is limited.

**A Note on Compile-Friendly Shape Conversion**:

An alternative to using `torch.Size()` directly is to convert a shape-like input into a `torch.Size` object using a `torch.compile`-friendly method. The following function demonstrates this technique:

```python
def _to_torch_size(shape: ShapeLike) -> torch.Size:
    """Convert a shape-like input to torch.Size in a torch.compile-friendly way."""
    # The "meta" device ensures no actual memory allocation occurs,
    # making this very fast while remaining compatible with torch.compile.
    return torch.empty(shape, device="meta").shape
```

This approach avoids the direct use of `torch.Size()` which can cause issues with Dynamo tracing. However, this was not chosen as the primary mechanism in `TensorContainer`'s constructor because it adds a `torch.empty` operation to the computation graph. While this is a valid technique, the design avoids adding this operation to the graph during initialization to keep the compiled graph as clean as possible, with the trade-off that the batch shape is treated as static when `fullgraph=True`.

## Design Decision 7: Device Management and Compatibility

### The Problem
Machine learning workflows often involve moving tensors between different devices (CPU, GPU, different GPU indices). Container classes need to ensure device consistency while providing flexible compatibility rules that don't overly restrict usage patterns.

### The Solution
TensorContainer implements flexible device compatibility through the [`resolve_device()`](src/tensorcontainer/utils.py) function and [`_is_device_compatible()`](src/tensorcontainer/tensor_container.py) method.

```python
# Device resolution at construction
container = MyContainer(shape=(4, 3), device="cuda")  # Resolves to current CUDA device (e.g., "cuda:0"), leaves must match exactly
container = MyContainer(shape=(4, 3), device=None)    # Allows mixed-device leaves

# Device resolution
device = resolve_device("cuda")  # Resolves to current CUDA device (e.g., "cuda:0")
```

### Why This Works
- **Construction-Time Resolution**: Device strings like "cuda" are resolved once at construction to a concrete device (e.g., "cuda:0") via [`resolve_device()`](src/tensorcontainer/utils.py)
- **Strict Equality Enforcement**: With a non-None container device, leaf devices must match exactly according to [`_is_device_compatible()`](src/tensorcontainer/tensor_container.py)
- **Mixed-Device Support**: Setting device=None allows mixed-device containers when needed
- **Dynamic Resolution**: The [`resolve_device()`](src/tensorcontainer/utils.py) function automatically resolves device strings to specific indices based on the current backend state
- **Backend Agnostic**: Works with any PyTorch backend that follows the torch.cuda pattern (cuda, xpu, etc.)

The device compatibility system resolves device strings at construction and enforces strict equality for all leaves unless device=None is specified.

## Type System

TensorContainer uses several type aliases to provide clear interfaces and maintain compatibility with PyTorch's internal typing:

### Core Type Aliases
- **[`ShapeLike`](src/tensorcontainer/types.py)**: `Union[torch.Size, list[int], tuple[int, ...]]` - Mirrors PyTorch's internal shape typing for flexible shape specification
- **[`DeviceLike`](src/tensorcontainer/types.py)**: `Union[str, torch.device, int]` - Mirrors PyTorch's device typing for flexible device specification
- **[`TCCompatible`](src/tensorcontainer/tensor_container.py)**: `Union[torch.Tensor, TensorContainer]` - Used for operations that can work with both tensors and containers

We have [`tests`](tests/test_types.py) that detect if the types included in torch deviate from our types.

## Subclassing `TensorContainer`

Creating a custom subclass of `TensorContainer` allows developers to build specialized, tensor-aware data structures that seamlessly integrate with PyTorch's ecosystem. To ensure compatibility and proper functionality, subclasses are expected to adhere to the following implementation contract.

### Required Inheritance Pattern

Subclasses must inherit from both `TensorContainer` and [`PytreeRegistered`](src/tensorcontainer/utils.py):

```python
from tensorcontainer import TensorContainer
from tensorcontainer.utils import PytreeRegistered

class MyContainer(TensorContainer, PytreeRegistered):
    def __init__(self, data, shape, device=None):
        super().__init__(shape, device)
        self.data = data
```

The [`PytreeRegistered`](src/tensorcontainer/utils.py) mixin automatically registers the subclass with PyTorch's PyTree system when the class is defined, enabling seamless integration with `torch.utils._pytree` functions.

### PyTree Integration (Required)

To enable PyTree compatibility using PyTorch's native `torch.utils._pytree` module, which is essential for `torch.compile` and functional programming patterns, you must implement three abstract methods:

- **`_pytree_flatten() -> tuple[list[torch.Tensor], Any]`**: Flattens the container into a list of tensor "leaves" and a "context" tuple. The context should contain all non-tensor data and metadata required to reconstruct the container.
- **`_pytree_unflatten(leaves: list[torch.Tensor], context: Any) -> TensorContainer`**: Reconstructs the container from the tensor leaves and the context provided by `_pytree_flatten`.
- **`_pytree_flatten_with_keys_fn() -> tuple[list[tuple[str, torch.Tensor]], Any]`**: Similar to `_pytree_flatten`, but it provides descriptive key paths for each leaf. This is crucial for clear error reporting in nested structures.

### Implementation Example

```python
class SimpleContainer(TensorContainer, PytreeRegistered):
    def __init__(self, data: dict[str, torch.Tensor], shape, device=None):
        super().__init__(shape, device)
        self.data = data
    
    def _pytree_flatten(self):
        keys = sorted(self.data.keys())
        values = [self.data[k] for k in keys]
        context = (keys, self.shape, self.device)
        return values, context
    
    @classmethod
    def _pytree_unflatten(cls, leaves, context):
        keys, shape, device = context
        data = dict(zip(keys, leaves))
        return cls(data, shape, device)
    
    def _pytree_flatten_with_keys_fn(self):
        keys = sorted(self.data.keys())
        key_values = [(k, self.data[k]) for k in keys]
        context = (keys, self.shape, self.device)
        return key_values, context
```

## Trade-offs and Limitations

### Benefits
- **Type Safety**: Batch/event separation prevents many common shape-related bugs
- **Performance**: `unsafe_construction` allows optimization in critical paths
- **Integration**: Seamless PyTorch ecosystem compatibility
- **Debugging**: Excellent error reporting for complex nested structures

### Limitations
- **Rigidity**: Uniform batch dimensions requirement prevents some use cases (no ragged batching)
- **Overhead**: PyTree operations and validation add computational cost
- **Complexity**: Implementation complexity is higher than simple tensor collections

