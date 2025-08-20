# TensorDict Deep Dive

TensorDict is a dictionary-style TensorContainer for managing collections of tensors that share the same leading batch shape, with flexible event dimensions. Use it when your schema changes during development or when you need key-based access.

## Quick start

```python
import torch
from tensorcontainer import TensorDict

# Create a TensorDict with related tensors
batch = TensorDict({
    'observations': torch.randn(32, 64, 64),  # 32 images of 64x64
    'actions': torch.randn(32, 4),            # 32 action vectors
    'rewards': torch.randn(32, 1)             # 32 reward values
}, shape=(32,))  # Batch dimension: 32 samples

# Dictionary-style access
obs = batch['observations']
batch['next_observations'] = torch.randn(32, 64, 64)

# Apply operations to ALL tensors at once (batch dims only)
batch = batch.to('cuda')          # Move all tensors to GPU  -> [`TensorContainer.to()`](src/tensorcontainer/tensor_container.py:599)
batch = batch.reshape(16, 2)      # Reshape batch to 16x2    -> [`TensorContainer.reshape()`](src/tensorcontainer/tensor_container.py:573)
batch = batch.detach()            # Detach from autograd     -> [`TensorContainer.detach()`](src/tensorcontainer/tensor_container.py:617)

print(f"Batch shape: {batch.shape}")
print(f"Keys: {list(batch.keys())}")
```

---

# TensorDict-Specific Features

TensorDict builds on the shared TensorContainer mechanics to provide dynamic, key-based collections, nested conversions, and key flattening tailored for real-world pipelines.

## Dictionary Semantics

TensorDict provides a full dictionary-like interface for managing tensor collections with shared batch dimensions.

### Key-Based Access

- Mapping methods:
  - Keys/Values/Items: [`TensorDict.keys()`](src/tensorcontainer/tensor_dict.py:322), [`TensorDict.values()`](src/tensorcontainer/tensor_dict.py:325), [`TensorDict.items()`](src/tensorcontainer/tensor_dict.py:328)
  - Size/Len: `len(td)` delegates to [`TensorDict.__len__()`](src/tensorcontainer/tensor_dict.py:316)
  - Membership: [`TensorDict.__contains__()`](src/tensorcontainer/tensor_dict.py:319)
- Get/Set by string key:
  - Get: [`TensorDict.__getitem__()`](src/tensorcontainer/tensor_dict.py:275) routes string keys to the underlying mapping
  - Set: [`TensorDict.__setitem__()`](src/tensorcontainer/tensor_dict.py:291) validates shape/device and auto-wraps nested dicts
- Bulk update:
  - [`TensorDict.update()`](src/tensorcontainer/tensor_dict.py:331) applies validations and nested wrapping

```python
td = TensorDict({
    'states': torch.randn(32, 128),
    'actions': torch.randn(32, 4),
}, shape=(32,))

states = td['states']              # key-based
td['values'] = torch.randn(32, 1)  # add new field
del td['actions']                  # remove field

# Bulk update
td.update({'logp': torch.randn(32, 1), 'entropy': torch.randn(32, 1)})
```

## Construction and Initialization

TensorDict offers flexible initialization options with automatic validation and nested structure support.

### Constructor Options

Constructor: [`TensorDict.__init__()`](src/tensorcontainer/tensor_dict.py:95)

Inputs:
- `data`: mapping from `str -> Tensor | TensorDict | dict`
- `shape`: batch shape prefix shared by all tensors
- `device`: optional device constraint; if `None`, mixed-device leaves are allowed (compat controlled via [`TensorContainer._is_device_compatible()`](src/tensorcontainer/tensor_container.py:321))

Nested `dict` values are normalized into TensorDict recursively via: [`TensorDict.data_from_dict()`](src/tensorcontainer/tensor_dict.py:116)

```python
# Basic
td = TensorDict({
    'obs': torch.randn(32, 128),
    'act': torch.randint(0, 4, (32,))
}, shape=(32,), device='cpu')

# From existing dict (auto-nested)
raw = {'agent': {'pos': torch.randn(16, 3), 'vel': torch.randn(16, 3)}}
agent_td = TensorDict(raw, shape=(16,), device='cpu')
assert isinstance(agent_td['agent'], TensorDict)

# Empty + dynamic fill
empty = TensorDict({}, shape=(8,), device='cuda')
empty['x'] = torch.randn(8, 2, device='cuda')
```

Shape semantics (enforced centrally by TensorContainer):
- Every leaf tensor must have at least `len(shape)` dims and its first `len(shape)` dims equal to `shape`. See: [`TensorContainer._validate_shape()`](src/tensorcontainer/tensor_container.py:328)
- Scalar and zero-sized support:
  - Scalar leaves are valid with `shape=()` (see tests: tests/tensor_dict/test_init.py)
  - Zero batch size accepted (see tests: tests/tensor_dict/test_shape.py)

Invalid examples (see tests):
- Wrong batch prefix -> error: tests/tensor_dict/test_init.py, tests/tensor_dict/test_shape.py
- Leaf has too few dims for specified `shape` -> error: tests/tensor_dict/test_shape.py

## Nested Structures

TensorDict automatically handles nested dictionaries, converting them to nested TensorDict instances for hierarchical data organization.

### Automatic Nesting

Nested Python dicts become nested TensorDicts automatically. This is consistently enforced on:
- Construction: [`TensorDict.data_from_dict()`](src/tensorcontainer/tensor_dict.py:116)
- Assignment of a dict to a key: [`TensorDict.__setitem__()`](src/tensorcontainer/tensor_dict.py:291) auto-wraps the dict using the parent's shape/device.

```python
nested = TensorDict({
    'agent': {
        'position': torch.randn(32, 3),
        'velocity': torch.randn(32, 3),
    },
    'env': {
        'goal': torch.randn(32, 2),
    },
}, shape=(32,), device='cpu')

# Access
pos = nested['agent']['position']
goal = nested['env']['goal']

# Dynamically extend nested paths
nested['agent']['inventory'] = {
    'items': torch.randint(0, 10, (32, 5)),
    'weights': torch.randn(32, 5),
}
```

### Operation Propagation

Operation propagation to nested children is automatic because all container ops are defined in the base class and traverse leaves via PyTree:
- Base map used by all ops (with key-path error decoration): [`TensorContainer._tree_map()`](src/tensorcontainer/tensor_container.py:300)

```python
reshaped = nested.reshape(8, 4)
on_cuda  = nested.to('cuda')
indexed  = nested[:, 0]
```

## Key Flattening

TensorDict provides powerful key flattening capabilities to convert nested structures into flat namespaces for easier processing.

### Flatten Keys

Flatten nested keys into a single-level namespace using dot or a custom separator:
- API: [`TensorDict.flatten_keys()`](src/tensorcontainer/tensor_dict.py:349)

```python
flat = nested.flatten_keys()
print(list(flat.keys()))  # ['agent.position', 'agent.velocity', 'env.goal', ...]

flat_slash = nested.flatten_keys(separator='/')  # ['agent/position', ...]
flat_under = nested.flatten_keys(separator='_')  # ['agent_position', ...]
```

Properties:
- Iterative traversal (non-recursive) prevents recursion depth/cycles.
- No copies — values in the flattened view reference the same tensors (see tests: tests/tensor_dict/test_flatten_keys.py).
- Works with `torch.compile` (see tests: tests/tensor_dict/test_flatten_keys.py).

## Advanced Indexing and Assignment

TensorDict provides sophisticated indexing capabilities that distinguish between key-based access and batch-based operations.

### Key vs Batch Indexing

String-key indexing delegates to mapping access, while non-string indexing defers to base batch indexing.


```python
# Key vs batch indexing
x = td['states']        # key (mapping)
sub = td[:8]            # batch slice (base container indexing)

# Boolean masking over batch dims
mask = torch.randint(0, 2, (32,), dtype=torch.bool)
active = td[mask]       # every leaf masked consistently

# Multi-d indexing
series = TensorDict({
    's': torch.randn(8, 10, 64),
    'a': torch.randn(8, 10, 4),
}, shape=(8, 10))
step5 = series[:, 5]         # shape becomes (8,)
subset = series[:4, :5]      # (4, 5)

# Ellipsis
complex_td = TensorDict({
    'multi': torch.randn(4, 8, 6, 128),
    'simple': torch.randn(4, 8, 6, 1),
}, shape=(4, 8, 6))
tail = complex_td[..., :3]   # slice last batch dim
```

### In-place Assignment

In-place assignment paths:
- Assign TensorDict into slice/mask: [`TensorContainer.__setitem__()`](src/tensorcontainer/tensor_container.py:511)

```python
template = TensorDict({
    'states': torch.zeros(2, 64),
    'actions': torch.zeros(2, 4),
}, shape=(2,))
td[:2] = template

mask = torch.tensor([True, False, True, False])
td[mask] = template  # broadcast rules apply at leaf level (PyTorch semantics)
```

Errors and constraints validated by tests:
- Mask shape mismatch raises IndexError (propagated from PyTorch leaves): tests/tensor_dict/test_mask_select.py

## Error Handling and Diagnostics

TensorDict provides comprehensive error handling with detailed diagnostics to help identify and resolve issues quickly.

### Validation with Context

TensorDict leans on base validation with detailed path reporting:
- Validation loop adds key-path context: [`TensorContainer._validate()`](src/tensorcontainer/tensor_container.py:340)
- Errors include "Validation error at key ..." with underlying cause.
- Any runtime thrown during `tree_map` operations is decorated with a readable key path: [`TensorContainer._tree_map()`](src/tensorcontainer/tensor_container.py:300)

Typical issues:
- Shape mismatch across batch dims
- Device mismatch when `device` is set on the container
- Too many indices or invalid ellipsis usage (normalized and checked in base)

Examples (from tests):
- Constructor raises on incompatible leaves: tests/tensor_dict/test_shape.py
- Nested mapping mismatch errors: tests/tensor_dict/test_shape.py


## Practical Examples

### Time-Series Batch with Nested Structure

```python
episodes = 8
timesteps = 10

rollout = TensorDict({
    'agent': {
        'obs': torch.randn(episodes, timesteps, 128),
        'act': torch.randn(episodes, timesteps, 4),
    },
    'env': {
        'rew': torch.randn(episodes, timesteps, 1),
        'done': torch.randint(0, 2, (episodes, timesteps, 1)),
    }
}, shape=(episodes, timesteps))

# Index timestep
t5 = rollout[:, 5]        # shape: (8,)
# Flatten for pipeline
flat = rollout.flatten_keys()
# Reshape to a single batch for training
flat = flat.reshape(episodes * timesteps)
# Move to GPU
flat = flat.to('cuda')
```

### Boolean Masking and Assignment

```python
td = TensorDict({
    'state': torch.randn(4, 16),
    'value': torch.randn(4, 1),
}, shape=(4,))

mask = torch.tensor([True, False, True, False])

# Select
sel = td[mask]  # shape: (2,)

# Assign template into masked positions
template = TensorDict({'state': torch.zeros(2, 16), 'value': torch.zeros(2, 1)}, shape=(2,))
td[mask] = template
```

### Stacking and Concatenating Rollouts

```python
rollout1 = TensorDict({'obs': torch.randn(16, 64)}, shape=(16,))
rollout2 = TensorDict({'obs': torch.randn(16, 64)}, shape=(16,))

stacked = torch.stack([rollout1, rollout2])  # (2, 16)
joined  = torch.cat([rollout1, rollout2])    # (32,)
```


---
