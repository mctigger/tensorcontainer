# TensorDict Deep Dive

TensorDict is a dictionary-style TensorContainer for managing collections of tensors that share the same leading batch shape, with flexible event dimensions. Use it when your schema changes during development or when you need key-based access and nested structures.

- Core implementation: [`TensorDict`](src/tensorcontainer/tensor_dict.py:57)
- Base mechanics shared by all containers: [`TensorContainer`](src/tensorcontainer/tensor_container.py:36)
- PyTree registration mixin: [`PytreeRegistered`](src/tensorcontainer/utils.py:13)
- Shape and device typing: [`ShapeLike`](src/tensorcontainer/types.py:13), [`DeviceLike`](src/tensorcontainer/types.py:16)

Note: This page is split into:
- Part A — Common mechanics shared by all TensorContainers (short).
- Part B — TensorDict-specific concepts, behaviors, and patterns (main content).

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

Relevant APIs:
- String-key get/set: [`TensorDict.__getitem__()`](src/tensorcontainer/tensor_dict.py:275), [`TensorDict.__setitem__()`](src/tensorcontainer/tensor_dict.py:291)
- Batch indexing and slicing (all containers): [`TensorContainer.__getitem__()`](src/tensorcontainer/tensor_container.py:464)

---

# Part A — Common mechanics shared by all TensorContainers

Most of the tensor-like behavior comes from the base class: [`TensorContainer`](src/tensorcontainer/tensor_container.py:36). TensorDict inherits these without modification.

### A1. Batch vs event dimensions

- Batch dims are the leading dimensions defined by `shape` (must match across all tensors).
- Event dims are the trailing dimensions (can differ tensor-by-tensor).

Validation rules (enforced centrally):
- Shape compatibility: [`TensorContainer._is_shape_compatible()`](src/tensorcontainer/tensor_container.py:318)
- Device compatibility: [`TensorContainer._is_device_compatible()`](src/tensorcontainer/tensor_container.py:321)
- Validation on construction/update: [`TensorContainer._validate()`](src/tensorcontainer/tensor_container.py:340)

```python
from tensorcontainer import TensorDict

data = TensorDict({
    'obs': torch.randn(4, 3, 128),  # event: (128,)
    'act': torch.randn(4, 3, 6),    # event: (6,)
    'rew': torch.randn(4, 3)        # event: ()
}, shape=(4, 3))

assert data.shape == torch.Size([4, 3])
```

### A2. Shape ops affect only batch dimensions

Use these on any TensorContainer:
- View/Reshape: [`TensorContainer.view()`](src/tensorcontainer/tensor_container.py:545), [`TensorContainer.reshape()`](src/tensorcontainer/tensor_container.py:573)
- Permute/Transpose/t: [`TensorContainer.permute()`](src/tensorcontainer/tensor_container.py:656), [`TensorContainer.transpose()`](src/tensorcontainer/tensor_container.py:720), [`TensorContainer.t()`](src/tensorcontainer/tensor_container.py:705)
- Unsqueeze/Squeeze: [`TensorContainer.unsqueeze()`](src/tensorcontainer/tensor_container.py:732), [`TensorContainer.squeeze()`](src/tensorcontainer/tensor_container.py:683)
- Expand: [`TensorContainer.expand()`](src/tensorcontainer/tensor_container.py:651)

```python
reshaped = data.reshape(12)     # (4,3) -> (12,)
permuted = data.permute(1, 0)   # (4,3) -> (3,4)
```

### A3. Indexing operates on batch dims

- All PyTorch indexing patterns are supported on batch dims.
- Ellipsis normalization is handled: [`TensorContainer.transform_ellipsis_index()`](src/tensorcontainer/tensor_container.py:371)
- Base indexing implementation: [`TensorContainer.__getitem__()`](src/tensorcontainer/tensor_container.py:464) and slice assignment: [`TensorContainer.__setitem__()`](src/tensorcontainer/tensor_container.py:511)

```python
first_sample  = data[0]     # shape: (3,)
time_slice    = data[:, 0]  # shape: (4,)
mask          = torch.tensor([True, False, True, False])
filtered      = data[mask]  # shape: (2, 3)
```

### A4. Device/dtype and math ops

- Device transfers: [`TensorContainer.to()`](src/tensorcontainer/tensor_container.py:599), [`TensorContainer.cuda()`](src/tensorcontainer/tensor_container.py:760), [`TensorContainer.cpu()`](src/tensorcontainer/tensor_container.py:756)
- Dtype casts: [`TensorContainer.float()`](src/tensorcontainer/tensor_container.py:767), [`TensorContainer.double()`](src/tensorcontainer/tensor_container.py:771), [`TensorContainer.long()`](src/tensorcontainer/tensor_container.py:779), etc.
- Detach/Clone: [`TensorContainer.detach()`](src/tensorcontainer/tensor_container.py:617), [`TensorContainer.clone()`](src/tensorcontainer/tensor_container.py:620)
- Basic elementwise math (propagates to all tensors): [`TensorContainer.add()`](src/tensorcontainer/tensor_container.py:791), [`TensorContainer.mul()`](src/tensorcontainer/tensor_container.py:799), etc.

### A5. torch functions and stacking/concatenation

- `torch.stack` and `torch.cat` are supported across homogenous containers via overrides:
  - [`@implements`](src/tensorcontainer/tensor_container.py:25) registration
  - [`_stack`](src/tensorcontainer/tensor_container.py:829)
  - [`_cat`](src/tensorcontainer/tensor_container.py:860)

```python
b1 = TensorDict({'x': torch.randn(16, 4)}, shape=(16,))
b2 = TensorDict({'x': torch.randn(16, 4)}, shape=(16,))
stacked = torch.stack([b1, b2])  # shape: (2, 16)
concat  = torch.cat([b1, b2])    # shape: (32,)
```

### A6. PyTree + torch.compile

- PyTree integration enables `tree_map`, functional transforms, and graph capture.
- TensorContainer aims for compile friendliness and paths that avoid graph breaks.

---

# Part B — TensorDict-specific concepts (primary)

TensorDict builds on the shared mechanics to provide dynamic, key-based collections, nested conversions, and key flattening tailored for real-world pipelines.

## B1. Dictionary semantics

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

Behavior validated by tests:
- Items equivalence: docs/tests — tests/tensor_dict/test_items.py
- Empty handling: docs/tests — tests/tensor_dict/test_items.py
- Nested dict item normalization: docs/tests — tests/tensor_dict/test_items.py

## B2. Construction and initialization

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

## B3. Nested structures

Nested Python dicts become nested TensorDicts automatically. This is consistently enforced on:
- Construction: [`TensorDict.data_from_dict()`](src/tensorcontainer/tensor_dict.py:116)
- Assignment of a dict to a key: [`TensorDict.__setitem__()`](src/tensorcontainer/tensor_dict.py:291) auto-wraps the dict using the parent’s shape/device.

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

Operation propagation to nested children is automatic because all container ops are defined in the base class and traverse leaves via PyTree:
- Base map used by all ops (with key-path error decoration): [`TensorContainer._tree_map()`](src/tensorcontainer/tensor_container.py:300)

```python
reshaped = nested.reshape(8, 4)
on_cuda  = nested.to('cuda')
indexed  = nested[:, 0]
```

## B4. Flattening keys (flat namespace)

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

## B5. Indexing, slicing, and assignment (TensorDict-specific surface)

String-key indexing delegates to mapping access, while non-string indexing defers to base batch indexing.

- Key-based get: [`TensorDict.__getitem__()`](src/tensorcontainer/tensor_dict.py:275)
- Key-based set with validation/auto-wrap: [`TensorDict.__setitem__()`](src/tensorcontainer/tensor_dict.py:291)
- Batch indexing/slicing (any TensorContainer):
  - Get: [`TensorContainer.__getitem__()`](src/tensorcontainer/tensor_container.py:464) (supports int, slice, boolean mask, tensor indices, and ellipsis)
  - Set: [`TensorContainer.__setitem__()`](src/tensorcontainer/tensor_container.py:511)
  - Ellipsis normalization: [`TensorContainer.transform_ellipsis_index()`](src/tensorcontainer/tensor_container.py:371)

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

## B6. Error handling and diagnostics

TensorDict leans on base validation with detailed path reporting:
- Validation loop adds key-path context: [`TensorContainer._validate()`](src/tensorcontainer/tensor_container.py:340)
- Errors include “Validation error at key ...” with underlying cause.
- Any runtime thrown during `tree_map` operations is decorated with a readable key path: [`TensorContainer._tree_map()`](src/tensorcontainer/tensor_container.py:300)

Typical issues:
- Shape mismatch across batch dims
- Device mismatch when `device` is set on the container
- Too many indices or invalid ellipsis usage (normalized and checked in base)

Examples (from tests):
- Constructor raises on incompatible leaves: tests/tensor_dict/test_shape.py
- Nested mapping mismatch errors: tests/tensor_dict/test_shape.py

## B7. Memory semantics and performance

Memory sharing:
- Leaf tensors are referenced — operations like `reshape` create views at the leaf level where appropriate.
- Cloning deep-copies leaves while preserving metadata (shape/device): [`TensorContainer.clone()`](src/tensorcontainer/tensor_container.py:620)

Device transfers and compile-safe construction:
- Transfers re-materialize the container under a context that disables validation during reconstruction to avoid transient mismatches: [`TensorContainer.unsafe_construction()`](src/tensorcontainer/tensor_container.py:241)
- After reassembly, validation state is reinstated.

Efficient updates:
- Prefer bulk `update()` over many per-key assignments where possible: [`TensorDict.update()`](src/tensorcontainer/tensor_dict.py:331)

## B8. Integration patterns

PyTree operations:
- Transform all tensors with a function using `torch.utils._pytree.tree_map`. Under the hood, base ops use similar traversal: [`TensorContainer._tree_map()`](src/tensorcontainer/tensor_container.py:300)

```python
import torch.utils._pytree as pytree

def norm(x):
    return (x - x.mean()) / (x.std() + 1e-5)

normalized = pytree.tree_map(norm, td)
```

DataLoader:
- You can store TensorDict samples and batch them in a standard `Dataset` and `DataLoader`. Batch dimension is preserved and compatible.

Model integration:
- Pass TensorDicts directly into `nn.Module` forward methods and operate on fields with normal tensor code; indices and ops propagate across the container.

torch.compile:
- Designed for graph capture and compatible with `fullgraph=True` paths.
- TensorDict flatten/unflatten is shallow and compile-friendly: [`TensorDict._pytree_flatten()`](src/tensorcontainer/tensor_dict.py:167), [`TensorDict._pytree_unflatten()`](src/tensorcontainer/tensor_dict.py:210), registration via [`PytreeRegistered`](src/tensorcontainer/utils.py:13)

## B9. Best practices

- Keys: Use descriptive, hierarchical names (optionally with dots) to align with `flatten_keys`.
- Structure: Group related data (inputs, targets, metadata) for clarity.
- Batch shape: Keep batch prefixes consistent and small; push variability into event dims.
- Device:
  - If you need mixed devices, construct with `device=None` (default) so validation permits them: base check [`TensorContainer._is_device_compatible()`](src/tensorcontainer/tensor_container.py:321)
  - Otherwise, set `device='cuda' | 'cpu'` to enforce consistency.
- Performance:
  - Prefer container-level ops over per-leaf loops.
  - Use `update()` for bulk writes.
  - Consider `flatten_keys()` for ML pipelines that benefit from flat namespaces.

## B10. Worked examples

### Example 1 — Time-series batch with nested structure

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

APIs used:
- Batch indexing: [`TensorContainer.__getitem__()`](src/tensorcontainer/tensor_container.py:464)
- Reshape: [`TensorContainer.reshape()`](src/tensorcontainer/tensor_container.py:573)
- Flatten keys: [`TensorDict.flatten_keys()`](src/tensorcontainer/tensor_dict.py:349)
- Device move: [`TensorContainer.to()`](src/tensorcontainer/tensor_container.py:599)

### Example 2 — Boolean masking and assignment

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

APIs used:
- Masking get/set: [`TensorContainer.__getitem__()`](src/tensorcontainer/tensor_container.py:464), [`TensorContainer.__setitem__()`](src/tensorcontainer/tensor_container.py:511)

### Example 3 — Stacking and concatenating rollouts

```python
rollout1 = TensorDict({'obs': torch.randn(16, 64)}, shape=(16,))
rollout2 = TensorDict({'obs': torch.randn(16, 64)}, shape=(16,))

stacked = torch.stack([rollout1, rollout2])  # (2, 16)
joined  = torch.cat([rollout1, rollout2])    # (32,)
```

APIs used:
- `torch.stack` override: [`_stack`](src/tensorcontainer/tensor_container.py:829)
- `torch.cat` override: [`_cat`](src/tensorcontainer/tensor_container.py:860)

---

# API reference quick-links

- Class and constructor
  - [`TensorDict`](src/tensorcontainer/tensor_dict.py:57)
  - [`TensorDict.__init__()`](src/tensorcontainer/tensor_dict.py:95)
  - [`TensorDict.data_from_dict()`](src/tensorcontainer/tensor_dict.py:116)

- Mapping
  - [`TensorDict.__getitem__()`](src/tensorcontainer/tensor_dict.py:275)
  - [`TensorDict.__setitem__()`](src/tensorcontainer/tensor_dict.py:291)
  - [`TensorDict.update()`](src/tensorcontainer/tensor_dict.py:331)
  - [`TensorDict.keys()`](src/tensorcontainer/tensor_dict.py:322), [`TensorDict.values()`](src/tensorcontainer/tensor_dict.py:325), [`TensorDict.items()`](src/tensorcontainer/tensor_dict.py:328)
  - [`TensorDict.flatten_keys()`](src/tensorcontainer/tensor_dict.py:349)

- Container-wide tensor ops (batch dims only)
  - [`TensorContainer.view()`](src/tensorcontainer/tensor_container.py:545), [`TensorContainer.reshape()`](src/tensorcontainer/tensor_container.py:573)
  - [`TensorContainer.permute()`](src/tensorcontainer/tensor_container.py:656), [`TensorContainer.transpose()`](src/tensorcontainer/tensor_container.py:720), [`TensorContainer.t()`](src/tensorcontainer/tensor_container.py:705)
  - [`TensorContainer.unsqueeze()`](src/tensorcontainer/tensor_container.py:732), [`TensorContainer.squeeze()`](src/tensorcontainer/tensor_container.py:683)
  - [`TensorContainer.expand()`](src/tensorcontainer/tensor_container.py:651)
  - [`TensorContainer.detach()`](src/tensorcontainer/tensor_container.py:617), [`TensorContainer.clone()`](src/tensorcontainer/tensor_container.py:620)
  - [`TensorContainer.to()`](src/tensorcontainer/tensor_container.py:599), [`TensorContainer.cuda()`](src/tensorcontainer/tensor_container.py:760), [`TensorContainer.cpu()`](src/tensorcontainer/tensor_container.py:756)
  - [`TensorContainer.float()`](src/tensorcontainer/tensor_container.py:767), [`TensorContainer.double()`](src/tensorcontainer/tensor_container.py:771), [`TensorContainer.long()`](src/tensorcontainer/tensor_container.py:779), etc.

- Indexing helpers and errors
  - [`TensorContainer.__getitem__()`](src/tensorcontainer/tensor_container.py:464), [`TensorContainer.__setitem__()`](src/tensorcontainer/tensor_container.py:511)
  - [`TensorContainer.transform_ellipsis_index()`](src/tensorcontainer/tensor_container.py:371)

- Torch function overrides
  - [`implements`](src/tensorcontainer/tensor_container.py:25)
  - [`_stack`](src/tensorcontainer/tensor_container.py:829)
  - [`_cat`](src/tensorcontainer/tensor_container.py:860)

- Validation and compile
  - [`TensorContainer._validate()`](src/tensorcontainer/tensor_container.py:340)
  - [`TensorContainer._tree_map()`](src/tensorcontainer/tensor_container.py:300)
  - [`TensorContainer.unsafe_construction()`](src/tensorcontainer/tensor_container.py:241)
  - [`PytreeRegistered`](src/tensorcontainer/utils.py:13), [`TensorDict._pytree_flatten()`](src/tensorcontainer/tensor_dict.py:167), [`TensorDict._pytree_unflatten()`](src/tensorcontainer/tensor_dict.py:210)

---

# When to choose TensorDict

Use TensorDict when you need:
- Dynamic schemas with key-based access and runtime evolution
- Nested hierarchical data with easy flattening
- Batch-consistent transformations across heterogeneous tensors
- Seamless integration with PyTorch, PyTree, and torch.compile

For static, type-safe schemas with IDE support, see TensorDataClass. For tensor-like distributions, see TensorDistribution. The overview remains your high-level map: docs/user_guide/overview.md.
