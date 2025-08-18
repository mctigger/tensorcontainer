# TensorDict

A dictionary-like, batched tensor container for structured data with shared batch dimensions.

## Key Benefits

- Unified batch ops: index, slice, and transform all tensors as one object
- Shape safety: enforces consistent leading batch dims across leaves
- Efficient views: zero-copy reshape/slice on batch dims when possible
- Device/dtype consistency: move/cast the entire container at once
- PyTree- and compile-ready: integrates with tree utilities and torch.compile
- Nested mappings: hierarchical data with the same batch semantics

## When to Use

### Use TensorDict when you:
- Need a dynamic mapping of tensors sharing batch dimensions
- Want container-wide operations (indexing, reshape, device/dtype)
- Require nested structures with consistent batch semantics
- Need PyTree compatibility or torch.compile(fullgraph=True)

### Prefer alternatives when you:
- Plain dict: you don’t need batch validation or container-wide behavior
- TensorDataClass: you have a fixed schema with typed, attribute-access fields
- Individual tensors: you operate on a single tensor without a container

## Mental Model

- Batch dimensions: leading dims identical across all leaves
- Event dimensions: trailing dims may differ per leaf
- Container ops affect batch dims only; event dims are preserved
- Validation ensures batch shape (and device if set) match across leaves

## Quick Start

```python
from tensorcontainer import TensorDict
import torch

td = TensorDict({"obs": torch.randn(32, 128), "act": torch.randn(32, 4)}, shape=(32,))
mini = td[:8]  # new TensorDict with shape (8,)
```

## Core Operations

### Indexing / Slicing
```python
from tensorcontainer import TensorDict
import torch

td = TensorDict({"x": torch.arange(4)}, shape=(4,))
td0 = td[0]    # shape: ()
head = td[:2]  # shape: (2,)
```

### Shape Ops (view / reshape / permute)
```python
from tensorcontainer import TensorDict
import torch

td = TensorDict({"x": torch.randn(2, 3, 5)}, shape=(2, 3))
td1 = td.view(6)        # (2,3) -> (6,)
td2 = td1.reshape(3, 2) # -> (3,2)
td3 = td2.permute(1, 0) # -> (2,3)
```

### Device / Dtype
```python
from tensorcontainer import TensorDict
import torch

td = TensorDict({"x": torch.randn(2, 3)}, shape=(2,))
td_f16 = td.to(torch.float16)
td_f32 = td.float()
```

### Stack / Cat
```python
from tensorcontainer import TensorDict
import torch

td1 = TensorDict({"x": torch.zeros(2, 3)}, shape=(2,))
td2 = TensorDict({"x": torch.ones(2, 3)}, shape=(2,))
stk = torch.stack([td1, td2], dim=0)  # shape: (2,2)
cat = torch.cat([td1, td2], dim=0)    # shape: (4,)
```

### Elementwise Math
```python
from tensorcontainer import TensorDict
import torch

a = TensorDict({"x": torch.tensor([[1., 2.], [3., 4.]])}, shape=(2,))
b = TensorDict({"x": torch.tensor([[10., 20.], [30., 40.]])}, shape=(2,))
s = a + b
z = a * 0.5
```

### Flatten Keys
```python
from tensorcontainer import TensorDict
import torch

td = TensorDict({"x": {"a": torch.tensor([1, 2]), "b": torch.tensor([3, 4])}}, shape=(2,))
flat = td.flatten_keys()  # keys: "x.a", "x.b"; values are views (no copy)
```

## Interop

- PyTree: flattens to leaf tensors + context; works with tree utilities
- torch.compile: compatible, including indexing and shape ops (fullgraph=True)
- torch ops: `torch.stack` and `torch.cat` dispatch via `__torch_function__`

## Limitations and Caveats

- Keys must be strings; non-tensor values don’t participate in tensor ops
- All leaves must share the exact leading batch shape
- If device is set/inferred, all leaves must be on that device
- Slice assignment: RHS must be a TensorDict with a compatible slice shape
- `view()` requires contiguous layout; use `reshape()` if unsure
- `permute()/transpose()/t()` operate on batch dims only
- `flatten_keys()` returns view-like values; edits affect the original
- Key collisions possible if original keys contain the separator
- Zero-dim containers require tuple indexing: `td[()]`
- Changing string keys in compiled code may trigger recompilation

## API Summary (selected)

- Construction and mapping
  - `TensorDict(mapping, shape, device=None, dtype=None)`
  - `td[key]`, `td[key] = value`, `td.update(mapping_or_td)`
  - `td.keys()`, `td.values()`, `td.items()`
  - `td.flatten_keys(separator=".")`
- Indexing and shape
  - `td[idx]`, `td[idx] = other_td`
  - `td.view(*shape)`, `td.reshape(*shape)`, `td.expand(*sizes)`
- Transforms
  - `td.permute(*dims)`, `td.transpose(dim0, dim1)`, `td.t()`
  - `td.squeeze(dim=None)`, `td.unsqueeze(dim)`
- Device/dtype and copy
  - `td.to(device_or_dtype)`, `td.cpu()`
  - `td.clone()`, `td.copy()`, `td.detach()`
- Torch ops
  - `torch.stack([td1, td2, ...], dim=...)`
  - `torch.cat([td1, td2, ...], dim=...)`

## References

- [src/tensorcontainer/tensor_dict.py](src/tensorcontainer/tensor_dict.py)
- [tests/tensor_dict/](tests/tensor_dict/)