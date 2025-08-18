"""
TensorDict

A dictionary-like, batched tensor container for structured data.

TensorDict is part of the TensorContainer family. It lets you organize a set of
tensors (and nested containers) under string keys while preserving a shared,
leading batch shape. Like a torch.Tensor, it supports indexing, slicing, shape
operations over batch dimensions, device/dtype moves, and container-wide math
operations. It integrates with PyTorch PyTree utilities and works with
torch.compile(fullgraph=True).

Key Benefits
- Mapping + tensor semantics: Treat a dict of tensors as one batched object.
- Clear shape model: Distinguishes batch dims (shared) vs event dims (per-leaf).
- Unified ops: View/reshape/permute/expand, stack/cat, math ops across leaves.
- Device and dtype consistency: Move/cast the entire container at once.
- PyTree + compile-ready: Plays well with tree utilities and torch.compile.

When to Use TensorDict
- Use TensorDict when you:
  - Need a dynamic mapping of tensors with a common batch shape.
  - Want nested structures that follow the same batch semantics.
  - Need container-wide operations (indexing, shape ops, device/dtype, math).
  - Want reliable PyTree behavior and torch.compile compatibility.

- Prefer a plain dict of tensors when you:
  - Do not need batch-shape validation or container-wide behavior.
  - Only pass around small structures without operations on the whole.

- Prefer TensorDataClass when you:
  - Have a fixed, typed schema (dataclass-like fields and static structure).
  - Want attribute access and type-checked fields.

Mental Model: Batch vs Event Dimensions
- Batch dims: Leading dimensions that are the same across all leaves.
- Event dims: Remaining trailing dimensions that can differ by leaf.
- Only batch dims are affected by container-wide indexing/shape ops.
- Event dims are preserved and may vary between leaves.

Minimal Examples
1) Construction, access, and updates
------------------------------------

Example: basic usage and nested wrapping.

```python
import torch
from tensorcontainer import TensorDict

# Construct with nested dicts; nested will be wrapped as TensorDict
td = TensorDict(
    {
        "x": {"a": torch.zeros(2, 3)},
        "y": torch.ones(2, 3),
    },
    shape=(2, 3),
)

# Access by key
assert td["y"].shape == (2, 3)

# Slice batch dims (returns a new TensorDict)
row0 = td[0]
assert row0.shape == (3,)

# Update with a new key and with a nested mapping
td.update({"z": torch.arange(6).reshape(2, 3)})
td.update({"x": {"b": torch.full((2, 3), 7)}})
```

2) Batch/event semantics and shape transforms
---------------------------------------------

Example: view/reshape/permute manipulate only batch dims.

```python
import torch
from tensorcontainer import TensorDict

td = TensorDict(
    {
        "obs": torch.randn(4, 5, 128),
        "act": torch.randn(4, 5, 6),
    },
    shape=(4, 5),
)

td2 = td.view(20)          # batch (4,5) -> (20,), event dims preserved
td3 = td2.reshape(2, 10)   # reshape batch dims
td4 = td3.permute(1, 0)    # permute batch dims (2,10) -> (10,2)
assert td4["obs"].shape == (10, 2, 128)
```

3) Slice assignment (container to container)
-------------------------------------------

Example: assign a sliced region from another TensorDict with compatible shape.

```python
import torch
from tensorcontainer import TensorDict

dest = TensorDict({"a": torch.zeros(2, 3, 4), "b": torch.zeros(2, 3)}, shape=(2, 3))
src  = TensorDict({"a": torch.randn(1, 3, 4), "b": torch.ones(1, 3)},  shape=(1, 3))

dest[0:1] = src  # assign into slice along batch dims
```

4) Stack and cat across containers
----------------------------------

Example: torch.stack inserts a new batch dim, torch.cat concatenates a batch dim.

```python
import torch
from tensorcontainer import TensorDict

td1 = TensorDict({"x": torch.arange(6).reshape(2, 3)}, shape=(2, 3))
td2 = TensorDict({"x": torch.arange(6, 12).reshape(2, 3)}, shape=(2, 3))

stacked = torch.stack([td1, td2], dim=0)  # result shape (2, 2, 3)
catted  = torch.cat([td1, td2], dim=1)    # result shape (2, 6)
```

5) Device, dtype, clone/copy/detach
-----------------------------------

Example: move/cast all leaves; understand copy semantics.

```python
import torch
from tensorcontainer import TensorDict

td = TensorDict({"a": torch.randn(4, 3)}, shape=(4,))
td_f16   = td.to(torch.float16)  # dtype change for leaves
td_clone = td.clone()            # deep copy of leaf tensors
td_copy  = td.copy()             # shares leaf tensors (shallow container copy)
td_det   = td.detach()           # shares storage, detached from autograd
```

6) Flatten keys (view-like, no copies of values)
------------------------------------------------

Example: flatten nested keys into a single level with a separator.

```python
import torch
from tensorcontainer import TensorDict

td = TensorDict(
    {
        "x": {
            "a": torch.tensor([[1, 2], [3, 4]]),
            "b": torch.tensor([[5, 6], [7, 8]]),
        }
    },
    shape=(2, 2),
)

flat = td.flatten_keys()  # keys: "x.a", "x.b"
assert flat["x.a"] is td["x"]["a"]  # values are the same objects (no copy)
```

How To Use (Core Behaviors)
- Construction
  - Create with a mapping of str keys to tensors or other containers.
  - Nested plain dicts are recursively wrapped as TensorDict.
  - All leaves must share the leading batch shape specified by shape=(...).
  - Device is inferred from leaves or set explicitly; all leaves must match it.

- Keys and Values
  - Keys are strings.
  - Values are tensors or TensorContainers. Non-tensor values may be preserved as
    metadata in PyTree contexts but will not participate in tensor ops.

- Indexing and Slicing
  - Indexing applies to batch dims only.
  - Supports integers, slices, ellipsis, boolean masks (like torch semantics).
  - 0-dim containers require tuple-based indexing (possibly empty).

- Setting Items
  - key-based: td["k"] = tensor_or_container (validates device and batch shape).
  - slice-based: td[idx] = other_td_of_same_type (RHS must be a TensorDict).

- Updates and Merge
  - td.update(mapping_or_td) adds/replaces keys. Nested dicts auto-wrap.
  - Device and batch shape validation applies to all inserted values.

- Batch Shape Transforms (batch dims only)
  - view/reshape: Reshape batch dims; reshape may copy if needed.
  - permute/transpose/t: Rearrange batch dims; event dims preserved.
  - squeeze/unsqueeze: Remove/add batch singleton dims.
  - expand: Broadcast batch dims (view-like if possible).

- Device/Dtype Ops
  - to(), cpu(), cuda() move all leaves consistently.
  - float()/double()/half()/long()/int() cast leaf dtypes.

- Copying and Autograd
  - copy(): Shallow container copy; leaves are the same tensors (shared).
  - clone(): Deep copy of leaves; separate storages.
  - detach(): Leaves are detached views sharing storage; no gradient tracking.

- Stacking and Concatenation
  - torch.stack(list_of_td, dim): Insert new batch dim at position dim.
  - torch.cat(list_of_td, dim): Concatenate along an existing batch dim.
  - Requires compatible shapes across inputs.

- Math Operations
  - Elementwise math across leaves: add, sub, mul, div, pow, abs, sqrt, log,
    neg, clamp, etc. Returns a new container.

- Flatten/Unflatten Keys
  - flatten_keys(separator=".") returns a flat mapping where nested keys are
    joined by the separator. Values are the same objects (no copying).

Interop
- PyTree Integration
  - TensorDict is a PyTree node; it flattens to leaf tensors plus context
    that records batch shape and metadata. Tree utilities work naturally.

- torch.compile Compatibility
  - Works with torch.compile(fullgraph=True) across creation, indexing, shape
    ops, stack/cat, device/dtype changes in the test suite.

- __torch_function__ Dispatch
  - Container overrides for selected torch ops (notably stack and cat).
  - Use instance methods for shape/device/math ops provided on the container.

Limitations and Caveats
- Keys and Values
  - Keys must be strings.
  - Assigning non-tensor values as leaves is not supported for tensor ops.
    Arbitrary objects lacking tensor attributes (like .shape) will fail on
    validation.
- Device Constraints
  - If a device is set or inferred, all assigned leaves must match; otherwise
    assignment/update raises an error.
- Shape Constraints
  - All leaves must share the leading batch shape. Mismatched shapes error out.
  - view() requires contiguity; reshape() may copy if needed.
- Slice Assignment
  - td[idx] = rhs requires rhs to be a TensorDict with a compatible slice shape.
  - Assigning a raw tensor or scalar at container-level raises ValueError
    (use key-based assignment instead).
- Permute/Transpose Dims
  - permute() expects exactly the batch dims; dims must be unique and in-range.
  - transpose()/t(): Pass indices that refer to batch dims to avoid touching
    event dims (event dims are per-leaf and should not be permuted here).
- Flattened Keys
  - flatten_keys() preserves the same value objects (no copies). Modifying a
    leaf in the flattened view affects the original structure.
  - If original keys contain the separator, collisions can occur; no escaping
    or collision resolution is performed.
- Compilation Recompilations
  - Compiled __getitem__(str) may recompile for different key constants.

API (Selected)
- Construction and Mapping
  - TensorDict(mapping, shape=(...), device=..., dtype=...)
  - td[key], td[key] = value, td.update(mapping_or_td)
  - td.keys(), td.values(), td.items()
  - td.flatten_keys(separator: str = ".")

- Shape and Indexing (batch dims only)
  - td[idx], td[idx] = other_td
  - td.view(*shape), td.reshape(*shape), td.expand(*sizes)
  - td.permute(*dims), td.transpose(dim0, dim1), td.t()
  - td.squeeze(dim=None), td.unsqueeze(dim)

- Device/Dtype and Copying
  - td.to(device_or_dtype), td.cpu(), td.cuda()
  - td.float(), td.double(), td.half(), td.long(), td.int()
  - td.clone(), td.copy(), td.detach()

- Torch Op Interop
  - torch.stack([td1, td2, ...], dim=...)
  - torch.cat([td1, td2, ...], dim=...)

Gotchas (Quick)
- Use reshape() instead of view() for non-contiguous layouts.
- For slice assignment, the RHS must be a TensorDict; use key assignment for
  individual tensors.
- Be careful with permute()/transpose(): operate only on batch dims.
- flatten_keys() is view-like regarding values; changes reflect in the original.
- Changing string keys in compiled code may trigger recompilation.

References (Implementation and Tests in this repo)
- Implementation
  - src/tensorcontainer/tensor_dict.py
  - src/tensorcontainer/tensor_container.py
  - src/tensorcontainer/tensor_dataclass.py
  - src/tensorcontainer/tensor_annotated.py
  - src/tensorcontainer/utils.py
  - src/tensorcontainer/types.py
- Tests
  - tests/tensor_dict/ (init/get/set, shape ops, device/casting, clone/copy/detach,
    update/stack/cat/expand, math ops, flatten_keys, metadata, compile, masks, pytree)

"""

# Optional: tiny smoke test block to quickly verify imports and a couple of
# core behaviors by running this file directly. Keep it minimal and fast.
if __name__ == "__main__":
    import torch
    from tensorcontainer import TensorDict

    td = TensorDict({"a": torch.randn(2, 3), "b": torch.ones(2, 3)}, shape=(2, 3))
    assert td.shape == (2, 3)
    assert td["a"].shape == (2, 3)

    td2 = td.view(6)
    assert td2.shape == (6,)
    assert td2["b"].shape == (6,)

    td3 = torch.stack([td2, td2], dim=0)
    assert td3.shape == (2, 6)

    print("TensorDict quick smoke test passed.")
