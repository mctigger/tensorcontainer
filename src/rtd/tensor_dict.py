from __future__ import annotations

import functools
from typing import Any, Dict, List, Mapping, Optional, Tuple, TypeAlias, Union

import torch

# Use the official PyTree utility from torch
import torch.utils._pytree as pytree
from torch import Tensor

# TypeAlias definitions remain the same
TDCompatible: TypeAlias = Union[Tensor, "TensorDict"]
NestedTDCompatible: TypeAlias = Union[TDCompatible, Dict[str, TDCompatible]]

HANDLED_FUNCTIONS = {}


def implements(torch_function):
    """Register a torch function override for TensorDict."""

    @functools.wraps(torch_function)
    def decorator(func):
        HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


class TensorDict:
    """
    A dictionary-like container for torch.Tensors that is compatible with
    torch.compile and standard PyTorch functions.
    """

    def __init__(
        self,
        data: Mapping[str, NestedTDCompatible],
        shape: Tuple[int],
        device: Optional[torch.device] = None,
    ):
        """
        Initializes the TensorDict. This constructor is kept simple for
        `torch.compile` compatibility, performing direct attribute assignment.
        """
        self.data = TensorDict.data_from_dict(data, shape, device)
        self.shape = shape
        self.device = device

    @classmethod
    def data_from_dict(cls, data, shape, device=None):
        result = {}
        for k, v in data.items():
            if isinstance(v, dict):
                result[k] = TensorDict(
                    TensorDict.data_from_dict(v, shape, device), shape, device
                )
            else:
                result[k] = v

        return result

    def _pytree_flatten(self) -> Tuple[List[Tensor], Tuple]:
        """
        Flattens the TensorDict into its tensor leaves and static metadata.
        """
        flat_leaves, children_spec = pytree.tree_flatten(self.data)
        batch_ndim = len(self.shape)
        event_ndims = tuple(leaf.ndim - batch_ndim for leaf in flat_leaves)
        context = (children_spec, event_ndims)
        return flat_leaves, context

    @classmethod
    def _pytree_unflatten(cls, flat_leaves: List[Tensor], context: Tuple) -> TensorDict:
        """
        Reconstructs a TensorDict by creating a new instance and manually
        populating its attributes. This approach is more robust for torch.compile's
        code generation phase.
        """
        (children_spec, event_ndims) = context  # Unpack the context

        if not flat_leaves:
            # Handle the empty case explicitly with direct instantiation
            obj = cls.__new__(cls)
            obj.data = {}
            obj.shape = []  # Or a sensible default for empty
            obj.device = None
            return obj

        # 1. Infer dynamic attributes (shape, device) from the new tensor leaves.
        # The fix: Manually reconstruct the object.
        children_spec, event_ndims = (
            context  # This unpacking should match flatten's context
        )

        # Reconstruct the nested dictionary structure using the unflattened leaves
        data = pytree.tree_unflatten(flat_leaves, children_spec)

        # Infer new_shape and new_device
        first_leaf_reconstructed = flat_leaves[0]

        # Simplified inference (common and works for stack/cat):
        new_device = first_leaf_reconstructed.device

        # Calculate new_shape based on the structure and first leaf.
        # For operations like `stack`, the batch shape changes.
        # If `_pytree_flatten` correctly passes `event_ndims`, then:
        if event_ndims[0] == 0:
            new_shape = first_leaf_reconstructed.shape
        else:
            new_shape = first_leaf_reconstructed.shape[: -event_ndims[0]]

        # Instead of calling `_reconstruct_tensordict` which wraps `cls(...)`,
        # directly use `cls.__new__` and set attributes.
        obj = cls.__new__(cls)
        obj.data = (
            data  # This is the reconstructed nested dictionary of tensors/TensorDicts
        )
        obj.shape = new_shape
        obj.device = new_device
        return obj

    # --- Standard MutableMapping methods ---
    def __getitem__(self, key: Union[str, Any]) -> TDCompatible:
        if isinstance(key, str):
            return self.data[key]

        return pytree.tree_map(lambda x: x[key], self)

    def __setitem__(self, key: str, value: TDCompatible):
        self.data[key] = value

    def __delitem__(self, key: str):
        del self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def keys(self):
        return self.data.keys()

    # --- Overloaded methods leveraging PyTrees ---

    def to(self, *args, **kwargs) -> TensorDict:
        return pytree.tree_map(lambda x: x.to(*args, **kwargs), self)

    def detach(self) -> TensorDict:
        return pytree.tree_map(lambda x: x.detach(), self)

    def clone(self) -> TensorDict:
        return pytree.tree_map(lambda x: x.clone(), self)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in HANDLED_FUNCTIONS or not all(
            issubclass(t, (torch.Tensor, TensorDict)) for t in types
        ):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def __repr__(self) -> str:
        # Infer device for representation if not set
        device = self.device
        if device is None and self.data:
            try:
                device = pytree.tree_leaves(self.data)[0].device
            except IndexError:
                pass

        def _format_item(key, value):
            if isinstance(value, TensorDict):
                return f"{key}: {value.__repr__()}"
            elif isinstance(value, torch.Tensor):
                return f"{key}: Tensor(shape={value.shape}, dtype={value.dtype})"
            else:
                return f"{key}: {repr(value)}"

        items_str = ", ".join(_format_item(k, v) for k, v in self.data.items())
        return f"TensorDict(shape={self.shape}, device={device}, {items_str})"

    def update(self, other: Union[Dict[str, NestedTDCompatible], TensorDict]):
        """
        Updates the TensorDict with values from another dictionary or TensorDict.
        """
        if isinstance(other, TensorDict):
            other = other.data
        for key, value in other.items():
            if isinstance(value, dict):
                value = TensorDict(value, shape=self.shape, device=self.device)
            if isinstance(value, torch.Tensor):
                if value.shape != self.shape:
                    raise RuntimeError(
                        f"Shape mismatch: value shape is {value.shape}, expected {self.shape}"
                    )
            self[key] = value

    def copy(self) -> TensorDict:
        """
        Creates a new TensorDict where all children anywhere in the data tree are also
        copied, but the leaves are the same. Implemented using pytree.
        """

        def copy_item(item):
            if isinstance(item, TensorDict):
                return item.copy()
            else:
                return item

        data = pytree.tree_map(copy_item, self.data)
        return TensorDict(data, self.shape, self.device)


# --- PyTree-aware implementations of torch functions ---


@implements(torch.stack)
def _stack(tensors: Union[Tuple[TensorDict, ...], List[TensorDict]], dim: int = 0):
    returns = pytree.tree_map(lambda *x: torch.stack(x, dim), *tensors)
    return returns


@implements(torch.cat)
def _cat(tensors: Union[Tuple[TensorDict, ...], List[TensorDict]], dim: int = 0):
    return pytree.tree_map(lambda *x: torch.cat(x, dim), *tensors)


# --- Register TensorDict as a PyTree node with PyTorch ---
pytree.register_pytree_node(
    TensorDict,
    lambda td: td._pytree_flatten(),
    TensorDict._pytree_unflatten,
)

if __name__ == "__main__":
    # --- Setup Example Data ---
    B, T = 4, 10
    batch_shape = torch.Size([B, T])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    td1 = TensorDict(
        {
            "obs": torch.randn(B, T, 3, 64, 64, device=device),
            "action": torch.randn(B, T, 8, device=device),
            "nested": TensorDict(
                {"state": torch.randn(B, T, 128, device=device)},
                shape=batch_shape,
                device=device,
            ),
        },
        shape=batch_shape,
        device=device,
    )
    td2 = td1.clone()

    print("--- Eager Mode Verification ---")
    eager_stacked_td = torch.stack([td1, td2], dim=0)
    print(f"Eager Stacked TensorDict: {eager_stacked_td}")
    expected_shape = torch.Size([2, B, T])
    assert eager_stacked_td.shape == expected_shape
    print("\nEager shape check passed!")

    print("\n" + "=" * 50 + "\n")

    # --- `torch.compile` Verification ---
    print("--- `torch.compile` Verification ---")

    @torch.compile
    def compiled_stack_fn(d1: TensorDict, d2: TensorDict) -> TensorDict:
        return torch.stack([d1, d2], dim=0)

    # Run the compiled function
    compiled_stacked_td = compiled_stack_fn(td1, td2)

    explanation = torch._dynamo.explain(compiled_stack_fn)(td1, td2)

    print("EXPLANATION", explanation.graph_break_count, explanation.break_reasons)

    print(f"Compiled Stacked TensorDict: {compiled_stacked_td}")
    assert compiled_stacked_td.shape == eager_stacked_td.shape
    assert torch.equal(compiled_stacked_td["obs"], eager_stacked_td["obs"])
    assert torch.equal(
        compiled_stacked_td["nested"]["state"], eager_stacked_td["nested"]["state"]
    )

    print("\n`torch.compile` verification successful!")
