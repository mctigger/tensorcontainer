from __future__ import annotations

import functools
from collections.abc import MutableMapping
from typing import Any, Dict, List, Mapping, Optional, Tuple, TypeAlias, Union

import torch

# Use the official PyTree utility from torch
import torch.utils._pytree as pytree
from torch import Tensor

from rtd.errors import ShapeMismatchError
from rtd.utils import apply_leaves, get_leaves

# TypeAlias definitions remain the same
TDCompatible: TypeAlias = Union[Tensor, "TensorDict"]
NestedTDCompatible: TypeAlias = Union[TDCompatible, Dict[str, TDCompatible]]

HANDLED_FUNCTIONS = {}


def normalize_device(dev: torch.device) -> torch.device:
    d = torch.device(dev)
    # If no index was given, fill in current_device() for CUDA, leave CPU as-is
    if d.type == "cuda" and d.index is None:
        idx = (
            torch.cuda.current_device()
        )  # e.g. 0 :contentReference[oaicite:4]{index=4}
        return torch.device(f"cuda:{idx}")
    return d


def implements(torch_function):
    """Register a torch function override"""

    @functools.wraps(torch_function)
    def decorator(func):
        HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


def get_leaves(data):
    return pytree.tree_leaves(data)


class TensorDict(MutableMapping[str, TDCompatible]):
    """Wrapper around (nested) dictionaries of torch.Tensor that is torch.compile compatible.

    Note: TensorDict does not copy the input data!

    This class is registered as a PyTorch PyTree, which allows torch.compile and
    functions like torch.stack, torch.cat, etc., to operate on it directly. It
    treats the individual tensors as the leaves of the tree.
    """

    validate_args: bool = config.validate_args

    # --- Add these two methods to your TensorDict class ---

    def __new__(
        cls,
        data: Mapping[str, NestedTDCompatible],
        shape: torch.Size,
        device: Optional[torch.device] = None,
    ):
        """
        Handles instance CREATION and VALIDATION.
        This is called before __init__. It validates the required `shape` and
        sets the device, providing a clean object for torch.compile to trace.
        """
        # --- 1. Validation Logic ---
        # The check for a mandatory shape now lives here.
        if shape is None:
            raise RuntimeError(
                "A 'shape' argument must be provided to the TensorDict constructor."
            )

        # --- 2. Create the raw class instance ---
        instance = super().__new__(cls)

        # --- 3. Attach the validated and inferred attributes ---
        instance.shape = shape

        # We can still handle optional device logic here
        if device is None:
            all_leaves = get_leaves(data)
            instance.device = all_leaves[0].device if all_leaves else None
        else:
            instance.device = device

        # --- 4. Return the prepared instance to be passed to __init__ ---
        return instance

    def __init__(
        self,
        data: Mapping[str, NestedTDCompatible],
        shape: torch.Size,
        device: Optional[torch.device] = None,
    ):
        """
        Handles simple instance INITIALIZATION.
        The `__new__` method has already validated and set `self.shape` and
        `self.device`. This method's only job is to assign the data.
        """
        # The `shape` and `device` arguments are ignored here.
        self.data = dict(data)

    def _check_tensor_shape(self, tensor: Tensor):
        """Check if a tensor's shape starts with the TensorDict's batch_shape."""
        if tensor.shape[: len(self.shape)] != self.shape:
            raise ShapeMismatchError(
                f"Tensor shape {tensor.shape} is incompatible with TensorDict shape {self.shape}.",
                tensor,
            )

    def _check_tensor_device(self, tensor: TDCompatible, device: torch.device):
        """
        Check if the tensor device matches the device of the tensordict.
        """
        if normalize_device(tensor.device) != normalize_device(device):
            raise ValueError(
                f"Tensor device {tensor.device} does not match expected device {device}."
            )

    @classmethod
    def generate_shape(cls, data: Mapping[str, Any]) -> Tuple[int, ...]:
        """
        Infers the common leading dimensions (batch shape) from the leaf tensors.
        """
        # Get the shapes of all tensors in the nested dictionary structure.
        shapes = get_leaves(apply_leaves(data, lambda x: x.shape))
        if not shapes:
            return ()

        # Find the common prefix of all shapes.
        shape = []
        for dims in zip(*shapes):
            if len(set(dims)) == 1:
                shape.append(dims[0])
            else:
                break

        return tuple(shape)

    def _pytree_flatten(self) -> Tuple[List[Tensor], Tuple]:
        """
        Flattens the TensorDict into its tensor leaves and a structural context.
        """
        # Use pytree to handle flattening the (potentially nested) dictionary
        flat_leaves, children_spec = pytree.tree_flatten(self.data)

        # For each actual tensor, determine how many of its dims are "event" dims
        event_ndims = tuple(leaf.ndim - len(self.shape) for leaf in flat_leaves)

        # THE FIX: The context should ONLY contain structural information.
        # - `children_spec` describes the internal layout (keys, nesting).
        # - `event_ndims` describes the batch/event split for each leaf.
        # - DO NOT include `self.device` or `self.shape` here.
        context = (children_spec, event_ndims)
        return flat_leaves, context

    @classmethod
    def _pytree_unflatten(cls, flat_leaves: List[Tensor], context: Tuple) -> TensorDict:
        """
        Reconstructs a TensorDict from tensor leaves and the structural context.
        """
        if not flat_leaves:
            # Handle empty case: create an empty TD with an empty shape and no device.
            return cls({}, shape=torch.Size(), device=None)

        children_spec, event_ndims = context

        # Calculate the new shape precisely from the first new leaf.
        first_leaf = flat_leaves[0]
        first_event_ndim = event_ndims[0]
        new_batch_ndim = first_leaf.ndim - first_event_ndim
        new_shape = first_leaf.shape[:new_batch_ndim]

        # Reconstruct the internal dictionary structure using the children_spec
        data = pytree.tree_unflatten(flat_leaves, children_spec)

        # Infer the device from the new leaves, which are the source of truth
        new_device = first_leaf.device

        return cls(data, shape=new_shape, device=new_device)

    # --- Standard MutableMapping methods ---

    def __getitem__(self, key: Union[str, Any]) -> TDCompatible:
        if isinstance(key, str):
            return self.data[key]
        # For slicing, indexing, etc., apply the operation to all leaf tensors.
        # This is automatically handled by the PyTree mechanism.
        return pytree.tree_map(lambda x: x[key], self)

    def __setitem__(self, key: str, value: TDCompatible):
        # TODO: Add shape validation for the new value if validate_args is on.
        self.data[key] = value

    def __delitem__(self, key: str):
        del self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    # --- Overloaded methods leveraging PyTrees ---
    def to(self, *args, **kwargs) -> TensorDict:
        """Moves all tensors to a specified device or dtype."""
        return pytree.tree_map(lambda x: x.to(*args, **kwargs), self)

    def detach(self) -> TensorDict:
        """Detaches all tensors from the computation graph."""
        return pytree.tree_map(lambda x: x.detach(), self)

    def copy(self) -> TensorDict:
        """Creates a shallow copy of the TensorDict."""
        return pytree.tree_map(lambda x: x.copy(), self)

    def clone(self) -> TensorDict:
        """Creates a deep copy of the TensorDict and its tensors."""
        return pytree.tree_map(lambda x: x.clone(), self)

    def view(self, *shape) -> TensorDict:
        """Reshapes the batch dimensions of all tensors."""
        return pytree.tree_map(
            lambda x: x.view(*shape, *x.shape[len(self.shape) :]), self
        )

    def unsqueeze(self, dim: int) -> TensorDict:
        """Unsqueezes all tensors at a given dimension."""
        return pytree.tree_map(lambda x: x.unsqueeze(dim), self)

    def expand(self, *shape) -> TensorDict:
        """Expands the batch dimensions of all tensors."""
        return pytree.tree_map(
            lambda x: x.expand(*shape, *x.shape[len(self.shape) :]), self
        )

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Extends torch with TensorDict

        See https://pytorch.org/docs/stable/notes/extending.html#extending-torch
        """
        if kwargs is None:
            kwargs = {}

        if func not in HANDLED_FUNCTIONS or not all(
            issubclass(t, (torch.Tensor, TensorDict)) for t in types
        ):
            return NotImplemented

        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def __contains__(self, key):
        return key in self.data

    def __repr__(self) -> str:
        # A helpful representation for debugging
        return f"TensorDict(shape={self.shape}, device={self.device}, keys={list(self.keys())})"


@implements(torch.stack)
def _stack(tensors: Union[Tuple[TensorDict, ...], List[TensorDict]], dim: int = 0):
    return pytree.tree_map(lambda *x: torch.stack(x, dim), *tensors)


@implements(torch.cat)
def _cat(tensors: Union[Tuple[TensorDict, ...], List[TensorDict]], dim: int = 0):
    return pytree.tree_map(lambda *x: torch.cat(x, dim), *tensors)


@implements(torch.unsqueeze)
def _unsqueeze(tensor: TensorDict, dim: int = 0):
    cls = type(tensor)
    assert dim <= len(tensor.shape)
    return pytree.tree_map(lambda x: torch.unsqueeze(x, dim), tensors)


# --- Register TensorDict as a PyTree node with PyTorch ---
# This step is crucial for torch.compile and other functions to understand TensorDict.
pytree.register_pytree_node(
    TensorDict,
    lambda td: td._pytree_flatten(),
    TensorDict._pytree_unflatten,
    flatten_with_keys_fn=None,  # Optional advanced usage
)


if __name__ == "__main__":
    # --- Example Usage ---
    # Demonstrate the fix for the "greedy shape" problem.
    # Both 'obs' and 'action' have the same trailing dimension (64), but it's
    # an event dimension, not a batch dimension.

    B, T = 4, 10
    batch_shape = torch.Size([B, T])

    td1 = TensorDict(
        {
            "obs": torch.randn(B, T, 3, 64, 64),
            "action": torch.randn(B, T, 8),
            "nested": {"state": torch.randn(B, T, 128)},
        },
        shape=batch_shape,
        device="cpu",
    )

    print(f"Original TensorDict: {td1}")
    print(f"Original 'obs' shape: {td1['obs'].shape}")

    td2 = td1.clone()

    print(td1, td2)
    # Stack two such TensorDicts along a new first dimension (dim=0)
    stacked_td = torch.stack([td1, td2], dim=0)

    print(f"\nStacked TensorDict: {stacked_td}")
    print(f"Stacked 'obs' shape: {stacked_td['obs'].shape}")
    print(f"Stacked nested 'state' shape: {stacked_td['nested']['state'].shape}")

    # Verify the new shape is correct: (2, B, T)
    expected_shape = torch.Size([2, B, T])
    print(f"\nExpected final shape: {expected_shape}")
    print(f"Actual final shape:   {stacked_td.shape}")

    assert stacked_td.shape == expected_shape
    print("\nShape check passed! The new batch shape was calculated correctly.")
