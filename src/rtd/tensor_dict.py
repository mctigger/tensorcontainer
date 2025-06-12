from __future__ import annotations

import functools
from typing import Any, Dict, List, Mapping, Optional, Tuple, TypeAlias, Union

import torch

# Use the official PyTree utility from torch
import torch.utils._pytree as pytree
from torch import Tensor

from rtd.tensor_container import TensorContainer
from rtd.utils import PytreeRegistered


# TypeAlias definitions remain the same
TDCompatible: TypeAlias = Union[Tensor, TensorContainer]
NestedTDCompatible: TypeAlias = Union[TDCompatible, Dict[str, TDCompatible]]

HANDLED_FUNCTIONS = {}


def implements(torch_function):
    """Register a torch function override for TensorDict."""

    @functools.wraps(torch_function)
    def decorator(func):
        HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


class TensorDict(TensorContainer, PytreeRegistered):
    """
    A dictionary-like container for torch.Tensors that is compatible with
    torch.compile and standard PyTorch functions.
    """

    def __init__(
        self,
        data: Mapping[str, NestedTDCompatible],
        shape: Tuple[int],
        device: Optional[Union[str, torch.device]] = None,
        validate_args: bool = True,
    ):
        """
        Initializes the TensorDict. This constructor is kept simple for
        `torch.compile` compatibility, performing direct attribute assignment.
        """
        super().__init__(shape)

        self.data = TensorDict.data_from_dict(data, shape, device)
        self.shape = shape
        self.device = device

        if validate_args:
            self._tree_validate_shape(data)

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

    def _is_shape_compatible(self, shape):
        batch_ndim = len(self.shape)
        leaf_ndim = len(shape)

        return leaf_ndim >= batch_ndim and shape[:batch_ndim] == self.shape

    def _is_device_compatible(self, device):
        if self.device is None:
            return True

        return self.device == device

    def _tree_validate_shape(self, data):
        """
        Validates that the shapes of all nested tensors in the TensorDict start
        with the expected batch shape.

        This method recursively traverses the entire data structure.
        """

        # Use tree_flatten_with_path to get a list of (path, leaf) pairs for
        # all leaves in the nested structure of self.data.
        keypath_leaf_pairs = pytree.tree_leaves_with_path(data)

        batch_shape = self.shape

        for key_path, leaf in keypath_leaf_pairs:
            path_str = pytree.keystr(key_path)

            if leaf.ndim > 0 and not self._is_shape_compatible(leaf.shape):
                # Use pytree.keystr to generate a readable path for the error message.
                raise ValueError(
                    f"Shape mismatch at '{path_str}': The tensor shape {leaf.shape} "
                    f"is not compatible with the TensorDict's batch shape {batch_shape}."
                )

    def _tree_validate_device(self, data):
        """
        Validates that the shapes of all nested tensors in the TensorDict start
        with the expected batch shape.

        This method recursively traverses the entire data structure.
        """

        # Use tree_flatten_with_path to get a list of (path, leaf) pairs for
        # all leaves in the nested structure of self.data.
        keypath_leaf_pairs = pytree.tree_leaves_with_path(data)

        for key_path, leaf in keypath_leaf_pairs:
            path_str = pytree.keystr(key_path)

            if not self._is_device_compatible(leaf.device):
                # Use pytree.keystr to generate a readable path for the error message.
                raise ValueError(
                    f"Device mismatch at '{path_str}': The tensor device {leaf.device} "
                    f"is not compatible with the TensorDict's device {self.device}."
                )

    def _get_pytree_context(
        self, flat_leaves: List[Tensor], children_spec: pytree.TreeSpec
    ) -> Tuple:
        """
        Private helper to compute the pytree context for this TensorDict.

        The context captures the necessary metadata to reconstruct the TensorDict
        from its leaves: the original structure of the contained data and the
        event dimensions of each tensor.
        """
        batch_ndim = len(self.shape)
        event_ndims = tuple(leaf.ndim - batch_ndim for leaf in flat_leaves)
        return (children_spec, event_ndims)

    def _pytree_flatten(self) -> Tuple[List[Tensor], Tuple]:
        """
        Flattens the TensorDict into its tensor leaves and static metadata.
        (Implementation for `flatten_fn` in `register_pytree_node`)
        """
        # Get the leaves and the spec describing the structure of self.data
        flat_leaves, children_spec = pytree.tree_flatten(self.data)

        # Use the helper to compute and return the context
        context = self._get_pytree_context(flat_leaves, children_spec)
        return flat_leaves, context

    def _pytree_flatten_with_keys_fn(
        self,
    ) -> Tuple[List[Tuple[pytree.KeyPath, Tensor]], Tuple]:
        """
        Flattens the TensorDict into key-path/leaf pairs and static metadata.
        (Implementation for `flatten_with_keys_fn` in `register_pytree_node`)
        """
        # Use the public API to robustly get key paths, leaves, and the spec
        keypath_leaf_list, children_spec = pytree.tree_flatten_with_path(self.data)

        # Extract just the leaves to pass to the context helper
        flat_leaves = [leaf for _, leaf in keypath_leaf_list]

        # Use the helper to compute and return the context
        context = self._get_pytree_context(flat_leaves, children_spec)
        return keypath_leaf_list, context

    @classmethod
    def _pytree_unflatten(cls, leaves: List[Tensor], context: Tuple) -> TensorDict:
        """
        Reconstructs a TensorDict by creating a new instance and manually
        populating its attributes. This approach is more robust for torch.compile's
        code generation phase.
        """
        (children_spec, event_ndims) = context  # Unpack the context

        if not leaves:
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
        data = pytree.tree_unflatten(leaves, children_spec)

        # Infer new_shape and new_device
        first_leaf_reconstructed = leaves[0]

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
        if not isinstance(value, (Tensor, TensorContainer)):
            raise ValueError("value must be a Tensor or TensorContainer")

        self._tree_validate_shape(value)
        self._tree_validate_device(value)

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

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def update(self, other: Union[Dict[str, NestedTDCompatible], TensorDict]):
        """
        Updates the TensorDict with values from another dictionary or TensorDict.
        """
        if isinstance(other, TensorDict):
            other = other.data
        for key, value in other.items():
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

    def flatten_keys(self) -> TensorDict:
        """
        Returns a TensorDict with flattened keys.
        """
        out = {}
        for key, value in self.data.items():
            if isinstance(value, TensorDict):
                sub_dict = value.flatten_keys()
                for sub_key, sub_value in sub_dict.data.items():
                    out[f"{key}.{sub_key}"] = sub_value
            else:
                out[key] = value
        return TensorDict(out, self.shape, self.device)

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
            elif isinstance(value, Tensor):
                return f"{key}: Tensor(shape={value.shape}, dtype={value.dtype})"
            else:
                return f"{key}: {repr(value)}"

        items_str = ", ".join(_format_item(k, v) for k, v in self.data.items())
        return f"TensorDict(shape={self.shape}, device={device}, {items_str})"
