from __future__ import annotations

import functools
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    TypeAlias,
    Union,
    overload,
)


import torch

# Use the official PyTree utility from torch
import torch.utils._pytree as pytree
from torch import Tensor

from rtd.tensor_container import TensorContainer
from rtd.utils import PytreeRegistered

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
    Dictionary-like container for batched Tensors sharing a common batch shape.

    - PyTree & torch.compile compatible
    - Standard mapping ops: getitem, setitem, update, etc.
    - Utilities: flatten_keys, copy, and more

    Example:
        >>> td = TensorDict({'x': torch.zeros(4, 3)}, shape=(4,))
        >>> td['x'].shape
        torch.Size([4, 3])
        >>> td.flatten_keys()
        TensorDict(shape=(4,), x: Tensor(shape=(4,3)))
    """

    def __init__(
        self,
        data: Mapping[str, NestedTDCompatible],
        shape: Tuple[int, ...],
        device: Optional[Union[str, torch.device]] = None,
        validate_args: bool = True,
    ):
        """
        Initializes the TensorDict. This constructor is kept simple for
        `torch.compile` compatibility, performing direct attribute assignment.
        """
        super().__init__(shape, device)

        self.data = TensorDict.data_from_dict(data, shape, device)

        if validate_args:
            self._tree_validate_shape(data)
            self._tree_validate_device(data)

    @classmethod
    def data_from_dict(cls, data, shape, device=None) -> Dict[str, TDCompatible]:
        result = {}
        for k, v in data.items():
            if isinstance(v, dict):
                result[k] = TensorDict(
                    TensorDict.data_from_dict(v, shape, device), shape, device
                )
            else:
                result[k] = v

        return result

    def _get_path_str(self, key_path):
        """Helper to construct path string from key_path, robust to torch.compile."""
        path_parts = []
        for k in key_path:
            if isinstance(k, tuple):  # Handle nested KeyPath tuples
                path_parts.append(self._get_path_str(k))
            elif hasattr(k, "key"):  # Access the 'key' attribute of the Key object
                path_parts.append(str(k.key))
            else:  # Fallback for unexpected elements
                path_parts.append(str(k))
        return ".".join(path_parts)

    def _tree_validate_shape(self, data):
        """
        Validates that the shapes of all nested tensors in the TensorDict start
        with the expected batch shape.

        This method recursively traverses the entire data structure.
        """
        keypath_leaf_pairs = pytree.tree_leaves_with_path(data)
        batch_shape = self.shape

        for key_path, leaf in keypath_leaf_pairs:
            path_str = self._get_path_str(key_path)

            if leaf.ndim > 0 and not self._is_shape_compatible(leaf.shape):
                raise ValueError(
                    f"Shape mismatch at '{path_str}': The tensor shape {leaf.shape} "
                    f"is not compatible with the TensorDict's batch shape {batch_shape}."
                )

    def _tree_validate_device(self, data):
        """
        Validates that the devices of all nested tensors in the TensorDict match
        the TensorDict's device if specified.
        """
        keypath_leaf_pairs = pytree.tree_leaves_with_path(data)

        for key_path, leaf in keypath_leaf_pairs:
            path_str = self._get_path_str(key_path)

            if not self._is_device_compatible(leaf.device):
                raise ValueError(
                    f"Device mismatch at '{path_str}': The tensor device {leaf.device} "
                    f"is not compatible with the TensorDict's device {self.device}."
                )

    def _get_pytree_context(
        self, flat_leaves: List[TDCompatible], children_spec: pytree.TreeSpec
    ) -> Tuple:
        """
        Private helper to compute the pytree context for this TensorDict.
        The context captures metadata to reconstruct the TensorDict:
        children_spec, event_ndims, original shape, and original device.
        """
        batch_ndim = len(self.shape)
        event_ndims = tuple(leaf.ndim - batch_ndim for leaf in flat_leaves)
        return (children_spec, event_ndims, self.shape, self.device)

    def _pytree_flatten(self) -> Tuple[List[TDCompatible], Tuple]:
        """
        Flattens the TensorDict into its tensor leaves and static metadata.
        """
        flat_leaves, children_spec = pytree.tree_flatten(self.data)
        context = self._get_pytree_context(flat_leaves, children_spec)
        return flat_leaves, context

    def _pytree_flatten_with_keys_fn(
        self,
    ) -> tuple[list[tuple[pytree.KeyEntry, Any]], Any]:
        """
        Flattens the TensorDict into key-path/leaf pairs and static metadata.
        """
        keypath_leaf_list, children_spec = pytree.tree_flatten_with_path(self.data)
        flat_leaves = [leaf for _, leaf in keypath_leaf_list]
        context = self._get_pytree_context(flat_leaves, children_spec)
        return keypath_leaf_list, context

    @classmethod
    def _pytree_unflatten(cls, leaves: List[Tensor], context: Tuple) -> TensorDict:
        """
        Reconstructs a TensorDict by creating a new instance and manually
        populating its attributes. This approach is more robust for torch.compile's
        code generation phase.
        """
        children_spec, event_ndims, shape_context, device_context = context

        obj = cls.__new__(cls)
        if not leaves:
            # Handle the empty case
            obj.data = {}
            obj.shape = shape_context  # Use shape from context
            obj.device = device_context  # For empty TD, use context device
            return obj

        first_leaf_device = leaves[0].device
        obj.device = first_leaf_device

        # Reconstruct the nested dictionary structure using the unflattened leaves
        data = pytree.tree_unflatten(leaves, children_spec)
        obj.data = data

        # Calculate new_shape based on the (potentially transformed) leaves and event_ndims from context.
        # This correctly determines the batch shape of the TensorDict after operations like stack/cat.
        # For copy(), where leaves are original, this also correctly yields the original shape.
        first_leaf_reconstructed = leaves[0]
        # event_ndims[0] is the event_ndim for the first leaf, relative to original batch shape.
        if (
            event_ndims[0] == 0
        ):  # Leaf was a scalar or had only batch dimensions originally
            reconstructed_shape = first_leaf_reconstructed.shape
        else:  # Leaf had event dimensions originally
            reconstructed_shape = first_leaf_reconstructed.shape[: -event_ndims[0]]

        obj.shape = reconstructed_shape

        return obj

    # --- Standard MutableMapping methods ---
    @overload
    def __getitem__(self, key: str) -> TDCompatible: ...

    @overload
    def __getitem__(self, key: slice) -> TensorDict: ...

    @overload
    def __getitem__(self, key: Tensor) -> TensorDict: ...

    def __getitem__(self, key: Any) -> TDCompatible:
        if isinstance(key, str):
            return self.data[key]

        return super().__getitem__(key)

    def __setitem__(self, key: str, value: TDCompatible):
        if not isinstance(value, (Tensor, TensorContainer)):
            raise ValueError(
                f"value must be a Tensor or TensorContainer, got value of type {type(value)}"
            )
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

    def update(self, other: Union[Dict[str, TDCompatible], TensorDict]):
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
        copied, but the leaves are the same. Implemented using _pytree_flatten and
        _pytree_unflatten for torch.compile compatibility.
        """
        flat_leaves, context = self._pytree_flatten()
        new_td = TensorDict._pytree_unflatten(flat_leaves, context)
        # Ensure device is preserved from the original TensorDict
        new_td.device = self.device
        return new_td

    def flatten_keys(self, separator: str = ".") -> TensorDict:
        """
        Returns a TensorDict with flattened keys.
        """
        out = {}

        def _flatten(data, prefix=""):
            if isinstance(data, TensorDict):
                for key, value in data.items():
                    new_prefix = prefix + key + separator
                    _flatten(value, new_prefix)
            else:
                out[prefix[:-1]] = data

        _flatten(self)

        return TensorDict(out, self.shape, self.device)

    def __repr__(self) -> str:
        # Infer device for representation if not set
        device_repr = self.device
        if device_repr is None and self.data:
            try:
                # Ensure there are leaves before trying to access device
                # pytree.tree_leaves can return an empty list
                leaves = pytree.tree_leaves(self.data)
                if leaves:
                    device_repr = leaves[0].device
            except IndexError:  # Should not happen if leaves is checked
                pass
            except Exception:  # Catch any other pytree or attribute errors
                pass

        def _format_item(key, value):
            if isinstance(value, TensorDict):
                return f"{key}: {value.__repr__()}"
            elif isinstance(value, Tensor):
                return f"{key}: Tensor(shape={value.shape}, dtype={value.dtype})"
            else:
                return f"{key}: {repr(value)}"

        items_str = ", ".join(_format_item(k, v) for k, v in self.data.items())
        return f"TensorDict(shape={self.shape}, device={device_repr}, {items_str})"
