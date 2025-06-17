from __future__ import annotations

import dataclasses
from typing import Optional, Union

from torch import Tensor

from rtd.tensor_container import TensorContainer
from rtd.utils import PytreeRegistered
from torch.utils import _pytree as pytree
from typing import Tuple, List
import torch

TDCompatible = Union[Tensor, TensorContainer]


@dataclasses.dataclass(eq=False)
class TensorDataclass(TensorContainer, PytreeRegistered):
    """A dataclass-based tensor container with PyTree compatibility."""

    shape: tuple
    device: Optional[torch.device]

    def __post_init__(self):
        """Initializes the TensorContainer part."""
        super().__init__(self.shape, self.device)

        self._tree_validate_device()
        self._tree_validate_shape()

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

    def _tree_validate_shape(self):
        keypath_leaf_pairs = pytree.tree_leaves_with_path(self)
        batch_shape = self.shape

        for key_path, leaf in keypath_leaf_pairs:
            path_str = self._get_path_str(key_path)

            if leaf.ndim > 0 and not self._is_shape_compatible(leaf.shape):
                raise ValueError(
                    f"Shape mismatch at '{path_str}': The tensor shape {leaf.shape} "
                    f"is not compatible with the TensorDataclass's batch shape {batch_shape}."
                )

    def _tree_validate_device(self):
        keypath_leaf_pairs = pytree.tree_leaves_with_path(self)

        for key_path, leaf in keypath_leaf_pairs:
            path_str = self._get_path_str(key_path)

            if not self._is_device_compatible(leaf.device):
                raise ValueError(
                    f"Device mismatch at '{path_str}': The tensor device {leaf.device} "
                    f"is not compatible with the TensorDataclass's device {self.device}."
                )

    def _get_pytree_context(
        self, flat_leaves: List[TDCompatible], children_spec: pytree.TreeSpec, meta_data
    ) -> Tuple:
        """
        Private helper to compute the pytree context for this TensorDict.
        The context captures metadata to reconstruct the TensorDict:
        children_spec, event_ndims, original shape, and original device.
        """
        batch_ndim = len(self.shape)
        event_ndims = tuple(leaf.ndim - batch_ndim for leaf in flat_leaves)

        return (children_spec, event_ndims, self.shape, self.device, meta_data)

    def _pytree_flatten(self) -> Tuple[List[TDCompatible], Tuple]:
        """
        Flattens the TensorDict into its tensor leaves and static metadata.
        """
        data = {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if isinstance(getattr(self, f.name), TDCompatible)
        }
        meta_data = {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if not isinstance(getattr(self, f.name), TDCompatible)
            and f.name not in ["shape", "device"]
        }
        flat_leaves, children_spec = pytree.tree_flatten(data)
        context = self._get_pytree_context(flat_leaves, children_spec, meta_data)
        return flat_leaves, context

    def _pytree_flatten_with_keys_fn(
        self,
    ) -> Tuple[List[Tuple[pytree.KeyPath, TDCompatible]], Tuple]:
        """
        Flattens the TensorDict into key-path/leaf pairs and static metadata.
        """
        data = {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if isinstance(getattr(self, f.name), TDCompatible)
        }
        meta_data = {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if not isinstance(getattr(self, f.name), TDCompatible)
            and f.name not in ["shape", "device"]
        }
        keypath_leaf_list, children_spec = pytree.tree_flatten_with_path(data)
        flat_leaves = [leaf for _, leaf in keypath_leaf_list]
        context = self._get_pytree_context(flat_leaves, children_spec, meta_data)

        return keypath_leaf_list, context

    @classmethod
    def _pytree_unflatten(cls, leaves: List[Tensor], context: Tuple) -> TensorDataclass:
        """Unflattens component values into a dataclass instance."""
        (
            children_spec,
            event_ndims,
            shape,
            device,
            meta_data,
        ) = context

        if not leaves:
            return cls(
                **meta_data,
                device=device,
                shape=shape,
            )

        reconstructed_device = leaves[0].device

        # Reconstruct the nested dictionary structure using the unflattened leaves
        data = pytree.tree_unflatten(leaves, children_spec)

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

        return cls(
            **data,
            **meta_data,
            device=reconstructed_device,
            shape=reconstructed_shape,
        )

    def __repr__(self) -> str:
        """Provides a developer-friendly representation of the TensorDataclass."""
        items = []
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if isinstance(value, Tensor):
                item_repr = (
                    f"Tensor(shape={value.shape}, dtype={value.dtype}, "
                    f"device={value.device})"
                )
            else:
                item_repr = repr(value)
            items.append(f"{f.name}={item_repr}")

        items_str = ",\n    ".join(items)
        return (
            f"{self.__class__.__name__}(\n"
            f"    shape={self.shape},\n"
            f"    device={self.device},\n"
            f"    fields=(\n        {items_str}\n    )\n)"
        )
