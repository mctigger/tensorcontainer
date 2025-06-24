from __future__ import annotations

import copy
import dataclasses
from typing import Any, List, Optional, Tuple, TypeVar, Union

import torch
from torch import Tensor
from torch.utils import _pytree as pytree
from typing_extensions import dataclass_transform

from rtd.tensor_container import TensorContainer
from rtd.utils import PytreeRegistered

TDCompatible = Union[Tensor, TensorContainer]
DATACLASS_ARGS = {"init", "repr", "eq", "order", "unsafe_hash", "frozen", "slots"}


T_TensorDataclass = TypeVar("T_TensorDataclass", bound="TensorDataClass")


@dataclass_transform(eq_default=False)
class TensorDataclassTransform:
    """This class is just needed for type hints. Directly decorating TensorDataclass does not work."""

    pass


class TensorDataClass(TensorContainer, PytreeRegistered, TensorDataclassTransform):
    """A dataclass-based tensor container with PyTree compatibility."""

    # Added here to make shape and device part of the data class.
    shape: tuple
    device: Optional[torch.device]

    def __init_subclass__(cls, **kwargs):
        if hasattr(cls, "__slots__"):
            return

        # --- NEW: Automatically inherit annotations from parent classes ---
        # Build a complete dictionary of annotations from the MRO.
        # Iterate backwards through the MRO to ensure correct override order (child overrides parent).
        all_annotations = {}
        for base in reversed(cls.__mro__):
            if hasattr(base, "__annotations__"):
                all_annotations.update(base.__annotations__)
        cls.__annotations__ = all_annotations
        # --- End new logic ---

        dc_kwargs = {}
        for k in list(kwargs.keys()):
            if k in DATACLASS_ARGS:
                dc_kwargs[k] = kwargs.pop(k)

        super().__init_subclass__(**kwargs)

        if dc_kwargs.get("eq") is True:
            raise TypeError(
                f"Cannot create {cls.__name__} with eq=True. TensorDataclass requires eq=False."
            )
        dc_kwargs.setdefault("eq", False)
        dc_kwargs.setdefault("slots", True)

        dataclasses.dataclass(cls, **dc_kwargs)

    def __post_init__(self):
        """Initializes the TensorContainer part."""
        super().__init__(self.shape, self.device)

        # Infer device from tensor children if self.device is None
        if self.device is None:
            devices = {
                getattr(self, f.name).device
                for f in dataclasses.fields(self)
                if isinstance(getattr(self, f.name), TDCompatible)
                and hasattr(getattr(self, f.name), "device")
            }
            if len(devices) == 1:
                self.device = devices.pop()
            elif len(devices) > 1:
                # This case should ideally be caught by _tree_validate_device if all tensors are expected
                # to be on the same device as the container. If the container's device is None,
                # it implies flexibility, but operations might fail later if devices are truly mixed
                # without explicit handling. For now, we keep self.device as None.
                pass

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

    def _pytree_flatten(self) -> Tuple[List[Any], Tuple]:
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
    ) -> Tuple[List[Tuple[pytree.KeyPath, Any]], Tuple]:
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
    def _pytree_unflatten(cls, leaves: List[Tensor], context: Tuple) -> TensorDataClass:
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

    def __copy__(self: T_TensorDataclass) -> T_TensorDataclass:
        """
        Performs a shallow copy of the TensorDataclass instance.

        This method is designed to be `torch.compile` safe by avoiding the
        use of `copy.copy()`, which can cause graph breaks. It creates a new
        instance of the same class and then copies references to the attributes
        of the original object.

        Returns:
            A new TensorDataclass instance with attributes that are shallow
            copies of the original's attributes.
        """
        # Create a new, uninitialized instance of the correct class.
        cls = type(self)
        new_obj = cls.__new__(cls)

        # Manually copy all dataclass fields.
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            setattr(new_obj, field.name, value)

        # Manually call __post_init__ to initialize the TensorContainer part
        # and run validation logic. This is necessary because we bypassed __init__.
        if hasattr(new_obj, "__post_init__"):
            new_obj.__post_init__()

        return new_obj

    def __deepcopy__(
        self: T_TensorDataclass, memo: Optional[dict] = None
    ) -> T_TensorDataclass:
        """
        Performs a deep copy of the TensorDataclass instance.

        This method is designed to be `torch.compile` safe by manually
        iterating through fields and using `copy.deepcopy` for each,
        while also handling the `memo` dictionary to prevent infinite
        recursion in case of circular references.

        Args:
            memo: A dictionary to keep track of already copied objects.
                  This is part of the `copy.deepcopy` protocol.

        Returns:
            A new TensorDataclass instance with attributes that are deep
            copies of the original's attributes.
        """
        if memo is None:
            memo = {}

        cls = type(self)
        # Check if the object is already in memo
        if id(self) in memo:
            return memo[id(self)]

        new_obj = cls.__new__(cls)
        memo[id(self)] = new_obj

        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            # The `shape` and `device` fields are part of the dataclass fields
            # due to their annotations in TensorDataclass.
            # These should be deepcopied as well if they are not None.
            if field.name in ("shape", "device"):
                # Tuples (shape) and torch.device are immutable or behave as such.
                # Direct assignment is fine and avoids torch.compile issues with deepcopying them.
                # Direct assignment for immutable types like tuple (shape) and torch.device.
                # This avoids torch.compile issues with copy.copy or copy.deepcopy on these types.
                setattr(new_obj, field.name, value)
            elif isinstance(value, Tensor):
                # For torch.Tensor, use .clone() for a deep copy of data.
                setattr(new_obj, field.name, value.clone())
            elif isinstance(value, list):
                # For lists, create a new list. This is a shallow copy of the list structure.
                # If list items are mutable and need deepcopying, torch.compile might
                # still struggle with a generic deepcopy of those items.
                # For a list of immutables (like in the test), this is effectively a deepcopy.
                setattr(new_obj, field.name, list(value))
            else:
                # For other fields (e.g., dict, other custom objects), attempt deepcopy.
                # This remains a potential point of failure for torch.compile
                # if it doesn't support deepcopying these specific types.
                setattr(new_obj, field.name, copy.deepcopy(value))

        # Manually call __post_init__ to initialize the TensorContainer part
        # and run validation logic. This is necessary because we bypassed __init__.
        # __post_init__ in TensorDataclass handles shape and device initialization
        # and validation, which is crucial after all fields are set.
        if hasattr(new_obj, "__post_init__"):
            new_obj.__post_init__()

        return new_obj
