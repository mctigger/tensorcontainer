from __future__ import annotations

from typing import Any, Dict, Iterable, List, Self, Tuple, TypeVar, Union, get_args

import torch
from torch import Tensor
from torch.utils import _pytree as pytree

from tensorcontainer.tensor_container import TensorContainer
from tensorcontainer.utils import PytreeRegistered

TDCompatible = Union[Tensor, TensorContainer]
DATACLASS_ARGS = {"init", "repr", "eq", "order", "unsafe_hash", "frozen", "slots"}


T_TensorAnnotated = TypeVar("T_TensorAnnotated", bound="TensorAnnotated")


class TensorAnnotated(TensorContainer, PytreeRegistered):
    def __init__(
        self,
        shape: torch.Size | List[int] | Tuple[int],
        device: str | torch.device | int | None
    ):
        super().__init__(shape, device, True)

    def _get_tensor_attributes(self):
        # In Python 3.9 __annotations__ also includes parent class
        # annotations, which is regarded a bug and changed from Python 3.10+
        # We use the following line to be backwards compatible for 3.9
        # In Python 3.10+ we could simply use cls.__annotations__.
        annotations = type(self).__dict__.get("__annotations__", {})

        tensor_attributes = {
            k: getattr(self, k)
            for k, v in annotations.items()
            if isinstance(getattr(self, k), get_args(TDCompatible))
        }

        return tensor_attributes

    def _get_meta_attributes(self):
        # In Python 3.9 __annotations__ also includes parent class
        # annotations, which is regarded a bug and changed from Python 3.10+
        # We use the following line to be backwards compatible for 3.9
        # In Python 3.10+ we could simply use cls.__annotations__.
        annotations = type(self).__dict__.get("__annotations__", {})

        meta_attributes = {
            k: getattr(self, k)
            for k, v in annotations.items()
            if not isinstance(getattr(self, k), get_args(TDCompatible))
        }

        return meta_attributes

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

    def _get_pytree_context(
        self, flat_names: List[str], flat_leaves: List[TDCompatible], meta_data
    ) -> Tuple:
        batch_ndim = len(self.shape)
        event_ndims = tuple(leaf.ndim - batch_ndim for leaf in flat_leaves)

        return flat_names, event_ndims, meta_data, self.device

    def _pytree_flatten(self) -> Tuple[List[Any], Any]:
        tensor_attributes = self._get_tensor_attributes()
        flat_names = list(tensor_attributes.keys())
        flat_values = list(tensor_attributes.values())

        meta_data = self._get_meta_attributes()

        context = self._get_pytree_context(flat_names, flat_values, meta_data)

        return flat_values, context

    def _pytree_flatten_with_keys_fn(
        self,
    ) -> tuple[list[tuple[pytree.KeyEntry, Any]], Any]:
        flat_values, context = self._pytree_flatten()
        flat_names = context[0]
        name_value_tuples = [
            (pytree.GetAttrKey(k), v) for k, v in zip(flat_names, flat_values)
        ]
        return name_value_tuples, context  # type: ignore[return-value]

    @classmethod
    def _pytree_unflatten(cls, leaves: Iterable[Any], context: pytree.Context) -> Self:
        flat_names, event_ndims, meta_data, device = context

        leaves = list(leaves)  # Convert to list to allow indexing

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

        return cls._init_from_reconstructed(
            dict(zip(flat_names, leaves)),
            {k: v for k, v in meta_data.items() if k not in ["device", "shape"]},
            device,
            reconstructed_shape,
        )

    @classmethod
    def _init_from_reconstructed(
        cls,
        tensor_attributes: Dict[str, TDCompatible],
        meta_attributes: Dict[str, Any],
        device,
        shape,
    ):
        return cls(**tensor_attributes, **meta_attributes, device=device, shape=shape)
