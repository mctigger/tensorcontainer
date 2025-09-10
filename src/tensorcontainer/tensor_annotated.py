from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, TypeVar, Union, get_args

from torch import Tensor
from torch.utils import _pytree as pytree
from typing_extensions import Self

from tensorcontainer.tensor_container import (
    TensorContainer,
    TensorContainerPytreeContext,
)
from tensorcontainer.types import DeviceLike, ShapeLike
from tensorcontainer.utils import PytreeRegistered

TDCompatible = Union[Tensor, TensorContainer]
DATACLASS_ARGS = {"init", "repr", "eq", "order", "unsafe_hash", "frozen", "slots"}


T_TensorAnnotated = TypeVar("T_TensorAnnotated", bound="TensorAnnotated")


# PyTree context metadata for reconstruction
@dataclass
class TensorAnnotatedPytreeContext(
    TensorContainerPytreeContext["TensorAnnotatedPytreeContext"]
):
    """TensorAnnotated PyTree context with enhanced error messages."""

    keys: list[str]
    event_ndims: list[int]
    metadata: dict[str, Any]

    def __str__(self) -> str:
        """Return human-readable description of this TensorAnnotated context."""
        # Try to get the actual class name from metadata
        class_name = self.metadata.get("class_name", "TensorDataClass")

        fields_str = f"fields={self.keys}"
        device_str = f"device={self.device}"

        return f"{class_name}({fields_str}, {device_str})"

    def analyze_mismatch_with(
        self, other: TensorAnnotatedPytreeContext, entry_index: int
    ) -> str:
        """Analyze specific mismatches between TensorAnnotated contexts."""
        # Start with base class analysis (device mismatch, if any)
        guidance = super().analyze_mismatch_with(other, entry_index)

        # Add TensorAnnotated-specific analysis
        self_fields = set(self.keys)
        other_fields = set(other.keys)

        if self_fields != other_fields:
            missing = self_fields - other_fields
            extra = other_fields - self_fields
            guidance += "Field mismatch detected."
            if missing:
                guidance += (
                    f" Missing fields in container {entry_index}: {sorted(missing)}."
                )
            if extra:
                guidance += (
                    f" Extra fields in container {entry_index}: {sorted(extra)}."
                )

        return guidance


class TensorAnnotated(TensorContainer, PytreeRegistered):
    def __init__(self, shape: ShapeLike, device: DeviceLike | None):
        super().__init__(shape, device)

    @classmethod
    def _get_annotations(cls, base_cls):
        annotations = {}

        # We collect annotations from all parent classes in MRO that are subclass of the base_cls.
        # This avoid collection annotations from TensorAnnotated (or any other class passed as base_cls)
        # parent classes, i.e. TensorContainer or PytreeRegistered.
        mro = list(reversed(cls.__mro__))
        mro_excluding_tensor_base = mro[mro.index(base_cls) + 1 :]
        for base in mro_excluding_tensor_base:
            # In Python 3.9 __annotations__ also includes parent class
            # annotations, which is regarded a bug and changed from Python 3.10+
            base_annotations = base.__dict__.get("__annotations__", {})

            if issubclass(base, base_cls):
                base_annotations = {
                    k: v
                    for k, v in base_annotations.items()
                    if k not in ["device", "shape"]
                }

            annotations.update(base_annotations)

        if "shape" in annotations or "device" in annotations:
            raise TypeError(f"Cannot define reserved fields in {cls.__name__}.")

        return annotations

    def _get_tensor_attributes(self):
        annotations = self._get_annotations(TensorAnnotated)

        tensor_attributes = {
            k: getattr(self, k)
            for k, v in annotations.items()
            if isinstance(getattr(self, k), get_args(TDCompatible))
        }

        return tensor_attributes

    def _get_meta_attributes(self):
        annotations = self._get_annotations(TensorAnnotated)

        meta_attributes = {
            k: getattr(self, k)
            for k, v in annotations.items()
            if not isinstance(getattr(self, k), get_args(TDCompatible))
        }

        return meta_attributes

    def _get_pytree_context(
        self, flat_names: list[str], flat_leaves: list[TDCompatible], meta_data
    ) -> TensorAnnotatedPytreeContext:
        batch_ndim = len(self.shape)
        event_ndims = [leaf.ndim - batch_ndim for leaf in flat_leaves]

        return TensorAnnotatedPytreeContext(
            self.device, flat_names, event_ndims, meta_data
        )

    def _pytree_flatten(self) -> tuple[list[Any], Any]:
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
        flat_names = context.keys
        name_value_tuples = [
            (pytree.GetAttrKey(k), v) for k, v in zip(flat_names, flat_values)
        ]
        return name_value_tuples, context  # type: ignore[return-value]

    @classmethod
    def _pytree_unflatten(
        cls, leaves: Iterable[Any], context: TensorAnnotatedPytreeContext
    ) -> Self:
        flat_names = context.keys
        event_ndims = context.event_ndims
        device = context.device
        meta_data = context.metadata

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
        tensor_attributes: dict[str, TDCompatible],
        meta_attributes: dict[str, Any],
        device,
        shape,
    ):
        return cls(**tensor_attributes, **meta_attributes, device=device, shape=shape)
