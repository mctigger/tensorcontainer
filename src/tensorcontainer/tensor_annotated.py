from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union, get_args
from collections.abc import Iterable

from torch import Tensor
from torch.utils import _pytree as pytree
from typing_extensions import Self

from tensorcontainer.tensor_container import (
    TensorContainer,
    TensorContainerPytreeContext,
)
from tensorcontainer.types import DeviceLike, ShapeLike
from tensorcontainer.utils import PytreeRegistered

"""Annotation-based tensor containers with automatic PyTree integration.

This module provides TensorAnnotated, a base class that uses type annotations to
automatically separate tensor fields from metadata during PyTree operations. This
enables seamless batching and transformation of complex data structures without
manual field management.

Key Components:
    TensorAnnotatedPytreeContext: Enhanced PyTree context with field-aware error analysis
    TensorAnnotated: Base class for annotation-driven tensor containers

The TensorAnnotated class serves as an intermediate layer between TensorContainer
(which provides batch/device semantics) and TensorDataClass (which provides dataclass
interface). Most users should prefer TensorDataClass for typical applications.
"""

TDCompatible = Union[Tensor, TensorContainer]


# PyTree context metadata for reconstruction
@dataclass
class TensorAnnotatedPytreeContext(
    TensorContainerPytreeContext["TensorAnnotatedPytreeContext"]
):
    """Enhanced PyTree context for TensorAnnotated with field-aware error analysis.

    Extends TensorContainerPytreeContext to add field name tracking, event dimension
    metadata, and enhanced mismatch analysis for annotation-based containers.

    Attributes:
        keys (list[str]): Names of tensor fields flattened into the PyTree.
        event_ndims (list[int]): Event dimensions for each field, used for shape reconstruction.
        metadata (dict[str, Any]): Non-tensor attributes preserved through PyTree operations.
        device (torch.device | None): Inherited from base class.
    """

    keys: list[str]
    event_ndims: list[int]
    metadata: dict[str, Any]

    def __str__(self) -> str:
        """Return human-readable description of this TensorAnnotated context."""
        # Try to get the actual class name from metadata
        class_name = self.metadata.get("class_name", "TensorAnnotated")

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
    """Base class for annotation-driven tensor containers with automatic field separation.

    Uses type annotations to automatically separate tensor fields (which become PyTree leaves)
    from metadata fields (preserved in PyTree context) during transformations. Extends
    TensorContainer with annotation-based field management.

    Example:
        ```python
        class MyContainer(TensorAnnotated):
            observations: torch.Tensor    # PyTree leaf
            actions: torch.Tensor         # PyTree leaf
            episode_id: int               # Metadata
            config: dict                  # Metadata
        ```

    Args:
        shape: Batch shape for tensor field validation (see TensorContainer).
        device: Target device for tensors (see TensorContainer).

    Note:
        Field classification as tensor or metadata depends on the runtime value type,
        not the type annotation. Annotations only indicate which fields TensorAnnotated
        should consider for tensor operations.
    """

    def __init__(self, shape: ShapeLike, device: DeviceLike | None):
        """Initialize TensorAnnotated container with batch shape and device constraints.

        Args:
            shape: Batch shape that all tensor fields must share as leading dimensions.
            device: Target device for tensors. None allows mixed devices.
        """
        super().__init__(shape, device)

    @classmethod
    def _get_annotations(cls, base_cls):
        """Collect field annotations from inheritance hierarchy, filtering out non-tensor mixins.

        Problem solved: When using multiple inheritance or mixins with TensorAnnotated classes,
        Python's standard annotation collection includes annotations from ALL parent classes,
        including unrelated mixins that may define fields incompatible with tensor operations.
        This method ensures only tensor-relevant annotations are collected by filtering the MRO
        to include only subclasses of base_cls, preventing annotation pollution that could cause
        incorrect field classification during PyTree operations.

        Example:
            ```python
            class MyMixin:
                helper_field: str  # Should NOT be collected

            class MyClass(TensorAnnotated, MyMixin):
                observations: torch.Tensor  # Should be collected
                actions: torch.Tensor       # Should be collected

            # MyClass._get_annotations(TensorAnnotated) returns:
            # {'observations': torch.Tensor, 'actions': torch.Tensor}
            # Note: 'helper_field' is excluded because MyMixin doesn't inherit from TensorAnnotated
            ```

        Walks the MRO to collect annotations only from subclasses of base_cls, preventing
        annotation pollution from unrelated mixins. Protects reserved 'shape'/'device' fields.

        Args:
            base_cls: Only collect annotations from subclasses of this class.

        Returns:
            dict: Field name to annotation mapping from inheritance hierarchy.

        Raises:
            TypeError: If 'shape' or 'device' are defined (reserved by TensorContainer).
        """
        annotations = {}

        # Start from current class and walk up MRO, excluding base_cls and its parent classes
        # This prevents collecting annotations from TensorContainer, PytreeRegistered, etc.
        mro = list(reversed(cls.__mro__))
        mro_excluding_tensor_base = mro[mro.index(base_cls) + 1 :]

        for base in mro_excluding_tensor_base:
            # Use __dict__.get() for Python 3.9 compatibility - avoids bug where
            # __annotations__ included parent class annotations automatically
            base_annotations = base.__dict__.get("__annotations__", {})

            # Only collect from classes that inherit from base_cls (filters out unrelated mixins)
            if issubclass(base, base_cls):
                # Remove reserved fields managed by TensorContainer
                base_annotations = {
                    k: annotation_type
                    for k, annotation_type in base_annotations.items()
                    if k not in ["device", "shape"]
                }

            annotations.update(base_annotations)

        # Protect TensorContainer's core fields from being overridden
        if "shape" in annotations or "device" in annotations:
            raise TypeError(f"Cannot define reserved fields in {cls.__name__}.")

        return annotations

    def _get_tensor_attributes(self):
        """Identify annotated fields containing tensor-compatible values for PyTree leaves.

        Uses runtime type checking on field values (not annotations) to handle cases
        like Optional fields that are None or Union types with mixed values.

        Returns:
            dict[str, TDCompatible]: Annotated fields containing tensor-compatible values.
        """
        annotations = self._get_annotations(TensorAnnotated)

        # Filter annotated fields to only include tensor-compatible runtime values
        # This uses isinstance() check on actual values, not type annotations
        tensor_attributes = {
            k: getattr(self, k)
            for k, _ in annotations.items()
            if isinstance(getattr(self, k), get_args(TDCompatible))
        }

        return tensor_attributes

    def _get_meta_attributes(self):
        """Identify annotated fields containing non-tensor values for metadata preservation.

        Uses runtime type checking to find annotated fields that do NOT contain
        tensor-compatible values. These are preserved in PyTree context during transformations.

        Returns:
            dict[str, Any]: Annotated fields containing non-tensor values.
        """
        annotations = self._get_annotations(TensorAnnotated)

        # Filter annotated fields to only include non-tensor runtime values
        # These become metadata preserved in PyTree context
        meta_attributes = {
            k: getattr(self, k)
            for k, _ in annotations.items()
            if not isinstance(getattr(self, k), get_args(TDCompatible))
        }

        return meta_attributes

    def _get_pytree_context(
        self, flat_names: list[str], flat_leaves: list[TDCompatible], meta_data
    ) -> TensorAnnotatedPytreeContext:
        """Create PyTree context with event dimension info for proper reconstruction.

        Calculates event dimensions (tensor.ndim - batch_ndim) for each tensor field
        to enable correct shape inference after PyTree transformations.

        Args:
            flat_names: Tensor field names in flattening order.
            flat_leaves: Tensor values in flattening order.
            meta_data: Non-tensor attributes to preserve.

        Returns:
            TensorAnnotatedPytreeContext: Context for reconstruction with event dimension info.
        """
        batch_ndim = len(self.shape)
        # Calculate event dimensions for each tensor field
        # This enables proper shape reconstruction after PyTree transformations
        event_ndims = [leaf.ndim - batch_ndim for leaf in flat_leaves]

        return TensorAnnotatedPytreeContext(
            self.device, flat_names, event_ndims, meta_data
        )

    def _pytree_flatten(self) -> tuple[list[Any], Any]:
        """Flatten container into tensor leaves and metadata context for PyTree operations.

        Separates tensor-compatible fields (become leaves) from metadata fields (preserved
        in context) using runtime type checking. Creates enhanced context with field names,
        event dimensions, and metadata for accurate reconstruction.

        Returns:
            tuple[list[Any], Any]: (tensor_values, context_with_metadata)
        """
        # Separate tensor and non-tensor attributes based on runtime type checking
        tensor_attributes = self._get_tensor_attributes()
        flat_names = list(tensor_attributes.keys())
        flat_values = list(tensor_attributes.values())

        # Preserve non-tensor attributes as metadata
        meta_data = self._get_meta_attributes()

        # Create enhanced context with field names and event dimension information
        context = self._get_pytree_context(flat_names, flat_values, meta_data)

        return flat_values, context

    def _pytree_flatten_with_keys_fn(
        self,
    ) -> tuple[list[tuple[pytree.KeyEntry, Any]], Any]:
        """Flatten with key paths for PyTorch operations that need field access tracking.

        Extends standard flattening by wrapping each tensor value with a GetAttrKey
        that identifies the field name, enabling PyTorch to track paths during operations.

        Returns:
            tuple[list[tuple[KeyEntry, Any]], Any]: (key_value_pairs, context)
        """
        # Start with standard flattening to get values and context
        flat_values, context = self._pytree_flatten()
        flat_names = context.keys

        # Create key-value pairs with GetAttrKey for PyTorch's path tracking
        name_value_tuples = [
            (pytree.GetAttrKey(k), v) for k, v in zip(flat_names, flat_values)
        ]
        return name_value_tuples, context  # type: ignore[return-value]

    @classmethod
    def _pytree_unflatten(
        cls, leaves: Iterable[Any], context: TensorAnnotatedPytreeContext
    ) -> Self:
        """Reconstruct container from transformed leaves with automatic batch shape inference.

        Uses stored event dimension counts to separate batch from event dimensions in
        transformed tensors, enabling automatic shape inference after PyTree operations.

        Args:
            leaves: Transformed tensor values from PyTree operation.
            context: Contains field names, event dimensions, device, and metadata.

        Returns:
            Self: Reconstructed container with inferred batch shape and restored fields.
        """
        flat_names = context.keys
        event_ndims = context.event_ndims
        device = context.device
        meta_data = context.metadata

        leaves = list(leaves)  # Convert to list to allow indexing

        # Infer new batch shape by analyzing the first transformed tensor
        # Uses stored event dimensions to separate batch dims from event dims
        first_leaf_reconstructed = leaves[0]
        first_leaf_event_ndims = event_ndims[0]

        if first_leaf_event_ndims == 0:
            # Original tensor had only batch dimensions - entire shape is batch shape
            reconstructed_shape = first_leaf_reconstructed.shape
        else:
            # Original tensor had event dimensions - remove them to get batch shape
            # E.g., tensor (2, 4, 3, 128) with event_ndims=1 -> batch shape (2, 4, 3)
            reconstructed_shape = first_leaf_reconstructed.shape[
                :-first_leaf_event_ndims
            ]

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
        """Extension point for subclasses to customize PyTree reconstruction logic.

        Combines tensor fields, metadata, device, and shape to create a new instance.
        Subclasses can override this to handle special reconstruction requirements.

        Args:
            tensor_attributes: Field names to tensor values from PyTree leaves.
            meta_attributes: Field names to metadata values from context.
            device: Target device for the container.
            shape: Inferred batch shape.

        Returns:
            Self: New instance with all fields and properties restored.
        """
        return cls(**tensor_attributes, **meta_attributes, device=device, shape=shape)
