"""Protocols for TensorContainer type safety and mixin constraints.

This module defines the protocol interfaces that TensorContainer mixins depend on,
providing static type safety and runtime verification capabilities.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable, ContextManager
from typing_extensions import Self

import torch
from torch.utils._pytree import Context


@runtime_checkable
class TensorContainerProtocol(Protocol):
    """Protocol defining the interface that TensorContainer mixins require.

    This protocol captures the essential methods and properties that mixins
    expect from TensorContainer implementations. Using this protocol ensures
    that mixins can only be applied to classes that provide the required
    TensorContainer interface.

    The protocol is marked with @runtime_checkable to enable isinstance()
    checks at runtime for additional type safety.
    """

    # Core container properties
    shape: torch.Size
    device: torch.device | None

    @property
    def ndim(self) -> int: ...

    # PyTree interface methods
    def _pytree_flatten(self) -> tuple[list[Any], Context]:
        """Flatten the container into a list of leaves and context."""
        ...

    def _pytree_flatten_with_keys_fn(self) -> tuple[list[tuple[Any, Any]], Any]:
        """Flatten the container with keys."""
        ...

    @classmethod
    def _pytree_unflatten(cls, leaves: Any, context: Context) -> Self:
        """Reconstruct the container from leaves and context."""
        ...

    @classmethod
    def _tree_map(
        cls,
        func: Callable[..., Any],
        tree: Any,
        *rests: Any,
        is_leaf: Callable[[Any], bool] | None = None,
    ) -> Self:
        """Apply a function to all leaves in the container tree."""
        ...

    # Core validation methods
    def _validate_shape(self, value) -> None:
        """Validate that a tensor has compatible shape."""
        ...

    def _validate_device(self, value) -> None:
        """Validate that a tensor has compatible device."""
        ...

    def _validate(self) -> None:
        """Validate the entire container state."""
        ...

    # Essential utility methods used by mixins
    @classmethod
    def unsafe_construction(cls) -> ContextManager[None]:
        """Context manager for unsafe construction operations."""
        ...

    @classmethod
    def tree_map_with_path(
        cls,
        func: Callable[..., Any],
        tree: Any,
        *rests: Any,
        is_leaf: Callable[[Any], bool] | None = None,
    ) -> Self:
        """Apply a function with path to all leaves in the container tree."""
        ...

    @classmethod
    def _is_shape_compatible(cls, parent: Any, child: Any) -> bool:
        """Check if child tensor shape is compatible with parent container."""
        ...

    @classmethod
    def _is_device_compatible(cls, parent: Any, child: Any) -> bool:
        """Check if child tensor device is compatible with parent container."""
        ...

    # Methods expected by device operations
    def to(self, *args, **kwargs) -> Self:
        """Move and/or cast the container to a device or dtype."""
        ...

    def clone(self, *, memory_format: torch.memory_format | None = None) -> Self:
        """Create a deep copy of the container."""
        ...
