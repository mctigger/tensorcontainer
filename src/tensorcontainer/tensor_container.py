from __future__ import annotations

from dataclasses import dataclass
import functools
import textwrap
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    Generic,
    TypeVar,
    Union,
)
from collections.abc import Iterable

import torch

import torch.utils._pytree as pytree
from torch import Tensor
from torch.utils._pytree import Context, KeyEntry, PyTree
from typing_extensions import Self, TypeAlias

from tensorcontainer.protocols import TensorContainerProtocol
from tensorcontainer.types import DeviceLike, IndexType, ShapeLike
from tensorcontainer.utils import (
    ContextWithAnalysis,
    diagnose_pytree_structure_mismatch,
    resolve_device,
    format_path,
)

# Global registry
HANDLED_FUNCTIONS = {}

TCCompatible: TypeAlias = Union[torch.Tensor, "TensorContainer"]


def implements(torch_function):
    """Register a torch function override for TensorContainer."""

    @functools.wraps(torch_function)
    def decorator(func):
        HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


U = TypeVar("U", bound="TensorContainerPytreeContext")


@dataclass
class TensorContainerPytreeContext(ContextWithAnalysis[U], Generic[U], ABC):
    """Base PyTree context class for tensor containers with common device handling."""

    device: torch.device | None

    def analyze_mismatch_with(self, other: U, entry_index: int) -> str:
        """Analyze mismatches with another TensorContainerPytreeContext, starting with device analysis."""
        # Check device mismatch first
        if self.device != other.device:
            return f"Device mismatch: container 0 device={self.device}, container {entry_index} device={other.device}. "

        # If devices match, return empty string for subclasses to add their analysis
        return ""


class TensorContainer(TensorContainerProtocol):
    """A foundational base class for PyTree-compatible tensor containers with batch semantics.

    TensorContainer provides a structured way to organize tensors that share common batch dimensions
    while allowing flexible event dimensions. It serves as the foundation for concrete implementations
    like TensorDict (dictionary-style) and TensorDataClass (dataclass-style).

    ## Core Concepts

    ### Shape Management
    TensorContainer enforces a clear distinction between batch and event dimensions:

    - **Batch Dimensions**: The leading dimensions defined by the `shape` parameter that must be
      consistent across all tensors in the container. These represent the batching structure
      (e.g., batch size, sequence length).

    - **Event Dimensions**: The trailing dimensions beyond the batch shape that can vary between
      different tensors in the container. These represent the actual data structure
      (e.g., feature dimensions, action spaces).

    Example:
        >>> # Container with batch shape (4, 3) - 4 samples, 3 time steps
        >>> container.shape == (4, 3)
        >>>
        >>> # Valid tensors within this container:
        >>> observations = torch.randn(4, 3, 128)    # Event dims: (128,)
        >>> actions = torch.randn(4, 3, 6)           # Event dims: (6,)
        >>> rewards = torch.randn(4, 3)              # Event dims: ()
        >>>
        >>> # All share batch dims (4, 3), different event dims allowed

    ### Device Management
    Device consistency is enforced through flexible compatibility rules:
    - If container device is None, any tensor device is accepted
    - If container device is not None, only tensors of the same device are accepted
    - If an operation changes the device of the container, it must also change the device of all children

    ### Metadata
    Data in a TensorContainer falls into two categories: tensor-likes and metadata.

    Tensor-likes participate in all transformations such as .view, .reshape, or .detach.
    Metadata does not participate. Metadata must remain constant during a transformation.
    For operations that include multiple TensorContainers (such as torch.stack), the
    metadata of all TensorContainers must be identical.

    ## Implementation details
    
    ### PyTree Integration

    TensorContainer is implemented as a PyTree for the following reasons:
    - PyTrees are `torch.compile` compatible
    - Efficient tree transformations using `torch.utils._pytree.tree_map`

    Treating TensorContainer as a PyTree enables straightforward implementation of operations that
    work on all children of a TensorContainer. Without torch.utils._pytree, we would need to
    implement equivalent tree traversal and transformation functionality from scratch.

    ### Torch Function Override System

    TensorContainer leverages the `__torch_function__` protocol to enable tensor-like behavior.
    See https://docs.pytorch.org/docs/stable/notes/extending.html#extending-torch-python-api.

    The `__torch_function__` protocol is used to intercept torch operations:
    - Register custom implementations via `@implements(torch_function)` decorator
    - Maintains compatibility with PyTorch's dispatch system
    - Enables container-aware versions of functions like `torch.stack`, `torch.cat`

    ## Subclassing Guide

    When creating TensorContainer subclasses:

    1. **Inherit from TensorContainer and PytreeRegistered**:
       ```python
       class MyContainer(TensorContainer, PytreeRegistered):
       ```
    2. **Implement PyTree methods**:
       - `_pytree_flatten()` - Convert to (leaves, context)
       - `_pytree_unflatten()` - Reconstruct from leaves and context
       - `_pytree_flatten_with_keys_fn()` - Provide key paths
    3. **Call super().__init__(shape, device)** in constructor
    4. **Override validation** if needed via `_is_shape_compatible()`, `_is_device_compatible()`
    5. **Register torch functions** using `@implements(torch_function)` decorator


    Args:
        shape (Tuple[int, ...]): The batch shape that all contained tensors must share
            as their leading dimensions. Defines the batching structure.
        device (Optional[Union[str, torch.device]]): The device all tensors should reside on.
            If None, no device consistency is enforced.

    Raises:
        ValueError: If tensor shapes are incompatible with the specified batch shape
        ValueError: If tensor devices are incompatible with the specified device
        IndexError: For invalid indexing operations (e.g., too many indices)
        RuntimeError: For invalid shape transformations or other tensor operations

    Example:
        >>> # Create a simple subclass for demonstration
        >>> class SimpleContainer(TensorContainer):
        ...     def __init__(self, data, shape, device=None):
        ...         super().__init__(shape, device)
        ...         self.data = data
        >>>
        >>> # Usage with batch shape (2, 3)
        >>> container = SimpleContainer({
        ...     'obs': torch.randn(2, 3, 64),  # Event dims: (64,)
        ...     'action': torch.randn(2, 3, 4) # Event dims: (4,)
        ... }, shape=(2, 3))
        >>>
        >>> # Batch operations preserve event structure
        >>> flattened = container.reshape(6)     # Shape becomes (6,), events preserved
        >>> first_batch = container[0]           # Shape becomes (3,), events preserved
    """

    shape: torch.Size
    device: torch.device | None

    # Thread-local storage for unsafe construction flag
    _validation_disabled = threading.local()

    def __init__(
        self,
        shape: ShapeLike,
        device: DeviceLike | None,
    ):
        super().__init__()

        self.shape = torch.Size(shape)
        self.device = None if device is None else resolve_device(device)

        self._validate()

    @classmethod
    @contextmanager
    def unsafe_construction(cls):
        """Temporarily skip safety checks when creating TensorContainers.

        Normally, when you create a TensorContainer, it automatically checks that:
        - All tensors have the same batch dimensions (shape consistency)
        - All tensors are on the same device (device consistency)

        This safety checking is usually good, but sometimes internal operations need
        to temporarily break these rules to complete successfully.

        For example, when you call container.to('cuda'), the operation needs to:
        1. Move all tensors to CUDA successfully
        2. Reconstruct the container with the moved tensors

        During step 2, the PyTreeContext still contains the old device information
        (e.g., 'cpu'), but the actual tensors are now on CUDA. When reconstructing
        the container, validation sees this disagreement and fails. This context
        manager allows that temporary mismatch between stored and actual device info.

        Technical details:
            During PyTree unflatten operations, the container reconstruction process
            can create temporarily invalid states that are necessary for the
            operation to complete. This context manager disables validation during
            these critical moments.

        Warning:
            Only use this for internal operations. If you manually create containers
            with mismatched tensor shapes or devices, your code will likely crash
            later with confusing error messages. When using unsafe_construction,
            you must ensure the container reaches a consistent state (matching
            shapes/devices).

        Example:
            >>> # This would normally fail due to device mismatch during construction
            >>> with TensorContainer.unsafe_construction():
            ...     container = MyContainer({
            ...         'a': torch.tensor([1, 2]).cuda(),
            ...         'b': torch.tensor([3, 4]).cpu()  # Different device!
            ...     })
            >>> # Validation is back on after the context ends

        Yields:
            None: Context manager yields nothing
        """
        old_value = getattr(cls._validation_disabled, "value", False)
        cls._validation_disabled.value = True
        try:
            yield
        finally:
            cls._validation_disabled.value = old_value

    @abstractmethod
    def _pytree_flatten(self) -> tuple[list[Any], Context]:
        pass

    @abstractmethod
    def _pytree_flatten_with_keys_fn(
        self,
    ) -> tuple[list[tuple[KeyEntry, Any]], Any]:
        pass

    @classmethod
    @abstractmethod
    def _pytree_unflatten(
        cls: type[Self], leaves: Iterable[Any], context: Context
    ) -> Self:
        pass

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in HANDLED_FUNCTIONS or not all(
            issubclass(t, (Tensor, TensorContainer)) for t in types
        ):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    @classmethod
    def _tree_map(
        cls,
        func: Callable[..., Any],
        tree: PyTree,
        *rests: PyTree,
        is_leaf: Callable[[PyTree], bool] | None = None,
    ) -> Self:
        def wrapped_func(keypath, x, *xs):
            try:
                return func(x, *xs)
            except Exception as e:
                path = format_path(keypath)
                message = f"Error at path {path}: {type(e).__name__}: {e}"
                raise type(e)(message) from e

        try:
            return pytree.tree_map_with_path(
                wrapped_func, tree, *rests, is_leaf=is_leaf
            )
        except Exception as e:
            # The following code is just to provide better error messages for operations that
            # work on multiple pytrees such as torch.stack() or torch.cat()
            # It is not necessary for TensorContainer to function properly.
            if len(rests) > 0:
                msg = diagnose_pytree_structure_mismatch(tree, *rests, is_leaf=is_leaf)
                if msg:
                    raise RuntimeError(msg) from e

            # Re-raise if it is an unknown error.
            raise e

    @classmethod
    def tree_map_with_path(
        cls,
        func: Callable[..., Any],
        tree: PyTree,
        *rests: PyTree,
        is_leaf: Callable[[PyTree], bool] | None = None,
    ) -> Self:
        # This is copied from pytree.tree_map_with_path()
        # We add the check for no leaves as operations are currently no supported for
        # empty TensorContainers.
        keypath_leaves, treespec = pytree.tree_flatten_with_path(tree, is_leaf)

        if len(keypath_leaves) == 0:
            raise RuntimeError(
                "TensorContainer does not allow operations on containers without leaves (i.e. not containing any tensors)."
            )

        keypath_leaves = list(zip(*keypath_leaves))
        all_keypath_leaves = keypath_leaves + [treespec.flatten_up_to(r) for r in rests]
        return treespec.unflatten(func(*xs) for xs in zip(*all_keypath_leaves))

    @classmethod
    def _is_shape_compatible(cls, parent: TensorContainer, child: TCCompatible):
        return child.shape[: parent.ndim] == parent.shape

    @classmethod
    def _is_device_compatible(cls, parent: TensorContainer, child: TCCompatible):
        if parent.device is None:
            return True

        return parent.device == child.device

    def _validate_shape(self, value):
        if not self._is_shape_compatible(self, value):
            raise RuntimeError(
                f"Invalid shape {value.shape}. Expected shape that is compatible to {self.shape}"
            )

    def _validate_device(self, value):
        if not self._is_device_compatible(self, value):
            raise RuntimeError(
                f"Invalid device {value.device}. Expected device that is compatible to {self.device}"
            )

    def _validate(self):
        # Check if validation is disabled via context manager
        if getattr(self._validation_disabled, "value", False):
            return

        key_value, _ = self._pytree_flatten_with_keys_fn()

        for k, v in key_value:
            try:
                self._validate_shape(v)
                self._validate_device(v)
            except RuntimeError as e:
                raise RuntimeError(f"Validation error at key {k}: {e.args}")

    @property
    def ndim(self):
        return len(self.shape)

    # --- Overloaded methods leveraging PyTrees ---

    def get_number_of_consuming_dims(self, item) -> int:
        """
        Returns the number of container dimensions consumed by an indexing item.

        This method is crucial for ellipsis expansion calculation. "Consuming" means
        the index item selects from existing container dimensions, reducing the
        container's rank. "Non-consuming" items either don't affect existing
        dimensions (Ellipsis) or add new dimensions (None).

        Args:
            item: An indexing element from an indexing tuple

        Returns:
            Number of container dimensions this item consumes:
            - 0 for non-consuming items (Ellipsis, None)
            - item.ndim for boolean tensors (advanced indexing)
            - 1 for standard consuming items (int, slice, non-bool tensor)

        Examples:
            >>> container.get_number_of_consuming_dims(0)          # int
            1
            >>> container.get_number_of_consuming_dims(slice(0, 2)) # slice
            1
            >>> container.get_number_of_consuming_dims(...)        # Ellipsis
            0
            >>> container.get_number_of_consuming_dims(None)       # None (newaxis)
            0
            >>> bool_mask = torch.tensor([[True, False], [False, True]])
            >>> container.get_number_of_consuming_dims(bool_mask)  # 2D bool tensor
            2
            >>> indices = torch.tensor([0, 2, 1])
            >>> container.get_number_of_consuming_dims(indices)    # non-bool tensor
            1

        Note:
            Used internally by transform_ellipsis_index to calculate how many ':'
            slices the ellipsis should expand to: rank - sum(consuming_dims)
        """
        if item is Ellipsis or item is None:
            return 0
        if isinstance(item, torch.Tensor) and item.dtype == torch.bool:
            return item.ndim

        return 1

    def transform_ellipsis_index(self, shape: torch.Size, idx: tuple) -> tuple:
        """
        Transforms an indexing tuple with ellipsis relative to container batch shape.

        This method is essential for TensorContainer's design: containers have batch dimensions
        (self.shape) but contain individual tensors with varying total shapes. Without this
        preprocessing, ellipsis (...) would expand differently for each tensor based on its
        individual shape, violating container semantics and batch/event dimension boundaries.

        Args:
            shape: The container's batch shape (self.shape), used as reference for ellipsis expansion
            idx: Indexing tuple potentially containing ellipsis (...)

        Returns:
            Equivalent indexing tuple with ellipsis expanded to explicit slices

        Example:
            Container with shape (4, 3) containing tensors (4, 3, 128) and (4, 3, 6, 64):

            # User indexing: container[..., 0]
            # This method transforms: (..., 0) -> (:, 0) based on container batch shape (4, 3)
            # Applied to tensors: [:, 0] works consistently on both tensor shapes
            # Result: Container shape becomes (4,) with tensors (4, 128) and (4, 6, 64)

            # Without this preprocessing, PyTorch would expand ellipsis per-tensor:
            # Tensor (4, 3, 128): [..., 0] -> [:, :, :, 0] (invalid - too many indices)
            # Tensor (4, 3, 6, 64): [..., 0] -> [:, :, :, :, 0] (invalid - too many indices)

        Raises:
            IndexError: If multiple ellipsis found or too many indices for container dimensions

        Note:
            This method is called internally during __getitem__ and __setitem__ operations
            to ensure consistent indexing behavior across all tensors in the container.
        """
        # Step 1: Count indices that "consume" container dimensions
        # - Ellipsis (...) and None don't consume dims (Ellipsis is placeholder, None adds new dim)
        # - int, slice, tensor indices consume 1 dim each (bool tensor consumes its ndim)
        # Example: (..., 0, :) has 2 consuming indices (0 and :), ellipsis doesn't count
        num_consuming_indices = sum(
            self.get_number_of_consuming_dims(item) for item in idx
        )

        # Step 2: Validate that we don't have more indices than container dimensions
        # Container shape (4, 3) has rank=2, so max 2 consuming indices allowed
        rank = len(shape)
        if num_consuming_indices > rank:
            raise IndexError(
                f"too many indices for container: container is {rank}-dimensional, "
                f"but {num_consuming_indices} were indexed"
            )

        # Step 3: Early return if no ellipsis - nothing to transform
        if Ellipsis not in idx:
            return idx

        # Step 4: Validate only one ellipsis exists (PyTorch/NumPy requirement)
        ellipsis_count = 0
        for item in idx:
            if item is Ellipsis:
                ellipsis_count += 1
        if ellipsis_count > 1:
            raise IndexError("an index can only have a single ellipsis ('...')")

        # Step 5: Core calculation - determine how many ':' slices ellipsis should expand to
        # Example: Container shape (4, 3), index (..., 0)
        # - rank=2, consuming_indices=1 -> ellipsis expands to 2-1=1 slice
        # - Result: (..., 0) becomes (:, 0)
        ellipsis_pos = idx.index(Ellipsis)
        num_slices_to_add = rank - num_consuming_indices

        # Step 6: Reconstruct index tuple by replacing ellipsis with explicit slices
        # Split around ellipsis: (a, ..., b) -> (a,) + (:, :, ...) + (b,)
        part_before_ellipsis = idx[:ellipsis_pos]  # Everything before ...
        part_after_ellipsis = idx[ellipsis_pos + 1 :]  # Everything after ...
        ellipsis_replacement = (
            slice(None),
        ) * num_slices_to_add  # (:, :, ...) - the ':' slices

        # Combine parts: before + replacement + after
        final_index = part_before_ellipsis + ellipsis_replacement + part_after_ellipsis

        return final_index

    def __repr__(self) -> str:
        # Use a consistent indent of 4 spaces, which is standard
        indent = "    "

        def _format_item(key, value):
            """Formats a key-value pair for representation."""
            key_repr = f"{str(key)}: "
            if isinstance(value, Tensor):
                # Custom, more informative representation for Tensors
                content = f"Tensor(shape={value.shape}, device={value.device}, dtype={value.dtype})"
            else:
                # For nested TensorDicts, repr() is called recursively.
                # The subsequent textwrap.indent handles the indentation of the nested structure.
                content = repr(value)

            return key_repr + content

        # Flatten the structure to get key-value pairs
        key_value_pairs, _ = self._pytree_flatten_with_keys_fn()

        # Create a string for all items, separated by newlines
        items_str = "\n".join(_format_item(k, v) for k, v in key_value_pairs)

        # Indent the entire block of items
        indented_items = textwrap.indent(items_str, indent)

        # Assemble the final, properly formatted representation string
        return (
            f"{self.__class__.__name__}(\n"
            f"{indent}shape={tuple(self.shape)},\n"
            f"{indent}device={self.device},\n"
            f"{indent}items=\n{textwrap.indent(indented_items, indent)}\n{indent}\n"
            f")"
        )

    def __getitem__(self: Self, key: IndexType) -> Self:
        """Index into the container along batch dimensions.

        Indexing operations are applied to the batch dimensions of all contained tensors.
        Event dimensions are preserved unchanged. Supports all PyTorch indexing patterns:

        - Integer indexing: reduces batch dimensions
        - Slice indexing: preserves batch structure
        - Boolean mask indexing: filters batch elements
        - Advanced indexing: tensor-based selection
        - Ellipsis (...): automatic dimension expansion

        Args:
            key: Index specification (int, slice, tensor, tuple, etc.)

        Returns:
            TensorContainer: New container with indexed tensors

        Raises:
            IndexError: If indexing a 0-dimensional container with non-tuple index
            IndexError: If ellipsis appears multiple times in index tuple

        Example:
            >>> container.shape == (4, 3)
            >>> # Integer indexing - reduces batch dimensions
            >>> sample = container[0]           # shape becomes (3,)
            >>> timestep = container[:, 0]      # shape becomes (4,)
            >>>
            >>> # Slice indexing - preserves structure
            >>> subset = container[1:3]         # shape becomes (2, 3)
            >>>
            >>> # Boolean mask - filters elements
            >>> mask = torch.tensor([True, False, True, False])
            >>> filtered = container[mask]      # shape becomes (2, 3)
            >>>
            >>> # Advanced indexing - tensor indices
            >>> indices = torch.tensor([0, 2, 1])
            >>> reordered = container[indices]  # shape becomes (3, 3)
        """
        if isinstance(key, tuple):
            key = self.transform_ellipsis_index(self.shape, key)
        elif self.ndim == 0:
            raise IndexError(
                "Cannot index a 0-dimensional TensorContainer with a single index. Use a tuple of indices matching the batch shape, or an empty tuple for a scalar."
            )

        return self._tree_map(lambda x: x[key], self)

    def __setitem__(self: Self, index: IndexType, value: Self) -> None:
        """
        Sets the value of a slice of the container in-place.

        This method mimics the behavior of `torch.Tensor.__setitem__`. It requires
        that the `value` be broadcastable to the shape of the slice `self[index]`.

        This approach correctly handles advanced indexing (e.g., boolean masks) by
        relying on PyTorch's underlying shape-checking for the leaf-level assignments.

        Args:
            index: The index or slice to set. Supports basic and advanced
                 indexing, including Ellipsis (`...`).
            value: The value to set. If it's a `TensorContainer`, its leaves must be
                   broadcastable to the corresponding sliced leaves of `self`. If it's
                   a scalar or `torch.Tensor`, it must be broadcastable to all sliced
                   leaves of `self`.
        """

        if not isinstance(value, type(self)):
            raise ValueError(f"Invalid value. Expected value of type {type(self)}")

        processed_index = index
        if isinstance(index, tuple):
            processed_index = self.transform_ellipsis_index(self.shape, index)

        for k, v in self._pytree_flatten_with_keys_fn()[0]:
            try:
                v[processed_index] = k.get(value)
            except Exception as e:
                raise type(e)(
                    f"Issue with key {str(k)} and index {processed_index} for value of shape {v.shape} and type {type(v)} and assignment of shape {tuple(value.shape)}"
                ) from e


# --- PyTree-aware implementations of torch functions ---
@implements(torch.stack)
def _stack(
    tensors: tuple[TensorContainer, ...] | list[TensorContainer], dim: int = 0
) -> TensorContainer:
    if not tensors:
        # Replicate PyTorch's error for an empty list
        raise RuntimeError("stack expects a non-empty TensorList")

    first_tc = tensors[0]
    batch_ndim = first_tc.ndim

    # Normalize dim to handle negative values; for stack, the new dim is added
    if dim < 0:
        dim = dim + batch_ndim + 1

    if dim < 0 or dim > batch_ndim:
        raise IndexError(
            f"Dimension {dim - batch_ndim - 1 if dim < 0 else dim} out of range "
            f"(expected 0 to {batch_ndim} for stack operation on shape {tuple(first_tc.shape)})"
        )

    # Pytree handles the stacking of individual tensors and metadata consistency
    result_td = TensorContainer._tree_map(lambda *x: torch.stack(x, dim), *tensors)

    return result_td


@implements(torch.cat)
def _cat(
    tensors: tuple[TensorContainer, ...] | list[TensorContainer], dim: int = 0
) -> TensorContainer:
    # Get the first tensor container to determine the base shape and type
    first_tc = tensors[0]
    batch_ndim = first_tc.ndim

    # Normalize dim to be positive
    if dim < 0:
        dim = dim + batch_ndim

    if dim < 0 or dim > batch_ndim - 1:
        raise IndexError(
            f"Dimension {dim - batch_ndim if dim < 0 else dim} out of range "
            f"(expected 0 to {batch_ndim - 1} for concatenation on shape {tuple(first_tc.shape)})"
        )

    # Create a new TensorContainer of the same type as the first one
    # and apply torch.cat to its internal tensors
    result_td = TensorContainer._tree_map(lambda *x: torch.cat(x, dim), *tensors)

    return result_td
