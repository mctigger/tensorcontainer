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
        """Flatten this container node into immediate children and reconstruction context.

        This method is part of PyTorch's PyTree protocol and enables automatic tree
        traversal operations like torch.stack, torch.cat, and functional transformations.
        It should decompose this container node into its immediate children and any
        metadata needed to reconstruct this specific node level.

        Unlike a full tree flattening, this method only extracts the **immediate children**
        of this container node. The PyTree system handles recursive traversal by calling
        this method on each node as it walks the tree structure.

        The flattening process separates this node into:
        - **Children**: Immediate child values (tensors, nested containers, etc.)
        - **Context**: Node-level metadata needed to reconstruct this container

        Returns:
            tuple[list[Any], Context]: A tuple containing:
                - list[Any]: Immediate children in a consistent order
                - Context: Node reconstruction metadata (keys, shape, device, etc.)

        Note:
            The order of children must be consistent with _pytree_flatten_with_keys_fn.
            The PyTree system will recursively process any nested containers in the
            children list.

        Example:
            For a TensorDict with {'a': tensor1, 'b': nested_container}:

            >>> children, context = container._pytree_flatten()
            >>> # children = [tensor1, nested_container]  # immediate children only
            >>> # context contains keys=['a', 'b'], shape, device for this node
        """
        pass

    @abstractmethod
    def _pytree_flatten_with_keys_fn(
        self,
    ) -> tuple[list[tuple[KeyEntry, Any]], Any]:
        """Flatten this container node with key paths for enhanced tree operations.

        This method extends _pytree_flatten by providing key paths that identify
        how to access each immediate child within this container node. Key paths enable
        advanced PyTree operations like tree_map_with_path and provide better
        error messages when operations fail.

        Each KeyEntry implements a protocol with:
        - get(parent): Retrieves the child value from this parent container
        - __str__(): String representation for error messages
        - __hash__() and __eq__(): Support for key-based operations

        The key paths provide navigation from this container to each immediate child,
        enabling precise error reporting and advanced tree manipulations.

        Returns:
            tuple[list[tuple[KeyEntry, Any]], Any]: A tuple containing:
                - list[tuple[KeyEntry, Any]]: List of (key_entry, child) pairs where
                  key_entry can navigate from this container to the child
                - Any: Same context as returned by _pytree_flatten

        Note:
            The order and context must match _pytree_flatten exactly. The children
            in both methods should be identical, with this method adding key entries.

        Example:
            For a TensorDict with {'a': tensor1, 'b': nested_container}:

            >>> key_children, context = container._pytree_flatten_with_keys_fn()
            >>> # key_children = [(MappingKey('a'), tensor1), (MappingKey('b'), nested_container)]
            >>> # Each KeyEntry navigates to immediate children only
        """
        pass

    @classmethod
    @abstractmethod
    def _pytree_unflatten(
        cls: type[Self], leaves: Iterable[Any], context: Context
    ) -> Self:
        """Reconstruct a container node from transformed children and context.

        This class method is the inverse of _pytree_flatten, taking the children
        and context produced by flattening and reconstructing this container node.
        It's called automatically by PyTree operations after applying transformations
        to the entire tree structure.

        The method must handle:
        - Reconstructing this container node from the provided context
        - Associating transformed children with their correct positions
        - Restoring node metadata (shape, device) from context
        - Validating that the reconstructed node is consistent

        Args:
            leaves (Iterable[Any]): Transformed children in the same order as
                produced by _pytree_flatten. These may be tensors, nested containers,
                or other values that have been processed by PyTree operations.
            context (Context): Reconstruction metadata from _pytree_flatten,
                containing node-level information like keys, shape, device.

        Returns:
            Self: A new container instance of this node type with the same structure
                as the original but containing the transformed children.

        Note:
            This method may need to use TensorContainer.unsafe_construction()
            context manager if validation would fail during intermediate steps
            of reconstruction (e.g., when device information is temporarily
            inconsistent during .to() operations).

        Example:
            After torch.stack([container1, container2]):

            >>> # PyTree calls _pytree_flatten on both containers
            >>> # Recursively processes the tree and applies torch.stack to leaf tensors
            >>> # Calls _pytree_unflatten with transformed children and context
            >>> result = cls._pytree_unflatten(transformed_children, context)
        """
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
        """Apply a function across PyTree structures with enhanced error reporting.

        This is the foundational infrastructure method that enables all tensor operations
        in TensorContainer. Every tensor-like operation (indexing, device transfers, shape
        operations, math operations, etc.) ultimately goes through this method to apply
        transformations across the container's PyTree structure.

        The method enhances PyTorch's standard `tree_map_with_path` with two layers of
        improved error reporting:

        1. **Keypath Error Wrapping**: Wraps the provided function to catch exceptions
           and enhance them with the specific keypath where the error occurred, making
           debugging much easier in nested structures.

        2. **Structure Mismatch Diagnosis**: For multi-tree operations (like torch.stack
           or torch.cat), provides detailed diagnostics when tree structures don't match.

        Args:
            func (Callable[..., Any]): Function to apply to each leaf. For single-tree
                operations, receives one argument per leaf. For multi-tree operations,
                receives corresponding leaves from all trees.
            tree (PyTree): The primary PyTree to map over (typically a TensorContainer).
            *rests (PyTree): Additional PyTrees for multi-tree operations. All trees
                must have compatible structures.
            is_leaf (Callable[[PyTree], bool] | None): Optional predicate to determine
                what counts as a leaf node during tree traversal.

        Returns:
            Self: A new TensorContainer of the same type with the function applied
                to all leaves, maintaining the original tree structure.

        Raises:
            RuntimeError: When tree structures don't match in multi-tree operations,
                with detailed diagnostic information about the mismatch.
            Exception: Any exception from the provided function, enhanced with keypath
                information showing exactly where the error occurred.

        Note:
            This method is internal infrastructure. End users typically don't call it
            directly, but rather use the tensor-like methods (`.view()`, `.cuda()`, etc.)
            that internally use `_tree_map` to provide their functionality.
        """

        def func_with_error_path(keypath, x, *xs):
            """
            This function wraps the given func just to provide error messages
            that include the path of the leaf that failed.
            """
            try:
                return func(x, *xs)
            except Exception as e:
                path = format_path(keypath)
                message = f"Error at path {path}: {type(e).__name__}: {e}"
                raise type(e)(message) from e

        try:
            return pytree.tree_map_with_path(
                func_with_error_path, tree, *rests, is_leaf=is_leaf
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
        """Apply a function with keypath information to PyTree leaves.

        This is the public interface for keypath-aware tree mapping operations.
        Unlike `_tree_map`, this method provides the full keypath to the function,
        enabling operations that need to know the location of each leaf within
        the tree structure.

        This method validates that the container is not empty before proceeding,
        as TensorContainer operations require at least one tensor to be present.

        Args:
            func (Callable[..., Any]): Function to apply that receives keypath
                as first argument, followed by leaf values. Signature should be
                `func(keypath, leaf, *other_leaves)`.
            tree (PyTree): The primary PyTree to map over.
            *rests (PyTree): Additional PyTrees for multi-tree operations.
            is_leaf (Callable[[PyTree], bool] | None): Optional leaf predicate.

        Returns:
            Self: New container with function applied to all leaves.

        Raises:
            RuntimeError: If the container has no leaves (empty container).

        Example:
            >>> def print_and_transform(keypath, tensor):
            ...     print(f"Processing {keypath}: {tensor.shape}")
            ...     return tensor.float()
            >>> result = MyContainer.tree_map_with_path(print_and_transform, container)

        Note:
            Most operations should use the simpler `_tree_map` method. Use this
            method only when you need access to the keypath information.
        """
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
        """Check if a child tensor/container's shape is compatible with parent's batch shape.

        Shape compatibility requires that the child's leading dimensions (batch dimensions)
        exactly match the parent container's shape. Event dimensions (trailing dimensions
        beyond the batch shape) can vary and are not checked.

        Args:
            parent (TensorContainer): The parent container defining the required batch shape.
            child (TCCompatible): The child tensor or container to check for compatibility.
                Can be either a torch.Tensor or another TensorContainer.

        Returns:
            bool: True if child's leading dimensions match parent's shape, False otherwise.

        Example:
            >>> parent.shape == (4, 3)
            >>> tensor1 = torch.randn(4, 3, 128)  # Compatible: batch (4, 3), event (128,)
            >>> tensor2 = torch.randn(2, 3, 64)   # Incompatible: batch (2, 3) != (4, 3)
            >>> cls._is_shape_compatible(parent, tensor1)  # True
            >>> cls._is_shape_compatible(parent, tensor2)  # False
        """
        return child.shape[: parent.ndim] == parent.shape

    @classmethod
    def _is_device_compatible(cls, parent: TensorContainer, child: TCCompatible):
        """Check if a child tensor/container's device is compatible with parent's device.

        Device compatibility follows these rules:
        - If parent.device is None, any child device is compatible (mixed devices allowed)
        - If parent.device is not None, child device must exactly match parent device

        Args:
            parent (TensorContainer): The parent container defining the required device.
            child (TCCompatible): The child tensor or container to check for compatibility.
                Can be either a torch.Tensor or another TensorContainer.

        Returns:
            bool: True if child's device is compatible with parent's device, False otherwise.

        Example:
            >>> parent_cpu = Container(shape=(2,), device='cpu')
            >>> parent_mixed = Container(shape=(2,), device=None)
            >>> tensor_cpu = torch.randn(2, 3)        # Default device: cpu
            >>> tensor_gpu = torch.randn(2, 3).cuda() # Device: cuda
            >>>
            >>> cls._is_device_compatible(parent_cpu, tensor_cpu)    # True
            >>> cls._is_device_compatible(parent_cpu, tensor_gpu)    # False
            >>> cls._is_device_compatible(parent_mixed, tensor_cpu)  # True
            >>> cls._is_device_compatible(parent_mixed, tensor_gpu)  # True
        """
        if parent.device is None:
            return True

        return parent.device == child.device

    def _validate_shape(self, value):
        """Validate that a value's shape is compatible with this container's batch shape.

        Performs shape validation by checking if the value's leading dimensions match
        this container's batch shape using _is_shape_compatible. Raises a descriptive
        error if validation fails.

        Args:
            value: The tensor or container to validate. Must have a .shape attribute
                and be compatible with TCCompatible type (torch.Tensor or TensorContainer).

        Raises:
            RuntimeError: If the value's shape is incompatible with this container's
                batch shape. The error message includes both the actual and expected shapes.

        Example:
            >>> container.shape == (4, 3)
            >>> good_tensor = torch.randn(4, 3, 128)  # Compatible
            >>> bad_tensor = torch.randn(2, 3, 64)    # Incompatible
            >>> container._validate_shape(good_tensor)  # Passes silently
            >>> container._validate_shape(bad_tensor)   # Raises RuntimeError
        """
        if not self._is_shape_compatible(self, value):
            raise RuntimeError(
                f"Invalid shape {value.shape}. Expected shape that is compatible to {self.shape}"
            )

    def _validate_device(self, value):
        """Validate that a value's device is compatible with this container's device.

        Performs device validation by checking if the value's device matches this
        container's device requirements using _is_device_compatible. Raises a
        descriptive error if validation fails.

        Args:
            value: The tensor or container to validate. Must have a .device attribute
                and be compatible with TCCompatible type (torch.Tensor or TensorContainer).

        Raises:
            RuntimeError: If the value's device is incompatible with this container's
                device. The error message includes both the actual and expected devices.

        Example:
            >>> container = TensorDict({}, shape=(2,), device='cpu')
            >>> cpu_tensor = torch.randn(2, 3)        # Default device: cpu
            >>> gpu_tensor = torch.randn(2, 3).cuda() # Device: cuda
            >>> container._validate_device(cpu_tensor)  # Passes silently
            >>> container._validate_device(gpu_tensor)  # Raises RuntimeError
        """
        if not self._is_device_compatible(self, value):
            raise RuntimeError(
                f"Invalid device {value.device}. Expected device that is compatible to {self.device}"
            )

    def _validate(self):
        """Validate shape and device compatibility for all tensors in this container.

        Performs comprehensive validation by checking that every tensor in the container
        satisfies both shape and device compatibility requirements. This method is called
        automatically during container construction and can be temporarily disabled using
        the unsafe_construction() context manager.

        The validation process:
        1. Checks if validation is disabled via unsafe_construction() context manager
        2. Flattens the container to get all key-value pairs
        3. For each tensor, validates both shape and device compatibility
        4. Provides enhanced error messages with the specific key path where validation failed

        Raises:
            RuntimeError: If any tensor in the container has incompatible shape or device.
                The error message includes the key path where validation failed and the
                underlying validation error details.
        """
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
