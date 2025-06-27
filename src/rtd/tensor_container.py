from __future__ import annotations

import functools
from typing import Any, List, Optional, Tuple, TypeAlias, TypeVar, Union

import torch

# Use the official PyTree utility from torch
import torch.utils._pytree as pytree
from torch import Tensor

HANDLED_FUNCTIONS = {}

TCCompatible: TypeAlias = Union[torch.Tensor, "TensorContainer"]
T = TypeVar("T", bound="TensorContainer")


def implements(torch_function):
    """Register a torch function override for TensorContainer."""

    @functools.wraps(torch_function)
    def decorator(func):
        HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


class TensorContainer:
    """A generic container for nested tensors.

    This class is a base class for more specific tensor containers like
    :class:`TensorDict` and :class:`TensorDataclass`. It provides common
    functionality for manipulating nested tensors, such as reshaping,
    casting, and cloning.

    The main idea behind this class is to provide a container that can be
    used with `torch.utils._pytree` to apply functions to all tensors
    in the container. This allows us to write code that is agnostic to the
    specific container type.

    Important: All methods do only apply to torch.Tensor or subclasses of TensorContainer.
    If a subclass defines non-tensor data (e.g. meta data), no transformations happens to this data.
    For example .clone() will only clone the meta data, but not the meta-data!

    Args:
        shape (torch.Size): The shape of the container.
        device (torch.device): The device of the container.
    """

    def __init__(self, shape, device):
        super().__init__()

        self.shape = shape
        self.device = device

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in HANDLED_FUNCTIONS or not all(
            issubclass(t, (Tensor, TensorContainer)) for t in types
        ):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def _is_shape_compatible(self, shape):
        batch_ndim = len(self.shape)
        leaf_ndim = len(shape)

        return leaf_ndim >= batch_ndim and shape[:batch_ndim] == self.shape

    def _is_device_compatible(self, leaf_device: torch.device):
        """Checks if the leaf_device is compatible with the TensorDict's device."""
        if self.device is None:
            # If TensorDict's device is not specified, any leaf device is considered compatible.
            return True

        td_device_obj = self.device
        if isinstance(self.device, str):
            try:
                td_device_obj = torch.device(self.device)
            except RuntimeError:
                return False

        if not isinstance(td_device_obj, torch.device):
            return False

        # Compare device types
        if td_device_obj.type != leaf_device.type:
            return False

        # Compare device indices
        # If both have an index, they must match
        if td_device_obj.index is not None and leaf_device.index is not None:
            return td_device_obj.index == leaf_device.index
        # If td_device_obj has no index (e.g., "cuda") and leaf_device has index 0 (e.g., "cuda:0"), they are compatible
        elif td_device_obj.index is None and leaf_device.index == 0:
            return True
        # If leaf_device has no index (e.g., "cuda") and td_device_obj has index 0 (e.g., "cuda:0"), they are compatible
        elif td_device_obj.index == 0 and leaf_device.index is None:
            return True
        # If neither has an index (e.g., both "cpu"), they are compatible
        elif td_device_obj.index is None and leaf_device.index is None:
            return True
        # Otherwise, they are not compatible (e.g., "cuda" vs "cuda:1")
        else:
            return False

    @property
    def ndim(self):
        return len(self.shape)

    # --- Overloaded methods leveraging PyTrees ---

    def get_number_of_consuming_dims(self, item):
        if item is Ellipsis:
            return 0
        if isinstance(item, torch.Tensor) and item.dtype == torch.bool:
            return item.ndim

        return 1

    def transform_ellipsis_index(self, shape: tuple[int, ...], idx: tuple) -> tuple:
        """
        Transforms an indexing tuple with an ellipsis into an equivalent one without it.
        ...
        """
        if Ellipsis not in idx:
            return idx

        ellipsis_count = 0
        for item in idx:
            if item is Ellipsis:
                ellipsis_count += 1
        if ellipsis_count > 1:
            raise IndexError("an index can only have a single ellipsis ('...')")

        ellipsis_pos = idx.index(Ellipsis)

        # Count how many items in the index "consume" an axis from the original shape.
        # `None` adds a new axis, so it's not counted.
        num_consuming_indices = sum(
            self.get_number_of_consuming_dims(item)
            for item in idx
            if item is not Ellipsis and item is not None
        )

        rank = len(shape)

        if num_consuming_indices > rank:
            raise IndexError(
                f"too many indices for array: array is {rank}-dimensional, "
                f"but {num_consuming_indices} were indexed"
            )

        # Calculate slices needed based on the consuming indices
        num_slices_to_add = rank - num_consuming_indices

        part_before_ellipsis = idx[:ellipsis_pos]
        part_after_ellipsis = idx[ellipsis_pos + 1 :]
        ellipsis_replacement = (slice(None),) * num_slices_to_add

        final_index = part_before_ellipsis + ellipsis_replacement + part_after_ellipsis

        return final_index

    def __getitem__(self: T, key: Any) -> T:
        if isinstance(key, tuple):
            key = self.transform_ellipsis_index(self.shape, key)
        elif self.ndim == 0:
            raise IndexError(
                "Cannot index a 0-dimensional TensorContainer with a single index. Use a tuple of indices matching the batch shape, or an empty tuple for a scalar."
            )
        return pytree.tree_map(lambda x: x[key], self)

    def _get_leaf_key(self, leaf_tensor: torch.Tensor, key_param: Any) -> Tuple:
        """
        Constructs the correct indexing key for a leaf tensor based on the
        original key provided to __setitem__.
        """
        current_key_tuple = key_param
        if not isinstance(current_key_tuple, tuple):
            current_key_tuple = (current_key_tuple,)

        try:
            # Handle Ellipsis: expand it to the correct number of slice(None)
            ellipsis_pos = current_key_tuple.index(Ellipsis)
            pre_ellipsis = current_key_tuple[:ellipsis_pos]
            post_ellipsis = current_key_tuple[ellipsis_pos + 1 :]

            num_ellipsis_dims = (
                leaf_tensor.ndim - len(pre_ellipsis) - len(post_ellipsis)
            )

            if num_ellipsis_dims < 0:
                raise IndexError(
                    f"Too many indices for tensor of dimension {leaf_tensor.ndim} "
                    f"after expanding Ellipsis for key '{key_param}'."
                )
            return pre_ellipsis + (slice(None),) * num_ellipsis_dims + post_ellipsis

        except ValueError:  # Ellipsis not found
            # No Ellipsis: append slice(None) for remaining event dimensions of the leaf.
            # current_key_tuple applies to the batch dimensions of the container.
            num_indices_in_key = len(current_key_tuple)

            if num_indices_in_key > leaf_tensor.ndim:
                raise IndexError(
                    f"too many indices for tensor of dimension {leaf_tensor.ndim} "  # Changed to lowercase 't'
                    f"for key '{key_param}'."
                )

            event_dims_to_slice = leaf_tensor.ndim - num_indices_in_key
            # This should ideally not happen if the above check is correct,
            # but as a safeguard:
            if event_dims_to_slice < 0:
                raise IndexError(
                    f"too many indices for tensor of dimension {leaf_tensor.ndim} "  # Changed to lowercase 't'
                    f"for key '{key_param}' (key has {num_indices_in_key} elements)."
                )
            return current_key_tuple + (slice(None),) * event_dims_to_slice

    def _format_path(self, path: pytree.KeyPath) -> str:
        """Helper to format a PyTree KeyPath into a readable string."""
        parts = []
        for entry in path:
            if isinstance(entry, tuple):  # Handle nested KeyPath tuples
                parts.append(self._format_path(entry))
            else:
                parts.append(str(entry))

        # Join parts and clean up leading dots if any
        formatted_path = "".join(parts)
        if formatted_path.startswith("."):
            formatted_path = formatted_path[1:]
        return formatted_path

    def __setitem__(self: T, key: Any, value: TCCompatible) -> None:
        """
        Sets the value of a slice of the container in-place.

        This method mimics the behavior of `torch.Tensor.__setitem__`. It requires
        that the `value` be broadcastable to the shape of the slice `self[key]`.

        This approach correctly handles advanced indexing (e.g., boolean masks) by
        relying on PyTorch's underlying shape-checking for the leaf-level assignments.

        Args:
            key: The index or slice to set. Supports basic and advanced
                 indexing, including Ellipsis (`...`).
            value: The value to set. If it's a `TensorContainer`, its leaves must be
                   broadcastable to the corresponding sliced leaves of `self`. If it's
                   a scalar or `torch.Tensor`, it must be broadcastable to all sliced
                   leaves of `self`.
        """
        processed_key = key
        if isinstance(key, tuple):
            processed_key = self.transform_ellipsis_index(self.shape, key)

        if isinstance(value, TensorContainer):
            self_leaves_with_path = pytree.tree_leaves_with_path(self)
            value_leaves_with_path = pytree.tree_leaves_with_path(value)

            if len(self_leaves_with_path) != len(value_leaves_with_path):
                raise ValueError(
                    f"Expected a container with {len(self_leaves_with_path)} leaves, but got one with {len(value_leaves_with_path)}."
                )

            # Assign leaf by leaf. This requires that `value_leaf` is broadcastable
            # to the shape of `self_leaf[processed_key]`, mimicking torch.Tensor behavior.
            for (self_path, self_leaf), (value_path, value_leaf) in zip(
                self_leaves_with_path, value_leaves_with_path
            ):
                try:
                    self_leaf[processed_key] = value_leaf
                except (RuntimeError, ValueError) as e:
                    path_info = self._format_path(self_path)

                    raise ValueError(
                        f"Assignment failed for leaf at path '{path_info}'. "
                        f"There might be a shape mismatch between the corresponding leaves of the source "
                        f"and destination containers. Original error: {e}"
                    ) from e
        else:
            # For a scalar or single tensor, iterate through leaves and assign.
            # PyTorch will raise a RuntimeError if `value` cannot be broadcast
            # to the shape of `self_leaf[processed_key]`.
            for self_leaf in pytree.tree_leaves(self):
                self_leaf[processed_key] = value

    def view(self: T, *shape: int) -> T:
        return pytree.tree_map(lambda x: x.view(*shape, *x.shape[self.ndim :]), self)

    def reshape(self: T, *shape: int) -> T:
        return pytree.tree_map(lambda x: x.reshape(*shape, *x.shape[self.ndim :]), self)

    def to(self: T, *args, **kwargs) -> T:
        # Move tensors and ensure they are contiguous
        return pytree.tree_map(lambda x: x.to(*args, **kwargs), self)

    def detach(self: T) -> T:
        return pytree.tree_map(lambda x: x.detach(), self)

    def clone(self: T, *, memory_format: Optional[torch.memory_format] = None) -> T:
        # If memory_format is not specified, use torch.preserve_format as default
        if memory_format is None:
            memory_format = torch.preserve_format

        cloned_td = pytree.tree_map(
            lambda x: x.clone(memory_format=memory_format), self
        )
        cloned_td.device = self.device
        return cloned_td

    def expand(self: T, *shape: int) -> T:
        return pytree.tree_map(lambda x: x.expand(*shape, *x.shape[self.ndim :]), self)

    def permute(self: T, *dims: int) -> T:
        """Permutes the batch dimensions of the container.

        This is equivalent to calling :meth:`torch.Tensor.permute` on each tensor
        in the container, but only for the batch dimensions.

        Args:
            *dims (int): The desired ordering of dimensions.

        Returns:
            A new container with the batch dimensions permuted.
        """
        if len(dims) != self.ndim:
            raise RuntimeError(
                f"permute() expected {self.ndim} dimensions but got {len(dims)}"
            )
        if len(set(dims)) != len(dims):
            raise RuntimeError("permute(): duplicate dimensions are not allowed")
        for dim in dims:
            if not 0 <= dim < self.ndim:
                raise RuntimeError(
                    f"permute(): dimension out of range (expected to be in range of [0, {self.ndim - 1}], but got {dim})"
                )
        return pytree.tree_map(
            lambda x: x.permute(*dims, *range(self.ndim, x.ndim)), self
        )

    def squeeze(self: T, dim: Optional[int] = None) -> T:
        """Squeezes the batch dimensions of the container.

        Args:
            dim (int, optional): The dimension to squeeze. If ``None``, all
                batch dimensions of size 1 are squeezed.

        Returns:
            A new container with the specified dimensions squeezed.
        """
        if dim is not None:
            if self.shape[dim] != 1:
                return self.clone()
            new_shape = list(self.shape)
            new_shape.pop(dim)
            return self.reshape(*new_shape)
        else:
            new_shape = [s for s in self.shape if s != 1]
            if len(new_shape) == len(self.shape):
                return self.clone()
            return self.reshape(*new_shape)

    def t(self: T) -> T:
        """Transposes the first two batch dimensions of the container.

        This is equivalent to ``self.transpose(0, 1)``.

        Returns:
            A new container with the first two batch dimensions transposed.
        """
        if self.ndim < 2:
            raise RuntimeError(
                "t() expects a tensor with at least 2 dimensions, but got a tensor with "
                f"{self.ndim} dimensions instead"
            )
        return self.transpose(0, 1)

    def transpose(self: T, dim0: int, dim1: int) -> T:
        """Transposes two batch dimensions of the container.

        Args:
            dim0 (int): The first dimension to transpose.
            dim1 (int): The second dimension to transpose.

        Returns:
            A new container with the specified dimensions transposed.
        """
        return pytree.tree_map(lambda x: x.transpose(dim0, dim1), self)

    def unsqueeze(self: T, dim: int) -> T:
        """Unsqueezes a batch dimension of the container.

        Args:
            dim (int): The dimension to unsqueeze.

        Returns:
            A new container with the specified dimension unsqueezed.
        """
        new_shape = torch.empty(self.shape).unsqueeze(dim).shape
        return self.reshape(*new_shape)

    def size(self) -> torch.Size:
        """Returns the size of the batch dimensions."""
        return torch.Size(self.shape)

    def dim(self) -> int:
        """Returns the number of batch dimensions."""
        return self.ndim

    def numel(self) -> int:
        """Returns the total number of elements in the batch dimensions."""
        return self.size().numel()

    def cpu(self: T) -> T:
        """Returns a new container with all tensors on the CPU."""
        return self.to("cpu")

    def cuda(self: T, device=None, non_blocking: bool = False) -> T:
        """Returns a new container with all tensors on the specified CUDA device."""
        return self.to(
            f"cuda:{device}" if device is not None else "cuda",
            non_blocking=non_blocking,
        )

    def float(self: T) -> T:
        """Casts all tensors to float type."""
        return pytree.tree_map(lambda x: x.float(), self)

    def double(self: T) -> T:
        """Casts all tensors to double type."""
        return pytree.tree_map(lambda x: x.double(), self)

    def half(self: T) -> T:
        """Casts all tensors to half type."""
        return pytree.tree_map(lambda x: x.half(), self)

    def long(self: T) -> T:
        """Casts all tensors to long type."""
        return pytree.tree_map(lambda x: x.long(), self)

    def int(self: T) -> T:
        """Casts all tensors to int type."""
        return pytree.tree_map(lambda x: x.int(), self)

    def abs(self: T) -> T:
        """Computes the absolute value of each tensor in the container."""
        return pytree.tree_map(lambda x: x.abs(), self)

    def add(self: T, other) -> T:
        """Adds a value to each tensor in the container."""
        return pytree.tree_map(lambda x: x.add(other), self)

    def sub(self: T, other) -> T:
        """Subtracts a value from each tensor in the container."""
        return pytree.tree_map(lambda x: x.sub(other), self)

    def mul(self: T, other) -> T:
        """Multiplies each tensor in the container by a value."""
        return pytree.tree_map(lambda x: x.mul(other), self)

    def div(self: T, other) -> T:
        """Divides each tensor in the container by a value."""
        return pytree.tree_map(lambda x: x.div(other), self)

    def pow(self: T, exponent) -> T:
        """Raises each tensor in the container to a power."""
        return pytree.tree_map(lambda x: x.pow(exponent), self)

    def sqrt(self: T) -> T:
        """Computes the square root of each tensor in the container."""
        return pytree.tree_map(lambda x: x.sqrt(), self)

    def log(self: T) -> T:
        """Computes the natural logarithm of each tensor in the container."""
        return pytree.tree_map(lambda x: x.log(), self)

    def neg(self: T) -> T:
        """Negates each tensor in the container."""
        return pytree.tree_map(lambda x: x.neg(), self)

    def clamp(self: T, min, max) -> T:
        """Clamps each tensor in the container to a range."""
        return pytree.tree_map(lambda x: x.clamp(min, max), self)


# --- PyTree-aware implementations of torch functions ---
@implements(torch.stack)
def _stack(
    tensors: Union[Tuple[TensorContainer, ...], List[TensorContainer]], dim: int = 0
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
        raise IndexError("Dimension out of range")

    shape_expected = first_tc.shape

    for t in tensors:
        shape_is = t.shape
        if shape_is != shape_expected:
            raise ValueError("stack expects each TensorContainer to be equal size")

    # Pytree handles the stacking of individual tensors and metadata consistency
    result_td = pytree.tree_map(lambda *x: torch.stack(x, dim), *tensors)

    return result_td


@implements(torch.cat)
def _cat(
    tensors: Union[Tuple[TensorContainer, ...], List[TensorContainer]], dim: int = 0
) -> TensorContainer:
    # Get the first tensor container to determine the base shape and type
    first_tc = tensors[0]
    batch_ndim = first_tc.ndim

    # Normalize dim to be positive
    if dim < 0:
        dim = dim + batch_ndim

    if dim < 0 or dim > batch_ndim - 1:
        raise IndexError("Dimension out of range")

    shape_expected = first_tc.shape[:dim] + first_tc.shape[dim + 1 :]

    for t in tensors:
        shape_is = t.shape[:dim] + t.shape[dim + 1 :]
        if shape_is != shape_expected:
            raise ValueError(
                "TensorContainer batch shapes must be identical except for 'dim'"
            )

    # Create a new TensorContainer of the same type as the first one
    # and apply torch.cat to its internal tensors
    result_td = pytree.tree_map(lambda *x: torch.cat(x, dim), *tensors)

    return result_td
