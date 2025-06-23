from __future__ import annotations

import functools
from typing import List, Optional, Tuple, TypeVar, Union

import torch

# Use the official PyTree utility from torch
import torch.utils._pytree as pytree

HANDLED_FUNCTIONS = {}


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
            issubclass(t, (torch.Tensor, TensorContainer)) for t in types
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
    returns = pytree.tree_map(lambda *x: torch.stack(x, dim), *tensors)
    return returns


@implements(torch.cat)
def _cat(
    tensors: Union[Tuple[TensorContainer, ...], List[TensorContainer]], dim: int = 0
) -> TensorContainer:
    # Get the first tensor container to determine the base shape and type
    first_td = tensors[0]
    batch_ndim = first_td.ndim

    # Normalize dim to be positive
    if dim < 0:
        dim = dim + batch_ndim

    if dim < 0 or dim > batch_ndim - 1:
        raise IndexError("Dimension out of range")

    shape_expected = first_td.shape[:dim] + first_td.shape[dim + 1 :]

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
