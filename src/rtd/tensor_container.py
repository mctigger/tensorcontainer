from __future__ import annotations

import functools
from typing import List, Tuple, Union

import torch

# Use the official PyTree utility from torch
import torch.utils._pytree as pytree


HANDLED_FUNCTIONS = {}


def implements(torch_function):
    """Register a torch function override for TensorContainer."""

    @functools.wraps(torch_function)
    def decorator(func):
        HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


class TensorContainer:
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
    def view(self, *shape: int) -> TensorContainer:
        return pytree.tree_map(lambda x: x.view(*shape, *x.shape[self.ndim :]), self)

    def reshape(self, *shape: int) -> TensorContainer:
        return pytree.tree_map(lambda x: x.reshape(*shape, *x.shape[self.ndim :]), self)

    def to(self, *args, **kwargs) -> TensorContainer:
        # Move tensors and ensure they are contiguous
        return pytree.tree_map(lambda x: x.to(*args, **kwargs), self)

    def detach(self) -> TensorContainer:
        return pytree.tree_map(lambda x: x.detach(), self)

    def clone(self) -> TensorContainer:
        return pytree.tree_map(lambda x: x.clone(), self)

    def expand(self, *shape: int) -> TensorContainer:
        return pytree.tree_map(lambda x: x.expand(*shape, *x.shape[self.ndim :]), self)


# --- PyTree-aware implementations of torch functions ---
@implements(torch.stack)
def _stack(
    tensors: Union[Tuple[TensorContainer, ...], List[TensorContainer]], dim: int = 0
):
    returns = pytree.tree_map(lambda *x: torch.stack(x, dim), *tensors)
    return returns


@implements(torch.cat)
def _cat(
    tensors: Union[Tuple[TensorContainer, ...], List[TensorContainer]], dim: int = 0
):
    return pytree.tree_map(lambda *x: torch.cat(x, dim), *tensors)
