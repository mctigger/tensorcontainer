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
    def __init__(self, shape):
        super().__init__()

        self.shape = shape

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in HANDLED_FUNCTIONS or not all(
            issubclass(t, (torch.Tensor, TensorContainer)) for t in types
        ):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    # --- Validation ---
    def _validate(self):
        if pytree.tree_any(self._has_shape_prefix, self):
            raise ValueError()

    def _has_shape_prefix(self, shape):
        return shape == self.shape[: len(shape)]


    @property
    def ndim(self):
        return len(self.shape)

    # --- Overloaded methods leveraging PyTrees ---
    def view(self, *shape: int) -> TensorContainer:
        return pytree.tree_map(lambda x: x.view(*shape, *x.shape[self.ndim:]), self)

    def reshape(self, *shape: int) -> TensorContainer:
        return pytree.tree_map(lambda x: x.reshape(*shape, *x.shape[self.ndim:]), self)

    def to(self, *args, **kwargs) -> TensorContainer:
        return pytree.tree_map(lambda x: x.to(*args, **kwargs), self)

    def detach(self) -> TensorContainer:
        return pytree.tree_map(lambda x: x.detach(), self)

    def clone(self) -> TensorContainer:
        return pytree.tree_map(lambda x: x.clone(), self)

    def expand(self, *shape: int) -> TensorContainer:
        return pytree.tree_map(lambda x: x.expand(*shape, *x.shape[self.ndim:]), self)


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
