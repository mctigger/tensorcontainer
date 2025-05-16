from __future__ import annotations
import functools
from typing import Any, Callable, List, Optional, Tuple, TypeAlias, Union

import torch
from torch import Tensor

from rtd.utils import NestedDict, apply_leaves, get_leaves, zip_apply_leaves
from rtd.validation import check_leaves_devices_match
from rtd import config


HANDLED_FUNCTIONS = {}


def implements(torch_function):
    """Register a torch function override"""

    @functools.wraps(torch_function)
    def decorator(func):
        HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


class TensorDict:
    """Wrapper around (nested) dictionaries of torch.Tensor that applies operations to all tensors in the dict.

    Note: TensorDict does not copy the input data!

    Often one needs to modify the batch dimension of tensors with different event dimensions, e.g. for complex
    observations in reinforcement learning. TensorDict takes the dictionary and the batch shape. Any operation executed
    on a TensorDict
    """

    validate_args: bool = config.validate_args

    def __init__(self, data: NestedDict, shape, device: Optional[torch.device] = None):
        if device is None:
            check_leaves_devices_match(data, validate_args=self.validate_args)

        self.device = device
        self.data = data
        self.shape = shape

    @classmethod
    def extract(cls, tensor_dict):
        if isinstance(tensor_dict, TensorDict):
            return {k: TensorDict.extract(v) for k, v in tensor_dict.data.items()}
        else:
            return tensor_dict

    @classmethod
    def generate_shape(cls, data):
        shape_dict = apply_leaves(data, lambda x: x.shape)
        shapes = get_leaves(shape_dict)

        shape = []
        for s in zip(*shapes):
            if not all(x == s[0] for x in s):
                break
            else:
                shape.append(s[0])

        return shape

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.data[key]

        return self.apply(lambda x: x[key])

    def __setitem__(self, key, value):
        self.data[key] = value

    def items(self):
        return self.data.items()

    def _update_shape(self, shape, fn):
        dummy = torch.empty(*shape, 1)
        new_dummy = fn(dummy)
        new_shape = new_dummy.shape[:-1]

        return new_shape

    @classmethod
    def _zip_update_shape(self, shapes, fn):
        dummy = [torch.empty(*s, 1) for s in shapes]
        new_dummy = fn(*dummy)
        new_shape = new_dummy.shape[:-1]

        return new_shape

    def apply(self, fn: Callable[[Tensor], Tensor]) -> TensorDict:
        data = apply_leaves(self.data, fn)
        shape = self._update_shape(self.shape, fn)

        return TensorDict(data, shape)

    @classmethod
    def zip_apply(
        cls, tensor_dicts: List[TensorDict], fn: Callable[[Tensor], Tensor]
    ) -> TensorDict:
        shapes = [t.shape for t in tensor_dicts]
        datas = [t.data for t in tensor_dicts]

        data = zip_apply_leaves(datas, fn)
        shape = TensorDict._zip_update_shape(shapes, fn)

        return TensorDict(data, shape)

    def to(self, *args, **kwargs):
        return self.apply(lambda x: x.to(*args, **kwargs))

    def detach(self, *args, **kwargs):
        return self.apply(lambda x: x.detach(*args, **kwargs))

    def view(self, *shape):
        return self.apply(lambda x: x.view(*shape, *x.shape[len(self.shape) :]))

    def expand(self, *shape):
        return self.apply(
            lambda x: x.expand(*shape, *x.shape[len(self.shape) :]),
        )

    def clone(self, *args, **kwargs):
        return self.apply(lambda x: x.clone(*args, **kwargs))

    def unsqueeze(self, *args, **kwargs):
        return torch.unsqueeze(self, *args, **kwargs)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Extends torch with TensorDict

        See https://pytorch.org/docs/stable/notes/extending.html#extending-torch
        """
        if kwargs is None:
            kwargs = {}

        if func not in HANDLED_FUNCTIONS or not all(
            issubclass(t, (torch.Tensor, TensorDict)) for t in types
        ):
            return NotImplemented

        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def __contains__(self, key):
        return key in self.data


# torch methods that already are implemented for TensorDict


@implements(torch.stack)
def _stack(tensors: Union[Tuple[TensorDict, ...], List[TensorDict]], dim: int = 0):
    return TensorDict.zip_apply(tensors, lambda *x: torch.stack(x, dim))


@implements(torch.cat)
def _cat(tensors: Union[Tuple[TensorDict, ...], List[TensorDict]], dim: int = 0):
    return TensorDict.zip_apply(tensors, lambda *x: torch.cat(x, dim))


@implements(torch.unsqueeze)
def _unsqueeze(tensor: TensorDict, dim: int = 0):
    assert dim <= len(tensor.shape)
    return TensorDict.apply(tensor, lambda x: torch.unsqueeze(x, dim))
