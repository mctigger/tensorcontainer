from __future__ import annotations

import functools
from collections.abc import MutableMapping
from typing import Callable, Dict, List, Mapping, Optional, Tuple, TypeAlias, Union

import torch
from torch import Tensor

from rtd import config
from rtd.errors import ShapeMismatchError
from rtd.utils import apply_leaves, get_leaves

TDCompatible: TypeAlias = Union[Tensor, "TensorDict"]
NestedTDCompatible: TypeAlias = Union[TDCompatible, Dict[str, TDCompatible]]

HANDLED_FUNCTIONS = {}


def implements(torch_function):
    """Register a torch function override"""

    @functools.wraps(torch_function)
    def decorator(func):
        HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


class TensorDict(MutableMapping[str, TDCompatible]):
    """Wrapper around (nested) dictionaries of torch.Tensor that applies operations to all tensors in the dict.

    Note: TensorDict does not copy the input data!

    Often one needs to modify the batch dimension of tensors with different event dimensions, e.g. for complex
    observations in reinforcement learning. TensorDict takes the dictionary and the batch shape. Any operation executed
    on a TensorDict
    """

    validate_args: bool = config.validate_args

    def __init__(
        self,
        dictionary: Mapping[str, NestedTDCompatible],
        shape: Tuple[int, ...],
        device: Optional[torch.device] = None,
    ):
        shape = torch.Size(shape)
        self.shape = shape
        self.device = device
        self.data = self._convert_dict_to_tensordict(dictionary, shape, device)

    def _convert_dict_to_tensordict(
        self,
        dictionary: Mapping[str, NestedTDCompatible],
        shape,
        device: Optional[torch.device] = None,
    ) -> Dict[str, TDCompatible]:
        data = {}
        for k, v in dictionary.items():
            #
            try:
                if isinstance(v, TensorDict):
                    data[k] = TensorDict(v.data, shape, device)
                elif isinstance(v, MutableMapping):
                    data[k] = TensorDict(v, shape, device)
                elif isinstance(v, Tensor):
                    self._check_tensor_shape(v, shape)
                    data[k] = v
                else:
                    raise TypeError(
                        f"Unsupported type {type(v)} for key {k}. Expected Tensor, TensorDict, or MutableMapping."
                    )
            except Exception as e:
                raise RuntimeError(f"Error adding key {k} to TensorDict. ") from e

        return data

    def _check_tensor_shape(self, tensor: TDCompatible, shape: torch.Size):
        """
        Check if the tensor shape matches the batch_shape of the tensordict.
        """
        if tensor.shape[: len(shape)] != shape:
            raise ShapeMismatchError(
                f"Tensor shape {tensor.shape} does not match expected batch_shape {shape}.",
                tensor,
            )

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

    def __delitem__(self, key: str):
        del self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key: str) -> Union[Tensor, TensorDict]:
        if isinstance(key, str):
            return self.data[key]

        return self.apply(lambda x: x[key])

    def __setitem__(self, key, value):
        self.data[key] = value

    def copy(self):
        data = {}
        for k, v in self.data.items():
            if isinstance(v, TensorDict):
                data[k] = v.copy()
            elif isinstance(v, Tensor):
                data[k] = v

        return TensorDict(data, self.shape, self.device)

    def items(self):
        return self.data.items()

    def update(self, mapping: Optional[Mapping[str, TDCompatible]] = {}, **kwargs):
        data = {**mapping, **kwargs}

        update_td = TensorDict(
            data,
            self.shape,
            self.device,
        )
        for k, v in update_td.items():
            self.data[k] = v

    def flatten_keys(self, sep: str = ".") -> TensorDict:
        """Flattens the keys of the TensorDict.

        Args:
            sep (str): Separator to use for flattening keys. Defaults to '.'.

        Returns:
            TensorDict: A new TensorDict with flattened keys.
        """
        data = {}
        for k, v in self.data.items():
            if isinstance(v, TensorDict):
                for sub_k, sub_v in v.flatten_keys(sep).items():
                    data[f"{k}{sep}{sub_k}"] = sub_v
            else:
                data[k] = v

        return TensorDict(data, self.shape)

    def _update_shape(self, shape, fn):
        dummy = torch.empty(*shape, 1)
        new_dummy = fn(dummy)
        new_shape = new_dummy.shape[:-1]

        return new_shape

    @classmethod
    def _zip_update_shape(cls, shapes, fn):
        dummy = [torch.empty(*s) for s in shapes]
        new_dummy = fn(dummy)
        new_shape = new_dummy.shape

        return new_shape

    def apply(self, fn: Callable[[TDCompatible], TDCompatible]) -> TensorDict:
        data = {k: fn(v) for k, v in self.data.items()}
        shape = self._update_shape(self.shape, fn)

        return TensorDict(data, shape)

    @classmethod
    def zip_apply(
        cls,
        tensor_dicts: Union[List[TensorDict], Tuple[TensorDict]],
        fn: Callable[[List[TDCompatible]], TDCompatible],
    ) -> TensorDict:
        data = {}
        for k in tensor_dicts[0].keys():
            for t in tensor_dicts:
                if k not in t:
                    raise KeyError(f"Key {k} not found in all TensorDicts.")

            data[k] = fn([t[k] for t in tensor_dicts])

        shapes = [t.shape for t in tensor_dicts]
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


@implements(torch.stack)
def _stack(tensors: Union[Tuple[TensorDict, ...], List[TensorDict]], dim: int = 0):
    return TensorDict.zip_apply(tensors, lambda x: torch.stack(x, dim))


@implements(torch.cat)
def _cat(tensors: Union[Tuple[TensorDict, ...], List[TensorDict]], dim: int = 0):
    return TensorDict.zip_apply(tensors, lambda x: torch.cat(x, dim))


@implements(torch.unsqueeze)
def _unsqueeze(tensor: TensorDict, dim: int = 0):
    assert dim <= len(tensor.shape)
    return TensorDict.apply(tensor, lambda x: torch.unsqueeze(x, dim))
