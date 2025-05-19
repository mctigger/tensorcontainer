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


def normalize_device(dev: torch.device) -> torch.device:
    d = torch.device(dev)
    # If no index was given, fill in current_device() for CUDA, leave CPU as-is
    if d.type == "cuda" and d.index is None:
        idx = (
            torch.cuda.current_device()
        )  # e.g. 0 :contentReference[oaicite:4]{index=4}
        return torch.device(f"cuda:{idx}")
    return d


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
        device: Optional[torch.device] = torch.device("cpu"),
    ):
        shape = torch.Size(shape)
        self.shape = shape
        self.device = device
        self.data = {}
        self._convert_dict_to_tensordict(dictionary, shape, device)

    def _convert_dict_to_tensordict(
        self,
        dictionary: Mapping[str, NestedTDCompatible],
        shape,
        device: Optional[torch.device] = None,
    ):
        for k, v in dictionary.items():
            #
            try:
                if isinstance(v, MutableMapping):
                    self[k] = TensorDict(v, shape, device)
                elif isinstance(v, (Tensor, TensorDict)):
                    self[k] = v
                else:
                    raise TypeError(
                        f"Unsupported type {type(v)} for key {k}. Expected Tensor, TensorDict, or MutableMapping."
                    )
            except Exception as e:
                raise RuntimeError(f"Error adding key {k} to TensorDict. ") from e

    def _check_tensor_shape(self, tensor: TDCompatible, shape: torch.Size):
        """
        Check if the tensor shape matches the batch_shape of the tensordict.
        """
        if tensor.shape[: len(shape)] != shape:
            raise ShapeMismatchError(
                f"Tensor shape {tensor.shape} does not match expected batch_shape {shape}.",
                tensor,
            )

    def _check_tensor_device(self, tensor: TDCompatible, device: torch.device):
        """
        Check if the tensor device matches the device of the tensordict.
        """
        if normalize_device(tensor.device) != normalize_device(device):
            raise ValueError(
                f"Tensor device {tensor.device} does not match expected device {device}."
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
        self._check_tensor_shape(value, self.shape)
        self._check_tensor_device(value, self.device)

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

        return TensorDict(data, self.shape, self.device)

    def _update_shape_and_device(self, shape, device, fn):
        dummy = torch.empty(*shape, 1, device=device)
        new_dummy = fn(dummy)
        new_shape = new_dummy.shape[:-1]
        new_device = new_dummy.device

        return new_shape, new_device

    @classmethod
    def _zip_update_shape_and_device(cls, shapes, device, fn):
        dummy = [torch.empty(*s, device=device) for s in shapes]
        new_dummy = fn(dummy)
        new_shape = new_dummy.shape
        new_device = new_dummy[0].device

        return new_shape, new_device

    def apply(self, fn: Callable[[TDCompatible], TDCompatible]) -> TensorDict:
        data = {k: fn(v) for k, v in self.data.items()}
        shape, device = self._update_shape_and_device(self.shape, self.device, fn)

        return TensorDict(data, shape, device)

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
        shape, device = TensorDict._zip_update_shape_and_device(
            shapes, tensor_dicts[0].device, fn
        )

        return TensorDict(data, shape, device)

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

    def __repr__(self) -> str:
        items = []
        for k, v in self.data.items():
            if isinstance(v, Tensor):
                items.append(f"{k}: Tensor(shape={tuple(v.shape)}, dtype={v.dtype})")
            else:  # Nested TensorDict
                nested = repr(v).replace("\n", "\n  ")
                items.append(f"{k}:\n  {nested}")
        header = f"{self.__class__.__name__}(shape={tuple(self.shape)}, device={self.device})"
        return "\n".join([header] + items)


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
