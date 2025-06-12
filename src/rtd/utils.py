from abc import ABC, abstractmethod
from typing import List, Tuple, TypeVar, Type

from torch import Tensor
import torch.utils._pytree as pytree

_PytreeRegistered = TypeVar("_PytreeRegistered", bound="PytreeRegistered")


class PytreeRegistered(ABC):
    """
    A mixin class that automatically registers any of its subclasses
    with the PyTorch PyTree system upon definition.

    Subclasses are expected to implement:
    - _get_flatten_spec(self)
    - _from_flatten_spec(cls, leaves, context)
    """

    def __init_subclass__(cls, **kwargs):
        # This method is called by Python when a class that inherits
        # from PytreeRegistered is defined. `cls` is the new subclass.
        super().__init_subclass__(**kwargs)

        pytree.register_pytree_node(
            cls,
            cls._pytree_flatten,
            cls._pytree_unflatten,
            flatten_with_keys_fn=cls._pytree_flatten_with_keys_fn,
        )

    @abstractmethod
    def _pytree_flatten(self) -> Tuple[List[Tensor], Tuple]:
        pass

    @classmethod
    @abstractmethod
    def _pytree_unflatten(
        cls: Type[_PytreeRegistered], leaves, context
    ) -> _PytreeRegistered:
        pass
