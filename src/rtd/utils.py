from abc import abstractmethod
from typing import Iterable, TypeVar, Type, Any

import torch.utils._pytree as pytree
from torch.utils._pytree import Context, PyTree, KeyEntry

_PytreeRegistered = TypeVar("_PytreeRegistered", bound="PytreeRegistered")


class PytreeRegistered:
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
        cls: Type[_PytreeRegistered], leaves: Iterable[Any], context: Context
    ) -> PyTree:
        pass
