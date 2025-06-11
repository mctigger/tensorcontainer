from abc import ABC, abstractmethod

import torch.utils._pytree as pytree


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

        pytree.register_pytree_node(cls, cls._pytree_flatten, cls._pytree_unflatten)

    @abstractmethod
    def _pytree_flatten(cls):
        pass

    @classmethod
    @abstractmethod
    def _pytree_unflatten(cls, leaves, context):
        pass
