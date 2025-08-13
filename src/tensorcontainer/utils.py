from abc import abstractmethod
from typing import Any, Iterable, List, Tuple, Type, TypeVar

import torch
import torch.utils._pytree as pytree
from torch.utils._pytree import Context, KeyEntry, PyTree

from tensorcontainer.types import DeviceLike

_PytreeRegistered = TypeVar("_PytreeRegistered", bound="PytreeRegistered")


class PytreeRegistered:
    """
    A mixin class that automatically registers any of its subclasses
    with the PyTorch PyTree system upon definition.
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
    def _pytree_flatten(self) -> Tuple[List[Any], Context]:
        pass

    @abstractmethod
    def _pytree_flatten_with_keys_fn(
        self,
    ) -> Tuple[List[Tuple[KeyEntry, Any]], Any]:
        pass

    @classmethod
    @abstractmethod
    def _pytree_unflatten(
        cls: Type[_PytreeRegistered], leaves: Iterable[Any], context: Context
    ) -> PyTree:
        pass


def resolve_device(device: DeviceLike) -> torch.device:
    """
    Resolves a device string to a torch.device object using manual device
    resolution without creating dummy tensors. Auto-indexes devices when
    no specific index is provided.

    If the device cannot be resolved or is not available, raises an appropriate
    exception instead of returning a fallback.

    Args:
        device_str: Device string or torch.device object to resolve

    Returns:
        torch.device: The resolved device object with proper indexing

    Raises:
        RuntimeError: If the device type is invalid or device is not available
        AssertionError: If the backend is not compiled/available
    """
    # Handle case where input is already a torch.device object
    if isinstance(device, torch.device):
        device = device
    elif device is not None:
        device = torch.device(device)
    else:
        raise ValueError("Device cannot be None")

    # Auto-index the device
    if device.index is None:
        if device.type == "cuda":
            # Check if CUDA is available before trying to get current device
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available")
            device = torch.device(device.type, index=torch.cuda.current_device())
        elif device.type not in ("cpu", "meta"):
            # For other device types (mps, xla, etc.), default to index 0
            # This will raise an error if the backend is not available
            device = torch.device(device.type, index=0)

    return device
