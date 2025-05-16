from typing import Dict, List

from torch import Tensor
from rtd.utils import NestedDict, get_leaves

import torch
from functools import wraps


def skip_if_disabled(func):
    """
    Decorator to skip execution of a validation function if validate_args=False is passed.
    The wrapped function does not need to return a value.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if kwargs.pop("validate_args", True):
            return func(*args, **kwargs)

    return wrapper


@skip_if_disabled
def check_shapes_match(tensors: List[Tensor]) -> None:
    """
    Raises a ValueError if the shapes of the given tensors do not match.
    """
    assert isinstance(tensors, list), f"Expected a list of Tensors, got {type(tensors)}"
    assert all(isinstance(t, Tensor) for t in tensors), (
        "All elements must be torch.Tensor"
    )

    if not tensors:
        return

    ref_shape = tensors[0].shape
    for i, t in enumerate(tensors[1:], start=1):
        if t.shape != ref_shape:
            raise ValueError(
                f"Shape mismatch at index {i}: expected {ref_shape}, got {t.shape}"
            )


@skip_if_disabled
def check_keys_match(dicts: List[Dict]) -> None:
    """
    Raises a ValueError if the dictionaries do not all have the same keys.
    """
    assert isinstance(dicts, list), f"Expected a list of dicts, got {type(dicts)}"
    assert all(isinstance(d, dict) for d in dicts), "All elements must be dictionaries"

    if not dicts:
        return

    ref_keys = dicts[0].keys()
    for i, d in enumerate(dicts[1:], start=1):
        if d.keys() != ref_keys:
            raise ValueError(
                f"Key mismatch at index {i}: expected keys {ref_keys}, got {d.keys()}"
            )


@skip_if_disabled
def check_devices_match(tensors: List[Tensor]) -> torch.device:
    """
    Raises a ValueError if the tensors are not all on the same device.
    Returns the common device if check passes.
    """
    assert isinstance(tensors, List), f"Expected a list of Tensors, got {type(tensors)}"
    for t in tensors:
        assert isinstance(t, Tensor), f"Expected a Tensor, got {type(t)}"

    if not tensors:
        raise ValueError("Expected at least one tensor to check devices.")

    ref_device = tensors[0].device
    for i, t in enumerate(tensors[1:], start=1):
        if t.device != ref_device:
            raise ValueError(
                f"Device mismatch at index {i}: expected {ref_device}, got {t.device}"
            )

    return ref_device


@skip_if_disabled
def check_leaves_devices_match(data: NestedDict) -> None:
    """
    Raises a ValueError if the leaves of the nested dictionary do not all have the same device.
    """
    leaves = get_leaves(data)
    check_devices_match(leaves)
