"""Custom types for tensorcontainer.

This module provides shared type aliases. The Shape alias is intended to mirror
PyTorch’s internal shape typing used in prims (torch._prims_common.ShapeType).
A separate test ensures our definition matches PyTorch’s current definition and
will fail if PyTorch changes it in the future.
"""

from typing import Union
import torch

# Mirror torch._prims_common.ShapeType without importing it directly.
# Current upstream definition is: Union[torch.Size, list[int], tuple[int, ...]]
Shape = Union[torch.Size, list[int], tuple[int, ...]]

__all__ = ["Shape"]
