"""Custom types for tensorcontainer.

This module provides shared type aliases. The Shape alias is intended to mirror
PyTorch’s internal shape typing used in prims (torch._prims_common.ShapeType).
A separate test ensures our definition matches PyTorch’s current definition and
will fail if PyTorch changes it in the future.
"""

from typing import List, Union
import torch

# Define EllipsisType for Python 3.9 compatibility
EllipsisType = type(...)

# Mirror torch._prims_common.ShapeType without importing it directly.
ShapeLike = Union[torch.Size, list[int], tuple[int, ...]]

# Mirror torch._prims_common.DeviceLikeType without importing it directly.
DeviceLike = Union[str, torch.device, int]

# Type for tensor indexing operations, covering all supported PyTorch indexing patterns
IndexType = Union[int, slice, torch.Tensor, tuple, EllipsisType, None, List[int]]

__all__ = ["ShapeLike", "DeviceLike", "IndexType"]
