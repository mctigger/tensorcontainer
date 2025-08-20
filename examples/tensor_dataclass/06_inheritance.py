"""
Inheritance patterns with TensorDataClass.

This example demonstrates how to inherit from TensorDataClass to extend
functionality while preserving base class behavior.

Key concepts demonstrated:
- Field inheritance from parent classes
- Automatic field merging in subclasses
- Consistent shape and device handling
"""

import torch
from tensorcontainer import TensorDataClass


class Base(TensorDataClass):
    x: torch.Tensor


class Child(Base):
    y: torch.Tensor


if __name__ == "__main__":
    # Create tensors with shared batch dimension
    x = torch.rand(2, 3)
    y = torch.rand(2, 5)

    # Child inherits all fields from Base
    data = Child(x=x, y=y, shape=(2,), device="cpu")

    # Verify inheritance works correctly
    assert data.shape == (2,)
    assert data.x.shape == (2, 3)
    assert data.y.shape == (2, 5)
