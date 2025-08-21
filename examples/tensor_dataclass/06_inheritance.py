"""
Inheritance patterns with TensorDataClass.

This example demonstrates how to create inheritance hierarchies with
TensorDataClass, showing how subclasses automatically inherit and merge
fields from parent classes while maintaining consistent shape and device
handling across the inheritance chain.

Key concepts demonstrated:
- Field inheritance: How subclasses automatically inherit all tensor fields
  from their parent TensorDataClass.
- Shape consistency: How inherited classes maintain consistent shape
  requirements across all fields in the inheritance hierarchy.
- Device handling: How device specifications are consistently applied
  across all inherited fields.
"""

import torch
from tensorcontainer import TensorDataClass


class Base(TensorDataClass):
    x: torch.Tensor


class Child(Base):
    y: torch.Tensor


def main() -> None:
    """Demonstrate inheritance patterns with TensorDataClass."""
    # Create tensors with shared batch dimension
    x = torch.rand(2, 3)
    y = torch.rand(2, 5)

    # Child inherits all fields from Base
    data = Child(x=x, y=y, shape=(2,), device="cpu")

    # Verify inheritance works correctly
    assert data.shape == (2,)
    assert data.x.shape == (2, 3)
    assert data.y.shape == (2, 5)


if __name__ == "__main__":
    main()
