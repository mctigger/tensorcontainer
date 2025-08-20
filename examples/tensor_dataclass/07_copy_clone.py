"""
Copying and cloning TensorDataClass instances.

This example demonstrates the difference between copy() and clone() methods
for creating new TensorDataClass instances.

Key concepts demonstrated:
- Shallow copy with shared tensor storage
- Deep clone with independent tensor storage
- Memory sharing behavior
"""

import torch
from tensorcontainer import TensorDataClass


class DataPair(TensorDataClass):
    x: torch.Tensor
    y: torch.Tensor


def main():
    """Demonstrate copy vs clone operations."""
    # Create tensors
    x = torch.rand(3, 3)
    y = torch.rand(3, 5)
    original = DataPair(x=x, y=y, shape=(3,), device="cpu")

    # Shallow copy shares tensor storage
    copied = original.copy()
    assert copied.x is original.x  # Same tensor objects
    assert copied.y is original.y

    # Deep clone creates independent tensor storage
    cloned = original.clone()
    assert cloned.x is not original.x  # Different tensor objects
    assert cloned.y is not original.y


if __name__ == "__main__":
    main()
