"""
Copying and cloning TensorDataClass instances.

This example demonstrates the difference between copy() and clone() methods
for creating new TensorDataClass instances, highlighting the distinction
between shallow copying (shared storage) and deep cloning (independent storage).

Key concepts demonstrated:
- Shallow copy behavior: How `copy()` creates a new TensorDataClass instance
  that shares the same underlying tensor storage as the original.
- Deep clone behavior: How `clone()` creates a completely independent
  TensorDataClass instance with its own tensor storage.
- Memory sharing implications: Understanding when tensor data is shared
  versus when it is duplicated in memory.
- Identity vs equality: How to distinguish between shallow copies and
  deep clones using identity checks on tensor objects.
"""

import torch
from tensorcontainer import TensorDataClass


class DataPair(TensorDataClass):
    x: torch.Tensor
    y: torch.Tensor


def main() -> None:
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
