"""
Cloning a TensorDataClass to create independent tensor storage.

This example demonstrates how to use `clone()` to create a new `TensorDataClass`
instance with detached tensor storage but identical values. The cloned instance
is completely independent - mutations to its tensors do not affect the original.

Note: `copy()` provides a deepcopy-like alternative but is not shown here.
"""

import torch
from tensorcontainer import TensorDataClass


# Define a simple TensorDataClass with two tensor fields
class DataPair(TensorDataClass):
    """
    A simple data container with two tensor fields.

    Args:
        x (torch.Tensor): The first tensor field.
        y (torch.Tensor): The second tensor field.
    """

    x: torch.Tensor
    y: torch.Tensor


def main():
    # Create deterministic tensors for our example
    x = torch.rand(3, 3)
    y = torch.rand(3, 5)
    # Instantiate the original TensorDataClass
    original = DataPair(x=x, y=y, shape=(3,), device="cpu")

    # Copy the original to create a shallow copy
    copied = original.copy()
    # Shares tensors with original
    assert copied.x is original.x
    assert copied.y is original.y

    # Clone the original to create an independent copy
    cloned = original.clone()
    # Does not share tensors with original. Tensors have also been cloned.
    assert cloned.x is not original.x
    assert cloned.y is not original.y


if __name__ == "__main__":
    main()
