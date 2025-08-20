"""
This example demonstrates how to inherit from a TensorDataClass to extend it
with additional tensor fields while preserving all base class behavior and fields.
"""

import torch
from tensorcontainer import TensorDataClass


class Base(TensorDataClass):
    """
    A base TensorDataClass with a single tensor field.
    """
    x: torch.Tensor


class Child(Base):
    """
    A subclass of Base that adds an additional tensor field.
    """
    y: torch.Tensor


if __name__ == "__main__":
    # Define tensor values for instantiation.
    x = torch.rand(2, 3)
    y = torch.rand(2, 5)

    # Create an instance of Child, which includes fields from both Base and Child.
    data = Child(x=x, y=y, shape=(2,), device="cpu")
