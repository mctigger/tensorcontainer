"""Subclassing (inheritance) of a TensorDataClass to add new fields.

This example demonstrates how to inherit from a TensorDataClass to extend it
with additional tensor fields while preserving all base class behavior and fields.
"""

import torch
from tensorcontainer import TensorDataClass


# Define a small base dataclass with one tensor field
class Base(TensorDataClass):
    x: torch.Tensor


# Define a subclass that inherits from Base and adds another tensor field
class Child(Base):
    y: torch.Tensor


if __name__ == "__main__":
    # Instantiate tensor values deterministically
    x = torch.tensor([[1, 2], [3, 4]])
    y = torch.tensor([10, 20])

    # Construct a Child instance
    data = Child(x=x, y=y, shape=(2,), device="cpu")

    # Show that the subclass instance contains both base and new fields
    print(repr(data))
    # Child(shape=torch.Size([2]), device=device(type='cpu'), x=tensor([[1, 2],
    #         [3, 4]]), y=tensor([10, 20]))

    # Print field values and their shapes
    print(f"data.x = {data.x}")
    # data.x = tensor([[1, 2],
    #         [3, 4]])

    print(f"data.y = {data.y}")
    # data.y = tensor([10, 20])

    print(f"data.x.shape = {data.x.shape}")
    # data.x.shape = torch.Size([2, 2])

    print(f"data.y.shape = {data.y.shape}")
    # data.y.shape = torch.Size([2])
