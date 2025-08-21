"""
Indexing and slicing TensorDataClass instances.

This example demonstrates tensor-like indexing and slicing operations
on TensorDataClass instances.

Key concepts demonstrated:
- Tensor-like indexing: How `TensorDataClass` supports integer and slice
  indexing, behaving consistently with `torch.Tensor`.
- View semantics: Indexing a `TensorDataClass` returns a new instance whose
  tensors are views of the original data. Modifications to the view are
  reflected in the original `TensorDataClass`.
- Assignment with indexing: How to assign a `TensorDataClass` instance to a
  slice of another `TensorDataClass`, provided their batch shapes match.
"""

import torch
from tensorcontainer import TensorDataClass


class DataPair(TensorDataClass):
    x: torch.Tensor
    y: torch.Tensor


def main() -> None:
    """Demonstrate indexing and slicing operations."""
    # Create batch of tensors
    x = torch.rand(3, 4)
    y = torch.rand(3)
    data = DataPair(x=x, y=y, shape=(3,), device="cpu")

    # Test indexing returns correct shapes
    single_item = data[0]
    assert single_item.shape == ()
    assert single_item.x.shape == (4,)
    assert single_item.y.shape == ()

    # Test slicing returns correct shapes
    slice_data = data[1:3]
    assert slice_data.shape == (2,)
    assert slice_data.x.shape == (2, 4)
    assert slice_data.y.shape == (2,)

    # You can assign to an indexed TensorDataClass using another instance with
    # matching batch shape.
    replacement = DataPair(
        x=torch.rand(2, 4),
        y=torch.rand(2),
        shape=(2,),
        device="cpu",
    )
    data[1:3] = replacement


if __name__ == "__main__":
    main()
