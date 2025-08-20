"""
This example showcases how `TensorDataClass` instances can be indexed and sliced
in a manner similar to `torch.Tensor`.

Indexing `TensorDataClass` instances allows for flexible access to sub-batches
or individual items, while maintaining the `TensorDataClass` structure.
This is crucial for operations that require processing parts of a batch,
such as mini-batching in training loops or selecting specific data points.

Key concepts demonstrated in this example:
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
    """
    Demonstrates indexing and slicing functionalities of `TensorDataClass`.
    """
    # Create a small batch of tensors. The leading dimension is the batch (B).
    # Indexing behaves like with torch.Tensor and returns a TensorDataClass.
    x = torch.rand(3, 4) 
    y = torch.rand(3)  
    data = DataPair(x=x, y=y, shape=(3,), device="cpu")

    data[0] # Return a DataPair of shape ()
    data[1:3] # Return a DataPair of shape (2,)

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

