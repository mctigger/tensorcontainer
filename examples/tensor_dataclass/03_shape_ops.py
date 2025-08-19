"""Reshaping a TensorDataClass.

This example demonstrates how to use reshape() on a TensorDataClass:
- How reshape() transforms the batch dimensions of all tensor fields
- How the leading (batch) dimensions change while preserving trailing (event) dimensions
- How shapes propagate consistently across all fields in the dataclass

The reshape() method works on the batch dimensions only, leaving the event dimensions
(trailing dimensions beyond the batch shape) unchanged for each field.
"""

import torch
from tensorcontainer import TensorDataClass


class Data(TensorDataClass):
    """A simple data container with two tensor fields."""

    x: torch.Tensor  # First tensor field
    y: torch.Tensor  # Second tensor field


def main():
    # Step 1: Create tensors with batch shape (2, 3)
    x_tensor = torch.tensor([[0, 1, 2], [3, 4, 5]])  # Shape: (2, 3)
    y_tensor = torch.tensor([[10, 11, 12], [13, 14, 15]])  # Shape: (2, 3)

    # Step 2: Create TensorDataClass instance with batch shape (2, 3)
    data = Data(
        x=x_tensor,
        y=y_tensor,
        shape=(2, 3),  # Batch dimensions: 2 rows, 3 columns
        device="cpu",
    )

    # Step 3: Print original shapes and values
    print("Original data:")
    print(f"  Batch shape: {data.shape}")
    print(f"  x.shape: {data.x.shape}, x: {data.x}")
    print(f"  y.shape: {data.y.shape}, y: {data.y}")
    # Original data:
    #   Batch shape: torch.Size([2, 3])
    #   x.shape: torch.Size([2, 3]), x: tensor([[0, 1, 2],
    #         [3, 4, 5]])
    #   y.shape: torch.Size([2, 3]), y: tensor([[10, 11, 12],
    #         [13, 14, 15]])

    # Step 4: Reshape from (2, 3) to (3, 2) - same total elements, different arrangement
    reshaped_data = data.reshape(3, 2)

    # Step 5: Print reshaped shapes and values
    print("\nAfter reshape(3, 2):")
    print(f"  Batch shape: {reshaped_data.shape}")
    print(f"  x.shape: {reshaped_data.x.shape}, x: {reshaped_data.x}")
    print(f"  y.shape: {reshaped_data.y.shape}, y: {reshaped_data.y}")
    # After reshape(3, 2):
    #   Batch shape: torch.Size([3, 2])
    #   x.shape: torch.Size([3, 2]), x: tensor([[0, 1],
    #         [2, 3],
    #         [4, 5]])
    #   y.shape: torch.Size([3, 2]), y: tensor([[10, 11],
    #         [12, 13],
    #         [14, 15]])


if __name__ == "__main__":
    main()
