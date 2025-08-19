"""Stacking multiple TensorDataClass instances along a new dimension.

This example demonstrates how to stack several TensorDataClass instances
into a single batched instance using torch.stack():
- How stacking creates a new leading batch dimension
- How tensor shapes and values combine when stacking
- How the resulting instance maintains TensorDataClass structure

Note: torch.cat() can also be used to concatenate along existing dimensions,
but this example focuses solely on stacking to keep the demonstration clear.
"""

import torch
from tensorcontainer import TensorDataClass


# Step 1: Define a simple TensorDataClass with two tensor fields
class DataPoint(TensorDataClass):
    """A simple data container with two tensor fields."""

    x: torch.Tensor  # First tensor field
    y: torch.Tensor  # Second tensor field


def main():
    # Step 2: Create three deterministic instances with simple tensor data
    # Each instance has shape=(3,) representing 3 elements per field
    point1 = DataPoint(
        x=torch.tensor([1, 2, 3]),  # Shape: (3,)
        y=torch.tensor([10, 20, 30]),  # Shape: (3,)
        shape=(3,),  # Batch shape: single dimension of size 3
        device="cpu",
    )

    point2 = DataPoint(
        x=torch.tensor([4, 5, 6]),  # Shape: (3,)
        y=torch.tensor([40, 50, 60]),  # Shape: (3,)
        shape=(3,),  # Same batch shape as point1
        device="cpu",
    )

    point3 = DataPoint(
        x=torch.tensor([7, 8, 9]),  # Shape: (3,)
        y=torch.tensor([70, 80, 90]),  # Shape: (3,)
        shape=(3,),  # Same batch shape as point1 and point2
        device="cpu",
    )

    # Step 3: Stack the three instances along a new leading dimension (dim=0)
    # This creates a new batch dimension, combining 3 instances into 1 batched instance
    stacked = torch.stack([point1, point2, point3], dim=0)

    # Step 4: Print the stacked result and examine the shapes
    print("Stacked result:")
    print(stacked)
    # Stacked result:
    # DataPoint(shape=torch.Size([3, 3]), device=device(type='cpu'), x=tensor([[1, 2, 3],
    #         [4, 5, 6],
    #         [7, 8, 9]]), y=tensor([[10, 20, 30],
    #         [40, 50, 60],
    #         [70, 80, 90]]))

    print(f"\nOriginal shape per instance: {point1.shape}")
    # Original shape per instance: torch.Size([3])

    print(f"Stacked shape: {stacked.shape}")
    # Stacked shape: torch.Size([3, 3])

    print(f"Stacked x tensor shape: {stacked.x.shape}")
    # Stacked x tensor shape: torch.Size([3, 3])

    print(f"Stacked y tensor shape: {stacked.y.shape}")
    # Stacked y tensor shape: torch.Size([3, 3])

    # Step 5: Show the actual tensor values after stacking
    print(f"\nStacked x values:\n{stacked.x}")
    # Stacked x values:
    # tensor([[1, 2, 3],
    #         [4, 5, 6],
    #         [7, 8, 9]])

    print(f"\nStacked y values:\n{stacked.y}")
    # Stacked y values:
    # tensor([[10, 20, 30],
    #         [40, 50, 60],
    #         [70, 80, 90]])


if __name__ == "__main__":
    main()
