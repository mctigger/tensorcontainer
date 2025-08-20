"""
Stacking TensorDataClass instances.

This example demonstrates stacking multiple TensorDataClass instances
along a new dimension.

Key concepts demonstrated:
- Creating new batch dimension with torch.stack
- Maintaining TensorDataClass structure
- Shape and value combination during stacking
"""

import torch
from tensorcontainer import TensorDataClass


class DataPoint(TensorDataClass):
    x: torch.Tensor
    y: torch.Tensor


def main():
    """Demonstrate stacking TensorDataClass instances."""
    point1 = DataPoint(
        x=torch.rand(3, 4),
        y=torch.rand(3, 5),
        shape=(3,),
        device="cpu",
    )

    point2 = DataPoint(
        x=torch.rand(3, 4),
        y=torch.rand(3, 5),
        shape=(3,),
        device="cpu",
    )

    # Stack instances along new leading dimension
    stacked = torch.stack([point1, point2], dim=0)

    # Verify stacking creates new batch dimension
    assert stacked.shape == (2, 3)
    assert stacked.x.shape == (2, 3, 4)
    assert stacked.y.shape == (2, 3, 5)


if __name__ == "__main__":
    main()
