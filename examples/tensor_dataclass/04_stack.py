"""Stacking multiple TensorDataClass instances along a new dimension.

This example demonstrates how to stack several TensorDataClass instances
into a single batched instance using `torch.stack()`.

It illustrates:
- How stacking creates a new leading batch dimension.
- How tensor shapes and values combine when stacking.
- How the resulting instance maintains the TensorDataClass structure.

Note: `torch.cat()` can also be used to concatenate along existing dimensions,
but this example focuses solely on stacking for clarity.
"""

import torch
from tensorcontainer import TensorDataClass


class DataPoint(TensorDataClass):
    """A simple data container with two tensor fields.

    Attributes:
        x (torch.Tensor): The first tensor field.
        y (torch.Tensor): The second tensor field.
    """

    x: torch.Tensor
    y: torch.Tensor


def main():
    """Run the example demonstrating stacking of TensorDataClass instances."""
    point1 = DataPoint(
        x=torch.arange(3),
        y=torch.arange(3),
        shape=(3,),
        device="cpu",
    )

    point2 = DataPoint(
        x=torch.arange(3),
        y=torch.arange(3),
        shape=(3,),
        device="cpu",
    )

    # Stack the three instances along a new leading dimension (dim=0).
    # This operation creates a new DataPoint of shape (2, 3).
    stacked = torch.stack([point1, point2], dim=0)


if __name__ == "__main__":
    main()
