"""
Stacking TensorDataClass instances.

This example demonstrates how to stack multiple TensorDataClass instances
along a new dimension using torch.stack, creating a new batch dimension
while preserving the TensorDataClass structure.

Key concepts demonstrated:
- Stacking multiple instances: How `torch.stack` can combine multiple
  TensorDataClass instances along a new leading dimension.
- Batch dimension creation: How stacking creates a new batch dimension
  that becomes the leading dimension of all tensor fields.
- Structure preservation: How the TensorDataClass structure and field
  relationships are maintained during the stacking operation.
- Shape consistency: How stacking ensures all tensor fields are combined
  with consistent shapes and dimensions.
"""

import torch
from tensorcontainer import TensorDataClass


class DataPoint(TensorDataClass):
    x: torch.Tensor
    y: torch.Tensor


class NotADataPoint(TensorDataClass):
    a: torch.Tensor


def main() -> None:
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

    # Attempt to stack with different TensorDataClasses
    try:
        not_a_point = NotADataPoint(a=torch.rand(3, 4), shape=(3,), device="cpu")
        torch.stack([point1, not_a_point], dim=0)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
