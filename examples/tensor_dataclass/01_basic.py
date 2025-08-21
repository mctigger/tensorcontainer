"""
Basic usage of TensorDataClass.

This example demonstrates how to define and instantiate a TensorDataClass
with shape and device parameters.

Key concepts demonstrated:
- Auto-generated constructor: How `TensorDataClass` simplifies object
  instantiation based on type hints.
- `shape` and `device` parameters: The necessity of providing `shape` and
  `device` during construction, which define the leading batch dimensions
  shared by all tensors within the `TensorDataClass` instance.
- Shape prefix requirement: The `shape` argument must be a prefix of every
  tensor's shape within the `TensorDataClass`. This ensures consistency and
  facilitates batch operations.
- Field access: How to access individual tensor fields using dot notation.
"""

import torch
from tensorcontainer import TensorDataClass


class SimpleData(TensorDataClass):
    observations: torch.Tensor
    actions: torch.Tensor


def main() -> None:
    """Demonstrate basic TensorDataClass functionality."""
    # Create tensors with shared batch dimension
    obs_tensor = torch.rand(2, 3)
    act_tensor = torch.rand(2)

    # Construct instance with required shape and device
    SimpleData(
        observations=obs_tensor,
        actions=act_tensor,
        shape=(2,),  # Batch dimensions shared by all fields
        device="cpu",
    )

    # Attempt to construct with a mismatched shape to demonstrate error handling.
    # The `shape` argument must be a prefix of every field's shape.
    try:
        SimpleData(
            observations=obs_tensor,
            actions=act_tensor,
            shape=(3,),  # This shape (3,) is not a prefix of (2, 3) or (2,)
            device="cpu",
        )
    except Exception as e:
        # An error is expected here due to the shape mismatch.
        print(e)


if __name__ == "__main__":
    main()
