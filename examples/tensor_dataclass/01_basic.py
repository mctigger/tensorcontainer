"""
This example showcases the basic usage of `TensorDataClass`.

`TensorDataClass` is a powerful tool for creating tensor containers with a
dataclass-like API. It automatically generates a constructor from annotated
fields, similar to Python's built-in `dataclasses`.

Key concepts demonstrated in this example:
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
    """
    Demonstrates the basic functionalities of `SimpleData` (a TensorDataClass).
    """
    # Create two example tensors with a shared leading batch dimension of size 2.
    obs_tensor = torch.rand(2, 3) 
    act_tensor = torch.rand(2) 

    # Construct a SimpleData instance.
    # The constructor is auto-generated from the annotated fields.
    # Unlike plain dataclasses, `shape` and `device` must also be passed.
    # `shape` defines the leading batch dimensions shared by all fields and
    # MUST be a prefix of each tensor's shape.
    data = SimpleData(
        observations=obs_tensor,
        actions=act_tensor,
        shape=(2,),  # Batch prefix shared by all fields
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
