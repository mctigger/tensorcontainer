"""
This example showcases how to reshape `TensorDataClass` instances.

Reshaping a `TensorDataClass` allows you to change the batch dimensions of all
contained tensors simultaneously, while preserving their event dimensions.
This is useful for operations like flattening or unflattening batches.

Key concepts demonstrated in this example:
- Reshaping batch dimensions: How `reshape()` transforms the leading (batch)
  dimensions of all tensor fields within the `TensorDataClass`.
- Preserving event dimensions: The trailing (event) dimensions of individual
  tensors remain unchanged during the reshape operation.
- Consistent shape propagation: How shape changes propagate consistently across
  all fields, ensuring the `TensorDataClass` remains coherent.
"""

import torch
from tensorcontainer import TensorDataClass


class Data(TensorDataClass):
    """A simple data container with two tensor fields."""

    x: torch.Tensor
    y: torch.Tensor


def main() -> None:
    """
    Demonstrates reshaping functionalities of `TensorDataClass`.
    """
    # Create two example tensors with a shared leading batch dimension of (2, 3).
    x_tensor = torch.rand(2, 3, 4)
    y_tensor = torch.rand(2, 3, 5)

    # Construct a Data instance.
    # The `shape` argument defines the leading batch dimensions shared by all fields.
    data = Data(
        x=x_tensor,
        y=y_tensor,
        shape=(2, 3),
        device="cpu",
    )

    # Reshape the TensorDataClass from batch shape (2, 3) to (6,).
    # The total number of elements in the batch dimensions must remain the same.
    # Event dimensions (e.g., 4 for x, 5 for y) are preserved.
    reshaped_data = data.reshape(6)


if __name__ == "__main__":
    main()
