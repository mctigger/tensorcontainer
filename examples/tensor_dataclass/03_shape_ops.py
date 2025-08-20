"""
Reshaping TensorDataClass instances.

Key concepts demonstrated:
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
    x: torch.Tensor
    y: torch.Tensor


def main() -> None:
    """Demonstrate reshaping operations."""
    x_tensor = torch.rand(2, 3, 4)
    y_tensor = torch.rand(2, 3, 5)

    data = Data(
        x=x_tensor,
        y=y_tensor,
        shape=(2, 3),
        device="cpu",
    )

    # Reshape batch dimensions while preserving event dimensions
    reshaped_data = data.reshape(6)

    # Verify reshape preserves total elements and event dimensions
    assert reshaped_data.shape == (6,)
    assert reshaped_data.x.shape == (6, 4)  # Event dimension (4,) preserved
    assert reshaped_data.y.shape == (6, 5)  # Event dimension (5,) preserved


if __name__ == "__main__":
    main()
