"""
Moving TensorDict instances between devices.

This example demonstrates how to move TensorDict instances between
different devices (CPU, GPU) using the `.to()` method, ensuring all tensor
values are moved consistently while maintaining the TensorDict structure.

Key concepts demonstrated:
- Device movement: How the `.to()` method moves all tensor values to a
  specified device while preserving the TensorDict structure.
- Automatic synchronization: How all tensor values are automatically moved
  to the target device in a single operation.
- Device consistency: How the TensorDict ensures all values remain
  on the same device after movement operations.
"""

import torch
from tensorcontainer import TensorDict


def main() -> None:
    """Demonstrate device movement operations."""
    x = torch.rand(2, 2)
    y = torch.rand(2, 5)

    # Create instance on CPU
    data_cpu = TensorDict({"x": x, "y": y}, shape=(2,), device="cpu")

    # Move to CUDA device
    data_cuda = data_cpu.to(device="cuda")

    # Verify all tensors moved to CUDA
    assert data_cuda.device == torch.device("cuda:0")
    assert data_cuda["x"].device == torch.device("cuda:0")
    assert data_cuda["y"].device == torch.device("cuda:0")


if __name__ == "__main__":
    main()
