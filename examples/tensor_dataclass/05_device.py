"""
Moving TensorDataClass instances between devices.

This example demonstrates how to move TensorDataClass instances between
different devices (CPU, GPU) using the `.to()` method, ensuring all tensor
fields are moved consistently while maintaining the TensorDataClass structure.

Key concepts demonstrated:
- Device movement: How the `.to()` method moves all tensor fields to a
  specified device while preserving the TensorDataClass structure.
- Automatic synchronization: How all tensor fields are automatically moved
  to the target device in a single operation.
- Device consistency: How the TensorDataClass ensures all fields remain
  on the same device after movement operations.
"""

import torch
from tensorcontainer import TensorDataClass


class MyData(TensorDataClass):
    x: torch.Tensor
    y: torch.Tensor


def main() -> None:
    """Demonstrate device movement operations."""
    x = torch.rand(2, 2)
    y = torch.rand(2, 5)

    # Create instance on CPU
    data_cpu = MyData(x=x, y=y, shape=(2,), device="cpu")

    # Move to CUDA device
    data_cuda = data_cpu.to(device="cuda")

    # Verify all tensors moved to CUDA
    assert data_cuda.device == torch.device("cuda:0")
    assert data_cuda.x.device == torch.device("cuda:0")
    assert data_cuda.y.device == torch.device("cuda:0")


if __name__ == "__main__":
    main()
