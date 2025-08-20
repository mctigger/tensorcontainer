"""
Moving TensorDataClass instances between devices.

This example demonstrates how to move TensorDataClass instances to different
devices using the `.to()` method.

Key concepts demonstrated:
- Device movement with .to() method
- Automatic device synchronization for all tensor fields
- Device consistency across the entire instance
"""

import torch
from tensorcontainer import TensorDataClass


class MyData(TensorDataClass):
    x: torch.Tensor
    y: torch.Tensor


def main():
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
