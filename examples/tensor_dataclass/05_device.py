"""Moving a TensorDataClass to a device using .to()

This example demonstrates how to use the .to() method to move all tensor fields
of a TensorDataClass instance to a specified device.
"""

import torch
from tensorcontainer import TensorDataClass


class DataClass(TensorDataClass):
    """Simple TensorDataClass with two tensor fields."""

    x: torch.Tensor
    y: torch.Tensor


def main():
    # Define deterministic tensors on CPU
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torch.tensor([10, 20])

    # Create TensorDataClass instance with CPU tensors
    # The shape parameter defines the batch dimensions (leading dimensions)
    data_cpu = DataClass(x=x, y=y, shape=(2,), device="cpu")
    print(f"Original data on: {data_cpu.x.device}")
    # cpu
    print(f"Original y on: {data_cpu.y.device}")
    # cpu

    # Move the entire TensorDataClass to CPU (demonstrating .to() method)
    # Note: For CUDA, you would use .to("cuda") if torch.cuda.is_available()
    data_on_cpu = data_cpu.to(device="cpu")

    # Verify all tensor fields are on the target device
    print(f"After .to('cpu'), x device: {data_on_cpu.x.device}")
    # cpu
    print(f"After .to('cpu'), y device: {data_on_cpu.y.device}")
    # cpu

    # Print the full instance representation
    print(f"Full instance: {data_on_cpu}")
    # Full instance: DataClass(shape=torch.Size([2]), device=device(type='cpu'), x=tensor([[1., 2.],
    #         [3., 4.]]), y=tensor([10, 20]))


if __name__ == "__main__":
    main()
