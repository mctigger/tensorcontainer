"""
This example demonstrates how to move a TensorDataClass instance to a specified device
using the `.to()` method.
"""

import torch
from tensorcontainer import TensorDataClass


class MyData(TensorDataClass):
    """
    A simple TensorDataClass with two tensor fields, `x` and `y`.
    """

    x: torch.Tensor
    y: torch.Tensor


def main():
    x = torch.rand(2, 2)
    y = torch.rand(2, 5)

    # Create a TensorDataClass instance with the CPU tensors.
    # The `shape` parameter defines the batch dimensions (leading dimensions) of the tensors.
    data_cpu = MyData(x=x, y=y, shape=(2,), device="cpu")

    data_cuda = data_cpu.to(device="cuda")

    # The TensorDataClass and all its tensors are moved to the device
    assert data_cuda.device == torch.device("cuda:0")
    assert data_cuda.x.device == torch.device("cuda:0")
    assert data_cuda.y.device == torch.device("cuda:0")


if __name__ == "__main__":
    main()
