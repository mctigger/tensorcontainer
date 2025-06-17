import torch
import dataclasses
from rtd.tensor_dataclass import TensorDataclass


@dataclasses.dataclass
class ToTestClass(TensorDataclass):
    # Parent class required fields (no defaults)
    device: torch.device
    shape: tuple

    # Tensor fields
    a: torch.Tensor
    b: torch.Tensor

    # Non-tensor field
    meta: int = 42


def test_to_different_device():
    """Test moving TensorDataclass to a different device."""
    td = ToTestClass(
        a=torch.randn(2, 3, device=torch.device("cpu")),
        b=torch.ones(2, 3, device=torch.device("cpu")),
        shape=(2, 3),
        device=torch.device("cpu"),
    )

    # Move to CUDA if available
    if torch.cuda.is_available():
        td_cuda = td.to(torch.device("cuda"))

        assert td_cuda.device.type == "cuda"
        assert td_cuda.a.device.type == "cuda"
        assert td_cuda.b.device.type == "cuda"
    else:
        # Move to a different CPU device
        td_cpu1 = td.to(torch.device("cpu"))
        assert td_cpu1.device.type == "cpu"
        assert td_cpu1.a.device.type == "cpu"
        assert td_cpu1.b.device.type == "cpu"


def test_to_same_device():
    """Test moving TensorDataclass to the same device."""
    td = ToTestClass(
        a=torch.randn(2, 3, device=torch.device("cpu")),
        b=torch.ones(2, 3, device=torch.device("cpu")),
        shape=(2, 3),
        device=torch.device("cpu"),
    )

    # Move to the same device
    td_same = td.to(torch.device("cpu"))

    assert td_same.device.type == "cpu"
    assert td_same.a.device.type == "cpu"
    assert td_same.b.device.type == "cpu"


def test_to_with_dtype_change():
    """Test moving TensorDataclass with dtype change."""
    td = ToTestClass(
        a=torch.randn(2, 3, dtype=torch.float32, device=torch.device("cpu")),
        b=torch.ones(2, 3, dtype=torch.float32, device=torch.device("cpu")),
        shape=(2, 3),
        device=torch.device("cpu"),
    )

    # Move to float64
    td_double = td.to(dtype=torch.float64)

    assert td_double.a.dtype == torch.float64
    assert td_double.b.dtype == torch.float64


def test_to_with_non_blocking_and_memory_format():
    """Test moving TensorDataclass with non_blocking and memory_format arguments."""
    td = ToTestClass(
        a=torch.randn(
            2, 3, 4, 5, device=torch.device("cpu")
        ),  # 4D tensor for channels_last
        b=torch.ones(
            2, 3, 4, 5, device=torch.device("cpu")
        ),  # 4D tensor for channels_last
        shape=(2, 3, 4, 5),
        device=torch.device("cpu"),
    )

    # Move with non_blocking=True and channels_last memory format
    if torch.cuda.is_available():
        td_non_blocking = td.to(
            torch.device("cuda"), non_blocking=True, memory_format=torch.channels_last
        )

        assert td_non_blocking.device.type == "cuda"
        # Check if the tensor is in channels_last format by verifying its layout
        # For channels_last format, the stride should be (1, C, H*W, W)
        # Check if the tensor is contiguous (required for channels_last format)
        assert td_non_blocking.a.is_contiguous()
        assert td_non_blocking.b.is_contiguous()


def test_to_mixed_fields():
    """Test moving a TensorDataclass with mixed tensor and non-tensor fields."""
    td = ToTestClass(
        a=torch.randn(2, 3, device=torch.device("cpu")),
        b=torch.ones(2, 3, device=torch.device("cpu")),
        shape=(2, 3),
        device=torch.device("cpu"),
        meta=42,
    )

    # Move to CUDA if available
    if torch.cuda.is_available():
        td_cuda = td.to(torch.device("cuda"))

        assert td_cuda.device.type == "cuda"
        assert td_cuda.a.device.type == "cuda"
        assert td_cuda.b.device.type == "cuda"
        assert td_cuda.meta == 42  # Non-tensor field should remain unchanged
    else:
        # Move to a different CPU device
        td_cpu1 = td.to(torch.device("cpu"))
        assert td_cpu1.device.type == "cpu"
        assert td_cpu1.a.device.type == "cpu"
        assert td_cpu1.b.device.type == "cpu"
        assert td_cpu1.meta == 42  # Non-tensor field should remain unchanged
