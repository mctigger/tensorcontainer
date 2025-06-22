import torch

from rtd.tensor_dataclass import TensorDataclass


class NdimTestClass(TensorDataclass):
    # Parent class required fields (no defaults)
    device: torch.device
    shape: tuple

    # Instance tensor fields
    a: torch.Tensor
    b: torch.Tensor

    # Optional field with default
    meta: int = 42


def test_ndim_basic():
    """Test ndim property for a TensorDataclass with multiple tensor fields."""
    td = NdimTestClass(
        a=torch.randn(2, 3, device=torch.device("cpu")),
        b=torch.ones(2, 3, device=torch.device("cpu")),
        shape=(2, 3),
        device=torch.device("cpu"),
    )

    assert td.ndim == 2  # ndim should be the length of the shape tuple
    assert td.a.ndim == 2  # Each tensor field should have the same ndim
    assert td.b.ndim == 2


def test_ndim_scalar():
    """Test ndim property for a TensorDataclass with a scalar tensor field."""
    td = NdimTestClass(
        a=torch.tensor(1.0, device=torch.device("cpu")),
        b=torch.tensor(2.0, device=torch.device("cpu")),
        shape=(),
        device=torch.device("cpu"),
    )

    assert td.ndim == 0  # ndim should be 0 for a scalar shape
    assert td.a.ndim == 0  # Each tensor field should be scalar
    assert td.b.ndim == 0


def test_ndim_empty_shape():
    """Test ndim property for a TensorDataclass with an empty shape."""
    td = NdimTestClass(
        a=torch.tensor([], device=torch.device("cpu")),
        b=torch.tensor([], device=torch.device("cpu")),
        shape=(),
        device=torch.device("cpu"),
    )

    assert td.ndim == 0  # ndim should be 0 for an empty shape
    assert td.a.ndim == 1  # Empty tensors have ndim of 1
    assert td.b.ndim == 1


def test_ndim_different_shapes():
    """Test ndim property when tensor fields have different shapes."""
    td = NdimTestClass(
        a=torch.randn(2, 3, 4, device=torch.device("cpu")),
        b=torch.ones(2, 3, device=torch.device("cpu")),
        shape=(2, 3),
        device=torch.device("cpu"),
    )

    assert td.ndim == 2  # ndim should be the length of the shape tuple
    assert td.a.ndim == 3  # a has 3 dimensions
    assert td.b.ndim == 2  # b has 2 dimensions
