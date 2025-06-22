from typing import Optional

import pytest
import torch

from rtd.tensor_dataclass import TensorDataclass


class ShapeTestClass(TensorDataclass):
    shape: tuple
    device: Optional[torch.device]
    a: torch.Tensor
    b: torch.Tensor
    meta: int = 42


# Note: The view and reshape tests have been moved to test_view_reshape.py
# This file now focuses on shape inference and other shape-related operations


def test_shape_inference_on_unflatten():
    original = ShapeTestClass(
        shape=(2, 5),
        device=torch.device("cpu"),
        a=torch.randn(2, 5),
        b=torch.ones(2, 5),
    )

    # Flatten and modify leaves
    leaves, context = original._pytree_flatten()
    modified_leaves = [t * 2 for t in leaves]

    # Reconstruct with original context
    reconstructed = ShapeTestClass._pytree_unflatten(modified_leaves, context)
    assert reconstructed.shape == (2, 5)
    assert reconstructed.a.shape == (2, 5)


def test_invalid_shape_raises():
    td = ShapeTestClass(
        shape=(4, 5),
        device=torch.device("cpu"),
        a=torch.randn(4, 5),
        b=torch.ones(4, 5),
    )

    with pytest.raises(RuntimeError):
        td.view(21)  # Invalid size


def test_shape_compile():
    td = ShapeTestClass(
        shape=(4, 5),
        device=torch.device("cpu"),
        a=torch.randn(4, 5),
        b=torch.ones(4, 5),
    )

    def view_fn(td):
        return td.view(20)

    compiled_fn = torch.compile(view_fn, fullgraph=True)
    result = compiled_fn(td)

    assert result.shape == (20,)
    assert result.a.shape == (20,)


def test_zero_sized_batch():
    """Test initialization and operations with a batch size of 0."""
    td = ShapeTestClass(
        shape=(0, 10),
        device=torch.device("cpu"),
        a=torch.randn(0, 10),
        b=torch.randn(0, 10),
    )

    assert td.shape == (0, 10)
    assert td.a.shape == (0, 10)
    assert td.b.shape == (0, 10)

    # Test clone
    cloned_td = td.clone()
    assert cloned_td.shape == (0, 10)
    assert torch.equal(cloned_td.a, td.a)

    # Test stack
    stacked_td = torch.stack([td, td], dim=0)
    assert stacked_td.shape == (2, 0, 10)


def test_inconsistent_trailing_shapes():
    """Test initialization with tensors that have different trailing shapes."""
    try:
        td = ShapeTestClass(
            shape=(4,),
            device=torch.device("cpu"),
            a=torch.randn(4, 10),
            b=torch.randn(4, 5),  # Different trailing dimension
        )
        assert td.shape == (4,)
        assert td.a.shape == (4, 10)
        assert td.b.shape == (4, 5)
    except ValueError:
        pytest.fail("Initialization failed with inconsistent trailing shapes.")


def test_no_tensor_fields():
    """Test a TensorDataclass with no tensor fields."""

    class NoTensorData(TensorDataclass):
        shape: tuple
        device: Optional[torch.device]
        meta: str

    # Initialization
    td = NoTensorData(shape=(2, 3), device=torch.device("cpu"), meta="test")
    assert td.shape == (2, 3)
    assert td.device == torch.device("cpu")
    assert td.meta == "test"

    # Clone
    cloned_td = td.clone()
    assert cloned_td.shape == (2, 3)
    assert cloned_td.meta == "test"
