import pytest
import torch
import dataclasses
from typing import Optional
from rtd.tensor_dataclass import TensorDataclass


@dataclasses.dataclass
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
