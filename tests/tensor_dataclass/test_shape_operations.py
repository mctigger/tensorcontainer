import pytest
import torch
import dataclasses
from rtd.tensor_dataclass import TensorDataclass


@dataclasses.dataclass
class ShapeTestClass(TensorDataclass):
    a: torch.Tensor
    b: torch.Tensor
    meta: int = 42


def test_view():
    td = ShapeTestClass(
        a=torch.randn(4, 5),
        b=torch.ones(4, 5),
        shape=(4, 5),
        device=torch.device("cpu"),
    )

    viewed = td.view(20)
    assert viewed.shape == (20,)
    assert viewed.a.shape == (20,)
    assert viewed.b.shape == (20,)
    assert viewed.meta == 42


def test_reshape():
    td = ShapeTestClass(
        a=torch.randn(2, 6),
        b=torch.ones(2, 6),
        shape=(2, 6),
        device=torch.device("cpu"),
    )

    reshaped = td.reshape(4, 3)
    assert reshaped.shape == (4, 3)
    assert reshaped.a.shape == (4, 3)
    assert reshaped.b.shape == (4, 3)


def test_shape_inference_on_unflatten():
    original = ShapeTestClass(
        a=torch.randn(2, 5),
        b=torch.ones(2, 5),
        shape=(2, 5),
        device=torch.device("cpu"),
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
        a=torch.randn(4, 5),
        b=torch.ones(4, 5),
        shape=(4, 5),
        device=torch.device("cpu"),
    )

    with pytest.raises(RuntimeError):
        td.view(21)  # Invalid size


def test_shape_compile():
    td = ShapeTestClass(
        a=torch.randn(4, 5),
        b=torch.ones(4, 5),
        shape=(4, 5),
        device=torch.device("cpu"),
    )

    def view_fn(td):
        return td.view(20)

    compiled_fn = torch.compile(view_fn, fullgraph=True)
    result = compiled_fn(td)

    assert result.shape == (20,)
    assert result.a.shape == (20,)
