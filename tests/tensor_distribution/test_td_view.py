import torch
import pytest
from rtd.tensor_distribution import TensorNormal
from rtd.tensor_dict import TensorDict
from tests.tensor_dict.compile_utils import run_and_compare_compiled


def test_view_basic():
    loc = torch.randn(4, 4)
    scale = torch.rand(4, 4)
    td = TensorNormal(loc, scale, reinterpreted_batch_ndims=1, shape=(4, 4))
    td_view = td.view(16)
    assert td_view.shape == (16,)
    assert torch.equal(td_view["loc"], td["loc"].view(16))
    assert torch.equal(td_view["scale"], td["scale"].view(16))


def test_view_reshape():
    loc = torch.randn(4, 4)
    scale = torch.rand(4, 4)
    td = TensorNormal(loc, scale, reinterpreted_batch_ndims=1, shape=(4, 4))
    td_view = td.view(2, 8)
    assert td_view.shape == (2, 8)
    assert torch.equal(td_view["loc"], td["loc"].view(2, 8))
    assert torch.equal(td_view["scale"], td["scale"].view(2, 8))


def test_view_multiple_keys():
    loc1 = torch.randn(2, 2)
    scale1 = torch.rand(2, 2)
    loc2 = torch.randn(2, 2)
    scale2 = torch.rand(2, 2)
    td = TensorDict(
        {
            "a": TensorNormal(loc1, scale1, reinterpreted_batch_ndims=1, shape=(2, 2)),
            "b": TensorNormal(loc2, scale2, reinterpreted_batch_ndims=1, shape=(2, 2)),
        },
        shape=(2, 2),
    )
    td_view = td.view(4)
    assert td_view.shape == (4,)
    assert isinstance(td_view["a"], TensorNormal)
    assert isinstance(td_view["b"], TensorNormal)
    assert td_view["a"].shape == (4,)
    assert td_view["b"].shape == (4,)
    assert torch.equal(td_view["a"]["loc"], td["a"]["loc"].view(4))
    assert torch.equal(td_view["a"]["scale"], td["a"]["scale"].view(4))
    assert torch.equal(td_view["b"]["loc"], td["b"]["loc"].view(4))
    assert torch.equal(td_view["b"]["scale"], td["b"]["scale"].view(4))


def test_view_invalid_shape():
    loc = torch.randn(2, 2)
    scale = torch.rand(2, 2)
    td = TensorNormal(loc, scale, reinterpreted_batch_ndims=1, shape=(2, 2))
    with pytest.raises(RuntimeError):
        td.view(3)


def test_view_single_element_tensordict():
    loc = torch.randn(1)
    scale = torch.rand(1)
    td = TensorNormal(loc, scale, reinterpreted_batch_ndims=1, shape=(1,))
    td_view = td.view(1)
    assert td.shape == (1,)
    assert td_view.shape == (1,)
    assert torch.equal(td["loc"], td_view["loc"])
    assert torch.equal(td["scale"], td_view["scale"])


@pytest.mark.parametrize("shape", [(4,), (2, 2)])
def test_compile_view(shape):
    loc = torch.randn(*shape, requires_grad=True)
    scale = torch.rand(*shape, requires_grad=True)
    td = TensorNormal(loc, scale, reinterpreted_batch_ndims=1, shape=shape)
    loc.requires_grad_(True)
    scale.requires_grad_(True)

    def compiled_fn(td):
        return td.view(torch.Size([4]).numel())

    run_and_compare_compiled(compiled_fn, td)
