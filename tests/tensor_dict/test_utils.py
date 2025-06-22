import pytest
import torch

from rtd.tensor_dict import TensorDict


def _get_tensor_dict(shape, device):
    """Helper to create a TensorDict with a given batch shape."""
    # Tensors in the dict should have a shape that is prefixed by the batch shape
    data = {
        "a": torch.randn(*shape, 2, device=device),
        "b": torch.randn(*shape, 3, 4, device=device),
    }
    return TensorDict(data, shape=shape)


def _test_size(td, shape):
    assert td.size() == torch.Size(shape)


def _test_dim(td, shape):
    assert td.dim() == len(shape)


def _test_numel(td, shape):
    assert td.numel() == torch.Size(shape).numel()


@pytest.mark.skipif_no_compile
class TestTensorDictUtils:
    @pytest.mark.parametrize("shape", [(), (1,), (2, 3), (2, 3, 4)])
    @pytest.mark.parametrize("device", ["cpu"])
    @pytest.mark.parametrize("compile", [False, True])
    def test_size(self, shape, device, compile):
        td = _get_tensor_dict(shape, device)
        test_fn = torch.compile(_test_size) if compile else _test_size
        test_fn(td, shape)

    @pytest.mark.parametrize("shape", [(), (1,), (2, 3), (2, 3, 4)])
    @pytest.mark.parametrize("device", ["cpu"])
    @pytest.mark.parametrize("compile", [False, True])
    def test_dim(self, shape, device, compile):
        td = _get_tensor_dict(shape, device)
        test_fn = torch.compile(_test_dim) if compile else _test_dim
        test_fn(td, shape)

    @pytest.mark.parametrize("shape", [(), (1,), (2, 3), (2, 3, 4)])
    @pytest.mark.parametrize("device", ["cpu"])
    @pytest.mark.parametrize("compile", [False, True])
    def test_numel(self, shape, device, compile):
        td = _get_tensor_dict(shape, device)
        test_fn = torch.compile(_test_numel) if compile else _test_numel
        test_fn(td, shape)
