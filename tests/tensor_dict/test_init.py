import pytest
from rtd.tensor_dict import TensorDict
import torch
from tests.tensor_dict.compile_utils import run_and_compare_compiled
from tests.tensor_dict import common


def test_valid_shape_prefix():
    with pytest.raises(
        ValueError,
        match="Shape mismatch for key 'a': expected \\[2\\] or a prefix, got torch.Size\\(\\[1, 2\\]\\)",
    ):
        TensorDict({"a": torch.arange(2, 4).reshape(1, 2)}, shape=[2])


nested_dict = common.nested_dict


def test_empty_tensordict():
    TensorDict({}, shape=[2, 2])


def test_init_compiled(nested_dict):
    shape = (2, 2)

    def init_fn(data, shape):
        return TensorDict(data, shape=shape)

    run_and_compare_compiled(init_fn, nested_dict(shape), shape)


def test_init_with_mixed_types():
    data = {"a": torch.arange(2), "b": torch.tensor(1), "c": "string"}
    with pytest.raises(
        ValueError,
        match="Shape mismatch for key 'b': expected \\[2\\] or a prefix, got torch.Size\\(\\[]\\)",
    ):
        TensorDict(data, shape=[2])


def test_init_with_zero_sized_tensor():
    data = {"a": torch.empty(0)}
    with pytest.raises(
        ValueError,
        match="Shape mismatch for key 'a': expected \\[2, 2\\] or a prefix, got torch.Size\\(\\[0\\]\\)",
    ):
        TensorDict(data, shape=[2, 2])


def test_init_with_scalar_tensor():
    data = {"a": torch.tensor(1)}
    td = TensorDict(data, shape=[])
    assert td["a"].shape == torch.Size([])
    assert torch.equal(td["a"], torch.tensor(1))
