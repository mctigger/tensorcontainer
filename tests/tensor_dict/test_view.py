import torch
import pytest
from rtd.tensor_dict import TensorDict


def test_view_basic():
    td = TensorDict({"a": torch.randn(4, 4)}, shape=(4, 4))
    td_view = td.view(16)
    assert td_view.shape == (16,)
    assert torch.equal(td_view["a"], td["a"].view(16))


def test_view_reshape():
    td = TensorDict({"a": torch.randn(4, 4)}, shape=(4, 4))
    td_view = td.view(2, 8)
    assert td_view.shape == (2, 8)
    assert torch.equal(td_view["a"], td["a"].view(2, 8))


def test_view_nested():
    td = TensorDict(
        {"a": TensorDict({"b": torch.randn(2, 2)}, shape=[2, 2])}, shape=[2, 2]
    )
    td_view = td.view(4)
    assert td_view.shape == (4,)


def test_view_multiple_keys():
    td = TensorDict({"a": torch.randn(2, 2), "b": torch.ones(2, 2)}, shape=(2, 2))
    td_view = td.view(4)
    assert td_view.shape == (4,)


def test_view_nested_shape_validation():
    # Valid case: child shape is the same as the parent shape after view
    td1 = TensorDict({"a": TensorDict({"b": torch.randn(4)}, shape=(4,))}, shape=(4,))
    td1.view(4)

    # Invalid case: child shape is not compatible with the parent shape after view
    td2 = TensorDict(
        {"a": TensorDict({"b": torch.randn(2, 2)}, shape=(2, 2))}, shape=(2, 2)
    )
    with pytest.raises(RuntimeError):
        td2.view(5)


def test_view_invalid_shape():
    td = TensorDict({"a": torch.randn(2, 2)}, shape=(2, 2))
    with pytest.raises(RuntimeError):
        td.view(3)


def test_view_single_element_tensordict():
    td = TensorDict({"a": torch.tensor([1.0])}, shape=(1,))
    assert td.shape == (1,)
    assert torch.equal(td["a"], torch.tensor([1.0]))
