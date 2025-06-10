import pytest
import torch

from rtd.tensor_dict import TensorDict


@pytest.fixture
def simple_nested():
    # A TensorDict with one level of nesting
    data = {
        "x": {
            "a": torch.tensor([[1, 2], [3, 4]]),
            "b": torch.tensor([[5, 6], [7, 8]]),
        },
        "y": torch.tensor([[9, 10], [11, 12]]),
    }
    return TensorDict(data, shape=(2, 2))


@pytest.fixture
def deep_nested():
    # A deeper nested structure
    data = {
        "a": {"b": {"c": torch.arange(4).reshape(2, 2)}},
        "d": {"e": torch.ones(2, 2)},
    }
    return TensorDict(data, shape=(2, 2))


def test_flatten_keys_simple(simple_nested):
    td = simple_nested
    flat = td.flatten_keys()
    # new object, original unmodified
    assert isinstance(flat, TensorDict)
    assert flat is not td
    assert set(td.keys()) == {"x", "y"}
    # flat keys
    assert set(flat.keys()) == {"x.a", "x.b", "y"}
    # values preserved
    assert torch.equal(flat["x.a"], td["x"]["a"])
    assert torch.equal(flat["x.b"], td["x"]["b"])
    assert torch.equal(flat["y"], td["y"])
    # shape unchanged
    assert flat.shape == td.shape


def test_flatten_keys_custom_sep(simple_nested):
    td = simple_nested
    flat = td.flatten_keys()
    assert set(flat.keys()) == {"x.a", "x.b", "y"}
    assert torch.equal(flat["x.a"], td["x"]["a"])
    assert torch.equal(flat["x.b"], td["x"]["b"])


def test_flatten_keys_deep(deep_nested):
    td = deep_nested
    flat = td.flatten_keys()
    # deep keys
    assert set(flat.keys()) == {"a.b.c", "d.e"}
    assert torch.equal(flat["a.b.c"], td["a"]["b"]["c"])
    assert torch.equal(flat["d.e"], td["d"]["e"])
    assert flat.shape == td.shape


def test_flatten_keys_idempotent_on_flat():
    # apply flatten_keys twice yields same as once
    data = {"z": torch.zeros(3, 4)}
    td = TensorDict(data, shape=(3, 4))
    flat1 = td.flatten_keys()
    flat2 = flat1.flatten_keys()
    assert set(flat1.keys()) == {"z"}
    assert set(flat2.keys()) == {"z"}
    assert flat1.shape == flat2.shape
    assert torch.equal(flat1["z"], flat2["z"])


def test_flatten_keys_empty():
    td = TensorDict({}, shape=())
    flat = td.flatten_keys()
    assert isinstance(flat, TensorDict)
    assert flat is not td
    assert list(flat.keys()) == []
    assert flat.shape == td.shape


def test_flatten_keys_complex():
    # A more complex nested structure
    data = {
        "a": {
            "b": {
                "c": torch.arange(4).reshape(2, 2),
                "d": torch.ones(2, 2),
            },
            "e": torch.zeros(2, 2),
        },
        "f": torch.full((2, 2), 2),
    }
    td = TensorDict(data, shape=(2, 2))
    flat = td.flatten_keys()
    # deep keys
    assert set(flat.keys()) == {"a.b.c", "a.b.d", "a.e", "f"}
    assert torch.equal(flat["a.b.c"], td["a"]["b"]["c"])
    assert torch.equal(flat["a.b.d"], td["a"]["b"]["d"])
    assert torch.equal(flat["a.e"], td["a"]["e"])
    assert torch.equal(flat["f"], td["f"])
    assert flat.shape == td.shape
