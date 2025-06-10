import pytest
import torch

from rtd.tensor_dict import TensorDict


@pytest.fixture
def nested_dict():
    def _make(shape):
        return {
            "x": {
                "a": torch.arange(0, 4).reshape(*shape),
                "b": torch.arange(4, 8).reshape(*shape),
            },
            "y": torch.arange(8, 12).reshape(*shape),
        }

    return _make


def test_update_adds_new_key_mapping(nested_dict):
    td = TensorDict(nested_dict((2, 2)), shape=(2, 2))
    new_tensor = torch.full((2, 2), 9)
    result = td.update({"new": new_tensor})
    # dict.update returns None
    assert result is None
    # new key has been added and matches
    assert "new" in td
    assert torch.equal(td["new"], new_tensor)


def test_update_overwrites_existing_key(nested_dict):
    td = TensorDict(nested_dict((2, 2)), shape=(2, 2))
    replacement = torch.full((2, 2), 5)
    td.update({"y": replacement})
    # old value replaced
    assert torch.equal(td["y"], replacement)


def test_update_with_kwargs(nested_dict):
    td = TensorDict(nested_dict((4, 1)), shape=(4, 1))
    w = torch.arange(4).reshape(4, 1)
    td.update({"y": w})
    assert torch.equal(td["y"], w)


def test_update_nested_mapping(nested_dict):
    td = TensorDict(nested_dict((2, 2)), shape=(2, 2))
    nested_map = {"x": {"c": torch.ones(2, 2) * 7}}
    td.update(nested_map)
    # ensure the nested dict was converted into a TensorDict
    assert isinstance(td["x"], TensorDict)
    assert "c" in td["x"]
    assert torch.equal(td["x"]["c"], nested_map["x"]["c"])


def test_update_incompatible_shape_raises(nested_dict):
    td = TensorDict(nested_dict((2, 2)), shape=(2, 2))
    bad = torch.ones(1, 3)
    with pytest.raises(RuntimeError):
        td.update({"z": bad})


def test_update_with_another_tensordict(nested_dict):
    td1 = TensorDict(nested_dict((2, 2)), shape=(2, 2))
    td2 = TensorDict({"z": torch.ones(2, 2) * 2}, shape=(2, 2))
    td1.update(td2)
    assert "z" in td1
    assert torch.equal(td1["z"], td2["z"])
