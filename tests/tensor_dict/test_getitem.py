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


def test_getitem_returns_new_tensordict(nested_dict):
    data = nested_dict((2, 2))
    td = TensorDict(data, shape=(2, 2))
    # slicing by an integer produces a new TensorDict
    sliced = td[1]
    assert isinstance(sliced, TensorDict)
    assert sliced is not td


def test_modify_top_level_structure_on_slice_does_not_affect_original(nested_dict):
    data = nested_dict((2, 2))
    td = TensorDict(data, shape=(2, 2))
    sliced = td[0]

    # add a new top‐level key to the slice
    sliced["extra"] = torch.tensor([1, 2, 3])
    assert "extra" in sliced
    assert "extra" not in td

    # delete an existing top‐level key from the slice
    del sliced["y"]
    assert "y" not in sliced
    assert "y" in td


def test_modify_nested_structure_on_slice_does_not_affect_original(nested_dict):
    data = nested_dict((2, 2))
    td = TensorDict(data, shape=(2, 2))
    sliced = td[1]
    nested = sliced["x"]

    # add a new nested key under "x"
    nested["c"] = torch.tensor([9, 9])
    assert "c" in nested
    assert "c" not in td["x"]

    # remove an existing nested key under "x"
    del nested["a"]
    assert "a" not in nested
    assert "a" in td["x"]


def test_slice_leaf_tensor_content_and_shape(nested_dict):
    data = nested_dict((2, 2))
    td = TensorDict(data, shape=(2, 2))

    # multi‐dimensional indexing
    slice_ = td[1, 0]
    # after slicing both batch dims, batch_shape becomes empty
    assert slice_.shape == torch.Size([])

    # leaf values match the underlying tensors
    assert torch.equal(slice_["x"]["a"], data["x"]["a"][1, 0])
    assert torch.equal(slice_["x"]["b"], data["x"]["b"][1, 0])
    assert torch.equal(slice_["y"], data["y"][1, 0])
