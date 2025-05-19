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


def test_copy_returns_distinct_tensordict_but_shares_leaf_tensors(nested_dict):
    data = nested_dict((2, 2))
    td = TensorDict(data, shape=(2, 2))
    td_copy = td.copy()

    # top‐level object is new
    assert td_copy is not td
    # shape and device preserved
    assert td_copy.shape == td.shape
    assert td_copy.device == td.device

    for key, val in td.items():
        copied_val = td_copy[key]
        # nested TensorDicts should be new objects
        if isinstance(val, TensorDict):
            assert isinstance(copied_val, TensorDict)
            assert copied_val is not val
            # their leaves must still be the same tensor objects
            for leaf_key, leaf in val.items():
                assert copied_val[leaf_key] is leaf
        else:
            # leaf tensors should be the same object
            assert copied_val is val


def test_mutating_nested_copy_does_not_affect_original(nested_dict):
    data = nested_dict((4, 1))
    td = TensorDict(data, shape=(4, 1))
    td_copy = td.copy()

    # modify the copy's nested structure
    td_copy["x"]["c"] = torch.tensor([[42], [43], [44], [45]])
    assert "c" in td_copy["x"]
    # original remains unchanged
    assert "c" not in td["x"]

    # remove a leaf from the copy
    del td_copy["x"]["a"]
    assert "a" not in td_copy["x"]
    # original still has its leaf
    assert "a" in td["x"]


def test_copy_of_empty_tensor_dict(nested_dict):
    # an empty dict should still copy correctly
    td = TensorDict({}, shape=())
    td_copy = td.copy()
    assert isinstance(td_copy, TensorDict)
    assert td_copy is not td
    assert td_copy.shape == torch.Size([])
    assert len(td_copy) == 0
