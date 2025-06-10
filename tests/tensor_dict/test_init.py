from rtd.tensor_dict import TensorDict
import torch
import pytest


def test_shape_mismatch():
    with pytest.raises(
        ValueError,
        match="Shape mismatch for key 'a': expected \\[2\\] or a prefix, got torch.Size\\(\\[1\\]\\)",
    ):
        TensorDict({"a": torch.arange(1)}, shape=[2])


def test_nested_shape_mismatch():
    with pytest.raises(
        ValueError,
        match="Shape mismatch for key 'c': expected \\[2\\] or a prefix, got torch.Size\\(\\[1\\]\\)",
    ):
        TensorDict(
            {"a": torch.arange(2), "b": TensorDict({"c": torch.arange(1)}, shape=[2])},
            shape=[2],
        )


def test_valid_shape_prefix():
    TensorDict({"a": torch.arange(2, 4).reshape(2, 1)}, shape=[2])


def test_empty_tensordict():
    TensorDict({}, shape=[2, 2])
