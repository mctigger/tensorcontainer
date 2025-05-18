import pytest
import torch

from rtd.tensor_dict import TensorDict


def test_constructor_raises_on_incompatible_leaf_shape():
    data = {
        "a": torch.zeros(1, 5),
        "b": torch.ones(1, 5),
    }
    with pytest.raises(RuntimeError) as excinfo:
        TensorDict(data, shape=(2,))


def test_constructor_raises_in_nested_mapping():
    data = {
        "x": {
            "inner": torch.randn(3, 4, 5),
        },
        "y": torch.randn(3, 4, 5),
    }
    with pytest.raises(RuntimeError):
        TensorDict(data, shape=(2, 4))


def test_constructor_accepts_flat_dict_leading_dim_only():
    data = {
        "a": torch.arange(6).reshape(2, 3),
        "b": torch.zeros(2, 5),
    }
    td = TensorDict(data, shape=(2,))
    # shape is stored as torch.Size
    assert td.shape == torch.Size([2])
    # contents unchanged
    assert torch.equal(td["a"], data["a"])
    assert torch.equal(td["b"], data["b"])


def test_constructor_accepts_multidimensional_leading_batch():
    data = {
        "x": torch.arange(24).reshape(2, 3, 4),
        "y": torch.arange(12).reshape(2, 6),
    }
    td = TensorDict(data, shape=(2,))
    # shape only enforces the first dim
    assert td.shape == torch.Size([2])
    assert td["x"].shape == (2, 3, 4)
    assert td["y"].shape == (2, 6)


def test_constructor_accepts_nested_dict():
    data = {
        "outer": {
            "inner_a": torch.randn(3, 4),
            "inner_b": torch.zeros(3, 1),
        },
        "leaf": torch.ones(3, 2),
    }
    td = TensorDict(data, shape=(3,))
    assert td.shape == torch.Size([3])
    # nested dict structure preserved
    assert set(td.keys()) == {"outer", "leaf"}
    assert torch.equal(td["outer"]["inner_a"], data["outer"]["inner_a"])
    assert torch.equal(td["outer"]["inner_b"], data["outer"]["inner_b"])
    assert torch.equal(td["leaf"], data["leaf"])


def test_constructor_accepts_tensordict_inputs():
    base = TensorDict({"a": torch.arange(8).reshape(4, 2)}, shape=(4,))
    td = TensorDict({"nested": base}, shape=(4,))
    # nested TensorDict re-wrapped with same batch_shape
    assert isinstance(td["nested"], TensorDict)
    assert td["nested"].shape == torch.Size([4])
    assert torch.equal(td["nested"]["a"], base["a"])


def test_constructor_accepts_empty_shape():
    # empty shape () means no batch dims — everything passes
    data = {
        "a": torch.randn(5, 6, 7),
        "b": torch.zeros(2),
    }
    td = TensorDict(data, shape=())
    assert td.shape == torch.Size([])
    assert td["a"].shape == (5, 6, 7)
    assert td["b"].shape == (2,)


def test_constructor_accepts_zero_batch_size():
    data = {
        "a": torch.zeros(0, 4),
        "b": torch.zeros(
            0,
        ),
    }
    td = TensorDict(data, shape=(0,))
    assert td.shape == torch.Size([0])
    assert td["a"].shape == (0, 4)
    assert td["b"].shape == (0,)


@pytest.mark.parametrize(
    "data, shape",
    [
        # leaf too many dims
        ({"a": torch.randn(2, 3)}, (2, 3, 1)),
        # nested leaf mismatch
        (
            {
                "x": {
                    "y": torch.randn(
                        5,
                    )
                },
                "z": torch.randn(
                    5,
                ),
            },
            (4,),
        ),
    ],
)
def test_constructor_raises_on_shape_too_long(data, shape):
    with pytest.raises(RuntimeError):
        TensorDict(data, shape=shape)


def test_constructor_raises_on_unsupported_type():
    data = {
        "a": [1, 2, 3],  # list is not a Tensor or dict
        "b": torch.zeros(3, 1),
    }
    with pytest.raises(RuntimeError):
        TensorDict(data, shape=(3,))
