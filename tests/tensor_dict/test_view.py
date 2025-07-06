import pytest
import torch

from tensorcontainer.tensor_dict import TensorDict


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
        {"a": TensorDict({"b": torch.randn(2, 2)}, shape=(2, 2))}, shape=(2, 2)
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


# New Tests for Edge Cases


def test_view_with_minus_one():
    """Tests if view can infer a dimension when -1 is used."""
    td = TensorDict({"a": torch.randn(4, 4, 4)}, shape=(4, 4))
    td_view = td.view(-1)
    assert td_view.shape == (16,)
    assert td_view["a"].shape == (16, 4)

    td_view_2 = td.view(2, -1)
    assert td_view_2.shape == (2, 8)
    assert td_view_2["a"].shape == (2, 8, 4)


def test_view_on_non_contiguous():
    """Tests that view raises an error on a non-contiguous tensor."""
    non_contiguous_tensor = torch.randn(4, 4).t()
    td = TensorDict({"a": non_contiguous_tensor}, shape=(4, 4))
    assert not td["a"].is_contiguous()
    with pytest.raises(
        RuntimeError,
        match="view size is not compatible with input tensor's size and stride",
    ):
        td.view(16)


def test_view_with_different_dtypes():
    """Tests that view works with tensors of different dtypes."""
    td = TensorDict(
        {"a": torch.randn(2, 4), "b": torch.randint(10, (2, 4), dtype=torch.int8)},
        shape=(2, 4),
    )
    td_view = td.view(8)
    assert td_view.shape == (8,)
    assert torch.equal(td_view["a"], td["a"].reshape(8))
    assert torch.equal(td_view["b"], td["b"].reshape(8))
    assert td_view["b"].dtype == torch.int8


def test_view_to_scalar_shape():
    """Tests viewing a single-element TensorDict to a scalar shape."""
    td = TensorDict({"a": torch.tensor(5.0)}, shape=())
    td_view = td.view(())
    assert td_view.shape == ()
    assert torch.equal(td_view["a"], td["a"])


def test_view_mixed_nested_and_tensor():
    """Tests viewing a TensorDict with mixed tensor and nested TensorDict values."""
    td = TensorDict(
        {
            "a": torch.randn(2, 2),
            "nested": TensorDict({"b": torch.ones(2, 2)}, shape=(2, 2)),
        },
        shape=(2, 2),
    )
    td_view = td.view(4)
    assert td_view.shape == (4,)
    assert torch.equal(td_view["a"], td["a"].reshape(4))
    assert td_view["nested"].shape == (4,)
    assert torch.equal(td_view["nested"]["b"], td["nested"]["b"].reshape(4))
