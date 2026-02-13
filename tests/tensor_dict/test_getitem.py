"""
Tests for TensorDict.__getitem__ functionality.

This module contains test classes that verify the getitem behavior of TensorDict,
including basic slicing, zero-dimensional indexing, and advanced indexing patterns.
"""

import pytest
import torch

from tensorcontainer.tensor_dict import TensorDict


def assert_tensor_indexing_parity(
    td_result: TensorDict,
    batch_shape: tuple[int, ...],
    index,
):
    """Assert that indexing a TensorDict produces the same batch shape as
    indexing a plain torch.Tensor with identical leading dimensions.

    This catches any drift between TensorDict's indexing semantics and
    PyTorch's native tensor indexing.
    """
    ref = torch.zeros(batch_shape)
    assert td_result.shape == ref[index].shape, (
        f"Shape mismatch: TensorDict {td_result.shape} vs Tensor {ref[index].shape} for index {index!r}"
    )


@pytest.fixture
def zero_dim_td():
    """A 0-dim TensorDict with a scalar and a vector leaf."""
    return TensorDict(
        {"scalar": torch.tensor(5.0), "vec": torch.zeros(3)},
        shape=(),
    )


# ---------------------------------------------------------------------------
# Index constants
# ---------------------------------------------------------------------------

BASIC_INDICES = [0, -1, slice(0, 2), slice(None, -1), slice(None, None, 2), ..., None]
ADVANCED_INDICES = [[0, 1], torch.tensor([0]), torch.tensor([0, 1])]
BOOLEAN_INDICES = [torch.tensor([True, False])]
MULTIDIM_INDICES = [(1, slice(None)), (slice(None), 1), (slice(0, 2), slice(None))]
EDGE_CASE_INDICES = [slice(0, 0), slice(1, 1)]

ALL_GETITEM_INDICES = (
    BASIC_INDICES
    + ADVANCED_INDICES
    + BOOLEAN_INDICES
    + MULTIDIM_INDICES
    + EDGE_CASE_INDICES
)


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestGetitemBasic:
    """Tests core getitem behavior: return type, shape, content, and device."""

    def test_getitem_returns_new_tensordict(self, nested_dict):
        """Indexing by int returns a new TensorDict with correct batch shape."""
        data = nested_dict((2, 2))
        td = TensorDict(data, shape=(2, 2))
        # slicing by an integer produces a new TensorDict
        sliced = td[1]
        assert isinstance(sliced, TensorDict)
        assert sliced is not td

        assert_tensor_indexing_parity(sliced, (2, 2), 1)

    def test_slice_leaf_tensor_content_and_shape(self, nested_dict):
        """Multi-dim indexing produces correct batch shape and leaf tensor values."""
        data = nested_dict((2, 2))
        td = TensorDict(data, shape=(2, 2))

        # multi‐dimensional indexing
        slice_ = td[1, 0]
        # after slicing both batch dims, batch_shape becomes empty
        assert slice_.shape == torch.Size([])

        assert_tensor_indexing_parity(slice_, (2, 2), (1, 0))

        # leaf values match the underlying tensors
        assert torch.equal(slice_["x"]["a"], data["x"]["a"][1, 0])
        assert torch.equal(slice_["x"]["b"], data["x"]["b"][1, 0])
        assert torch.equal(slice_["y"], data["y"][1, 0])

    def test_slice_preserves_device(self, nested_dict, device):
        """Sliced container and all its leaves stay on the original device."""
        data = nested_dict((2, 2))

        def to_device(obj):
            if isinstance(obj, torch.Tensor):
                return obj.to(device)
            return {k: to_device(v) for k, v in obj.items()}

        data = to_device(data)

        td = TensorDict(data, shape=(2, 2), device=device)
        sliced = td[0]

        assert sliced.device.type == device.type
        assert sliced["y"].device.type == device.type
        assert sliced["x"]["a"].device.type == device.type
        assert sliced["x"]["b"].device.type == device.type

    def test_getitem_preserves_nested_batch_shape(self):
        """Indexing a nested TensorDict slices the child's batch shape too."""
        inner = TensorDict({"a": torch.zeros(4, 3, 5)}, shape=(4, 3))
        outer = TensorDict({"inner": inner, "b": torch.zeros(4, 3)}, shape=(4, 3))
        result = outer[0]
        assert result.shape == torch.Size([3])
        assert result["inner"].shape == torch.Size([3])
        assert result["inner"]["a"].shape == torch.Size([3, 5])

    def test_getitem_single_dim_container(self):
        """Int, slice, and tensor indexing work correctly on a 1-dim container."""
        td = TensorDict({"a": torch.arange(5).float(), "b": torch.zeros(5, 3)}, shape=(5,))
        assert td[0].shape == torch.Size([])
        assert td[:3].shape == torch.Size([3])
        assert td[torch.tensor([1, 3])].shape == torch.Size([2])


class TestGetitemIsolation:
    """Tests that slicing produces structurally independent containers."""

    def test_modify_top_level_structure_on_slice_does_not_affect_original(self, nested_dict):
        """Adding or deleting top-level keys in a slice leaves the original unchanged."""
        data = nested_dict((2, 2))
        td = TensorDict(data, shape=(2, 2))
        sliced = td[0]

        # add a new top‐level key to the slice
        sliced["extra"] = torch.tensor([[1, 2], [1, 2]])
        assert "extra" in sliced
        assert "extra" not in td

        # delete an existing top‐level key from the slice
        del sliced["y"]
        assert "y" not in sliced
        assert "y" in td

    def test_modify_nested_structure_on_slice_does_not_affect_original(self, nested_dict):
        """Adding or deleting nested keys in a slice leaves the original unchanged."""
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

    def test_leaf_tensor_data_is_shared(self):
        """Sliced leaf tensors share storage with the original (view, not copy)."""
        a = torch.arange(6).reshape(2, 3).float()
        td = TensorDict({"a": a}, shape=(2, 3))
        sliced = td[0]
        assert sliced["a"].data_ptr() == a[0].data_ptr()


class TestGetitemZeroDim:
    """Tests zero-dimensional TensorDict indexing semantics."""

    def test_getitem_none_on_zero_dim(self, zero_dim_td):
        """td[None] on a 0-dim TensorDict should add a leading dim, matching torch.Tensor semantics."""
        result = zero_dim_td[None]
        assert result.shape == torch.Size([1])
        assert_tensor_indexing_parity(result, (), None)
        assert result["scalar"].shape == torch.Size([1])
        assert result["vec"].shape == torch.Size([1, 3])

    def test_getitem_ellipsis_on_zero_dim(self, zero_dim_td):
        """td[...] on a 0-dim TensorDict should return identity, matching torch.Tensor semantics."""
        result = zero_dim_td[...]
        assert result.shape == torch.Size([])
        assert_tensor_indexing_parity(result, (), Ellipsis)
        assert torch.equal(result["scalar"], zero_dim_td["scalar"])
        assert torch.equal(result["vec"], zero_dim_td["vec"])

    @pytest.mark.parametrize("value", [True, False])
    def test_getitem_bool_on_zero_dim(self, zero_dim_td, value):
        """td[True]/td[False] on a 0-dim TensorDict should match torch.Tensor semantics."""
        result = zero_dim_td[value]
        expected_len = 1 if value else 0
        assert result.shape == torch.Size([expected_len])
        assert_tensor_indexing_parity(result, (), value)
        assert result["scalar"].shape == torch.Size([expected_len])
        assert result["vec"].shape == torch.Size([expected_len, 3])

    @pytest.mark.parametrize("value", [True, False])
    def test_getitem_bool_tensor_on_zero_dim(self, zero_dim_td, value):
        """td[torch.tensor(True/False)] on a 0-dim TensorDict should match torch.Tensor semantics."""
        index = torch.tensor(value)
        result = zero_dim_td[index]
        expected_len = 1 if value else 0
        assert result.shape == torch.Size([expected_len])
        assert_tensor_indexing_parity(result, (), index)
        assert result["scalar"].shape == torch.Size([expected_len])
        assert result["vec"].shape == torch.Size([expected_len, 3])


class TestGetitemSlicing:
    """Tests torch.Tensor indexing parity across all index types."""

    @pytest.mark.parametrize("index", ALL_GETITEM_INDICES, ids=str)
    def test_getitem_with_slicing_indices(self, index):
        """Indexing a TensorDict should match torch.Tensor indexing for shape and values."""
        a = torch.arange(30).reshape(2, 3, 5).float()
        b = torch.arange(6).reshape(2, 3).float()
        td = TensorDict({"a": a, "b": b}, shape=(2, 3))

        result = td[index]
        assert_tensor_indexing_parity(result, (2, 3), index)
        torch.testing.assert_close(result["a"], a[index])
        torch.testing.assert_close(result["b"], b[index])


class TestGetitemErrors:
    """Tests that invalid indices raise appropriate errors."""

    def test_too_many_indices_raises_error(self):
        """Indexing with more dims than batch rank raises IndexError."""
        td = TensorDict({"a": torch.zeros(2, 3, 4)}, shape=[2, 3])
        with pytest.raises(
            IndexError,
            match="too many indices for container: container is 2-dimensional, but 3 were indexed",
        ):
            td[:, :, 0]

        # torch.Tensor parity: torch also rejects too many indices on a 2-d tensor
        with pytest.raises(IndexError):
            torch.zeros(2, 3)[:, :, 0]

    def test_multiple_ellipsis_raises_error(self):
        """Using more than one ellipsis in an index raises IndexError."""
        td = TensorDict({"a": torch.zeros(2, 3)}, shape=(2, 3))
        with pytest.raises(IndexError, match="single ellipsis"):
            td[..., ..., 0]
