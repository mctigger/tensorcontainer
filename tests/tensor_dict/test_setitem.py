"""
Tests for TensorDict.__setitem__ functionality.

This module contains test classes that verify the setitem behavior of TensorDict,
including slicing-based assignments and key-based dictionary-like operations.
"""

import pytest
import torch

from src.tensorcontainer.tensor_dict import TensorDict
from tests.conftest import skipif_no_cuda

# Define a common set of indices for slicing tests
SLICING_INDICES = [
    (1, slice(None)),  # Single integer index
    (slice(0, 2), slice(None)),  # Basic slice
    (slice(0, 2, 1), slice(None)),  # Slice with step
    ([0, 1], slice(None)),  # List of indices
    (torch.LongTensor([0, 1]), slice(None)),  # LongTensor of indices
    (torch.rand(2) > 0.5, slice(None)),  # BoolTensor mask
    (slice(None), 1),  # Indexing second dimension
    (slice(None), slice(0, 2)),  # Slicing second dimension
    (torch.tensor([0, 1]), torch.tensor([0, 1])),  # Tensor indexing
    (slice(None), torch.rand(3) > 0.5),  # BoolTensor mask on second dim
]


class TestTensorDictSlicingSetitem:
    """
    Tests the slicing-based __setitem__ functionality of TensorDict.

    This suite verifies that:
    - Valid slice assignments work correctly with various indexing patterns
    - Invalid slice assignments raise appropriate errors
    - Tensor data is correctly updated when assigning TensorDict slices
    """

    @pytest.mark.parametrize("idx", SLICING_INDICES)
    def test_valid_slicing_assignment(self, idx):
        """
        Tests assigning a source TensorDict to a slice of a destination TensorDict.
        Verifies that the assignment correctly updates the sliced portion while
        preserving the rest of the data.
        """
        td_dest = TensorDict(
            data={"features": torch.ones(2, 3, 4), "labels": torch.ones(2, 3)},
            shape=(2, 3),
        )

        # Determine the shape of the slice to create a compatible source TensorDict
        slice_shape = td_dest[idx].shape

        td_source = TensorDict(
            data={
                "features": torch.randn(*slice_shape, 4),
                "labels": torch.randn(*slice_shape),
            },
            shape=slice_shape,
        )

        original_dest_features = td_dest["features"].clone()
        original_dest_labels = td_dest["labels"].clone()

        # Perform the assignment
        td_dest[idx] = td_source

        # Calculate expected results
        expected_features = original_dest_features
        expected_labels = original_dest_labels
        expected_features[idx] = td_source["features"]
        expected_labels[idx] = td_source["labels"]

        torch.testing.assert_close(td_dest["features"], expected_features)
        torch.testing.assert_close(td_dest["labels"], expected_labels)

    @pytest.mark.parametrize(
        "idx, value_shape, error",
        [
            ((slice(None),), (1, 3), KeyError),  # Shape mismatch
            (2, None, ValueError),  # Index out of bounds
            ((slice(None), slice(None), 0), None, ValueError),  # Too many indices
        ],
    )
    def test_invalid_slicing_assignment(self, idx, value_shape, error):
        """
        Tests invalid slice assignments that should raise errors.
        Verifies that appropriate exceptions are raised for incompatible shapes,
        out-of-bounds indices, and invalid indexing patterns.
        """
        td = TensorDict(
            data={"features": torch.randn(2, 3, 4), "labels": torch.randn(2, 3)},
            shape=(2, 3),
        )

        if value_shape is not None:
            value = TensorDict(
                data={"features": torch.randn(*value_shape, 4)}, shape=value_shape
            )
        else:
            value = 0.0

        with pytest.raises(error):
            td[idx] = value


class TestTensorDictKeyBasedSetitem:
    """
    Tests the dictionary-like __setitem__ functionality of TensorDict.

    This suite verifies that:
    - New tensors can be added with compatible shapes
    - Existing tensors can be updated correctly
    - Nested TensorDicts can be assigned as values
    - Invalid assignments raise appropriate errors for shape, type, and device mismatches
    """

    def test_valid_key_assignments(self):
        """
        Tests valid key-based assignments including adding new tensors,
        updating existing ones, and assigning nested TensorDicts.
        """
        td = TensorDict(
            data={"features": torch.randn(2, 3)},
            shape=(2,),
        )

        # Add a new tensor with a compatible shape
        new_labels = torch.zeros(2)
        td["labels"] = new_labels
        assert "labels" in td
        torch.testing.assert_close(td["labels"], new_labels)

        # Update an existing tensor
        updated_features = torch.ones(2, 3)
        td["features"] = updated_features
        torch.testing.assert_close(td["features"], updated_features)

        # Add a nested TensorDict with a compatible shape
        nested_td = TensorDict({"nested_feat": torch.rand(2, 4)}, shape=(2,))
        td["nested"] = nested_td
        assert "nested" in td
        assert isinstance(td["nested"], TensorDict)
        torch.testing.assert_close(
            td["nested"]["nested_feat"], nested_td["nested_feat"]
        )

    def test_invalid_shape_assignment(self):
        """
        Tests that assigning a tensor with an incompatible batch shape raises RuntimeError.
        """
        td = TensorDict(
            data={"features": torch.randn(2, 3)},
            shape=(2,),
        )

        with pytest.raises(
            RuntimeError, match="Invalid shape.*Expected shape that is compatible"
        ):
            td["bad_shape"] = torch.randn(3, 3)  # Mismatched batch dim

    def test_invalid_type_assignment(self):
        """
        Tests that assigning a non-Tensor/TensorContainer value raises AttributeError.
        """
        td = TensorDict(
            data={"features": torch.randn(2, 3)},
            shape=(2,),
        )

        with pytest.raises(
            AttributeError, match="'list' object has no attribute 'shape'"
        ):
            td["bad_type"] = [1, 2, 3]  # Invalid type

    @skipif_no_cuda
    def test_invalid_device_assignment(self):
        """
        Tests that assigning a tensor with a different device raises RuntimeError
        when device validation is enabled.
        """
        td_cpu = TensorDict(
            data={"features": torch.randn(2, 3, device="cpu")},
            shape=(2,),
            device="cpu",
        )
        with pytest.raises(
            RuntimeError, match="Invalid device.*Expected device that is compatible"
        ):
            td_cpu["bad_device"] = torch.randn(2, 3, device="cuda")
