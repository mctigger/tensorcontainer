"""
Tests for TensorDict.__setitem__ functionality.

This module contains test classes that verify the setitem behavior of TensorDict,
including slicing-based assignments and key-based dictionary-like operations.
"""

import pytest
import torch

from src.tensorcontainer.tensor_dict import TensorDict
from tests.conftest import skipif_no_cuda

# Basic indexing - creates views when possible, predictable shape transformations
BASIC_INDICES = [
    0,
    1,
    -1,
    -2,
    slice(0, 2),
    slice(1, 2),
    slice(-2, None),
    slice(None, -1),
    slice(-3, -1),
    slice(None, None, 2),
    slice(1, None, 2),
    ...,
    None,
]

# Advanced indexing - may require copying, can gather elements from arbitrary positions
ADVANCED_INDICES = [
    [0, 1],
    torch.tensor([0]),
    torch.tensor([1]),
    torch.tensor([0, 1]),
    torch.LongTensor([0, 1]),
]

# Boolean indexing - filters elements based on conditions, typically flattens result
BOOLEAN_INDICES = [
    torch.rand(2) > 0.5,
    torch.rand(2, 3) > 0.5,
]

# Multi-dimensional indexing - tests interaction between different index types
MULTIDIM_INDICES = [
    (1, slice(None)),
    (slice(0, 2), slice(None)),
    (slice(0, 2, 1), slice(None)),
    ([0, 1], slice(None)),
    (torch.LongTensor([0, 1]), slice(None)),
    (torch.rand(2) > 0.5, slice(None)),
    (slice(None), 1),
    (slice(None), slice(0, 2)),
    (torch.tensor([0, 1]), torch.tensor([0, 1])),
    (slice(None), torch.rand(3) > 0.5),
]

# Edge cases - empty slices, boundary conditions, and special behaviors
EDGE_CASE_INDICES = [
    slice(0, 0),
    slice(1, 1),
    slice(2, 2),
]

SLICING_INDICES = (
    BASIC_INDICES
    + ADVANCED_INDICES
    + BOOLEAN_INDICES
    + MULTIDIM_INDICES
    + EDGE_CASE_INDICES
)


class TestTensorDictSlicingSetitem:
    """
    Tests the slicing-based __setitem__ functionality of TensorDict.

    This suite verifies that:
    - Valid slice assignments work correctly with various indexing patterns
    - Invalid slice assignments raise appropriate errors
    - Tensor data is correctly updated when assigning TensorDict slices
    """

    @pytest.mark.parametrize("idx", SLICING_INDICES, ids=str)
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
        slice_shape = torch.ones(2, 3)[idx].shape

        td_source = TensorDict(
            data={
                "features": torch.randn(*slice_shape, 4),
                "labels": torch.randn(*slice_shape),
            },
            shape=slice_shape,
        )

        # Calculate expected results
        expected_features = td_dest["features"].clone()
        expected_labels = td_dest["labels"].clone()

        # Manually assign from source tensors to destination tensors
        expected_features[idx] = td_source["features"]
        expected_labels[idx] = td_source["labels"]

        # Perform the assignment
        td_dest[idx] = td_source.clone()

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


class TestTensorDictCrossDeviceSliceAssignment:
    """
    Tests that demonstrate PyTorch's cross-device assignment behavior for slice assignments.

    This test class shows that PyTorch allows cross-device tensor assignments by automatically
    transferring data to the target tensor's device, maintaining device consistency.
    """

    @skipif_no_cuda
    def test_cpu_to_cuda_slice_assignment_succeeds(self):
        """
        Tests that slice assignment from CPU TensorDict to CUDA TensorDict succeeds
        with PyTorch automatically transferring data to CUDA.
        """
        # Create CUDA destination container
        td_cuda = TensorDict(
            data={"features": torch.ones(2, 3, device="cuda")},
            shape=(2,),
            device="cuda",
        )

        # Create CPU source container with different values
        td_cpu = TensorDict(
            data={"features": torch.ones(2, 3, device="cpu") * 5},
            shape=(2,),
            device="cpu",
        )

        # Perform slice assignment - should succeed with automatic device transfer
        td_cuda[0] = td_cpu[0]

        # Verify the assignment worked and device consistency is maintained
        expected = torch.ones(2, 3, device="cuda")
        expected[0] = 5  # First row should be 5s, second row still 1s

        torch.testing.assert_close(td_cuda["features"], expected)
        assert td_cuda["features"].device.type == "cuda"
        assert td_cuda.device.type == "cuda"

    @skipif_no_cuda
    def test_cuda_to_cpu_slice_assignment_succeeds(self):
        """
        Tests that slice assignment from CUDA TensorDict to CPU TensorDict succeeds
        with PyTorch automatically transferring data to CPU.
        """
        # Create CPU destination container
        td_cpu = TensorDict(
            data={"features": torch.ones(2, 3, device="cpu")},
            shape=(2,),
            device="cpu",
        )

        # Create CUDA source container with different values
        td_cuda = TensorDict(
            data={"features": torch.ones(2, 3, device="cuda") * 7},
            shape=(2,),
            device="cuda",
        )

        # Perform slice assignment - should succeed with automatic device transfer
        td_cpu[0] = td_cuda[0]

        # Verify the assignment worked and device consistency is maintained
        expected = torch.ones(2, 3, device="cpu")
        expected[0] = 7  # First row should be 7s, second row still 1s

        torch.testing.assert_close(td_cpu["features"], expected)
        assert td_cpu["features"].device.type == "cpu"
        assert td_cpu.device.type == "cpu"

    @skipif_no_cuda
    def test_cross_device_slice_range_assignment_succeeds(self):
        """
        Tests that slice range assignment across devices succeeds with automatic transfer.
        """
        # Create CUDA destination container
        td_cuda = TensorDict(
            data={"features": torch.ones(4, 2, device="cuda")},
            shape=(4,),
            device="cuda",
        )

        # Create CPU source container with different values for slice
        td_cpu = TensorDict(
            data={"features": torch.ones(2, 2, device="cpu") * 3},
            shape=(2,),
            device="cpu",
        )

        # Perform slice range assignment - should succeed
        td_cuda[1:3] = td_cpu

        # Verify the assignment worked
        expected = torch.ones(4, 2, device="cuda")
        expected[1:3] = 3  # Middle rows should be 3s

        torch.testing.assert_close(td_cuda["features"], expected)
        assert td_cuda["features"].device.type == "cuda"
        assert td_cuda.device.type == "cuda"

    @skipif_no_cuda
    def test_cross_device_boolean_mask_fails_due_to_stricter_requirements(self):
        """
        Tests that boolean mask assignment has stricter device requirements than
        regular slice assignment - it requires ALL tensors (mask, target, source)
        to be on the same device.
        """
        # Create CUDA destination container
        td_cuda = TensorDict(
            data={"features": torch.ones(4, 2, device="cuda")},
            shape=(4,),
            device="cuda",
        )

        # Create CPU source container
        td_cpu = TensorDict(
            data={"features": torch.ones(2, 2, device="cpu") * 9},
            shape=(2,),
            device="cpu",
        )

        # Create boolean mask on CUDA (same as target device)
        mask = torch.tensor([True, False, True, False], device="cuda")

        # Boolean mask assignment fails because PyTorch requires ALL tensors
        # (mask, target, AND source) to be on the same device for boolean indexing
        with pytest.raises(
            RuntimeError,
            match="Expected all tensors to be on the same device|Issue with key",
        ):
            td_cuda[mask] = td_cpu

    @skipif_no_cuda
    def test_device_validation_still_enforced_for_key_assignment(self):
        """
        Tests that device validation is still enforced for key-based assignments,
        demonstrating that slice assignment behavior is PyTorch's standard behavior.
        """
        td_cuda = TensorDict(
            data={"features": torch.ones(2, 3, device="cuda")},
            shape=(2,),
            device="cuda",
        )

        # Key-based assignment should still enforce device validation
        with pytest.raises(
            RuntimeError, match="Invalid device.*Expected device that is compatible"
        ):
            td_cuda["new_tensor"] = torch.ones(2, 3, device="cpu")
