import pytest
import torch

from src.rtd.tensor_dataclass import TensorDataClass


class MyTensorDataClass(TensorDataClass):
    features: torch.Tensor  # e.g., shape (B, N, D)
    labels: torch.Tensor  # e.g., shape (B, N)


params = [
    (0, slice(None)),  # Single integer index for first batch dim
    (slice(2, 15), slice(None)),  # Basic slice object for first batch dim
    (slice(0, 20, 3), slice(None)),  # Slice with custom step for first batch dim
    ([0, 1], slice(None)),  # List of integer indices for first batch dim
    (
        torch.LongTensor([0, 1]),
        slice(None),
    ),  # torch.LongTensor of indices for first batch dim
    (
        torch.rand(2) > 0.5,
        slice(None),
    ),  # torch.BoolTensor mask for first batch dim
    (slice(None), 0),  # Tuple with slice and integer for second batch dim
    (slice(2, 15), slice(None)),  # Tuple with two slices
    (
        slice(None),
        slice(2, 5),
    ),  # Tuple with slice and another slice for second batch dim
    (torch.tensor([0, 1]), slice(None)),  # Tuple with tensor and slice
    (
        slice(None),
        torch.tensor([0, 1]),
    ),  # Tuple with slice and tensor for second batch dim
    (torch.tensor([0, 1]), torch.tensor([0, 1])),  # Tuple with two tensors
    (torch.rand(2) > 0.5, slice(None)),  # Tuple with bool tensor and slice
    (
        slice(None),
        torch.rand(3) > 0.5,
    ),  # Tuple with slice and bool tensor for second batch dim
    # New test cases for Ellipsis and None
    (None, slice(None), slice(None)),  # New axis at the beginning
    (slice(None), None, slice(None)),  # New axis in the middle of batch dims
    (slice(None), slice(None), None),  # New axis at the end (after batch dims)
    (None, None, slice(None), slice(None)),  # Two new axes at the beginning
    (slice(None), slice(None), None, None),  # Two new axes at the end
]


@pytest.mark.parametrize(
    "idx",
    params,
    # Add this `ids` argument to generate descriptive names
    ids=[str(p) for p in params],
)
def test_getitem_standard_indices(idx):
    # Setup: Create a sample TensorDataClass instance
    features = torch.randn(2, 3, 4)
    labels = torch.arange(2 * 3).reshape(2, 3)
    tdc = MyTensorDataClass(
        features=features, labels=labels, shape=(2, 3), device=features.device
    )

    # Apply the index to the TensorDataClass instance
    result = tdc[idx]

    # Verify that result is an instance of TensorDataClass
    assert isinstance(result, MyTensorDataClass)

    # Apply the same index to the original source tensors
    expected_features = features[idx]
    expected_labels = labels[idx]

    # Confirm that result.features is equal to expected_features
    torch.testing.assert_close(result.features, expected_features)

    # Confirm that result.labels is equal to expected_labels
    torch.testing.assert_close(result.labels, expected_labels)

    # Verify shape and device
    if isinstance(idx, tuple):
        assert result.shape == expected_features.shape[:-1]
    else:
        assert result.shape == expected_features.shape
    assert result.device == expected_features.device


@pytest.mark.parametrize(
    "idx",
    [
        # Too many indices
        (slice(None), slice(None), 0),
        (slice(None), slice(None), slice(0, 1)),
        (slice(None), slice(None), torch.tensor([0])),
        # Index out of bounds for batch dim
        (20, slice(None)),  # Single integer index out of bounds
        (slice(None), 5),  # Single integer index out of bounds for second dim
        (torch.tensor([0, 20]), slice(None)),  # Tensor index out of bounds
        (
            slice(None),
            torch.tensor([0, 5]),
        ),  # Tensor index out of bounds for second dim
        (torch.rand(21) > 0.5, slice(None)),  # Bool tensor mask with wrong size
        (
            slice(None),
            torch.rand(6) > 0.5,
        ),  # Bool tensor mask with wrong size for second dim
        # Indexing a 0-dim TensorDataClass with a single index
        0,
    ],
)
def test_getitem_invalid_index(idx):
    features = torch.randn(2, 3, 4)
    labels = torch.arange(2 * 3).reshape(2, 3)
    tdc = MyTensorDataClass(
        features=features, labels=labels, shape=(2, 3), device=features.device
    )

    # For the 0-dim case, create a 0-dim TensorDataClass
    if idx == 0:
        features_0d = torch.randn(10)
        labels_0d = torch.arange(10)
        tdc = MyTensorDataClass(
            features=features_0d, labels=labels_0d, shape=(), device=features_0d.device
        )

    with pytest.raises(IndexError):
        tdc[idx]


params_ellipsis = [
    ((Ellipsis,), (2, 3)),
    ((Ellipsis, 0), (2,)),
    ((0, Ellipsis), (3,)),
    ((None, Ellipsis), (1, 2, 3)),
    ((Ellipsis, None), (2, 3, 1)),
    ((0, Ellipsis, None), (3, 1)),
    ((None, Ellipsis, 0), (1, 2)),
]


@pytest.mark.parametrize(
    "idx, expected_shape",
    params_ellipsis,
    ids=[str(p[0]) for p in params_ellipsis],
)
def test_getitem_with_ellipsis(idx, expected_shape):
    # Setup: Create a sample TensorDataClass instance

    batch_shape = (2, 3)
    features = torch.randn(2, 3, 4)
    labels = torch.arange(2 * 3).reshape(2, 3)
    tdc = MyTensorDataClass(
        features=features, labels=labels, shape=batch_shape, device=features.device
    )

    # Apply the index to the TensorDataClass instance
    result = tdc[idx]

    # Verify that result is an instance of TensorDataClass
    assert isinstance(result, MyTensorDataClass)

    # Apply the same index to the original source tensors
    expected_features = features[idx + (slice(None),)]
    expected_labels = labels[idx]

    # Confirm that result.features is equal to expected_features
    torch.testing.assert_close(result.features, expected_features)

    # Confirm that result.labels is equal to expected_labels
    torch.testing.assert_close(result.labels, expected_labels)

    # Verify shape and device
    assert result.shape == expected_shape
    assert result.device == expected_features.device


class TestTransformEllipsisIndex:
    """Test the transform_ellipsis_index method directly."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tdc = MyTensorDataClass(
            features=torch.randn(2, 3, 4),
            labels=torch.arange(6).reshape(2, 3),
            shape=(2, 3),
            device=torch.device("cpu"),
        )

    def test_no_ellipsis_returns_unchanged(self):
        """Test that indices without ellipsis are returned unchanged."""
        shape = (2, 3, 4)
        idx = (0, slice(None))
        result = self.tdc.transform_ellipsis_index(shape, idx)
        assert result == idx

        # Test with empty tuple
        idx = ()
        result = self.tdc.transform_ellipsis_index(shape, idx)
        assert result == idx

        # Test with single element
        idx = (0,)
        result = self.tdc.transform_ellipsis_index(shape, idx)
        assert result == idx

    def test_basic_ellipsis_expansion(self):
        """Test basic ellipsis expansion functionality."""
        shape = (2, 3, 4)

        # Ellipsis at beginning
        idx = (Ellipsis, 0)
        result = self.tdc.transform_ellipsis_index(shape, idx)
        expected = (slice(None), slice(None), 0)
        assert result == expected

        # Ellipsis at end
        idx = (0, Ellipsis)
        result = self.tdc.transform_ellipsis_index(shape, idx)
        expected = (0, slice(None), slice(None))
        assert result == expected

        # Ellipsis in middle
        idx = (0, Ellipsis, 1)
        result = self.tdc.transform_ellipsis_index(shape, idx)
        expected = (0, slice(None), 1)
        assert result == expected

        # Ellipsis only
        idx = (Ellipsis,)
        result = self.tdc.transform_ellipsis_index(shape, idx)
        expected = (slice(None), slice(None), slice(None))
        assert result == expected

    def test_ellipsis_with_none_indices(self):
        """Test ellipsis expansion with None (newaxis) indices."""
        shape = (2, 3)

        # None before ellipsis
        idx = (None, Ellipsis)
        result = self.tdc.transform_ellipsis_index(shape, idx)
        expected = (None, slice(None), slice(None))
        assert result == expected

        # None after ellipsis
        idx = (Ellipsis, None)
        result = self.tdc.transform_ellipsis_index(shape, idx)
        expected = (slice(None), slice(None), None)
        assert result == expected

        # None around ellipsis
        idx = (None, Ellipsis, None)
        result = self.tdc.transform_ellipsis_index(shape, idx)
        expected = (None, slice(None), slice(None), None)
        assert result == expected

        # Multiple None indices
        idx = (None, None, Ellipsis, None)
        result = self.tdc.transform_ellipsis_index(shape, idx)
        expected = (None, None, slice(None), slice(None), None)
        assert result == expected

    def test_ellipsis_zero_expansion(self):
        """Test cases where ellipsis expands to zero slices."""
        shape = (2, 3)

        # All dimensions consumed by explicit indices
        idx = (0, Ellipsis, 1)
        result = self.tdc.transform_ellipsis_index(shape, idx)
        expected = (0, 1)  # Ellipsis becomes empty tuple
        assert result == expected

        # Test with slice objects
        idx = (slice(None), Ellipsis, slice(1, 2))
        result = self.tdc.transform_ellipsis_index(shape, idx)
        expected = (slice(None), slice(1, 2))
        assert result == expected

    def test_ellipsis_with_slices(self):
        """Test ellipsis with various slice objects."""
        shape = (2, 3, 4, 5)

        idx = (slice(1, 2), Ellipsis, slice(None, None, 2))
        result = self.tdc.transform_ellipsis_index(shape, idx)
        expected = (slice(1, 2), slice(None), slice(None), slice(None, None, 2))
        assert result == expected

        idx = (slice(None), Ellipsis)
        result = self.tdc.transform_ellipsis_index(shape, idx)
        expected = (slice(None), slice(None), slice(None), slice(None))
        assert result == expected

    def test_multiple_ellipsis_error(self):
        """Test that multiple ellipsis raise IndexError."""
        shape = (2, 3, 4)
        idx = (Ellipsis, 0, Ellipsis)

        with pytest.raises(
            IndexError, match="an index can only have a single ellipsis"
        ):
            self.tdc.transform_ellipsis_index(shape, idx)

        # Test with three ellipsis
        idx = (Ellipsis, Ellipsis, Ellipsis)
        with pytest.raises(
            IndexError, match="an index can only have a single ellipsis"
        ):
            self.tdc.transform_ellipsis_index(shape, idx)

    def test_too_many_indices_error(self):
        """Test that too many consuming indices raise IndexError."""
        shape = (2, 3)

        # More indices than dimensions
        idx = (0, 1, Ellipsis, 2)
        with pytest.raises(IndexError, match="too many indices for array"):
            self.tdc.transform_ellipsis_index(shape, idx)

        # Exactly too many indices
        idx = (0, 1, 2, Ellipsis)
        with pytest.raises(IndexError, match="too many indices for array"):
            self.tdc.transform_ellipsis_index(shape, idx)

    def test_edge_cases_empty_and_single_dim(self):
        """Test edge cases with empty shapes and single dimensions."""
        # Empty shape
        shape = ()
        idx = (Ellipsis,)
        result = self.tdc.transform_ellipsis_index(shape, idx)
        expected = ()
        assert result == expected

        # Single dimension
        shape = (5,)
        idx = (Ellipsis,)
        result = self.tdc.transform_ellipsis_index(shape, idx)
        expected = (slice(None),)
        assert result == expected

        # Single dimension with index
        idx = (0, Ellipsis)
        result = self.tdc.transform_ellipsis_index(shape, idx)
        expected = (0,)
        assert result == expected

    def test_advanced_indexing_limitations(self):
        """Test current limitations with advanced indexing types."""
        shape = (2, 3, 4)

        # Boolean arrays - currently treated as single consuming index
        # This is a known limitation that should be documented
        bool_mask = torch.tensor([True, False])
        idx = (bool_mask, Ellipsis)
        result = self.tdc.transform_ellipsis_index(shape, idx)
        expected = (bool_mask, slice(None), slice(None))
        assert result == expected

        # Tensor indices - currently treated as single consuming index
        # This is also a known limitation
        tensor_idx = torch.tensor([0, 1])
        idx = (tensor_idx, Ellipsis)
        result = self.tdc.transform_ellipsis_index(shape, idx)
        expected = (tensor_idx, slice(None), slice(None))
        assert result == expected

        # List indices - currently treated as single consuming index
        list_idx = [0, 1]
        idx = (list_idx, Ellipsis)
        result = self.tdc.transform_ellipsis_index(shape, idx)
        expected = (list_idx, slice(None), slice(None))
        assert result == expected

    @pytest.mark.parametrize(
        "invalid_shape",
        [
            (-1, 2, 3),  # Negative dimension
            (2.5, 3),  # Float dimension
            ("2", 3),  # String dimension
        ],
    )
    def test_invalid_shape_handling(self, invalid_shape):
        """Test behavior with invalid shape tuples."""
        # Note: The current implementation doesn't validate shape
        # This test documents the current behavior
        idx = (Ellipsis,)
        try:
            result = self.tdc.transform_ellipsis_index(invalid_shape, idx)
            # If it doesn't raise an error, check that it returns something
            assert isinstance(result, tuple)
        except (TypeError, ValueError):
            # Some invalid shapes might cause errors in len() or other operations
            pass

    def test_complex_nested_cases(self):
        """Test complex combinations of ellipsis with various index types."""
        shape = (2, 3, 4, 5, 6)

        # Mix of slices, integers, None, and ellipsis
        idx = (slice(1, 2), None, Ellipsis, 0, None)
        result = self.tdc.transform_ellipsis_index(shape, idx)
        expected = (slice(1, 2), None, slice(None), slice(None), slice(None), 0, None)
        assert result == expected

        # Multiple None indices with ellipsis
        idx = (None, slice(None), None, Ellipsis, None)
        result = self.tdc.transform_ellipsis_index(shape, idx)
        expected = (
            None,
            slice(None),
            None,
            slice(None),
            slice(None),
            slice(None),
            slice(None),
            None,
        )
        assert result == expected

    def test_large_dimension_expansion(self):
        """Test ellipsis expansion with large number of dimensions."""
        shape = tuple(range(2, 12))  # 10 dimensions: (2, 3, 4, 5, 6, 7, 8, 9, 10, 11)

        idx = (0, Ellipsis, 1)
        result = self.tdc.transform_ellipsis_index(shape, idx)
        expected = (0,) + (slice(None),) * 8 + (1,)  # 8 slice(None) in the middle
        assert result == expected
        assert len(result) == 10  # Same as original shape length

    def test_memory_efficiency_large_expansion(self):
        """Test that large ellipsis expansions don't cause memory issues."""
        # This is a stress test for the slice(None) * num_slices_to_add operation
        shape = tuple([2] * 100)  # 100 dimensions

        idx = (0, Ellipsis)
        result = self.tdc.transform_ellipsis_index(shape, idx)
        expected = (0,) + (slice(None),) * 99
        assert result == expected
        assert len(result) == 100


class TestTransformEllipsisIndexIntegration:
    """Integration tests for transform_ellipsis_index with actual tensor operations."""

    def setup_method(self):
        """Set up test fixtures."""
        # TensorDataClass with shape (2, 3) containing:
        # - features: shape (2, 3, 4, 5) where (4, 5) are event dims
        # - labels: shape (2, 3) matching batch shape exactly
        self.features = torch.randn(2, 3, 4, 5)
        self.labels = torch.arange(2 * 3).reshape(2, 3)
        self.tdc = MyTensorDataClass(
            features=self.features,
            labels=self.labels,
            shape=(2, 3),
            device=torch.device("cpu"),
        )

    @pytest.mark.parametrize(
        "idx,expected_tdc_shape,expected_features_shape,expected_labels_shape",
        [
            # Basic ellipsis patterns
            ((...,), (2, 3), (2, 3, 4, 5), (2, 3)),
            ((0, ...), (3,), (3, 4, 5), (3,)),
            ((..., 0), (2,), (2, 4, 5), (2,)),
            ((1, ...), (3,), (3, 4, 5), (3,)),
            # Ellipsis with None (newaxis)
            ((None, ...), (1, 2, 3), (1, 2, 3, 4, 5), (1, 2, 3)),
            ((..., None), (2, 3, 1), (2, 3, 1, 4, 5), (2, 3, 1)),
            ((None, 0, ...), (1, 3), (1, 3, 4, 5), (1, 3)),
            ((0, None, ...), (1, 3), (1, 3, 4, 5), (1, 3)),
            # Ellipsis with slices
            ((slice(None), ...), (2, 3), (2, 3, 4, 5), (2, 3)),
            ((slice(0, 1), ...), (1, 3), (1, 3, 4, 5), (1, 3)),
            ((..., slice(None)), (2, 3), (2, 3, 4, 5), (2, 3)),
            ((slice(1, 2), ...), (1, 3), (1, 3, 4, 5), (1, 3)),
            # Complex combinations
            ((None, slice(0, 2), ...), (1, 2, 3), (1, 2, 3, 4, 5), (1, 2, 3)),
            ((slice(None), None, ...), (2, 1, 3), (2, 1, 3, 4, 5), (2, 1, 3)),
            ((0, None, ...), (1, 3), (1, 3, 4, 5), (1, 3)),
        ],
        ids=[
            "ellipsis_only",
            "int_ellipsis",
            "ellipsis_int",
            "int1_ellipsis",
            "none_ellipsis",
            "ellipsis_none",
            "none_int_ellipsis",
            "int_none_ellipsis",
            "slice_ellipsis",
            "slice_0_1_ellipsis",
            "ellipsis_slice",
            "slice_1_2_ellipsis",
            "none_slice_ellipsis",
            "slice_none_ellipsis",
            "int_none_ellipsis_2",
        ],
    )
    def test_ellipsis_integration_shapes(
        self, idx, expected_tdc_shape, expected_features_shape, expected_labels_shape
    ):
        """Test that ellipsis transformation produces correct shapes."""
        result = self.tdc[idx]

        # Verify TensorDataClass shape transformation
        assert result.shape == expected_tdc_shape, (
            f"TensorDataClass shape mismatch for {idx}"
        )

        # Verify individual tensor shapes (event dims preserved)
        assert result.features.shape == expected_features_shape, (
            f"Features shape mismatch for {idx}"
        )
        assert result.labels.shape == expected_labels_shape, (
            f"Labels shape mismatch for {idx}"
        )

        # Verify result is still a TensorDataClass
        assert isinstance(result, MyTensorDataClass)

    def test_ellipsis_integration_content_int_first(self):
        """Test integer index at start with ellipsis."""
        result = self.tdc[0, ...]
        expected_features = self.features[0, ...]
        expected_labels = self.labels[0, ...]
        torch.testing.assert_close(result.features, expected_features)
        torch.testing.assert_close(result.labels, expected_labels)

    def test_ellipsis_integration_content_int_second(self):
        """Test second integer index with ellipsis."""
        result = self.tdc[1, ...]
        expected_features = self.features[1, ...]
        expected_labels = self.labels[1, ...]
        torch.testing.assert_close(result.features, expected_features)
        torch.testing.assert_close(result.labels, expected_labels)

    def test_ellipsis_integration_content_int_last(self):
        """Test ellipsis with integer at end."""
        result = self.tdc[..., 0]
        expected_features = self.features[:, 0, :]
        expected_labels = self.labels[:, 0]
        torch.testing.assert_close(result.features, expected_features)
        torch.testing.assert_close(result.labels, expected_labels)

    def test_ellipsis_integration_content_slice(self):
        """Test slice with ellipsis."""
        result = self.tdc[slice(0, 1), ...]
        expected_features = self.features[slice(0, 1), ...]
        expected_labels = self.labels[slice(0, 1), ...]
        torch.testing.assert_close(result.features, expected_features)
        torch.testing.assert_close(result.labels, expected_labels)

    def test_ellipsis_integration_content_none_int(self):
        """Test None (newaxis) with integer and ellipsis."""
        result = self.tdc[None, 0, ...]
        expected_features = self.features[0, ...].unsqueeze(0)
        expected_labels = self.labels[0, ...].unsqueeze(0)
        torch.testing.assert_close(result.features, expected_features)
        torch.testing.assert_close(result.labels, expected_labels)

    def test_ellipsis_error_propagation(self):
        """Test that errors from transform_ellipsis_index propagate correctly."""
        # Test multiple ellipsis error
        with pytest.raises(IndexError, match="single ellipsis"):
            self.tdc[..., 0, ...]

        # Test too many indices error - trying to index beyond batch dims
        with pytest.raises(IndexError, match="too many indices"):
            self.tdc[0, 1, 2, ...]

    def test_direct_method_access(self):
        """Test accessing the transform_ellipsis_index method directly."""
        # Test with shape (2, 3, 4) - matches a tensor shape
        shape = (2, 3, 4)
        idx = (0, ..., 1)

        result = self.tdc.transform_ellipsis_index(shape, idx)
        expected = (0, slice(None), 1)
        assert result == expected

    def test_event_dim_preservation(self):
        """Test that event dimensions are always preserved regardless of batch transformations."""
        # Original TensorDataClass: shape=(2,3), features=(2,3,4,5), labels=(2,3)
        # Event dims for features: (4,5), for labels: ()

        test_cases = [
            # (indexing, expected_batch_shape, expected_features_event_shape, expected_labels_event_shape)
            ((0,), (3,), (4, 5), ()),
            ((slice(0, 1),), (1, 3), (4, 5), ()),
            ((None, 0), (1, 3), (4, 5), ()),
            ((...,), (2, 3), (4, 5), ()),
            ((0, None), (1, 3), (4, 5), ()),
        ]

        for (
            idx,
            expected_batch_shape,
            expected_features_event_shape,
            expected_labels_event_shape,
        ) in test_cases:
            result = self.tdc[idx]

            # Check that batch shape is transformed correctly
            assert result.shape == expected_batch_shape

            # Check that event dimensions are preserved
            actual_features_event_shape = result.features.shape[
                len(expected_batch_shape) :
            ]
            actual_labels_event_shape = result.labels.shape[len(expected_batch_shape) :]

            assert actual_features_event_shape == expected_features_event_shape, (
                f"Features event dims not preserved for {idx}: got {actual_features_event_shape}, expected {expected_features_event_shape}"
            )
            assert actual_labels_event_shape == expected_labels_event_shape, (
                f"Labels event dims not preserved for {idx}: got {actual_labels_event_shape}, expected {expected_labels_event_shape}"
            )


class TestTransformEllipsisIndexErrorMessages:
    """Test specific error messages and edge cases for better debugging."""

    def setup_method(self):
        self.tdc = MyTensorDataClass(
            features=torch.randn(2, 3, 4),
            labels=torch.arange(6).reshape(2, 3),
            shape=(2, 3),
            device=torch.device("cpu"),
        )

    def test_multiple_ellipsis_error_message(self):
        """Test specific error message for multiple ellipsis."""
        with pytest.raises(IndexError) as exc_info:
            self.tdc.transform_ellipsis_index((2, 3), (..., 0, ...))

        error_msg = str(exc_info.value)
        assert "single ellipsis" in error_msg
        assert "..." in error_msg

    def test_too_many_indices_error_message(self):
        """Test specific error message for too many indices."""
        with pytest.raises(IndexError) as exc_info:
            self.tdc.transform_ellipsis_index((2, 3), (0, 1, 2, ...))

        error_msg = str(exc_info.value)
        assert "too many indices" in error_msg
        assert "2-dimensional" in error_msg
        assert "3 were indexed" in error_msg

    def test_boundary_conditions(self):
        """Test boundary conditions that might cause off-by-one errors."""
        shape = (5,)

        # Exactly matching number of indices
        idx = (0, ...)
        result = self.tdc.transform_ellipsis_index(shape, idx)
        assert result == (0,)

        shape = (2, 3, 4)

        # Just under the limit should work
        idx = (0, 1, ...)
        result = self.tdc.transform_ellipsis_index(shape, idx)
        expected = (0, 1, slice(None))
        assert result == expected
