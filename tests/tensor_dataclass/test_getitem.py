import pytest
import torch

from src.rtd.tensor_dataclass import TensorDataClass
from tests.tensor_dict.compile_utils import run_and_compare_compiled


class SampleTensorDataClass(TensorDataClass):
    """Test dataclass with features and labels tensors."""

    features: torch.Tensor  # shape (B, N, D)
    labels: torch.Tensor  # shape (B, N)


def create_sample_tdc(batch_shape=(2, 3), feature_dims=4):
    """Helper to create a sample TensorDataClass for testing."""
    features = torch.randn(*batch_shape, feature_dims)
    total_elements = 1
    for dim in batch_shape:
        total_elements *= dim
    labels = torch.arange(total_elements).reshape(batch_shape)
    return SampleTensorDataClass(
        features=features, labels=labels, shape=batch_shape, device=features.device
    )


def random_mask(*shape):
    """Creates a random boolean mask of a given shape."""
    return torch.rand(shape) > 0.5


def _get_item(tdc, idx):
    return tdc[idx]


# Standard indexing test cases
STANDARD_INDEXING_CASES = [
    # Single dimension indexing
    ("single_int_first_dim", (0, slice(None))),
    ("slice_first_dim", (slice(2, 15), slice(None))),
    ("slice_with_step", (slice(0, 20, 3), slice(None))),
    ("list_indices", ([0, 1], slice(None))),
    ("tensor_indices", (torch.LongTensor([0, 1]), slice(None))),
    ("bool_mask_first_dim", (random_mask(2), slice(None))),
    # Second dimension indexing
    ("single_int_second_dim", (slice(None), 0)),
    ("slice_second_dim", (slice(None), slice(2, 5))),
    ("tensor_second_dim", (slice(None), torch.tensor([0, 1]))),
    ("bool_mask_second_dim", (slice(None), random_mask(3))),
    # Combined indexing
    ("both_slices", (slice(2, 15), slice(None))),
    ("tensor_and_slice", (torch.tensor([0, 1]), slice(None))),
    ("two_tensors", (torch.tensor([0, 1]), torch.tensor([0, 1]))),
    ("bool_and_slice", (random_mask(2), slice(None))),
    # None (newaxis) indexing
    ("newaxis_start", (None, slice(None), slice(None))),
    ("newaxis_middle", (slice(None), None, slice(None))),
    ("newaxis_end", (slice(None), slice(None), None)),
    ("two_newaxis_start", (None, None, slice(None), slice(None))),
    ("two_newaxis_end", (slice(None), slice(None), None, None)),
]


@pytest.mark.parametrize(
    "test_name,idx",
    STANDARD_INDEXING_CASES,
    ids=[case[0] for case in STANDARD_INDEXING_CASES],
)
def test_standard_indexing(test_name, idx, request):
    """Test standard indexing operations on TensorDataClass.

    Verifies that indexing operations work correctly and return the expected
    tensor shapes and values for various index types including integers,
    slices, lists, tensors, and boolean masks.
    """
    tdc = create_sample_tdc()

    fullgraph = True
    try:
        torch.compile(_get_item, fullgraph=True)(tdc.labels, idx)
    except Exception:
        fullgraph = False

    # Apply the index to the TensorDataClass instance
    result, _ = run_and_compare_compiled(
        _get_item,
        tdc,
        idx,
        fullgraph=fullgraph,
        expected_graph_breaks=0 if fullgraph else None,
    )

    # Verify that result is an instance of TensorDataClass
    assert isinstance(result, SampleTensorDataClass), (
        f"Expected SampleTensorDataClass, got {type(result)}"
    )

    # Apply the same index to the original source tensors
    expected_features = tdc.features[idx]
    expected_labels = tdc.labels[idx]
    # Confirm that result tensors match expected values
    torch.testing.assert_close(
        result.features, expected_features, msg=f"Features mismatch for {test_name}"
    )
    torch.testing.assert_close(
        result.labels, expected_labels, msg=f"Labels mismatch for {test_name}"
    )
    # Verify shape and device
    if isinstance(idx, tuple):
        assert result.shape == expected_features.shape[:-1], (
            f"Shape mismatch for {test_name}"
        )
    else:
        assert result.shape == expected_features.shape, (
            f"Shape mismatch for {test_name}"
        )
    assert result.device == expected_features.device, f"Device mismatch for {test_name}"


# Invalid indexing test cases
INVALID_INDEXING_CASES = [
    # Too many indices
    ("too_many_indices_int", (slice(None), slice(None), 0)),
    ("too_many_indices_slice", (slice(None), slice(None), slice(0, 1))),
    ("too_many_indices_tensor", (slice(None), slice(None), torch.tensor([0]))),
    # Index out of bounds
    ("out_of_bounds_first_dim", (20, slice(None))),
    ("out_of_bounds_second_dim", (slice(None), 5)),
    ("tensor_out_of_bounds_first", (torch.tensor([0, 20]), slice(None))),
    ("tensor_out_of_bounds_second", (slice(None), torch.tensor([0, 5]))),
    # Wrong size boolean masks
    ("bool_mask_wrong_size_first", (random_mask(21), slice(None))),
    ("bool_mask_wrong_size_second", (slice(None), random_mask(6))),
]


@pytest.mark.parametrize(
    "test_name,idx",
    INVALID_INDEXING_CASES,
    ids=[case[0] for case in INVALID_INDEXING_CASES],
)
def test_invalid_indexing_raises_error(test_name, idx):
    """Test that invalid indexing operations raise appropriate errors.

    Verifies that various invalid indexing scenarios (too many indices,
    out of bounds indices, wrong size masks) raise IndexError.
    """
    tdc = create_sample_tdc()

    with pytest.raises(IndexError, match=".*"):
        _ = tdc[idx]


def test_zero_dim_indexing_raises_error():
    """Test that indexing a 0-dim TensorDataClass raises IndexError."""
    features_0d = torch.randn(10)
    labels_0d = torch.arange(10)
    tdc = SampleTensorDataClass(
        features=features_0d, labels=labels_0d, shape=(), device=features_0d.device
    )

    with pytest.raises(IndexError, match=".*"):
        _ = tdc[0]


# Ellipsis indexing test cases
ELLIPSIS_INDEXING_CASES = [
    ("ellipsis_only", (Ellipsis,), (2, 3)),
    ("ellipsis_with_int", (Ellipsis, 0), (2,)),
    ("int_with_ellipsis", (0, Ellipsis), (3,)),
    ("newaxis_ellipsis", (None, Ellipsis), (1, 2, 3)),
    ("ellipsis_newaxis", (Ellipsis, None), (2, 3, 1)),
    ("int_ellipsis_newaxis", (0, Ellipsis, None), (3, 1)),
    ("newaxis_ellipsis_int", (None, Ellipsis, 0), (1, 2)),
]


@pytest.mark.parametrize(
    "test_name,idx,expected_shape",
    ELLIPSIS_INDEXING_CASES,
    ids=[case[0] for case in ELLIPSIS_INDEXING_CASES],
)
def test_ellipsis_indexing(test_name, idx, expected_shape, request):
    """Test ellipsis (...) indexing operations on TensorDataClass.

    Verifies that ellipsis indexing works correctly with various combinations
    of ellipsis, integers, and None (newaxis) indices.
    """
    tdc = create_sample_tdc()

    fullgraph = True
    try:
        torch.compile(_get_item, fullgraph=True)(tdc.labels, idx)
    except Exception:
        fullgraph = False

    # Apply the index to the TensorDataClass instance
    result, _ = run_and_compare_compiled(
        _get_item,
        tdc,
        idx,
        fullgraph=fullgraph,
        expected_graph_breaks=0 if fullgraph else None,
    )

    # Verify that result is an instance of TensorDataClass
    assert isinstance(result, SampleTensorDataClass), (
        f"Expected SampleTensorDataClass, got {type(result)}"
    )

    # Apply the same index to the original source tensors
    expected_features = tdc.features[idx + (slice(None),)]
    expected_labels = tdc.labels[idx]

    # Confirm that result tensors match expected values
    # Confirm that result tensors match expected values
    torch.testing.assert_close(
        result.features, expected_features, msg=f"Features mismatch for {test_name}"
    )
    torch.testing.assert_close(
        result.labels, expected_labels, msg=f"Labels mismatch for {test_name}"
    )

    # Verify shape and device
    assert result.shape == expected_shape, (
        f"Shape mismatch for {test_name}: expected {expected_shape}, got {result.shape}"
    )
    assert result.device == expected_features.device, f"Device mismatch for {test_name}"


# Boolean mask indexing test cases
BOOLEAN_MASK_CASES = [
    # Basic boolean masking
    ("bool_mask_1d", (10,), (random_mask(10),)),
    ("bool_mask_2d", (10, 5), (random_mask(10, 5),)),
    ("bool_mask_second_dim", (10, 5), (slice(None), random_mask(5))),
    ("bool_mask_first_dim", (10, 5), (random_mask(10), slice(None))),
    # Boolean masking with ellipsis
    ("bool_mask_with_ellipsis_end", (10, 5, 6), (..., random_mask(6))),
    ("bool_mask_with_ellipsis_start", (10, 5, 6), (random_mask(10), ...)),
    # Boolean masking with integer indexing
    ("int_then_bool_mask", (10, 5), (0, random_mask(5))),
    ("bool_mask_then_int", (10, 5), (random_mask(10), 0)),
    # Edge cases: all false/true masks
    ("all_false_mask_2d", (10, 5), (torch.zeros(10, 5, dtype=torch.bool),)),
    ("all_true_mask_2d", (10, 5), (torch.ones(10, 5, dtype=torch.bool),)),
    (
        "all_false_mask_with_slice",
        (10, 5),
        (slice(None), torch.zeros(5, dtype=torch.bool)),
    ),
    (
        "all_true_mask_with_slice",
        (10, 5),
        (slice(None), torch.ones(5, dtype=torch.bool)),
    ),
    # Multi-dimensional masks (flattening behavior)
    ("multidim_mask_start", (2, 3, 4), (random_mask(2, 3), ...)),
    ("multidim_mask_end", (2, 3, 4), (..., random_mask(3, 4))),
    # Mixed indexing with boolean masks
    ("mixed_slice_mask_int", (10, 5, 8), (slice(2, 7), random_mask(5), 3)),
    ("mixed_mask_int_slice", (10, 5, 8), (random_mask(10), 2, slice(1, 4))),
]


@pytest.mark.skipif_no_compile
class TestBooleanMaskIndexing:
    """Test suite for boolean mask indexing operations."""

    @pytest.mark.parametrize(
        "test_name,shape,idx",
        BOOLEAN_MASK_CASES,
        ids=[case[0] for case in BOOLEAN_MASK_CASES],
    )
    def test_boolean_mask_indexing(self, test_name, shape, idx, request):
        """Test boolean mask indexing on TensorDataClass.

        Verifies that boolean masking works correctly with various mask shapes
        and positions, including edge cases like all-false/all-true masks.
        """
        # Setup with extra feature dimensions
        features = torch.randn(*shape, 4, 5)  # two event dims
        labels = torch.randn(*shape)
        tdc = SampleTensorDataClass(
            features=features, labels=labels, shape=shape, device=features.device
        )

        fullgraph = True
        try:
            torch.compile(_get_item, fullgraph=True)(tdc.labels, idx)
        except Exception:
            fullgraph = False

        # Apply index
        result, _ = run_and_compare_compiled(
            _get_item,
            tdc,
            idx,
            fullgraph=fullgraph,
            expected_graph_breaks=0 if fullgraph else None,
        )

        # Verify type
        assert isinstance(result, SampleTensorDataClass), (
            f"Expected SampleTensorDataClass, got {type(result)}"
        )

        # Apply to source tensors
        expected_features = features[idx + (slice(None), slice(None))]
        expected_labels = labels[idx]

        # Check content
        torch.testing.assert_close(
            result.features, expected_features, msg=f"Features mismatch for {test_name}"
        )
        torch.testing.assert_close(
            result.labels, expected_labels, msg=f"Labels mismatch for {test_name}"
        )

        # Verify shape and device
        assert result.device == expected_features.device, (
            f"Device mismatch for {test_name}"
        )
        assert result.shape == expected_labels.shape, f"Shape mismatch for {test_name}"

    @pytest.mark.parametrize(
        "test_name,mask_value",
        [("zero_dim_true_mask", True), ("zero_dim_false_mask", False)],
    )
    def test_zero_dim_boolean_masks(self, test_name, mask_value, request):
        """Test boolean masking on 0-dimensional TensorDataClass.

        Verifies that 0-dim boolean masks work correctly on 0-dim tensors.
        """
        # Setup: Create a 0-dim TensorDataClass
        features = torch.randn(10)
        labels = torch.arange(10)
        tdc = SampleTensorDataClass(
            features=features, labels=labels, shape=(), device=features.device
        )

        idx = (torch.tensor(mask_value),)

        fullgraph = True
        try:
            torch.compile(_get_item, fullgraph=True)(tdc.labels, idx)
        except Exception:
            fullgraph = False

        # Apply index
        result, _ = run_and_compare_compiled(
            _get_item,
            tdc,
            idx,
            fullgraph=fullgraph,
            expected_graph_breaks=0 if fullgraph else None,
        )

        # Verify type
        assert isinstance(result, SampleTensorDataClass), (
            f"Expected SampleTensorDataClass, got {type(result)}"
        )

        # Apply to source tensors
        expected_features = features[idx]
        expected_labels = labels[idx]

        # Check content
        torch.testing.assert_close(
            result.features, expected_features, msg=f"Features mismatch for {test_name}"
        )
        torch.testing.assert_close(
            result.labels, expected_labels, msg=f"Labels mismatch for {test_name}"
        )

        # Verify shape and device
        assert result.device == expected_features.device, (
            f"Device mismatch for {test_name}"
        )
        assert (*result.shape, 10) == expected_labels.shape, (
            f"Labels shape mismatch for {test_name}"
        )
