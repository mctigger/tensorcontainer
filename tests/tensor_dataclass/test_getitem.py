import pytest
import torch

from src.rtd.tensor_dataclass import TensorDataClass
from tests.tensor_dict.compile_utils import run_and_compare_compiled


class SampleTensorDataClass(TensorDataClass):
    """Test dataclass with features and labels tensors."""

    features: torch.Tensor  # shape (B, N, D) or (B, N, D1, D2)
    labels: torch.Tensor  # shape (B, N)


def create_sample_tdc(batch_shape=(2, 3), feature_dims=4, num_event_dims=1):
    """Helper to create a sample TensorDataClass for testing."""
    if num_event_dims == 1:
        features = torch.randn(*batch_shape, feature_dims)
    elif num_event_dims == 2:
        features = torch.randn(
            *batch_shape, feature_dims, 5
        )  # Adding another event dim
    else:
        raise ValueError("num_event_dims must be 1 or 2")

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


class TestGetItem:
    """Consolidated test suite for __getitem__ operations on TensorDataClass."""

    def _run_and_verify(self, tdc, idx, test_name):
        """
        Central helper method to encapsulate common testing logic for __getitem__.
        """

        # This function is defined locally to be pickled correctly by torch.compile
        def _get_item(tdc, idx):
            return tdc[idx]

        fullgraph = True
        try:
            # Use a copy for the compilation check to avoid modifying the original
            torch.compile(_get_item, fullgraph=True)(tdc.clone(), idx)
        except Exception:
            fullgraph = False

        result, _ = run_and_compare_compiled(
            _get_item,
            tdc,
            idx,
            fullgraph=fullgraph,
            expected_graph_breaks=0 if fullgraph else None,
        )

        assert isinstance(result, SampleTensorDataClass), (
            f"Expected SampleTensorDataClass, got {type(result)} for {test_name}"
        )

        labels_event_dims = len(tdc.labels.shape) - len(tdc.shape)
        features_event_dims = len(tdc.features.shape) - len(tdc.shape)

        expected_labels = tdc.labels[idx]

        if features_event_dims > labels_event_dims:
            slicing = (slice(None),) * (features_event_dims - labels_event_dims)
            features_idx = idx + slicing if isinstance(idx, tuple) else (idx,) + slicing
        else:
            features_idx = idx

        expected_features = tdc.features[features_idx]

        torch.testing.assert_close(
            result.features, expected_features, msg=f"Features mismatch for {test_name}"
        )
        torch.testing.assert_close(
            result.labels, expected_labels, msg=f"Labels mismatch for {test_name}"
        )

        assert result.device == expected_features.device, (
            f"Device mismatch for {test_name}"
        )

        labels_event_shape = tdc.labels.shape[len(tdc.shape) :]
        assert result.shape + labels_event_shape == expected_labels.shape, (
            f"Shape mismatch for {test_name} (labels): expected {expected_labels.shape}, got {result.shape} (batch) + {labels_event_shape} (event)"
        )

        features_event_shape = tdc.features.shape[len(tdc.shape) :]
        assert result.shape + features_event_shape == expected_features.shape, (
            f"Shape mismatch for {test_name} (features): expected {expected_features.shape}, got {result.shape} (batch) + {features_event_shape} (event)"
        )

    # Combined indexing test cases
    COMBINED_INDEXING_CASES = [
        # Standard indexing
        ("single_int_first_dim", (0, slice(None))),
        ("slice_first_dim", (slice(2, 15), slice(None))),
        ("slice_with_step", (slice(0, 20, 3), slice(None))),
        ("list_indices", ([0, 1], slice(None))),
        ("tensor_indices", (torch.LongTensor([0, 1]), slice(None))),
        ("bool_mask_first_dim", (random_mask(2), slice(None))),
        ("single_int_second_dim", (slice(None), 0)),
        ("slice_second_dim", (slice(None), slice(2, 5))),
        ("tensor_second_dim", (slice(None), torch.tensor([0, 1]))),
        ("bool_mask_second_dim", (slice(None), random_mask(3))),
        ("both_slices", (slice(2, 15), slice(None))),
        ("tensor_and_slice", (torch.tensor([0, 1]), slice(None))),
        ("two_tensors", (torch.tensor([0, 1]), torch.tensor([0, 1]))),
        ("bool_and_slice", (random_mask(2), slice(None))),
        # Ellipsis indexing
        ("ellipsis_only", (Ellipsis,)),
        ("ellipsis_with_int", (Ellipsis, 0)),
        ("int_with_ellipsis", (0, Ellipsis)),
        ("newaxis_ellipsis", (None, Ellipsis)),
        ("ellipsis_newaxis", (Ellipsis, None)),
        ("int_ellipsis_newaxis", (0, Ellipsis, None)),
        ("newaxis_ellipsis_int", (None, Ellipsis, 0)),
        # Newaxis indexing
        ("newaxis_start", (None, slice(None), slice(None))),
        ("newaxis_middle", (slice(None), None, slice(None))),
        ("newaxis_end", (slice(None), slice(None), None)),
        ("two_newaxis_start", (None, None, slice(None), slice(None))),
        ("two_newaxis_end", (slice(None), slice(None), None, None)),
    ]

    @pytest.mark.parametrize(
        "test_name,idx",
        COMBINED_INDEXING_CASES,
        ids=[case[0] for case in COMBINED_INDEXING_CASES],
    )
    def test_various_indexing(self, test_name, idx):
        """Test various indexing operations on TensorDataClass."""
        tdc = create_sample_tdc()
        self._run_and_verify(tdc, idx, test_name)

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

    @pytest.mark.parametrize(
        "test_name,shape,idx",
        BOOLEAN_MASK_CASES,
        ids=[case[0] for case in BOOLEAN_MASK_CASES],
    )
    def test_boolean_mask_indexing(self, test_name, shape, idx):
        """Test boolean mask indexing on TensorDataClass."""
        tdc = create_sample_tdc(batch_shape=shape, num_event_dims=2)
        self._run_and_verify(tdc, idx, test_name)

    @pytest.mark.parametrize(
        "test_name,mask_value",
        [("zero_dim_true_mask", True), ("zero_dim_false_mask", False)],
    )
    def test_zero_dim_boolean_masks(self, test_name, mask_value):
        """Test boolean masking on 0-dimensional TensorDataClass."""
        features = torch.randn(10)
        labels = torch.arange(10)
        tdc = SampleTensorDataClass(
            features=features, labels=labels, shape=(), device=features.device
        )
        idx = (torch.tensor(mask_value),)
        self._run_and_verify(tdc, idx, test_name)

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
    def test_invalid_indexing_raises_error(self, test_name, idx):
        """Test that invalid indexing operations raise appropriate errors."""
        tdc = create_sample_tdc()
        with pytest.raises(IndexError, match=".*"):
            _ = tdc[idx]

    def test_zero_dim_indexing_raises_error(self):
        """Test that indexing a 0-dim TensorDataClass raises IndexError."""
        features_0d = torch.randn(10)
        labels_0d = torch.arange(10)
        tdc = SampleTensorDataClass(
            features=features_0d, labels=labels_0d, shape=(), device=features_0d.device
        )
        with pytest.raises(IndexError, match=".*"):
            _ = tdc[0]
