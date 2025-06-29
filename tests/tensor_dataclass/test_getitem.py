import pytest
import torch

from src.tensorcontainer.tensor_dataclass import TensorDataClass
from tests.compile_utils import run_and_compare_compiled


class SampleTensorDataClass(TensorDataClass):
    """Test dataclass with features and labels tensors."""

    features: torch.Tensor  # shape (B, N, D) or (B, N, D1, D2)
    labels: torch.Tensor  # shape (B, N)


def create_sample_tdc(batch_shape=(20, 5), feature_dims=10, num_event_dims=1):
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
    labels = torch.arange(total_elements).reshape(batch_shape).float()
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

    def _run_and_verify_getitem(self, tdc, idx, test_name, fullgraph=True):
        """
        Central helper method to encapsulate common testing logic for __getitem__.
        """
        try:
            # Use a copy for the compilation check to avoid modifying the original
            torch.compile(_get_item, fullgraph=fullgraph)(tdc.clone(), idx)
        except Exception:
            if fullgraph:
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

    # Basic indexing cases (integers, slices)
    BASIC_INDEXING_CASES = [
        ("int_first_dim", 0),
        ("int_second_dim", (slice(None), 3)),
        ("slice_first_dim", slice(2, 15)),
        ("slice_second_dim", (slice(None), slice(1, 4))),
        ("slice_with_step", slice(0, 20, 3)),
        ("int_slice_tuple", (4, slice(1, 4))),
        ("slice_int_tuple", (slice(2, 8), 3)),
        ("slice_slice_tuple", (slice(1, 10, 2), slice(0, 4, 2))),
    ]

    @pytest.mark.parametrize(
        "test_name,idx",
        BASIC_INDEXING_CASES,
        ids=[case[0] for case in BASIC_INDEXING_CASES],
    )
    def test_getitem_basic_indexing(self, test_name, idx):
        """Tests basic indexing with integers and slices.

        This test covers standard tensor indexing using integers and slice objects.
        It verifies that selecting a part of the TensorDataClass works as
        expected and that the operation is torch.compile compatible.

        Example with torch.Tensor:
            >>> tensor = torch.randn(10, 5)
            >>> tensor[0]
            >>> tensor[2:5]
            >>> tensor[:, 1:3]
        """
        tdc = create_sample_tdc()
        self._run_and_verify_getitem(tdc, idx, test_name)

    # Advanced indexing cases (lists, tensors)
    ADVANCED_INDEXING_CASES = [
        ("list_indices", [0, 4, 2, 19, 7]),
        ("long_tensor_indices", torch.tensor([0, 4, 2, 19, 7])),
        ("tensor_slice_tuple", (torch.tensor([0, 1, 2]), slice(None))),
        ("slice_tensor_tuple", (slice(None), torch.tensor([0, 1, 2]))),
        ("tensor_tensor_tuple", (torch.tensor([0, 1]), torch.tensor([2, 3]))),
    ]

    @pytest.mark.parametrize(
        "test_name,idx",
        ADVANCED_INDEXING_CASES,
        ids=[case[0] for case in ADVANCED_INDEXING_CASES],
    )
    def test_getitem_advanced_indexing(self, test_name, idx):
        """Tests advanced indexing with lists and tensors.

        This test covers advanced indexing using lists of integers or long tensors.
        It checks that selecting elements specified by advanced indexing works
        correctly and compiles.

        Example with torch.Tensor:
            >>> tensor = torch.randn(10)
            >>> indices = [0, 4, 8]
            >>> tensor[indices]
            >>> tensor[torch.tensor([0, 4, 8])]
        """
        tdc = create_sample_tdc()
        self._run_and_verify_getitem(tdc, idx, test_name)

    # Ellipsis indexing cases
    ELLIPSIS_INDEXING_CASES = [
        ("ellipsis_only", Ellipsis),
        ("ellipsis_int_end", (Ellipsis, 0)),
        ("int_ellipsis_start", (0, Ellipsis)),
        ("ellipsis_slice", (Ellipsis, slice(0, 2))),
        ("slice_ellipsis", (slice(0, 10), Ellipsis)),
    ]

    @pytest.mark.parametrize(
        "test_name,idx",
        ELLIPSIS_INDEXING_CASES,
        ids=[case[0] for case in ELLIPSIS_INDEXING_CASES],
    )
    def test_getitem_ellipsis_indexing(self, test_name, idx):
        """Tests indexing with Ellipsis (...).

        This test checks that using an Ellipsis (...) for indexing works
        correctly and compiles. Ellipsis is used to select all dimensions not
        explicitly sliced.

        Example with torch.Tensor:
            >>> tensor = torch.randn(2, 3, 4)
            >>> tensor[..., 0]
            >>> tensor[0, ...]
        """
        tdc = create_sample_tdc()
        self._run_and_verify_getitem(tdc, idx, test_name)

    # Newaxis indexing cases
    NEWAXIS_INDEXING_CASES = [
        ("newaxis_start", (None,)),
        ("newaxis_end", (Ellipsis, None)),
        ("newaxis_middle", (slice(None), None, slice(None))),
        ("int_ellipsis_newaxis", (0, Ellipsis, None)),
        ("newaxis_ellipsis_int", (None, Ellipsis, 0)),
    ]

    @pytest.mark.parametrize(
        "test_name,idx",
        NEWAXIS_INDEXING_CASES,
        ids=[case[0] for case in NEWAXIS_INDEXING_CASES],
    )
    def test_getitem_newaxis_indexing(self, test_name, idx):
        """Tests indexing with `None` (newaxis).

        This test checks that using `None` to insert a new dimension of size 1
        works correctly and compiles.

        Example with torch.Tensor:
            >>> tensor = torch.randn(10, 5)
            >>> tensor[None, ...].shape # (1, 10, 5)
            >>> tensor[..., None].shape # (10, 5, 1)
        """
        tdc = create_sample_tdc()
        self._run_and_verify_getitem(tdc, idx, test_name)

    # Boolean mask indexing test cases
    BOOLEAN_MASK_INDEXING_CASES = [
        ("bool_mask_1d", (20,), (random_mask(20),)),
        ("bool_mask_2d", (20, 5), (random_mask(20, 5),)),
        ("bool_mask_first_dim", (20, 5), (random_mask(20), slice(None))),
        ("bool_mask_second_dim", (20, 5), (slice(None), random_mask(5))),
        ("int_then_bool_mask", (20, 5), (0, random_mask(5))),
        ("bool_mask_then_int", (20, 5), (random_mask(20), 0)),
        ("all_false_mask", (20, 5), (torch.zeros(20, 5, dtype=torch.bool),)),
        ("all_true_mask", (20, 5), (torch.ones(20, 5, dtype=torch.bool),)),
    ]

    @pytest.mark.parametrize(
        "test_name,shape,idx",
        BOOLEAN_MASK_INDEXING_CASES,
        ids=[case[0] for case in BOOLEAN_MASK_INDEXING_CASES],
    )
    def test_getitem_boolean_mask_indexing(self, test_name, shape, idx):
        """Tests boolean mask indexing.

        This test verifies that selecting elements from a TensorDataClass using a
        boolean mask works correctly. Due to the nature of boolean mask
        indexing, this is tested with `fullgraph=False`.

        Example with torch.Tensor:
            >>> tensor = torch.randn(5)
            >>> mask = torch.tensor([True, False, True, False, True])
            >>> tensor[mask] # selects elements at indices 0, 2, 4
        """
        tdc = create_sample_tdc(batch_shape=shape, num_event_dims=2)
        self._run_and_verify_getitem(tdc, idx, test_name, fullgraph=False)

    @pytest.mark.parametrize(
        "test_name,mask_value",
        [("zero_dim_true_mask", True), ("zero_dim_false_mask", False)],
    )
    def test_getitem_zero_dim_boolean_masks(self, test_name, mask_value):
        """Tests boolean masking on a 0-dimensional TensorDataClass.

        This test checks the edge case of indexing a 0-dim TensorDataClass
        (which represents a batch of data with a single element) with a 0-dim
        boolean tensor.

        Example with torch.Tensor:
            >>> tensor = torch.randn(4) # A single item in a batch
            >>> tensor[torch.tensor(True)].shape # torch.Size([1, 4])
            >>> tensor[torch.tensor(False)].shape # torch.Size([0, 4])
        """
        features = torch.randn(10)
        labels = torch.arange(10)
        tdc = SampleTensorDataClass(
            features=features, labels=labels, shape=(), device=features.device
        )
        idx = torch.tensor(mask_value)
        self._run_and_verify_getitem(tdc, (idx,), test_name)

    # Invalid indexing test cases
    INVALID_INDEXING_CASES = [
        (
            "too_many_indices",
            (slice(None), slice(None), 0),
            IndexError,
            "too many indices",
        ),
        (
            "out_of_bounds_first_dim",
            20,
            IndexError,
            "index 20 is out of bounds for dimension 0 with size 20",
        ),
        (
            "out_of_bounds_second_dim",
            (slice(None), 5),
            IndexError,
            "index 5 is out of bounds for dimension 1 with size 5",
        ),
        (
            "tensor_out_of_bounds",
            torch.tensor([0, 20]),
            IndexError,
            "out of bounds",
        ),
        (
            "bool_mask_wrong_size",
            random_mask(21),
            IndexError,
            "The shape of the mask .* at index 0 does not match the shape of the indexed tensor .* at index 0",
        ),
    ]

    @pytest.mark.parametrize(
        "test_name,idx,error,match",
        INVALID_INDEXING_CASES,
        ids=[case[0] for case in INVALID_INDEXING_CASES],
    )
    def test_getitem_invalid_indexing_raises_error(self, test_name, idx, error, match):
        """Tests that invalid indexing operations raise appropriate errors.

        This test ensures that `__getitem__` correctly raises errors in eager
        mode for various invalid scenarios, such as out-of-bounds indices and
        incorrectly shaped boolean masks.

        Example with torch.Tensor:
            >>> import pytest
            >>> tensor = torch.zeros(5)
            >>> with pytest.raises(IndexError, match="out of bounds"):
            ...     tensor[10]
        """
        tdc = create_sample_tdc()
        with pytest.raises(error, match=match):
            _ = tdc[idx]

    def test_getitem_zero_dim_indexing_raises_error(self):
        """Test that indexing a 0-dim TensorDataClass raises IndexError."""
        features_0d = torch.randn(10)
        labels_0d = torch.arange(10)
        tdc = SampleTensorDataClass(
            features=features_0d, labels=labels_0d, shape=(), device=features_0d.device
        )
        with pytest.raises(
            IndexError,
            match="Cannot index a 0-dimensional TensorContainer with a single index",
        ):
            _ = tdc[0]
