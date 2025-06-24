import pytest
import torch

from src.rtd.tensor_dataclass import TensorDataClass


class MyTensorDataClass(TensorDataClass):
    features: torch.Tensor  # e.g., shape (B, N, D)
    labels: torch.Tensor  # e.g., shape (B, N)


@pytest.mark.parametrize(
    "idx",
    [
        (5, slice(None)),  # Single integer index for first batch dim
        (slice(2, 15), slice(None)),  # Basic slice object for first batch dim
        (slice(0, 20, 3), slice(None)),  # Slice with custom step for first batch dim
        ([0, 4, 2, 19, 7], slice(None)),  # List of integer indices for first batch dim
        (
            torch.LongTensor([0, 4, 2, 19, 7]),
            slice(None),
        ),  # torch.LongTensor of indices for first batch dim
        (
            torch.rand(20) > 0.5,
            slice(None),
        ),  # torch.BoolTensor mask for first batch dim
        (slice(None), 3),  # Tuple with slice and integer for second batch dim
        (slice(2, 15), slice(None)),  # Tuple with two slices
        (
            slice(None),
            slice(2, 5),
        ),  # Tuple with slice and another slice for second batch dim
        (torch.tensor([0, 1, 2]), slice(None)),  # Tuple with tensor and slice
        (
            slice(None),
            torch.tensor([0, 1, 2]),
        ),  # Tuple with slice and tensor for second batch dim
        (torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2])),  # Tuple with two tensors
        (torch.rand(20) > 0.5, slice(None)),  # Tuple with bool tensor and slice
        (
            slice(None),
            torch.rand(5) > 0.5,
        ),  # Tuple with slice and bool tensor for second batch dim
        (Ellipsis, None),  # Add dimension with Ellipsis and None
        (None, Ellipsis),  # Add dimension at the start with None
    ],
)
def test_setitem_with_scalar(idx):
    features = torch.randn(20, 5, 10)
    labels = torch.arange(20 * 5).reshape(20, 5).float()
    tdc = MyTensorDataClass(
        features=features.clone(),
        labels=labels.clone(),
        shape=(20, 5),
        device=features.device,
    )

    original_features = features.clone()
    original_labels = labels.clone()

    tdc[idx] = 0.0

    expected_features = original_features
    expected_labels = original_labels
    expected_features[idx] = 0.0
    expected_labels[idx] = 0.0

    torch.testing.assert_close(tdc.features, expected_features)
    torch.testing.assert_close(tdc.labels, expected_labels)


@pytest.mark.parametrize(
    "idx",
    [
        (5, slice(None)),  # Single integer index for first batch dim
        (slice(2, 15), slice(None)),  # Basic slice object for first batch dim
        (slice(0, 20, 3), slice(None)),  # Slice with custom step for first batch dim
        ([0, 4, 2, 19, 7], slice(None)),  # List of integer indices for first batch dim
        (
            torch.LongTensor([0, 4, 2, 19, 7]),
            slice(None),
        ),  # torch.LongTensor of indices for first batch dim
        (
            torch.rand(20) > 0.5,
            slice(None),
        ),  # torch.BoolTensor mask for first batch dim
        (slice(None), 3),  # Tuple with slice and integer for second batch dim
        (slice(2, 15), slice(None)),  # Tuple with two slices
        (
            slice(None),
            slice(2, 5),
        ),  # Tuple with slice and another slice for second batch dim
        (torch.tensor([0, 1, 2]), slice(None)),  # Tuple with tensor and slice
        (
            slice(None),
            torch.tensor([0, 1, 2]),
        ),  # Tuple with slice and tensor for second batch dim
        (torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2])),  # Tuple with two tensors
        (torch.rand(20) > 0.5, slice(None)),  # Tuple with bool tensor and slice
        (
            slice(None),
            torch.rand(5) > 0.5,
        ),  # Tuple with slice and bool tensor for second batch dim
        (Ellipsis, None),  # Add dimension with Ellipsis and None
        (None, Ellipsis),  # Add dimension at the start with None
    ],
)
def test_setitem_with_tensor_dataclass(idx):
    tdc_dest = MyTensorDataClass(
        features=torch.ones(20, 5, 10),
        labels=torch.ones(20, 5),
        shape=(20, 5),
        device=torch.device("cpu"),
    )

    dummy_tensor = torch.empty(20, 5)
    slice_shape = dummy_tensor[idx].shape

    tdc_source = MyTensorDataClass(
        features=torch.randn(*slice_shape, 10),
        labels=torch.randn(*slice_shape),
        shape=slice_shape,
        device=torch.device("cpu"),
    )

    original_dest_features = tdc_dest.features.clone()
    original_dest_labels = tdc_dest.labels.clone()

    tdc_dest[idx] = tdc_source

    expected_features = original_dest_features
    expected_labels = original_dest_labels

    def _get_leaf_key(leaf_tensor, key, batch_ndim):
        """Constructs the correct key for a leaf tensor."""
        if not isinstance(key, tuple):
            key = (key,)

        event_ndim = leaf_tensor.ndim - batch_ndim
        num_none = key.count(None)

        try:
            ellipsis_pos = key.index(Ellipsis)
            pre_key = key[:ellipsis_pos]
            post_key = key[ellipsis_pos + 1 :]
            n_ellipsis = batch_ndim + num_none - len(key) + 1
            if n_ellipsis < 0:
                raise IndexError("too many indices for tensor")
            expanded_batch_key = pre_key + (slice(None),) * n_ellipsis + post_key
        except ValueError:
            expanded_batch_key = key

        return expanded_batch_key + (slice(None),) * event_ndim

    final_key_features = _get_leaf_key(expected_features, idx, tdc_dest.ndim)
    final_key_labels = _get_leaf_key(expected_labels, idx, tdc_dest.ndim)

    expected_features[final_key_features] = tdc_source.features
    expected_labels[final_key_labels] = tdc_source.labels

    torch.testing.assert_close(tdc_dest.features, expected_features)
    torch.testing.assert_close(tdc_dest.labels, expected_labels)


@pytest.mark.parametrize(
    "idx, value_shape, error",
    [
        ((slice(None),), (19, 5), ValueError),  # Shape mismatch
        (20, None, IndexError),  # Index out of bounds
        ((slice(None), slice(None), 0), None, IndexError),  # Too many indices
    ],
)
def test_setitem_invalid(idx, value_shape, error):
    tdc = MyTensorDataClass(
        features=torch.randn(20, 5, 10),
        labels=torch.randn(20, 5),
        shape=(20, 5),
        device=torch.device("cpu"),
    )

    value = None
    if value_shape is not None:
        value = MyTensorDataClass(
            features=torch.randn(*value_shape, 10),
            labels=torch.randn(*value_shape),
            shape=value_shape,
            device=torch.device("cpu"),
        )
    else:
        # For index out of bounds, we can assign a scalar or a dummy tensor
        # The error should come from the indexing itself, not the value type
        value = 0.0

    with pytest.raises(error):
        tdc[idx] = value
