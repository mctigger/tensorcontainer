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
    ],
)
def test_getitem(idx):
    # Setup: Create a sample TensorDataClass instance
    features = torch.randn(20, 5, 10)
    labels = torch.arange(20 * 5).reshape(20, 5)
    tdc = MyTensorDataClass(
        features=features, labels=labels, shape=(20, 5), device=features.device
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
    assert result.shape == expected_features.shape[:-1]
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
    features = torch.randn(20, 5, 10)
    labels = torch.arange(20 * 5).reshape(20, 5)
    tdc = MyTensorDataClass(
        features=features, labels=labels, shape=(20, 5), device=features.device
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
