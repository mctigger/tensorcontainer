import pytest
import torch

from src.rtd.tensor_dataclass import TensorDataClass


def random_mask(*shape):
    """Creates a random boolean mask of a given shape."""
    return torch.rand(shape) > 0.5


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
        random_mask(2),
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
    (random_mask(2), slice(None)),  # Tuple with bool tensor and slice
    (
        slice(None),
        random_mask(3),
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
        (random_mask(21), slice(None)),  # Bool tensor mask with wrong size
        (
            slice(None),
            random_mask(6),
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


class TestGetitemBooleanMask:
    boolean_mask_params = [
        # id, shape, idx
        ("a[m]_1d", (10,), (random_mask(10),)),
        ("a[m]_2d", (10, 5), (random_mask(10, 5),)),
        ("a[:, m]", (10, 5), (slice(None), random_mask(5))),
        ("a[m, :]", (10, 5), (random_mask(10), slice(None))),
        ("a[..., m]", (10, 5, 6), (..., random_mask(6))),
        ("a[m, ...]", (10, 5, 6), (random_mask(10), ...)),
        ("a[0, m]", (10, 5), (0, random_mask(5))),
        ("a[m, 0]", (10, 5), (random_mask(10), 0)),
        ("a[m]_all_false", (10, 5), (torch.zeros(10, 5, dtype=torch.bool),)),
        ("a[m]_all_true", (10, 5), (torch.ones(10, 5, dtype=torch.bool),)),
        ("a[:, m]_all_false", (10, 5), (slice(None), torch.zeros(5, dtype=torch.bool))),
        ("a[:, m]_all_true", (10, 5), (slice(None), torch.ones(5, dtype=torch.bool))),
        (
            "a[m, ...]_multidim_mask",
            (2, 3, 4),
            (random_mask(2, 3), ...),
        ),  # PyTorch will flatten the first two batch dims
        (
            "a[..., m]_multidim_mask",
            (2, 3, 4),
            (..., random_mask(3, 4)),
        ),  # PyTorch will flatten the last two batch dims
        ("mixed_indexing_1", (10, 5, 8), (slice(2, 7), random_mask(5), 3)),
        ("mixed_indexing_2", (10, 5, 8), (random_mask(10), 2, slice(1, 4))),
    ]

    @pytest.mark.parametrize(
        "shape, idx",
        [p[1:] for p in boolean_mask_params],
        ids=[p[0] for p in boolean_mask_params],
    )
    def test_getitem_boolean_mask(self, shape, idx):
        # Setup
        features = torch.randn(*shape, 4, 5)  # two event dims
        labels = torch.randn(*shape)
        tdc = MyTensorDataClass(
            features=features, labels=labels, shape=shape, device=features.device
        )

        # Apply index
        result = tdc[idx]

        # Verify type
        assert isinstance(result, MyTensorDataClass)

        # Apply to source tensors
        expected_features = features[idx + (slice(None), slice(None))]
        expected_labels = labels[idx]

        # Check content
        torch.testing.assert_close(result.features, expected_features)
        torch.testing.assert_close(result.labels, expected_labels)

        # Verify shape and device
        assert result.device == expected_features.device
        assert result.shape == expected_labels.shape

    boolean_mask_0_dim_params = [
        ("0_dim_tdc_true_mask", (), (torch.tensor(True),)),
        ("0_dim_tdc_false_mask", (), (torch.tensor(False),)),
    ]

    @pytest.mark.parametrize(
        "shape, idx",
        [p[1:] for p in boolean_mask_0_dim_params],
        ids=[p[0] for p in boolean_mask_0_dim_params],
    )
    def test_getitem_0_dim_masks(self, shape, idx):
        # Setup: Create a 0-dim TensorDataClass
        features = torch.randn(10)
        labels = torch.arange(10)
        tdc = MyTensorDataClass(
            features=features, labels=labels, shape=shape, device=features.device
        )

        # Apply index
        result = tdc[idx]

        # Verify type
        assert isinstance(result, MyTensorDataClass)

        # Apply to source tensors
        expected_features = features[idx]
        expected_labels = labels[idx]

        # Check content
        torch.testing.assert_close(result.features, expected_features)
        torch.testing.assert_close(result.labels, expected_labels)

        # Verify shape and device
        assert result.device == expected_features.device
        assert (*result.shape, 10) == expected_labels.shape
