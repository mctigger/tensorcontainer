import pytest
import torch

from src.rtd.tensor_dict import TensorDict


# Define a common set of indices for slicing tests
SLICING_INDICES = [
    (5, slice(None)),  # Single integer index
    (slice(2, 15), slice(None)),  # Basic slice
    (slice(0, 20, 3), slice(None)),  # Slice with step
    ([0, 4, 2, 19, 7], slice(None)),  # List of indices
    (torch.LongTensor([0, 4, 2, 19, 7]), slice(None)),  # LongTensor of indices
    (torch.rand(20) > 0.5, slice(None)),  # BoolTensor mask
    (slice(None), 3),  # Indexing second dimension
    (slice(None), slice(2, 5)),  # Slicing second dimension
    (torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2])),  # Tensor indexing
    (slice(None), torch.rand(5) > 0.5),  # BoolTensor mask on second dim
]


@pytest.mark.parametrize("idx", SLICING_INDICES)
def test_setitem_slicing_with_scalar(idx):
    """
    Tests assigning a scalar to a slice of a TensorDict.
    Verifies that the scalar is broadcast to all tensors in the dict for the given slice.
    """
    features = torch.randn(20, 5, 10)
    labels = torch.arange(20 * 5).reshape(20, 5).float()
    td = TensorDict(
        data={"features": features.clone(), "labels": labels.clone()},
        shape=(20, 5),
    )

    original_features = features.clone()
    original_labels = labels.clone()

    # Perform the slice assignment
    td[idx] = 0.0

    # Calculate expected results
    expected_features = original_features
    expected_labels = original_labels
    expected_features[idx] = 0.0
    expected_labels[idx] = 0.0

    torch.testing.assert_close(td["features"], expected_features)
    torch.testing.assert_close(td["labels"], expected_labels)


@pytest.mark.parametrize("idx", SLICING_INDICES)
def test_setitem_slicing_with_tensordict(idx):
    """
    Tests assigning a source TensorDict to a slice of a destination TensorDict.
    """
    td_dest = TensorDict(
        data={"features": torch.ones(20, 5, 10), "labels": torch.ones(20, 5)},
        shape=(20, 5),
    )

    # Determine the shape of the slice to create a compatible source TensorDict
    dummy_tensor = torch.empty(20, 5)
    slice_shape = dummy_tensor[idx].shape

    td_source = TensorDict(
        data={
            "features": torch.randn(*slice_shape, 10),
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
        ((slice(None),), (19, 5), ValueError),  # Shape mismatch
        (20, None, IndexError),  # Index out of bounds
        ((slice(None), slice(None), 0), None, IndexError),  # Too many indices
    ],
)
def test_setitem_slicing_invalid(idx, value_shape, error):
    """
    Tests invalid slice assignments that should raise errors.
    """
    td = TensorDict(
        data={"features": torch.randn(20, 5, 10), "labels": torch.randn(20, 5)},
        shape=(20, 5),
    )

    if value_shape is not None:
        value = TensorDict(
            data={"features": torch.randn(*value_shape, 10)}, shape=value_shape
        )
    else:
        # For index errors, the value can be a simple scalar.
        value = 0.0

    with pytest.raises(error):
        td[idx] = value


def test_setitem_key_based():
    """
    Tests the dictionary-like `__setitem__` with string keys.
    """
    td = TensorDict(
        data={"features": torch.randn(10, 5)},
        shape=(10,),
    )

    # === Valid Cases ===

    # 1. Add a new tensor with a compatible shape
    new_labels = torch.zeros(10)
    td["labels"] = new_labels
    assert "labels" in td
    torch.testing.assert_close(td["labels"], new_labels)

    # 2. Update an existing tensor
    updated_features = torch.ones(10, 5)
    td["features"] = updated_features
    torch.testing.assert_close(td["features"], updated_features)

    # 3. Add a nested TensorDict with a compatible shape
    nested_td = TensorDict({"nested_feat": torch.rand(10, 2)}, shape=(10,))
    td["nested"] = nested_td
    assert "nested" in td
    assert isinstance(td["nested"], TensorDict)
    torch.testing.assert_close(td["nested"]["nested_feat"], nested_td["nested_feat"])

    # === Invalid Cases ===

    # 1. Assigning a tensor with an incompatible batch shape
    with pytest.raises(
        ValueError, match="is not compatible with the TensorDict's batch shape"
    ):
        td["bad_shape"] = torch.randn(9, 5)  # Mismatched batch dim

    # 2. Assigning a non-Tensor/TensorContainer value
    with pytest.raises(ValueError, match="must be a Tensor or TensorContainer"):
        td["bad_type"] = [1, 2, 3]  # Invalid type

    # 3. Assigning a tensor with a different device (if validation is on)
    if torch.cuda.is_available():
        td_cpu = TensorDict(
            data={"features": torch.randn(10, 5, device="cpu")},
            shape=(10,),
            device="cpu",
        )
        with pytest.raises(
            ValueError, match="is not compatible with the TensorDict's device"
        ):
            td_cpu["bad_device"] = torch.randn(10, 5, device="cuda")
