import pytest
import torch

from tensorcontainer.tensor_dict import TensorDict


def normalize_device(dev: torch.device) -> torch.device:
    d = torch.device(dev)
    # If no index was given, fill in current_device() for CUDA, leave CPU as-is
    if d.type == "cuda" and d.index is None:
        if torch.cuda.is_available():
            idx = (
                torch.cuda.current_device()
            )  # e.g. 0 :contentReference[oaicite:4]{index=4}
            return torch.device(f"cuda:{idx}")
        else:
            # If CUDA is not available, return the device as-is
            return d
    return d


def assert_tensor_indexing_parity(
    td_result: TensorDict,
    batch_shape: tuple[int, ...],
    index,
):
    """Assert that indexing a TensorDict produces the same batch shape as
    indexing a plain torch.Tensor with identical leading dimensions.

    This catches any drift between TensorDict's indexing semantics and
    PyTorch's native tensor indexing.
    """
    ref = torch.zeros(batch_shape)
    assert td_result.shape == ref[index].shape


@pytest.fixture
def nested_dict():
    def _make(shape):
        return {
            "x": {
                "a": torch.arange(0, 4).reshape(*shape),
                "b": torch.arange(4, 8).reshape(*shape),
            },
            "y": torch.arange(8, 12).reshape(*shape),
        }

    return _make


def test_getitem_returns_new_tensordict(nested_dict):
    data = nested_dict((2, 2))
    td = TensorDict(data, shape=(2, 2))
    # slicing by an integer produces a new TensorDict
    sliced = td[1]
    assert isinstance(sliced, TensorDict)
    assert sliced is not td

    assert_tensor_indexing_parity(sliced, (2, 2), 1)


def test_modify_top_level_structure_on_slice_does_not_affect_original(nested_dict):
    data = nested_dict((2, 2))
    td = TensorDict(data, shape=(2, 2))
    sliced = td[0]

    # add a new top‐level key to the slice
    sliced["extra"] = torch.tensor([[1, 2], [1, 2]])
    assert "extra" in sliced
    assert "extra" not in td

    # delete an existing top‐level key from the slice
    del sliced["y"]
    assert "y" not in sliced
    assert "y" in td


def test_modify_nested_structure_on_slice_does_not_affect_original(nested_dict):
    data = nested_dict((2, 2))
    td = TensorDict(data, shape=(2, 2))
    sliced = td[1]
    nested = sliced["x"]

    # add a new nested key under "x"
    nested["c"] = torch.tensor([9, 9])
    assert "c" in nested
    assert "c" not in td["x"]

    # remove an existing nested key under "x"
    del nested["a"]
    assert "a" not in nested
    assert "a" in td["x"]


def test_slice_leaf_tensor_content_and_shape(nested_dict):
    data = nested_dict((2, 2))
    td = TensorDict(data, shape=(2, 2))

    # multi‐dimensional indexing
    slice_ = td[1, 0]
    # after slicing both batch dims, batch_shape becomes empty
    assert slice_.shape == torch.Size([])

    assert_tensor_indexing_parity(slice_, (2, 2), (1, 0))

    # leaf values match the underlying tensors
    assert torch.equal(slice_["x"]["a"], data["x"]["a"][1, 0])
    assert torch.equal(slice_["x"]["b"], data["x"]["b"][1, 0])
    assert torch.equal(slice_["y"], data["y"][1, 0])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_slice_preserves_device(nested_dict, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # prepare data on the target device
    data = nested_dict((2, 2))

    def to_device(obj):
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        # dict of tensors or nested dicts
        return {k: to_device(v) for k, v in obj.items()}

    data = to_device(data)

    # create and slice
    td = TensorDict(data, shape=(2, 2), device=torch.device(device))
    sliced = td[0]

    # TensorDict.device should be unchanged
    assert normalize_device(sliced.device) == normalize_device(td.device)

    # leaf tensors should live on the same device
    assert normalize_device(sliced["y"].device) == normalize_device(
        torch.device(device)
    )
    nested = sliced["x"]
    assert normalize_device(nested["a"].device) == normalize_device(
        torch.device(device)
    )
    assert normalize_device(nested["b"].device) == normalize_device(
        torch.device(device)
    )


def test_invalid_getitem_raises_error():
    td = TensorDict({"a": torch.randn(2, 3, 4)}, shape=[2, 3])
    with pytest.raises(
        IndexError,
        match="too many indices for container: container is 2-dimensional, but 3 were indexed",
    ):
        td[:, :, 0]

    # torch.Tensor parity: torch also rejects too many indices on a 2-d tensor
    with pytest.raises(IndexError):
        torch.randn(2, 3)[:, :, 0]


def test_getitem_none_on_zero_dim():
    """td[None] on a 0-dim TensorDict should add a leading dim, matching torch.Tensor semantics."""
    td = TensorDict(
        {"scalar": torch.tensor(5.0), "vec": torch.randn(3)},
        shape=(),
    )
    result = td[None]
    assert result.shape == torch.Size([1])
    assert_tensor_indexing_parity(result, (), None)
    assert result["scalar"].shape == torch.Size([1])
    assert result["vec"].shape == torch.Size([1, 3])


def test_getitem_ellipsis_on_zero_dim():
    """td[...] on a 0-dim TensorDict should return identity, matching torch.Tensor semantics."""
    td = TensorDict(
        {"scalar": torch.tensor(5.0), "vec": torch.randn(3)},
        shape=(),
    )
    result = td[...]
    assert result.shape == torch.Size([])
    assert_tensor_indexing_parity(result, (), Ellipsis)
    assert torch.equal(result["scalar"], td["scalar"])
    assert torch.equal(result["vec"], td["vec"])


def test_getitem_bool_on_zero_dim():
    """td[True]/td[False] on a 0-dim TensorDict should match torch.Tensor semantics."""
    td = TensorDict(
        {"scalar": torch.tensor(5.0), "vec": torch.randn(3)},
        shape=(),
    )
    result_true = td[True]
    assert result_true.shape == torch.Size([1])
    assert_tensor_indexing_parity(result_true, (), True)
    assert result_true["scalar"].shape == torch.Size([1])
    assert result_true["vec"].shape == torch.Size([1, 3])

    result_false = td[False]
    assert result_false.shape == torch.Size([0])
    assert_tensor_indexing_parity(result_false, (), False)
    assert result_false["scalar"].shape == torch.Size([0])
    assert result_false["vec"].shape == torch.Size([0, 3])


def test_getitem_bool_tensor_on_zero_dim():
    """td[torch.tensor(True/False)] on a 0-dim TensorDict should match torch.Tensor semantics."""
    td = TensorDict(
        {"scalar": torch.tensor(5.0), "vec": torch.randn(3)},
        shape=(),
    )
    result_true = td[torch.tensor(True)]
    assert result_true.shape == torch.Size([1])
    assert_tensor_indexing_parity(result_true, (), torch.tensor(True))
    assert result_true["scalar"].shape == torch.Size([1])
    assert result_true["vec"].shape == torch.Size([1, 3])

    result_false = td[torch.tensor(False)]
    assert result_false.shape == torch.Size([0])
    assert_tensor_indexing_parity(result_false, (), torch.tensor(False))
    assert result_false["scalar"].shape == torch.Size([0])
    assert result_false["vec"].shape == torch.Size([0, 3])
