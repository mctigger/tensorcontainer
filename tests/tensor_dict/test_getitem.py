import pytest
import torch

from tensorcontainer.tensor_dict import TensorDict


def normalize_device(dev: torch.device) -> torch.device:
    d = torch.device(dev)
    # If no index was given, fill in current_device() for CUDA, leave CPU as-is
    if d.type == "cuda" and d.index is None:
        idx = (
            torch.cuda.current_device()
        )  # e.g. 0 :contentReference[oaicite:4]{index=4}
        return torch.device(f"cuda:{idx}")
    return d


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
