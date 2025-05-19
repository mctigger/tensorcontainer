import pytest
import torch
from rtd.tensor_dict import TensorDict  # Adjust as needed


def test_tensordict_homogeneous_device_cpu():
    data = {
        "a": torch.randn(4, 3),
        "b": torch.zeros(4, 5),
    }
    td = TensorDict(data, shape=(4,), device=torch.device("cpu"))
    for v in td.data.values():
        assert v.device.type == "cpu"
    assert td.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tensordict_mixed_device_creation_raises():
    data = {
        "a": torch.randn(4, 3, device="cpu"),
        "b": torch.randn(4, 3, device="cuda"),
    }
    with pytest.raises(Exception):  # Replace with your specific device error if defined
        TensorDict(data, shape=(4,))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tensordict_update_with_mismatched_device_raises():
    td = TensorDict(
        {"a": torch.randn(4, 3, device="cuda")}, shape=(4,), device=torch.device("cuda")
    )
    with pytest.raises(Exception):  # Replace with your specific device error
        td.update({"b": torch.randn(4, 3, device="cpu")})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tensordict_to_changes_device():
    td = TensorDict({"a": torch.randn(4, 3)}, shape=(4,), device=torch.device("cpu"))
    td_cuda = td.to(torch.device("cuda"))
    assert td_cuda.device.type == "cuda"
    for v in td_cuda.data.values():
        assert v.device.type == "cuda"


def test_device_persistence_across_operations():
    td = TensorDict({"a": torch.randn(4, 3)}, shape=(4,), device=torch.device("cpu"))
    td2 = td.view(2, 2)
    assert td2.device == td.device

    td3 = td.clone()
    assert td3.device == td.device

    td4 = td.expand(4)
    assert td4.device == td.device
