import pytest
import torch
from tensorcontainer.tensor_dict import TensorDict  # Adjust as needed


def are_devices_equal(device1, device2):
    """
    Compare two devices for equality, normalizing their representations.

    Args:
        device1: First device (str or torch.device)
        device2: Second device (str or torch.device)

    Returns:
        bool: True if the devices are equivalent
    """
    # Convert string to torch.device if needed
    if isinstance(device1, str):
        device1 = torch.device(device1)
    if isinstance(device2, str):
        device2 = torch.device(device2)

    # Now both should be torch.device objects
    # This properly compares only the device type, not the index
    return device1.type == device2.type


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tensordict_to_changes_device():
    td = TensorDict({"a": torch.randn(4, 3)}, shape=(4,), device=torch.device("cpu"))
    td_cuda = td.to(torch.device("cuda"))

    # Check if device is 'cuda' or a cuda device object
    assert are_devices_equal(td_cuda.device, "cuda")

    # Check tensor devices
    for v in td_cuda.data.values():
        assert are_devices_equal(v.device, "cuda")


def test_tensordict_homogeneous_device_cpu():
    data = {
        "a": torch.randn(4, 3),
        "b": torch.zeros(4, 5),
    }
    td = TensorDict(data, shape=(4,), device=torch.device("cpu"))
    for v in td.values():
        assert are_devices_equal(v.device, "cpu")

    # Check if device is 'cpu' or a cpu device object
    assert are_devices_equal(td.device, "cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tensordict_update_with_mismatched_device_raises():
    td = TensorDict({"a": torch.randn(4, 3, device="cuda")}, shape=(4,), device="cuda")
    with pytest.raises(RuntimeError):
        td.update({"b": torch.randn(4, 3, device="cpu")})


def test_device_persistence_across_operations():
    td = TensorDict({"a": torch.randn(4, 3)}, shape=(4, 3), device=torch.device("cpu"))
    td2 = td.view(1, 12)
    assert are_devices_equal(td2.device, td.device)

    td3 = td.clone()
    assert are_devices_equal(td3.device, td.device)

    td4 = td2.expand(12, 12)
    assert are_devices_equal(td4.device, td.device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tensordict_cpu_method():
    td = TensorDict(
        {"a": torch.randn(4, 3, device="cuda")},
        shape=(4,),
        device="cuda",
    )
    td_cpu = td.cpu()

    assert are_devices_equal(td_cpu.device, "cpu")
    for v in td_cpu.data.values():
        assert are_devices_equal(v.device, "cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tensordict_cuda_method():
    td = TensorDict({"a": torch.randn(4, 3)}, shape=(4,), device=torch.device("cpu"))
    td_cuda = td.cuda()

    assert are_devices_equal(td_cuda.device, "cuda")
    for v in td_cuda.data.values():
        assert are_devices_equal(v.device, "cuda")
