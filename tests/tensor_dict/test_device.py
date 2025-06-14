import pytest
import torch
from rtd.tensor_dict import TensorDict  # Adjust as needed


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
    # This properly compares both device type and index
    return device1 == device2


def get_device_type(device):
    """
    Get the device type regardless of whether it's a string or torch.device.

    Args:
        device: Device (str or torch.device)

    Returns:
        str: Device type (without index)
    """
    if isinstance(device, str):
        # For string devices like 'cuda:0', extract just the type part
        return device.split(":")[0]
    else:
        # For torch.device objects, use the type attribute
        return device.type


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tensordict_to_changes_device():
    td = TensorDict({"a": torch.randn(4, 3)}, shape=(4,), device=torch.device("cpu"))
    td_cuda = td.to(torch.device("cuda"))

    # Check if device is 'cuda' or a cuda device object
    if isinstance(td_cuda.device, str):
        assert td_cuda.device == "cuda" or td_cuda.device.startswith("cuda:")
    else:
        assert td_cuda.device.type == "cuda"

    # Check tensor devices
    for v in td_cuda.data.values():
        assert v.device.type == "cuda"


def test_tensordict_homogeneous_device_cpu():
    data = {
        "a": torch.randn(4, 3),
        "b": torch.zeros(4, 5),
    }
    td = TensorDict(data, shape=(4,), device=torch.device("cpu"))
    for v in td.values():
        assert v.device.type == "cpu"

    # Check if device is 'cpu' or a cpu device object
    if isinstance(td.device, str):
        assert td.device == "cpu"
    else:
        assert td.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tensordict_update_with_mismatched_device_raises():
    td = TensorDict(
        {"a": torch.randn(4, 3, device="cuda")}, shape=(4,), device=torch.device("cuda")
    )
    with pytest.raises(Exception):  # Replace with your specific device error
        td.update({"b": torch.randn(4, 3, device="cpu")})


def test_device_persistence_across_operations():
    td = TensorDict({"a": torch.randn(4, 3)}, shape=(4, 3), device=torch.device("cpu"))
    td2 = td.view(1, 12)
    assert are_devices_equal(td2.device, td.device)

    td3 = td.clone()
    assert are_devices_equal(td3.device, td.device)

    td4 = td2.expand(12, 12)
    assert are_devices_equal(td4.device, td.device)
