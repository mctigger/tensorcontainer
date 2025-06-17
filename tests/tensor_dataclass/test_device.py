import pytest
import torch
import dataclasses
from rtd.tensor_dataclass import TensorDataclass


@dataclasses.dataclass
class DeviceTestClass(TensorDataclass):
    # Parent class required fields (no defaults)
    device: torch.device
    shape: tuple

    # Instance tensor fields
    a: torch.Tensor
    b: torch.Tensor

    # Optional field with default
    meta: int = 42


def test_device_propagation():
    td = DeviceTestClass(
        a=torch.randn(2, 3, device=torch.device("cuda")),
        b=torch.ones(2, 3, device=torch.device("cuda")),
        shape=(2, 3),
        device=torch.device("cuda"),
    )

    assert td.device.type == "cuda"
    assert td.a.device.type == "cuda"
    assert td.b.device.type == "cuda"


def test_to_device():
    td = DeviceTestClass(
        device=torch.device("cpu"),
        shape=(2, 3),
        a=torch.randn(2, 3),
        b=torch.ones(2, 3),
    ).to(torch.device("cuda"))

    assert td.device.type == "cuda"
    assert td.a.device.type == "cuda"
    assert td.b.device.type == "cuda"


def test_device_consistency_check():
    with pytest.raises(ValueError):
        DeviceTestClass(
            a=torch.randn(2, 3, device=torch.device("cuda")),
            b=torch.ones(2, 3),
            shape=(2, 3),
            device=torch.device("cpu"),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_compile():
    def device_fn(td):
        return td.to("cuda")

    td = DeviceTestClass(
        a=torch.randn(2, 3),
        b=torch.ones(2, 3),
        shape=(2, 3),
        device=torch.device("cpu"),
    )

    compiled_fn = torch.compile(device_fn, fullgraph=True)
    result = compiled_fn(td)

    assert result.device.type == "cuda"
    assert result.a.device.type == "cuda"
