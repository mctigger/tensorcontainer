from typing import Optional

import pytest
import torch

from rtd.tensor_dataclass import TensorDataclass


class DeviceTestClass(TensorDataclass):
    # Parent class required fields (no defaults)
    device: Optional[torch.device]
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

    assert td.device is not None
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

    assert td.device is not None
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

    assert result.device is not None
    assert result.device.type == "cuda"
    assert result.a.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_nested_device_mismatch_raises():
    """Test that device validation catches mismatches in nested TensorDataclasses."""

    class Inner(TensorDataclass):
        shape: tuple
        device: Optional[torch.device]
        c: torch.Tensor

    class Outer(TensorDataclass):
        shape: tuple
        device: Optional[torch.device]
        inner: Inner

    with pytest.raises(ValueError, match="Device mismatch"):
        Outer(
            shape=(2,),
            device=torch.device("cpu"),
            inner=Inner(
                shape=(2,),
                device=torch.device("cuda"),
                c=torch.randn(2, device="cuda"),
            ),
        )
