"""Tests the `__repr__` method of `TensorDict` and `TensorDataClass`."""

import pytest
import torch

from src.tensorcontainer.tensor_dataclass import TensorDataClass
from src.tensorcontainer.tensor_dict import TensorDict


class MyData(TensorDataClass):
    """A simple TensorDataClass for testing."""

    features: torch.Tensor
    labels: torch.Tensor


@pytest.fixture(scope="class")
def tensordict_instance():
    """Returns a sample TensorDict for testing `__repr__`."""
    return TensorDict(
        {
            "a": {"aa": torch.randn(2, 4), "ab": torch.randn(2, 4)},
            "b": torch.randn(2, 4),
        },
        shape=(2,),
        device="cpu",
    )


@pytest.fixture(scope="class")
def tensordataclass_instance():
    """Returns a sample TensorDataClass for testing `__repr__`."""
    return MyData(
        features=torch.randn(4, 10),
        labels=torch.arange(4).float(),
        shape=torch.Size([4]),
        device=torch.device("cpu"),
    )


class TestTensorDictRepr:
    """
    Tests the `__repr__` method of `TensorDict`.

    This suite verifies that the `__repr__` string for a `TensorDict` includes:
    - The `TensorDict` class name.
    - The correct shape.
    - The correct device.
    - A representation of its items (tensors and nested containers).
    """

    def test_class_name(self, tensordict_instance):
        """Tests that the class name is in the `__repr__` string."""
        repr_str = repr(tensordict_instance)
        assert "TensorDict" in repr_str

    def test_shape(self, tensordict_instance):
        """Tests that the shape is in the `__repr__` string."""
        repr_str = repr(tensordict_instance)
        assert "shape=torch.Size([2])" in repr_str

    def test_device(self, tensordict_instance):
        """Tests that the device is in the `__repr__` string."""
        repr_str = repr(tensordict_instance)
        assert "device=cpu" in repr_str

    def test_items(self, tensordict_instance):
        """Tests that items are correctly represented in the `__repr__` string."""
        repr_str = repr(tensordict_instance)
        # 'a' is a nested TensorDict, so its repr should indicate that.
        assert "['a']: TensorDict(" in repr_str
        assert (
            "['b']: Tensor(shape=torch.Size([2, 4]), device=cpu, dtype=torch.float32)"
            in repr_str
        )


class TestTensorDataClassRepr:
    """
    Tests the `__repr__` method of `TensorDataClass`.

    This suite verifies that the `__repr__` string for a `TensorDataClass` includes:
    - The subclass name (`MyData`).
    - The correct shape.
    - The correct device.
    - A representation of its tensor fields.
    """

    def test_class_name(self, tensordataclass_instance):
        """Tests that the class name is in the `__repr__` string."""
        repr_str = repr(tensordataclass_instance)
        assert "MyData" in repr_str

    def test_shape(self, tensordataclass_instance):
        """Tests that the shape is in the `__repr__` string."""
        repr_str = repr(tensordataclass_instance)
        assert "shape=torch.Size([4])" in repr_str

    def test_device(self, tensordataclass_instance):
        """Tests that the device is in the `__repr__` string."""
        repr_str = repr(tensordataclass_instance)
        assert "device=cpu" in repr_str

    def test_fields(self, tensordataclass_instance):
        """Tests that fields are correctly represented in the `__repr__` string."""
        repr_str = repr(tensordataclass_instance)
        assert (
            ".features: Tensor(shape=torch.Size([4, 10]), device=cpu, dtype=torch.float32)"
            in repr_str
        )
        assert (
            ".labels: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.float32)"
            in repr_str
        )
