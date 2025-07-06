"""
Tests the `__repr__` method implementations for tensor container classes.

This module contains test suites for verifying the string representation
functionality of `TensorDict` and `TensorDataClass` classes, ensuring
they provide informative and correctly formatted output.
"""

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

    This suite verifies that:
    - The class name 'TensorDict' appears in the string representation.
    - The shape information is correctly displayed in the repr string.
    - The device information is properly included in the output.
    - Nested tensor items are accurately represented with their properties.
    """

    def test_class_name(self, tensordict_instance):
        """
        Verifies that the TensorDict class name appears in the repr output.

        This ensures users can easily identify the object type when debugging.
        """
        repr_str = repr(tensordict_instance)
        assert "TensorDict" in repr_str

    def test_shape(self, tensordict_instance):
        """
        Verifies that the shape information is correctly displayed in repr.

        The shape is crucial for understanding tensor dimensions during debugging.
        """
        repr_str = repr(tensordict_instance)
        assert "shape=(2,)" in repr_str

    def test_device(self, tensordict_instance):
        """
        Verifies that the device information is included in the repr output.

        Device information is essential for debugging GPU/CPU placement issues.
        """
        repr_str = repr(tensordict_instance)
        assert "device=cpu" in repr_str

    def test_items(self, tensordict_instance):
        """
        Verifies that nested items are properly represented in the repr string.

        This ensures both nested TensorDicts and individual tensors show
        their structure and properties clearly for debugging purposes.
        """
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

    This suite verifies that:
    - The subclass name appears correctly in the string representation.
    - The shape information is properly formatted and displayed.
    - The device information is accurately included in the output.
    - All tensor fields are represented with their shape, device, and dtype information.
    """

    def test_class_name(self, tensordataclass_instance):
        """
        Verifies that the subclass name appears in the repr output.

        This helps users identify the specific TensorDataClass subtype
        when working with multiple dataclass types.
        """
        repr_str = repr(tensordataclass_instance)
        assert "MyData" in repr_str

    def test_shape(self, tensordataclass_instance):
        """
        Verifies that the shape information is correctly displayed in repr.

        Shape information is critical for understanding the batch dimensions
        of the dataclass instance during debugging.
        """
        repr_str = repr(tensordataclass_instance)
        assert "shape=torch.Size([4])" in repr_str

    def test_device(self, tensordataclass_instance):
        """
        Verifies that the device information is included in the repr output.

        Device information helps identify where tensors are located
        for debugging GPU/CPU placement issues.
        """
        repr_str = repr(tensordataclass_instance)
        assert "device=device(type='cpu')" in repr_str

    def test_fields(self, tensordataclass_instance):
        """
        Verifies that tensor fields are properly represented with their properties.

        Each field should show its shape, device, and dtype information
        to provide complete debugging context for the dataclass contents.
        """
        repr_str = repr(tensordataclass_instance)
        assert "features=tensor(" in repr_str
        assert "labels=tensor(" in repr_str
