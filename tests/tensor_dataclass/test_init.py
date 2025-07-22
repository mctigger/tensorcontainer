import pytest
import torch

from tensorcontainer.tensor_dataclass import TensorDataClass
from tests.conftest import skipif_no_cuda
from tests.tensor_dataclass.conftest import (
    DeviceTestClass,
    OptionalFieldsTestClass,
    SimpleTensorData,
)


class TestInitDevice:
    """Test class for TensorDataClass initialization and validation."""

    def test_device_mismatch_raises(self):
        """Test that device consistency validation catches mismatches."""

        with pytest.raises(RuntimeError):
            DeviceTestClass(
                a=torch.randn(2, 3, device="cpu:0"),
                b=torch.ones(2, 3, device="cpu:0"),
                shape=(2, 3),
                device=torch.device("cpu:1"),
            )

    @skipif_no_cuda
    def test_nested_device_mismatch_raises(self):
        """Test that device validation catches mismatches in nested TensorDataclasses."""

        class Inner(TensorDataClass):
            c: torch.Tensor

        class Outer(TensorDataClass):
            inner: Inner

        with pytest.raises(RuntimeError):
            Outer(
                shape=(2, 3),
                device=torch.device("cpu"),
                inner=Inner(
                    shape=(2, 3),
                    device=torch.device("meta"),
                    c=torch.randn(2, device="meta"),
                ),
            )

    def test_device_none_allows_mixed_devices(self):
        """Test that device=None allows tensors on different devices."""

        td = SimpleTensorData(
            a=torch.randn(2, 3, device="cpu"),
            b=torch.randn(2, 3, device="meta"),
            shape=(2, 3),
            device=None,
        )

        # Device should remain None (no inference)
        assert td.device is None
        # Tensors keep their original devices

        assert td.a.device == torch.device("cpu")
        assert td.b.device == torch.device("meta")


class TestInitShape:
    def test_shape_mismatch_raises(self):
        """Test that initialization with incompatible (non-prefix) shapes raises."""

        with pytest.raises(
            RuntimeError, match=r"Validation error at key \.b: \('Invalid shape torch\.Size\(\[2, 4\]\)\. Expected shape that is compatible to \(2, 3\)',\)"
        ):
            SimpleTensorData(
                a=torch.randn(2, 3),  
                b=torch.randn(2, 4), 
                shape=(2, 3),
                device=torch.device("cpu"),
            )

    def test_nested_shape_mismatch_raises(self):
        """Test that nested TensorDataClass initialization with incompatible inner tensor shape raises."""

        class Inner(TensorDataClass):
            c: torch.Tensor

        class Outer(TensorDataClass):
            inner: Inner

        with pytest.raises(
            RuntimeError,
            match=r"Validation error at key \.inner: \('Invalid shape \(2, 4\)\. Expected shape that is compatible to \(2, 3\)',\)",
        ):
            Outer(
                shape=(2, 3),
                device=torch.device("cpu"),
                inner=Inner(
                    c=torch.randn(2, 4), shape=(2, 4), device=torch.device("cpu")
                ),
            )

    def test_zero_sized_batch_dimensions(self):
        """Test initialization with zero-sized batch dimensions."""

        shape = (0, 5)
        td = SimpleTensorData(
            a=torch.randn(*shape, 10),
            b=torch.randn(*shape, 20),
            shape=shape,
            device=torch.device("cpu"),
        )
        assert td.shape == shape
        assert td.a.shape == (0, 5, 10)
        assert td.b.shape == (0, 5, 20)

    def test_empty_batch_shape(self):
        """Test initialization with an empty batch shape (scalar-like)."""

        shape = ()
        td = SimpleTensorData(
            a=torch.randn(10),
            b=torch.randn(20),
            shape=shape,
            device=torch.device("cpu"),
        )
        assert td.shape == shape
        assert td.a.shape == (10,)
        assert td.b.shape == (20,)


class TestInitFields:
    def test_metadata_only_instance(self):
        """Test initialization of a TensorDataClass with only metadata fields."""

        class MetadataOnly(TensorDataClass):
            name: str
            value: int

        td = MetadataOnly(name="test", value=123, shape=(), device=None)
        assert td.name == "test"
        assert td.value == 123
        assert td.shape == ()
        assert td.device is None

    def test_optional_and_default_fields_set(self):
        """Test initialization with optional and default_factory fields set."""

        # Test with all fields provided
        td1 = OptionalFieldsTestClass(
            obs=torch.randn(2, 3),
            reward=torch.randn(2, 1),
            info=["a", "b"],
            optional_meta="test_meta",
            shape=(2,),
            device=torch.device("cpu"),
        )
        assert td1.obs.shape == (2, 3)
        assert td1.reward is not None
        assert td1.reward.shape == (2, 1)
        assert td1.info == ["a", "b"]
        assert td1.optional_meta == "test_meta"
        assert td1.default_tensor.shape == (2, 3, 4)

    def test_optional_and_default_fields_unset(self):
        """Test initialization with optional and default_factory fields not set."""
        td2 = OptionalFieldsTestClass(
            obs=torch.randn(2, 3), reward=None, shape=(2,), device=None
        )
        assert td2.obs.shape == (2, 3)
        assert td2.reward is None
        assert td2.info == []  # Default factory for list
        assert td2.optional_meta is None
        assert td2.default_tensor.shape == (2, 3, 4)

    def test_inherited_fields(self):
        """Test that a child TensorDataClass correctly inherits and initializes fields from a parent."""

        class ParentTensorData(TensorDataClass):
            parent_tensor: torch.Tensor
            parent_meta: str

        class ChildTensorData(ParentTensorData):
            child_tensor: torch.Tensor
            child_meta: int

        td = ChildTensorData(
            parent_tensor=torch.randn(2, 3),
            parent_meta="parent_val",
            child_tensor=torch.randn(2, 4),
            child_meta=123,
            shape=(2,),
            device=torch.device("cpu"),
        )

        assert td.parent_tensor.shape == (2, 3)
        assert td.parent_meta == "parent_val"
        assert td.child_tensor.shape == (2, 4)
        assert td.child_meta == 123
        assert td.shape == (2,)
        assert td.device == torch.device("cpu")


class TestInitMisc:
    def test_eq_true_subclassing_raises(self):
        """Test that defining TensorDataClass with eq=True raises a TypeError."""
        with pytest.raises(TypeError, match="TensorDataClass requires eq=False."):

            class MyEqTensorDataClass(TensorDataClass, eq=True):  # type: ignore
                a: torch.Tensor
