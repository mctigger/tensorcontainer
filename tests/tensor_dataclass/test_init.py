import dataclasses
from typing import cast

import pytest
import torch

from tensorcontainer.tensor_dataclass import TensorDataClass
from tests.compile_utils import run_and_compare_compiled
from tests.conftest import skipif_no_compile, skipif_no_cuda
from tests.tensor_dataclass.conftest import (
    DeviceTestClass,
    OptionalFieldsTestClass,
    SimpleTensorData,
)


class InitFalseTestClass(TensorDataClass):
    a: torch.Tensor
    b: torch.Tensor = dataclasses.field(init=False)
    c: str = dataclasses.field(init=False, default="hello")

    def __post_init__(self):
        super().__post_init__()
        # b must be initialized here based on other fields
        if hasattr(self, "a"):
            self.b = self.a + 1


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

        def _test_nested_device_mismatch():
            return Outer(
                shape=(2, 3),
                device=torch.device("cpu"),
                inner=Inner(
                    shape=(2, 3),
                    device=torch.device("meta"),
                    c=torch.randn(2, device="meta"),
                ),
            )

        with pytest.raises(RuntimeError):
            run_and_compare_compiled(_test_nested_device_mismatch)

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

        def _test_nested_shape_mismatch():
            return Outer(
                shape=(2, 3),
                device=torch.device("cpu"),
                inner=Inner(
                    c=torch.randn(2, 4), shape=(2, 4), device=torch.device("cpu")
                ),
            )

        with pytest.raises(
            RuntimeError,
            match=r"Validation error at key \.inner: \('Invalid shape \(2, 4\)\. Expected shape that is compatible to \(2, 3\)',\)",
        ):
            run_and_compare_compiled(_test_nested_shape_mismatch)

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

        def _test_all_fields():
            return OptionalFieldsTestClass(
                obs=torch.randn(2, 3),
                reward=torch.randn(2, 1),
                info=["a", "b"],
                optional_meta="test_meta",
                shape=(2,),
                device=torch.device("cpu"),
            )

        td1, _ = run_and_compare_compiled(_test_all_fields)
        assert td1.obs.shape == (2, 3)
        assert td1.reward is not None
        assert td1.reward.shape == (2, 1)
        assert td1.info == ["a", "b"]
        assert td1.optional_meta == "test_meta"
        assert td1.default_tensor.shape == (2, 3, 4)

    def test_optional_and_default_fields_unset(self):
        """Test initialization with optional and default_factory fields not set."""
        
        def _test_unset_fields():
            return OptionalFieldsTestClass(
                obs=torch.randn(2, 3), reward=None, shape=(2,), device=None
            )

        td2, _ = run_and_compare_compiled(_test_unset_fields)
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


class TestOptionalTensorInitialization:
    """
    Tests the initialization logic of TensorDataclass with optional fields.

    This suite verifies that:
    - The dataclass can be initialized when an Optional[Tensor] field is None.
    - The dataclass can be initialized when an Optional[Tensor] field is a Tensor.
    - Shape validation correctly raises an error when a tensor has an incompatible batch size, even with optional fields.
    """

    def test_init_with_optional_tensor_as_none(self):
        """
        Tests that the dataclass can be initialized when an Optional[Tensor]
        field is None.
        """

        def _test_init_none():
            return OptionalFieldsTestClass(
                shape=(2, 3),
                device=None,
                obs=torch.ones(2, 3, 10),
                reward=None,
            )

        instance, _ = run_and_compare_compiled(_test_init_none)
        assert instance.obs.shape == (2, 3, 10)
        assert instance.reward is None
        assert instance.shape == (2, 3)

    def test_init_with_optional_tensor_as_tensor(self):
        """
        Tests that the dataclass can be initialized when an Optional[Tensor]
        field is a Tensor.
        """

        def _test_init_all_fields():
            return OptionalFieldsTestClass(
                shape=(2, 3),
                device=None,
                obs=torch.ones(2, 3, 10),
                reward=torch.zeros(2, 3, 2),
            )

        instance, _ = run_and_compare_compiled(_test_init_all_fields)
        assert instance.obs.shape == (2, 3, 10)
        assert instance.reward is not None
        assert instance.reward.shape == (2, 3, 2)
        assert instance.shape == (2, 3)

    def test_init_raises_error_on_shape_mismatch(self):
        with pytest.raises(RuntimeError):
            OptionalFieldsTestClass(
                shape=(2, 3),
                device=None,
                obs=torch.ones(3, 10),
                reward=None,
            )


class TestDefaultValueFields:
    """
    Tests the behavior of fields defined with `default` or `default_factory` in
    TensorDataClass.

    This suite verifies that:
    - Metadata fields with `default_factory` are correctly handled during cloning.
    - Tensor fields with `default_factory` are correctly handled during stacking.
    - Using a mutable `default` for a tensor field raises a `ValueError`.
    """

    def test_default_factory_metadata_clone(self):
        """
        Tests that metadata fields with `default_factory` are correctly handled
        during cloning.
        """

        def _test_default_factory():
            data = OptionalFieldsTestClass(
                shape=(2, 3), device=None, obs=torch.ones(2, 3, 5), reward=None
            )
            return data, data.clone()

        (data, cloned_data), _ = run_and_compare_compiled(_test_default_factory)
        assert data.info == []
        assert cloned_data.info == []
        assert cloned_data.info is data.info

    @skipif_no_compile
    def test_default_factory_tensor_stack(self):
        """
        Tests that tensor fields with `default_factory` are correctly handled
        during stacking.
        """

        def _test_default_factory_tensor():
            data1 = OptionalFieldsTestClass(
                shape=(2, 3),
                device=None,
                obs=torch.ones(2, 3, 32, 32),
                reward=None,
                info=["step1"],
            )
            data2 = OptionalFieldsTestClass(
                shape=(2, 3),
                device=None,
                obs=torch.ones(2, 3, 32, 32),
                reward=None,
                info=["step1"],
            )
            stacked_data = cast(
                OptionalFieldsTestClass,
                torch.stack([data1, data2], dim=0),  # type: ignore
            )
            return data1, stacked_data

        (data1, stacked_data), _ = run_and_compare_compiled(
            _test_default_factory_tensor
        )
        assert data1.default_tensor.shape == (2, 3, 4)
        assert torch.equal(data1.default_tensor, torch.zeros(2, 3, 4))
        assert stacked_data.default_tensor.shape == (2, 2, 3, 4)
        assert torch.equal(stacked_data.default_tensor, torch.zeros(2, 2, 3, 4))


class TestInitFalseFields:
    """
    Tests the behavior of fields with `init=False` in TensorDataClass.

    This suite verifies that:
    - Fields with `init=False` are not expected in the constructor.
    - `__post_init__` can be used to initialize `init=False` fields.
    - `init=False` fields with defaults are correctly set.
    - Stacking works correctly with `init=False` fields.
    """

    def test_init_false_with_default_value(self):
        """
        Tests that a non-tensor field with init=False and a default value is
        correctly initialized.
        """

        class InitFalseDefault(TensorDataClass):
            a: torch.Tensor
            b: int = dataclasses.field(init=False, default=10)

        def _test():
            return InitFalseDefault(a=torch.ones(4), shape=(4,), device=None)

        instance, _ = run_and_compare_compiled(_test)
        assert instance.b == 10
        assert instance.shape == (4,)


class TestFieldArguments:
    """
    Tests other `dataclasses.field` arguments like `repr`.

    This suite verifies that:
    - Fields with `repr=False` are excluded from the string representation.
    """

    def test_repr_false(self):
        """
        Tests that a field with repr=False is excluded from the string
        representation.
        """

        class ReprFalseTestClass(TensorDataClass):
            a: torch.Tensor
            b: torch.Tensor = dataclasses.field(repr=False)

        def _test():
            return ReprFalseTestClass(
                a=torch.ones(2), b=torch.zeros(2), shape=(2,), device=None
            )

        instance, _ = run_and_compare_compiled(_test)
        instance_repr = repr(instance)
        assert "a=" in instance_repr
        assert "b=" not in instance_repr


class TestInitMisc:
    def test_eq_true_subclassing_raises(self):
        """Test that defining TensorDataClass with eq=True raises a TypeError."""
        with pytest.raises(TypeError, match="TensorDataClass requires eq=False."):

            class MyEqTensorDataClass(TensorDataClass, eq=True):  # type: ignore
                a: torch.Tensor
