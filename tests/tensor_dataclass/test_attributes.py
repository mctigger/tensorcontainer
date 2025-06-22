import pytest
import torch

from tests.conftest import skipif_no_compile
from .conftest import (
    SimpleTensorData,
    DeviceTestClass,
    CloneTestClass,
    ShapeTestClass,
    assert_tensor_properties,
    assert_compilation_works,
    assert_raises_with_message,
    COMMON_DEVICES,
)


class TestAttributes:
    """Tests attribute access and method inheritance for TensorDataclass."""

    @pytest.mark.parametrize(
        "fixture_name,expected_shape,tensor_fields",
        [
            ("simple_tensor_data_instance", (3, 4), ["a", "b"]),
            ("device_test_instance", (2, 3), ["a", "b"]),
            ("clone_test_instance", (2, 3), ["a", "b"]),
            ("shape_test_instance", (4, 5), ["a", "b"]),
        ],
    )
    def test_getattr_tensor_fields(
        self, request, fixture_name, expected_shape, tensor_fields
    ):
        """Tests direct attribute access for tensor fields across different TensorDataclass types."""
        instance = request.getfixturevalue(fixture_name)

        # Test direct attribute access for each tensor field
        for field_name in tensor_fields:
            tensor = getattr(instance, field_name)
            assert isinstance(tensor, torch.Tensor)
            assert tensor.shape == expected_shape

    @pytest.mark.parametrize(
        "fixture_name,dataclass_type",
        [
            ("simple_tensor_data_instance", SimpleTensorData),
            ("device_test_instance", DeviceTestClass),
            ("clone_test_instance", CloneTestClass),
            ("shape_test_instance", ShapeTestClass),
        ],
    )
    def test_method_inheritance(self, request, fixture_name, dataclass_type):
        """Tests TensorContainer method inheritance across different TensorDataclass types."""
        instance = request.getfixturevalue(fixture_name)

        # Test that clone returns the correct type
        cloned = instance.clone()
        assert isinstance(cloned, dataclass_type)

        # Test that view returns the correct type
        viewed = instance.view(-1)
        assert isinstance(viewed, dataclass_type)

    @pytest.mark.parametrize(
        "fixture_name,invalid_attr",
        [
            ("simple_tensor_data_instance", "invalid"),
            ("device_test_instance", "nonexistent"),
            ("clone_test_instance", "missing_field"),
            ("shape_test_instance", "undefined_attr"),
        ],
    )
    def test_invalid_attribute_access(self, request, fixture_name, invalid_attr):
        """Tests that accessing invalid attributes raises AttributeError."""
        instance = request.getfixturevalue(fixture_name)

        assert_raises_with_message(
            AttributeError,
            f"'{type(instance).__name__}' object has no attribute '{invalid_attr}'",
            getattr,
            instance,
            invalid_attr,
        )

    def test_optional_fields_attribute_access(self, optional_fields_instance):
        """Tests attribute access for TensorDataclass with optional fields."""
        # Test tensor field access
        assert_tensor_properties(
            optional_fields_instance.obs,
            expected_shape=(4, 32, 32),
            expected_device=torch.device("cpu"),
        )

        # Test optional field access
        assert optional_fields_instance.reward is None
        assert optional_fields_instance.info == ["step1"]
        assert optional_fields_instance.optional_meta is None
        assert optional_fields_instance.optional_meta_val == "value"

        # Test default factory field
        assert_tensor_properties(
            optional_fields_instance.default_tensor,
            expected_shape=(4,),
            expected_device=torch.device("cpu"),
        )

    @pytest.mark.parametrize("device", COMMON_DEVICES)
    def test_device_attribute_consistency(self, device):
        """Tests device attribute consistency across different devices."""
        if device.type == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        instance = DeviceTestClass(
            a=torch.randn(2, 3, device=device),
            b=torch.ones(2, 3, device=device),
            shape=(2, 3),
            device=device,
        )

        # Test device attribute access
        assert instance.device == device
        assert_tensor_properties(instance.a, expected_device=device)
        assert_tensor_properties(instance.b, expected_device=device)

    @skipif_no_compile
    @pytest.mark.parametrize(
        "fixture_name,operation",
        [
            ("simple_tensor_data_instance", lambda td: td.view(-1)),
            ("simple_tensor_data_instance", lambda td: td.clone()),
            ("shape_test_instance", lambda td: td.view(20)),
        ],
    )
    def test_compile_operations(self, request, fixture_name, operation):
        """Tests that various operations on TensorDataclass can be torch.compiled."""
        instance = request.getfixturevalue(fixture_name)

        def func(td):
            return operation(td)

        assert_compilation_works(func, instance)

    @skipif_no_compile
    def test_compile_attribute_access(self, simple_tensor_data_instance):
        """Tests that attribute access within compiled functions works correctly."""

        def func(td: SimpleTensorData) -> torch.Tensor:
            # Test accessing tensor attributes within compiled function
            return td.a + td.b

        assert_compilation_works(func, simple_tensor_data_instance)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_attribute_access(self, cuda_device_test_instance):
        """Tests attribute access for CUDA tensors."""
        # Test tensor field access on CUDA
        assert_tensor_properties(
            cuda_device_test_instance.a,
            expected_shape=(2, 3),
            expected_device=torch.device("cuda"),
        )
        assert_tensor_properties(
            cuda_device_test_instance.b,
            expected_shape=(2, 3),
            expected_device=torch.device("cuda"),
        )

        # Test device attribute
        assert cuda_device_test_instance.device.type == "cuda"
