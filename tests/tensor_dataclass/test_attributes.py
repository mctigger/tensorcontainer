import pytest
import torch

from tests.conftest import skipif_no_compile

from .conftest import (
    NestedTensorDataClass,
    assert_compilation_works,
    assert_raises_with_message,
    assert_tensor_properties,
)


class TestAttributes:
    """Tests attribute access and method inheritance for TensorDataclass."""

    def test_getattr_tensor_fields(
        self, nested_tensor_data_class: NestedTensorDataClass
    ):
        """Tests direct attribute access for tensor fields."""
        # Test direct attribute access for the main dataclass
        assert isinstance(nested_tensor_data_class.tensor, torch.Tensor)
        assert nested_tensor_data_class.tensor.shape == (2, 3, 4)

        # Test direct attribute access for the nested dataclass
        nested = nested_tensor_data_class.tensor_data_class
        assert isinstance(nested.tensor, torch.Tensor)
        assert nested.tensor.shape == (2, 3, 4)

        # Test non-tensor attribute
        assert nested_tensor_data_class.meta_data == "meta_data_str"

    def test_method_inheritance(self, nested_tensor_data_class: NestedTensorDataClass):
        """Tests TensorContainer method inheritance."""
        # Test that clone returns the correct type
        cloned = nested_tensor_data_class.clone()
        assert isinstance(cloned, NestedTensorDataClass)
        assert cloned.device == nested_tensor_data_class.device

        # Test that view returns the correct type
        # Test that view returns the correct type and reshapes correctly
        viewed = nested_tensor_data_class.view(2, 1)
        assert isinstance(viewed, NestedTensorDataClass)
        assert viewed.device.type == nested_tensor_data_class.device.type
        assert viewed.shape == (2, 1)

        # Check that underlying tensors are reshaped correctly
        # The new batch shape (2, 1) is prepended to the non-batch dimensions (3, 4)
        expected_shape = (2, 1, 3, 4)
        assert viewed.tensor.shape == expected_shape
        assert viewed.tensor_data_class.tensor.shape == expected_shape

    def test_invalid_attribute_access(
        self, nested_tensor_data_class: NestedTensorDataClass
    ):
        """Tests that accessing invalid attributes raises AttributeError."""
        assert_raises_with_message(
            AttributeError,
            f"'{type(nested_tensor_data_class).__name__}' object has no attribute 'invalid_attr'",
            getattr,
            nested_tensor_data_class,
            "invalid_attr",
        )

    def test_optional_fields_attribute_access(
        self, nested_tensor_data_class: NestedTensorDataClass
    ):
        """Tests attribute access for optional fields."""
        assert nested_tensor_data_class.optional_tensor is None
        assert nested_tensor_data_class.optional_meta_data is None

    def test_device_attribute_consistency(self, nested_tensor_data_class):
        """Tests device attribute consistency."""
        device = nested_tensor_data_class.device
        assert nested_tensor_data_class.device == device
        assert_tensor_properties(
            nested_tensor_data_class.tensor, expected_device=device
        )
        assert_tensor_properties(
            nested_tensor_data_class.tensor_data_class.tensor, expected_device=device
        )

    @skipif_no_compile
    @pytest.mark.parametrize(
        "operation",
        [
            lambda td: td.view(-1),
            lambda td: td.clone(),
        ],
    )
    def test_compile_operations(
        self, nested_tensor_data_class: NestedTensorDataClass, operation
    ):
        """Tests that various operations on TensorDataclass can be torch.compiled."""

        def func(td):
            return operation(td)

        assert_compilation_works(func, nested_tensor_data_class)

    @skipif_no_compile
    def test_compile_attribute_access(
        self, nested_tensor_data_class: NestedTensorDataClass
    ):
        """Tests that attribute access within compiled functions works correctly."""

        def func(td: NestedTensorDataClass) -> torch.Tensor:
            # Test accessing tensor attributes within compiled function
            return td.tensor + td.tensor_data_class.tensor

        assert_compilation_works(func, nested_tensor_data_class)

    def test_device_attribute_access(
        self, nested_tensor_data_class: NestedTensorDataClass
    ):
        # Test tensor field access on CUDA
        assert_tensor_properties(
            nested_tensor_data_class.tensor,
            expected_shape=(2, 3, 4),
            expected_device=nested_tensor_data_class.device,
        )
        assert_tensor_properties(
            nested_tensor_data_class.tensor_data_class.tensor,
            expected_shape=(2, 3, 4),
            expected_device=nested_tensor_data_class.device,
        )
