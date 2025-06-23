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

    def test_getattr_tensor_fields(self, nested_tensor_data_class):
        """Tests direct attribute access for tensor fields."""
        # The dataclass shape is the batch_shape
        assert nested_tensor_data_class.shape == (2, 3)

        # The tensor's shape is batch_shape + event_shape
        expected_tensor_shape = (2, 3, 4, 5)
        assert nested_tensor_data_class.tensor.shape == expected_tensor_shape
        assert (
            nested_tensor_data_class.tensor_data_class.tensor.shape
            == expected_tensor_shape
        )

        # Test non-tensor attribute
        assert nested_tensor_data_class.meta_data == "meta_data_str"

    def test_method_inheritance(self, nested_tensor_data_class):
        """Tests TensorContainer method inheritance."""
        # Test that clone returns the correct type
        cloned = nested_tensor_data_class.clone()
        assert isinstance(cloned, NestedTensorDataClass)
        if nested_tensor_data_class.device is not None:
            assert cloned.device.type == nested_tensor_data_class.device.type
        else:
            assert cloned.device is None

        # Test that view returns the correct type and reshapes the batch dimension
        viewed = nested_tensor_data_class.view(6)
        assert isinstance(viewed, NestedTensorDataClass)
        if nested_tensor_data_class.device is not None:
            assert viewed.device.type == nested_tensor_data_class.device.type
        else:
            assert viewed.device is None
        assert viewed.shape == (6,)

        # Check that underlying tensors are reshaped correctly
        # The new batch shape (6,) is prepended to the event_shape (4, 5)
        assert viewed.tensor.shape == (6, 4, 5)
        assert viewed.tensor_data_class.tensor.shape == (6, 4, 5)

    def test_invalid_attribute_access(self, nested_tensor_data_class):
        """Tests that accessing invalid attributes raises AttributeError."""
        assert_raises_with_message(
            AttributeError,
            f"'{type(nested_tensor_data_class).__name__}' object has no attribute 'invalid_attr'",
            getattr,
            nested_tensor_data_class,
            "invalid_attr",
        )

    def test_optional_fields_attribute_access(self, nested_tensor_data_class):
        """Tests attribute access for optional fields."""
        assert nested_tensor_data_class.optional_tensor is None
        assert nested_tensor_data_class.optional_meta_data is None

    def test_device_attribute_consistency(self, nested_tensor_data_class):
        """Tests device attribute consistency."""
        device = nested_tensor_data_class.device
        assert nested_tensor_data_class.device.type == device.type
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
            lambda td: td.view(6),
            lambda td: td.clone(),
        ],
    )
    def test_compile_operations(self, nested_tensor_data_class, operation):
        """Tests that various operations on TensorDataclass can be torch.compiled."""

        def func(td):
            return operation(td)

        assert_compilation_works(func, nested_tensor_data_class)

    @skipif_no_compile
    def test_compile_attribute_access(self, nested_tensor_data_class):
        """Tests that attribute access within compiled functions works correctly."""

        def func(td: NestedTensorDataClass) -> torch.Tensor:
            # Test accessing tensor attributes within compiled function
            return td.tensor + td.tensor_data_class.tensor

        assert_compilation_works(func, nested_tensor_data_class)

    def test_cuda_attribute_access(self, nested_tensor_data_class):
        """Tests attribute access for CUDA tensors."""
        if nested_tensor_data_class.device.type != "cuda":
            pytest.skip("Test requires CUDA device")

        # Test tensor field access on CUDA
        assert_tensor_properties(
            nested_tensor_data_class.tensor,
            expected_shape=(2, 3, 4, 5),
            expected_device=torch.device("cuda"),
        )
        assert_tensor_properties(
            nested_tensor_data_class.tensor_data_class.tensor,
            expected_shape=(2, 3, 4, 5),
            expected_device=torch.device("cuda"),
        )

        # Test device attribute
        assert nested_tensor_data_class.device.type == "cuda"
