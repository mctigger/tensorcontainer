from typing import Optional

import pytest
import torch

from rtd.tensor_dataclass import TensorDataclass
from tests.conftest import skipif_no_compile
from tests.tensor_dataclass.conftest import (
    assert_raises_with_message,
    assert_tensor_equal_and_different_objects,
)


class SubclassedTensorDataclass(TensorDataclass):
    """Test subclass with custom __post_init__ method."""

    my_tensor: torch.Tensor
    initialized_value: int = 0

    def __post_init__(self):
        # Call the base class's __post_init__
        super().__post_init__()
        # Custom initialization logic
        self.initialized_value = 100


class NoSuperPostInit(TensorDataclass):
    """Test subclass that doesn't call super().__post_init__()."""

    shape: tuple
    device: Optional[torch.device]
    a: torch.Tensor

    def __post_init__(self):
        # This subclass does not call super().__post_init__()
        pass


class TestPostInitSubclassing:
    """Test suite for TensorDataclass subclassing with custom __post_init__ methods."""

    def test_subclass_with_post_init(self):
        """Test that a subclass can implement __post_init__ without crashing."""
        # Test instantiation
        tensor_data = torch.randn(2, 3)
        instance = SubclassedTensorDataclass(
            my_tensor=tensor_data, shape=(2, 3), device=torch.device("cpu")
        )

        # Verify that the base TensorDataclass __post_init__ logic was executed
        assert instance.shape == (2, 3)
        assert instance.device == torch.device("cpu")
        assert torch.equal(instance.my_tensor, tensor_data)

        # Verify that the subclass's custom __post_init__ logic was executed
        assert instance.initialized_value == 100

    def test_subclass_methods_work(self):
        """Test that TensorDataclass methods still work on subclasses."""
        tensor_data = torch.randn(2, 3)
        instance = SubclassedTensorDataclass(
            my_tensor=tensor_data, shape=(2, 3), device=torch.device("cpu")
        )

        # Test that TensorDataclass methods still work
        cloned_instance = instance.clone()
        assert isinstance(cloned_instance, SubclassedTensorDataclass)
        assert_tensor_equal_and_different_objects(
            cloned_instance.my_tensor, instance.my_tensor
        )
        assert cloned_instance.initialized_value == instance.initialized_value

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_subclass_device_validation(self):
        """Test that device validation still works in subclasses."""
        assert_raises_with_message(
            ValueError,
            "Device mismatch",
            SubclassedTensorDataclass,
            my_tensor=torch.randn(2, 3, device=torch.device("cuda")),
            shape=(2, 3),
            device=torch.device("cpu"),
        )

    def test_subclass_shape_validation(self):
        """Test that shape validation still works in subclasses."""
        assert_raises_with_message(
            ValueError,
            "Shape mismatch",
            SubclassedTensorDataclass,
            my_tensor=torch.randn(2, 4),
            shape=(2, 3),
            device=torch.device("cpu"),
        )

    def test_post_init_no_super_call(self):
        """Test that validation is skipped if subclass doesn't call super().__post_init__()."""
        # This should not raise an error, because validation is skipped
        td = NoSuperPostInit(
            shape=(3, 4),
            device=torch.device("cpu"),
            a=torch.randn(5, 6),  # Inconsistent shape
        )
        assert td.shape == (3, 4)
        assert td.a.shape == (5, 6)

    @skipif_no_compile
    def test_subclass_compile_compatibility(self):
        """Test that subclassed TensorDataclass works with torch.compile."""
        from tests.tensor_dict.compile_utils import run_and_compare_compiled

        def func(td: SubclassedTensorDataclass):
            return td.clone()

        tensor_data = torch.randn(2, 3)
        instance = SubclassedTensorDataclass(
            my_tensor=tensor_data, shape=(2, 3), device=torch.device("cpu")
        )

        run_and_compare_compiled(func, instance)
