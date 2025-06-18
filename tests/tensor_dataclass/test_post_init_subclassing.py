import pytest
import torch
from src.rtd.tensor_dataclass import TensorDataclass


class SubclassedTensorDataclass(TensorDataclass):
    my_tensor: torch.Tensor
    initialized_value: int = 0

    def __post_init__(self):
        # Call the base class's __post_init__
        super().__post_init__()
        # Custom initialization logic
        self.initialized_value = 100


def test_subclass_with_post_init():
    """
    Test that a subclass of TensorDataclass can implement __post_init__
    without crashing and that the base __post_init__ is called.
    """

    # Test instantiation
    tensor_data = torch.randn(2, 3)
    instance = SubclassedTensorDataclass(
        my_tensor=tensor_data, shape=(2, 3), device=torch.device("cpu")
    )

    # Verify that the base TensorDataclass __post_init__ logic was executed
    # (e.g., shape and device validation)
    assert instance.shape == (2, 3)
    assert instance.device == torch.device("cpu")
    assert torch.equal(instance.my_tensor, tensor_data)

    # Verify that the subclass's custom __post_init__ logic was executed
    assert instance.initialized_value == 100

    # Test that TensorDataclass methods still work
    cloned_instance = instance.clone()
    assert isinstance(cloned_instance, SubclassedTensorDataclass)
    assert torch.equal(cloned_instance.my_tensor, instance.my_tensor)
    assert cloned_instance.initialized_value == instance.initialized_value
    assert cloned_instance.my_tensor is not instance.my_tensor  # Should be a new tensor

    # Test with a device mismatch to ensure base validation still works
    with pytest.raises(ValueError, match="Device mismatch"):
        SubclassedTensorDataclass(
            my_tensor=torch.randn(2, 3, device=torch.device("cuda"))
            if torch.cuda.is_available()
            else torch.randn(2, 3),
            shape=(2, 3),
            device=torch.device("cpu"),
        )

    # Test with a shape mismatch to ensure base validation still works
    with pytest.raises(ValueError, match="Shape mismatch"):
        SubclassedTensorDataclass(
            my_tensor=torch.randn(2, 4), shape=(2, 3), device=torch.device("cpu")
        )
