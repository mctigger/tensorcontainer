import torch

from tensorcontainer.tensor_dataclass import TensorDataClass
from tests.conftest import skipif_no_cuda


class TestSubclassing:
    """Test subclassing functionality of TensorDataclass."""

    def test_subclassing_of_subclass(self):
        """Test that subclasses of TensorDataclass subclasses work correctly."""

        class A(TensorDataClass):
            x: torch.Tensor

        class B(A):
            new_attribute: str

        # Test instantiation of B
        x_tensor = torch.randn(2, 3)
        b_instance = B(x=x_tensor, new_attribute="hello", shape=(2, 3), device="cpu")

        assert isinstance(b_instance, B)
        assert isinstance(b_instance, A)
        assert isinstance(b_instance, TensorDataClass)

        assert torch.equal(b_instance.x, x_tensor)
        assert b_instance.new_attribute == "hello"
        assert b_instance.shape == (2, 3)
        assert b_instance.device == torch.device("cpu")

        # Test that methods from TensorDataclass are available
        cloned_b = b_instance.clone()
        assert isinstance(cloned_b, B)
        assert torch.equal(cloned_b.x, b_instance.x)
        assert cloned_b.new_attribute == b_instance.new_attribute

        # Test that B's attributes are correctly handled by TensorDataclass methods
        assert cloned_b.new_attribute == "hello"

    def test_subclass_with_tensor_attribute(self):
        """Test subclass with additional tensor attributes."""

        class A(TensorDataClass):
            x: torch.Tensor

        class B(A):
            y: torch.Tensor  # New tensor attribute

        # Test instantiation of B
        x_tensor = torch.randn(2, 3)
        y_tensor = torch.ones(2, 3)
        b_instance = B(x=x_tensor, y=y_tensor, shape=(2, 3), device=torch.device("cpu"))

        assert isinstance(b_instance, B)
        assert isinstance(b_instance, A)
        assert isinstance(b_instance, TensorDataClass)

        assert torch.equal(b_instance.x, x_tensor)
        assert torch.equal(b_instance.y, y_tensor)
        assert b_instance.shape == (2, 3)
        assert b_instance.device == torch.device("cpu")

        # Test that methods from TensorDataclass are available and handle the new tensor attribute
        cloned_b = b_instance.clone()
        assert isinstance(cloned_b, B)
        assert torch.equal(cloned_b.x, b_instance.x)
        assert torch.equal(cloned_b.y, b_instance.y)
        assert cloned_b.x is not b_instance.x
        assert cloned_b.y is not b_instance.y

        detached_b = b_instance.detach()
        assert isinstance(detached_b, B)
        assert not detached_b.x.requires_grad
        assert not detached_b.y.requires_grad
        assert torch.equal(detached_b.x, b_instance.x)
        assert torch.equal(detached_b.y, b_instance.y)

    @skipif_no_cuda
    def test_subclass_cuda_operations(self):
        """Test CUDA operations on subclassed TensorDataclass."""

        class A(TensorDataClass):
            x: torch.Tensor

        class B(A):
            y: torch.Tensor

        x_tensor = torch.randn(2, 3)
        y_tensor = torch.ones(2, 3)
        b_instance = B(x=x_tensor, y=y_tensor, shape=(2, 3), device=torch.device("cpu"))

        to_cuda_b = b_instance.to(torch.device("cuda"))
        assert to_cuda_b.device is not None and to_cuda_b.device.type == "cuda"
        assert to_cuda_b.x.device.type == "cuda"
        assert to_cuda_b.y.device.type == "cuda"
