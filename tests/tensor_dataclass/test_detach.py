from typing import Optional

import torch

from src.rtd.tensor_dataclass import TensorDataclass


class TestDetach:
    def test_basic_detach(self):
        """Test that tensors are detached and still have the same data."""

        class TestClass(TensorDataclass):
            shape: tuple
            device: Optional[torch.device]
            a: torch.Tensor
            b: torch.Tensor

        # Create instance with tensors
        test_instance = TestClass(
            shape=(3,),
            device=torch.device("cpu"),
            a=torch.tensor([1.0, 2.0, 3.0]),
            b=torch.tensor([4.0, 5.0, 6.0]),
        )

        # Detach the instance
        detached_instance = test_instance.detach()

        # Check that tensors are detached
        assert not detached_instance.a.requires_grad
        assert not detached_instance.b.requires_grad

        # Check that original tensors still have gradients
        assert test_instance.a.requires_grad == False
        assert test_instance.b.requires_grad == False

        # Check that data is preserved
        assert torch.equal(detached_instance.a, test_instance.a)
        assert torch.equal(detached_instance.b, test_instance.b)

    def test_detach_with_gradients(self):
        """Test detach with tensors requiring gradients."""

        class TestClass(TensorDataclass):
            shape: tuple
            device: Optional[torch.device]
            a: torch.Tensor
            b: torch.Tensor

        # Create instance with tensors requiring gradients
        test_instance = TestClass(
            shape=(3,),
            device=torch.device("cpu"),
            a=torch.tensor([1.0, 2.0, 3.0], requires_grad=True),
            b=torch.tensor([4.0, 5.0, 6.0], requires_grad=True),
        )

        # Detach the instance
        detached_instance = test_instance.detach()

        # Check that tensors are detached
        assert not detached_instance.a.requires_grad
        assert not detached_instance.b.requires_grad

        # Check that original tensors still have gradients
        assert test_instance.a.requires_grad
        assert test_instance.b.requires_grad

        # Check that data is preserved
        assert torch.equal(detached_instance.a, test_instance.a)
        assert torch.equal(detached_instance.b, test_instance.b)

    def test_nested_detach(self):
        """Test detach with nested TensorDataclass instances."""

        class NestedClass(TensorDataclass):
            shape: tuple
            device: Optional[torch.device]
            c: torch.Tensor

        class TestClass(TensorDataclass):
            shape: tuple
            device: Optional[torch.device]
            a: torch.Tensor
            b: NestedClass

        # Create nested instance
        nested = NestedClass(
            shape=(3,),
            device=torch.device("cpu"),
            c=torch.tensor([7.0, 8.0, 9.0], requires_grad=True),
        )
        test_instance = TestClass(
            shape=(3,),
            device=torch.device("cpu"),
            a=torch.tensor([1.0, 2.0, 3.0], requires_grad=True),
            b=nested,
        )

        # Detach the instance
        detached_instance = test_instance.detach()

        # Check that tensors are detached
        assert not detached_instance.a.requires_grad
        assert not detached_instance.b.c.requires_grad

        # Check that original tensors still have gradients
        assert test_instance.a.requires_grad
        assert test_instance.b.c.requires_grad

        # Check that data is preserved
        assert torch.equal(detached_instance.a, test_instance.a)
        assert torch.equal(detached_instance.b.c, test_instance.b.c)

    def test_non_tensor_fields(self):
        """Test that non-tensor fields are preserved."""

        class TestClass(TensorDataclass):
            shape: tuple
            device: Optional[torch.device]
            a: torch.Tensor
            b: int
            c: str

        # Create instance with mixed fields
        test_instance = TestClass(
            shape=(3,),
            device=torch.device("cpu"),
            a=torch.tensor([1.0, 2.0, 3.0]),
            b=42,
            c="test",
        )

        # Detach the instance
        detached_instance = test_instance.detach()

        # Check that non-tensor fields are preserved
        assert detached_instance.b == 42
        assert detached_instance.c == "test"

        # Check that tensors are detached
        assert not detached_instance.a.requires_grad

        # Check that data is preserved
        assert torch.equal(detached_instance.a, test_instance.a)
