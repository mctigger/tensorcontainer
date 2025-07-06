import pytest
import torch

from tensorcontainer.tensor_dataclass import TensorDataClass
from tests.conftest import skipif_no_compile
from tests.tensor_dataclass.conftest import SimpleTensorData


@skipif_no_compile
class TestDetach:
    """Test suite for detach operations on TensorDataclass instances."""

    @staticmethod
    def _detach_operation(tensor_dataclass_instance):
        """Helper method for detach operations."""
        return tensor_dataclass_instance.detach()

    @pytest.mark.parametrize("compile_mode", [False, True])
    def test_basic_detach(self, simple_tensor_data_instance, compile_mode):
        """Test that tensors are detached and still have the same data in both eager and compile modes."""
        if compile_mode:
            pytest.importorskip("torch._dynamo", reason="Compilation not available")

        test_instance = simple_tensor_data_instance

        if compile_mode:
            compiled_detach = torch.compile(self._detach_operation, fullgraph=True)
            detached_instance = compiled_detach(test_instance)
        else:
            detached_instance = self._detach_operation(test_instance)

        # Check that tensors are detached
        assert not detached_instance.a.requires_grad
        assert not detached_instance.b.requires_grad

        # Check that original tensors still have gradients (should be False for fixture)
        assert not test_instance.a.requires_grad
        assert not test_instance.b.requires_grad

        # Check that data is preserved
        assert torch.equal(detached_instance.a, test_instance.a)
        assert torch.equal(detached_instance.b, test_instance.b)

    @pytest.mark.parametrize("compile_mode", [False, True])
    def test_detach_with_gradients(self, compile_mode):
        """Test detach with tensors requiring gradients in both eager and compile modes."""
        if compile_mode:
            pytest.importorskip("torch._dynamo", reason="Compilation not available")

        # Create instance with tensors requiring gradients
        test_instance = SimpleTensorData(
            a=torch.tensor([1.0, 2.0, 3.0], requires_grad=True),
            b=torch.tensor([4.0, 5.0, 6.0], requires_grad=True),
            shape=(3,),
            device="cpu",
        )

        if compile_mode:
            compiled_detach = torch.compile(self._detach_operation, fullgraph=True)
            detached_instance = compiled_detach(test_instance)
        else:
            detached_instance = self._detach_operation(test_instance)

        # Check that tensors are detached
        assert not detached_instance.a.requires_grad
        assert not detached_instance.b.requires_grad

        # Check that original tensors still have gradients
        assert test_instance.a.requires_grad
        assert test_instance.b.requires_grad

        # Check that data is preserved
        assert torch.equal(detached_instance.a, test_instance.a)
        assert torch.equal(detached_instance.b, test_instance.b)

    @pytest.mark.parametrize("compile_mode", [False, True])
    def test_nested_detach(self, compile_mode):
        """Test detach with nested TensorDataclass instances in both eager and compile modes."""
        if compile_mode:
            pytest.importorskip("torch._dynamo", reason="Compilation not available")

        class NestedClass(TensorDataClass):
            c: torch.Tensor

        class TestClass(TensorDataClass):
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

        if compile_mode:
            compiled_detach = torch.compile(self._detach_operation, fullgraph=True)
            detached_instance = compiled_detach(test_instance)
        else:
            detached_instance = self._detach_operation(test_instance)

        # Check that tensors are detached
        assert not detached_instance.a.requires_grad
        assert not detached_instance.b.c.requires_grad

        # Check that original tensors still have gradients
        assert test_instance.a.requires_grad
        assert test_instance.b.c.requires_grad

        # Check that data is preserved
        assert torch.equal(detached_instance.a, test_instance.a)
        assert torch.equal(detached_instance.b.c, test_instance.b.c)

    @pytest.mark.parametrize("compile_mode", [False, True])
    def test_non_tensor_fields(self, compile_mode):
        """Test that non-tensor fields are preserved in both eager and compile modes."""
        if compile_mode:
            pytest.importorskip("torch._dynamo", reason="Compilation not available")

        class TestClass(TensorDataClass):
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

        if compile_mode:
            compiled_detach = torch.compile(self._detach_operation, fullgraph=True)
            detached_instance = compiled_detach(test_instance)
        else:
            detached_instance = self._detach_operation(test_instance)

        # Check that non-tensor fields are preserved
        assert detached_instance.b == 42
        assert detached_instance.c == "test"

        # Check that tensors are detached
        assert not detached_instance.a.requires_grad

        # Check that data is preserved
        assert torch.equal(detached_instance.a, test_instance.a)
