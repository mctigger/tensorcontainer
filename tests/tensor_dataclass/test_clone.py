from typing import Optional

import pytest
import torch

from rtd.tensor_dataclass import TensorDataclass
from tests.tensor_dataclass.conftest import (
    CloneTestClass,
    assert_tensor_equal_and_different_objects,
    create_nested_tensor_dataclass,
)
from tests.conftest import skipif_no_compile


class TestClone:
    @pytest.mark.parametrize("requires_grad", [True, False])
    def test_basic_clone(self, clone_test_instance, requires_grad):
        """Test that clone creates a new instance with copied tensors."""
        # Modify the fixture instance to test both grad scenarios
        if not requires_grad:
            clone_test_instance.a.requires_grad_(False)
            clone_test_instance.b.requires_grad_(False)

        cloned_td = clone_test_instance.clone()

        # Check that it's a new instance
        assert cloned_td is not clone_test_instance
        assert cloned_td.a is not clone_test_instance.a
        assert cloned_td.b is not clone_test_instance.b

        # Check that data is equal but objects are different
        assert_tensor_equal_and_different_objects(cloned_td.a, clone_test_instance.a)
        assert_tensor_equal_and_different_objects(cloned_td.b, clone_test_instance.b)

        # Check that requires_grad is preserved
        assert cloned_td.a.requires_grad == clone_test_instance.a.requires_grad
        assert cloned_td.b.requires_grad == clone_test_instance.b.requires_grad

        # Check non-tensor fields are copied
        assert cloned_td.meta == clone_test_instance.meta

        # Modify original and check cloned is unchanged
        original_a = clone_test_instance.a.clone()
        clone_test_instance.a = torch.zeros_like(clone_test_instance.a)
        assert torch.equal(cloned_td.a, original_a)
        assert not torch.equal(cloned_td.a, clone_test_instance.a)

    def test_clone_nested(self):
        """Test clone with nested TensorDataclass instances."""
        outer, inner = create_nested_tensor_dataclass()
        cloned_outer = outer.clone()

        # Check that all objects are different
        assert cloned_outer is not outer
        assert cloned_outer.inner is not outer.inner  # type: ignore
        assert cloned_outer.inner.c is not outer.inner.c  # type: ignore

        # Check that data is equal but objects are different
        assert_tensor_equal_and_different_objects(cloned_outer.inner.c, outer.inner.c)  # type: ignore

        # Check that requires_grad is preserved
        assert cloned_outer.inner.c.requires_grad == outer.inner.c.requires_grad  # type: ignore

    def test_clone_empty_dataclass(self):
        """Test cloning an empty TensorDataclass."""

        class EmptyCloneTestClass(TensorDataclass):
            shape: tuple
            device: Optional[torch.device]

        td = EmptyCloneTestClass(shape=(), device=torch.device("cpu"))
        cloned_td = td.clone()

        assert cloned_td is not td
        assert cloned_td.shape == td.shape
        assert cloned_td.device == td.device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_clone_on_cuda(self):
        """Test cloning a TensorDataclass on CUDA."""
        td = CloneTestClass(
            a=torch.randn(2, 3, device="cuda", requires_grad=True),
            b=torch.ones(2, 3, device="cuda", requires_grad=True),
            shape=(2, 3),
            device=torch.device("cuda"),
        )
        cloned_td = td.clone()

        assert cloned_td.a.device.type == "cuda"  # type: ignore
        assert cloned_td.b.device.type == "cuda"  # type: ignore
        assert cloned_td.device.type == "cuda"  # type: ignore
        assert_tensor_equal_and_different_objects(cloned_td.a, td.a)  # type: ignore
        assert_tensor_equal_and_different_objects(cloned_td.b, td.b)  # type: ignore
        assert cloned_td.a.requires_grad == td.a.requires_grad  # type: ignore
        assert cloned_td.b.requires_grad == td.b.requires_grad  # type: ignore

    @skipif_no_compile
    def test_clone_compile(self):
        """Tests that a function using TensorDataclass.clone() can be torch.compiled."""
        from tests.tensor_dict.compile_utils import run_and_compare_compiled

        class MyData(TensorDataclass):
            shape: tuple
            device: Optional[torch.device]
            x: torch.Tensor
            y: torch.Tensor

        def func(td: MyData) -> MyData:
            return td.clone()  # type: ignore

        data = MyData(
            x=torch.ones(3, 4),
            y=torch.zeros(3, 4),
            shape=(3, 4),
            device=torch.device("cpu"),
        )
        run_and_compare_compiled(func, data)

    def test_clone_mutable_metadata(self):
        """Test that clone() deepcopies mutable metadata."""

        class MutableMetadata(TensorDataclass):
            shape: tuple
            device: Optional[torch.device]
            a: torch.Tensor
            metadata: list

        td = MutableMetadata(
            shape=(2,),
            device=torch.device("cpu"),
            a=torch.randn(2),
            metadata=[1, 2],
        )

        cloned_td = td.clone()
        assert cloned_td.metadata == [1, 2]  # type: ignore
        assert cloned_td.metadata is not td.metadata  # type: ignore

        cloned_td.metadata.append(3)  # type: ignore
        assert td.metadata == [1, 2]  # type: ignore
        assert cloned_td.metadata == [1, 2, 3]  # type: ignore
