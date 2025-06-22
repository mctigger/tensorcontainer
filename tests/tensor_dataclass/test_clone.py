from typing import Optional

import pytest
import torch

from rtd.tensor_dataclass import TensorDataclass


class CloneTestClass(TensorDataclass):
    a: torch.Tensor
    b: torch.Tensor
    meta: int = 42


@pytest.mark.skipif_no_compile
class TestClone:
    def test_basic_clone(self):
        """Test that clone creates a new instance with copied tensors."""
        td = CloneTestClass(
            a=torch.randn(2, 3, requires_grad=True),
            b=torch.ones(2, 3, requires_grad=True),
            shape=(2, 3),
            device=torch.device("cpu"),
        )
        cloned_td = td.clone()

        # Check that it's a new instance
        assert cloned_td is not td
        assert cloned_td.a is not td.a
        assert cloned_td.b is not td.b

        # Check that data is equal
        assert torch.equal(cloned_td.a, td.a)
        assert torch.equal(cloned_td.b, td.b)

        # Check that requires_grad is preserved
        assert cloned_td.a.requires_grad == td.a.requires_grad
        assert cloned_td.b.requires_grad == td.b.requires_grad

        # Check non-tensor fields are copied
        assert cloned_td.meta == td.meta

        # Modify original and check cloned is unchanged
        td.a = torch.zeros_like(td.a)
        assert not torch.equal(cloned_td.a, td.a)

    def test_clone_no_grad(self):
        """Test clone with tensors not requiring gradients."""
        td = CloneTestClass(
            a=torch.randn(2, 3, requires_grad=False),
            b=torch.ones(2, 3, requires_grad=False),
            shape=(2, 3),
            device=torch.device("cpu"),
        )
        cloned_td = td.clone()

        assert not cloned_td.a.requires_grad
        assert not cloned_td.b.requires_grad
        assert torch.equal(cloned_td.a, td.a)
        assert torch.equal(cloned_td.b, td.b)

    def test_clone_nested(self):
        """Test clone with nested TensorDataclass instances."""

        class NestedCloneTestClass(TensorDataclass):
            shape: tuple
            device: Optional[torch.device]
            c: torch.Tensor

        td_nested = NestedCloneTestClass(
            c=torch.randn(2, 3, requires_grad=True),
            shape=(2, 3),
            device=torch.device("cpu"),
        )

        class CloneWithNested(TensorDataclass):
            a: torch.Tensor
            b: NestedCloneTestClass
            meta: int = 42

        td = CloneWithNested(
            a=torch.randn(2, 3, requires_grad=True),
            b=td_nested,
            shape=(2, 3),
            device=torch.device("cpu"),
        )
        cloned_td = td.clone()

        assert cloned_td is not td
        assert cloned_td.a is not td.a
        assert cloned_td.b is not td.b
        assert cloned_td.b.c is not td.b.c

        assert torch.equal(cloned_td.a, td.a)
        assert torch.equal(cloned_td.b.c, td.b.c)

        assert cloned_td.a.requires_grad == td.a.requires_grad
        assert cloned_td.b.c.requires_grad == td.b.c.requires_grad

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

        assert cloned_td.a.device.type == "cuda"
        assert cloned_td.b.device.type == "cuda"
        assert cloned_td.device.type == "cuda"
        assert torch.equal(cloned_td.a, td.a)
        assert torch.equal(cloned_td.b, td.b)
        assert cloned_td.a.requires_grad == td.a.requires_grad
        assert cloned_td.b.requires_grad == td.b.requires_grad

    def test_clone_compile(self):
        """Tests that a function using TensorDataclass.clone() can be torch.compiled."""
        from tests.tensor_dict.compile_utils import run_and_compare_compiled

        class MyData(TensorDataclass):
            shape: tuple
            device: Optional[torch.device]
            x: torch.Tensor
            y: torch.Tensor

        def func(td: MyData) -> MyData:
            return td.clone()

        data = MyData(
            x=torch.ones(3, 4),
            y=torch.zeros(3, 4),
            shape=(3, 4),
            device=torch.device("cpu"),
        )
        run_and_compare_compiled(func, data)


def test_clone_mutable_metadata():
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
    assert cloned_td.metadata == [1, 2]
    assert cloned_td.metadata is not td.metadata

    cloned_td.metadata.append(3)
    assert td.metadata == [1, 2]
    assert cloned_td.metadata == [1, 2, 3]
