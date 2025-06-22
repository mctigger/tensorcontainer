from typing import Optional

import pytest
import torch

from rtd.tensor_dataclass import TensorDataclass
from tests.tensor_dataclass.conftest import (
    StackTestClass,
)
from tests.conftest import skipif_no_compile


class TestStack:
    """Test suite for torch.stack operations on TensorDataclass instances."""

    def test_basic_stack(self, stack_test_instances):
        """Test basic torch.stack operation on a list of TensorDataclass instances."""
        td1, td2 = stack_test_instances

        stacked_td = torch.stack([td1, td2], dim=0)  # type: ignore

        expected_shape = (2,) + td1.shape  # Stack adds a new dimension at the front
        assert stacked_td.shape == expected_shape
        assert isinstance(stacked_td, StackTestClass)
        assert stacked_td.a.shape == expected_shape
        assert stacked_td.b.shape == expected_shape
        assert (
            stacked_td.meta == 42
        )  # Non-tensor fields should be preserved (first one)

        assert torch.equal(stacked_td.a[0], td1.a)
        assert torch.equal(stacked_td.a[1], td2.a)
        assert torch.equal(stacked_td.b[0], td1.b)
        assert torch.equal(stacked_td.b[1], td2.b)

    def test_stack_different_dim(self, stack_test_instances):
        """Test torch.stack with a different dimension."""
        td1, td2 = stack_test_instances

        stacked_td = torch.stack([td1, td2], dim=1)  # type: ignore

        assert isinstance(stacked_td, StackTestClass)
        assert stacked_td.shape == (2, 2, 3)
        assert stacked_td.a.shape == (2, 2, 3)
        assert stacked_td.b.shape == (2, 2, 3)

        assert torch.equal(stacked_td.a[:, 0], td1.a)
        assert torch.equal(stacked_td.a[:, 1], td2.a)

    def test_stack_inconsistent_shapes_raises(self):
        """Test that stacking with inconsistent shapes raises an error."""
        td1 = StackTestClass(
            a=torch.randn(2, 3),
            b=torch.ones(2, 3),
            shape=(2, 3),
            device=torch.device("cpu"),
        )
        td2 = StackTestClass(
            a=torch.randn(2, 4),  # Inconsistent shape
            b=torch.ones(2, 4),
            shape=(2, 4),
            device=torch.device("cpu"),
        )

        with pytest.raises(ValueError):
            torch.stack([td1, td2], dim=0)  # type: ignore

    def test_stack_inconsistent_meta_data_raises(self):
        """Test that stacking with inconsistent meta data raises an error."""
        td1 = StackTestClass(
            a=torch.randn(2, 3),
            b=torch.ones(2, 3),
            shape=(2, 3),
            device=torch.device("cpu"),
            meta=42,
        )
        td2 = StackTestClass(
            a=torch.randn(2, 3),
            b=torch.ones(2, 3),
            shape=(2, 3),
            device=torch.device("cpu"),
            meta=99,  # Inconsistent meta data
        )

        # The current pytree implementation for TensorDataclass will take the meta_data
        # from the first element. If this behavior is desired, this test should be removed
        # or modified to assert that the first meta_data is kept.
        # For now, we assume inconsistent meta data should raise an error if it's not handled
        # by the pytree unflattening in a way that preserves consistency or raises.
        # Given the current _pytree_unflatten, it will just take the first meta_data.
        # So, this test should check if the meta data is taken from the first element.
        with pytest.raises(ValueError):
            torch.stack([td1, td2], dim=0)  # type: ignore

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_stack_on_cuda(self):
        """Test torch.stack on CUDA devices."""
        td1 = StackTestClass(
            a=torch.randn(2, 3, device="cuda"),
            b=torch.ones(2, 3, device="cuda"),
            shape=(2, 3),
            device=torch.device("cuda"),
        )
        td2 = StackTestClass(
            a=torch.randn(2, 3, device="cuda"),
            b=torch.ones(2, 3, device="cuda"),
            shape=(2, 3),
            device=torch.device("cuda"),
        )

        stacked_td = torch.stack([td1, td2], dim=0)  # type: ignore

        assert isinstance(stacked_td, StackTestClass)
        assert stacked_td.device.type == "cuda"
        assert stacked_td.a.device.type == "cuda"
        assert stacked_td.b.device.type == "cuda"
        assert stacked_td.shape == (2, 2, 3)

    @skipif_no_compile
    def test_stack_compile(self):
        """Tests that a function using torch.stack with TensorDataclass can be torch.compiled."""
        from tests.tensor_dict.compile_utils import run_and_compare_compiled

        class MyData(TensorDataclass):
            shape: tuple
            device: Optional[torch.device]
            x: torch.Tensor
            y: torch.Tensor

        def func(tds):
            return torch.stack(tds, dim=0)  # type: ignore

        data1 = MyData(
            x=torch.ones(3, 4),
            y=torch.zeros(3, 4),
            shape=(3, 4),
            device=torch.device("cpu"),
        )
        data2 = MyData(
            x=torch.ones(3, 4) * 2,
            y=torch.zeros(3, 4) * 2,
            shape=(3, 4),
            device=torch.device("cpu"),
        )
        run_and_compare_compiled(func, [data1, data2])


def test_stack_empty_list_raises():
    """Test that torch.stack on an empty list raises a RuntimeError."""
    with pytest.raises(RuntimeError, match="stack expects a non-empty TensorList"):
        torch.stack([], dim=0)


def test_stack_mixed_optional_raises():
    """Test that stacking with mixed None and Tensor for an optional field raises."""

    class OptionalStack(TensorDataclass):
        shape: tuple
        device: Optional[torch.device]
        a: torch.Tensor
        b: Optional[torch.Tensor] = None

    td1 = OptionalStack(
        shape=(3,),
        device=torch.device("cpu"),
        a=torch.randn(3),
        b=torch.ones(3),
    )
    td2 = OptionalStack(
        shape=(3,),
        device=torch.device("cpu"),
        a=torch.randn(3),
        b=None,  # b is None here
    )

    with pytest.raises(ValueError, match="Node arity mismatch"):
        torch.stack([td1, td2], dim=0)  # type: ignore
