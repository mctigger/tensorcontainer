from typing import Optional

import pytest

import torch
from torch._dynamo import exc

from tests.conftest import skipif_no_compile

from rtd.tensor_dataclass import TensorDataclass


class A(TensorDataclass):
    shape: tuple
    device: Optional[torch.device]
    a: torch.Tensor
    b: torch.Tensor


# Parametrize test cases for both eager and compile modes
class TestViewAndReshape:
    def test_view_eager(self):
        """Test the view method of TensorDataclass."""
        td = A(
            a=torch.randn(4, 5),
            b=torch.ones(4, 5),
            shape=(4, 5),
            device=torch.device("cpu"),
        )

        def view_fn(td):
            return td.view(20)

        result = view_fn(td)

        assert result.shape == (20,)
        assert result.a.shape == (20,)
        assert result.b.shape == (20,)
        assert torch.equal(result.a, td.a.view(20))
        assert torch.equal(result.b, td.b.view(20))

    @skipif_no_compile
    def test_view_compile(self):
        """Test the view method of TensorDataclass."""
        td = A(
            a=torch.randn(4, 5),
            b=torch.ones(4, 5),
            shape=(4, 5),
            device=torch.device("cpu"),
        )

        def view_fn(td):
            return td.view(20)

        compiled_view = torch.compile(view_fn, fullgraph=True)
        result = compiled_view(td)

        assert result.shape == (20,)
        assert result.a.shape == (20,)
        assert result.b.shape == (20,)
        assert torch.equal(result.a, td.a.view(20))
        assert torch.equal(result.b, td.b.view(20))

    def test_reshape_eager(self):
        """Test the reshape method of TensorDataclass."""
        td = A(
            a=torch.randn(2, 6),
            b=torch.ones(2, 6),
            shape=(2, 6),
            device=torch.device("cpu"),
        )

        def reshape_fn(td):
            return td.reshape(4, 3)

        result = reshape_fn(td)

        assert result.shape == (4, 3)
        assert result.a.shape == (4, 3)
        assert result.b.shape == (4, 3)
        assert torch.equal(result.a, td.a.reshape(4, 3))
        assert torch.equal(result.b, td.b.reshape(4, 3))

    @skipif_no_compile
    def test_reshape_compile(self):
        """Test the reshape method of TensorDataclass."""
        td = A(
            a=torch.randn(2, 6),
            b=torch.ones(2, 6),
            shape=(2, 6),
            device=torch.device("cpu"),
        )

        def reshape_fn(td):
            return td.reshape(4, 3)

        compiled_reshape = torch.compile(reshape_fn, fullgraph=True)
        result = compiled_reshape(td)

        assert result.shape == (4, 3)
        assert result.a.shape == (4, 3)
        assert result.b.shape == (4, 3)
        assert torch.equal(result.a, td.a.reshape(4, 3))
        assert torch.equal(result.b, td.b.reshape(4, 3))

    def test_invalid_view_raises_eager(self):
        """Test that invalid view operations raise RuntimeError."""
        td = A(
            a=torch.randn(4, 5),
            b=torch.ones(4, 5),
            shape=(4, 5),
            device=torch.device("cpu"),
        )

        def invalid_view_fn(td):
            return td.view(21)  # Invalid size

        with pytest.raises(RuntimeError):
            invalid_view_fn(td)

    @skipif_no_compile
    def test_invalid_view_raises_compile(self):
        """Test that invalid view operations raise RuntimeError."""
        td = A(
            a=torch.randn(4, 5),
            b=torch.ones(4, 5),
            shape=(4, 5),
            device=torch.device("cpu"),
        )

        def invalid_view_fn(td):
            return td.view(21)  # Invalid size

        compiled_invalid_view = torch.compile(invalid_view_fn, fullgraph=True)
        with pytest.raises(RuntimeError):
            compiled_invalid_view(td)

    def test_invalid_reshape_raises_eager(self):
        """Test that invalid reshape operations raise RuntimeError."""
        td = A(
            a=torch.randn(4, 5),
            b=torch.ones(4, 5),
            shape=(4, 5),
            device=torch.device("cpu"),
        )

        def invalid_reshape_fn(td):
            return td.reshape(3, 7)  # Invalid size

        with pytest.raises(RuntimeError):
            invalid_reshape_fn(td)

    @skipif_no_compile
    def test_invalid_reshape_raises_compile(self):
        """Test that invalid reshape operations raise RuntimeError."""
        td = A(
            a=torch.randn(4, 5),
            b=torch.ones(4, 5),
            shape=(4, 5),
            device=torch.device("cpu"),
        )

        def invalid_reshape_fn(td):
            return td.reshape(3, 7)  # Invalid size

        compiled_invalid_reshape = torch.compile(invalid_reshape_fn, fullgraph=True)
        with pytest.raises(RuntimeError):
            compiled_invalid_reshape(td)

    def test_view_reshape_eager(self):
        """Test that view and reshape operations can be compiled."""
        td = A(
            a=torch.randn(4, 5),
            b=torch.ones(4, 5),
            shape=(4, 5),
            device=torch.device("cpu"),
        )

        def view_reshape_fn(td):
            viewed = td.view(20)
            reshaped = viewed.reshape(4, 5)
            return reshaped

        result = view_reshape_fn(td)

        assert result.shape == (4, 5)
        assert result.a.shape == (4, 5)
        assert result.b.shape == (4, 5)
        assert torch.equal(result.a, td.a)
        assert torch.equal(result.b, td.b)

    @skipif_no_compile
    def test_view_reshape_compile(self):
        """Test that view and reshape operations can be compiled."""
        td = A(
            a=torch.randn(4, 5),
            b=torch.ones(4, 5),
            shape=(4, 5),
            device=torch.device("cpu"),
        )

        def view_reshape_fn(td):
            viewed = td.view(20)
            reshaped = viewed.reshape(4, 5)
            return reshaped

        compiled_view_reshape = torch.compile(view_reshape_fn, fullgraph=True)
        result = compiled_view_reshape(td)

        assert result.shape == (4, 5)
        assert result.a.shape == (4, 5)
        assert result.b.shape == (4, 5)
        assert torch.equal(result.a, td.a)
        assert torch.equal(result.b, td.b)

    def test_non_contiguous_view_raises_eager(self):
        """Test that view() on non-contiguous tensor raises a RuntimeError."""
        # Create a dataclass with a tensor that can be transposed
        td_orig = A(
            shape=(5, 4),
            device=torch.device("cpu"),
            a=torch.randn(5, 4),
            b=torch.randn(5, 4),
        )

        # Create a non-contiguous version by transposing
        td_non_contiguous = td_orig.transpose(0, 1)
        assert not td_non_contiguous.a.is_contiguous()

        def view_fn(td):
            return td.view(20)

        with pytest.raises(RuntimeError, match="view size is not compatible"):
            view_fn(td_non_contiguous)

    @skipif_no_compile
    def test_non_contiguous_view_raises_compile(self):
        """Test that view() on non-contiguous tensor raises a RuntimeError."""
        # Create a dataclass with a tensor that can be transposed
        td_orig = A(
            shape=(5, 4),
            device=torch.device("cpu"),
            a=torch.randn(5, 4),
            b=torch.randn(5, 4),
        )

        # Create a non-contiguous version by transposing
        td_non_contiguous = td_orig.transpose(0, 1)
        assert not td_non_contiguous.a.is_contiguous()

        def view_fn(td):
            return td.view(20)

        compiled_view = torch.compile(view_fn, fullgraph=True)
        with pytest.raises(exc.TorchRuntimeError):
            compiled_view(td_non_contiguous)
