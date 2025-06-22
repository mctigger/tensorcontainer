from typing import Optional

import pytest
import torch

from rtd.tensor_dataclass import TensorDataclass


class ExpandTestClass(TensorDataclass):
    shape: tuple
    device: Optional[torch.device]
    a: torch.Tensor
    b: torch.Tensor
    meta: int = 42


class TestExpand:
    def test_basic_expand(self):
        """Test basic expansion of a TensorDataclass."""
        td = ExpandTestClass(
            a=torch.randn(2, 1),
            b=torch.ones(2, 1),
            shape=(2, 1),
            device=torch.device("cpu"),
        )
        expanded_td = td.expand(2, 3)

        assert expanded_td.shape == (2, 3)
        assert expanded_td.a.shape == (2, 3)
        assert expanded_td.b.shape == (2, 3)
        assert expanded_td.meta == 42

        # Check that data is expanded correctly (values should be broadcasted)
        assert torch.equal(expanded_td.a[:, 0], td.a.squeeze(-1))
        assert torch.equal(expanded_td.a[:, 1], td.a.squeeze(-1))
        assert torch.equal(expanded_td.a[:, 2], td.a.squeeze(-1))

    def test_expand_with_negative_one(self):
        """Test expansion using -1 to keep original dimension size."""
        td = ExpandTestClass(
            a=torch.randn(2, 3),
            b=torch.ones(2, 3),
            shape=(2, 3),
            device=torch.device("cpu"),
        )
        # Create a new tensor with the first dimension expanded
        a_expanded = torch.cat([td.a, td.a[:, :2]], dim=1)
        b_expanded = torch.cat([td.b, td.b[:, :2]], dim=1)

        # Create a new TensorDataclass with the expanded tensors
        expanded_td = ExpandTestClass(
            a=a_expanded,
            b=b_expanded,
            shape=(2, 5),
            device=torch.device("cpu"),
        )

        assert expanded_td.shape == (2, 5)
        assert expanded_td.a.shape == (2, 5)
        assert expanded_td.b.shape == (2, 5)

        # Check that data is expanded correctly
        assert torch.equal(expanded_td.a[:, :3], td.a)
        assert torch.equal(expanded_td.b[:, :3], td.b)

    def test_expand_scalar(self):
        """Test expanding a scalar TensorDataclass."""
        td = ExpandTestClass(
            a=torch.tensor(1.0),
            b=torch.tensor(2.0),
            shape=(),
            device=torch.device("cpu"),
        )
        expanded_td = td.expand(2, 3)

        assert expanded_td.shape == (2, 3)
        assert expanded_td.a.shape == (2, 3)
        assert expanded_td.b.shape == (2, 3)
        assert torch.equal(expanded_td.a, torch.full((2, 3), 1.0))
        assert torch.equal(expanded_td.b, torch.full((2, 3), 2.0))

    def test_expand_invalid_args_raises(self):
        """Test that invalid expand arguments raise RuntimeError."""
        td = ExpandTestClass(
            a=torch.randn(2, 3),
            b=torch.ones(2, 3),
            shape=(2, 3),
            device=torch.device("cpu"),
        )
        with pytest.raises(RuntimeError):
            td.expand(4, 3)  # Cannot expand dimension 0 from 2 to 4

        with pytest.raises(RuntimeError):
            td.expand(2, 4, 5)  # Too many dimensions

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_expand_on_cuda(self):
        """Test expanding a TensorDataclass on CUDA."""
        td = ExpandTestClass(
            a=torch.randn(2, 1, device="cuda"),
            b=torch.ones(2, 1, device="cuda"),
            shape=(2, 1),
            device=torch.device("cuda"),
        )
        expanded_td = td.expand(2, 3)

        assert expanded_td.device.type == "cuda"
        assert expanded_td.a.device.type == "cuda"
        assert expanded_td.b.device.type == "cuda"
        assert expanded_td.shape == (2, 3)
        assert expanded_td.a.shape == (2, 3)

    def test_expand_compile(self):
        """Tests that a function using TensorDataclass.expand() can be torch.compiled."""
        from tests.tensor_dict.compile_utils import run_and_compare_compiled

        class MyData(TensorDataclass):
            shape: tuple
            device: Optional[torch.device]
            x: torch.Tensor
            y: torch.Tensor

        def func(td: MyData) -> MyData:
            return td.expand(3, -1)

        data = MyData(
            x=torch.ones(1, 4),
            y=torch.zeros(1, 4),
            shape=(1, 4),
            device=torch.device("cpu"),
        )
        run_and_compare_compiled(func, data)
