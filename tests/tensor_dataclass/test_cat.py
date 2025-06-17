import pytest
import torch
import dataclasses
from rtd.tensor_dataclass import TensorDataclass
from typing import Optional


@dataclasses.dataclass
class CatTestClass(TensorDataclass):
    shape: tuple
    device: Optional[torch.device]
    a: torch.Tensor
    b: torch.Tensor
    meta: int = 42


class TestCat:
    def test_basic_cat(self):
        """Test basic torch.cat operation on a list of TensorDataclass instances."""
        td1 = CatTestClass(
            a=torch.randn(2, 3),
            b=torch.ones(2, 3),
            shape=(2, 3),
            device=torch.device("cpu"),
        )
        td2 = CatTestClass(
            a=torch.randn(2, 3),
            b=torch.ones(2, 3),
            shape=(2, 3),
            device=torch.device("cpu"),
        )

        cat_td = torch.cat([td1, td2], dim=0)

        assert cat_td.shape == (4, 3)
        assert cat_td.a.shape == (4, 3)
        assert cat_td.b.shape == (4, 3)
        assert cat_td.meta == 42  # Non-tensor fields should be preserved (first one)

        assert torch.equal(cat_td.a[:2], td1.a)
        assert torch.equal(cat_td.a[2:], td2.a)
        assert torch.equal(cat_td.b[:2], td1.b)
        assert torch.equal(cat_td.b[2:], td2.b)

    def test_cat_different_dim(self):
        """Test torch.cat with a different dimension."""
        td1 = CatTestClass(
            a=torch.randn(2, 3),
            b=torch.ones(2, 3),
            shape=(2, 3),
            device=torch.device("cpu"),
        )
        td2 = CatTestClass(
            a=torch.randn(2, 3),
            b=torch.ones(2, 3),
            shape=(2, 3),
            device=torch.device("cpu"),
        )

        cat_td = torch.cat([td1, td2], dim=1)

        assert cat_td.shape == (2, 6)
        assert cat_td.a.shape == (2, 6)
        assert cat_td.b.shape == (2, 6)

        assert torch.equal(cat_td.a[:, :3], td1.a)
        assert torch.equal(cat_td.a[:, 3:], td2.a)

    def test_cat_inconsistent_shapes_raises(self):
        """Test that concatenating with inconsistent shapes raises an error."""
        td1 = CatTestClass(
            a=torch.randn(2, 3),
            b=torch.ones(2, 3),
            shape=(2, 3),
            device=torch.device("cpu"),
        )
        td2 = CatTestClass(
            a=torch.randn(2, 4),  # Inconsistent shape
            b=torch.ones(2, 4),
            shape=(2, 4),
            device=torch.device("cpu"),
        )

        with pytest.raises(ValueError):
            torch.cat([td1, td2], dim=0)

    def test_cat_inconsistent_meta_data_raises(self):
        """Test that concatenating with inconsistent meta data raises an error."""
        td1 = CatTestClass(
            a=torch.randn(2, 3),
            b=torch.ones(2, 3),
            shape=(2, 3),
            device=torch.device("cpu"),
            meta=42,
        )
        td2 = CatTestClass(
            a=torch.randn(2, 3),
            b=torch.ones(2, 3),
            shape=(2, 3),
            device=torch.device("cpu"),
            meta=99,  # Inconsistent meta data
        )

        with pytest.raises(ValueError):
            torch.cat([td1, td2], dim=0)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cat_on_cuda(self):
        """Test torch.cat on CUDA devices."""
        td1 = CatTestClass(
            a=torch.randn(2, 3, device="cuda"),
            b=torch.ones(2, 3, device="cuda"),
            shape=(2, 3),
            device=torch.device("cuda"),
        )
        td2 = CatTestClass(
            a=torch.randn(2, 3, device="cuda"),
            b=torch.ones(2, 3, device="cuda"),
            shape=(2, 3),
            device=torch.device("cuda"),
        )

        cat_td = torch.cat([td1, td2], dim=0)

        assert cat_td.device.type == "cuda"
        assert cat_td.a.device.type == "cuda"
        assert cat_td.b.device.type == "cuda"
        assert cat_td.shape == (4, 3)

    def test_cat_compile(self):
        """Tests that a function using torch.cat with TensorDataclass can be torch.compiled."""
        from tests.tensor_dict.compile_utils import run_and_compare_compiled

        @dataclasses.dataclass(eq=False)
        class MyData(TensorDataclass):
            shape: tuple
            device: Optional[torch.device]
            x: torch.Tensor
            y: torch.Tensor

        def func(tds: list[MyData]) -> MyData:
            return torch.cat(tds, dim=0)

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
