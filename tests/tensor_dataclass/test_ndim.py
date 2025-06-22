import pytest
import torch

from tests.conftest import skipif_no_compile
from tests.tensor_dataclass.conftest import SimpleTensorData


@skipif_no_compile
class TestNdim:
    """Test class for ndim property of TensorDataclass."""

    @pytest.mark.parametrize("compile_mode", [False, True])
    def test_ndim_basic(self, simple_tensor_data_instance, compile_mode):
        """Test ndim property for a TensorDataclass with multiple tensor fields."""
        td = simple_tensor_data_instance

        def get_ndim():
            return td.ndim

        if compile_mode:
            get_ndim = torch.compile(get_ndim)

        result = get_ndim()
        assert result == 2  # ndim should be the length of the shape tuple (3, 4)
        assert td.a.ndim == 2  # Each tensor field should have the same ndim
        assert td.b.ndim == 2

    @pytest.mark.parametrize("compile_mode", [False, True])
    def test_ndim_scalar(self, compile_mode):
        """Test ndim property for a TensorDataclass with a scalar tensor field."""
        td = SimpleTensorData(
            a=torch.tensor(1.0, device=torch.device("cpu")),
            b=torch.tensor(2.0, device=torch.device("cpu")),
            shape=(),
            device=torch.device("cpu"),
        )

        def get_ndim():
            return td.ndim

        if compile_mode:
            get_ndim = torch.compile(get_ndim)

        result = get_ndim()
        assert result == 0  # ndim should be 0 for a scalar shape
        assert td.a.ndim == 0  # Each tensor field should be scalar
        assert td.b.ndim == 0

    @pytest.mark.parametrize("compile_mode", [False, True])
    def test_ndim_empty_shape(self, compile_mode):
        """Test ndim property for a TensorDataclass with an empty shape."""
        td = SimpleTensorData(
            a=torch.tensor([], device=torch.device("cpu")),
            b=torch.tensor([], device=torch.device("cpu")),
            shape=(),
            device=torch.device("cpu"),
        )

        def get_ndim():
            return td.ndim

        if compile_mode:
            get_ndim = torch.compile(get_ndim)

        result = get_ndim()
        assert result == 0  # ndim should be 0 for an empty shape
        assert td.a.ndim == 1  # Empty tensors have ndim of 1
        assert td.b.ndim == 1

    @pytest.mark.parametrize("compile_mode", [False, True])
    def test_ndim_different_shapes(self, compile_mode):
        """Test ndim property when tensor fields have different shapes."""
        td = SimpleTensorData(
            a=torch.randn(2, 3, 4, device=torch.device("cpu")),
            b=torch.ones(2, 3, device=torch.device("cpu")),
            shape=(2, 3),
            device=torch.device("cpu"),
        )

        def get_ndim():
            return td.ndim

        if compile_mode:
            get_ndim = torch.compile(get_ndim)

        result = get_ndim()
        assert result == 2  # ndim should be the length of the shape tuple
        assert td.a.ndim == 3  # a has 3 dimensions
        assert td.b.ndim == 2  # b has 2 dimensions
