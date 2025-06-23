from typing import Optional

import pytest
import torch

from rtd.tensor_dataclass import TensorDataClass
from tests.conftest import skipif_no_compile
from tests.tensor_dataclass.conftest import DeviceTestClass


class TestExpand:
    """Test class for TensorDataclass expand functionality."""

    @pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
    @pytest.mark.parametrize(
        "expand_shape,original_shape,expected_shape",
        [
            # Basic expansion
            ((2, 3), (2, 1), (2, 3)),
            # Scalar expansion
            ((2, 3), (), (2, 3)),
            # Multi-dimensional expansion
            ((3, 2, 4), (1, 2, 1), (3, 2, 4)),
            # Keep some dimensions unchanged with -1
            ((-1, 5), (2, 1), (2, 5)),
        ],
    )
    def test_expand_basic(
        self, execution_mode, expand_shape, original_shape, expected_shape
    ):
        """Test basic expansion of a TensorDataclass."""
        if execution_mode == "compiled":
            pytest.importorskip("torch", minversion="2.0")

        # Create test instance with the specified original shape
        if original_shape == ():
            # Scalar case
            td = DeviceTestClass(
                a=torch.tensor(1.0),
                b=torch.tensor(2.0),
                shape=original_shape,
                device=torch.device("cpu"),
            )
        else:
            td = DeviceTestClass(
                a=torch.randn(original_shape),
                b=torch.ones(original_shape),
                shape=original_shape,
                device=torch.device("cpu"),
            )

        def expand_func(td_input):
            return td_input.expand(*expand_shape)

        if execution_mode == "compiled":
            expand_func = torch.compile(expand_func)

        expanded_td = expand_func(td)

        assert expanded_td.shape == expected_shape
        assert expanded_td.a.shape == expected_shape
        assert expanded_td.b.shape == expected_shape
        assert expanded_td.meta == 42

        # Check that data is expanded correctly for non-scalar cases
        if original_shape != ():
            # For basic expansion, verify broadcasting worked correctly
            if expand_shape == (2, 3) and original_shape == (2, 1):
                assert torch.equal(expanded_td.a[:, 0], td.a.squeeze(-1))
                assert torch.equal(expanded_td.a[:, 1], td.a.squeeze(-1))
                assert torch.equal(expanded_td.a[:, 2], td.a.squeeze(-1))
        else:
            # For scalar expansion, check values are broadcasted
            if expand_shape == (2, 3):
                assert torch.equal(expanded_td.a, torch.full((2, 3), 1.0))
                assert torch.equal(expanded_td.b, torch.full((2, 3), 2.0))

    @pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
    @pytest.mark.parametrize(
        "invalid_expand_shape,original_shape",
        [
            # Cannot expand dimension 0 from 2 to 4
            ((4, 3), (2, 3)),
            # Too many dimensions
            ((2, 4, 5), (2, 3)),
        ],
    )
    def test_expand_invalid_args_raises(
        self, execution_mode, invalid_expand_shape, original_shape
    ):
        """Test that invalid expand arguments raise RuntimeError."""
        if execution_mode == "compiled":
            pytest.importorskip("torch", minversion="2.0")

        td = DeviceTestClass(
            a=torch.randn(original_shape),
            b=torch.ones(original_shape),
            shape=original_shape,
            device=torch.device("cpu"),
        )

        def expand_func(td_input):
            return td_input.expand(*invalid_expand_shape)

        if execution_mode == "compiled":
            expand_func = torch.compile(expand_func)

        with pytest.raises(RuntimeError):
            expand_func(td)

    @pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_expand_on_cuda(self, execution_mode):
        """Test expanding a TensorDataclass on CUDA."""
        if execution_mode == "compiled":
            pytest.importorskip("torch", minversion="2.0")

        td = DeviceTestClass(
            a=torch.randn(2, 1, device="cuda"),
            b=torch.ones(2, 1, device="cuda"),
            shape=(2, 1),
            device=torch.device("cuda"),
        )

        def expand_func(td_input):
            return td_input.expand(2, 3)

        if execution_mode == "compiled":
            expand_func = torch.compile(expand_func)

        expanded_td = expand_func(td)

        assert expanded_td.device is not None and expanded_td.device.type == "cuda"
        assert expanded_td.a.device.type == "cuda"
        assert expanded_td.b.device.type == "cuda"
        assert expanded_td.shape == (2, 3)
        assert expanded_td.a.shape == (2, 3)

    @skipif_no_compile
    def test_expand_compile_integration(self):
        """Tests that a function using TensorDataclass.expand() can be torch.compiled."""
        from tests.tensor_dict.compile_utils import run_and_compare_compiled

        class MyData(TensorDataClass):
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
