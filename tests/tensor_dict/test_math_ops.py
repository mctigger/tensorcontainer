import pytest
import torch

from rtd.tensor_dict import TensorDict


from .common import compare_nested_dict


@pytest.mark.skipif_no_compile
class TestTensorDictMathOps:
    @pytest.mark.parametrize("compile_op", [True, False])
    def test_abs(self, nested_dict, compile_op):
        shape = (2, 2)
        data = nested_dict(shape)

        # Negate some values to test abs
        data["x"]["a"] = -data["x"]["a"]
        data["y"] = -data["y"]

        td = TensorDict(data, shape, device="cpu")

        def op(t):
            return t.abs()

        if compile_op:
            op = torch.compile(op, fullgraph=True)

        result_td = op(td)

        assert isinstance(result_td, TensorDict)
        assert result_td is not td
        assert result_td.shape == td.shape
        assert str(result_td.device) == td.device

        compare_nested_dict(td.data, result_td.data, torch.abs)

    @pytest.mark.parametrize("compile_op", [True, False])
    def test_add(self, nested_dict, compile_op):
        shape = (2, 2)
        data = nested_dict(shape)
        td = TensorDict(data, shape, device="cpu")

        def op(t):
            return t.add(2)

        if compile_op:
            op = torch.compile(op, fullgraph=True)

        result_td = op(td)

        assert isinstance(result_td, TensorDict)
        assert result_td is not td
        assert result_td.shape == td.shape
        assert str(result_td.device) == td.device

        compare_nested_dict(td.data, result_td.data, lambda x: x + 2)

    @pytest.mark.parametrize("compile_op", [True, False])
    def test_sub(self, nested_dict, compile_op):
        shape = (2, 2)
        data = nested_dict(shape)
        td = TensorDict(data, shape, device="cpu")

        def op(t):
            return t.sub(2)

        if compile_op:
            op = torch.compile(op, fullgraph=True)

        result_td = op(td)

        assert isinstance(result_td, TensorDict)
        assert result_td is not td
        assert result_td.shape == td.shape
        assert str(result_td.device) == td.device

        compare_nested_dict(td.data, result_td.data, lambda x: x - 2)

    @pytest.mark.parametrize("compile_op", [True, False])
    def test_mul(self, nested_dict, compile_op):
        shape = (2, 2)
        data = nested_dict(shape)
        td = TensorDict(data, shape, device="cpu")

        def op(t):
            return t.mul(2)

        if compile_op:
            op = torch.compile(op, fullgraph=True)

        result_td = op(td)

        assert isinstance(result_td, TensorDict)
        assert result_td is not td
        assert result_td.shape == td.shape
        assert str(result_td.device) == td.device

        compare_nested_dict(td.data, result_td.data, lambda x: x * 2)

    @pytest.mark.parametrize("compile_op", [True, False])
    def test_div(self, nested_dict, compile_op):
        shape = (2, 2)
        data = nested_dict(shape)
        td = TensorDict(data, shape, device="cpu")

        def op(t):
            return t.div(2)

        if compile_op:
            op = torch.compile(op, fullgraph=True)

        result_td = op(td)

        assert isinstance(result_td, TensorDict)
        assert result_td is not td
        assert result_td.shape == td.shape
        assert str(result_td.device) == td.device

        compare_nested_dict(td.data, result_td.data, lambda x: x / 2)

    @pytest.mark.parametrize("compile_op", [True, False])
    def test_pow(self, nested_dict, compile_op):
        shape = (2, 2)
        data = nested_dict(shape)
        td = TensorDict(data, shape, device="cpu")

        def op(t):
            return t.pow(2)

        if compile_op:
            op = torch.compile(op, fullgraph=True)

        result_td = op(td)

        assert isinstance(result_td, TensorDict)
        assert result_td is not td
        assert result_td.shape == td.shape
        assert str(result_td.device) == td.device

        compare_nested_dict(td.data, result_td.data, lambda x: x**2)

    @pytest.mark.parametrize("compile_op", [True, False])
    def test_sqrt(self, nested_dict, compile_op):
        shape = (2, 2)
        data = nested_dict(shape)
        td = TensorDict(data, shape, device="cpu")

        def op(t):
            return t.sqrt()

        if compile_op:
            op = torch.compile(op, fullgraph=True)

        result_td = op(td)

        assert isinstance(result_td, TensorDict)
        assert result_td is not td
        assert result_td.shape == td.shape
        assert str(result_td.device) == td.device

        compare_nested_dict(td.data, result_td.data, torch.sqrt)

    @pytest.mark.parametrize("compile_op", [True, False])
    def test_log(self, nested_dict, compile_op):
        shape = (2, 2)
        data = nested_dict(shape)
        # Ensure values are positive for log
        for k, v in data.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    data[k][sub_k] = sub_v + 1
            else:
                data[k] = v + 1
        td = TensorDict(data, shape, device="cpu")

        def op(t):
            return t.log()

        if compile_op:
            op = torch.compile(op, fullgraph=True)

        result_td = op(td)

        assert isinstance(result_td, TensorDict)
        assert result_td is not td
        assert result_td.shape == td.shape
        assert str(result_td.device) == td.device

        compare_nested_dict(td.data, result_td.data, torch.log)

    @pytest.mark.parametrize("compile_op", [True, False])
    def test_neg(self, nested_dict, compile_op):
        shape = (2, 2)
        data = nested_dict(shape)
        td = TensorDict(data, shape, device="cpu")

        def op(t):
            return t.neg()

        if compile_op:
            op = torch.compile(op, fullgraph=True)

        result_td = op(td)

        assert isinstance(result_td, TensorDict)
        assert result_td is not td
        assert result_td.shape == td.shape
        assert str(result_td.device) == td.device

        compare_nested_dict(td.data, result_td.data, lambda x: -x)

    @pytest.mark.parametrize("compile_op", [True, False])
    def test_clamp(self, nested_dict, compile_op):
        shape = (2, 2)
        data = nested_dict(shape)
        td = TensorDict(data, shape, device="cpu")

        def op(t):
            return t.clamp(min=5, max=10)

        if compile_op:
            op = torch.compile(op, fullgraph=True)

        result_td = op(td)

        assert isinstance(result_td, TensorDict)
        assert result_td is not td
        assert result_td.shape == td.shape
        assert str(result_td.device) == td.device

        compare_nested_dict(
            td.data, result_td.data, lambda x: torch.clamp(x, min=5, max=10)
        )
