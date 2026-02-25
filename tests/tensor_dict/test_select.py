import pytest
import torch

from tensorcontainer import TensorDict
from tests.compile_utils import run_and_compare_compiled
from tests.conftest import skipif_no_compile


class TestSelect:
    def test_single_key(self):
        td = TensorDict(
            {
                "obs": torch.randn(4, 10),
                "act": torch.randn(4, 3),
                "rew": torch.randn(4),
            },
            shape=(4,),
        )
        result = td.select("obs")
        assert list(result.keys()) == ["obs"]
        assert torch.equal(result["obs"], td["obs"])

    def test_multiple_keys(self):
        td = TensorDict(
            {
                "obs": torch.randn(4, 10),
                "act": torch.randn(4, 3),
                "rew": torch.randn(4),
            },
            shape=(4,),
        )
        result = td.select("obs", "act")
        assert set(result.keys()) == {"obs", "act"}
        assert torch.equal(result["obs"], td["obs"])
        assert torch.equal(result["act"], td["act"])

    def test_preserves_order(self):
        td = TensorDict(
            {"a": torch.randn(4), "b": torch.randn(4), "c": torch.randn(4)},
            shape=(4,),
        )
        result = td.select("c", "a")
        assert list(result.keys()) == ["c", "a"]

    def test_nested_tensordict_selected_as_whole(self):
        nested = TensorDict(
            {"x": torch.randn(4, 2), "y": torch.randn(4, 3)}, shape=(4,)
        )
        td = TensorDict({"nested": nested, "scalar": torch.randn(4)}, shape=(4,))
        result = td.select("nested")
        assert list(result.keys()) == ["nested"]
        assert isinstance(result["nested"], TensorDict)
        assert set(result["nested"].keys()) == {"x", "y"}

    def test_missing_key_raises(self):
        td = TensorDict({"obs": torch.randn(4)}, shape=(4,))
        with pytest.raises(KeyError, match="Keys not found"):
            td.select("obs", "missing")

    def test_preserves_shape(self):
        td = TensorDict({"x": torch.randn(4, 5, 6)}, shape=(4, 5))
        result = td.select("x")
        assert result.shape == torch.Size([4, 5])

    def test_preserves_device(self):
        td = TensorDict({"x": torch.randn(4)}, shape=(4,), device="cpu")
        result = td.select("x")
        assert result.device == td.device

    def test_subclass_preserved(self):
        class MyTensorDict(TensorDict):
            pass

        td = MyTensorDict({"x": torch.randn(4), "y": torch.randn(4)}, shape=(4,))
        result = td.select("x")
        assert isinstance(result, MyTensorDict)


@skipif_no_compile
class TestSelectCompile:
    def test_select_compiled(self):
        td = TensorDict(
            {"obs": torch.randn(4, 10), "act": torch.randn(4, 3)},
            shape=(4,),
        )

        def select_fn(t):
            return t.select("obs")

        eager_result, compiled_result = run_and_compare_compiled(select_fn, td)
        assert list(compiled_result.keys()) == ["obs"]
        assert compiled_result.shape == td.shape
