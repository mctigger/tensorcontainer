import pytest
import torch

from tensorcontainer import TensorDict
from tests.compile_utils import run_and_compare_compiled
from tests.conftest import skipif_no_compile


class TestExclude:
    def test_single_key(self):
        td = TensorDict(
            {
                "obs": torch.randn(4, 10),
                "act": torch.randn(4, 3),
                "rew": torch.randn(4),
            },
            shape=(4,),
        )
        result = td.exclude("rew")
        assert set(result.keys()) == {"obs", "act"}
        assert torch.equal(result["obs"], td["obs"])
        assert torch.equal(result["act"], td["act"])

    def test_multiple_keys(self):
        td = TensorDict(
            {
                "obs": torch.randn(4, 10),
                "act": torch.randn(4, 3),
                "rew": torch.randn(4),
            },
            shape=(4,),
        )
        result = td.exclude("act", "rew")
        assert list(result.keys()) == ["obs"]
        assert torch.equal(result["obs"], td["obs"])

    def test_exclude_all_but_one(self):
        td = TensorDict(
            {"a": torch.randn(4), "b": torch.randn(4), "c": torch.randn(4)},
            shape=(4,),
        )
        result = td.exclude("a", "b")
        assert list(result.keys()) == ["c"]

    def test_nested_tensordict_excluded(self):
        nested = TensorDict(
            {"x": torch.randn(4, 2), "y": torch.randn(4, 3)}, shape=(4,)
        )
        td = TensorDict({"nested": nested, "scalar": torch.randn(4)}, shape=(4,))
        result = td.exclude("nested")
        assert list(result.keys()) == ["scalar"]

    def test_missing_key_raises(self):
        td = TensorDict({"obs": torch.randn(4)}, shape=(4,))
        with pytest.raises(KeyError, match="Keys not found"):
            td.exclude("missing")

    def test_preserves_shape(self):
        td = TensorDict(
            {"x": torch.randn(4, 5, 6), "y": torch.randn(4, 5)}, shape=(4, 5)
        )
        result = td.exclude("y")
        assert result.shape == torch.Size([4, 5])

    def test_preserves_device(self):
        td = TensorDict(
            {"x": torch.randn(4), "y": torch.randn(4)}, shape=(4,), device="cpu"
        )
        result = td.exclude("y")
        assert result.device == td.device

    def test_subclass_preserved(self):
        class MyTensorDict(TensorDict):
            pass

        td = MyTensorDict({"x": torch.randn(4), "y": torch.randn(4)}, shape=(4,))
        result = td.exclude("y")
        assert isinstance(result, MyTensorDict)


@skipif_no_compile
class TestExcludeCompile:
    def test_exclude_compiled(self):
        td = TensorDict(
            {
                "obs": torch.randn(4, 10),
                "act": torch.randn(4, 3),
                "rew": torch.randn(4),
            },
            shape=(4,),
        )

        def exclude_fn(t):
            return t.exclude("rew")

        eager_result, compiled_result = run_and_compare_compiled(exclude_fn, td)
        assert set(compiled_result.keys()) == {"obs", "act"}
        assert compiled_result.shape == td.shape
