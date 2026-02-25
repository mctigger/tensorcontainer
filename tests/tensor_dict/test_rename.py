import pytest
import torch

from tensorcontainer import TensorDict
from tests.compile_utils import run_and_compare_compiled
from tests.conftest import skipif_no_compile


class TestRename:
    def test_single_key(self):
        td = TensorDict({"obs": torch.randn(4, 10), "act": torch.randn(4, 3)}, shape=(4,))
        result = td.rename({"obs": "observation"})
        assert set(result.keys()) == {"observation", "act"}
        assert torch.equal(result["observation"], td["obs"])
        assert torch.equal(result["act"], td["act"])

    def test_multiple_keys(self):
        td = TensorDict({"obs": torch.randn(4, 10), "act": torch.randn(4, 3)}, shape=(4,))
        result = td.rename({"obs": "observation", "act": "action"})
        assert set(result.keys()) == {"observation", "action"}
        assert torch.equal(result["observation"], td["obs"])
        assert torch.equal(result["action"], td["act"])

    def test_partial_rename(self):
        td = TensorDict(
            {"a": torch.randn(4), "b": torch.randn(4), "c": torch.randn(4)},
            shape=(4,),
        )
        result = td.rename({"a": "alpha"})
        assert set(result.keys()) == {"alpha", "b", "c"}
        assert torch.equal(result["alpha"], td["a"])
        assert torch.equal(result["b"], td["b"])
        assert torch.equal(result["c"], td["c"])

    def test_missing_key_raises(self):
        td = TensorDict({"obs": torch.randn(4)}, shape=(4,))
        with pytest.raises(KeyError, match="Keys not found"):
            td.rename({"missing": "new_name"})

    def test_duplicate_key_raises(self):
        td = TensorDict({"a": torch.randn(4), "b": torch.randn(4)}, shape=(4,))
        with pytest.raises(ValueError, match="duplicate key"):
            td.rename({"a": "b"})

    def test_swap_keys_works(self):
        td = TensorDict({"a": torch.randn(4), "b": torch.randn(4)}, shape=(4,))
        result = td.rename({"a": "b", "b": "a"})
        assert set(result.keys()) == {"a", "b"}
        assert torch.equal(result["a"], td["b"])
        assert torch.equal(result["b"], td["a"])

    def test_nested_tensordict_renamed(self):
        nested = TensorDict({"x": torch.randn(4, 2)}, shape=(4,))
        td = TensorDict({"nested": nested, "scalar": torch.randn(4)}, shape=(4,))
        result = td.rename({"nested": "inner"})
        assert set(result.keys()) == {"inner", "scalar"}
        assert isinstance(result["inner"], TensorDict)
        assert list(result["inner"].keys()) == ["x"]

    def test_preserves_shape(self):
        td = TensorDict({"x": torch.randn(4, 5, 6)}, shape=(4, 5))
        result = td.rename({"x": "y"})
        assert result.shape == torch.Size([4, 5])

    def test_preserves_device(self):
        td = TensorDict({"x": torch.randn(4)}, shape=(4,), device="cpu")
        result = td.rename({"x": "y"})
        assert result.device == td.device

    def test_empty_mapping(self):
        td = TensorDict({"x": torch.randn(4), "y": torch.randn(4)}, shape=(4,))
        result = td.rename({})
        assert set(result.keys()) == {"x", "y"}

    def test_subclass_preserved(self):
        class MyTensorDict(TensorDict):
            pass

        td = MyTensorDict({"x": torch.randn(4)}, shape=(4,))
        result = td.rename({"x": "y"})
        assert isinstance(result, MyTensorDict)


@skipif_no_compile
class TestRenameCompile:
    def test_rename_compiled(self):
        td = TensorDict(
            {"obs": torch.randn(4, 10), "act": torch.randn(4, 3)},
            shape=(4,),
        )

        def rename_fn(t):
            return t.rename({"obs": "observation"})

        eager_result, compiled_result = run_and_compare_compiled(rename_fn, td)
        assert set(compiled_result.keys()) == {"observation", "act"}
        assert compiled_result.shape == td.shape
