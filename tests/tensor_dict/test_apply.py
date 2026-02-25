import torch

from tensorcontainer import TensorDict
from tests.compile_utils import run_and_compare_compiled
from tests.conftest import skipif_no_compile


class TestApply:
    def test_simple_function(self):
        td = TensorDict(
            {"x": torch.ones(4, 3), "y": torch.ones(4, 2)},
            shape=(4,),
        )
        result = td.apply(lambda t: t * 2)
        assert torch.equal(result["x"], torch.ones(4, 3) * 2)
        assert torch.equal(result["y"], torch.ones(4, 2) * 2)

    def test_nested_structure(self):
        nested = TensorDict({"a": torch.ones(4, 2)}, shape=(4,))
        td = TensorDict({"nested": nested, "b": torch.ones(4, 3)}, shape=(4,))
        result = td.apply(lambda t: t * 3)
        assert torch.equal(result["b"], torch.ones(4, 3) * 3)
        assert torch.equal(result["nested"]["a"], torch.ones(4, 2) * 3)

    def test_deeply_nested(self):
        inner = TensorDict({"x": torch.ones(4)}, shape=(4,))
        middle = TensorDict({"inner": inner}, shape=(4,))
        td = TensorDict({"middle": middle, "top": torch.ones(4, 2)}, shape=(4,))
        result = td.apply(lambda t: t + 1)
        assert torch.equal(result["top"], torch.ones(4, 2) + 1)
        assert torch.equal(result["middle"]["inner"]["x"], torch.ones(4) + 1)

    def test_type_transformation(self):
        td = TensorDict(
            {"x": torch.ones(4, dtype=torch.int32)},
            shape=(4,),
        )
        result = td.apply(lambda t: t.float())
        assert result["x"].dtype == torch.float32

    def test_preserves_shape(self):
        td = TensorDict({"x": torch.randn(4, 5, 6)}, shape=(4, 5))
        result = td.apply(lambda t: t * 2)
        assert result.shape == torch.Size([4, 5])

    def test_preserves_device(self):
        td = TensorDict({"x": torch.randn(4)}, shape=(4,), device="cpu")
        result = td.apply(lambda t: t * 2)
        assert result.device == td.device

    def test_original_unchanged(self):
        original = torch.randn(4, 3)
        td = TensorDict({"x": original.clone()}, shape=(4,))
        td.apply(lambda t: t * 2)
        assert torch.equal(td["x"], original)

    def test_returns_new_tensordict(self):
        td = TensorDict({"x": torch.randn(4)}, shape=(4,))
        result = td.apply(lambda t: t)
        assert result is not td

    def test_subclass_preserved(self):
        class MyTensorDict(TensorDict):
            pass

        td = MyTensorDict({"x": torch.randn(4)}, shape=(4,))
        result = td.apply(lambda t: t * 2)
        assert isinstance(result, MyTensorDict)


@skipif_no_compile
class TestApplyCompile:
    def test_apply_compiled(self):
        td = TensorDict(
            {"x": torch.randn(4, 10), "y": torch.randn(4, 3)},
            shape=(4,),
        )

        def apply_fn(t):
            return t.apply(lambda x: x * 2)

        eager_result, compiled_result = run_and_compare_compiled(apply_fn, td)
        assert compiled_result.shape == td.shape

    def test_apply_nested_compiled(self):
        nested = TensorDict({"a": torch.randn(4, 2)}, shape=(4,))
        td = TensorDict({"nested": nested, "b": torch.randn(4, 3)}, shape=(4,))

        def apply_fn(t):
            return t.apply(lambda x: x + 1)

        eager_result, compiled_result = run_and_compare_compiled(apply_fn, td)
        assert isinstance(compiled_result["nested"], TensorDict)
