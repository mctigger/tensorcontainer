import pytest
import torch

from tensorcontainer import TensorDict


class TestLike:
    def test_basic(self):
        td = TensorDict(
            {"obs": torch.randn(4, 10), "act": torch.randn(4, 3)}, shape=(4,)
        )
        td2 = td.like({"obs": td["obs"]})
        assert list(td2.keys()) == ["obs"]
        assert td2.shape == torch.Size([4])
        assert td2.device == td.device

    def test_empty(self):
        td = TensorDict({"x": torch.zeros(2, 3)}, shape=(2,))
        td2 = td.like({})
        assert len(td2) == 0
        assert td2.shape == torch.Size([2])

    def test_nested(self):
        inner = TensorDict({"a": torch.ones(4, 2)}, shape=(4,))
        td = TensorDict({"inner": inner, "b": torch.zeros(4)}, shape=(4,))
        td2 = td.like({"inner": inner})
        assert isinstance(td2["inner"], TensorDict)
        assert td2.shape == torch.Size([4])

    def test_shape_mismatch_raises(self):
        td = TensorDict({"x": torch.zeros(4)}, shape=(4,))
        with pytest.raises(RuntimeError):
            td.like({"x": torch.zeros(5)})

    def test_different_data(self):
        td = TensorDict({"x": torch.zeros(4)}, shape=(4,))
        new_x = torch.ones(4)
        td2 = td.like({"x": new_x})
        assert torch.equal(td2["x"], new_x)

    def test_subclass(self):
        class MyTensorDict(TensorDict):
            pass

        td = MyTensorDict({"x": torch.zeros(4)}, shape=(4,))
        td2 = td.like({"x": torch.ones(4)})
        assert isinstance(td2, MyTensorDict)

    def test_device_override(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        td = TensorDict({"x": torch.zeros(4)}, shape=(4,), device="cpu")
        cuda_data = {"x": torch.zeros(4, device="cuda")}

        # Should fail without override
        with pytest.raises(RuntimeError):
            td.like(cuda_data)

        # Should succeed with override
        td2 = td.like(cuda_data, device="cuda")
        assert td2.device.type == "cuda"
        assert td2["x"].device.type == "cuda"
