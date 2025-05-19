import pytest
import torch
from torch import Tensor

from rtd.tensor_dict import TensorDict

# Skip these tests if torch.compile is not available (requires PyTorch ≥2.0)
compile_fn = getattr(torch, "compile", None)


@pytest.mark.skipif(compile_fn is None, reason="torch.compile not available")
def test_compile_tensor_dict_creation():
    a = torch.randn(5, 2)

    def make_td():
        data = {
            "a": a,
            "b": torch.arange(10, dtype=torch.float32).reshape(5, 2),
        }
        # create a new TensorDict inside the compiled function
        return TensorDict(data, shape=(5,), device=torch.device("cpu"))

    # compile the constructor
    make_td_compiled = torch.compile(make_td)

    td_ref = make_td()
    td_cmp = make_td_compiled()

    # Both outputs must be TensorDict with same keys, shape, and contents
    assert isinstance(td_cmp, TensorDict)
    assert td_cmp.shape == td_ref.shape
    assert td_cmp.device == td_ref.device
    assert set(td_cmp.data.keys()) == set(td_ref.data.keys())

    for k in td_ref.data:
        v_ref = td_ref.data[k]
        v_cmp = td_cmp.data[k]
        assert torch.allclose(v_cmp, v_ref), f"Mismatch in key '{k}'"


@pytest.mark.skipif(compile_fn is None, reason="torch.compile not available")
def test_compile_simple_arithmetic():
    # Create a TensorDict with two tensors
    data = {
        "a": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "b": torch.ones(2, 3, dtype=torch.float32) * 2.0,
    }
    td = TensorDict(data, shape=(2,), device=torch.device("cpu"))

    # Define a simple function that uses TensorDict operations
    def fn(x: TensorDict) -> TensorDict:
        # element‐wise add and multiply
        y = x.copy()
        y["sum"] = x["a"] + x["b"]
        y["scaled"] = (x["a"] * 3.0).view(2, 3)
        return y

    # Compile the function
    fn_compiled = torch.compile(fn)

    # Execute both compiled and uncompiled
    out_ref = fn(td)
    out_cmp = fn_compiled(td)

    # Both should be TensorDict of same class
    assert isinstance(out_ref, TensorDict)
    assert isinstance(out_cmp, TensorDict)
    # keys match
    assert set(out_cmp.data.keys()) == set(out_ref.data.keys())

    # values equal
    for k in out_ref.data:
        assert torch.allclose(out_cmp.data[k], out_ref.data[k]), f"Mismatch in key {k}"


@pytest.mark.skipif(compile_fn is None, reason="torch.compile not available")
def test_compile_stack_and_to():
    # Create two TensorDicts
    td1 = TensorDict({"x": torch.randn(4, 5)}, shape=(4,), device=torch.device("cpu"))
    td2 = TensorDict({"x": torch.randn(4, 5)}, shape=(4,), device=torch.device("cpu"))

    # Function that stacks, moves to CPU, and clones
    def fn(a: TensorDict, b: TensorDict) -> TensorDict:
        s = torch.stack([a, b], dim=0)
        s = s.to(torch.device("cpu"))
        return s.clone()

    fn_compiled = torch.compile(fn)

    out_ref = fn(td1, td2)
    out_cmp = fn_compiled(td1, td2)

    # Both should be TensorDict
    assert isinstance(out_cmp, TensorDict)
    # Shape should be (2, 4)
    assert out_cmp.shape == out_ref.shape == (2, 4)
    # Underlying tensor data equal
    for k in out_ref.data:
        assert torch.allclose(out_cmp.data[k], out_ref.data[k])


@pytest.mark.skipif(
    compile_fn is None or not torch.cuda.is_available(),
    reason="torch.compile or CUDA not available",
)
def test_compile_on_cuda():
    # Test that compile works on GPU
    td = TensorDict(
        {"y": torch.randn(3, 3, device="cuda")},
        shape=(3,),
        device=torch.device("cuda"),
    )

    def fn(x: TensorDict) -> Tensor:
        # sample a simple operation and return a Tensor
        return (x["y"] * 0.5).sum()

    fn_compiled = torch.compile(fn)

    out_ref = fn(td)
    out_cmp = fn_compiled(td)

    assert torch.is_tensor(out_cmp)
    assert torch.allclose(out_cmp, out_ref)

    # Ensure the result is on CUDA
    assert out_cmp.device.type == "cuda"
