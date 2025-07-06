import pytest
import torch
import torch._dynamo
import torch._dynamo.utils
from torch._dynamo.testing import CompileCounter  # New import

from tensorcontainer.tensor_dict import TensorDict
from tests.conftest import skipif_no_compile

keys = ["a", "b", "c", "d", "e", "f", "g"]


def _get_td():
    return TensorDict(
        {k: torch.randn(3, 4, 5) for k in keys},
        shape=(3, 4),
    )


@skipif_no_compile
@pytest.mark.parametrize("key", ["a", "b"])
def test_getitem_recompilation(key):
    torch._dynamo.reset()
    # Global counters clear, though CompileCounter is preferred
    torch._dynamo.utils.counters.clear()

    # --- Sanity check with a simple lambda using CompileCounter ---
    lambda_compile_counter = CompileCounter()
    # Using fullgraph=True as it was in the original attempts
    compiled_lambda = torch.compile(
        lambda x: x + 1, backend=lambda_compile_counter, fullgraph=True
    )

    _ = compiled_lambda(torch.randn(1))
    assert lambda_compile_counter.frame_count == 1, (
        "Simple lambda should compile 1 frame on first call"
    )

    _ = compiled_lambda(torch.randn(1))  # Second call, different tensor instance
    assert lambda_compile_counter.frame_count == 1, (
        "Simple lambda, second call, should not recompile frame"
    )
    # --- End sanity check ---

    # --- Now for TensorDict ---
    # Reset Dynamo for a cleaner state for the TensorDict part
    torch._dynamo.reset()
    # No need to clear torch._dynamo.utils.counters if we are relying on CompileCounter

    td = _get_td()

    td_getitem_compile_counter = CompileCounter()
    # Compile the __getitem__ method of the specific td instance
    compiled_getitem = torch.compile(
        td.__getitem__, backend=td_getitem_compile_counter, fullgraph=True
    )

    # First call with the initial 'key'
    _ = compiled_getitem(key)
    assert td_getitem_compile_counter.frame_count == 1, (
        f"TensorDict[key='{key}'] first call should compile 1 frame"
    )

    # Second call with the same 'key'
    _ = compiled_getitem(key)
    assert td_getitem_compile_counter.frame_count == 1, (
        f"TensorDict[key='{key}'] second call (same key) should not recompile frame"
    )

    # Call with a different key ('other_key')
    other_key = "c"
    _ = compiled_getitem(other_key)
    assert td_getitem_compile_counter.frame_count == 2, (
        f"TensorDict[other_key='{other_key}'] should compile a new frame (total 2)"
    )
