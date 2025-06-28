import torch
import torch._dynamo.testing
import torch._dynamo.utils
import torch.utils._pytree as pytree

from rtd.tensor_container import TensorContainer


def assert_tc_equal(tc_a: TensorContainer, tc_b: TensorContainer):
    """
    Asserts that two TensorContainers are equal in shape, device, structure, and values.
    """
    assert tc_a.shape == tc_b.shape, "Shape mismatch"
    assert tc_a.device == tc_b.device, "Device mismatch"

    leaves_a, spec_a = pytree.tree_flatten(tc_a)
    leaves_b, spec_b = pytree.tree_flatten(tc_b)

    assert spec_a == spec_b, "PyTree spec mismatch (keys or nesting)"

    for tensor_a, tensor_b in zip(leaves_a, leaves_b):
        assert torch.allclose(tensor_a, tensor_b), "Tensor values mismatch"


def _compare_results(eager_result, compiled_result):
    """
    Recursively compares eager and compiled results.
    """
    if isinstance(eager_result, TensorContainer):
        assert_tc_equal(eager_result, compiled_result)
    elif isinstance(eager_result, torch.Tensor):
        assert torch.allclose(eager_result, compiled_result)
    elif isinstance(eager_result, (tuple, list)):
        assert len(eager_result) == len(compiled_result)
        for er, cr in zip(eager_result, compiled_result):
            _compare_results(er, cr)
    elif isinstance(eager_result, dict):
        assert eager_result.keys() == compiled_result.keys()
        for k in eager_result:
            _compare_results(eager_result[k], compiled_result[k])
    else:
        assert eager_result == compiled_result, "Eager and compiled results mismatch"


def run_and_compare_compiled(
    fn,
    *args,
    fullgraph=True,
    expected_graph_breaks=None,
    **kwargs,
):
    """
    Runs a function in eager mode and compiled mode, compares results,
    and asserts no graph breaks.
    """
    # Reset dynamo counters for graph break detection
    torch._dynamo.utils.reset_graph_break_dup_checker()

    # Eager run
    torch.manual_seed(0)
    eager_result = fn(*args, **kwargs)

    # Compiled run
    # Temporarily enable dynamic shape capture for compilation
    original_capture_dynamic = torch._dynamo.config.capture_dynamic_output_shape_ops
    torch._dynamo.config.capture_dynamic_output_shape_ops = fullgraph
    try:
        torch.manual_seed(0)
        counter = torch._dynamo.testing.CompileCounter()
        compiled_fn = torch.compile(fn, fullgraph=fullgraph)
        compiled_result = compiled_fn(*args, **kwargs)
    finally:
        # Reset to original value
        torch._dynamo.config.capture_dynamic_output_shape_ops = original_capture_dynamic

    # Assert results are equal
    _compare_results(eager_result, compiled_result)

    if expected_graph_breaks is not None:
        assert counter.frame_count == expected_graph_breaks, (
            f"Expected {expected_graph_breaks} graph breaks, got {counter.frame_count}"
        )

    return eager_result, compiled_result
