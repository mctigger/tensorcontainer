import torch
import torch.utils._pytree as pytree
import torch._dynamo.utils

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


def run_and_compare_compiled(fn, *args, **kwargs):
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
    torch.manual_seed(0)
    compiled_fn = torch.compile(fn, fullgraph=True)
    compiled_result = compiled_fn(*args, **kwargs)

    # Assert results are equal
    _compare_results(eager_result, compiled_result)

    # Assert no graph breaks
    assert torch._dynamo.utils.counters["graph_break"]["total"] == 0, (
        f"Graph breaks detected: {torch._dynamo.utils.counters['graph_break']['total']}"
    )

    return eager_result, compiled_result
