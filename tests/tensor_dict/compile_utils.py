import torch
import torch.utils._pytree as pytree
import torch._dynamo.utils

from rtd.tensor_dict import TensorDict


def assert_td_equal(td_a: TensorDict, td_b: TensorDict):
    """
    Asserts that two TensorDicts are equal in shape, device, structure, and values.
    """
    assert td_a.shape == td_b.shape, "Shape mismatch"
    assert td_a.device == td_b.device, "Device mismatch"

    leaves_a, spec_a = pytree.tree_flatten(td_a)
    leaves_b, spec_b = pytree.tree_flatten(td_b)

    assert spec_a == spec_b, "PyTree spec mismatch (keys or nesting)"

    for tensor_a, tensor_b in zip(leaves_a, leaves_b):
        assert torch.allclose(tensor_a, tensor_b), "Tensor values mismatch"


def _compare_results(eager_result, compiled_result):
    """
    Recursively compares eager and compiled results.
    """
    if isinstance(eager_result, TensorDict):
        assert_td_equal(eager_result, compiled_result)
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
    compiled_fn = torch.compile(fn)
    compiled_result = compiled_fn(*args, **kwargs)

    # Assert results are equal
    _compare_results(eager_result, compiled_result)

    # Assert no graph breaks
    assert torch._dynamo.utils.counters["graph_break"]["total"] == 0, (
        f"Graph breaks detected: {torch._dynamo.utils.counters['graph_break']['total']}"
    )

    return eager_result, compiled_result
