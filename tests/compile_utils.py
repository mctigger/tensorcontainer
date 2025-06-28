import torch
import torch._dynamo
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
    and asserts the number of graph breaks.
    """
    # Reset dynamo counters
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()

    # Eager run
    torch.manual_seed(0)
    eager_result = fn(*args, **kwargs)

    # Compiled run
    torch.manual_seed(0)
    compiled_fn = torch.compile(fn, fullgraph=fullgraph)
    compiled_result = compiled_fn(*args, **kwargs)

    # Assert results are equal
    _compare_results(eager_result, compiled_result)

    if expected_graph_breaks is not None:
        graph_breaks = sum(torch._dynamo.utils.counters["graph_break"].values())
        assert graph_breaks == expected_graph_breaks, (
            f"Expected {expected_graph_breaks} graph breaks, "
            f"got {graph_breaks} graph breaks"
        )

    return eager_result, compiled_result


def run_and_count_recompiles(fn, *args, expected_recompiles: int):
    """
    Runs a function multiple times with different inputs and asserts the number of
    recompilations.
    This function compiles the given function `fn` and then runs it with each set
    of arguments provided in `args`. It tracks the number of compilations that
    occur and asserts that the number of recompilations (i.e., compilations
    after the first one) matches `expected_recompiles`.
    Args:
        fn: The function to be compiled and tested.
        *args: A variable number of argument tuples. Each tuple represents a
            separate call to the function `fn`. For example, to call `fn` twice
            with different tensors, you might pass `(torch.randn(2),), (torch.randn(3),)`.
        expected_recompiles (int): The expected number of recompilations.
    """
    # Reset dynamo counters to ensure a clean measurement
    torch._dynamo.reset()

    if not args:
        num_compiles = 0
    else:
        # Use CompileCounter as a backend to count compilations
        counter = torch._dynamo.testing.CompileCounter()
        compiled_fn = torch.compile(fn, backend=counter, fullgraph=True)

        # Invoke the compiled function with each set of arguments
        for arg_set in args:
            compiled_fn(*arg_set)

        num_compiles = counter.frame_count

    # Recompilations are compilations that occur after the initial one.
    # If there are 0 or 1 total compilations, there are no recompilations.
    actual_recompiles = max(0, num_compiles - 1)

    # Assert that the actual number of recompilations matches the expected number
    assert actual_recompiles == expected_recompiles, (
        f"Expected {expected_recompiles} recompiles, but got {actual_recompiles} "
        f"({num_compiles} total compilations)."
    )
