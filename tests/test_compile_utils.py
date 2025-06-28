import torch
import pytest
import torch._dynamo

from tests.compile_utils import run_and_compare_compiled, run_and_count_recompiles


def no_break(x):
    # This operation should not cause a graph break
    return x[0] + 1


def one_break(x):
    # This operation is known to cause a graph break
    torch._dynamo.graph_break()
    return x[0] + 1


def two_breaks(x):
    # These operations are known to cause graph breaks
    tensor = x[0]
    torch._dynamo.graph_break()
    torch._dynamo.graph_break()
    return tensor[0] + 1


def three_breaks(x):
    # These operations are known to cause graph breaks
    tensor = x[0]
    torch._dynamo.graph_break()
    torch._dynamo.graph_break()
    torch._dynamo.graph_break()
    return tensor[0] + 1


def test_run_and_compare_compiled_no_break():
    # A single frame is counted for the initial compilation, so no additional breaks should occur.
    x = torch.randn(3, 4)
    run_and_compare_compiled(no_break, (x,), expected_graph_breaks=0)


def test_run_and_compare_compiled_with_break():
    # Accounting for the additional break from the print statement.
    x = torch.randn(3, 4)
    run_and_compare_compiled(one_break, (x,), expected_graph_breaks=1, fullgraph=False)


def test_run_and_compare_compiled_with_two_breaks():
    x = torch.randn(3, 4)
    run_and_compare_compiled(two_breaks, (x,), expected_graph_breaks=2, fullgraph=False)


def test_run_and_compare_compiled_with_three_breaks():
    x = torch.randn(3, 4)
    run_and_compare_compiled(
        three_breaks, (x,), expected_graph_breaks=3, fullgraph=False
    )


def test_run_and_compare_compiled_fails():
    # Assert that this raises an AssertionError.
    x = torch.randn(3, 4)
    with pytest.raises(AssertionError):
        # The 'one_break' function causes 1 graph break, but we expect 0.
        # This should trigger an AssertionError.
        run_and_compare_compiled(
            one_break, (x,), expected_graph_breaks=0, fullgraph=False
        )


def recursive_recompile(i):
    if i > 0:
        i -= 1
        return recursive_recompile(i)
    else:
        return torch.tensor([1])


class TestRunAndCountRecompiles:
    def test_one_recompile(self):
        """Tests that a stateful functor recompiles once on the second call."""
        run_and_count_recompiles(recursive_recompile, (1,), expected_recompiles=0)

    def test_two_recompile(self):
        """Tests that a stateful functor recompiles once on the second call."""
        run_and_count_recompiles(recursive_recompile, (1,), (3,), expected_recompiles=1)

    def test_three_recompile(self):
        """Tests that a stateful functor recompiles once on the second call."""
        run_and_count_recompiles(
            recursive_recompile, (1,), (2,), (3,), expected_recompiles=2
        )
