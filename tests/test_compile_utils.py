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


class TestRunAndCountRecompiles:
    def test_zero_recompiles(self):
        """Tests that a function recompiles zero times with same input shape."""

        def func(x):
            return x * 2

        args1 = (torch.randn(4),)
        args2 = (torch.randn(4),)

        run_and_count_recompiles(func, args1, args2, expected_recompiles=0)

    def test_one_recompile(self):
        """Tests that a function recompiles once with different input shapes."""

        def func(x):
            return x * 2

        args1 = (torch.randn(4),)
        args2 = (torch.randn(8),)

        run_and_count_recompiles(func, args1, args2, expected_recompiles=1)

    def test_run_and_count_recompiles_two_recompiles(self):
        """Tests that a function recompiles twice with three different input shapes."""

        def func(x):
            if x.shape[0] > 10:
                return x * 2
            return x + 1

        args1 = (torch.randn(4),)
        args2 = (torch.randn(8),)
        args3 = (torch.randn(16),)

        run_and_count_recompiles(func, args1, args2, args3, expected_recompiles=2)

    def test_run_and_count_recompiles_three_recompiles(self):
        """Tests that a function recompiles three times with four different input shapes."""

        def func(x):
            if x.shape[0] > 20:
                return x * 2
            elif x.shape[0] > 10:
                return x + 2
            return x + 1

        args1 = (torch.randn(4),)
        args2 = (torch.randn(8),)
        args3 = (torch.randn(16),)
        args4 = (torch.randn(32),)

        run_and_count_recompiles(
            func, args1, args2, args3, args4, expected_recompiles=3
        )
