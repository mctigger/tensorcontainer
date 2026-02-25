"""Tests for _tree_map behavior under torch.compile.

There are two distinct concerns:

1. **Internal operation fan-out**: Methods like abs(), float(), neg() each pass
   a different lambda through _tree_map. When these are called within a single
   compiled function, Dynamo inlines them into one graph — no recompilation.

2. **Dynamic callable fan-out**: When _tree_map or apply() is the compiled entry
   point and called repeatedly with different callables, Dynamo guards on callable
   identity and recompiles each time. This is expected Dynamo behavior, not a
   TensorContainer bug.
"""

import pytest
import torch
import torch._dynamo
from torch._dynamo.testing import CompileCounter

from tensorcontainer.tensor_dict import TensorDict
from tests.compile_utils import run_and_count_graph_breaks
from tests.conftest import skipif_no_compile

keys = ["a", "b", "c", "d", "e", "f", "g"]


def _get_td():
    return TensorDict(
        {k: torch.randn(3, 4, 5) for k in keys},
        shape=(3, 4),
    )


# --- Named callables for fan-out tests ---

def _clone_leaf(x):
    return x.clone()


def _detach_leaf(x):
    return x.detach()


def _float_leaf(x):
    return x.float()


def _double_leaf(x):
    return x.double()


def _half_leaf(x):
    return x.half()


def _long_leaf(x):
    return x.long()


def _int_leaf(x):
    return x.int()


def _abs_leaf(x):
    return x.abs()


def _neg_leaf(x):
    return x.neg()


def _sqrt_leaf(x):
    return x.sqrt()


@skipif_no_compile
class TestInternalOpFanOut:
    """Tests that many different internal ops in one compiled function work without
    graph breaks or recompilation issues.

    This is the common usage pattern: the user compiles their function which
    internally calls multiple TensorContainer methods. Dynamo inlines through
    each method and sees each lambda exactly once — no recompilation.
    """

    def test_many_ops_single_compiled_function_no_graph_break(self):
        """Many different _tree_map-based operations in one function produce
        zero graph breaks."""
        td = _get_td()

        def many_ops(td):
            a = td.abs()
            b = td.float()
            c = td.neg()
            d = td.sqrt()
            e = td.clone()
            f = td.detach()
            g = td[0]
            h = td.double()
            i = td.half()
            j = td.long()
            return a, b, c, d, e, f, g, h, i, j

        run_and_count_graph_breaks(many_ops, td, expected_graph_breaks=0)

    def test_many_ops_no_recompilation(self):
        """Calling a compiled function with many ops repeatedly doesn't recompile."""
        td = _get_td()

        def many_ops(td):
            a = td.abs()
            b = td.float()
            c = td.neg()
            d = td.sqrt()
            e = td.clone()
            f = td.detach()
            g = td[0]
            h = td.double()
            i = td.half()
            j = td.long()
            return a, b, c, d, e, f, g, h, i, j

        torch._dynamo.reset()
        counter = CompileCounter()
        compiled = torch.compile(many_ops, backend=counter, fullgraph=True)

        with torch._dynamo.config.patch(
            recompile_limit=8,
            cache_size_limit=8,
            fail_on_recompile_limit_hit=True,
        ):
            compiled(td)
            compiled(_get_td())  # second call with same-shaped input

        assert counter.frame_count == 1

    def test_apply_in_compiled_function_no_graph_break(self):
        """apply() with a lambda inside a compiled function produces no graph breaks."""
        td = _get_td()

        def fn(td):
            return td.apply(lambda x: x * 2)

        run_and_count_graph_breaks(fn, td, expected_graph_breaks=0)


@skipif_no_compile
class TestDynamicCallableFanOut:
    """Tests documenting expected Dynamo behavior when _tree_map/apply is called
    with different callables across separate compiled invocations.

    Dynamo guards on callable identity (func.__code__). When a compiled function
    receives a different callable, it must recompile because the callable determines
    the computation graph. This is fundamental Dynamo behavior, not a TensorContainer
    issue.
    """

    def test_tree_map_recompiles_with_different_callables(self):
        """Directly compiling _tree_map and calling with different callables
        causes recompilation — one per unique callable."""
        torch._dynamo.reset()
        td = _get_td()

        counter = CompileCounter()
        compiled_tree_map = torch.compile(
            TensorDict._tree_map, backend=counter, fullgraph=True
        )

        compiled_tree_map(_clone_leaf, td)
        compiled_tree_map(_detach_leaf, td)
        compiled_tree_map(_float_leaf, td)

        # Each unique callable causes a recompilation of the _tree_map frame
        assert counter.frame_count == 3

    def test_apply_recompiles_with_different_callables(self):
        """apply() called from a compiled wrapper with different callables
        causes recompilation — expected Dynamo behavior."""
        torch._dynamo.reset()
        td = _get_td()

        def use_apply(td, fn):
            return td.apply(fn)

        counter = CompileCounter()
        compiled = torch.compile(use_apply, backend=counter, fullgraph=True)

        compiled(td, _abs_leaf)
        compiled(td, _neg_leaf)
        compiled(td, _sqrt_leaf)

        assert counter.frame_count == 3

    def test_callable_fanout_hits_recompile_limit(self):
        """With strict limits, many different callables exhaust the recompile budget.

        This documents the expected failure mode. Users who need many different
        callables should either:
        - Increase recompile_limit/cache_size_limit
        - Use apply() in eager mode outside the compiled region
        - Compose operations using built-in methods instead of apply()
        """
        torch._dynamo.reset()
        td = _get_td()

        compiled_tree_map = torch.compile(TensorDict._tree_map, fullgraph=True)

        all_fns = [
            _clone_leaf, _detach_leaf, _float_leaf, _double_leaf, _half_leaf,
            _long_leaf, _int_leaf, _abs_leaf, _neg_leaf, _sqrt_leaf,
        ]

        with torch._dynamo.config.patch(
            recompile_limit=8,
            cache_size_limit=8,
            fail_on_recompile_limit_hit=True,
        ):
            with pytest.raises(torch._dynamo.exc.FailOnRecompileLimitHit):
                for fn in all_fns:
                    compiled_tree_map(fn, td)


@skipif_no_compile
class TestGetitemCompilation:
    """Tests for __getitem__ compilation behavior."""

    @pytest.mark.parametrize("key", ["a", "b"])
    def test_getitem_compiles_and_caches(self, key):
        """__getitem__ compiles on first call and caches for same key."""
        torch._dynamo.reset()
        td = _get_td()

        counter = CompileCounter()
        compiled_getitem = torch.compile(
            td.__getitem__, backend=counter, fullgraph=True
        )

        compiled_getitem(key)
        assert counter.frame_count == 1

        # Same key, no recompilation
        compiled_getitem(key)
        assert counter.frame_count == 1

    def test_getitem_recompiles_for_different_key(self):
        """__getitem__ with a different key triggers recompilation."""
        torch._dynamo.reset()
        td = _get_td()

        counter = CompileCounter()
        compiled_getitem = torch.compile(
            td.__getitem__, backend=counter, fullgraph=True
        )

        compiled_getitem("a")
        assert counter.frame_count == 1

        compiled_getitem("c")
        assert counter.frame_count == 2
