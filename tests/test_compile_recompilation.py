"""Test that TensorContainer operations don't cause _tree_map recompilation.

The bug: ~25 methods each pass a different lambda to the shared _tree_map
classmethod.  When Dynamo compiles _tree_map as its own frame (which happens
in complex real-world models but NOT in simple fullgraph=True tests), it guards
on func.__code__ and recompiles for every unique lambda until the limit (8) is
exhausted → RecompileError.

The fix: each method calls pytree.tree_map directly when
torch.compiler.is_compiling(), so _tree_map is never entered during compilation
and no callable-identity guard accumulates.

Testing strategy:
    We cannot rely on Dynamo naturally creating a separate frame for _tree_map
    in simple tests — it always inlines in isolated scenarios.  Instead we
    verify the fix mechanism with two complementary tests:

    1. Confirm that the underlying Dynamo behavior (recompile on callable
       identity) still exists by directly compiling _tree_map with >8 callables.

    2. Confirm that each method has the is_compiling() fast path by compiling
       a function that calls all methods — inside the compiled region,
       is_compiling() returns True, so methods must bypass _tree_map.  We
       verify by checking _tree_map is NOT called during compilation (its
       compile counter stays at 0 frames).
"""

import pytest
import torch
import torch._dynamo

from tensorcontainer.tensor_dict import TensorDict
from tests.conftest import skipif_no_compile

keys = ["a", "b", "c", "d", "e", "f", "g"]


def _get_td():
    return TensorDict(
        {k: torch.randn(3, 4, 5) for k in keys},
        shape=(3, 4),
    )


@skipif_no_compile
def test_methods_bypass_tree_map_when_compiling():
    """When compiled, methods must call pytree.tree_map directly instead of
    _tree_map.  We verify by separately compiling _tree_map and checking its
    frame count stays at 0 when other methods are called from a compiled region.

    Without the is_compiling() fast paths, _tree_map is called by every method
    and its frame count would be >= 1.  With the fast paths, _tree_map is
    bypassed entirely — its frame count stays at 0.
    """
    torch._dynamo.reset()
    td = _get_td()

    # Track whether _tree_map gets compiled as a frame.
    # We wrap _tree_map with a compile counter backend.
    tree_map_call_count = 0
    original_tree_map = TensorDict._tree_map.__func__

    @classmethod
    def counting_tree_map(cls, func, tree, *rests, is_leaf=None):
        nonlocal tree_map_call_count
        tree_map_call_count += 1
        return original_tree_map(cls, func, tree, *rests, is_leaf=is_leaf)

    def many_ops(td):
        # 12 distinct operations — each would call _tree_map without the fix
        x = td.detach()
        x = x.abs()
        x = x.float()
        x = x.clone()
        x = x.neg()
        x = x.mul(2.0)
        x = x.add(1.0)
        x = x.sub(0.5)
        x = x.div(2.0)
        x = x.clamp(-1.0, 1.0)
        x = x.view(12)
        x = x.reshape(3, 4)
        return x

    # Compile and run the function.  Inside the compiled region,
    # is_compiling() returns True.  Methods with the fast path call
    # pytree.tree_map directly; methods without it call _tree_map.
    compiled_fn = torch.compile(many_ops, fullgraph=True)

    # Patch _tree_map to count calls
    TensorDict._tree_map = counting_tree_map
    try:
        tree_map_call_count = 0
        compiled_fn(td)
        compiled_call_count = tree_map_call_count
    finally:
        TensorDict._tree_map = classmethod(original_tree_map)

    # With the fix: _tree_map should NOT be called during compilation
    # (is_compiling() fast path bypasses it).
    # Without the fix: _tree_map would be called 12 times.
    assert compiled_call_count == 0, (
        f"_tree_map was called {compiled_call_count} times during compilation. "
        f"Each method must have an `if torch.compiler.is_compiling(): "
        f"return pytree.tree_map(...)` fast path to bypass _tree_map."
    )


@skipif_no_compile
def test_tree_map_directly_compiled_with_many_callables_hits_limit():
    """Baseline: directly compiling _tree_map and calling it with >8 different
    callables MUST hit the recompile limit.  This confirms the underlying
    Dynamo behavior that our fix is designed to avoid.

    If this test starts passing (i.e. Dynamo no longer guards on callable
    identity), the is_compiling() fast paths become unnecessary — but harmless.
    """
    torch._dynamo.reset()
    td = _get_td()

    compiled_tree_map = torch.compile(TensorDict._tree_map, fullgraph=True)

    callables = [
        lambda x: x.clone(),
        lambda x: x.detach(),
        lambda x: x.float(),
        lambda x: x.double(),
        lambda x: x.half(),
        lambda x: x.long(),
        lambda x: x.int(),
        lambda x: x.abs(),
        lambda x: x.neg(),
        lambda x: x.sqrt(),
    ]

    with torch._dynamo.config.patch(
        recompile_limit=8,
        cache_size_limit=8,
        fail_on_recompile_limit_hit=True,
    ):
        with pytest.raises(torch._dynamo.exc.FailOnRecompileLimitHit):
            for fn in callables:
                compiled_tree_map(fn, td)


@skipif_no_compile
def test_compiled_output_matches_eager():
    """Compiled output must match eager output for all modified operations."""
    torch._dynamo.reset()

    def fn(td):
        x = td.detach()
        x = x.abs()
        x = x.float()
        x = x.clone()
        x = x.neg()
        x = x.mul(2.0)
        x = x.add(1.0)
        x = x.sub(0.5)
        x = x.div(2.0)
        x = x.clamp(-1.0, 1.0)
        x = x.view(12)
        x = x.reshape(3, 4)
        return x

    compiled_fn = torch.compile(fn, fullgraph=True)
    td = _get_td()

    result = compiled_fn(td)
    expected = fn(td)

    for key in result.keys():
        torch.testing.assert_close(result[key], expected[key])
