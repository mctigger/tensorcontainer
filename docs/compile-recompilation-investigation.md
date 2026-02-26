# `_tree_map` Recompilation Under `torch.compile`

## Problem

When using TensorContainer inside a `torch.compile` region, users hit `RecompileError` or exhaust the recompile limit:

```
torch._dynamo.exc.RecompileError: Recompiling function _tree_map
    triggered by the following guard failure(s):
    - 4/0: ___check_obj_id(func.__code__, 126633386583440)
```

This occurs on both PyTorch 2.8 and 2.10 in real-world workloads.

## Root Cause

### How `_tree_map` works

Every tensor operation in TensorContainer (`abs()`, `view()`, `float()`, `detach()`, `__getitem__`, etc.) routes through a single shared classmethod `_tree_map`:

```python
# In abs():
return self._tree_map(lambda x: x.abs(), self)

# In float():
return self._tree_map(lambda x: x.float(), self)

# In __getitem__():
return self._tree_map(lambda x: x[key], self)
```

There are ~26 call sites across 6 files, each passing a **different lambda** to `_tree_map`.

### Why Dynamo recompiles

Dynamo treats `_tree_map` as its own compiled frame (not inlined into the caller). When a compiled function calls multiple TensorContainer methods, each method passes a different lambda through `_tree_map`. Dynamo guards on `func.__code__` at the `_tree_map` frame boundary:

1. `td.detach()` → `_tree_map(lambda x: x.detach(), self)` → **compile frame 1**
2. `td.view(20)` → `_tree_map(lambda x: x.view(20, ...), self)` → **recompile** (different `func.__code__`)
3. `td.float()` → `_tree_map(lambda x: x.float(), self)` → **recompile** (different `func.__code__`)
4. ... and so on until the recompile limit (default 8) is exhausted

Each unique lambda has a different `__code__` object, triggering a new recompilation of the `_tree_map` frame.

### Why `_tree_map` is a separate frame

We initially expected Dynamo to **inline** `_tree_map` into the calling function (eliminating the frame boundary). This does happen in isolated test scenarios on PyTorch 2.10:

```python
# This works: 1 frame, 0 recompilations on torch 2.10
@torch.compile
def step(td):
    a = td.abs()
    b = td.float()
    c = td.view(12)
    return a, b, c
```

However, in real-world workloads (e.g., a Dreamer model's `imagine()` function), `_tree_map` is **not inlined** — even on torch 2.10. The multi-frame compilation context `[4/8]` in the warning confirms `_tree_map` is compiled as its own frame. The exact reason Dynamo doesn't inline in these cases may relate to:

- The complexity/depth of the calling function
- Classmethod dispatch overhead
- Dynamo's inlining budget being exhausted by the time it reaches `_tree_map`
- Interaction with other compilation features (cudagraphs, device copies, etc.)

## Investigation Details

### What we tested

| Scenario | Result |
|---|---|
| Simple test: 10 ops in one compiled function (torch 2.10, isolated) | 1 frame, 0 recompiles |
| 10 levels of TensorDict nesting with `.view()` (torch 2.10, isolated) | 1 frame, 0 recompiles |
| Directly compiling `_tree_map` with 10 different callables | `FailOnRecompileLimitHit` after 8 |
| `apply(fn)` from compiled wrapper with different `fn` values | Recompile per unique `fn` |
| Real Dreamer model `imagine()` function (torch 2.8) | `RecompileError` on `_tree_map` |
| Real Dreamer model `imagine()` function (torch 2.10) | `recompile_limit` hit on `_tree_map` |
| `_tree_map_fn()` factory replacing per-method guards (real workload) | `recompile_limit` hit on per-method lambdas (rank mismatch) |

### Mechanisms explored to avoid callable guards

| Mechanism | Applicable? | Notes |
|---|---|---|
| `torch.compiler.is_compiling()` fast path in `_tree_map` | No | Guard fires before function body |
| `torch.compiler.allow_in_graph` | No | Doesn't affect callable guards |
| `torch.compiler.disable` on `_tree_map` | No | Causes graph breaks instead |
| `torch._dynamo.config.recompile_limit` increase | Partial | Band-aid, doesn't fix root cause |
| `guard_filter_fn` (drop `ID_MATCH` guards) | Unsafe | Silent incorrectness if wrong compiled code runs |
| `set_stance(skip_guard_eval_unsafe=True)` | Unsafe | Skips all guard evaluation |
| `torch._dynamo.nonstrict_trace` + `register_constant` | Experimental | Callable body not optimized by compiler |
| `substitute_in_graph` on classmethod `_tree_map.__func__` | No | Classmethod descriptor identity mismatch; see below |
| `substitute_in_graph` on module-level `_tree_map_impl` | No | Dynamo still compiles as its own frame in complex models; see below |
| `_tree_map_fn()` classmethod factory | No | Moves the problem from `_tree_map` to per-method lambdas; see below |
| Per-method `is_compiling()` guards calling `pytree.tree_map` directly | **Yes** | Eliminates the shared frame entirely |

### Why `_tree_map_fn()` factory doesn't work

We attempted to centralize the `is_compiling()` dispatch into a single classmethod factory:

```python
@classmethod
def _tree_map_fn(cls):
    if torch.compiler.is_compiling():
        return pytree.tree_map
    return cls._tree_map

# Then each method becomes a single line:
def abs(self) -> Self:
    return self._tree_map_fn()(lambda x: x.abs(), self)

def detach(self) -> Self:
    return self._tree_map_fn()(lambda x: x.detach(), self)
```

This passes unit tests (where Dynamo inlines everything) but **fails in real-world workloads** with a different recompilation pattern. Instead of recompiling `_tree_map`, Dynamo now compiles each method's **lambda** as its own frame and recompiles it due to **tensor rank mismatches**:

```
[24/8] torch._dynamo hit config.recompile_limit (8)
   function: '<lambda>' (tensorcontainer/tensor_container.py:892)
   last reason: 24/7: tensor 'x' rank mismatch. expected 3, actual 2

[17/8] torch._dynamo hit config.recompile_limit (8)
   function: '<lambda>' (tensorcontainer/mixins/device_operations.py:86)
   last reason: 17/7: tensor 'x' rank mismatch. expected 3, actual 2
```

**Why this happens**: With the factory pattern, `_tree_map_fn()` returns `pytree.tree_map` during compilation. `pytree.tree_map` is traced natively by Dynamo — it flattens the tree and applies the lambda to each leaf. But the lambda itself (`lambda x: x.detach()`) becomes its own compiled frame. When the same method (e.g., `detach()`) is called on TensorContainers with different structures (different numbers of tensors, or tensors with different ranks), Dynamo recompiles the lambda frame because the tensor rank guard fails.

With the per-method `if is_compiling()` pattern, the lambda is written **inline in the method body** next to the `pytree.tree_map` call. Dynamo traces through `pytree.tree_map` natively (no frame boundary for it), and the lambda is inlined into the caller's graph as part of the tree-map trace. There's no separate frame for the lambda, so no rank-guard recompilation.

**Key insight**: The factory pattern changes *where* the lambda is seen by Dynamo. With the factory, the lambda is passed as an argument to `pytree.tree_map` from a different scope, causing Dynamo to treat it as a separate frame. With the inline pattern, the lambda is part of the same trace context as the `pytree.tree_map` call, so it gets inlined into the graph.

### Why `pytree.tree_map` is special

`pytree.tree_map` is handled natively by Dynamo — it's **traced through**, not compiled as a Python frame. When a method calls `pytree.tree_map(fn, self)` directly, Dynamo traces the tree flattening, applies `fn` to each leaf symbolically, and generates the graph. There's no frame boundary and no guard on `fn` at a frame level.

The problem only arises when `pytree.tree_map` is called **inside** a Python function (`_tree_map`) that Dynamo compiles as its own frame. Then the guard is on `_tree_map`'s `func` parameter, not on `pytree.tree_map`'s.

### Why `substitute_in_graph` doesn't work

We explored two approaches with `substitute_in_graph`:

#### Approach 1: On the classmethod's `__func__`

```python
@torch.compiler.substitute_in_graph(
    TensorContainer._tree_map.__func__, skip_signature_check=True
)
def _tree_map_substitute(cls, func, tree, *rests, is_leaf=None):
    return pytree.tree_map(func, tree, *rests, is_leaf=is_leaf)
```

This fails because `substitute_in_graph` registers the polyfill keyed on the function's `id()`. When Dynamo encounters `self._tree_map(...)`, it resolves through the classmethod descriptor — the resulting bound method has a different `id()` than `_tree_map.__func__`, so the polyfill lookup fails.

#### Approach 2: On a module-level function

To avoid the classmethod descriptor issue, we extracted the implementation into a module-level `_tree_map_impl` function with a stable `id()` and registered a polyfill on it:

```python
def _tree_map_impl(func, tree, *rests, is_leaf=None):
    # ... error-wrapping implementation ...

@torch.compiler.substitute_in_graph(_tree_map_impl)
def _tree_map_compile(func, tree, *rests, is_leaf=None):
    return pytree.tree_map(func, tree, *rests, is_leaf=is_leaf)
```

This passes unit tests (where Dynamo inlines `_tree_map_impl` into the caller's graph), but **still fails in real-world workloads**. The reason: `substitute_in_graph` operates at the **tracing** level — it tells Dynamo how to trace through a function when it encounters a call to it *within* a frame being traced. But when Dynamo decides to compile `_tree_map_impl` as its **own separate frame** (which happens in complex models), it's not tracing through a call — it's treating `_tree_map_impl` as a top-level compilation entry point. At that point, `substitute_in_graph` has no effect.

**Key insight**: `substitute_in_graph` helps when a function is *inlined during tracing* of an outer frame. It does not help when a function is compiled as its own frame — which is exactly the scenario that causes the recompilation bug.

## Fix (implemented)

### Two layers of defense

The fix uses per-method `is_compiling()` guards as the primary defense, with a `substitute_in_graph` polyfill on the module-level `_tree_map_impl` as additional structural insurance:

#### Layer 1: `is_compiling()` guards (proven fix)

Each method calls `pytree.tree_map` directly when `torch.compiler.is_compiling()` returns `True`, bypassing `_tree_map` entirely:

```python
def abs(self) -> Self:
    if torch.compiler.is_compiling():
        return pytree.tree_map(lambda x: x.abs(), self)
    return self._tree_map(lambda x: x.abs(), self)
```

This ensures:
- **Compile mode**: Each method is its own frame with a stable, fixed lambda. No shared `_tree_map` frame, no callable-identity guard churn.
- **Eager mode**: `_tree_map` still provides rich error messages with keypath information and structure mismatch diagnostics.

#### Layer 2: Module-level `_tree_map_impl` with polyfill (structural)

The `_tree_map` classmethod delegates to a module-level `_tree_map_impl` function, which has a `substitute_in_graph` polyfill registered. In scenarios where Dynamo does inline `_tree_map_impl` (simple tests, future Dynamo improvements), the polyfill provides an additional optimization path:

```python
def _tree_map_impl(func, tree, *rests, is_leaf=None):
    """Module-level function with stable identity for substitute_in_graph."""
    # ... error-wrapping implementation ...

@torch.compiler.substitute_in_graph(_tree_map_impl)
def _tree_map_compile(func, tree, *rests, is_leaf=None):
    return pytree.tree_map(func, tree, *rests, is_leaf=is_leaf)

class TensorContainer:
    @classmethod
    def _tree_map(cls, func, tree, *rests, is_leaf=None):
        return _tree_map_impl(func, tree, *rests, is_leaf=is_leaf)
```

### `apply(fn)` — expected limitation

`apply(fn)` passes a user-provided callable. If a compiled function calls `apply()` with different `fn` values across invocations, Dynamo **must** recompile because different callables produce different computation graphs. This is fundamental Dynamo behavior, not a TensorContainer bug.

Recommended workarounds for users:
- Use built-in methods (`abs`, `float`, `view`, etc.) instead of `apply()` when possible
- Call `apply()` outside the compiled region
- Increase `recompile_limit` / `cache_size_limit` if many unique callables are expected

### Affected call sites (26 total)

**`mixins/math_operations.py`** (10): `abs`, `add`, `sub`, `mul`, `div`, `pow`, `sqrt`, `log`, `neg`, `clamp`

**`mixins/type_operations.py`** (5): `float`, `double`, `half`, `long`, `int`

**`mixins/device_operations.py`** (3): `detach`, `clone`, `copy`

**`mixins/shape_operations.py`** (5): `view`, `reshape`, `expand`, `permute`, `transpose`

**`tensor_container.py`** (3): `__getitem__`, `_stack`, `_cat`

## Verification

### Integration test

The fix was verified against the original failing command:

```bash
python scripts/benchmark_unsupervised.py \
  +experiment/miniworld_maze=nine_rooms_episodic_large_goals \
  +execution=local +preset=leg \
  +preset/deployment/miniworld_maze=visual_small \
  execution.compile=reduce-overhead
```

This previously raised `RecompileError` on `_tree_map`. After the fix, compilation
succeeds and training proceeds. Note: `reduce-overhead` mode sets
`torch._dynamo.config.error_on_recompile = True`, which makes any recompilation
a hard error. Normal recompilations (e.g., Dynamo guarding on `requires_grad`
for individual tensors) still occur, but these are standard Dynamo behavior — they
only become errors if `error_on_recompile` is set.

### Unit tests

Tests are in `tests/test_compile_recompilation.py`.

#### Testing challenge: Dynamo inlining

The bug only manifests when Dynamo compiles `_tree_map` as a **separate frame**.
In simple test functions with `fullgraph=True`, Dynamo inlines `_tree_map` into
the caller's graph — eliminating the frame boundary and the callable-identity
guard entirely. This means a naive test that compiles a function calling 12+
TensorContainer methods will **pass even without the fix**.

We confirmed this empirically: removing all `is_compiling()` fast paths and
running the `fullgraph=True` tests still produced 0 failures.

#### Test strategy

Instead of trying to force Dynamo to create a separate frame (which depends on
graph complexity, inlining budget, and PyTorch version), the tests verify the
fix mechanism directly:

1. **`test_substitute_in_graph_registered`**: Verifies that `_tree_map_impl` has
   a polyfill registered via `substitute_in_graph`, confirming the structural
   defense is in place.

2. **`test_methods_bypass_tree_map_when_compiling`**: Patches `_tree_map` with a
   counting wrapper, compiles a function that calls 12 different methods, and
   asserts `_tree_map` was called **0 times** during compiled execution. This
   works because `torch.compiler.is_compiling()` returns `True` inside the
   compiled region — methods with the fast path call `pytree.tree_map` directly.
   **Without the fix, this test fails** (`_tree_map` is called 12 times).

3. **`test_tree_map_directly_compiled_with_many_callables_hits_limit`**: Baseline
   test confirming the underlying Dynamo behavior still exists. Directly compiles
   `_tree_map_impl` and calls it with 10 different lambdas — asserts
   `FailOnRecompileLimitHit` is raised after 8. If a future PyTorch version
   removes callable-identity guards, this test will fail, signaling the fast
   paths are no longer needed (but remain harmless).

4. **`test_compiled_output_matches_eager`**: Correctness test — verifies compiled
   output matches eager output for all modified operations.
