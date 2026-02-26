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

### The `is_compiling()` fast path doesn't help

We added an early return in `_tree_map` to skip the `try/except` error-wrapping in compile mode:

```python
@classmethod
def _tree_map(cls, func, tree, *rests, is_leaf=None):
    if torch.compiler.is_compiling():
        return pytree.tree_map(func, tree, *rests, is_leaf=is_leaf)
    # ... eager path with try/except error wrapping
```

This does **not** fix the recompilation issue because the `func.__code__` guard is evaluated **at the frame boundary** — before the function body executes. Dynamo decides whether to recompile `_tree_map` based on its arguments before entering it.

The `is_compiling()` branch is still useful for eliminating `try/except` overhead (which can cause graph breaks on older PyTorch versions), but it doesn't prevent the callable-identity guard.

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

### Mechanisms explored to avoid callable guards

| Mechanism | Applicable? | Notes |
|---|---|---|
| `torch.compiler.is_compiling()` fast path | No | Guard fires before function body |
| `torch.compiler.allow_in_graph` | No | Doesn't affect callable guards |
| `torch.compiler.disable` on `_tree_map` | No | Causes graph breaks instead |
| `torch._dynamo.config.recompile_limit` increase | Partial | Band-aid, doesn't fix root cause |
| `guard_filter_fn` (drop `ID_MATCH` guards) | Unsafe | Silent incorrectness if wrong compiled code runs |
| `set_stance(skip_guard_eval_unsafe=True)` | Unsafe | Skips all guard evaluation |
| `torch._dynamo.nonstrict_trace` + `register_constant` | Experimental | Callable body not optimized by compiler |
| Direct `pytree.tree_map` calls from each method | **Yes** | Eliminates the shared frame entirely |

### Why `pytree.tree_map` is special

`pytree.tree_map` is handled natively by Dynamo — it's **traced through**, not compiled as a Python frame. When a method calls `pytree.tree_map(fn, self)` directly, Dynamo traces the tree flattening, applies `fn` to each leaf symbolically, and generates the graph. There's no frame boundary and no guard on `fn` at a frame level.

The problem only arises when `pytree.tree_map` is called **inside** a Python function (`_tree_map`) that Dynamo compiles as its own frame. Then the guard is on `_tree_map`'s `func` parameter, not on `pytree.tree_map`'s.

## Fix

Each method must call `pytree.tree_map` directly in compile mode, bypassing `_tree_map` entirely:

```python
def abs(self) -> Self:
    if torch.compiler.is_compiling():
        return pytree.tree_map(lambda x: x.abs(), self)
    return self._tree_map(lambda x: x.abs(), self)
```

This ensures:
- **Compile mode**: Each method is its own frame with a stable, fixed lambda. No shared `_tree_map` frame, no callable-identity guard churn.
- **Eager mode**: `_tree_map` still provides rich error messages with keypath information and structure mismatch diagnostics.

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
