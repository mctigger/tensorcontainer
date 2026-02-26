"""Definitive test: when does _map help vs _tree_map?

Three scenarios:
1. Compiled directly as entry point: torch.compile(fn)(different_callables...)
2. Called inside a compiled function with baked-in lambdas
3. Methods called inside a compiled function
"""

import torch
import torch._dynamo

from tensorcontainer.tensor_container import TensorContainer, _map
from tensorcontainer.tensor_dict import TensorDict

keys = ["a", "b", "c", "d", "e", "f", "g"]


def _get_td():
    return TensorDict({k: torch.randn(3, 4, 5) for k in keys}, shape=(3, 4))


def run(name, fn):
    torch._dynamo.reset()
    td = _get_td()
    try:
        with torch._dynamo.config.patch(
            recompile_limit=0, cache_size_limit=64, fail_on_recompile_limit_hit=True,
        ):
            fn(td)
            fn(td)
        print(f"  PASS  {name}")
    except torch._dynamo.exc.FailOnRecompileLimitHit:
        print(f"  FAIL  {name}")


# === Scenario 1: compiled directly, called with different callables ===
def run_direct(name, map_fn):
    torch._dynamo.reset()
    td = _get_td()
    compiled = torch.compile(map_fn, fullgraph=True)
    fns = [
        lambda x: x.abs(), lambda x: x.neg(), lambda x: x.float(),
        lambda x: x.double(), lambda x: x.clone(), lambda x: x.detach(),
        lambda x: x.sqrt(), lambda x: x.half(), lambda x: x.long(), lambda x: x.int(),
    ]
    try:
        with torch._dynamo.config.patch(
            recompile_limit=8, cache_size_limit=8, fail_on_recompile_limit_hit=True,
        ):
            for f in fns:
                compiled(f, td)
        print(f"  PASS  {name}")
    except torch._dynamo.exc.FailOnRecompileLimitHit:
        print(f"  FAIL  {name}")


# === Scenario 2: baked-in lambdas inside compiled function ===
def baked_tree_map(td):
    TensorContainer._tree_map(lambda x: x.abs(), td)
    TensorContainer._tree_map(lambda x: x.neg(), td)
    TensorContainer._tree_map(lambda x: x.float(), td)
    TensorContainer._tree_map(lambda x: x.double(), td)
    TensorContainer._tree_map(lambda x: x.clone(), td)
    TensorContainer._tree_map(lambda x: x.detach(), td)
    TensorContainer._tree_map(lambda x: x.sqrt(), td)
    TensorContainer._tree_map(lambda x: x.half(), td)
    TensorContainer._tree_map(lambda x: x.long(), td)
    return TensorContainer._tree_map(lambda x: x.int(), td)


def baked_map(td):
    _map(lambda x: x.abs(), td)
    _map(lambda x: x.neg(), td)
    _map(lambda x: x.float(), td)
    _map(lambda x: x.double(), td)
    _map(lambda x: x.clone(), td)
    _map(lambda x: x.detach(), td)
    _map(lambda x: x.sqrt(), td)
    _map(lambda x: x.half(), td)
    _map(lambda x: x.long(), td)
    return _map(lambda x: x.int(), td)


# === Scenario 3: methods ===
def via_methods(td):
    td.abs()
    td.neg()
    td.float()
    td.double()
    td.clone()
    td.detach()
    td.sqrt()
    td.half()
    td.long()
    return td.int()


print()
print("Scenario 1: compiled DIRECTLY, called with different callables each time")
run_direct("_tree_map", TensorContainer._tree_map)
run_direct("_map", _map)

print()
print("Scenario 2: baked-in lambdas inside one compiled function")
run("_tree_map with baked-in lambdas", torch.compile(baked_tree_map, fullgraph=True))
run("_map with baked-in lambdas", torch.compile(baked_map, fullgraph=True))

print()
print("Scenario 3: TensorDict methods inside one compiled function")
run("methods (currently use _map)", torch.compile(via_methods, fullgraph=True))
