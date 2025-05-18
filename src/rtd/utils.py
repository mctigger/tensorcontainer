from collections.abc import Mapping
from typing import Any, Callable, Dict, Iterable, List, TypeAlias, Union

from torch import Tensor

NestedDict: TypeAlias = Dict[str, Union[Tensor, "NestedDict"]]


def get_leaves(dictionary: NestedDict) -> List[Any]:
    if not isinstance(dictionary, Dict):
        return [dictionary]

    leaves = []
    for value in dictionary.values():
        leaves.extend(get_leaves(value))

    return leaves


def zip_apply_leaves(
    dictionaries: Iterable[NestedDict], operation: Callable[..., Any]
) -> Union[NestedDict, Any]:
    if not isinstance(dictionaries[0], Dict):
        return operation(*dictionaries)

    result = {}

    for k in dictionaries[0].keys():
        result[k] = zip_apply_leaves([td[k] for td in dictionaries], operation)

    return result


def apply_leaves(
    dictionary: NestedDict, operation: Callable[[Any], Any]
) -> Union[NestedDict, Any]:
    if not isinstance(dictionary, Dict):
        return operation(dictionary)

    result = {}

    for k, v in dictionary.items():
        result[k] = apply_leaves(v, operation)

    return result


def map_nested(obj: Any, fn: Callable[[Any], Any]) -> Any:
    """
    Recursively traverse a nested dict, calling fn on each node _after_ its children
    have been processed.  If fn returns something new, that replaces the node.

    Args:
      obj: either a Mapping or a leaf value
      fn:  a function that takes the current object (leaf or dict) and returns
           either the original object or a replacement.
    Returns:
      The transformed object tree.
    """
    if isinstance(obj, Mapping):
        # First recurse into children
        new_dict = {k: map_nested(v, fn) for k, v in obj.items()}
        # Then call fn on the reconstructed dict
        return fn(new_dict)
    else:
        # Leaf node: just call fn on it
        return fn(obj)
