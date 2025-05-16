from typing import Any, Callable, Dict, Iterable, List, Union, TypeAlias


NestedDict: TypeAlias = Dict[str, Union[Any, "NestedDict"]]


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
