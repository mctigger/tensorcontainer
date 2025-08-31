from abc import abstractmethod
from typing import Any, Callable, Iterable, List, Tuple, Type, TypeVar

import torch
import torch.utils._pytree as pytree
from torch.utils._pytree import (
    Context,
    KeyEntry,
    PyTree,
    TreeSpec,
    SUPPORTED_NODES,
    _get_node_type,
    KeyPath,
    is_namedtuple
)

from tensorcontainer.types import DeviceLike

_PytreeRegistered = TypeVar("_PytreeRegistered", bound="PytreeRegistered")


class PytreeRegistered:
    """
    A mixin class that automatically registers any of its subclasses
    with the PyTorch PyTree system upon definition.
    """

    def __init_subclass__(cls, **kwargs):
        # This method is called by Python when a class that inherits
        # from PytreeRegistered is defined. `cls` is the new subclass.
        super().__init_subclass__(**kwargs)

        pytree.register_pytree_node(
            cls,
            cls._pytree_flatten,
            cls._pytree_unflatten,
            flatten_with_keys_fn=cls._pytree_flatten_with_keys_fn,
        )

    @abstractmethod
    def _pytree_flatten(self) -> Tuple[List[Any], Context]:
        pass

    @abstractmethod
    def _pytree_flatten_with_keys_fn(
        self,
    ) -> Tuple[List[Tuple[KeyEntry, Any]], Any]:
        pass

    @classmethod
    @abstractmethod
    def _pytree_unflatten(
        cls: Type[_PytreeRegistered], leaves: Iterable[Any], context: Context
    ) -> PyTree:
        pass


def resolve_device(device: DeviceLike) -> torch.device:
    """
    Resolves a device string to a torch.device object using manual device
    resolution without creating dummy tensors. Auto-indexes devices when
    no specific index is provided.

    If the device cannot be resolved or is not available, raises an appropriate
    exception instead of returning a fallback.

    Args:
        device_str: Device string or torch.device object to resolve

    Returns:
        torch.device: The resolved device object with proper indexing

    Raises:
        RuntimeError: If the device type is invalid or device is not available
        AssertionError: If the backend is not compiled/available
    """
    # Handle case where input is already a torch.device object
    if isinstance(device, torch.device):
        device = device
    elif device is not None:
        device = torch.device(device)
    else:
        raise ValueError("Device cannot be None")

    # Auto-index the device
    if device.index is None:
        if device.type == "cuda":
            # Check if CUDA is available before trying to get current device
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available")
            device = torch.device(device.type, index=torch.cuda.current_device())
        elif device.type not in ("cpu", "meta"):
            # For other device types (mps, xla, etc.), default to index 0
            # This will raise an error if the backend is not available
            device = torch.device(device.type, index=0)

    return device


def _get_differing_namedtuple_fields(contexts: tuple) -> dict[str, list]:
    """Get fields that differ between namedtuple contexts.
    
    Args:
        contexts: Tuple of namedtuple contexts to compare
        
    Returns:
        Dictionary mapping field names to their values across contexts
    """
    first_context = contexts[0]
    differing_fields = {}
    
    for field in first_context._fields:
        field_values = [getattr(ctx, field) for ctx in contexts]
        first_value = getattr(first_context, field)
        
        if not all(value == first_value for value in field_values):
            differing_fields[field] = field_values
            
    return differing_fields


def _create_context_error_message(contexts: tuple) -> str:
    """Create a descriptive error message for differing contexts.
    
    Args:
        contexts: Tuple of contexts that differ
        
    Returns:
        Formatted error message string
    """
    if not is_namedtuple(contexts[0]):
        return f"Contexts differ: {contexts}"
    
    differing_fields = _get_differing_namedtuple_fields(contexts)
    
    if not differing_fields:
        return f"Contexts differ: {contexts}"
    
    field_descriptions = []
    for field, values in differing_fields.items():
        field_descriptions.append(f"{field}: {values}")
    
    return f"Contexts differ in fields: {', '.join(field_descriptions)}"


def _assert_all_contexts_equal(contexts: tuple) -> None:
    """Assert that all contexts are equal.
    
    Args:
        contexts: Tuple of contexts to compare
        
    Raises:
        AssertionError: If contexts are not all equal
    """
    first_context = contexts[0]
    
    if all(ctx == first_context for ctx in contexts):
        return
        
    error_message = _create_context_error_message(contexts)
    raise AssertionError(error_message)


def _ensure_generators_exhausted(generators: list) -> None:
    """Ensure all generators are exhausted (same length).
    
    Args:
        generators: List of generators to check
        
    Raises:
        AssertionError: If any generator has remaining elements
    """
    for i, gen in enumerate(generators):
        try:
            next(gen)
            raise AssertionError(f"Tree {i} has more elements than others. This is a bug. Report to developers")
        except StopIteration:
            pass


def _assert_all_keypaths_equal(keypaths: tuple[KeyPath]) -> None:
    """Assert that all key paths are equal.
    
    Args:
        keypaths: Tuple of key paths to compare
        
    Raises:
        AssertionError: If key paths differ
    """
    first_keypath = keypaths[0]
    assert all(kp == first_keypath for kp in keypaths), (
        f"Key paths differ: {keypaths}"
    )


def _assert_all_types_equal(node_types: tuple[type]) -> None:
    """Assert that all node types are equal.
    
    Args:
        node_types: Tuple of node types to compare
        
    Raises:
        AssertionError: If node types differ
    """
    first_node_type = node_types[0]
    assert all(nt == first_node_type for nt in node_types), (
        f"Node types differ: {node_types}"
    )


def _assert_pytrees_equal(trees: list[PyTree]):
    """Assert that all PyTrees in the list have identical structure.

    Iterates through the key paths, node types, and contexts of each PyTree
    and asserts that they are equal across all trees.

    Args:
        trees: List of PyTrees to compare

    Raises:
        AssertionError: If the PyTrees have different structures
    """
    if not trees:
        return

    generators = [_generate_key_path_context((), tree) for tree in trees]
    
    for knc in zip(*generators):
        keypaths, node_types, contexts = tuple(zip(*knc))
        
        _assert_all_keypaths_equal(keypaths)

        try:
            _assert_all_types_equal(node_types)
            _assert_all_contexts_equal(contexts)
        except Exception as e:
            raise ValueError(f"Error at keypath {str(keypaths[0][0])}: {str(e)}") from e

    _ensure_generators_exhausted(generators)


def _generate_key_path_context(
    key_path: KeyPath,
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> Iterable[tuple[KeyPath, type, Context]]:
    """Recursively traverses a PyTree and yields key paths, node types, and contexts.

    This generator function walks through a PyTree structure, yielding information
    about each node that has a registered handler in the PyTorch PyTree system.
    It stops at leaf nodes or unregistered node types.

    Args:
        key_path: Tuple representing the sequence of keys to reach the current node.
        tree: The current PyTree node being processed.
        is_leaf: Optional callable that determines if a node should be treated as a leaf.
            If provided and returns True for the current tree, traversal stops.

    Yields:
        Tuple of (key_path, node_type, context) for each registered node encountered.

    Raises:
        ValueError: If a node type is registered but lacks a flatten_with_keys_fn.
    """
    # Check if current node should be treated as a leaf
    if is_leaf and is_leaf(tree):
        return

    # Determine the type of the current node
    node_type = _get_node_type(tree)
    # Lookup the handler for this node type from the registry
    handler = SUPPORTED_NODES.get(node_type)
    if not handler:
        # Node type not registered, treat as leaf and stop traversal
        return

    # Get the flatten_with_keys function from the handler
    flatten_with_keys = handler.flatten_with_keys_fn
    if flatten_with_keys:
        # Flatten the node to get children and context
        keys_and_children, context = flatten_with_keys(tree)
        # Yield the current node's information
        yield key_path, node_type, context

        # Recursively process each child node
        for k, c in keys_and_children:
            new_key_path = (*key_path, k)
            yield from _generate_key_path_context(new_key_path, c, is_leaf)
    else:
        # Handler exists but no flatten_with_keys_fn provided
        raise ValueError(
            f"Did not find a flatten_with_keys_fn for type: {node_type}. "
            "Please pass a flatten_with_keys_fn argument to register_pytree_node."
        )
