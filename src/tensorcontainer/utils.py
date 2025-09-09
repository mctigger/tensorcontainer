from abc import abstractmethod
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
)

import torch
import torch.utils._pytree as pytree
from torch.utils._pytree import (
    Context,
    KeyEntry,
    PyTree,
    SUPPORTED_NODES,
    _get_node_type,
    KeyPath,
)

from tensorcontainer.types import DeviceLike

_PytreeRegistered = TypeVar("_PytreeRegistered", bound="PytreeRegistered")


@dataclass
class StructureMismatch:
    """Base class for PyTree structure mismatch errors."""

    @abstractmethod
    def __str__(self) -> str:
        """Return a human-readable error message describing the mismatch."""
        pass


@dataclass
class KeyPathMismatch(StructureMismatch):
    """Represents a mismatch in key paths between PyTrees."""

    keypaths: Tuple[KeyPath, ...]

    def __str__(self) -> str:
        # Format each keypath for better readability
        formatted_paths = []
        for i, keypath in enumerate(self.keypaths):
            if keypath:
                path_str = "container" + "".join(str(element) for element in keypath)
            else:
                path_str = "container (root)"
            formatted_paths.append(f"Container {i}: {path_str}")
        
        paths_display = "\n".join(formatted_paths)
        
        return (
            f"Structure traversal mismatch: containers have different nesting patterns.\n\n"
            f"{paths_display}\n\n"
            f"Fix: Ensure all containers have identical nested structure at each level."
        )


@dataclass
class TypeMismatch(StructureMismatch):
    """Represents a type mismatch between PyTree nodes."""

    expected_type: type
    actual_type: type
    entry_index: int
    key_path: KeyPath

    def __str__(self) -> str:
        path_str = format_path(self.key_path)
        location = f" at {path_str}" if path_str else ""
        
        # Generate conversion guidance based on common type pairs
        guidance = f"Convert container {self.entry_index} to {self.expected_type.__name__}"
        
        # Add specific guidance for common conversions
        if self.expected_type.__name__ == "TensorDict" and self.actual_type in (list, tuple):
            guidance += " using TensorDict({'key_0': item[0], 'key_1': item[1], ...})"
        elif self.expected_type in (list, tuple) and self.actual_type.__name__ == "TensorDict":
            guidance += f" using {self.expected_type.__name__}(tensor_dict.values())"
        elif self.expected_type is list and self.actual_type is tuple:
            guidance += " using list(container)"
        elif self.expected_type is tuple and self.actual_type is list:
            guidance += " using tuple(container)"
        
        return (
            f"Type mismatch{location}: incompatible container types.\n\n"
            f"Expected: {self.expected_type.__name__}\n"
            f"Found:    {self.actual_type.__name__} (in container {self.entry_index})\n\n"
            f"Fix: {guidance}"
        )


@dataclass
class ContextMismatch(StructureMismatch):
    """Represents a context mismatch between PyTree nodes."""

    expected_context: Context
    actual_context: Context
    entry_index: int
    key_path: KeyPath

    def __str__(self) -> str:
        path_str = format_path(self.key_path)
        location = f" at {path_str}" if path_str else ""
        
        # Extract readable context information
        expected_info = _extract_context_info(self.expected_context)
        actual_info = _extract_context_info(self.actual_context)
        
        # Detect specific types of mismatches for targeted guidance
        guidance = "Ensure all containers have identical structure."
        
        # Try to provide specific guidance based on context differences
        if hasattr(self.expected_context, 'keys') and hasattr(self.actual_context, 'keys'):
            expected_keys = set(getattr(self.expected_context, 'keys', []))
            actual_keys = set(getattr(self.actual_context, 'keys', []))
            if expected_keys != actual_keys:
                missing = expected_keys - actual_keys
                extra = actual_keys - expected_keys
                if missing or extra:
                    guidance = "Key mismatch detected."
                    if missing:
                        guidance += f" Missing keys in container {self.entry_index}: {sorted(missing)}."
                    if extra:
                        guidance += f" Extra keys in container {self.entry_index}: {sorted(extra)}."
        
        return (
            f"Structure mismatch{location}: containers have incompatible layouts.\n\n"
            f"Container 0{location}: {expected_info}\n"
            f"Container {self.entry_index}{location}: {actual_info}\n\n"
            f"Fix: {guidance}"
        )


def format_path(path: KeyPath) -> str:
    """Formats a PyTree KeyPath into a PyTorch-style path string.

    Args:
        path: The KeyPath tuple to format, typically containing keys from PyTree traversal.

    Returns:
        A PyTorch-style path string (e.g., "container['outer']['inner']", "container[0][1]").
    """
    if not path:
        return ""
    
    # KeyPath elements already format nicely: MappingKey -> "['key']", SequenceKey -> "[0]"  
    path_parts = [str(element) for element in path]
    return "container" + "".join(path_parts)


def _extract_context_info(context: Context) -> str:
    """Extract human-readable information from PyTree context objects.
    
    Args:
        context: PyTree context object from _pytree_flatten.
        
    Returns:
        Human-readable string describing the context.
    """
    if hasattr(context, 'keys') and hasattr(context, 'shape'):
        # TensorDict-like context
        keys = getattr(context, 'keys', [])
        shape = getattr(context, 'shape', ())
        device = getattr(context, 'device', None)
        
        keys_str = f"keys={list(keys)}" if keys else "keys=[]"
        shape_str = f"shape={tuple(shape)}" if shape else "shape=()"
        device_str = f"device={device}" if device is not None else "device=None"
        
        return f"({keys_str}, {shape_str}, {device_str})"
    elif hasattr(context, '__dataclass_fields__'):
        # TensorDataClass-like context
        fields = list(getattr(context, '__dataclass_fields__', {}).keys())
        shape = getattr(context, 'shape', ())
        device = getattr(context, 'device', None)
        
        fields_str = f"fields={fields}"
        shape_str = f"shape={tuple(shape)}"
        device_str = f"device={device}" if device is not None else "device=None"
        
        return f"({fields_str}, {shape_str}, {device_str})"
    else:
        # Fallback to string representation
        return str(context)


class PytreeRegistered:
    """
    A mixin class that automatically registers any of its subclasses
    with the PyTorch PyTree system upon definition. This enables seamless
    integration with PyTorch's tree operations (e.g., flattening/unflattening)
    without manual registration calls.
    """

    def __init_subclass__(cls, **kwargs):
        """Automatically registers the subclass with PyTorch's PyTree system.

        This method is invoked when a subclass is defined, ensuring the class
        is treated as a PyTree node for operations like tree_map or tree_flatten.
        No manual intervention is required, reducing boilerplate.
        """
        super().__init_subclass__(**kwargs)

        pytree.register_pytree_node(
            cls,
            cls._pytree_flatten,
            cls._pytree_unflatten,
            flatten_with_keys_fn=cls._pytree_flatten_with_keys_fn,
        )

    @abstractmethod
    def _pytree_flatten(self) -> Tuple[List[Any], Context]:
        """Flattens the instance into leaves and context for PyTree operations.

        Subclasses must implement this to define how their structure is decomposed
        into a list of child values and a context object for reconstruction.

        Returns:
            A tuple of (leaves, context), where leaves are the flattened children
            and context contains metadata for unflattening.
        """
        pass

    @abstractmethod
    def _pytree_flatten_with_keys_fn(
        self,
    ) -> Tuple[List[Tuple[KeyEntry, Any]], Any]:
        """Flattens the instance with keys for advanced PyTree traversal.

        Similar to _pytree_flatten, but includes key information for each leaf,
        enabling key-aware operations like tree_map_with_path.

        Returns:
            A tuple of (keys_and_children, context), where keys_and_children
            is a list of (KeyEntry, child) pairs.
        """
        pass

    @classmethod
    @abstractmethod
    def _pytree_unflatten(
        cls: Type[_PytreeRegistered], leaves: Iterable[Any], context: Context
    ) -> PyTree:
        """Reconstructs an instance from flattened leaves and context.

        Subclasses must implement this to reverse the flattening process,
        ensuring the original structure is restored accurately.

        Args:
            leaves: The flattened child values from _pytree_flatten.
            context: The context object from _pytree_flatten.

        Returns:
            A reconstructed instance of the class.
        """
        pass


def resolve_device(device: DeviceLike) -> torch.device:
    """
    Resolves a device string to a torch.device object using manual device
    resolution without creating dummy tensors. Auto-indexes devices when
    no specific index is provided.

    If the device cannot be resolved or is not available, raises an appropriate
    exception instead of returning a fallback.

    Args:
        device: Device string or torch.device object to resolve.

    Returns:
        torch.device: The resolved device object with proper indexing.

    Raises:
        RuntimeError: If the device type is invalid or device is not available.
        AssertionError: If the backend is not compiled/available.
        ValueError: If device is None.
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


def diagnose_pytree_structure_mismatch(
    tree: PyTree,
    *rests: tuple[PyTree],
    is_leaf: Optional[Callable[[PyTree], bool]] = None,
) -> StructureMismatch | None:
    """Diagnoses if all PyTrees have identical structure for operations requiring uniformity.

    Iterates through key paths, node types, and contexts, checking for equality.
    Useful for debugging mismatches before operations like stacking or mapping.

    Args:
        tree: The first PyTree to compare.
        rests: Additional PyTrees to compare.
        is_leaf: Optional callable to determine if a node is a leaf.

    Returns:
        StructureMismatch | None: Structured error describing the mismatch if structures differ, None if they match.
    """

    def diagnose_keypaths_equal(
        keypaths: Tuple[KeyPath, ...],
    ) -> KeyPathMismatch | None:
        """Check if all key paths are equal.

        Args:
            keypaths: Tuple of KeyPath to compare.

        Returns:
            KeyPathMismatch | None: KeyPathMismatch if key paths differ, None if they match.
        """
        if not all(kp == keypaths[0] for kp in keypaths):
            return KeyPathMismatch(keypaths=keypaths)

    def diagnose_types_equal(
        node_types: Tuple[type, ...], key_path: KeyPath = ()
    ) -> TypeMismatch | None:
        """Check if all node types are equal to the first type.

        Args:
            node_types: Tuple of types to compare.
            key_path: The key path for error reporting.

        Returns:
            TypeMismatch | None: TypeMismatch if types differ, None if they match.
        """
        for i, n in enumerate(node_types[1:]):
            if n != node_types[0]:
                return TypeMismatch(
                    expected_type=node_types[0],
                    actual_type=n,
                    entry_index=i + 1,
                    key_path=key_path,
                )

    def diagnose_contexts_equal(
        contexts: Tuple[Context, ...], key_path: KeyPath = ()
    ) -> ContextMismatch | None:
        """Check if all contexts are equal.

        Args:
            contexts: Tuple of Context to compare.
            key_path: The key path for error reporting.

        Returns:
            ContextMismatch | None: ContextMismatch if contexts differ, None if they match.
        """
        for i, c in enumerate(contexts[1:]):
            if c != contexts[0]:
                return ContextMismatch(
                    expected_context=contexts[0],
                    actual_context=c,
                    entry_index=i + 1,
                    key_path=key_path,
                )

    def generate_key_path_context(
        key_path: KeyPath,
        tree: PyTree,
        is_leaf: Callable[[PyTree], bool] | None = None,
    ) -> Iterator[Tuple[KeyPath, type, Context]]:
        """Recursively traverse a PyTree, yielding key paths, node types, and contexts.

        This function performs a depth-first traversal of the PyTree structure,
        yielding information about each node including its key path, type, and
        flattening context. It handles registered PyTree nodes and treats
        unregistered types as leaves.

        Args:
            key_path: The current key path in the tree traversal, represented as
                a tuple of keys leading to the current node.
            tree: The PyTree structure to traverse. This can be any nested
                structure composed of registered PyTree node types.
            is_leaf: Optional callable that determines if a node should be
                treated as a leaf. If provided, nodes for which this returns
                True will not be traversed further. If None, uses the default
                PyTree leaf detection.

        Yields:
            Tuple[KeyPath, type, Context]: A tuple containing:
                - key_path: The full key path to the current node
                - node_type: The type of the current node
                - context: The context object from flattening the node

        Raises:
            ValueError: If a registered node type does not have a
                flatten_with_keys_fn function available.
        """
        # Early return for leaf nodes
        if is_leaf and is_leaf(tree):
            return

        node_type = _get_node_type(tree)
        handler = SUPPORTED_NODES.get(node_type)
        if not handler:
            return  # Unregistered type treated as leaf

        flatten_fn = handler.flatten_with_keys_fn
        if not flatten_fn:
            raise ValueError(
                f"No flatten_with_keys_fn for type: {node_type}. "
                "Provide one when registering the PyTree node."
            )

        # Flatten and yield current node
        keys_and_children, context = flatten_fn(tree)
        yield key_path, node_type, context

        # Recurse on children
        for key, child in keys_and_children:
            new_path = (*key_path, key)
            yield from generate_key_path_context(new_path, child, is_leaf)

    if len(rests) == 0:
        return

    # Create generators for each tree's structure to enable synchronized traversal and comparison.
    # Each generator yields tuples of (key_path, node_type, context) for every node in the PyTree,
    # allowing us to iterate through multiple trees in parallel and detect structural mismatches
    # by comparing corresponding nodes across all trees at each level of nesting.
    # Example: For a nested TensorDict, the generator might yield:
    #   - keypath: ("a",), TensorDict, TensorDictContext(...)
    #   - keypath: ("a", "b"), TensorDict, TensorDictContext(...)
    #   - ...
    generators = [
        generate_key_path_context((), tree, is_leaf) for tree in [tree, *rests]
    ]

    # Iterate through corresponding nodes in all trees simultaneously.
    # zip(*generators) provides the next (keypath, type, context) from each tree at the same structural level.
    # Unpack into separate lists and check for mismatches in keypaths, types, or contexts.
    # Return detailed error message on first mismatch found.
    # Example: zipped_items = ((("a",), TensorDict, ctx1), (("a",), TensorDict, ctx2), ...)
    for zipped_items in zip(*generators):
        # Unpack the zipped items: transform from ((kp1, t1, c1), (kp2, t2, c2), ...)
        # to separate tuples: keypaths=(kp1, kp2, ...), node_types=(t1, t2, ...), contexts=(c1, c2, ...)
        # This groups corresponding elements from each tree for easy comparison.
        keypaths, node_types, contexts = tuple(zip(*zipped_items))

        mismatch = (
            diagnose_keypaths_equal(keypaths)
            or diagnose_types_equal(node_types, keypaths[0])
            or diagnose_contexts_equal(contexts, keypaths[0])
        )

        if mismatch:
            return mismatch

    return None



T = TypeVar('T', bound="ContextWithAnalysis")
class ContextWithAnalysis(Generic[T]):
    """Base class for PyTree structure mismatch errors."""

    @abstractmethod
    def __str__(self) -> str:
        """Return a human-readable description of the context"""
        pass

    @abstractmethod
    def analyze_mismatch_with(self, other: T, entry_index: int) -> str:
        pass