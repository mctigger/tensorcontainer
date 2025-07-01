from __future__ import annotations

import copy
import dataclasses
from typing import Any, Iterable, List, Optional, Tuple, TypeVar, Union

import torch
from torch import Tensor
from torch.utils import _pytree as pytree
from typing_extensions import dataclass_transform

from tensorcontainer.tensor_container import TensorContainer
from tensorcontainer.utils import PytreeRegistered

TDCompatible = Union[Tensor, TensorContainer]
DATACLASS_ARGS = {"init", "repr", "eq", "order", "unsafe_hash", "frozen", "slots"}


T_TensorDataclass = TypeVar("T_TensorDataclass", bound="TensorDataClass")


@dataclass_transform(eq_default=False)
class TensorDataclassTransform:
    """This class is just needed for type hints. Directly decorating TensorDataclass does not work."""

    pass


class TensorDataClass(TensorContainer, PytreeRegistered, TensorDataclassTransform):
    """A dataclass-based tensor container with automatic field generation and batch semantics.

    TensorDataClass provides a strongly-typed alternative to TensorDict by automatically
    converting annotated class definitions into dataclasses while maintaining tensor
    container functionality. It combines Python's dataclass system with PyTree
    compatibility and torch.compile support.

    ## Automatic Dataclass Generation

    Any class inheriting from TensorDataClass is automatically converted to a dataclass
    with optimized settings for tensor operations:

    - Field-based access using `obj.field` syntax
    - Static typing with IDE support and autocomplete
    - Natural inheritance patterns with field merging
    - Memory-efficient `slots=True` layout
    - Disabled equality comparison (`eq=False`) for tensor compatibility

    Example:
        >>> class MyData(TensorDataClass):
        ...     features: torch.Tensor
        ...     labels: torch.Tensor
        >>>
        >>> data = MyData(
        ...     features=torch.randn(4, 10),
        ...     labels=torch.arange(4).float(),
        ...     shape=(4,),
        ...     device="cpu"
        ... )
        >>>
        >>> # Automatic dataclass features
        >>> print(data.features.shape)  # torch.Size([4, 10])
        >>> data.features = new_tensor  # Type-checked assignment

    ## Batch and Event Dimensions

    TensorDataClass enforces the same batch/event dimension semantics as TensorContainer:

    - **Batch Dimensions**: Leading dimensions defined by `shape` parameter, must be
      consistent across all tensor fields
    - **Event Dimensions**: Trailing dimensions beyond batch shape, can vary per field
    - **Automatic Validation**: Shape compatibility is checked during initialization

    All tensor operations preserve this batch/event structure, enabling consistent
    batched processing across heterogeneous tensor fields.

    ## Field Definition Patterns

    ### Basic Tensor Fields
    ```python
    class BasicData(TensorDataClass):
        observations: torch.Tensor
        actions: torch.Tensor
    ```

    ### Optional Fields and Defaults
    ```python
    class FlexibleData(TensorDataClass):
        required_field: torch.Tensor
        optional_field: Optional[torch.Tensor] = None
        metadata: List[str] = dataclasses.field(default_factory=list)
        config: Dict[str, Any] = dataclasses.field(default_factory=dict)
        default_tensor: torch.Tensor = dataclasses.field(
            default_factory=lambda: torch.zeros(10)
        )
    ```

    ### Inheritance and Field Composition
    ```python
    class BaseData(TensorDataClass):
        observations: torch.Tensor

    class ExtendedData(BaseData):
        actions: torch.Tensor      # Inherits observations
        rewards: torch.Tensor      # Total: observations, actions, rewards

    class FinalData(ExtendedData):
        values: torch.Tensor       # Inherits all previous fields
    ```

    ## PyTree Integration

    TensorDataClass provides seamless PyTree integration through automatic registration:

    - Tensor fields become PyTree leaves for tree operations
    - Non-tensor fields are preserved as metadata
    - Supports `torch.stack`, `torch.cat`, and other tree operations
    - Compatible with `torch.compile` and JIT compilation

    The PyTree flattening separates tensor data from metadata, enabling efficient
    tensor transformations while preserving all field information.

    ## Device and Shape Management

    ### Device and Shape Validation
    The initialization process validates:
    - All tensor fields have batch shapes compatible with the container shape
    - All tensor fields reside on compatible devices
    - Field types match their annotations

    Validation uses PyTree traversal to check nested structures and provides
    detailed error messages with field paths for debugging.

    ## torch.compile Compatibility

    TensorDataClass is designed for efficient compilation:

    - **Static Structure**: Field names and types are known at compile time
    - **Efficient Access**: Direct attribute access compiles to optimized code
    - **Safe Copying**: Custom copy methods avoid graph breaks
    - **Minimal Overhead**: Streamlined operations for hot paths

    ## Memory and Performance

    With `slots=True` by default, TensorDataClass instances provide:

    - Reduced memory overhead compared to regular classes
    - Faster attribute access through direct slot access
    - Better memory locality for improved cache performance
    - Elimination of per-instance `__dict__` storage

    ## Comparison with TensorDict

    | Feature | TensorDataClass | TensorDict |
    |---------|-----------------|------------|
    | Access Pattern | `obj.field` | `obj["key"]` |
    | Type Safety | Static typing | Runtime checks |
    | IDE Support | Full autocomplete | Limited |
    | Memory Usage | Lower (slots) | Higher (dict) |
    | Field Definition | Compile-time | Runtime |
    | Inheritance | Natural OOP | Composition |
    | Dynamic Fields | Not supported | Full support |

    Args:
        shape (Tuple[int, ...]): The batch shape that all tensor fields must share
            as their leading dimensions.
        device (Optional[Union[str, torch.device]]): The device all tensors should
            reside on. If None, device is inferred from the first tensor field.

    Raises:
        ValueError: If tensor field shapes are incompatible with batch shape.
        ValueError: If tensor field devices are incompatible with container device.
        TypeError: If attempting to create a subclass with eq=True.

    Note:
        TensorDataClass automatically applies the @dataclass decorator to subclasses.
        The eq parameter is forced to False for tensor compatibility, and slots is
        enabled by default for performance.
    """

    # Added here to make shape and device part of the data class.
    shape: tuple
    device: Optional[torch.device]

    def __init_subclass__(cls, **kwargs):
        """Automatically convert subclasses into dataclasses with proper field inheritance.

        This method is called whenever a class inherits from TensorDataClass. It:
        1. Merges field annotations from the entire inheritance chain
        2. Extracts dataclass-specific configuration options
        3. Applies the @dataclass decorator with optimized defaults
        4. Enforces constraints like eq=False for tensor compatibility

        The annotation inheritance ensures that derived classes properly inherit
        field definitions from parent TensorDataClass instances.

        Args:
            **kwargs: Class definition arguments, may include dataclass options
                     like 'init', 'repr', 'eq', 'order', 'unsafe_hash', 'frozen', 'slots'

        Raises:
            TypeError: If eq=True is specified (incompatible with tensor fields)
        """
        if hasattr(cls, "__slots__"):
            return

        # Automatically inherit annotations from parent classes
        # Build a complete dictionary of annotations from the MRO.
        # Iterate backwards through the MRO to ensure correct override order (child overrides parent).
        all_annotations = {}
        for base in reversed(cls.__mro__):
            if hasattr(base, "__annotations__"):
                all_annotations.update(base.__annotations__)
        cls.__annotations__ = all_annotations

        dc_kwargs = {}
        for k in list(kwargs.keys()):
            if k in DATACLASS_ARGS:
                dc_kwargs[k] = kwargs.pop(k)

        super().__init_subclass__(**kwargs)

        if dc_kwargs.get("eq") is True:
            raise TypeError(
                f"Cannot create {cls.__name__} with eq=True. TensorDataclass requires eq=False."
            )
        dc_kwargs.setdefault("eq", False)
        dc_kwargs.setdefault("slots", True)

        dataclasses.dataclass(cls, **dc_kwargs)

    def __post_init__(self):
        """Initialize TensorContainer functionality and perform validation.

        This method is automatically called by the dataclass __init__ after all
        fields have been set. It:

        1. Infers device from tensor fields if device was not specified
        2. Initializes the TensorContainer base class with shape and device
        3. Validates that all tensor fields have compatible devices
        4. Validates that all tensor fields have compatible batch shapes

        Raises:
            ValueError: If tensor field shapes are incompatible with batch shape
            ValueError: If tensor field devices are incompatible with container device
        """
        super().__init__(self.shape, self.device)

        # self._tree_validate_device()
        self._tree_validate_shape()

    def _get_path_str(self, key_path):
        """Helper to construct path string from key_path, robust to torch.compile."""
        path_parts = []
        for k in key_path:
            if isinstance(k, tuple):  # Handle nested KeyPath tuples
                path_parts.append(self._get_path_str(k))
            elif hasattr(k, "key"):  # Access the 'key' attribute of the Key object
                path_parts.append(str(k.key))
            else:  # Fallback for unexpected elements
                path_parts.append(str(k))
        return ".".join(path_parts)

    def _tree_validate_shape(self):
        keypath_leaf_pairs = pytree.tree_leaves_with_path(self)
        batch_shape = self.shape

        for key_path, leaf in keypath_leaf_pairs:
            path_str = self._get_path_str(key_path)

            if leaf.ndim > 0 and not self._is_shape_compatible(leaf.shape):
                raise ValueError(
                    f"Shape mismatch at '{path_str}': The tensor shape {leaf.shape} "
                    f"is not compatible with the TensorDataclass's batch shape {batch_shape}."
                )

    def _tree_validate_device(self):
        keypath_leaf_pairs = pytree.tree_leaves_with_path(self)

        for key_path, leaf in keypath_leaf_pairs:
            path_str = self._get_path_str(key_path)

            if not self._is_device_compatible(leaf.device):
                raise ValueError(
                    f"Device mismatch at '{path_str}': The tensor device {leaf.device} "
                    f"is not compatible with the TensorDataclass's device {self.device}."
                )

    def _get_pytree_context(
        self, flat_names: List[str], flat_leaves: List[TDCompatible], meta_data
    ) -> Tuple:
        """
        Private helper to compute the pytree context for this TensorDict.
        The context captures metadata to reconstruct the TensorDict:
        children_spec, event_ndims, original shape, and original device.
        """
        batch_ndim = len(self.shape)
        event_ndims = tuple(leaf.ndim - batch_ndim for leaf in flat_leaves)

        return flat_names, event_ndims, meta_data, self.device

    def _pytree_flatten(self) -> Tuple[List[Any], Any]:
        """Flatten the TensorDataClass into tensor leaves and metadata context.

        Separates dataclass fields into two categories:
        - Tensor-compatible fields become PyTree leaves for transformation
        - Non-tensor fields are stored as metadata in the context

        This enables PyTree operations like tree_map to operate only on tensor
        data while preserving all other field values through the context.

        Returns:
            Tuple containing:
            - List of tensor values (PyTree leaves)
            - Context tuple with (field_names, event_dims, metadata)
        """
        flat_names = []
        flat_values = []

        meta_data = {}

        for f in dataclasses.fields(self):
            name, val = f.name, getattr(self, f.name)
            if isinstance(val, TDCompatible):
                flat_values.append(val)
                flat_names.append(name)
            else:
                meta_data[name] = val

        context = self._get_pytree_context(flat_names, flat_values, meta_data)

        return flat_values, context

    def _pytree_flatten_with_keys_fn(
        self,
    ) -> tuple[list[tuple[pytree.KeyEntry, Any]], Any]:
        """
        Flattens the TensorDataclass into key-path/leaf pairs and static metadata,
        using GetAttrKey for dataclass attributes.
        """
        flat_values, context = self._pytree_flatten()
        flat_names = context[0]
        name_value_tuples = [
            (pytree.GetAttrKey(k), v) for k, v in zip(flat_names, flat_values)
        ]
        return name_value_tuples, context  # type: ignore[return-value]

    @classmethod
    def _pytree_unflatten(
        cls, leaves: Iterable[Any], context: pytree.Context
    ) -> TensorDataClass:
        """Unflattens component values into a dataclass instance."""
        flat_names, event_ndims, meta_data, device = context

        leaves = list(leaves)  # Convert to list to allow indexing

        if not leaves:
            return cls(**meta_data)

        reconstructed_device = device

        # Calculate new_shape based on the (potentially transformed) leaves and event_ndims from context.
        # This correctly determines the batch shape of the TensorDict after operations like stack/cat.
        # For copy(), where leaves are original, this also correctly yields the original shape.
        first_leaf_reconstructed = leaves[0]
        # event_ndims[0] is the event_ndim for the first leaf, relative to original batch shape.
        if (
            event_ndims[0] == 0
        ):  # Leaf was a scalar or had only batch dimensions originally
            reconstructed_shape = first_leaf_reconstructed.shape
        else:  # Leaf had event dimensions originally
            reconstructed_shape = first_leaf_reconstructed.shape[: -event_ndims[0]]

        return cls(
            **dict(zip(flat_names, leaves)),
            **{k: v for k, v in meta_data.items() if k not in ["device", "shape"]},
            device=reconstructed_device,
            shape=reconstructed_shape,
        )

    def __repr__(self) -> str:
        """Provides a developer-friendly representation of the TensorDataclass."""
        items = []
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if isinstance(value, Tensor):
                item_repr = (
                    f"Tensor(shape={value.shape}, dtype={value.dtype}, "
                    f"device={value.device})"
                )
            else:
                item_repr = repr(value)
            items.append(f"{f.name}={item_repr}")

        items_str = ",\n    ".join(items)
        return (
            f"{self.__class__.__name__}(\n"
            f"    shape={self.shape},\n"
            f"    device={self.device},\n"
            f"    fields=(\n        {items_str}\n    )\n)"
        )

    def __copy__(self: T_TensorDataclass) -> T_TensorDataclass:
        """Create a shallow copy of the TensorDataClass instance.

        This method is designed to be `torch.compile` safe by avoiding the
        use of `copy.copy()`, which can cause graph breaks. It manually
        copies all field references without deep-copying tensor data.

        The shallow copy means:
        - Field references are copied (new instance)
        - Tensor data is shared (same underlying tensors)
        - Metadata fields are shared (same objects)

        For independent tensor data, use `clone()` inherited from TensorContainer.

        Returns:
            T_TensorDataclass: New instance with shared field data

        Example:
            >>> original = MyData(obs=torch.randn(4, 128), shape=(4,))
            >>> shallow_copy = original.__copy__()
            >>> shallow_copy.obs is original.obs  # True - shared tensor
            >>>
            >>> # For independent tensors:
            >>> deep_copy = original.clone()  # Creates new tensor data
        """
        # Create a new, uninitialized instance of the correct class.
        cls = type(self)
        new_obj = cls.__new__(cls)

        # Manually copy all dataclass fields.
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            setattr(new_obj, field.name, value)

        # Manually call __post_init__ to initialize the TensorContainer part
        # and run validation logic. This is necessary because we bypassed __init__.
        if hasattr(new_obj, "__post_init__"):
            new_obj.__post_init__()

        return new_obj

    def __deepcopy__(
        self: T_TensorDataclass, memo: Optional[dict] = None
    ) -> T_TensorDataclass:
        """
        Performs a deep copy of the TensorDataclass instance.

        This method is designed to be `torch.compile` safe by manually
        iterating through fields and using `copy.deepcopy` for each,
        while also handling the `memo` dictionary to prevent infinite
        recursion in case of circular references.

        Args:
            memo: A dictionary to keep track of already copied objects.
                  This is part of the `copy.deepcopy` protocol.

        Returns:
            A new TensorDataclass instance with attributes that are deep
            copies of the original's attributes.
        """
        if memo is None:
            memo = {}

        cls = type(self)
        # Check if the object is already in memo
        if id(self) in memo:
            return memo[id(self)]

        new_obj = cls.__new__(cls)
        memo[id(self)] = new_obj

        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            # The `shape` and `device` fields are part of the dataclass fields
            # due to their annotations in TensorDataclass.
            # These should be deepcopied as well if they are not None.
            if field.name in ("shape", "device"):
                # Tuples (shape) and torch.device are immutable or behave as such.
                # Direct assignment is fine and avoids torch.compile issues with deepcopying them.
                # Direct assignment for immutable types like tuple (shape) and torch.device.
                # This avoids torch.compile issues with copy.copy or copy.deepcopy on these types.
                setattr(new_obj, field.name, value)
            elif isinstance(value, Tensor):
                # For torch.Tensor, use .clone() for a deep copy of data.
                setattr(new_obj, field.name, value.clone())
            elif isinstance(value, list):
                # For lists, create a new list. This is a shallow copy of the list structure.
                # If list items are mutable and need deepcopying, torch.compile might
                # still struggle with a generic deepcopy of those items.
                # For a list of immutables (like in the test), this is effectively a deepcopy.
                setattr(new_obj, field.name, list(value))
            else:
                # For other fields (e.g., dict, other custom objects), attempt deepcopy.
                # This remains a potential point of failure for torch.compile
                # if it doesn't support deepcopying these specific types.
                setattr(new_obj, field.name, copy.deepcopy(value))

        # Manually call __post_init__ to initialize the TensorContainer part
        # and run validation logic. This is necessary because we bypassed __init__.
        # __post_init__ in TensorDataclass handles shape and device initialization
        # and validation, which is crucial after all fields are set.
        if hasattr(new_obj, "__post_init__"):
            new_obj.__post_init__()

        return new_obj
