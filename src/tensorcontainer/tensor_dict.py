"""TensorDict provides a dictionary-like container for batched tensors that share a common
leading batch shape.

Features:
- Compatible with torch.utils._pytree (shallow flatten) and torch.compile.
- Supports standard mapping operations (getitem, setitem, update, iteration).
- Includes utilities such as `flatten_keys` to promote nested keys to a flat namespace.

PyTree:
- Flattening is shallow: leaves are the immediate values stored in the mapping.
- Reconstruction uses a context tuple capturing keys, per-leaf event_ndims, batch shape, and device.

Notes:
- See class and method docstrings below for detailed behavior and examples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Union,
    cast,
    overload,
    get_args,
)
from collections.abc import Iterable, Mapping

import torch
from torch import Tensor
from torch.utils._pytree import (
    KeyEntry,
    MappingKey,
    PyTree,
)

from tensorcontainer.tensor_container import (
    TensorContainer,
    TensorContainerPytreeContext,
)
from tensorcontainer.types import DeviceLike, ShapeLike
from tensorcontainer.utils import PytreeRegistered

TDCompatible = Union[Tensor, TensorContainer]


# PyTree context metadata for reconstruction
@dataclass
class TensorDictPytreeContext(TensorContainerPytreeContext["TensorDictPytreeContext"]):
    """TensorDict PyTree context with enhanced error messages."""

    keys: list[str]
    event_ndims: list[int]
    metadata: dict[str, Any]

    def __str__(self) -> str:
        """Return human-readable description of this TensorDict context."""
        keys_str = f"keys={list(self.keys)}" if self.keys else "keys=[]"
        device_str = f"device={self.device}" if self.device else "device=None"

        return f"TensorDict({keys_str}, {device_str})"

    def analyze_mismatch_with(
        self, other: TensorDictPytreeContext, entry_index: int
    ) -> str:
        """Analyze specific mismatches with another TensorDict context."""
        # Start with base class analysis (device mismatch, if any)
        guidance = super().analyze_mismatch_with(other, entry_index)

        # Add TensorDict-specific analysis
        self_keys = set(self.keys)
        other_keys = set(other.keys)

        if self_keys != other_keys:
            missing = self_keys - other_keys
            extra = other_keys - self_keys
            guidance += "Key mismatch detected."
            if missing:
                guidance += (
                    f" Missing keys in container {entry_index}: {sorted(missing)}."
                )
            if extra:
                guidance += f" Extra keys in container {entry_index}: {sorted(extra)}."

        return guidance


class TensorDict(TensorContainer, PytreeRegistered):
    """Dictionary-like container for batched tensors that share the same leading batch shape.

    Args:
      data: Mapping from string keys to tensors, ``TensorDict`` instances, or nested
        dicts. Nested dicts are recursively wrapped as ``TensorDict`` instances with
        the same batch shape and device.
      shape: Expected batch shape prefix. All tensors must have shapes beginning
        with this prefix (validated on assignment where applicable).
      device: Optional device constraint; when set, all tensors must reside on this
        device (validated on assignment).

    Attributes:
      data: Underlying mapping storing tensors or nested ``TensorDict`` instances.
      shape: Common batch shape shared by all leaves.
      device: Device constraint for all leaves when specified.

    Notes:
      - Mapping semantics: ``__getitem__``/``__setitem__`` handle string keys as dictionary
        access; slicing/indexing delegates to ``TensorContainer``.
      - PyTree: flattening is shallow; leaves correspond to immediate values in ``data``.
        The context captures keys, per-leaf ``event_ndims``, batch shape, and device to
        support reconstruction and ``torch.compile``.
      - Nested dict handling: plain dict values are wrapped into ``TensorDict`` recursively
        with the same shape/device invariants.

    Examples:
      >>> td = TensorDict({'x': torch.zeros(4, 3)}, shape=(4,))
      >>> td['x'].shape
      torch.Size([4, 3])
      >>> td2 = TensorDict({'a': {'b': torch.ones(4, 2)}}, shape=(4,))
      >>> isinstance(td2['a'], TensorDict)
      True
      >>> td_flat = td2.flatten_keys()
      >>> 'a.b' in td_flat.keys()
      True
    """

    def __init__(
        self,
        data: Mapping[str, Any],
        shape: ShapeLike,
        device: DeviceLike | None = None,
    ):
        """Initialize a TensorDict with minimal overhead for torch.compile.

        Args:
          data: Mapping whose nested dicts are wrapped into ``TensorDict`` instances.
          shape: Expected batch shape prefix for all tensors.
          device: Optional device constraint applied to tensors.

        Notes:
          - The constructor performs minimal work to remain ``torch.compile``-friendly.
          - Shape and device validations are enforced via setters and update paths.
        """
        self.data = TensorDict.data_from_dict(data, shape, device)

        super().__init__(shape, device)

    @classmethod
    def data_from_dict(
        cls, data: Mapping[str, Any], shape: ShapeLike, device=None
    ) -> dict[str, Any]:
        """Recursively wrap nested dict values into TensorDict instances.

        Args:
          data: Input mapping possibly containing nested plain dictionaries.
          shape: Batch shape to assign to nested ``TensorDict`` instances.
          device: Optional device to assign to nested ``TensorDict`` instances.

        Returns:
          Dict[str, TDCompatible]: A dictionary whose nested dicts are converted
          to ``TensorDict`` instances with the provided shape and device.

        Notes:
          No validation is performed here; callers enforce shape/device constraints.
          This preserves structure while normalizing nested dictionaries to ``TensorDict``.
        """
        result = {}
        for k, v in data.items():
            if isinstance(v, dict):
                result[k] = TensorDict(
                    TensorDict.data_from_dict(v, shape, device), shape, device
                )
            else:
                result[k] = v

        return result

    def _get_pytree_context(
        self,
        keys: list[str],
        flat_leaves: list[Any],
        metadata: dict[str, Any],
    ) -> TensorDictPytreeContext:
        """Compute pytree context metadata for reconstructing this TensorDict.

        Args:
          keys: Top-level keys in insertion order.
          flat_leaves: Leaves corresponding to the keys.
          metadata: Dictionary of non-TDCompatible metadata.

        Returns:
          TensorDictPytreeContext: Context capturing keys, per-leaf ``event_ndims``,
          original batch shape, device, and metadata.
        """
        batch_ndim = len(self.shape)
        event_ndims = list(leaf.ndim - batch_ndim for leaf in flat_leaves)
        return TensorDictPytreeContext(self.device, list(keys), event_ndims, metadata)

    def _pytree_flatten(
        self,
    ) -> tuple[list[Any], TensorDictPytreeContext]:
        """Shallow flatten into leaves and context.

        Returns:
          Tuple[List[TDCompatible], TensorDictPytreeContext]: The TDCompatible leaves in key order and
          the reconstruction context (keys, per-leaf ``event_ndims``, shape, device, metadata).
        """
        td_compatible_leaves: list[Any] = []
        td_compatible_keys: list[str] = []
        metadata: dict[str, Any] = {}

        for key, value in self.data.items():
            if isinstance(value, get_args(TDCompatible)):
                td_compatible_leaves.append(value)
                td_compatible_keys.append(key)
            else:
                metadata[key] = value

        context = self._get_pytree_context(
            td_compatible_keys, td_compatible_leaves, metadata
        )
        return td_compatible_leaves, context

    def _pytree_flatten_with_keys_fn(
        self,
    ) -> tuple[list[tuple[KeyEntry, Any]], Any]:
        """Return ``(keypath, leaf)`` pairs and context for pytree APIs.

        Returns:
          tuple[list[tuple[KeyEntry, Any]], TensorDictPytreeContext]: Pairs of
          ``MappingKey(key)`` with each leaf, and the same context as
          :meth:`_pytree_flatten`.
        """
        leaves, context = self._pytree_flatten()
        # Pair MappingKey(key) with each leaf for pytree APIs.
        key_value_pairs = [
            (cast(KeyEntry, MappingKey(k)), cast(Any, v))
            for k, v in zip(context.keys, leaves)
        ]
        return key_value_pairs, context

    @classmethod
    def _pytree_unflatten(
        cls, leaves: Iterable[Any], context: TensorDictPytreeContext
    ) -> PyTree:
        """Reconstruct a TensorDict from leaves and context.

        Args:
          leaves: Iterable of leaves in key order.
          context: ``TensorDictPytreeContext`` carrying keys, per-leaf ``event_ndims``,
            shape, and device.

        Returns:
          TensorDict: Reconstructed object with data mapped from keys to leaves.

        Notes:
          - Batch shape is inferred from the first leaf and its corresponding ``event_ndims``:
            if ``event_ndims[0] == 0``, the batch shape equals ``first_leaf.shape``; otherwise,
            it is ``first_leaf.shape[:-event_ndims[0]]``.
        """
        # Access context fields
        keys = context.keys
        event_ndims = context.event_ndims
        device_context = context.device
        metadata = context.metadata

        obj = cls.__new__(cls)
        obj.device = device_context
        leaves_list = list(leaves)

        # Reconstruct mapping from keys and leaves
        data = dict(zip(keys, leaves_list))
        # Add metadata back to the data
        data.update(metadata)
        obj.data = data

        first_leaf = leaves_list[0]

        # Infer batch shape from first leaf and event_ndims
        if (
            event_ndims and event_ndims[0] == 0
        ):  # Leaf was a scalar or had only batch dimensions originally
            reconstructed_shape = first_leaf.shape
        else:  # Leaf had event dimensions originally
            reconstructed_shape = first_leaf.shape[: -event_ndims[0]]

        obj.shape = reconstructed_shape

        return obj

    # --- Standard MutableMapping methods ---
    @overload
    def __getitem__(self, key: str) -> Any: ...

    @overload
    def __getitem__(self, key: slice) -> TensorDict: ...

    @overload
    def __getitem__(self, key: Tensor) -> TensorDict: ...

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            return self.data[key]

        return super().__getitem__(key)

    @overload
    def __setitem__(self, key: str, value: Any) -> None: ...

    @overload
    def __setitem__(
        self,
        key: slice | Tensor | int | tuple[slice | Tensor | int, ...],
        value: Any,
    ) -> None: ...

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(key, str):
            if isinstance(value, dict):
                value = TensorDict(value, self.shape, self.device)
            else:
                self._validate_device(value)
                self._validate_shape(value)

            self.data[key] = value
        else:
            # Handle slicing/indexing assignments via TensorContainer
            if isinstance(value, (float, int)):
                # Promote Python scalars to tensors to support slice assignment paths
                value = torch.tensor(
                    value, device=self.device, dtype=torch.float32
                )  # scalar promotion for container setitem

            super().__setitem__(key, value)

    def __delitem__(self, key: str):
        del self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def update(self, other: dict[str, Any] | TensorDict):
        """Update entries from a mapping or another TensorDict.

        Args:
          other: A mapping or ``TensorDict`` whose key-value pairs will be written
            into this container.

        Notes:
          - If ``other`` is a ``TensorDict``, its internal data mapping is used.
          - Assignment follows ``__setitem__`` validations, including wrapping plain
            dicts into ``TensorDict`` with the same shape/device and enforcing
            device and shape compatibility for tensors.
        """
        if isinstance(other, TensorDict):
            other = other.data
        for key, value in other.items():
            self[key] = value

    def flatten_keys(self, separator: str = ".") -> TensorDict:
        """Return a new TensorDict whose keys are flattened using the given separator.

        Args:
          separator: String used to join nested keys (default: ``"."``).

        Returns:
          TensorDict: A new ``TensorDict`` with flattened keys. Values are the same
          tensor objects (no copies), and the original batch shape and device are preserved.

        Notes:
          Traversal is iterative (non-recursive) to avoid recursion and temporary
          reference cycles. Nested ``TensorDict`` keys are joined like ``"parent.child"``
          by default.
        """
        out = {}
        # Stack for iterative traversal: (data, prefix)
        stack: list[tuple[Any, str]] = [(self, "")]

        while stack:
            data, prefix = stack.pop()

            if isinstance(data, TensorDict):
                for key, value in data.items():
                    new_prefix = prefix + key + separator
                    stack.append((value, new_prefix))
            else:
                # Store the flattened value with its full key
                out[prefix[:-1]] = data

        return TensorDict(out, self.shape, self.device)
