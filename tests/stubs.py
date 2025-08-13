from typing import Any, Iterable, Type

import torch

from tensorcontainer.tensor_container import TensorContainer


class StubTensorContainer(TensorContainer):
    """Minimal TensorContainer implementation for testing PyTree operations."""

    # Class-level call tracking
    _flatten_calls = 0
    _unflatten_calls = 0
    _in_context = False

    def __init__(self, tensor_a: Any, tensor_b: Any, shape, device=None):
        self.tensor_a = tensor_a
        self.tensor_b = tensor_b

        # Use unsafe_construction to avoid calling flatten for validation
        # in TensorContainer __init__
        with TensorContainer.unsafe_construction():
            super().__init__(shape, device)

    def _pytree_flatten(self) -> tuple[list[Any], Any]:
        """Return tensors in consistent order for reconstruction."""
        if not StubTensorContainer._in_context:
            raise RuntimeError(
                "StubTensorContainer must be used within a 'with' statement"
            )
        StubTensorContainer._flatten_calls += 1
        return [self.tensor_a, self.tensor_b], (self.shape, self.device)

    def _pytree_flatten_with_keys_fn(self) -> tuple[list[tuple[Any, Any]], Any]:
        """Return (key, tensor) pairs and context for reconstruction."""
        if not StubTensorContainer._in_context:
            raise RuntimeError(
                "StubTensorContainer must be used within a 'with' statement"
            )

        return [("tensor_a", self.tensor_a), ("tensor_b", self.tensor_b)], (
            self.shape,
            self.device,
        )

    @classmethod
    def _pytree_unflatten(
        cls: Type["StubTensorContainer"], leaves: Iterable[Any], context: Any
    ) -> "StubTensorContainer":
        """Reconstruct from flattened tensors and context."""
        if not cls._in_context:
            raise RuntimeError(
                "StubTensorContainer must be used within a 'with' statement"
            )
        cls._unflatten_calls += 1
        shape, device = context
        tensor_a, tensor_b = leaves

        # Simplified shape inference - use tensor shape or fallback to original
        new_shape = (
            tensor_a.shape[: len(shape)]
            if hasattr(tensor_a, "shape") and len(tensor_a.shape) >= len(shape)
            else shape
        )

        with cls.unsafe_construction():
            instance = cls.__new__(cls)
            instance.tensor_a = tensor_a
            instance.tensor_b = tensor_b
            instance.shape = torch.Size(new_shape)
            instance.device = device
            return instance

    @classmethod
    def reset_call_tracking(cls):
        """Reset call counters for testing."""
        cls._flatten_calls = 0
        cls._unflatten_calls = 0

    def __enter__(self):
        """Enter context manager, set _in_context to True and reset call tracking."""
        StubTensorContainer._in_context = True
        StubTensorContainer.reset_call_tracking()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager, reset _in_context to False."""
        StubTensorContainer._in_context = False
        return False


# Register the StubTensorContainer as a PyTree node
torch.utils._pytree.register_pytree_node(
    StubTensorContainer,
    StubTensorContainer._pytree_flatten,
    StubTensorContainer._pytree_unflatten,
    flatten_with_keys_fn=StubTensorContainer._pytree_flatten_with_keys_fn,
)
