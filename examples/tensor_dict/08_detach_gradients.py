"""
Detaching gradients in TensorDict instances.

This example demonstrates how to detach gradients from TensorDict
instances using the `detach()` method, which stops gradient flow while
preserving the tensor data and structure of the original instance.

Key concepts demonstrated:
- Detach operation: How the `detach()` method creates a new TensorDict
  instance with the same data but without gradient tracking.
"""

import torch
from tensorcontainer import TensorDict


def main() -> None:
    """Demonstrate gradient detachment operations."""
    # Create batch with gradient tracking
    batch = TensorDict(
        {
            "observations": torch.randn(4, 10, requires_grad=True),
            "actions": torch.randn(4, 3, requires_grad=True),
        },
        shape=(4,),
        device="cpu",
    )

    # Verify gradients are initially tracked
    assert batch["observations"].requires_grad
    assert batch["actions"].requires_grad

    # Detach to stop gradient flow
    detached_batch = batch.detach()

    # Verify gradients are no longer tracked
    assert not detached_batch["observations"].requires_grad
    assert not detached_batch["actions"].requires_grad


if __name__ == "__main__":
    main()