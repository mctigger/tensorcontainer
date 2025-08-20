"""
Detaching gradients in TensorDataClass instances.

This example demonstrates how to detach gradients from TensorDataClass
instances to stop gradient flow while preserving tensor data.

Key concepts demonstrated:
- Gradient tracking in tensor fields
- Detach operation on TensorDataClass
- Independent gradient flow control
"""

import torch
from tensorcontainer import TensorDataClass


class TrainingBatch(TensorDataClass):
    """Training batch with observation and action tensors.

    Args:
        observations: Observation tensor data
        actions: Action tensor data
    """
    observations: torch.Tensor
    actions: torch.Tensor


def main() -> None:
    """Demonstrate gradient detachment operations."""
    # Create batch with gradient tracking
    batch = TrainingBatch(
        observations=torch.randn(4, 10, requires_grad=True),
        actions=torch.randn(4, 3, requires_grad=True),
        shape=(4,),
        device="cpu",
    )

    # Verify gradients are initially tracked
    assert batch.observations.requires_grad
    assert batch.actions.requires_grad

    # Detach to stop gradient flow
    detached_batch = batch.detach()

    # Verify gradients are no longer tracked
    assert not detached_batch.observations.requires_grad
    assert not detached_batch.actions.requires_grad

if __name__ == "__main__":
    main()