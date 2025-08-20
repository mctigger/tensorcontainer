"""
This example demonstrates detaching gradients in TensorDataClass instances.
"""

import torch
from tensorcontainer import TensorDataClass


class TrainingBatch(TensorDataClass):
    observations: torch.Tensor
    actions: torch.Tensor


def main() -> None:
    # Create a training batch with gradient tracking
    batch = TrainingBatch(
        observations=torch.randn(4, 10, requires_grad=True),
        actions=torch.randn(4, 3, requires_grad=True),
        shape=(4,),
        device="cpu",
    )

    assert batch.observations.requires_grad 
    assert batch.actions.requires_grad 

    # Detach the batch to stop gradient flow
    detached_batch = batch.detach()

    assert not detached_batch.observations.requires_grad 
    assert not detached_batch.actions.requires_grad 

if __name__ == "__main__":
    main()