"""
Example demonstrating method chaining with TensorNormal.

This example shows how TensorNormal supports seamless chaining of tensor operations
like .view(), .detach(), .permute(), and .expand() while maintaining the distribution
interface.
"""

import torch
from tensorcontainer.tensor_distribution import TensorNormal


def demonstrate_tensor_normal_chaining():
    """Demonstrate chaining operations on TensorNormal distribution."""
    # Create a TensorNormal with batch shape (2, 3) and gradients enabled
    loc = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], requires_grad=True)
    scale = torch.tensor([[1.0, 0.5, 2.0], [1.5, 0.8, 1.2]], requires_grad=True)

    normal = TensorNormal(loc=loc, scale=scale)

    # Chain operations: view -> detach -> permute -> expand
    chained = normal.view(6).detach().permute(0).expand(2, 6)

    # Verify the distribution interface still works
    samples = chained.sample()
    log_probs = chained.log_prob(samples)
    entropy = chained.entropy()

    # More complex chaining with 3D batch shape
    loc_3d = torch.randn(2, 2, 3, requires_grad=True)
    scale_3d = torch.abs(torch.randn(2, 2, 3, requires_grad=True)) + 0.1

    normal_3d = TensorNormal(loc=loc_3d, scale=scale_3d)

    # Chain: reshape -> permute -> detach -> expand
    complex_chained = (
        normal_3d.view(4, 3)  # Reshape first two dims
        .permute(1, 0)  # Swap dimensions
        .detach()  # Remove gradients
        .expand(3, 4)
    )  # Expand existing dimensions

    # Verify complex chaining works
    complex_samples = complex_chained.sample()

    return chained, complex_chained


if __name__ == "__main__":
    chained_normal, complex_chained_normal = demonstrate_tensor_normal_chaining()
